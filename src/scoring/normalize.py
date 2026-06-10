"""src/scoring/normalize.py
Score normalization, winsorizing, fallback reweighting, and explainability.

Markowitz (1990 Nobel) — portfolio construction requires comparable, bounded
signals. Raw scores from heterogeneous sources (EDGAR, FMP, yfinance) span
different scales; normalization makes them combinable without distortion.

Winsorizing at the 1st/99th percentile caps extreme outliers that would
dominate a naive rank — a direct application of Tukey's robustness principle.

Public API:
    winsorize(arr, lo=1, hi=99)               → np.ndarray  [0-100 range]
    normalize_score(arr, lo_pct=1, hi_pct=99) → np.ndarray  [0-100]
    fallback_reweight(weights, available_mask) → np.ndarray  (renormalized)
    build_explain(ticker, scores, evidence_ids, weights) → dict
    persist_explain(ticker, explain_dict, cache_root)

Usage:
    from src.scoring.normalize import normalize_score, winsorize, fallback_reweight

    raw = np.array([0.1, 0.5, 100.0, 0.3, 0.9])
    normed = normalize_score(raw)   # [0, 100] with outlier capping
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

_EXPLAIN_ROOT = Path(__file__).parent.parent.parent / ".cache" / "explain"


# ── Winsorize ──────────────────────────────────────────────────────────────────

def winsorize(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Markowitz (1990 Nobel) — cap extreme values at lo/hi percentiles.

    Clips values outside the [lo, hi] percentile range to those boundary
    values.  This prevents single outliers from dominating composite scores.

    $x_{winsor} = \\max(P_{lo},\\; \\min(P_{hi},\\; x))$

    Args:
        arr: 1-D array of raw scores (any scale).
        lo:  Lower percentile (default 1st — removes bottom 1%).
        hi:  Upper percentile (default 99th — removes top 1%).

    Returns:
        Winsorized array (same shape as arr, same dtype float64).
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    p_lo = float(np.nanpercentile(arr, lo))
    p_hi = float(np.nanpercentile(arr, hi))
    return np.clip(arr, p_lo, p_hi)


# ── Normalize ─────────────────────────────────────────────────────────────────

def normalize_score(
    arr:     np.ndarray,
    lo_pct:  float = 1.0,
    hi_pct:  float = 99.0,
    out_min: float = 0.0,
    out_max: float = 100.0,
) -> np.ndarray:
    """Markowitz (1990 Nobel) — winsorize then linearly scale to [out_min, out_max].

    Step 1: winsorize at [lo_pct, hi_pct].
    Step 2: min-max scale to [out_min, out_max].

    $x_{norm} = out_{min} + \\frac{x_{winsor} - \\min}{\\max - \\min}(out_{max} - out_{min})$

    If all values are equal (zero range), returns array filled with out_min.

    Args:
        arr:     1-D array of raw scores.
        lo_pct:  Winsorize lower percentile.
        hi_pct:  Winsorize upper percentile.
        out_min: Lower bound of output range (default 0).
        out_max: Upper bound of output range (default 100).

    Returns:
        Normalized float64 array in [out_min, out_max].
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr

    w = winsorize(arr, lo=lo_pct, hi=hi_pct)
    mn, mx = float(np.nanmin(w)), float(np.nanmax(w))
    if mx == mn:
        return np.full_like(w, out_min)
    return out_min + (w - mn) / (mx - mn) * (out_max - out_min)


# ── Fallback reweighting ───────────────────────────────────────────────────────

def fallback_reweight(
    weights:        Sequence[float],
    available_mask: Sequence[bool],
) -> np.ndarray:
    """Markowitz (1990 Nobel) — proportionally renormalize weights for available components.

    When a scoring component is missing (e.g. FMP unavailable), its weight is
    redistributed proportionally among the remaining components so the composite
    score still sums correctly.

    Rule:
      - Identify indices where available_mask is True.
      - Redistribute total missing weight proportionally to available components.
      - If no components are available, return uniform weights (all equal).

    $w'_i = \\frac{w_i \\cdot \\mathbf{1}[available_i]}{\\sum_{j: available_j} w_j}$

    Args:
        weights:        Original weight vector (must sum to > 0).
        available_mask: Boolean mask of same length — True = component present.

    Returns:
        Renormalized weight array (same length, sums to 1.0, zeros for missing).
    """
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(available_mask, dtype=bool)

    if w.shape != m.shape:
        raise ValueError(
            f"weights length {len(w)} != available_mask length {len(m)}"
        )

    masked = w * m.astype(np.float64)
    total  = masked.sum()
    if total <= 0.0:
        # Nothing available: equal weight over all (fallback of last resort)
        return np.full_like(w, 1.0 / len(w) if len(w) > 0 else 0.0)

    return masked / total


# ── Explainability ─────────────────────────────────────────────────────────────

def build_explain(
    ticker:       str,
    scores:       Dict[str, float],
    weights:      Dict[str, float],
    evidence_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a structured explainability record for a single ticker.

    Markowitz (1990 Nobel): portfolio decisions require transparency; each
    composite score must be decomposable into constituent factors.

    Args:
        ticker:       Ticker symbol.
        scores:       Component name → raw score (e.g. {"insider": 0.8, "momentum": 0.6}).
        weights:      Component name → weight used in composite (same keys as scores).
        evidence_ids: List of filing evidence IDs that contributed to the signal.

    Returns:
        Explain dict with: ticker, composite, breakdown (per component),
        evidence (list of evidence_ids), computed_at.
    """
    available_mask = [scores.get(k) is not None for k in weights]
    w_arr  = np.array([weights.get(k, 0.0) for k in weights])
    s_arr  = np.array([scores.get(k, 0.0) or 0.0 for k in weights])
    w_norm = fallback_reweight(w_arr, available_mask)

    composite = float(np.dot(s_arr, w_norm))

    breakdown = {
        k: {
            "raw":    scores.get(k),
            "weight": float(w_norm[i]),
            "contribution": float(s_arr[i] * w_norm[i]),
        }
        for i, k in enumerate(weights)
    }

    return {
        "ticker":      ticker,
        "composite":   round(composite, 4),
        "breakdown":   breakdown,
        "evidence":    evidence_ids or [],
        "computed_at": time.time(),
    }


def persist_explain(
    ticker:     str,
    explain:    Dict[str, Any],
    cache_root: Optional[Path] = None,
) -> None:
    """Atomically write the explainability record for ticker to disk.

    UI reads .cache/explain/<ticker>.json — no external fetch required.

    Args:
        ticker:     Ticker symbol (used as filename).
        explain:    Dict from build_explain().
        cache_root: Override root dir (default .cache/explain/).
    """
    root = Path(cache_root) if cache_root else _EXPLAIN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    p       = root / f"{ticker}.json"
    payload = json.dumps(explain, indent=2, ensure_ascii=False).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(root))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_explain(
    ticker:     str,
    cache_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load explain record for ticker from cache (None if missing)."""
    root = Path(cache_root) if cache_root else _EXPLAIN_ROOT
    p    = root / f"{ticker}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
