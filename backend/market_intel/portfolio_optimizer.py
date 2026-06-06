# Path: backend/market_intel/portfolio_optimizer.py
"""MVO portfolio optimizer — no qlib dependency.

Imported by generate_top_lists.py to add portfolio_weight to top_lists.json.

Fallback chain (always produces a valid weight vector):
    MVO (SLSQP) → risk parity → score-proportional

VIX vol-targeting applied post-optimization (scales weights down, never up).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

_COVARIANCE_PATH = Path("logs/covariance_matrix.npz")
_IC_REPORT_PATH = Path("research/ic_report.json")

_MAX_POSITION = 0.10        # max 10% per position
_MAX_SECTOR_WEIGHT = 0.30   # max 30% per sector
_MAX_TURNOVER = 0.20        # max 20% portfolio turnover

TARGET_VOL: dict[str, float] = {
    "normal": 0.15,   # VIX < 25
    "bear":   0.10,   # VIX 25–30
    "panic":  0.05,   # VIX >= 30
}


def _vix_regime(vix: float) -> str:
    if vix >= 30:
        return "panic"
    if vix >= 25:
        return "bear"
    return "normal"


def _score_proportional(scores: np.ndarray) -> np.ndarray:
    w = np.clip(scores, 0.0, None)
    total = w.sum()
    if total <= 0:
        n = len(scores)
        return np.ones(n) / n
    w = w / total
    w = np.clip(w, 0.0, _MAX_POSITION)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(scores)) / len(scores)


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols <= 1e-8, 1e-8, vols)
    w = 1.0 / vols
    return w / w.sum()


def _mvo(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    sector_ids: list[int],
    n_sectors: int,
    prev_weights: Optional[np.ndarray],
) -> np.ndarray:
    n = len(expected_returns)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ expected_returns)
        vol = float(np.sqrt(w @ cov @ w + 1e-10))
        return -ret / vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    for s in range(n_sectors):
        mask = np.array([1.0 if sector_ids[i] == s else 0.0 for i in range(n)])
        if mask.sum() > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w, m=mask: _MAX_SECTOR_WEIGHT - float((w * m).sum())}
            )

    if prev_weights is not None and len(prev_weights) == n:
        constraints.append(
            {"type": "ineq", "fun": lambda w: _MAX_TURNOVER - float(np.abs(w - prev_weights).sum())}
        )

    bounds = [(0.0, _MAX_POSITION)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"MVO did not converge: {result.message}")
    return np.array(result.x)


def _load_covariance(tickers: list[str]) -> Optional[np.ndarray]:
    if not _COVARIANCE_PATH.exists():
        log.debug("No covariance matrix at %s — will use identity fallback", _COVARIANCE_PATH)
        return None
    try:
        data = np.load(_COVARIANCE_PATH, allow_pickle=True)
        stored_tickers: list[str] = list(data["tickers"])
        full_cov: np.ndarray = data["covariance"]
        idx = [stored_tickers.index(t) for t in tickers if t in stored_tickers]
        if len(idx) != len(tickers):
            log.warning("Covariance ticker mismatch: expected %d, found %d", len(tickers), len(idx))
            return None
        return full_cov[np.ix_(idx, idx)]
    except Exception as exc:
        log.warning("Failed to load covariance matrix: %s", exc)
        return None


def _ic_estimate() -> float:
    try:
        data = json.loads(_IC_REPORT_PATH.read_text())
        ics = [v["mean_ic"] for v in data.values() if isinstance(v, dict) and "mean_ic" in v]
        return max(0.01, float(np.mean(ics))) if ics else 0.03
    except Exception:
        return 0.03  # conservative default


def run_optimizer(
    tickers: list[str],
    scores: list[float],
    sectors: list[str],
    vix: float,
    prev_weights: Optional[dict[str, float]] = None,
) -> tuple[dict[str, float], str]:
    """Compute portfolio weights for the given tickers.

    Args:
        tickers:      Ticker list (top-20 by composite score).
        scores:       Composite scores, same order as tickers.
        sectors:      Sector strings, same order as tickers.
        vix:          Current VIX level for vol-targeting.
        prev_weights: Previous weight dict (ticker → weight) for turnover control.

    Returns:
        (weights_dict, method_used)
        method_used in {"MVO", "risk_parity", "score_proportional"}
    """
    n = len(tickers)
    scores_arr = np.array(scores, dtype=float)

    unique_sectors = sorted(set(sectors))
    sector_map = {s: i for i, s in enumerate(unique_sectors)}
    sector_ids = [sector_map[s] for s in sectors]

    cov = _load_covariance(tickers)
    if cov is None:
        cov = np.eye(n) * 0.04  # 20% annual vol fallback

    ic = _ic_estimate()
    z = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)
    expected_returns = ic * z

    prev_arr = (
        np.array([prev_weights.get(t, 0.0) for t in tickers])
        if prev_weights else None
    )

    method = "MVO"
    try:
        weights = _mvo(expected_returns, cov, sector_ids, len(unique_sectors), prev_arr)
    except Exception as exc:
        log.warning("MVO failed (%s), trying risk parity", exc)
        method = "risk_parity"
        try:
            weights = _risk_parity(cov)
        except Exception as exc2:
            log.warning("Risk parity failed (%s), using score-proportional", exc2)
            method = "score_proportional"
            weights = _score_proportional(scores_arr)

    # VIX vol-targeting — scale down only, never up
    regime = _vix_regime(vix)
    target_vol = TARGET_VOL[regime]
    port_vol = float(np.sqrt(weights @ cov @ weights + 1e-10))
    if port_vol > 1e-8:
        scale = min(1.0, target_vol / port_vol)
        weights = weights * scale

    return {t: float(w) for t, w in zip(tickers, weights)}, method
