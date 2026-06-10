# Path: backend/market_intel/_score_compositor.py
#
# STATUS: REFERENCE ONLY — not called in live pipeline as of 2026-06.
# Live scoring path:
#   run_pipeline._score_ticker() → neutralize_factors() → renormalize_weights_for_market()
# This module is kept for the Piotroski gate logic reference and test validation.
# To wire into live pipeline: replace _to_entry()'s inline weight calculation
# with a call to compute_composite_score() from this module.
#
# Drop-in replacement for the composite score computation block in
# generate_top_lists.py (or wherever _compute_composite_score is defined).
#
# USAGE — replace the existing composite score function with this module:
#
#   from backend.market_intel._score_compositor import compute_composite_score
#
# WHAT CHANGES vs the previous implementation
# ────────────────────────────────────────────
# 1. Calls get_weights(ticker) instead of using a global WEIGHTS dict.
#    → US tickers:   WEIGHTS_US   (no behavioural change whatsoever)
#    → EU/Asia:      WEIGHTS_GLOBAL (congress weight redistributed)
#
# 2. congress_score is FORCED to 0.0 for non-US tickers regardless of what
#    the upstream scorer returned. This is a safety guard against accidental
#    contamination — the weight is already 0.0 in WEIGHTS_GLOBAL, but the
#    explicit zero prevents any future drift.
#
# 3. Adds `region` and `weights_set` keys to the returned metadata dict so
#    callers (Discord formatter, canary validation, audit payload) can surface
#    which weight set was applied without re-computing the region.
#
# 4. quality_piotroski is applied as a MULTIPLICATIVE GATE on the BUY score
#    (same logic as before, now also in WEIGHTS_GLOBAL at 0.03 weighted).
#    The gate is evaluated AFTER the weighted sum — order unchanged.
#
# CONSTRAINTS (unchanged)
# ───────────────────────
# - Soft failures return (0.0, "none") — never raise
# - _cross_sectional_normalize() is NOT called here — caller responsibility
# - _apply_vix_overlay() is NOT called here — caller responsibility
# - sum(weights.values()) == 1.0 enforced by weights.py at import time

from __future__ import annotations

import logging
from typing import Any

from src.config.weights import get_weights, get_region, is_congress_eligible

logger = logging.getLogger(__name__)

# Piotroski gate thresholds (unchanged from sprint step 6 spec)
_PIOTROSKI_FULL_WEIGHT   = 6    # F-Score ≥ 6 → full weight (multiplier 1.0)
_PIOTROSKI_PARTIAL_FLOOR = 3    # F-Score 3-5 → partial weight (multiplier 0.6)
_PIOTROSKI_GATE_ZERO     = 3    # F-Score < 3 → BUY suppressed (multiplier 0.0)
_PIOTROSKI_PARTIAL_MULT  = 0.6


def compute_composite_score(
    ticker: str,
    factor_scores: dict[str, float],
    piotroski_raw: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute the composite weighted score for *ticker*.

    Parameters
    ----------
    ticker:
        Ticker symbol, e.g. "AAPL", "SAP.DE", "7203.T".
    factor_scores:
        Dict mapping factor name → normalised score in [0.0, 1.0].
        Expected keys (all optional — missing → 0.0):
            insider_conviction, insider_breadth, congress,
            news_sentiment, news_buzz, momentum_long, volume_attention,
            analyst_consensus, quality_piotroski
    piotroski_raw:
        Raw Piotroski F-Score integer (0–9). Used for the multiplicative gate.
        If None, gate is not applied (score passes through).

    Returns
    -------
    (composite_score, metadata)
        composite_score : float in [0.0, 1.0]
        metadata        : dict with keys:
            weights_set     : "US" | "EU" | "ASIA"
            region          : "US" | "EU" | "ASIA"
            weighted_factors: {factor: weighted_contribution}
            congress_masked : bool  (True when congress forced to 0.0)
            piotroski_gate  : float (multiplier applied: 0.0 / 0.6 / 1.0)
            piotroski_raw   : int | None
    """
    try:
        region = get_region(ticker)
        weights = get_weights(ticker)
        weights_set = region  # "US", "EU", or "ASIA"

        # ── Safety guard: force congress to 0.0 for non-US ──────────────────
        congress_masked = False
        scores = dict(factor_scores)   # defensive copy
        if not is_congress_eligible(ticker):
            if scores.get("congress", 0.0) != 0.0:
                logger.warning(
                    "%s: congress_score=%s forced to 0.0 (non-US ticker, "
                    "WEIGHTS_GLOBAL already sets weight=0.0 but raw score "
                    "contamination detected — check upstream scorer)",
                    ticker, scores.get("congress"),
                )
                congress_masked = True   # contamination detected and zeroed
            scores["congress"] = 0.0

        # ── Weighted sum ──────────────────────────────────────────────────────
        weighted_factors: dict[str, float] = {}
        raw_sum = 0.0
        for factor, weight in weights.items():
            s = float(scores.get(factor, 0.0))
            contribution = weight * s
            weighted_factors[factor] = round(contribution, 6)
            raw_sum += contribution

        # ── Piotroski multiplicative gate ─────────────────────────────────────
        # Applied AFTER the weighted sum — suppresses the full composite for
        # financially distressed companies regardless of insider/momentum signals.
        gate_mult = _piotroski_gate_multiplier(piotroski_raw)
        composite = raw_sum * gate_mult

        # Clamp to [0.0, 1.0] — floating-point rounding safety
        composite = max(0.0, min(1.0, composite))

        metadata = {
            "weights_set":      weights_set,
            "region":           region,
            "weighted_factors": weighted_factors,
            "congress_masked":  congress_masked,
            "piotroski_gate":   gate_mult,
            "piotroski_raw":    piotroski_raw,
        }
        return composite, metadata

    except Exception:
        logger.exception("%s: compute_composite_score failed — returning 0.0", ticker)
        return 0.0, {
            "weights_set":      "US",
            "region":           "US",
            "weighted_factors": {},
            "congress_masked":  False,
            "piotroski_gate":   1.0,
            "piotroski_raw":    None,
        }


def _piotroski_gate_multiplier(piotroski_raw: int | None) -> float:
    """Return the Piotroski multiplicative gate (0.0 / 0.6 / 1.0).

    None → gate not applied (1.0). Matches sprint step 6 spec.
    """
    if piotroski_raw is None:
        return 1.0
    if piotroski_raw < _PIOTROSKI_GATE_ZERO:
        return 0.0
    if piotroski_raw < _PIOTROSKI_FULL_WEIGHT:
        return _PIOTROSKI_PARTIAL_MULT
    return 1.0
