# Path: src/risk/regime.py
"""Shared RiskRegime — single source of truth for VIX thresholds.

NORMAL      : VIX < 20   — full positioning
BEAR        : 20 ≤ VIX < 30 — graduated caution
CAPITULATION: VIX ≥ 30   — distressed alpha regime (NOT a blind kill-switch)

At CAPITULATION, the pipeline does NOT go silent. It surfaces the highest-quality,
lowest-beta structural anchors — assets historically offering asymmetric risk-reward
during market capitulation events (Greenblatt 1980; Druckenmiller drawdown theory).

Criteria for CAPITULATION survivors:
  - Beta ≤ 1.2 (trailing 30-day rolling vs SPY/benchmark)
  - quality_piotroski (normalized) ≥ 0.70  OR  debt_to_equity (normalized) ≤ 0.30
"""
from __future__ import annotations

import math
from enum import Enum
from typing import List

_BEAR_THRESHOLD = 20.0
_CAPITULATION_THRESHOLD = 30.0

# Public aliases — consumers must import these instead of hardcoding VIX levels.
BEAR_THRESHOLD = _BEAR_THRESHOLD
CAPITULATION_THRESHOLD = _CAPITULATION_THRESHOLD
_BETA_MAX = 1.2       # assets above this beta are filtered in CAPITULATION
_PIOTROSKI_FLOOR = 0.70   # normalized score threshold (≈ F-Score ≥ 7)
_DE_CEILING = 0.30    # normalized D/E — bottom quintile qualifier


class RiskRegime(str, Enum):
    NORMAL = "NORMAL"
    BEAR = "BEAR"
    CAPITULATION = "CAPITULATION"


def get_regime(vix: float) -> RiskRegime:
    if not isinstance(vix, (int, float)) or math.isnan(vix) or vix < 0:
        raise ValueError(f"Invalid VIX: {vix!r}")
    if vix >= _CAPITULATION_THRESHOLD:
        return RiskRegime.CAPITULATION
    if vix >= _BEAR_THRESHOLD:
        return RiskRegime.BEAR
    return RiskRegime.NORMAL


def is_panic(vix: float) -> bool:
    """True when VIX triggers CAPITULATION regime (≥ 30). Alias for legacy compatibility."""
    return get_regime(vix) == RiskRegime.CAPITULATION


def score_multiplier(regime: RiskRegime) -> float:
    return {
        RiskRegime.NORMAL: 1.00,
        RiskRegime.BEAR: 0.80,
        RiskRegime.CAPITULATION: 0.50,
    }[regime]


def strategy_label(regime: RiskRegime) -> str:
    return {
        RiskRegime.NORMAL: "NORMAL / FULL POSITIONING",
        RiskRegime.BEAR: "DEFENSIVE / GRADUATED POSITIONING ACTIVATED",
        RiskRegime.CAPITULATION: "CAPITULATION DISTRESSED REGIME / HIGH-QUALITY ANCHORS ONLY",
    }[regime]


def _is_capitulation_survivor(entry: dict) -> bool:
    """True if entry qualifies for CAPITULATION distressed alpha shortlist.

    Survivor criteria (beta gate AND quality gate):
      beta       ≤ 1.2                                   (low-beta filter)
      piotroski  ≥ 0.70  OR  debt_to_equity ≤ 0.30      (quality gate)
    """
    factors = entry.get("factors", {})
    beta = float(factors.get("beta") or factors.get("beta_30d") or 0.0)
    piotroski = float(factors.get("quality_piotroski") or 0.0)
    de_ratio = float(factors.get("debt_to_equity") or 1.0)  # default high if missing

    if beta > _BETA_MAX:
        return False
    return piotroski >= _PIOTROSKI_FLOOR or de_ratio <= _DE_CEILING


def apply_capitulation_filter(entries: List[dict], vix: float) -> List[dict]:
    """Apply CAPITULATION DISTRESSED REGIME filter when VIX ≥ 30.

    Surfaces highest-quality, lowest-beta structural anchors with 0.50× dampening.
    All survivors are force-badged WATCHLIST — no new BUY signals during Panic/Crash.
    Callers (cook_toplists.py) must move survivors from top_buys_* into a watchlist
    key so the Discord embed does not render them as "Active Buy Signals."
    Non-capitulation regime: returns entries unchanged.
    """
    if not is_panic(vix):
        return entries

    mult = score_multiplier(RiskRegime.CAPITULATION)
    result = []
    for entry in entries:
        if not _is_capitulation_survivor(entry):
            continue
        e = dict(entry)
        e["final_score"] = round(float(e.get("final_score", 0)) * mult, 4)
        # Force WATCHLIST — no BUY labels during CAPITULATION regime.
        # (After 0.50× dampening the score is always < 0.60, so HIGH BUY /
        # TACTICAL BUY thresholds can never trigger anyway; making intent explicit.)
        e["badge"] = "WATCHLIST"
        e["_capitulation_survivor"] = True
        result.append(e)
    return result
