# Path: backend/quant_models/prediction_controller.py
"""Laureate regime classifier and Minsky moment detector.

Combines HMM state, monetary regime, and volatility regime into the
4-state Laureate framework: BULL / OVERHEATED / FRAGILE / CRASH.

Minsky Moment: 3-condition alert system (Minsky 1986, "Stabilizing an Unstable Economy"):
    1. GARCH persistence ≥ 0.98  (volatility clustering = permanent regime)
    2. CAPE ≥ 95th percentile     (valuation extreme)
    3. Yield curve inverted       (Harvey 1988 recession predictor)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_LAUREATE_ICONS = {
    "BULL":       "🟢",
    "OVERHEATED": "🟡",
    "FRAGILE":    "🟣",
    "CRASH":      "🔴",
}

_MINSKY_ICONS = {
    "CRITICAL": "🚨",
    "WARNING":  "⚠️",
    "WATCH":    "👁️",
    "CLEAR":    "✅",
}

_SCALE = {
    "BULL":       1.00,
    "OVERHEATED": 0.70,
    "FRAGILE":    0.40,
    "CRASH":      0.00,
}


@dataclass
class MinskyResult:
    alert_level:       str    # CRITICAL | WARNING | WATCH | CLEAR
    conditions_met:    int    # 0-3
    narrative:         str
    garch_persistence: float
    cape_percentile:   float
    yield_spread_bps:  float

    @property
    def icon(self) -> str:
        return _MINSKY_ICONS.get(self.alert_level, "❓")


def classify_regime(
    hmm_label:   str,
    mon_regime:  str,
    vol_regime:  str,
) -> str:
    """Map HMM + monetary + volatility state onto 4 Laureate states.

    Decision matrix (first matching rule wins):

    | HMM   | Monetary    | Volatility  | Laureate    |
    |-------|-------------|-------------|-------------|
    | BULL  | EXPANSION   | STABLE      | BULL        |
    | BULL  | EXPANSION   | EXPANDING   | OVERHEATED  |
    | BULL  | *           | CLUSTERING  | FRAGILE     |
    | BULL  | TIGHTENING  | *           | FRAGILE     |
    | BULL  | STAGFLATION | *           | OVERHEATED  |
    | BEAR  | *           | CLUSTERING  | CRASH       |
    | BEAR  | TIGHTENING  | *           | CRASH       |
    | BEAR  | *           | *           | FRAGILE     |
    | NEUTRAL| *          | CLUSTERING  | FRAGILE     |
    | NEUTRAL| EXPANSION  | *           | OVERHEATED  |
    | *     | *           | *           | FRAGILE     |  (safe fallback)
    """
    h = (hmm_label  or "NEUTRAL").upper()
    m = (mon_regime or "NEUTRAL").upper()
    v = (vol_regime or "STABLE").upper()

    if h == "BULL":
        if v == "CLUSTERING" or m == "TIGHTENING":
            return "FRAGILE"
        if m == "STAGFLATION" or v == "EXPANDING":
            return "OVERHEATED"
        if m == "EXPANSION" and v == "STABLE":
            return "BULL"
        return "OVERHEATED"   # bull but uncertain

    if h == "BEAR":
        if v == "CLUSTERING" or m == "TIGHTENING":
            return "CRASH"
        return "FRAGILE"

    # NEUTRAL
    if v == "CLUSTERING":
        return "FRAGILE"
    if m == "EXPANSION":
        return "OVERHEATED"

    return "FRAGILE"   # conservative default


def combined_position_scale(laureate: str) -> float:
    """Return [0.0, 1.0] position size scalar for each Laureate state.

    BULL=1.00  OVERHEATED=0.70  FRAGILE=0.40  CRASH=0.00
    """
    return _SCALE.get(laureate.upper(), 0.40)


def minsky_moment(
    garch_persistence: float,
    cape_pct:          float,
    yield_spread_bps:  float,
) -> MinskyResult:
    """Evaluate the 3-condition Minsky moment alert.

    Conditions:
        1. garch_persistence ≥ 0.98  (volatility regime integration)
        2. cape_pct ≥ 95.0           (valuation extreme — 95th percentile)
        3. yield_spread_bps < 0      (curve inversion — Harvey 1988)

    Alert levels:
        3/3 → CRITICAL   (all three conditions fire simultaneously)
        2/3 → WARNING
        1/3 → WATCH
        0/3 → CLEAR
    """
    cond1 = garch_persistence >= 0.98
    cond2 = cape_pct >= 95.0
    cond3 = yield_spread_bps < 0.0
    n = sum([cond1, cond2, cond3])

    if n == 3:
        level = "CRITICAL"
        narrative = (
            f"All 3 Minsky conditions active: "
            f"GARCH persistence={garch_persistence:.3f}, "
            f"CAPE p{cape_pct:.0f}, "
            f"yield spread {yield_spread_bps:.0f}bps"
        )
    elif n == 2:
        triggered = []
        if cond1: triggered.append(f"GARCH={garch_persistence:.3f}")
        if cond2: triggered.append(f"CAPE p{cape_pct:.0f}")
        if cond3: triggered.append(f"spread {yield_spread_bps:.0f}bps")
        level = "WARNING"
        narrative = f"2/3 Minsky conditions: {', '.join(triggered)}"
    elif n == 1:
        if cond1:
            narrative = f"Volatility clustering: GARCH persistence={garch_persistence:.3f}"
        elif cond2:
            narrative = f"Valuation extreme: CAPE at p{cape_pct:.0f}"
        else:
            narrative = f"Yield curve: spread {yield_spread_bps:.0f}bps"
        level = "WATCH"
    else:
        level = "CLEAR"
        narrative = (
            f"All clear — persistence={garch_persistence:.3f}, "
            f"CAPE p{cape_pct:.0f}, spread {yield_spread_bps:.0f}bps"
        )

    return MinskyResult(
        alert_level=level,
        conditions_met=n,
        narrative=narrative,
        garch_persistence=garch_persistence,
        cape_percentile=cape_pct,
        yield_spread_bps=yield_spread_bps,
    )


def laureate_icon(state: str) -> str:
    return _LAUREATE_ICONS.get(state.upper(), "❓")
