"""decision_matrix/engine/risk.py
Portfolio risk scoring — normalized to 0-100.
"""
from __future__ import annotations

from typing import Dict, List

from decision_matrix.engine.config import (
    MAX_COMPONENT_RISK,
    MAX_TOTAL_RISK_RAW,
    REGIME_RISK_MAP,
)
from decision_matrix.engine.models import Position, RegimeState, RiskState
from decision_matrix.engine.utils import clamp, normalize_risk, safe_div


def _clamp_component(x: float) -> float:
    return clamp(x, 0.0, MAX_COMPONENT_RISK)


def compute_risk_components(
    positions: List[Position],
    regime: RegimeState,
    concentration_score: float,
    intel_score: float,
) -> Dict[str, float]:
    """Return raw component scores in [0, MAX_COMPONENT_RISK].

    Components:
      Regime        : pre-mapped risk level for the current HMM regime label.
      Concentration : largest single position as fraction of total MV.
      Intel         : penalises low average intel score across held positions.
    """
    regime_risk = float(REGIME_RISK_MAP.get(regime.label, 25))

    # Concentration: top-weight mapped 0..1 -> 0..30 (capped at MAX-10 = 30)
    conc_cap = MAX_COMPONENT_RISK - 10
    conc_risk = _clamp_component(concentration_score * conc_cap)

    # Intel: low intel is risky — (1 - avg_intel) * 30
    intel_risk = _clamp_component((1.0 - clamp(intel_score, 0.0, 1.0)) * 30.0)

    return {
        "Regime":        round(regime_risk, 1),
        "Concentration": round(conc_risk,   1),
        "Intel":         round(intel_risk,  1),
    }


def compute_risk_state(
    positions: List[Position],
    regime: RegimeState,
    concentration_score: float,
    intel_score: float,
) -> RiskState:
    """Compute normalised risk state (0-100) from components.

    Args:
        positions:           Open positions (used for future component extensions).
        regime:              Current regime state.
        concentration_score: Largest position fraction [0, 1].
        intel_score:         Average intel score for held symbols [0, 1].

    Returns:
        RiskState with total in [0, 100] and per-component breakdown.
    """
    components = compute_risk_components(
        positions, regime, concentration_score, intel_score
    )
    raw_total = sum(components.values())
    total = normalize_risk(raw_total, MAX_TOTAL_RISK_RAW)
    return RiskState(total=round(total, 1), breakdown=components)
