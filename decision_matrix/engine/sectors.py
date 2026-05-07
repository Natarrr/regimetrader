"""decision_matrix/engine/sectors.py
Sector concentration analysis and regime alignment warnings.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from decision_matrix.engine.config import SECTOR_CAP
from decision_matrix.engine.models import Position, RegimeState
from decision_matrix.engine.utils import safe_div


_REGIME_SECTOR_RISK: Dict[str, List[str]] = {
    "Bear":    ["Information Technology", "Consumer Discretionary", "Communication Services"],
    "Panic":   ["Information Technology", "Consumer Discretionary", "Financials"],
    "Crash":   ["Information Technology", "Consumer Discretionary", "Financials", "Energy"],
    "Mania":   ["Information Technology", "Consumer Discretionary"],
    "Euphoria":["Information Technology"],
}


def sector_analysis(
    positions: List[Position],
    symbol_to_sector: Dict[str, str],
    regime: RegimeState,
) -> Tuple[List[str], Dict[str, float]]:
    """Compute sector weights and return concentration + alignment warnings.

    Args:
        positions:        Open positions list.
        symbol_to_sector: Ticker -> GICS sector string mapping.
        regime:           Current regime state (used for alignment check).

    Returns:
        (warnings, sector_weights)
        warnings:       List of human-readable warning strings for the Brief.
        sector_weights: Dict[sector, fraction_of_total_MV].
    """
    sec_mv: Dict[str, float] = {}
    for p in positions:
        sec = symbol_to_sector.get(p.symbol, "Unknown")
        sec_mv[sec] = sec_mv.get(sec, 0.0) + p.market_value

    total = sum(sec_mv.values())
    weights: Dict[str, float] = {}
    warnings: List[str] = []

    for sec, mv in sec_mv.items():
        w = safe_div(mv, total, 0.0)
        weights[sec] = round(w, 4)

        # Concentration check
        if sec != "Unknown" and w > SECTOR_CAP:
            warnings.append(
                f"**{sec}** ({w:.0%} of MV > {int(SECTOR_CAP*100)}% cap)"
            )

    # Regime alignment check: warn if overweight in a risky sector for this regime
    risky_secs = _REGIME_SECTOR_RISK.get(regime.label, [])
    for sec in risky_secs:
        w = weights.get(sec, 0.0)
        if w > 0.10:
            warnings.append(
                f"REGIME MISMATCH: {sec} overweight ({w:.0%}) in {regime.label} regime -- reduce exposure"
            )

    return warnings, weights
