"""decision_matrix/engine/regime.py
Regime-level overrides applied after action computation.
"""
from __future__ import annotations

from typing import List

from decision_matrix.engine.config import ACTION_URGENCY, REGIME_OVERRIDES
from decision_matrix.engine.models import ActionRow, RegimeState
from decision_matrix.engine.utils import clamp


def apply_regime_overrides(
    actions: List[ActionRow],
    regime: RegimeState,
) -> List[ActionRow]:
    """Convert blocked actions to the regime-mandated fallback.

    Example: in Crash regime, BUY MORE and ADD are converted to SELL.
    Returns a new list — does not mutate inputs.

    Why: the trader brief says "no new longs" but without this override the
    action engine can still emit BUY/ADD for high-conviction names.
    """
    rules = REGIME_OVERRIDES.get(regime.label)
    if not rules:
        return actions

    block = rules["block"]
    force = rules["force"]
    extra = rules["reason"]
    out: List[ActionRow] = []

    for a in actions:
        if a.action in block:
            import dataclasses
            a = dataclasses.replace(
                a,
                action=force,
                urgency=ACTION_URGENCY.get(force, 99),
                reason=a.reason + f" | {extra}",
            )
        out.append(a)
    return out


def apply_crash_conviction_override(
    conviction_map: dict,
    regime: RegimeState,
) -> dict:
    """Force all conviction scores to 0 in Crash regime.

    The Senior Trader's Brief explicitly states "Unified Conviction = 0 across
    all symbols" in Crash.  Without this the engine still uses real scores.

    Returns a new dict — does not mutate input.
    """
    if regime.label != "Crash":
        return conviction_map

    from decision_matrix.engine.models import ConvictionScore
    return {
        sym: ConvictionScore(symbol=sym, conviction=0.0, grade="C", grade_color="#ff4444")
        for sym in conviction_map
    }
