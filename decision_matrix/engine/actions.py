"""decision_matrix/engine/actions.py
Build and sort action rows for all open positions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from decision_matrix.engine.config import ACTION_URGENCY, ATR_MULT_DEFAULT, ATR_MULT_DEFENSIVE
from decision_matrix.engine.models import (
    ActionRow,
    ConvictionScore,
    Position,
    RegimeState,
    TechnicalSignal,
)
from decision_matrix.engine.utils import clamp, safe_div


def _atr_mult_for_regime(regime_label: str) -> float:
    """Return ATR stop multiplier based on regime — tighter in defensive regimes."""
    if regime_label in ("Bear", "Panic", "Crash"):
        return ATR_MULT_DEFENSIVE
    return ATR_MULT_DEFAULT


def _action_from_combined(
    combined: float,
    regime_label: str,
    unreal_pct: float,
) -> Tuple[str, str, str]:
    """Core action rating logic — mirrors existing _action_rating() in streamlit_app.py.

    Returns (action_label, hex_color, reason).
    Kept in sync with the original so wiring is a drop-in replacement.
    """
    bull_regime = regime_label in ("Bull", "Euphoria", "Mania")
    bear_regime = regime_label in ("Bear", "Panic", "Crash")

    if regime_label == "Crash":
        return "SELL", "#ff4444", "Crash regime -- capital preservation, exit all longs"
    if regime_label == "Panic":
        return "TRIM", "#ffbb33", "Panic regime -- reduce all exposure immediately"

    if unreal_pct < -0.12:
        return "TRIM", "#ffbb33", f"Down {unreal_pct*100:.1f}% -- review stop"

    if bear_regime:
        if combined >= 0.55:
            return "TRIM", "#ffbb33", "Bear regime -- hold but do not add"
        return "SELL", "#ff4444", "Weak conviction in bear regime -- exit"

    if combined >= 0.70 and bull_regime:
        return "BUY MORE", "#00c851", "Strong bull + high conviction"
    if combined >= 0.62 and not bear_regime and abs(combined - 0.5) > 0.04:
        return "ADD", "#7cb342", "Positive signals -- consider adding"
    if combined >= 0.44:
        return "HOLD", "#9e9e9e", "Neutral -- hold current size"
    if combined >= 0.32:
        return "TRIM", "#ffbb33", "Weak signals -- reduce position"
    return "SELL", "#ff4444", "Strong sell signal"


def compute_actions(
    positions: List[Position],
    regime: RegimeState,
    intel_scores: Dict[str, float],
    tech_signals: Dict[str, TechnicalSignal],
    conviction_scores: Dict[str, ConvictionScore],
) -> List[ActionRow]:
    """Build one ActionRow per position and sort by (urgency, -market_value, unreal_pct).

    Sort order:
      1. urgency ascending  (SELL before TRIM before HOLD)
      2. market_value descending  (largest dollar risk first within same urgency)
      3. unreal_pct ascending     (worst performer last resort tiebreak)

    Why -mv: the audit flags that within the same urgency tier, the largest-dollar
    exposure should surface first so the trader sees the biggest risk immediately.
    """
    atr_mult = _atr_mult_for_regime(regime.label)
    rows: List[ActionRow] = []

    for p in positions:
        sym = p.symbol
        intel = clamp(float(intel_scores.get(sym, 0.5)), 0.0, 1.0)
        tech = tech_signals.get(sym, TechnicalSignal(symbol=sym))
        cv = conviction_scores.get(sym, ConvictionScore(symbol=sym))

        # Combined score mirrors the existing formula in _action_rating
        pnl_contrib = clamp(p.unreal_pct, -0.15, 0.15)
        combined = clamp(
            0.50 * intel + 0.30 * cv.conviction + 0.20 * (0.5 + pnl_contrib),
            0.0, 1.0,
        )
        action, act_color, reason = _action_from_combined(combined, regime.label, p.unreal_pct)

        atr_stop: Optional[float] = None
        risk_usd: Optional[float] = None
        if tech.atr14 and p.price > 0:
            atr_stop = round(p.price - atr_mult * tech.atr14, 2)
            risk_usd = round(p.market_value * safe_div(tech.atr14, p.price, 0.0), 2)

        rows.append(ActionRow(
            symbol=sym,
            action=action,
            urgency=ACTION_URGENCY.get(action, 99),
            reason=reason,
            intel=round(intel, 4),
            unreal_pct=p.unreal_pct,
            qty=p.qty,
            cost=p.avg_cost,
            price=p.price,
            mv=p.market_value,
            pnl=p.unreal_pnl,
            trend=tech.trend_status,
            trend_color=tech.trend_color,
            rsi=tech.rsi14,
            rsi_label=tech.rsi_label,
            rsi_color=tech.rsi_color,
            atr_stop=atr_stop,
            risk_usd=risk_usd,
            conviction=cv.conviction,
            grade=cv.grade,
            grade_color=cv.grade_color,
            atr_alert=tech.atr_alert,
            atr_pct=tech.atr_pct,
        ))

    # Audit fix: sort by urgency, then largest-dollar first, then worst P&L last
    rows.sort(key=lambda r: (r.urgency, -r.mv, r.unreal_pct))
    return rows
