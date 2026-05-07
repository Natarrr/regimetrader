"""decision_matrix/engine/compute.py
Top-level orchestrator — single entry point for the Decision Matrix engine.

Usage (Streamlit Phase 1 replacement):
    from decision_matrix.engine import compute_decision_matrix_state

    state = compute_decision_matrix_state(
        positions=_positions,
        regime=RegimeState(label=regime_lbl, confidence=conf, ...),
        intel_scores=_intel_scores,
        tech_signals=_tech_signals_map,
        conviction_scores=_conviction_map,
        symbol_to_sector=_sym_sector_map,
        equity=equity,
        cash=cash,
        daily_pnl=daily_pnl,
    )
    # Then pass state to the render layer.
"""
from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

from decision_matrix.engine.actions import compute_actions
from decision_matrix.engine.config import (
    CAPE_PERCENTILE_THRESHOLD,
    CONVICTION_GRADES,
    PERSISTENCE_THRESHOLD,
    YIELD_SPREAD_THRESHOLD_BPS,
)
from decision_matrix.engine.conviction import crash_conviction_override
from decision_matrix.engine.models import (
    ActionRow,
    ConvictionScore,
    DecisionMatrixState,
    Position,
    RegimeState,
    RiskState,
    TechnicalSignal,
)
from decision_matrix.engine.regime import apply_regime_overrides
from decision_matrix.engine.risk import compute_risk_state
from decision_matrix.engine.sectors import sector_analysis
from decision_matrix.engine.utils import clamp, safe_div
from decision_matrix.engine.volatility import detect_volatility


# ── Trigger provenance ────────────────────────────────────────────────────────

def compute_minsky_trace(
    persistence: float,
    cape_percentile: float,
    yield_spread_bps: float,
) -> Dict:
    """Return a provenance dict with raw values, booleans, and count.

    This dict is attached to DecisionMatrixState.minsky_trace so the render
    layer can show raw values next to each trigger for auditability.
    """
    persistence_trigger = persistence >= PERSISTENCE_THRESHOLD
    cape_trigger        = cape_percentile >= CAPE_PERCENTILE_THRESHOLD
    yield_trigger       = yield_spread_bps < YIELD_SPREAD_THRESHOLD_BPS
    conditions_met = int(persistence_trigger) + int(cape_trigger) + int(yield_trigger)

    return {
        "timestamp":           datetime.datetime.utcnow().isoformat(),
        "persistence":         round(persistence, 6),
        "persistence_trigger": persistence_trigger,
        "persistence_thresh":  PERSISTENCE_THRESHOLD,
        "cape_percentile":     round(cape_percentile, 2),
        "cape_trigger":        cape_trigger,
        "cape_thresh":         CAPE_PERCENTILE_THRESHOLD,
        "yield_spread_bps":    round(yield_spread_bps, 1),
        "yield_trigger":       yield_trigger,
        "yield_thresh":        YIELD_SPREAD_THRESHOLD_BPS,
        "conditions_met":      conditions_met,
        "alert_level":         ["CLEAR", "WATCH", "WARNING", "CRITICAL"][conditions_met],
    }


# ── Brief builder ─────────────────────────────────────────────────────────────

def build_brief_items(
    actions: List[ActionRow],
    regime: RegimeState,
    sector_warnings: List[str],
    vol_symbols: List[str],
) -> List[Tuple[str, str, str, str]]:
    """Compile the Senior Trader's Brief items.

    Returns: List of (badge, hex_color, title, description) tuples.
    Deduplicates on (badge, title) so the same symbol can't appear twice.
    """
    items: List[Tuple[str, str, str, str]] = []

    # 1. Regime-level items
    if regime.label == "Crash":
        items.append(("HALT",      "#ff4444", "REGIME: CRASH DETECTED",
                       "Hard halt on ALL new long positions. Unified Conviction = 0 across all symbols. Consider full liquidation."))
    elif regime.label == "Panic":
        items.append(("REDUCE",    "#ff4444", "REGIME: PANIC MODE",
                       "Reduce all position sizes immediately. No new long entries permitted. Capital preservation priority."))
    elif regime.label == "Bear":
        items.append(("DEFENSIVE", "#ff8800", "REGIME: BEAR",
                       "No new long entries. Hold only Grade-A conviction names with hard stops below ATR."))

    # 2. Volatility alerts
    for sym in vol_symbols:
        items.append(("ALERT", "#ff8800", f"VOLATILITY ALERT: {sym}",
                       "ATR significantly above 30-day baseline. Widen stops or reduce position size immediately."))

    # 3. Conviction exits
    for a in actions:
        rsi_val = a.rsi
        if rsi_val and rsi_val > 70 and regime.label in ("Bull", "Euphoria", "Mania"):
            items.append(("TRIM", "#ffbb33", f"TRIM {a.symbol}",
                           f"Overbought (RSI {rsi_val:.0f}) in {regime.label} regime. Take partial profits on strength."))

        if a.trend.startswith("Death") and regime.label in ("Bear", "Panic", "Crash"):
            items.append(("LIQUIDATE", "#ff4444", f"LIQUIDATE {a.symbol}",
                           f"Price below SMA200 (Death Cross confirmed) in {regime.label} regime. Investment thesis invalidated."))

        if a.atr_alert:
            items.append(("ALERT", "#ff8800", f"VOLATILITY ALERT: {a.symbol}",
                           f"ATR is {a.atr_pct:.0f}% above 30-day average. Widen stops or reduce position size immediately."))

        if a.grade == "C" and a.mv > 0:
            items.append(("EXIT", "#ff4444", f"EXIT {a.symbol}",
                           f"Unified Conviction = {a.conviction:.0%} (Grade C, regime-adjusted). Risk/reward below threshold."))

    # 4. Sector warnings
    for sw in sector_warnings:
        items.append(("CONCENTRATION", "#ff8800", "SECTOR OVERWEIGHT",
                       f"{sw} -- breaches sector cap. Rebalance to reduce correlated exposure."))

    # 5. Deduplicate on (badge, title)
    seen: set = set()
    clean: List[Tuple] = []
    for badge, clr, title, desc in items:
        key = (badge, title)
        if key not in seen:
            seen.add(key)
            clean.append((badge, clr, title, desc))

    return clean


# ── Portfolio metrics ─────────────────────────────────────────────────────────

def compute_portfolio_beta(
    positions: List[Position],
    tech_signals: Dict[str, TechnicalSignal],
) -> Optional[float]:
    """Market-value-weighted portfolio beta vs SPY."""
    total_mv = sum(p.market_value for p in positions)
    if total_mv <= 0:
        return None
    parts = [
        ts.beta * safe_div(p.market_value, total_mv, 0.0)
        for p in positions
        if (ts := tech_signals.get(p.symbol)) and ts.beta is not None
    ]
    return round(sum(parts), 2) if parts else None


# ── Main orchestrator ─────────────────────────────────────────────────────────

def compute_decision_matrix_state(
    positions: List[Position],
    regime: RegimeState,
    intel_scores: Dict[str, float],
    tech_signals: Dict[str, TechnicalSignal],
    conviction_scores: Dict[str, ConvictionScore],
    symbol_to_sector: Dict[str, str],
    equity: float,
    cash: float,
    daily_pnl: float,
    # Optional Minsky inputs — use 0.0 as neutral defaults if unavailable
    garch_persistence: float = 0.0,
    cape_percentile: float = 0.0,
    yield_spread_bps: float = 50.0,
) -> DecisionMatrixState:
    """Pure compute function — no Streamlit calls, no side effects.

    Wiring to existing helpers:
        positions  → from portf["positions"] (adapt dicts to Position dataclass)
        regime     → from _get_regime_state() (adapt dict to RegimeState)
        intel_scores → from _scores_df_m DataFrame (symbol -> composite)
        tech_signals → from _get_technical_signals() (adapt df rows to TechnicalSignal)
        conviction_scores → from _get_unified_conviction() (adapt df rows to ConvictionScore)
        symbol_to_sector  → from _get_quality_scores() sector column
    """
    # 1. Apply Crash conviction override
    conviction_scores = crash_conviction_override(conviction_scores, regime)

    # 2. Risk (uses top-position concentration as proxy for concentration_score)
    total_mv = sum(p.market_value for p in positions)
    top_weight = (
        max(p.market_value for p in positions) / total_mv
        if positions and total_mv > 0 else 0.0
    )
    avg_intel = (
        sum(intel_scores.get(p.symbol, 0.5) for p in positions) / len(positions)
        if positions else 0.5
    )
    risk_state = compute_risk_state(
        positions=positions,
        regime=regime,
        concentration_score=top_weight,
        intel_score=avg_intel,
    )

    # 3. Actions + regime overrides
    actions = compute_actions(
        positions=positions,
        regime=regime,
        intel_scores=intel_scores,
        tech_signals=tech_signals,
        conviction_scores=conviction_scores,
    )
    actions = apply_regime_overrides(actions, regime)

    # 4. Volatility
    _any_vol, vol_syms = detect_volatility(tech_signals)

    # 5. Sectors
    sector_warnings, _sector_weights = sector_analysis(
        positions=positions,
        symbol_to_sector=symbol_to_sector,
        regime=regime,
    )

    # 6. Brief (deduped)
    brief_items = build_brief_items(
        actions=actions,
        regime=regime,
        sector_warnings=sector_warnings,
        vol_symbols=vol_syms if _any_vol else [],
    )

    # 7. Minsky trace
    minsky_trace = compute_minsky_trace(garch_persistence, cape_percentile, yield_spread_bps)

    # 8. Portfolio-level aggregates
    beta = compute_portfolio_beta(positions, tech_signals)
    total_unreal = sum(p.unreal_pnl for p in positions)
    daily_pnl_pct = safe_div(daily_pnl, max(equity, 1.0), 0.0)
    alloc_frac = clamp(1.0 - safe_div(cash, max(equity, 1.0), 0.0), 0.0, 1.0)

    return DecisionMatrixState(
        regime=regime,
        risk=risk_state,
        action_rows=actions,
        brief_items=brief_items,
        sector_warnings=sector_warnings,
        volatility_symbols=vol_syms,
        portfolio_beta=beta,
        total_mv=total_mv,
        total_unreal=total_unreal,
        daily_pnl_pct=daily_pnl_pct,
        alloc_frac=alloc_frac,
        minsky_trace=minsky_trace,
    )
