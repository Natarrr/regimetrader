"""backend/quant_models/prediction_controller.py
Module E — The Prediction Controller (Lucas / Sargent / Minsky).

Composite regime classification combining the HMM state, monetary pulse, and
volatility brain into a four-state laureate label. Also fires the Minsky Moment
alert when preconditions are jointly stressed.

All functions are pure — no module-level state, no I/O side effects.
"""
from __future__ import annotations

from typing import Dict, Literal

from backend.data.schemas import MinskyStatusOut


# ── Laureate label type ────────────────────────────────────────────────────────

LaureateRegime = Literal["BULL", "OVERHEATED", "FRAGILE", "CRASH"]


# ── Predicate helpers (single source of truth for string normalization) ────────

def is_risk_off(hmm_label: str) -> bool:
    """True for clearly distressed HMM states: Bear, Panic, Crash."""
    return str(hmm_label).capitalize() in ("Bear", "Panic", "Crash")


def is_bullish(hmm_label: str) -> bool:
    """True for clearly bullish HMM states: Bull, Euphoria, Mania."""
    return str(hmm_label).capitalize() in ("Bull", "Euphoria", "Mania")


def is_tight_or_inverted(mon_regime: str) -> bool:
    """True when monetary_regime() returned TIGHTENING (inverted curve or falling M2V)."""
    return str(mon_regime).upper() == "TIGHTENING"


def is_stressed_vol(vol_regime: str) -> bool:
    """True when volatility_regime() returned CLUSTERING (GARCH persistence > 0.98)."""
    return str(vol_regime).upper() == "CLUSTERING"


# ── classify_regime ────────────────────────────────────────────────────────────

def classify_regime(
    hmm_label: str,
    monetary_regime: str,
    volatility_regime: str,
) -> LaureateRegime:
    """Lucas (1995 Nobel) + Sargent (2011 Nobel) — Conservative four-state regime map.

    Rational expectations: agents jointly price HMM price action, monetary
    conditions, and volatility regime. The key conservatism constraint is that
    CRASH requires risk-off price action AND at least one macro stress confirmation.
    A lone Bear label with benign macro maps to FRAGILE, not CRASH.

    Rule table (checked in order, first match wins):

    | HMM signal     | Monetary   | Vol       | Laureate   |
    |----------------|------------|-----------|------------|
    | risk-off       | TIGHTENING | *         | CRASH      |
    | risk-off       | *          | CLUSTERING| CRASH      |
    | bullish        | TIGHTENING | *         | OVERHEATED |
    | bullish        | *          | CLUSTERING| OVERHEATED |
    | risk-off       | NEUTRAL    | STABLE    | FRAGILE    |
    | neutral/unknown| TIGHTENING | *         | FRAGILE    |
    | neutral/unknown| *          | CLUSTERING| FRAGILE    |
    | bullish        | NEUTRAL    | STABLE    | BULL       |
    | neutral/unknown| NEUTRAL    | STABLE    | FRAGILE    |

    # $\\text{regime} = f(\\hat{q}_t^{\\text{HMM}}, \\Delta_t, \\sigma^2_t)$

    Args:
        hmm_label:         Raw label from RegimeClassifier (Bear / Neutral / Bull …).
        monetary_regime:   From monetary_pulse.monetary_regime() → TIGHTENING/NEUTRAL/EASING.
        volatility_regime: From volatility_brain.volatility_regime() → CLUSTERING/STABLE.

    Returns:
        One of BULL | OVERHEATED | FRAGILE | CRASH.
    """
    _risk_off = is_risk_off(hmm_label)
    _bullish  = is_bullish(hmm_label)
    _tight    = is_tight_or_inverted(monetary_regime)
    _stressed = is_stressed_vol(volatility_regime)

    # ── CRASH ─────────────────────────────────────────────────────────────────
    # Requires price-action stress AND at least one macro confirmation.
    # Prevents Bear + NEUTRAL + STABLE from triggering CRASH.
    if _risk_off and _tight and _stressed:
        return "CRASH"

    # ── OVERHEATED ────────────────────────────────────────────────────────────
    # Bullish price action undermined by restrictive monetary policy or vol spike.
    if _bullish and (_tight or _stressed):
        return "OVERHEATED"

    # ── FRAGILE (risk-off, no macro confirmation yet) ─────────────────────────
    # Bear/Panic/Crash label but monetary + vol not yet stressed.
    # Represents early deterioration — reduce positions but not full de-risk.
    if _risk_off:
        return "FRAGILE"

    # ── FRAGILE (neutral under stress) ────────────────────────────────────────
    # Neutral/Unknown HMM meets monetary tightening or vol clustering.
    if not _bullish and (_tight or _stressed):
        return "FRAGILE"

    # ── BULL ──────────────────────────────────────────────────────────────────
    # Bullish HMM + accommodative or neutral monetary + stable vol.
    if _bullish:
        return "BULL"

    # ── Conservative default ──────────────────────────────────────────────────
    # Neutral/Unknown HMM with benign macro — preserve capital but don't de-risk.
    return "FRAGILE"


# ── Position scale mapping (Sargent 2011) ─────────────────────────────────────

# Policy regime → suggested allocation fraction.
# Based on Sargent's policy-regime framework: each macro state maps to a
# distinct optimal portfolio weight.
_LAUREATE_SCALES: Dict[str, float] = {
    "BULL":       1.00,   # full allocation — accommodative macro, bullish momentum
    "OVERHEATED": 0.70,   # moderately reduce — monetary headwind, trim longs
    "FRAGILE":    0.40,   # significant reduction — fragile environment, preserve capital
    "CRASH":      0.00,   # de-risk flat — systemic stress, hold cash
}


def laureate_position_scale(label: str) -> float:
    """Sargent (2011 Nobel) — Policy regime → position size scalar.

    Maps the four-state laureate label to a position fraction in [0.0, 1.0].
    Unknown labels default to 0.40 (FRAGILE-equivalent, conservative).

    # $s_t^{\\text{laureate}} \\in \\{0.0, 0.4, 0.7, 1.0\\}$
    """
    key = str(label).upper() if label else "FRAGILE"
    return max(0.0, min(1.0, _LAUREATE_SCALES.get(key, 0.40)))


def combined_position_scale(hmm_scale: float, laureate_label: str) -> float:
    """Combine HMM micro-signal with laureate macro-signal into a final scale.

    Both signals carry independent information:
    - hmm_scale: from RegimeClassifier (state-level Sharpe / drawdown profile)
    - laureate_scale: from the three-signal macro label

    Final scale = hmm_scale × laureate_scale, clipped to [0.0, 1.0].
    This ensures that either signal alone can cut allocation to zero (CRASH → 0).

    # $s_t^{\\text{final}} = \\text{clip}(s_t^{\\text{HMM}} \\cdot s_t^{\\text{laureate}}, 0, 1)$

    Args:
        hmm_scale:      Position scale from RegimeClassifier.predict_current() [0–1].
        laureate_label: Four-state macro label from classify_regime().

    Returns:
        float in [0.0, 1.0].
    """
    hmm_clipped     = max(0.0, min(1.0, float(hmm_scale) if hmm_scale is not None else 1.0))
    laureate_s      = laureate_position_scale(laureate_label)
    return max(0.0, min(1.0, hmm_clipped * laureate_s))


# ── Minsky Moment ──────────────────────────────────────────────────────────────

def minsky_moment(
    garch_persistence: float,
    cape_percentile: float,
    yield_spread_bps: float,
) -> MinskyStatusOut:
    """Minsky (Financial Instability Hypothesis) — Alert when all 3 extremes align.

    Hyman Minsky argued that prolonged stability breeds fragility: agents take on
    excessive leverage during calm periods, making the system brittle.
    The Minsky Moment is the sudden reversal when over-leveraged positions unwind.

    Three Nobel-grounded preconditions (all three must breach for CRITICAL):
    1. GARCH persistence > 0.98 → volatility clustering (Engle 2003 Nobel)
    2. CAPE percentile > 95     → extreme overvaluation (Shiller 2013 Nobel)
    3. Yield spread < 0 bps     → yield curve inverted (Friedman 1968 Nobel)

    Alert levels:
    - CRITICAL (3/3): All conditions met — imminent systemic risk.
    - WARNING  (2/3): Two conditions met — elevated fragility.
    - WATCH    (1/3): One condition met — monitor closely.
    - CLEAR    (0/3): No thresholds breached — normal regime.

    # $\\text{Minsky} = \\mathbf{1}[\\rho > 0.98] \\cdot
    #                   \\mathbf{1}[P_{CAPE} > 95] \\cdot
    #                   \\mathbf{1}[\\Delta < 0]$

    Args:
        garch_persistence: α + β + γ/2 from GJR-GARCH fit.
        cape_percentile:   CAPE percentile vs 40-year history [0–100].
        yield_spread_bps:  10Y−2Y spread in basis points.

    Returns:
        MinskyStatusOut with triggered flag, alert_level, conditions_met, and narrative.
    """
    cond_garch = garch_persistence >= 0.98
    cond_cape  = cape_percentile   >= 95.0
    cond_yield = yield_spread_bps  < 0.0
    conditions_met = int(cond_garch) + int(cond_cape) + int(cond_yield)

    alert_level: Literal["CLEAR", "WATCH", "WARNING", "CRITICAL"] = (
        "CRITICAL" if conditions_met == 3 else
        "WARNING"  if conditions_met == 2 else
        "WATCH"    if conditions_met == 1 else
        "CLEAR"
    )

    triggered = conditions_met == 3

    parts = []
    if cond_garch:
        parts.append(f"GARCH persistence {garch_persistence:.3f} > 0.98 (volatility clustering)")
    if cond_cape:
        parts.append(f"CAPE at {cape_percentile:.0f}th percentile (extreme valuation)")
    if cond_yield:
        parts.append(f"Yield curve inverted ({yield_spread_bps:.0f} bps)")

    narrative = (
        "MINSKY MOMENT TRIGGERED — " + " | ".join(parts)
        if triggered
        else (
            "Elevated risk: " + " | ".join(parts)
            if parts
            else "No Minsky thresholds breached. Regime appears stable."
        )
    )

    return MinskyStatusOut(
        triggered=triggered,
        alert_level=alert_level,
        conditions_met=conditions_met,
        garch_persistence=round(garch_persistence, 4),
        cape_percentile=round(cape_percentile, 2),
        yield_spread_bps=round(yield_spread_bps, 1),
        narrative=narrative,
    )
