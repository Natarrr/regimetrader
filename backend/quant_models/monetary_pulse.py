"""backend/quant_models/monetary_pulse.py
Module A — The Monetary Pulse (Friedman / Kuznets / Prescott).

All functions are pure: no module-level state, no I/O side effects.
Data fetching is separated into backend/data/fred_service.py; callers
pass in pre-fetched pd.Series so these functions are fully testable
without network access.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


# ── Yield Curve ────────────────────────────────────────────────────────────────

def yield_spread(gs10: pd.Series, gs2: pd.Series) -> pd.Series:
    """Friedman (1968 Nobel) — Monetary tightening leads recession by 12-18M.

    The 10Y-2Y Treasury spread is the canonical leading indicator of credit
    conditions. When it inverts (Δ < 0) the Fed has tightened faster than
    long-run expectations, signalling impending economic contraction.

    # $\Delta_t = r_{10Y,t} - r_{2Y,t}$

    Args:
        gs10: 10-year Treasury CMT yield (% p.a.), FRED GS10.
        gs2:  2-year Treasury CMT yield (% p.a.), FRED GS2.

    Returns:
        Daily/monthly spread series in basis points (× 100 for display).
    """
    aligned = pd.concat([gs10, gs2], axis=1).dropna()
    spread_pct = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    spread_bps = spread_pct * 100
    spread_bps.name = "yield_spread_bps"
    return spread_bps


def is_inverted(spread_bps: pd.Series) -> bool:
    """Friedman (1968 Nobel) — Returns True when the latest spread is negative.

    # $\text{inverted} = \mathbf{1}[\Delta_t < 0]$
    """
    return float(spread_bps.iloc[-1]) < 0.0


# ── M2 Velocity ────────────────────────────────────────────────────────────────

def m2_velocity_trend(m2v: pd.Series, window: int = 4) -> str:
    """Kuznets (1971 Nobel) — National income accounting identifies growth cycles.

    Compares the rolling mean of the most recent `window` quarters against the
    rolling mean of the preceding `window` quarters. Using means on both sides
    removes sensitivity to a single noisy observation at the tail of the series.

    # $V_t = \frac{GDP_t}{M2_t}$

    Args:
        m2v:    M2 velocity series (quarterly), FRED M2V.
        window: Rolling window in quarters for each comparison period (default 4).

    Returns:
        'RISING' | 'FALLING' | 'STABLE'
    """
    if len(m2v) < 2 * window:
        return "STABLE"
    recent = float(m2v.iloc[-window:].mean())
    baseline = float(m2v.iloc[-(2 * window):-window].mean())
    pct_change = (recent - baseline) / abs(baseline)
    if pct_change > 0.02:
        return "RISING"
    if pct_change < -0.02:
        return "FALLING"
    return "STABLE"


# ── HP Filter ──────────────────────────────────────────────────────────────────

def hp_filter_trend(series: pd.Series, lam: int = 1600) -> Tuple[pd.Series, pd.Series]:
    """Prescott (2004 Nobel) — HP filter decomposes cyclical from trend component.

    Standard smoothing parameter: λ=1600 for quarterly data (Hodrick-Prescott 1997).
    Use λ=129600 for monthly, λ=6.25 for annual.

    # $\min_{\tau} \sum_{t}(y_t - \tau_t)^2 + \lambda \sum_{t}(\Delta^2 \tau_t)^2$

    Args:
        series: Time series to decompose (quarterly GDP preferred).
        lam:    Smoothing parameter λ.

    Returns:
        (trend, cycle) — both as pd.Series indexed like `series`.
    """
    clean = series.dropna().astype(float)
    cycle_arr, trend_arr = hpfilter(clean, lamb=lam)
    trend = pd.Series(trend_arr, index=clean.index, name="hp_trend")
    cycle = pd.Series(cycle_arr, index=clean.index, name="hp_cycle")
    return trend, cycle


# ── Monetary Regime ────────────────────────────────────────────────────────────

def monetary_regime(spread_bps: pd.Series, m2v: pd.Series) -> str:
    """Friedman (1968 Nobel) + Kuznets (1971 Nobel) — Classify monetary environment.

    Combines the yield curve signal with M2 velocity trend:
    - Inverted spread + falling velocity → TIGHTENING (credit contraction)
    - Steep spread + rising velocity      → EASING     (credit expansion)
    - Otherwise                           → NEUTRAL

    # $\text{regime} = f(\Delta_t, V_t)$

    Args:
        spread_bps: Yield spread in basis points (from yield_spread()).
        m2v:        Raw M2 velocity series (from FRED M2V).

    Returns:
        'TIGHTENING' | 'NEUTRAL' | 'EASING'
    """
    inverted = is_inverted(spread_bps)
    vel_trend = m2_velocity_trend(m2v)
    latest_spread = float(spread_bps.iloc[-1])

    if inverted or vel_trend == "FALLING":
        return "TIGHTENING"
    if latest_spread > 100 and vel_trend == "RISING":
        return "EASING"
    return "NEUTRAL"
