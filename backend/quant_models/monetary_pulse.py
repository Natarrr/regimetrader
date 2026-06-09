# Path: backend/quant_models/monetary_pulse.py
"""Monetary regime signals from yield curve and M2 velocity.

Theory:
    Inverted yield curve (GS2 > GS10) has predicted every US recession with
    a 12-24 month lead since 1955 (Harvey 1988). Combined with M2 velocity
    trend gives a richer EXPANSION / TIGHTENING / STAGFLATION / NEUTRAL read.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def yield_spread(gs10: pd.Series, gs2: pd.Series) -> pd.Series:
    """Compute 10Y-2Y yield spread in basis points.

    Positive spread = normal (risk-on). Negative spread = inverted (risk-off).
    Aligns both series to the intersection of their indices.
    """
    aligned = pd.concat([gs10.rename("gs10"), gs2.rename("gs2")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    spread_pct = aligned["gs10"] - aligned["gs2"]
    return (spread_pct * 100).rename("spread_bps")


def is_inverted(spread: pd.Series) -> bool:
    """True when the most recent yield spread reading is negative (2Y > 10Y)."""
    if spread.empty:
        return False
    return float(spread.iloc[-1]) < 0.0


def monetary_regime(spread: pd.Series, m2v: pd.Series) -> str:
    """Classify monetary environment into one of four states.

    Logic:
        EXPANSION   — spread > 0 AND M2V trend rising
        TIGHTENING  — spread < 0 (inverted curve) regardless of M2V
        STAGFLATION — spread > 0 but M2V trend falling (growth with slowing money)
        NEUTRAL     — spread near zero (±25 bps) and M2V flat
    """
    spread_now = float(spread.iloc[-1]) if not spread.empty else 0.0
    m2v_trend  = m2_velocity_trend(m2v)

    if spread_now < -25.0:
        return "TIGHTENING"
    if spread_now > 25.0 and m2v_trend == "RISING":
        return "EXPANSION"
    if spread_now > 0 and m2v_trend == "FALLING":
        return "STAGFLATION"
    return "NEUTRAL"


def m2_velocity_trend(m2v: pd.Series) -> str:
    """Classify M2 velocity as RISING / FALLING / STABLE over the last 4 quarters.

    Threshold: >2% change = directional signal; ≤2% = STABLE.
    """
    if m2v.empty or len(m2v) < 2:
        return "STABLE"
    recent = m2v.dropna()
    if len(recent) < 2:
        return "STABLE"
    last  = float(recent.iloc[-1])
    prior = float(recent.iloc[max(0, len(recent) - 5)])  # ~4 quarters back
    if prior == 0:
        return "STABLE"
    change = (last - prior) / prior
    if change > 0.02:
        return "RISING"
    if change < -0.02:
        return "FALLING"
    return "STABLE"
