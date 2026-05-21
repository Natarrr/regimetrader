"""backend/data/schemas.py
Pydantic response models for The Laureate Engine API.

Mirrors the frozen dataclasses in core/models.py as JSON-serialisable
BaseModel subclasses so FastAPI can validate and document every endpoint.
"""
from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


# ── Module A — Monetary Pulse ──────────────────────────────────────────────────

class MonetaryPulseOut(BaseModel):
    """Friedman (1968 Nobel) + Kuznets (1971 Nobel) monetary regime snapshot."""
    yield_spread_bps: float = Field(description="10Y minus 2Y spread in basis points")
    is_inverted: bool = Field(description="True when spread < 0")
    m2_velocity: float = Field(description="Latest M2 money velocity (GDP/M2)")
    m2_velocity_trend: str = Field(description="RISING | FALLING | STABLE")
    monetary_regime: str = Field(description="TIGHTENING | NEUTRAL | EASING")
    hp_cycle_value: float = Field(description="HP-filter cyclical component of GDP (latest)")


# ── Module B — Volatility Brain ────────────────────────────────────────────────

class GARCHOut(BaseModel):
    """Engle (2003 Nobel) GJR-GARCH(1,1) parameter estimates."""
    symbol: str
    omega: float = Field(description="Constant variance term ω")
    alpha: float = Field(description="ARCH coefficient α (symmetric shock)")
    gamma: float = Field(description="GJR leverage term γ (asymmetric negative shock)")
    beta: float = Field(description="GARCH persistence coefficient β")
    persistence: float = Field(description="α + β + γ/2; > 0.98 triggers Minsky check")
    volatility_regime: str = Field(description="CLUSTERING | STABLE")
    latest_conditional_vol_ann: float = Field(description="Annualised conditional σ (current)")


class MertonOut(BaseModel):
    """Merton (1997 Nobel) Distance-to-Default for systemic bank risk."""
    symbol: str
    asset_value: float = Field(description="Implied firm asset value V")
    asset_vol: float = Field(description="Implied asset volatility σ_V")
    d2d: float = Field(description="Distance-to-Default; < 1.5 = distress zone")
    prob_default: float = Field(description="Risk-neutral default probability N(-D2D)")


# ── Module C — Valuation Radar ─────────────────────────────────────────────────

class CAPEOut(BaseModel):
    """Shiller (2013 Nobel) CAPE + Thaler (2017 Nobel) Excess CAPE Yield."""
    cape: float = Field(description="Current Shiller CAPE ratio")
    cape_percentile: float = Field(description="Percentile vs 40-year history [0–100]")
    ecy: float = Field(description="Excess CAPE Yield = 1/CAPE − real 10Y yield")
    is_danger_zone: bool = Field(description="True when cape_percentile > 95")
    real_10y_yield: float = Field(description="10Y yield minus trailing CPI inflation")


# ── Module D — Contagion Web ───────────────────────────────────────────────────

class ContagionOut(BaseModel):
    """Leontief (1973 Nobel) + Tirole (2014 Nobel) supply-chain shock propagation."""
    shock_sector: str = Field(description="Sector that received the initial −20% demand shock")
    sector_impacts: Dict[str, float] = Field(description="{sector: pct_output_impact}")
    critical_nodes: List[str] = Field(description="Top 3 sectors by Leontief multiplier")
    total_gdp_impact_pct: float = Field(description="Estimated aggregate GDP impact %")


# ── Module E — Prediction Controller ──────────────────────────────────────────

class MinskyStatusOut(BaseModel):
    """Composite Minsky Moment indicator (Engle + Shiller + Friedman thresholds)."""
    triggered: bool = Field(description="True when ALL three threshold conditions are met")
    alert_level: Literal["CLEAR", "WATCH", "WARNING", "CRITICAL"]
    conditions_met: int = Field(description="Number of 3 thresholds currently breached [0–3]")
    garch_persistence: float
    cape_percentile: float
    yield_spread_bps: float
    narrative: str = Field(description="Human-readable summary of current risk posture")


class RegimeOut(BaseModel):
    """Lucas (1995 Nobel) + Sargent (2011 Nobel) composite regime label."""
    symbol: str
    laureate_regime: Literal["BULL", "OVERHEATED", "FRAGILE", "CRASH"]
    hmm_label: str = Field(description="Raw HMM regime label from RegimeClassifier")
    monetary_regime: str
    volatility_regime: str
    position_scale: float = Field(description="Suggested position sizing scalar [0.0–1.0]")
    is_uncertain: bool
