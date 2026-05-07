"""backend/routers/monetary.py
Module A router — The Monetary Pulse (Friedman / Kuznets / Prescott).

GET /api/monetary/pulse — returns a full MonetaryPulseOut snapshot.
"""
from __future__ import annotations

import backend.dependencies  # noqa: F401 — ensures sys.path is set

from fastapi import APIRouter, HTTPException

from backend.data.fred_service import (
    fetch_10y_yield,
    fetch_2y_yield,
    fetch_m2_velocity,
    fetch_real_gdp,
)
from backend.data.schemas import MonetaryPulseOut
from backend.quant_models.monetary_pulse import (
    yield_spread,
    is_inverted,
    m2_velocity_trend,
    hp_filter_trend,
    monetary_regime,
)

router = APIRouter(tags=["Monetary Pulse"])


@router.get("/pulse", response_model=MonetaryPulseOut)
def get_monetary_pulse() -> MonetaryPulseOut:
    """Friedman (1968 Nobel) + Kuznets (1971 Nobel) + Prescott (2004 Nobel).

    Fetches live FRED data and returns the current monetary regime snapshot:
    - 10Y-2Y yield spread (basis points)
    - M2 velocity trend
    - HP-filter cyclical component of real GDP
    - Composite monetary regime label
    """
    try:
        gs10 = fetch_10y_yield()
        gs2 = fetch_2y_yield()
        m2v = fetch_m2_velocity()
        gdp = fetch_real_gdp()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"FRED data fetch failed: {exc}")

    spread = yield_spread(gs10, gs2)
    spread_latest = float(spread.iloc[-1])

    vel_trend = m2_velocity_trend(m2v)
    m2v_latest = float(m2v.iloc[-1])

    _, cycle = hp_filter_trend(gdp)
    hp_cycle_latest = float(cycle.iloc[-1])

    regime = monetary_regime(spread, m2v)

    return MonetaryPulseOut(
        yield_spread_bps=round(spread_latest, 1),
        is_inverted=is_inverted(spread),
        m2_velocity=round(m2v_latest, 4),
        m2_velocity_trend=vel_trend,
        monetary_regime=regime,
        hp_cycle_value=round(hp_cycle_latest, 2),
    )
