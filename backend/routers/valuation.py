"""backend/routers/valuation.py
Module C router — The Valuation Radar (Shiller / Thaler).

GET /api/valuation/cape — Shiller CAPE with percentile rank and ECY.
"""
from __future__ import annotations

import backend.dependencies  # noqa: F401

from fastapi import APIRouter, HTTPException

from backend.data.fred_service import fetch_10y_yield, fetch_cpi
from backend.data.schemas import CAPEOut
from backend.quant_models.valuation_radar import (
    fetch_shiller_cape_series,
    cape_percentile,
    excess_cape_yield,
    real_yield,
    is_valuation_danger_zone,
)

router = APIRouter(tags=["Valuation Radar"])


@router.get("/cape", response_model=CAPEOut)
def get_cape() -> CAPEOut:
    """Shiller (2013 Nobel) + Thaler (2017 Nobel) — CAPE and Excess CAPE Yield.

    Fetches Shiller's CAPE dataset, ranks current CAPE against 40-year history,
    and computes the ECY signal for equity vs bond relative value.
    """
    try:
        cape_series = fetch_shiller_cape_series()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CAPE data fetch failed: {exc}")

    if cape_series.empty:
        raise HTTPException(status_code=503, detail="CAPE series unavailable.")

    current_cape = float(cape_series.iloc[-1])

    try:
        gs10 = fetch_10y_yield()
        cpi = fetch_cpi()
        nominal_10y = float(gs10.iloc[-1])
        cpi_12m = float(cpi.pct_change(12).dropna().iloc[-1] * 100)
        real_10y = real_yield(nominal_10y, cpi_12m)
    except Exception:
        real_10y = 0.015  # fallback: 1.5% real yield assumption

    pct = cape_percentile(cape_series, current_cape)
    ecy = excess_cape_yield(current_cape, real_10y)

    return CAPEOut(
        cape=round(current_cape, 2),
        cape_percentile=pct,
        ecy=round(ecy, 6),
        is_danger_zone=is_valuation_danger_zone(pct),
        real_10y_yield=round(real_10y, 4),
    )
