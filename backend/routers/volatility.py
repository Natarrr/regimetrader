"""backend/routers/volatility.py
Module B router — The Volatility Brain (Engle / Merton).

GET /api/volatility/{symbol}         — GJR-GARCH persistence for any ticker.
GET /api/volatility/merton/{symbol}  — Merton Distance-to-Default.
"""
from __future__ import annotations

import backend.dependencies  # noqa: F401

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query

from backend.data.market_service import MarketData
from backend.data.schemas import GARCHOut, MertonOut
from backend.quant_models.volatility_brain import (
    fit_gjr_garch,
    volatility_regime,
    merton_distance_to_default,
)

router = APIRouter(tags=["Volatility Brain"])

_LOOKBACK_YEARS = 5


@router.get("/{symbol}", response_model=GARCHOut)
def get_garch(
    symbol: str = Path(description="Ticker symbol e.g. SPY, AAPL"),
    years: int = Query(default=_LOOKBACK_YEARS, ge=2, le=10),
) -> GARCHOut:
    """Engle (2003 Nobel) — GJR-GARCH(1,1) volatility clustering analysis.

    Fetches daily OHLCV, computes log-returns, fits GJR-GARCH, and returns
    parameter estimates including the persistence metric used in the Minsky check.
    """
    try:
        md = MarketData(symbol)
        bars = md.get_historical_bars(years=years)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Market data unavailable: {exc}")

    if len(bars) < 100:
        raise HTTPException(status_code=422, detail="Insufficient data for GARCH fit (< 100 bars).")

    log_returns = np.log(bars["Close"] / bars["Close"].shift(1)).dropna().values

    try:
        result = fit_gjr_garch(log_returns)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GARCH fit failed: {exc}")

    return GARCHOut(
        symbol=symbol.upper(),
        omega=round(result["omega"], 8),
        alpha=round(result["alpha"], 6),
        gamma=round(result["gamma"], 6),
        beta=round(result["beta"], 6),
        persistence=round(result["persistence"], 6),
        volatility_regime=volatility_regime(result["persistence"]),
        latest_conditional_vol_ann=round(result["latest_conditional_vol_ann"], 4),
    )


@router.get("/merton/{symbol}", response_model=MertonOut)
def get_merton(
    symbol: str = Path(description="Ticker symbol for D2D computation"),
    face_value_debt: float = Query(description="Total face value of liabilities (same currency as market cap)"),
    risk_free_rate: float = Query(default=0.045, description="Annual risk-free rate as decimal"),
) -> MertonOut:
    """Merton (1997 Nobel) — Distance-to-Default for systemic solvency risk.

    Uses live market cap and equity volatility from yfinance; caller supplies
    face_value_debt (total liabilities from balance sheet).
    """
    try:
        md = MarketData(symbol)
        bars = md.get_historical_bars(years=3)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Market data unavailable: {exc}")

    log_returns = np.log(bars["Close"] / bars["Close"].shift(1)).dropna().values
    equity_vol = float(np.std(log_returns) * np.sqrt(252))
    current_price = float(bars["Close"].iloc[-1])

    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        shares = info.get("sharesOutstanding", 1e9)
        equity_value = current_price * shares
    except Exception:
        equity_value = current_price * 1e9  # fallback: assume 1B shares

    result = merton_distance_to_default(
        equity_value=equity_value,
        face_value_debt=face_value_debt,
        risk_free_rate=risk_free_rate,
        equity_vol=equity_vol,
    )

    return MertonOut(
        symbol=symbol.upper(),
        asset_value=round(result["asset_value"], 2),
        asset_vol=round(result["asset_vol"], 4),
        d2d=round(result["d2d"], 4),
        prob_default=round(result["prob_default"], 6),
    )
