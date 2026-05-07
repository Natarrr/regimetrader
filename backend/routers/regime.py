"""backend/routers/regime.py
Module E router — The Prediction Controller (Lucas / Sargent).

GET /api/regime/{symbol} — HMM-based laureate regime for a ticker.
GET /api/regime/minsky   — Composite Minsky Moment alert.
"""
from __future__ import annotations

import backend.dependencies  # noqa: F401

import numpy as np
from fastapi import APIRouter, HTTPException, Path

from backend.data.market_service import MarketData
from backend.data.schemas import MinskyStatusOut, RegimeOut
from backend.quant_models.prediction_controller import classify_regime, minsky_moment

# Lazy imports of existing project modules (available after sys.path injection in dependencies.py)
try:
    from feature_engineering.feature_engineering import FeatureEngineer
    from hmm_engine.classifier import RegimeClassifier
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

router = APIRouter(tags=["Regime Prediction"])

_LOOKBACK_YEARS = 3
_MIN_BARS = 504  # RegimeClassifier requires ≥ 2 trading years


@router.get("/minsky", response_model=MinskyStatusOut)
def get_minsky_status(
    symbol: str = "SPY",
) -> MinskyStatusOut:
    """Lucas (1995 Nobel) + Sargent (2011 Nobel) + Minsky — Composite crisis alert.

    Assembles the three Minsky preconditions from live data:
    1. GJR-GARCH persistence from the requested symbol (default SPY)
    2. Shiller CAPE percentile (from valuation_radar module)
    3. 10Y-2Y yield spread (from monetary_pulse module)
    """
    from backend.routers.monetary import get_monetary_pulse
    from backend.routers.volatility import get_garch
    from backend.routers.valuation import get_cape

    try:
        monetary = get_monetary_pulse()
        garch = get_garch(symbol=symbol)
        cape = get_cape()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data aggregation failed: {exc}")

    return minsky_moment(
        garch_persistence=garch.persistence,
        cape_percentile=cape.cape_percentile,
        yield_spread_bps=monetary.yield_spread_bps,
    )


@router.get("/{symbol}", response_model=RegimeOut)
def get_regime(
    symbol: str = Path(description="Ticker symbol e.g. SPY, QQQ, AAPL"),
) -> RegimeOut:
    """Lucas (1995 Nobel) + Sargent (2011 Nobel) — Composite laureate regime label.

    Fits an HMM regime classifier on historical bars (if available), then combines
    the HMM state, monetary regime, and volatility regime into the four-state
    laureate label: BULL | OVERHEATED | FRAGILE | CRASH.
    """
    if not _HMM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="hmm_engine not importable. Ensure the project root is on sys.path.",
        )

    try:
        md = MarketData(symbol)
        bars = md.get_historical_bars(years=_LOOKBACK_YEARS)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Market data unavailable: {exc}")

    if len(bars) < _MIN_BARS:
        raise HTTPException(
            status_code=422,
            detail=f"Need ≥ {_MIN_BARS} bars for HMM fit; got {len(bars)}.",
        )

    fe = FeatureEngineer()
    features, returns, _ = fe.build(bars)

    clf = RegimeClassifier()
    clf.fit(features, returns)
    state = clf.predict_current(features[-20:])

    hmm_label = state.confirmed_label or state.raw_label or "Unknown"

    from backend.routers.monetary import get_monetary_pulse
    from backend.routers.volatility import get_garch

    try:
        monetary = get_monetary_pulse()
        mon_regime = monetary.monetary_regime
    except Exception:
        mon_regime = "NEUTRAL"

    try:
        garch = get_garch(symbol=symbol)
        vol_regime = garch.volatility_regime
    except Exception:
        vol_regime = "STABLE"

    laureate = classify_regime(hmm_label, mon_regime, vol_regime)

    return RegimeOut(
        symbol=symbol.upper(),
        laureate_regime=laureate,
        hmm_label=hmm_label,
        monetary_regime=mon_regime,
        volatility_regime=vol_regime,
        position_scale=state.position_scale,
        is_uncertain=state.is_uncertain,
    )
