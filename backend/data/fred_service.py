"""backend/data/fred_service.py
FRED data ingestion service for The Laureate Engine.

Uses FRED's public CSV endpoint directly (no API key, no pandas_datareader).
pandas_datareader is broken on Python 3.14 (deprecate_kwarg signature mismatch),
so we hit https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES&vintage_date=END
with plain requests + pandas.read_csv.
"""
from __future__ import annotations

from datetime import datetime
from io import StringIO

import pandas as pd
import requests

_DEFAULT_START = "1980-01-01"
_DEFAULT_END = datetime.today().strftime("%Y-%m-%d")
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT = 20


def _fred(series_id: str, start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> pd.Series:
    """Fetch a single FRED series via the public CSV endpoint (no API key required).

    Endpoint: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID
    Returns observations from `start` to `end`, NaN rows dropped.
    """
    params = {"id": series_id}
    resp = requests.get(_FRED_CSV, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), index_col=0)
    df.columns = [series_id]
    df = df.replace(".", float("nan")).astype(float).dropna()
    df.index = pd.to_datetime(df.index)
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    s = df.loc[mask, series_id]
    s.name = series_id
    return s


# ── Treasury yields ────────────────────────────────────────────────────────────

def fetch_10y_yield(start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> pd.Series:
    """Friedman (1968 Nobel) — 10-year Treasury constant-maturity yield (% p.a.).
    FRED series GS10.
    """
    return _fred("GS10", start, end)


def fetch_2y_yield(start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> pd.Series:
    """Friedman (1968 Nobel) — 2-year Treasury constant-maturity yield (% p.a.).
    FRED series GS2.
    """
    return _fred("GS2", start, end)


# ── Money supply ───────────────────────────────────────────────────────────────

def fetch_m2_velocity(start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> pd.Series:
    """Kuznets (1971 Nobel) — M2 money velocity (quarterly).
    # $V_t = \frac{GDP_t}{M2_t}$
    FRED series M2V. Declining velocity implies liquidity trap risk.
    """
    return _fred("M2V", start, end)


# ── Inflation ──────────────────────────────────────────────────────────────────

def fetch_cpi(start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> pd.Series:
    """Shiller (2013 Nobel) — CPI-U All Urban (monthly index, 1982–84 = 100).
    Used to deflate S&P 500 earnings for CAPE construction.
    FRED series CPIAUCSL.
    """
    return _fred("CPIAUCSL", start, end)


# ── GDP ────────────────────────────────────────────────────────────────────────

def fetch_real_gdp(start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> pd.Series:
    """Prescott (2004 Nobel) — Real GDP (billions of chained 2017 $, quarterly).
    FRED series GDPC1. Used as numerator for M2 velocity and HP filter input.
    """
    return _fred("GDPC1", start, end)
