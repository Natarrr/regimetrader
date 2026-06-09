# Path: backend/data/fred_service.py
"""FRED macro data service — yields and M2 velocity.

Data fetched from FRED via pandas_datareader.
Cache TTL: 4 hours (macro signals update daily at most).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_CACHE_DIR  = Path(".cache/fred")
_CACHE_TTL  = 4 * 3600   # 4 hours

_SERIES = {
    "GS10": "10Y Treasury yield (%)",
    "GS2":  "2Y Treasury yield (%)",
    "M2V":  "M2 velocity of money",
}


def _cache_path(series_id: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{series_id}.json"


def _cache_load(series_id: str) -> Optional[pd.Series]:
    path = _cache_path(series_id)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
        if time.time() - blob["ts"] > _CACHE_TTL:
            return None
        return pd.Series(blob["values"], index=pd.to_datetime(blob["index"]))
    except Exception:
        return None


def _cache_save(series_id: str, series: pd.Series) -> None:
    path = _cache_path(series_id)
    blob = {
        "ts":     time.time(),
        "index":  [str(i) for i in series.index],
        "values": list(series.values.astype(float)),
    }
    path.write_text(json.dumps(blob))


def _fetch_series(series_id: str, years_back: int = 5) -> pd.Series:
    cached = _cache_load(series_id)
    if cached is not None:
        return cached
    try:
        import pandas_datareader.data as web  # noqa: PLC0415
        end   = pd.Timestamp.now()
        start = end - pd.DateOffset(years=years_back)
        series = web.DataReader(series_id, "fred", start, end)[series_id].dropna()
        _cache_save(series_id, series)
        return series
    except Exception as exc:
        log.warning("FRED fetch %s failed: %s", series_id, exc)
        return pd.Series(dtype=float)


def fetch_10y_yield(years_back: int = 5) -> pd.Series:
    """10-year Treasury constant maturity yield (GS10, %). FRED."""
    return _fetch_series("GS10", years_back)


def fetch_2y_yield(years_back: int = 5) -> pd.Series:
    """2-year Treasury constant maturity yield (GS2, %). FRED."""
    return _fetch_series("GS2", years_back)


def fetch_m2_velocity(years_back: int = 10) -> pd.Series:
    """M2 money velocity (M2V, quarterly). FRED."""
    return _fetch_series("M2V", years_back)
