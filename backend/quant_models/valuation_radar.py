# Path: backend/quant_models/valuation_radar.py
"""Shiller CAPE valuation signal.

Theory:
    Shiller (2000) — "Irrational Exuberance". CAPE = price / (10-year avg real EPS).
    CAPE percentile vs historical distribution identifies valuation extremes.
    A CAPE in the 95th percentile of its own history signals over-valuation.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_SHILLER_URL = "https://shiller.econ.yale.edu/data/ie_data.xls"


def fetch_shiller_cape_series() -> pd.Series:
    """Download Shiller monthly CAPE data from Yale.

    Falls back to SPY trailing P/E proxy (via yfinance) if the Shiller file
    is unavailable (network timeout or URL change).

    Returns a pd.Series of monthly CAPE values indexed by date.
    """
    try:
        df = pd.read_excel(
            _SHILLER_URL,
            sheet_name="Data",
            skiprows=7,
            header=0,
            usecols="A,E",
        )
        df.columns = ["date_raw", "cape"]
        df = df.dropna(subset=["cape"])
        df = df[pd.to_numeric(df["cape"], errors="coerce").notna()]
        df["cape"] = pd.to_numeric(df["cape"])
        df["date"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y.%m",
                                    errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")["cape"].sort_index()
        log.debug("Shiller CAPE: %d records, latest %.1f", len(df), df.iloc[-1])
        return df
    except Exception as exc:
        log.warning("Shiller CAPE download failed (%s) — using SPY P/E fallback", exc)
        return _spy_pe_fallback()


def _spy_pe_fallback() -> pd.Series:
    """Approximate CAPE from SPY trailing P/E via yfinance."""
    try:
        import yfinance as yf  # noqa: PLC0415
        spy = yf.Ticker("SPY")
        pe  = spy.info.get("trailingPE")
        if pe and pe > 0:
            now = pd.Timestamp.now().normalize()
            return pd.Series([float(pe)], index=[now])
    except Exception as exc:
        log.debug("SPY P/E fallback failed: %s", exc)
    return pd.Series(dtype=float)


def cape_percentile(series: pd.Series, current: Optional[float] = None) -> float:
    """Return the current CAPE value's percentile rank within *series*.

    If *current* is None, uses the most recent value in *series*.
    Returns 0.0 if series has fewer than 10 observations.
    """
    if series.empty or len(series) < 10:
        return 0.0
    val = current if current is not None else float(series.iloc[-1])
    below = (series < val).sum()
    return round(float(below) / len(series) * 100, 1)
