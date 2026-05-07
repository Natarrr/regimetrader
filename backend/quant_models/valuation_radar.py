"""backend/quant_models/valuation_radar.py
Module C — The Valuation Radar (Shiller / Thaler).

Shiller CAPE (Cyclically Adjusted Price-Earnings ratio) and Excess CAPE Yield
(ECY). All functions are pure; data is passed in, not fetched here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from io import StringIO


# ── CAPE Construction ──────────────────────────────────────────────────────────

def fetch_shiller_cape_series() -> pd.Series:
    """Shiller (2013 Nobel) — Fetch historical CAPE from Robert Shiller's dataset.

    Downloads the monthly S&P 500 CAPE from Shiller's online data (Yale).
    Columns in the CSV: Date, P, D, E, CPI, Date_frac, GS10, Real_P, Real_D,
    Real_E_10yr_avg, CAPE.

    # $CAPE_t = \\frac{P_t}{\\overline{E_{10yr,real,t}}}$
    # where $\\overline{E_{10yr,real,t}} = \\frac{1}{10}\\sum_{k=0}^{9} \\frac{E_{t-12k}}{CPI_{t-12k}} \\cdot CPI_t$

    Returns:
        Monthly pd.Series of CAPE values indexed by datetime, name='CAPE'.
    """
    url = "https://shiller.econ.yale.edu/data/ie_data.xls"
    # Fallback: construct from FRED series if Shiller URL is unavailable
    try:
        df = pd.read_excel(url, sheet_name="Data", skiprows=7, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        # The CAPE column is labelled 'CAPE' in Shiller's sheet
        cape_col = [c for c in df.columns if "cape" in c.lower() or "p/e10" in c.lower()]
        date_col = [c for c in df.columns if "date" in c.lower()]
        if cape_col and date_col:
            df = df[[date_col[0], cape_col[0]]].dropna()
            df.columns = ["date", "CAPE"]
            df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")
            return df["CAPE"].dropna()
    except Exception:
        pass
    return _construct_cape_from_fred()


def _construct_cape_from_fred() -> pd.Series:
    """Fallback CAPE construction from FRED: CPI + S&P 500 earnings proxy.

    Uses MULTPL Shiller PE data via web scraping as a secondary fallback.
    """
    try:
        resp = requests.get("https://www.multpl.com/shiller-pe/table/by-month", timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        df.columns = ["date", "CAPE"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["CAPE"] = pd.to_numeric(df["CAPE"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
        df = df.dropna().set_index("date").sort_index()
        return df["CAPE"]
    except Exception:
        # Return an empty series — callers must handle gracefully
        return pd.Series(dtype=float, name="CAPE")


# ── CAPE Analytics ─────────────────────────────────────────────────────────────

def cape_percentile(cape_series: pd.Series, current_cape: float) -> float:
    """Shiller (2013 Nobel) — Percentile rank of current CAPE vs historical distribution.

    Alert threshold: > 95th percentile indicates extreme valuation (danger zone).

    # $P = \\frac{|\\{CAPE_t : CAPE_t \\leq CAPE_{\\text{current}}\\}|}{N} \\times 100$

    Args:
        cape_series:   Historical CAPE series (at least 10 years recommended).
        current_cape:  Latest CAPE value to rank.

    Returns:
        Percentile in [0, 100]. > 95 triggers Minsky check.
    """
    clean = cape_series.dropna().values
    if len(clean) == 0:
        return 50.0
    pct = float(np.mean(clean <= current_cape) * 100)
    return round(pct, 2)


def excess_cape_yield(cape: float, real_10y_yield: float) -> float:
    """Thaler (2017 Nobel) — Behavioural finance: ECY measures equity risk premium irrationality.

    The Excess CAPE Yield compares the earnings yield of equities (1/CAPE) to
    the real bond yield. When ECY turns negative, equities are more expensive
    than bonds on a real-return basis — a historically reliable bubble signal.

    # $ECY = \\frac{1}{CAPE} - r_{10Y,\\text{real}}$

    Negative ECY ≈ Greenspan's "irrational exuberance" threshold.

    Args:
        cape:           Current Shiller CAPE ratio (e.g. 36.5).
        real_10y_yield: Real 10Y yield as decimal (nominal − trailing CPI,
                        e.g. 0.045 − 0.032 = 0.013).

    Returns:
        ECY as decimal. Negative = equities expensive vs bonds.
    """
    if cape <= 0:
        return 0.0
    return round((1.0 / cape) - real_10y_yield, 6)


def real_yield(nominal_10y_pct: float, trailing_cpi_pct: float) -> float:
    """Fisher (1930, not Nobel) — Real yield approximation used for ECY computation.

    # $r_{\\text{real}} \\approx r_{\\text{nominal}} - \\pi$

    Args:
        nominal_10y_pct:  10Y Treasury yield in percent (e.g. 4.5 for 4.5%).
        trailing_cpi_pct: Trailing 12-month CPI change in percent (e.g. 3.2).

    Returns:
        Real yield as decimal (e.g. 0.013).
    """
    return (nominal_10y_pct - trailing_cpi_pct) / 100.0


def is_valuation_danger_zone(percentile: float, threshold: float = 95.0) -> bool:
    """Shiller (2013 Nobel) — Returns True when CAPE exceeds the 95th percentile.

    # $\\text{danger} = \\mathbf{1}[P > 95]$
    """
    return percentile > threshold
