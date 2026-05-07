"""backend/utils/volatility.py
Canonical volatility unit conversion — Engle (2003 Nobel).

Single source of truth for the arch/GARCH conditional-variance → annualised
percentage volatility pipeline. Import this in backend modules instead of
inlining the conversion arithmetic to prevent double-annualisation and
off-by-100² errors.

Unit convention (project-wide default: units="percent")
-------------------------------------------------------
units="percent"  Model was fitted on returns × 100 (%-pt scale).
                 h_t is in %-pt² (e.g. 0.8 = 0.8 %-pt²/day).
                 Returns annualised vol in plain percent (e.g. 10.25).

units="decimal"  Model was fitted on raw log-returns (decimal scale).
                 h_t is in decimal² (e.g. 0.00008 = 0.008%²/day).
                 Returns annualised vol in plain percent (e.g. 10.25).

To change to decimal units: pass units="decimal" everywhere h_t originates
from a model estimated on un-scaled log-returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualise_vol_from_condvar(
    h_t: "np.ndarray | pd.Series",
    units: str = "percent",
) -> pd.Series:
    """Engle (2003 Nobel) — Convert daily conditional variance to annualised vol%.

    GJR-GARCH(1,1) variance recurrence:
      h_t = omega + (alpha + gamma * 1[eps<0]) * eps^2_{t-1} + beta * h_{t-1}

    Unit pipeline (units="percent"):
      h_t [%-pt^2]  ->  sqrt(h_t) [%-pt/day]  ->  * sqrt(252)  [%-pt/yr]

    Unit pipeline (units="decimal"):
      h_t [dec^2]  ->  * 10^4  [%-pt^2]  ->  sqrt  ->  * sqrt(252)  [%-pt/yr]

    Args:
        h_t:   1-D array / Series of daily conditional variance.
               • units="percent" → h_t in %-pt²  (model estimated on r × 100)
               • units="decimal" → h_t in decimal² (model estimated on raw log-ret)
        units: "percent" (default) or "decimal".

    Returns:
        pd.Series of annualised volatility as plain percentage (10.25 = 10.25%/yr).

    Raises:
        ValueError: if units is not "percent" or "decimal".
    """
    if units not in ("percent", "decimal"):
        raise ValueError(f"units must be 'percent' or 'decimal', got {units!r}")

    h = pd.Series(h_t, dtype=float)

    if units == "decimal":
        # decimal² → %-pt²: multiply variance by (100)²
        h = h * (100.0 ** 2)

    # h is now in %-pt² → daily std in %-pt → annualise
    daily_std_pct = np.sqrt(h)                           # %-pt / day
    annual_std_pct = daily_std_pct * np.sqrt(TRADING_DAYS)
    return annual_std_pct
