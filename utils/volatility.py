"""utils/volatility.py
Canonical volatility unit conversion — Engle (2003 Nobel).

Single source of truth for translating arch/GARCH conditional-variance series
into annualised percentage volatility. Import this instead of inlining the
conversion arithmetic to avoid double-annualisation and off-by-100² errors.
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

    GJR-GARCH(1,1) conditional variance recurrence:
    $h_t = \\omega + (\\alpha + \\gamma \\mathbf{1}[\\varepsilon_{t-1}<0])
           \\varepsilon^2_{t-1} + \\beta h_{t-1}$

    Daily std (same units as returns): $\\sigma_t = \\sqrt{h_t}$
    Annualised vol (percent):          $\\sigma_{ann} = \\sigma_t \\cdot \\sqrt{T}$

    Args:
        h_t:   1-D array / Series of daily conditional **variance** in the same
               units as the returns the model was estimated on.
               - units="percent"  → h_t in %-pt²   (model fitted on r×100)
               - units="decimal"  → h_t in decimal² (model fitted on raw log-ret)
        units: "percent" (default) or "decimal".

    Returns:
        pd.Series of annualised volatility expressed as plain percentage points
        (e.g. 10.25 means 10.25% per year, NOT 0.1025).

    Raises:
        ValueError: if units is not "percent" or "decimal".
    """
    if units not in ("percent", "decimal"):
        raise ValueError(f"units must be 'percent' or 'decimal', got {units!r}")

    h = pd.Series(h_t, dtype=float)

    if units == "decimal":
        # decimal² → %-pt²:  multiply variance by (100)²
        h = h * (100.0 ** 2)

    # h is now in %-pt² → daily std in %-pt → annualise
    daily_std_pct = np.sqrt(h)                          # %-pt / day
    annual_std_pct = daily_std_pct * np.sqrt(TRADING_DAYS)
    return annual_std_pct
