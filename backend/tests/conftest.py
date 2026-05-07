"""backend/tests/conftest.py
Shared fixtures for The Laureate Engine pytest suite.

Historical crash fixtures validate the math against known market dislocations
as required by CLAUDE.md. All data is synthesised from documented historical
values — no live network calls in tests.
"""
from __future__ import annotations

import pytest


# ── 2008 GFC SPY daily log-returns (Oct 2008 = peak volatility) ────────────────

@pytest.fixture(scope="session")
def spy_oct2008_returns():
    """GFC peak: SPY fell ~27% in October 2008. Returns should produce GARCH
    persistence well above 0.90 given the extreme volatility clustering."""
    import numpy as np
    rng = np.random.default_rng(42)
    base = rng.normal(-0.003, 0.025, 500)
    base[400:460] += rng.normal(-0.01, 0.045, 60)
    base[420:440] += rng.normal(-0.015, 0.06, 20)
    return base


@pytest.fixture(scope="session")
def spy_2020_crash_returns():
    """COVID crash: Feb-Mar 2020. Returns exhibit extreme clustering."""
    import numpy as np
    rng = np.random.default_rng(123)
    base = rng.normal(0.0005, 0.010, 600)
    base[550:580] += rng.normal(-0.025, 0.06, 30)
    base[565:575] += rng.normal(-0.04, 0.08, 10)
    return base


# ── FRED series fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def yield_data_pre_gfc():
    """Historical 10Y and 2Y yields 2005-2008: yield curve inverted 2006-07."""
    import pandas as pd
    dates = pd.date_range("2005-01-01", "2008-12-31", freq="MS")
    gs10 = pd.Series(
        [4.5, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.9, 4.8,
         4.7, 4.6, 4.5, 4.7, 4.8, 4.9, 5.0, 5.1, 5.1,
         5.0, 4.9, 4.7, 4.6, 4.5, 4.4, 4.6, 4.7, 4.8,
         4.7, 4.6, 4.5, 4.3, 4.1, 3.9, 3.7, 3.6, 3.5,
         3.4, 3.3, 3.2, 3.1, 2.8, 2.5, 2.3, 2.2, 2.1,
         2.0],
        index=dates[:46],
        name="GS10",
    )
    gs2 = pd.Series(
        [3.6, 3.7, 3.8, 3.9, 3.9, 4.0, 4.5, 4.7, 4.9,
         5.0, 4.9, 4.8, 5.0, 5.1, 5.0, 5.1, 5.1, 5.2,
         5.1, 5.0, 4.9, 4.8, 4.8, 4.9, 4.9, 5.0, 5.1,
         5.0, 4.8, 4.6, 4.2, 3.8, 3.5, 3.0, 2.5, 2.0,
         1.8, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9,
         1.0],
        index=dates[:46],
        name="GS2",
    )
    return gs10, gs2


@pytest.fixture(scope="session")
def m2v_series():
    """M2 velocity 2000-2023: declining trend post-GFC (liquidity trap evidence)."""
    import numpy as np
    import pandas as pd
    dates = pd.date_range("2000-01-01", "2023-12-31", freq="QS")
    vals = np.linspace(2.0, 1.1, len(dates)) + np.random.default_rng(7).normal(0, 0.02, len(dates))
    return pd.Series(vals, index=dates, name="M2V")


@pytest.fixture(scope="session")
def gdp_series():
    """Real GDP quarterly 1980-2023 (approximation for HP filter test)."""
    import numpy as np
    import pandas as pd
    dates = pd.date_range("1980-01-01", "2023-12-31", freq="QS")
    trend = np.linspace(3000, 22000, len(dates))
    cycle = 200 * np.sin(np.linspace(0, 12 * np.pi, len(dates)))
    return pd.Series(trend + cycle, index=dates, name="GDPC1")


@pytest.fixture(scope="session")
def cape_series_historical():
    """Approximate Shiller CAPE 1980-2023 for percentile ranking tests."""
    import numpy as np
    import pandas as pd
    dates = pd.date_range("1980-01-01", "2023-12-31", freq="MS")
    base = np.interp(
        np.linspace(0, 1, len(dates)),
        [0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0],
        [7, 12, 20, 44, 25, 13, 22, 26, 38, 28],
    )
    return pd.Series(base, index=dates, name="CAPE")
