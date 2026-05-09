"""
Sanity tests — every package in requirements-ci.txt must be importable.

If any of these fail in CI it means requirements-ci.txt is out of sync
with the actual imports in the test suite or production modules.

Based on Black-Scholes / import-graph invariants (Merton 1973 Nobel Prize):
every dependency node must resolve before analysis can proceed.
"""


def test_pydantic():
    """Pydantic: data validation used by backend/data/schemas.py."""
    import pydantic

    assert pydantic.__version__ >= "2"


def test_requests():
    """Requests: HTTP client used by valuation_radar and discovery_scanner."""
    import requests  # noqa: F401


def test_numpy():
    import numpy as np

    assert np.__version__


def test_pandas():
    import pandas as pd

    assert pd.__version__


def test_scipy():
    """SciPy: Merton distance-to-default in volatility_brain."""
    from scipy import stats  # noqa: F401


def test_statsmodels():
    """Statsmodels: yield-spread OLS in monetary_pulse."""
    import statsmodels.api  # noqa: F401


def test_hmmlearn():
    """hmmlearn: HMM regime detector (Sargent 2011 application)."""
    from hmmlearn import hmm  # noqa: F401


def test_sklearn():
    """scikit-learn: ML regime classifier in regime_detector."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401


def test_arch():
    """arch: GJR-GARCH volatility model (Engle 2003 Nobel)."""
    from arch import arch_model  # noqa: F401


def test_anthropic():
    """anthropic: Anthropic SDK for ClaudeClient wrapper."""
    import anthropic  # noqa: F401


def test_yfinance():
    """yfinance: market data for regime detection and credit signals."""
    import yfinance  # noqa: F401


def test_requests_mock():
    """requests-mock: HTTP mocking in discovery_scanner tests."""
    import requests_mock  # noqa: F401


def test_dotenv():
    """python-dotenv: .env loading for local dev."""
    import dotenv  # noqa: F401
