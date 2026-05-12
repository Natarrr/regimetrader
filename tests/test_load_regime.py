"""tests/test_load_regime.py
Tests for _load_regime() in regime_trader/ui/streamlit_app.py.

Fama (2013 Nobel) — efficient markets: VIX is the market's expectation of
30-day volatility; a regime classifier built on VIX must be deterministic
given the same reading.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Stub streamlit before any app import ─────────────────────────────────────
def _cache_data_stub(*a, **kw):
    def dec(f):
        f.clear = lambda: None
        return f
    return dec

_st_mock = MagicMock()
_st_mock.cache_data = _cache_data_stub
_st_mock.set_page_config = MagicMock()

sys.modules.setdefault("streamlit", _st_mock)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_vix_df(close: float) -> pd.DataFrame:
    """Build a minimal yfinance-shaped DataFrame for ^VIX."""
    idx = pd.to_datetime(["2026-05-09", "2026-05-10"])
    return pd.DataFrame({"Close": [close - 1, close]}, index=idx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_load_regime():
    """Return _load_regime from the app, reloading to clear st.cache_data state."""
    import importlib
    with patch.dict(sys.modules, {"streamlit": _st_mock}):
        import regime_trader.ui.streamlit_app as app
        importlib.reload(app)
        return app._load_regime


# ── Success paths ─────────────────────────────────────────────────────────────

class TestLoadRegimeSuccess:
    def test_returns_regime_and_vix_keys(self):
        fn = _get_load_regime()
        vix_df = _make_vix_df(18.0)
        with (
            patch("yfinance.download", return_value=vix_df),
            patch("regime_trader.models.regime_detector.vix_rule", return_value="Neutral"),
        ):
            result = fn()
        assert "regime" in result
        assert "vix" in result

    def test_vix_value_matches_last_close(self):
        fn = _get_load_regime()
        vix_df = _make_vix_df(22.5)
        with (
            patch("yfinance.download", return_value=vix_df),
            patch("regime_trader.models.regime_detector.vix_rule", return_value="Bear"),
        ):
            result = fn()
        assert abs(result["vix"] - 22.5) < 1e-6

    def test_regime_label_forwarded_from_vix_rule(self):
        fn = _get_load_regime()
        vix_df = _make_vix_df(35.0)
        with (
            patch("yfinance.download", return_value=vix_df),
            patch("regime_trader.models.regime_detector.vix_rule", return_value="Panic"),
        ):
            result = fn()
        assert result["regime"] == "Panic"

    @pytest.mark.parametrize("vix,expected_regime", [
        (12.0, "Bull"),
        (20.0, "Neutral"),
        (28.0, "Bear"),
        (40.0, "Panic"),
        (60.0, "Crash"),
    ])
    def test_vix_thresholds_passed_to_vix_rule(self, vix, expected_regime):
        fn = _get_load_regime()
        vix_df = _make_vix_df(vix)
        with (
            patch("yfinance.download", return_value=vix_df),
            patch("regime_trader.models.regime_detector.vix_rule", return_value=expected_regime) as mock_rule,
        ):
            result = fn()
        mock_rule.assert_called_once()
        assert result["regime"] == expected_regime


# ── Fallback / error paths ────────────────────────────────────────────────────

class TestLoadRegimeFallback:
    def test_empty_df_returns_unknown(self):
        fn = _get_load_regime()
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = fn()
        assert result["regime"] == "Unknown"
        assert result["vix"] is None

    def test_exception_returns_unknown(self):
        fn = _get_load_regime()
        with patch("yfinance.download", side_effect=RuntimeError("network down")):
            result = fn()
        assert result["regime"] == "Unknown"
        assert result["vix"] is None

    def test_vix_rule_exception_returns_unknown(self):
        fn = _get_load_regime()
        vix_df = _make_vix_df(20.0)
        with (
            patch("yfinance.download", return_value=vix_df),
            patch("regime_trader.models.regime_detector.vix_rule", side_effect=ValueError("bad")),
        ):
            result = fn()
        assert result["regime"] == "Unknown"
