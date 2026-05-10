"""tests/test_thread_loaders.py
Tests for _load_commodity_prices() and _load_macro_indicators() thread-pool
loaders in regime_trader/ui/streamlit_app.py.

Black / Scholes / Merton (1997 Nobel) — bounded computation time is as
important as bounded risk; the timeout guard prevents a single slow data
source from stalling the entire dashboard render.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Stub streamlit ─────────────────────────────────────────────────────────────
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


def _get_app():
    import importlib
    with patch.dict(sys.modules, {"streamlit": _st_mock}):
        import regime_trader.ui.streamlit_app as app
        importlib.reload(app)
        return app


# ── Commodity loader ──────────────────────────────────────────────────────────

class TestLoadCommodityPrices:
    def test_happy_path_returns_dict_keyed_by_ticker(self):
        app = _get_app()

        fake_universe = [
            {"ticker": "GC=F", "name": "Gold"},
            {"ticker": "CL=F", "name": "Crude Oil"},
        ]
        fake_prices = {"price": 1900.0, "ret_1d": 0.01, "ret_5d": 0.02}

        with (
            patch("regime_trader.market_intel_macro.COMMODITY_UNIVERSE", fake_universe),
            patch("regime_trader.market_intel_macro.fetch_commodity_prices",
                  return_value=fake_prices),
        ):
            result = app._load_commodity_prices()

        assert "GC=F" in result
        assert "CL=F" in result
        assert result["GC=F"]["price"] == 1900.0

    def test_fetch_exception_marks_ticker_as_none(self):
        app = _get_app()

        fake_universe = [{"ticker": "GC=F", "name": "Gold"}]

        def boom(commodity):
            raise RuntimeError("feed down")

        with (
            patch("regime_trader.market_intel_macro.COMMODITY_UNIVERSE", fake_universe),
            patch("regime_trader.market_intel_macro.fetch_commodity_prices", side_effect=boom),
        ):
            result = app._load_commodity_prices()

        assert result["GC=F"] is None

    def test_partial_failure_returns_partial_dict(self):
        app = _get_app()

        fake_universe = [
            {"ticker": "GC=F", "name": "Gold"},
            {"ticker": "CL=F", "name": "Crude Oil"},
        ]
        fake_prices = {"price": 1900.0, "ret_1d": 0.0, "ret_5d": 0.0}

        call_count = {"n": 0}

        def maybe_fail(commodity):
            call_count["n"] += 1
            if commodity["ticker"] == "CL=F":
                raise RuntimeError("offline")
            return fake_prices

        with (
            patch("regime_trader.market_intel_macro.COMMODITY_UNIVERSE", fake_universe),
            patch("regime_trader.market_intel_macro.fetch_commodity_prices",
                  side_effect=maybe_fail),
        ):
            result = app._load_commodity_prices()

        assert result["GC=F"] is not None
        assert result["CL=F"] is None

    def test_empty_universe_returns_empty_dict(self):
        app = _get_app()

        with (
            patch("regime_trader.market_intel_macro.COMMODITY_UNIVERSE", []),
            patch("regime_trader.market_intel_macro.fetch_commodity_prices",
                  return_value={}),
        ):
            result = app._load_commodity_prices()

        assert result == {}


# ── Macro indicator loader ─────────────────────────────────────────────────────

class TestLoadMacroIndicators:
    def test_happy_path_returns_dict_keyed_by_ticker(self):
        app = _get_app()

        fake_indicators = [
            {"ticker": "^VIX", "name": "VIX"},
            {"ticker": "DX-Y.NYB", "name": "DXY"},
        ]
        fake_data = {"price": 18.5, "ret_1d": -0.02, "ret_5d": 0.01}

        with (
            patch("regime_trader.market_intel_macro.MACRO_INDICATORS", fake_indicators),
            patch("regime_trader.market_intel_macro.fetch_macro_indicator",
                  return_value=fake_data),
        ):
            result = app._load_macro_indicators()

        assert "^VIX" in result
        assert "DX-Y.NYB" in result

    def test_fetch_exception_marks_ticker_as_none(self):
        app = _get_app()

        fake_indicators = [{"ticker": "^VIX", "name": "VIX"}]

        with (
            patch("regime_trader.market_intel_macro.MACRO_INDICATORS", fake_indicators),
            patch("regime_trader.market_intel_macro.fetch_macro_indicator",
                  side_effect=ConnectionError("timeout")),
        ):
            result = app._load_macro_indicators()

        assert result["^VIX"] is None

    def test_partial_failure_returns_partial_dict(self):
        app = _get_app()

        fake_indicators = [
            {"ticker": "^VIX", "name": "VIX"},
            {"ticker": "DX-Y.NYB", "name": "DXY"},
        ]
        fake_data = {"price": 18.5, "ret_1d": 0.0, "ret_5d": 0.0}

        def maybe_fail(ticker):
            if ticker == "DX-Y.NYB":
                raise RuntimeError("feed down")
            return fake_data

        with (
            patch("regime_trader.market_intel_macro.MACRO_INDICATORS", fake_indicators),
            patch("regime_trader.market_intel_macro.fetch_macro_indicator",
                  side_effect=maybe_fail),
        ):
            result = app._load_macro_indicators()

        assert result["^VIX"] is not None
        assert result["DX-Y.NYB"] is None


# ── Timeout guard (fast simulation) ──────────────────────────────────────────

class TestLoaderTimeoutHandling:
    @pytest.mark.slow
    def test_commodity_timeout_marks_remaining_none(self):
        """Simulate a slow fetch that exceeds the 30-second wall timeout.

        Marked slow — runs only on protected branches / local dev.
        Uses a very short timeout override via monkeypatching as_completed.
        """
        app = _get_app()

        fake_universe = [
            {"ticker": "GC=F", "name": "Gold"},
            {"ticker": "SLOW", "name": "Slow Feed"},
        ]

        completed_count = {"n": 0}

        original_as_completed = __import__(
            "concurrent.futures", fromlist=["as_completed"]
        ).as_completed

        def patched_as_completed(fs, timeout=None):
            # Only yield first future; simulate timeout by stopping early
            for i, f in enumerate(fs):
                if i >= 1:
                    from concurrent.futures import TimeoutError as FTE
                    raise FTE()
                yield f

        import concurrent.futures as cf

        fast_data = {"price": 1900.0, "ret_1d": 0.0, "ret_5d": 0.0}

        def fetch(commodity):
            if commodity["ticker"] == "SLOW":
                time.sleep(0.05)
            return fast_data

        with (
            patch("regime_trader.market_intel_macro.COMMODITY_UNIVERSE", fake_universe),
            patch("regime_trader.market_intel_macro.fetch_commodity_prices",
                  side_effect=fetch),
            patch("regime_trader.ui.streamlit_app.as_completed", patched_as_completed),
        ):
            result = app._load_commodity_prices()

        # At least one ticker was completed before "timeout", one is None
        none_count = sum(1 for v in result.values() if v is None)
        assert none_count >= 1, "Timeout path should mark at least one ticker None"
