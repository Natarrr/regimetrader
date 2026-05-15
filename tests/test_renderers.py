"""tests/test_renderers.py
Renderer smoke tests for regime_trader/ui/streamlit_app.py.

Monkeypatches the streamlit module so tab renderers can be called without a
running Streamlit server. Validates that renderers do not raise and that they
call the expected st.* methods.

Stiglitz (2001 Nobel) — information asymmetry: UI rendering failures are
silent to the end-user (they see a blank tab) but detectable in tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "canned_discovery_payload.json"


def _load_canned() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ── Module-level st mock: imported ONCE; not reloaded between tests ───────────

def _make_cache_data():
    """Return a cache_data stand-in that adds .clear() to decorated fns."""
    def cache_data(*args, **kwargs):
        def decorator(f):
            f.clear = MagicMock()
            return f
        return decorator
    return cache_data


_ST = MagicMock(name="streamlit")
_ST.cache_data = _make_cache_data()
_ST.set_page_config = MagicMock()

# columns() must return a list whose length matches the spec.
# Each column mock has .button() → False so renderers never enter the "force" branch.
def _make_col():
    col = MagicMock()
    col.button = MagicMock(return_value=False)
    return col


def _columns(spec, **kw):
    n = spec if isinstance(spec, int) else len(spec)
    return [_make_col() for _ in range(n)]

_ST.columns = MagicMock(side_effect=_columns)
_ST.tabs    = MagicMock(side_effect=lambda labels: [MagicMock() for _ in labels])

_cm = MagicMock()
_cm.__enter__ = lambda s: s
_cm.__exit__  = MagicMock(return_value=False)
_ST.expander = MagicMock(return_value=_cm)
_ST.spinner  = MagicMock(return_value=_cm)
_ST.sidebar  = MagicMock()
_ST.sidebar.expander = MagicMock(return_value=_cm)

# Inject mock and reload the app so _app.st is _ST regardless of collection order.
import importlib  # noqa: E402

_real_streamlit = sys.modules.get("streamlit")  # save before overwrite
sys.modules["streamlit"] = _ST
import regime_trader.ui.streamlit_app as _app  # noqa: E402
importlib.reload(_app)  # one reload — ensures _app.st is _ST
# Restore real streamlit in sys.modules so other test files are unaffected.
if _real_streamlit is not None:
    sys.modules["streamlit"] = _real_streamlit


# ── Autouse fixture: ensure _app.st == _ST for every renderer test ────────────

@pytest.fixture(autouse=True)
def _bind_st(monkeypatch):
    """Pin _app.st to _ST so renderers use our mock regardless of import order."""
    monkeypatch.setattr(_app, "st", _ST)
    yield


# ── Helper ────────────────────────────────────────────────────────────────────

def _loader(return_value):
    """MagicMock callable that also has .clear() — mimics a st.cache_data loader."""
    m = MagicMock(return_value=return_value)
    m.clear = MagicMock()
    return m


def _reset_st():
    """Reset all call counts on _ST between tests."""
    _ST.reset_mock()
    _ST.columns  = MagicMock(side_effect=_columns)
    _ST.tabs     = MagicMock(side_effect=lambda labels: [MagicMock() for _ in labels])
    _ST.expander = MagicMock(return_value=_cm)
    _ST.spinner  = MagicMock(return_value=_cm)
    _ST.slider   = MagicMock(return_value=5)
    _ST.button   = MagicMock(return_value=False)


# ── _render_live_monitor ──────────────────────────────────────────────────────

class TestRenderLiveMonitor:
    def setup_method(self):
        _reset_st()

    def test_no_alpaca_shows_warning_and_dash_metrics(self, monkeypatch):
        monkeypatch.setattr(_app, "_HAS_ALPACA", False)
        monkeypatch.setattr(_app, "_load_regime",      _loader({"regime": "Neutral", "vix": 18.5}))
        monkeypatch.setattr(_app, "_load_vix_history", _loader([18.0, 18.5]))

        _app._render_live_monitor()

        _ST.warning.assert_called()

    def test_with_alpaca_shows_metrics(self, monkeypatch):
        monkeypatch.setattr(_app, "_HAS_ALPACA", True)
        monkeypatch.setattr(_app, "_load_regime",        _loader({"regime": "Bull", "vix": 14.0}))
        monkeypatch.setattr(_app, "_load_vix_history",   _loader([13.5, 14.0]))
        monkeypatch.setattr(_app, "_load_alpaca_account", _loader({
            "equity": 100000.0, "buying_power": 50000.0,
            "portfolio_value": 100000.0, "daily_pnl": 500.0,
            "daily_pnl_pct": 0.5, "positions": [],
            "cash": 25000.0, "amount_invested": 75000.0,
            "status": "ACTIVE", "paper": True,
        }))

        _app._render_live_monitor()

        # st.metric is called on column objects (col1.metric, col2.metric ...) not on st
        # directly; verify we reached that section by checking st.columns was called
        assert _ST.columns.called, "Expected st.columns to be called for metric layout"

    def test_alpaca_error_shows_st_error(self, monkeypatch):
        monkeypatch.setattr(_app, "_HAS_ALPACA", True)
        monkeypatch.setattr(_app, "_load_regime",        _loader({"regime": "Unknown", "vix": None}))
        monkeypatch.setattr(_app, "_load_vix_history",   _loader(None))
        monkeypatch.setattr(_app, "_load_alpaca_account", _loader({"error": "connection refused"}))

        _app._render_live_monitor()

        _ST.error.assert_called()

    def test_positions_table_rendered_when_present(self, monkeypatch):
        monkeypatch.setattr(_app, "_HAS_ALPACA", True)
        monkeypatch.setattr(_app, "_load_regime",        _loader({"regime": "Bull", "vix": 14.0}))
        monkeypatch.setattr(_app, "_load_vix_history",   _loader([13.5, 14.0]))
        monkeypatch.setattr(_app, "_load_alpaca_account", _loader({
            "equity": 100000.0, "buying_power": 50000.0,
            "portfolio_value": 100000.0, "daily_pnl": 100.0,
            "daily_pnl_pct": 0.1,
            "cash": 25000.0, "amount_invested": 75000.0,
            "positions": [
                {"Symbol": "AAPL", "Side": "Long", "Qty": 10.0,
                 "Entry": 170.0, "Price": 175.0, "Mkt Value": 1750.0,
                 "Unreal. P&L": 50.0, "Unreal. %": 2.94,
                 "Day P&L": 20.0, "Day %": 1.16},
            ],
            "status": "ACTIVE", "paper": True,
        }))

        _app._render_live_monitor()

        _ST.dataframe.assert_called()


# ── _render_market_intel ──────────────────────────────────────────────────────

def _make_market_state(alpha_picks=None) -> dict:
    """Build a minimal market_state.json structure for renderer tests."""
    return {
        "last_updated": "2026-05-15T14:00:00+00:00",
        "macro_status": {
            "regime": "Bull",
            "conviction": 0.75,
            "kill_switch_active": False,
            "vix_latest": 16.0,
        },
        "alpha_picks": alpha_picks if alpha_picks is not None else [],
    }


def _canned_as_picks() -> list:
    """Convert canned discovery fixture results to alpha_picks format."""
    return [
        {**r, "risk_block": False,
         "institutional_net_shares": r.get("institutional_net_shares", 0.0),
         "institutional_pct_change": r.get("institutional_pct_change", 0.0),
         "key_insider_roles": r.get("key_insider_roles", []),
         "market_cap": r.get("market_cap", 0.0),
         "insider_value_pct_mcap": r.get("insider_value_pct_mcap", 0.0),
         "insider_score": r.get("insider_score", 0.0),
         "institutional_score": r.get("institutional_score", 0.0),
         "momentum_score": r.get("momentum_score", 0.0),
        }
        for r in _load_canned().get("results", [])
    ]


class TestRenderMarketIntel:
    def setup_method(self):
        _reset_st()

    def test_no_results_shows_info(self, monkeypatch):
        monkeypatch.setattr(_app, "_load_market_state",
                            _loader(_make_market_state(alpha_picks=[])))

        _app._render_market_intel()

        _ST.info.assert_called()

    def test_results_render_dataframe(self, monkeypatch):
        monkeypatch.setattr(_app, "_load_market_state",
                            _loader(_make_market_state(alpha_picks=_canned_as_picks())))

        _app._render_market_intel()

        _ST.dataframe.assert_called()

    def test_explainability_expander_opened_per_result(self, monkeypatch):
        monkeypatch.setattr(_app, "_load_market_state",
                            _loader(_make_market_state(alpha_picks=_canned_as_picks())))

        _app._render_market_intel()

        assert _ST.expander.call_count >= 1


# ── _render_macro_intel ───────────────────────────────────────────────────────

class TestRenderMacroIntel:
    def setup_method(self):
        _reset_st()

    def _patch_macro(self, monkeypatch, prices, indicators):
        from unittest.mock import patch

        fake_universe  = [{"ticker": "GC=F", "name": "Gold", "unit": "USD/oz"}]
        fake_indics    = [{"ticker": "^VIX", "name": "VIX", "unit": "pts"}]
        conviction_ret = {"conviction_label": "Neutral", "composite": 0.5}

        monkeypatch.setattr(_app, "_load_commodity_prices", _loader(prices))
        monkeypatch.setattr(_app, "_load_macro_indicators", _loader(indicators))

        return (
            patch("regime_trader.scanners.market_intel_macro.COMMODITY_UNIVERSE", fake_universe),
            patch("regime_trader.scanners.market_intel_macro.MACRO_INDICATORS",   fake_indics),
            patch("regime_trader.scanners.market_intel_macro.calc_macro_conviction",
                  MagicMock(return_value=conviction_ret)),
            patch("regime_trader.scanners.market_intel_macro.check_macro_shocks",
                  MagicMock(return_value=[])),
            patch("regime_trader.scanners.market_intel_macro.generate_macro_synthesis",
                  MagicMock(return_value=["ok"])),
        )

    def test_no_data_renders_partial_badge(self, monkeypatch):
        patches = self._patch_macro(monkeypatch, {"GC=F": None}, {"^VIX": None})
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            _app._render_macro_intel()
        # partial data warning should have been emitted
        assert _ST.warning.called or _ST.info.called or _ST.caption.called

    def test_full_data_renders_dataframe(self, monkeypatch):
        prices_data   = {"price": 1900.0, "ret_1d": 0.01, "ret_5d": 0.02, "rsi14": 55.0}
        indicator_data = {"price": 18.5, "ret_1d": -0.01, "ret_5d": 0.02}

        patches = self._patch_macro(monkeypatch,
                                    {"GC=F": prices_data}, {"^VIX": indicator_data})
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            _app._render_macro_intel()

        _ST.dataframe.assert_called()
