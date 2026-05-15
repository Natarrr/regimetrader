"""tests/test_alpaca_account.py
Tests for _load_alpaca_account() in regime_trader/ui/streamlit_app.py.

Markowitz (1952 Nobel) — portfolio theory: a portfolio monitor is only as
trustworthy as the account loader that feeds it. Test the loader contract
independently of the UI rendering.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
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


# ── Fake Alpaca objects ────────────────────────────────────────────────────────

def _fake_account(equity="100000.00", last_equity="99000.00",
                   buying_power="50000.00", portfolio_value="100000.00",
                   cash="25000.00", status="ACTIVE"):
    ns = SimpleNamespace(
        equity=equity,
        last_equity=last_equity,
        buying_power=buying_power,
        portfolio_value=portfolio_value,
        cash=cash,
        status=SimpleNamespace(value=status),
    )
    return ns


def _fake_position(symbol, side_val, qty, avg_entry, cur_price,
                    market_value, unreal_pl, unreal_plpc,
                    intraday_pl, intraday_plpc):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value=side_val),
        qty=str(qty),
        avg_entry_price=str(avg_entry),
        current_price=str(cur_price),
        market_value=str(market_value),
        unrealized_pl=str(unreal_pl),
        unrealized_plpc=str(unreal_plpc),
        unrealized_intraday_pl=str(intraday_pl),
        unrealized_intraday_plpc=str(intraday_plpc),
    )


def _get_loader():
    import importlib
    with patch.dict(sys.modules, {"streamlit": _st_mock}):
        import regime_trader.ui.streamlit_app as app
        importlib.reload(app)
        return app._load_alpaca_account


# ── Success path ───────────────────────────────────────────────────────────────

class TestLoadAlpacaAccountSuccess:
    def test_returns_expected_keys(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_A001")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_A001")

        fn = _get_loader()
        acct = _fake_account()
        pos = _fake_position("AAPL", "long", 10, 170.0, 175.0,
                              1750.0, 50.0, 0.0294, 20.0, 0.0116)
        fake_client = MagicMock()
        fake_client.get_account.return_value = acct
        fake_client.get_all_positions.return_value = [pos]

        with patch("alpaca.trading.client.TradingClient", return_value=fake_client):
            result = fn()

        assert "error" not in result
        for key in ("equity", "buying_power", "portfolio_value",
                     "daily_pnl", "daily_pnl_pct", "positions", "status", "paper"):
            assert key in result, f"missing key: {key}"

    def test_daily_pnl_calculation(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_A002")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_A002")

        fn = _get_loader()
        acct = _fake_account(equity="101000.00", last_equity="100000.00")
        fake_client = MagicMock()
        fake_client.get_account.return_value = acct
        fake_client.get_all_positions.return_value = []

        with patch("alpaca.trading.client.TradingClient", return_value=fake_client):
            result = fn()

        assert abs(result["daily_pnl"] - 1000.0) < 0.01
        assert abs(result["daily_pnl_pct"] - 1.0) < 0.01

    def test_positions_sorted_by_market_value_desc(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_A003")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_A003")

        fn = _get_loader()
        acct = _fake_account()
        pos_small = _fake_position("TSLA", "long", 1, 200.0, 200.0,
                                    200.0, 0.0, 0.0, 0.0, 0.0)
        pos_large = _fake_position("AAPL", "long", 10, 170.0, 175.0,
                                    1750.0, 50.0, 0.03, 20.0, 0.01)
        fake_client = MagicMock()
        fake_client.get_account.return_value = acct
        fake_client.get_all_positions.return_value = [pos_small, pos_large]

        with patch("alpaca.trading.client.TradingClient", return_value=fake_client):
            result = fn()

        symbols = [r["Symbol"] for r in result["positions"]]
        assert symbols[0] == "AAPL"  # larger market value first
        assert symbols[1] == "TSLA"

    def test_position_row_has_all_columns(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_A004")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_A004")

        fn = _get_loader()
        acct = _fake_account()
        pos = _fake_position("MSFT", "long", 5, 380.0, 385.0,
                              1925.0, 25.0, 0.013, 10.0, 0.005)
        fake_client = MagicMock()
        fake_client.get_account.return_value = acct
        fake_client.get_all_positions.return_value = [pos]

        with patch("alpaca.trading.client.TradingClient", return_value=fake_client):
            result = fn()

        row = result["positions"][0]
        for col in ("Symbol", "Side", "Qty", "Entry", "Price",
                     "Mkt Value", "Unreal. P&L", "Unreal. %",
                     "Day P&L", "Day %"):
            assert col in row, f"missing column: {col}"

    def test_paper_flag_true_when_paper_in_base_url(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        fn = _get_loader()
        fake_client = MagicMock()
        fake_client.get_account.return_value = _fake_account()
        fake_client.get_all_positions.return_value = []

        with patch("alpaca.trading.client.TradingClient", return_value=fake_client):
            result = fn()

        assert result["paper"] is True


# ── Error path ─────────────────────────────────────────────────────────────────

class TestLoadAlpacaAccountError:
    def test_connection_error_returns_error_key(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_TEST001")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_TEST001")

        fn = _get_loader()
        with patch("alpaca.trading.client.TradingClient",
                   side_effect=RuntimeError("connection refused")):
            result = fn()

        assert "error" in result
        assert "positions" not in result

    def test_error_value_is_string(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_TEST002")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_TEST002")

        fn = _get_loader()
        with patch("alpaca.trading.client.TradingClient",
                   side_effect=Exception("feed timed out")):
            result = fn()

        assert isinstance(result["error"], str)
        assert "timed out" in result["error"]

    def test_last_equity_zero_gives_zero_pct(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "FAKE_ALPACA_KEY_TEST003")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_ALPACA_SECRET_TEST003")

        fn = _get_loader()
        acct = _fake_account(equity="1000.00", last_equity="0.00")
        fake_client = MagicMock()
        fake_client.get_account.return_value = acct
        fake_client.get_all_positions.return_value = []

        with patch("alpaca.trading.client.TradingClient", return_value=fake_client):
            result = fn()

        assert result["daily_pnl_pct"] == 0.0
