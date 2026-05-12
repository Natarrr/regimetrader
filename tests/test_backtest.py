"""tests/test_backtest.py
Unit tests for regime_trader.tools.backtest.

Sharpe (1990 Nobel) — risk-adjusted backtest must be reproducible and
cache-driven (offline-capable). All yfinance calls are mocked.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from regime_trader.tools.backtest import (
    run_backtest,
    _forward_return,
    _load_picks_from_explain,
)
from regime_trader.scoring.normalize import build_explain, persist_explain


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def explain_cache(tmp_path: Path):
    """Populate a temp explain cache with 3 known tickers."""
    for ticker, composite in [("AAPL", 0.90), ("MSFT", 0.80), ("NVDA", 0.70)]:
        explain = build_explain(
            ticker       = ticker,
            scores       = {"x": composite},
            weights      = {"x": 1.0},
            evidence_ids = [f"ev_{ticker.lower()}"],
        )
        persist_explain(ticker, explain, cache_root=tmp_path)
    return tmp_path


def _make_prices(start: str, end: str, tickers: list, growth: float = 0.1) -> dict:
    """Synthetic price series: linear growth from 100 to 100*(1+growth)."""
    s   = date.fromisoformat(start)
    e   = date.fromisoformat(end)
    n   = (e - s).days + 1
    idx = [s + timedelta(days=i) for i in range(n)]
    out = {}
    for t in tickers:
        prices = np.linspace(100.0, 100.0 * (1 + growth), n)
        out[t] = dict(zip([str(d) for d in idx], prices.tolist()))
    return out


# ── _forward_return ───────────────────────────────────────────────────────────

class TestForwardReturn:
    def test_positive_return_computed(self):
        prices = {
            date(2026, 1, 1): 100.0,
            date(2026, 1, 15): 110.0,
            date(2026, 1, 31): 120.0,
        }
        r = _forward_return(prices, date(2026, 1, 1), 30)
        assert r == pytest.approx(0.20, abs=0.01)

    def test_returns_none_if_no_exit_data(self):
        prices = {date(2026, 1, 1): 100.0}
        r = _forward_return(prices, date(2026, 1, 1), 30)
        assert r is None

    def test_returns_none_if_entry_before_data(self):
        prices = {date(2026, 2, 1): 100.0}
        r = _forward_return(prices, date(2026, 1, 1), 7)
        assert r is None

    def test_string_keys_supported(self):
        prices = {
            "2026-01-01": 100.0,
            "2026-01-10": 110.0,
            "2026-02-01": 120.0,
        }
        r = _forward_return(prices, date(2026, 1, 1), 30)
        assert r is not None
        assert r == pytest.approx(0.20, abs=0.01)


# ── _load_picks_from_explain ──────────────────────────────────────────────────

class TestLoadPicksFromExplain:
    def test_loads_top_n_by_composite(self, explain_cache: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_EXPLAIN_ROOT", explain_cache)
        picks = _load_picks_from_explain(top_n=2)
        assert len(picks) == 2
        # AAPL has highest composite (0.90)
        assert picks[0]["ticker"] == "AAPL"

    def test_empty_dir_returns_empty(self, tmp_path: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_EXPLAIN_ROOT", tmp_path / "no_dir")
        picks = _load_picks_from_explain(top_n=5)
        assert picks == []


# ── run_backtest ──────────────────────────────────────────────────────────────

class TestRunBacktest:
    def test_returns_expected_keys(self, explain_cache: Path, tmp_path: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_EXPLAIN_ROOT", explain_cache)
        monkeypatch.setattr(mod, "_PRICE_ROOT", tmp_path / "prices")

        # Mock yfinance download
        tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
        prices  = _make_prices("2026-01-01", "2026-04-30", tickers, growth=0.10)

        def fake_yf_prices(t_list, start, end):
            return {t: {date.fromisoformat(k): v for k, v in prices.get(t, {}).items()}
                    for t in t_list}

        monkeypatch.setattr(mod, "_load_prices_yfinance", fake_yf_prices)
        monkeypatch.setattr(mod, "_load_cached_prices", lambda *_: None)

        result = run_backtest("2026-01-01", "2026-04-30", top_n=3)

        assert "precision_at_n" in result
        assert "forward_returns" in result
        assert "acceptance" in result
        assert "tickers" in result

    def test_precision_is_fraction(self, explain_cache: Path, tmp_path: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_EXPLAIN_ROOT", explain_cache)
        monkeypatch.setattr(mod, "_PRICE_ROOT", tmp_path / "prices")

        tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
        prices  = _make_prices("2026-01-01", "2026-04-30", tickers, growth=0.15)
        monkeypatch.setattr(mod, "_load_prices_yfinance",
                            lambda t, s, e: {t_: {date.fromisoformat(k): v
                                                   for k, v in prices.get(t_, {}).items()}
                                             for t_ in t})
        monkeypatch.setattr(mod, "_load_cached_prices", lambda *_: None)

        result = run_backtest("2026-01-01", "2026-04-30", top_n=3)
        p = result["precision_at_n"]
        assert 0.0 <= p <= 1.0

    def test_no_price_data_returns_error(self, explain_cache: Path, tmp_path: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_EXPLAIN_ROOT", explain_cache)
        monkeypatch.setattr(mod, "_PRICE_ROOT", tmp_path / "prices")
        monkeypatch.setattr(mod, "_load_prices_yfinance", lambda *_: {})
        monkeypatch.setattr(mod, "_load_cached_prices", lambda *_: None)

        result = run_backtest("2026-01-01", "2026-04-30", top_n=3)
        assert result.get("error") == "no_price_data"

    def test_ticker_override_skips_explain_cache(self, tmp_path: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_PRICE_ROOT", tmp_path / "prices")

        tickers = ["AAPL", "MSFT", "SPY"]
        prices  = _make_prices("2026-01-01", "2026-04-30", tickers)
        monkeypatch.setattr(mod, "_load_prices_yfinance",
                            lambda t, s, e: {t_: {date.fromisoformat(k): v
                                                   for k, v in prices.get(t_, {}).items()}
                                             for t_ in t})
        monkeypatch.setattr(mod, "_load_cached_prices", lambda *_: None)

        result = run_backtest(
            "2026-01-01", "2026-04-30",
            top_n=2,
            tickers=["AAPL", "MSFT"],
        )
        assert set(result["tickers"]) == {"AAPL", "MSFT"}

    def test_acceptance_keys_present(self, explain_cache: Path, tmp_path: Path, monkeypatch):
        import regime_trader.tools.backtest as mod
        monkeypatch.setattr(mod, "_EXPLAIN_ROOT", explain_cache)
        monkeypatch.setattr(mod, "_PRICE_ROOT", tmp_path / "prices")

        prices = _make_prices("2026-01-01", "2026-04-30", ["AAPL", "MSFT", "NVDA", "SPY"])
        monkeypatch.setattr(mod, "_load_prices_yfinance",
                            lambda t, s, e: {t_: {date.fromisoformat(k): v
                                                   for k, v in prices.get(t_, {}).items()}
                                             for t_ in t})
        monkeypatch.setattr(mod, "_load_cached_prices", lambda *_: None)

        result = run_backtest("2026-01-01", "2026-04-30", top_n=3)
        acc = result["acceptance"]
        assert "tickers_processed" in acc
        assert "precision_met" in acc
        assert isinstance(acc["precision_met"], bool)
