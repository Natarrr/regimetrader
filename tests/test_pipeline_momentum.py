"""tests/test_pipeline_momentum.py
Unit tests for fetch_price_data and fetch_fmp_profiles pipeline functions.

Thaler (2017 Nobel) — behavioral momentum: prices continue trending
in the direction of institutional conviction. Tests use injected rows;
no live yfinance calls in CI.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.run_pipeline import (
    fetch_price_data,
    fetch_fmp_profiles,
)


class TestFetchPriceData:
    def _fake_rows(self, start_price: float, end_price: float, n: int = 260) -> list:
        """Build FMP historical-price-eod/full rows (newest-first).

        n=260 gives >252 bars so Jegadeesh-Titman 12-1m period is computable.
        """
        step = (end_price - start_price) / max(n - 1, 1)
        rows = []
        for i in range(n):
            rows.append({
                "date":   f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "close":  start_price + step * i,
                "volume": 1_000_000.0,
            })
        return list(reversed(rows))   # newest-first

    def _patch(self, rows):
        return patch(
            "regime_trader.services.fmp_client.FMPClient.get_historical_prices",
            return_value=rows,
        )

    def test_positive_momentum_detected(self):
        with self._patch(self._fake_rows(100.0, 110.0)):
            result = fetch_price_data("AAPL")
        assert result["return_12_1m"] is not None
        assert result["return_12_1m"] > 0.0

    def test_negative_momentum_detected(self):
        with self._patch(self._fake_rows(100.0, 90.0)):
            result = fetch_price_data("AAPL")
        assert result["return_12_1m"] is not None
        assert result["return_12_1m"] < 0.0

    def test_flat_market_returns_near_zero(self):
        with self._patch(self._fake_rows(100.0, 100.0)):
            result = fetch_price_data("AAPL")
        assert abs(result["return_12_1m"]) < 1e-6

    def test_empty_data_returns_none(self):
        """Insufficient history → return_12_1m=None (dead signal, not 0.0)."""
        with self._patch([]):
            result = fetch_price_data("INVALID")
        assert result["return_12_1m"] is None

    def test_exception_returns_none(self):
        """Exception → default dict with return_12_1m=None."""
        with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
                   side_effect=Exception("network error")):
            result = fetch_price_data("AAPL")
        assert result["return_12_1m"] is None

    def test_return_key_present(self):
        with self._patch(self._fake_rows(100.0, 105.0)):
            result = fetch_price_data("MSFT")
        assert "return_12_1m" in result


class TestFetchFmpProfilesBatch:
    def test_returns_market_caps_for_all_tickers(self):
        """fetch_fmp_profiles returns a market cap for each ticker via batch-quote."""
        tickers = ["AAPL", "MSFT", "GOOGL"]

        def fake_get(self, *args, **kwargs):
            params = kwargs.get("params", {}) or {}
            symbols = params.get("symbols", "")
            syms = [s for s in symbols.split(",") if s]
            mock = MagicMock()
            mock.status_code = 200
            mock.raise_for_status = MagicMock()
            mock.json.return_value = [{"symbol": sym, "marketCap": 1e12} for sym in syms]
            return mock

        import os
        with patch.dict(os.environ, {"FMP_API_KEY": "test-key"}), \
             patch("requests.Session.get", side_effect=fake_get):
            result = fetch_fmp_profiles(tickers)

        assert all(result.get(t, 0) > 0 for t in tickers), f"Missing caps: {result}"

    def test_no_key_returns_zero_caps(self):
        """When FMP_API_KEY is absent, all caps are 0.0 (no yfinance fallback)."""
        import os
        tickers = ["AAPL", "MSFT"]
        with patch.dict(os.environ, {}, clear=True):
            result = fetch_fmp_profiles(tickers)
        assert result == {"AAPL": 0.0, "MSFT": 0.0}
