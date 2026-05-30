"""tests/test_pipeline_momentum.py
Unit tests for momentum factor and pipeline wiring changes.

Thaler (2017 Nobel) — behavioral momentum: prices continue trending
in the direction of institutional conviction. Tests use injected DataFrames;
no live yfinance calls in CI.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.run_pipeline import (
    fetch_price_data,
    score_momentum,
    fetch_fmp_profiles,
)


class TestScoreMomentum:
    def test_positive_return_above_neutral(self):
        # volume_spike=3.0 gives vol_score=0.5; combined with +10% relative return → > 0.5
        assert score_momentum(0.10, spy_return_20d=0.0, volume_spike=3.0) > 0.5

    def test_negative_return_below_neutral(self):
        assert score_momentum(-0.10, spy_return_20d=0.0, volume_spike=1.0) < 0.5

    def test_zero_return_near_neutral(self):
        # Equal returns, moderate volume → combined ≈ 0.5*0.65 + 0.25*0.35 = 0.4125; test range
        assert 0.3 < score_momentum(0.0, spy_return_20d=0.0, volume_spike=2.0) < 0.7

    def test_large_positive_capped(self):
        """Returns > 30% are clamped — score should equal score at +30%."""
        assert score_momentum(0.99, spy_return_20d=0.0, volume_spike=1.0) == pytest.approx(
            score_momentum(0.30, spy_return_20d=0.0, volume_spike=1.0), abs=1e-6
        )

    def test_large_negative_capped(self):
        assert score_momentum(-0.99, spy_return_20d=0.0, volume_spike=1.0) == pytest.approx(
            score_momentum(-0.30, spy_return_20d=0.0, volume_spike=1.0), abs=1e-6
        )

    def test_output_bounded_0_to_1(self):
        for r in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            assert 0.0 <= score_momentum(r, spy_return_20d=0.0, volume_spike=1.0) <= 1.0


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
        batch = {t: {"symbol": t, "marketCap": 1e12} for t in tickers}

        import os
        with patch.dict(os.environ, {"FMP_API_KEY": "test-key"}), \
             patch("regime_trader.services.fmp_client.FMPClient.get_batch_quotes",
                   return_value=batch):
            result = fetch_fmp_profiles(tickers)

        assert all(result.get(t, 0) > 0 for t in tickers), f"Missing caps: {result}"

    def test_no_key_returns_zero_caps(self):
        """When FMP_API_KEY is absent, all caps are 0.0 (no yfinance fallback)."""
        import os
        tickers = ["AAPL", "MSFT"]
        with patch.dict(os.environ, {}, clear=True):
            result = fetch_fmp_profiles(tickers)
        assert result == {"AAPL": 0.0, "MSFT": 0.0}
