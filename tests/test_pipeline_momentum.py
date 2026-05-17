"""tests/test_pipeline_momentum.py
Unit tests for momentum factor and pipeline wiring changes.

Thaler (2017 Nobel) — behavioral momentum: prices continue trending
in the direction of institutional conviction. Tests use injected DataFrames;
no live yfinance calls in CI.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock
from pathlib import Path

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
        assert score_momentum(0.10, spy_return_20d=0.0, volume_spike=1.0) > 0.5

    def test_negative_return_below_neutral(self):
        assert score_momentum(-0.10, spy_return_20d=0.0, volume_spike=1.0) < 0.5

    def test_zero_return_near_neutral(self):
        assert score_momentum(0.0, spy_return_20d=0.0, volume_spike=1.0) == pytest.approx(0.5, abs=0.02)

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
    def _fake_df(self, start_price: float, end_price: float, n: int = 21) -> pd.DataFrame:
        """Create a fake yfinance Close DataFrame."""
        prices = np.linspace(start_price, end_price, n)
        idx    = pd.date_range("2026-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": prices}, index=idx)

    def test_positive_momentum_detected(self):
        fake = self._fake_df(100.0, 110.0)   # +10% over the period
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("AAPL")
        assert result["return_20d"] > 0.0

    def test_negative_momentum_detected(self):
        fake = self._fake_df(100.0, 90.0)    # -10%
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("AAPL")
        assert result["return_20d"] < 0.0

    def test_flat_market_returns_near_zero(self):
        fake = self._fake_df(100.0, 100.0)
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("AAPL")
        assert abs(result["return_20d"]) < 1e-6

    def test_empty_dataframe_returns_zero(self):
        fake = pd.DataFrame()
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("INVALID")
        assert result["return_20d"] == pytest.approx(0.0)

    def test_exception_returns_zero(self):
        with patch("yfinance.download", side_effect=Exception("network error")):
            result = fetch_price_data("AAPL")
        assert result["return_20d"] == pytest.approx(0.0)

    def test_return_key_present(self):
        fake = self._fake_df(100.0, 105.0)
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("MSFT")
        assert "return_20d" in result


class TestFetchFmpProfilesChunking:
    def test_165_tickers_makes_two_calls(self):
        """165 tickers chunked at 100 → 2 FMP GET calls."""
        tickers = [f"T{i:03d}" for i in range(165)]

        call_count = 0
        def fake_fmp_get(path, params, timeout=20):
            nonlocal call_count
            call_count += 1
            symbols = params.get("symbol", "").split(",")
            return [{"symbol": s, "mktCap": 1e9} for s in symbols]

        with patch("scripts.run_pipeline._fmp_get", side_effect=fake_fmp_get):
            result = fetch_fmp_profiles(tickers)

        assert call_count == 2
        assert len(result) == 165

    def test_50_tickers_makes_one_call(self):
        tickers = [f"T{i:02d}" for i in range(50)]
        call_count = 0
        def fake_fmp_get(path, params, timeout=20):
            nonlocal call_count
            call_count += 1
            return [{"symbol": s, "mktCap": 1e9} for s in params["symbol"].split(",")]
        with patch("scripts.run_pipeline._fmp_get", side_effect=fake_fmp_get):
            fetch_fmp_profiles(tickers)
        assert call_count == 1
