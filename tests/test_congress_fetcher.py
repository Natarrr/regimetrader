"""tests/test_congress_fetcher.py
Unit tests for congressional trading data fetching and scoring.

Stiglitz (2001 Nobel) — asymmetric information: congressional trading
is a credible signal of non-public knowledge. Tests use mocked S3 feeds;
no live network calls in CI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.run_pipeline import fetch_congress_buys, score_congress


class TestScoreCongress:
    def test_no_data_returns_neutral(self):
        assert score_congress(None) == pytest.approx(0.5, abs=1e-4)

    def test_empty_dict_returns_neutral(self):
        assert score_congress({}) == pytest.approx(0.5, abs=1e-4)

    def test_all_purchases_above_neutral(self):
        score = score_congress({"purchases": 4, "sales": 0, "total": 4})
        assert score > 0.5

    def test_all_sales_below_neutral(self):
        score = score_congress({"purchases": 0, "sales": 4, "total": 4})
        assert score < 0.5

    def test_equal_purchases_and_sales_near_neutral(self):
        score = score_congress({"purchases": 3, "sales": 3, "total": 6})
        assert score == pytest.approx(0.5, abs=0.05)

    def test_output_bounded_0_to_1(self):
        for p, s in [(10, 0), (0, 10), (5, 5), (1, 0)]:
            score = score_congress({"purchases": p, "sales": s, "total": p + s})
            assert 0.0 <= score <= 1.0

    def test_zero_total_returns_neutral(self):
        score = score_congress({"purchases": 0, "sales": 0, "total": 0})
        assert score == pytest.approx(0.5, abs=1e-4)


class TestFetchCongressBuys:
    def _make_mock_get(self, house_data, senate_data):
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "house" in url:
                resp.json.return_value = house_data
            else:
                resp.json.return_value = senate_data
            return resp
        return side_effect

    def test_counts_house_purchases(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            {"transaction_date": "2026-04-01", "ticker": "AAPL", "type": "purchase"},
            {"transaction_date": "2026-04-10", "ticker": "AAPL", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)

        assert "AAPL" in result
        assert result["AAPL"]["purchases"] == 2
        assert result["AAPL"]["sales"] == 0

    def test_counts_senate_sales(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        senate = [
            {"transaction_date": "2026-04-05", "ticker": "MSFT", "type": "sale"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get([], senate)):
            result = fetch_congress_buys(lookback_days=90)

        assert "MSFT" in result
        assert result["MSFT"]["sales"] == 1

    def test_ignores_transactions_outside_lookback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            # 200 days ago — outside 90-day window
            {"transaction_date": "2025-10-01", "ticker": "NVDA", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)

        assert "NVDA" not in result

    def test_skips_invalid_tickers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            {"transaction_date": "2026-04-01", "ticker": "N/A", "type": "purchase"},
            {"transaction_date": "2026-04-01", "ticker": "", "type": "purchase"},
            {"transaction_date": "2026-04-01", "ticker": "--", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)

        assert result == {}

    def test_network_failure_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        with patch("requests.get", side_effect=Exception("timeout")):
            result = fetch_congress_buys(lookback_days=90)

        assert result == {}

    def test_cache_is_used_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            {"transaction_date": "2026-04-01", "ticker": "JPM", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])) as mock_get:
            fetch_congress_buys(lookback_days=90)
            call_count_first = mock_get.call_count
            fetch_congress_buys(lookback_days=90)
            # Second call should not hit network
            assert mock_get.call_count == call_count_first
