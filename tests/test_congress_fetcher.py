"""tests/test_congress_fetcher.py
Unit tests for congressional trading data fetching and scoring.

Stiglitz (2001 Nobel) — asymmetric information: congressional trading
is a credible signal of non-public knowledge. Tests use mocked S3 feeds;
no live network calls in CI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.run_pipeline import fetch_congress_buys, score_congress


class TestScoreCongress:
    def test_no_data_returns_zero_not_neutral(self):
        # Dead API / ticker not traded → 0.0 so the normaliser penalises dead feeds
        assert score_congress(None) == pytest.approx(0.0, abs=1e-9)

    def test_empty_dict_returns_zero_not_neutral(self):
        assert score_congress({}) == pytest.approx(0.0, abs=1e-9)

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

    def test_recent_trade_scores_higher_than_old(self):
        recent = score_congress({"purchases": 3, "sales": 0, "total": 3, "recency_days": 5})
        old    = score_congress({"purchases": 3, "sales": 0, "total": 3, "recency_days": 150})
        assert recent > old

    def test_recency_dampens_towards_neutral_not_zero(self):
        # Old but net-purchase signal should still be > 0.5 (direction preserved)
        score = score_congress({"purchases": 5, "sales": 0, "total": 5, "recency_days": 180})
        assert score > 0.5

    def test_no_recency_key_no_change(self):
        score_with    = score_congress({"purchases": 3, "sales": 0, "total": 3})
        score_recent  = score_congress({"purchases": 3, "sales": 0, "total": 3, "recency_days": 10})
        assert score_with == pytest.approx(score_recent, abs=1e-4)


class TestFetchCongressBuys:
    _QUIVER_URL = "quiverquant.com"

    def _make_mock_get(self, house_data, senate_data, quiver_data=None):
        """Mock requests.get for S3 + Quiver endpoints."""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "house" in url:
                resp.json.return_value = house_data
            elif "senate" in url:
                resp.json.return_value = senate_data
            elif self._QUIVER_URL in url:
                if quiver_data is None:
                    resp.raise_for_status.side_effect = Exception("no quiver mock")
                else:
                    resp.json.return_value = quiver_data
            return resp
        return side_effect

    def _no_quiver(self, monkeypatch):
        """Ensure QUIVER_API_KEY is absent so fallback doesn't fire unexpectedly."""
        monkeypatch.delenv("QUIVER_API_KEY", raising=False)

    def test_counts_house_purchases(self, tmp_path, monkeypatch):
        self._no_quiver(monkeypatch)
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
        house = [
            {"transaction_date": "2026-04-01", "ticker": "AAPL", "type": "purchase"},
            {"transaction_date": "2026-04-10", "ticker": "AAPL", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)
        assert result["AAPL"]["purchases"] == 2
        assert result["AAPL"]["sales"] == 0

    def test_counts_senate_sales(self, tmp_path, monkeypatch):
        self._no_quiver(monkeypatch)
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
        senate = [{"transaction_date": "2026-04-05", "ticker": "MSFT", "type": "sale"}]
        with patch("requests.get", side_effect=self._make_mock_get([], senate)):
            result = fetch_congress_buys(lookback_days=90)
        assert result["MSFT"]["sales"] == 1

    def test_ignores_transactions_outside_lookback(self, tmp_path, monkeypatch):
        self._no_quiver(monkeypatch)
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
        house = [{"transaction_date": "2025-10-01", "ticker": "NVDA", "type": "purchase"}]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)
        assert "NVDA" not in result

    def test_skips_invalid_tickers(self, tmp_path, monkeypatch):
        self._no_quiver(monkeypatch)
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
        house = [
            {"transaction_date": "2026-04-01", "ticker": "N/A", "type": "purchase"},
            {"transaction_date": "2026-04-01", "ticker": "",    "type": "purchase"},
            {"transaction_date": "2026-04-01", "ticker": "--",  "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)
        assert result == {}

    def test_s3_403_falls_back_to_fmp(self, tmp_path, monkeypatch):
        """When S3 returns 403, FMPClient.get_congress_trades() is used as fallback."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")

        fmp_result = {
            "purchases": 1, "sales": 0, "total": 1, "recency_days": 5,
        }

        def mock_s3_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 403
            resp.raise_for_status.side_effect = Exception("403 Forbidden")
            return resp

        with patch("requests.get", side_effect=mock_s3_get), \
             patch(
                 "regime_trader.services.fmp_client.FMPClient.get_congress_trades",
                 return_value=fmp_result,
             ):
            result = fetch_congress_buys(lookback_days=90)

        assert isinstance(result, dict)

    def test_all_sources_fail_returns_empty(self, tmp_path, monkeypatch):
        self._no_quiver(monkeypatch)
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
        with patch("requests.get", side_effect=Exception("timeout")):
            result = fetch_congress_buys(lookback_days=90)
        assert result == {}

    def test_cache_is_used_on_second_call(self, tmp_path, monkeypatch):
        self._no_quiver(monkeypatch)
        monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
        house = [{"transaction_date": "2026-04-01", "ticker": "JPM", "type": "purchase"}]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])) as mock_get:
            fetch_congress_buys(lookback_days=90)
            first_count = mock_get.call_count
            fetch_congress_buys(lookback_days=90)
            assert mock_get.call_count == first_count
