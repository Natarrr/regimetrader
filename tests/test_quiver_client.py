"""tests/test_quiver_client.py
Unit tests for QuiverClient service module.

Stiglitz (2001 Nobel) — congressional trading exploits non-public information.
All tests use mocked HTTP sessions; no live network calls in CI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from regime_trader.services.quiver_client import QuiverClient

CONGRESS_RECORD = {
    "Representative": "Nancy Pelosi",
    "BioGuideID": "P000197",
    "ReportDate": "2026-04-15",
    "TransactionDate": "2026-04-01",
    "Ticker": "NVDA",
    "Transaction": "Purchase",
    "Range": "$50,001 - $100,000",
    "House": "House",
    "Amount": 75000,
    "Party": "Democrat",
    "TickerType": "ST",
    "ExcessReturn": 5.2,
    "PriceChange": 3.1,
}
INSIDER_RECORD = {
    "Name": "Jensen Huang",
    "Title": "CEO",
    "Date": "2026-04-01",
    "Ticker": "NVDA",
    "AcquisitionOrDisposition": "A",
    "Shares": 10000,
    "PricePerShare": 800.0,
    "TotalValue": 8000000.0,
    "FilingURL": "https://sec.gov/cgi-bin/browse-edgar",
}
F13_RECORD = {
    "Date": "2025-12-31",
    "Shares": 1500000,
    "Value": 1200000000,
    "Pct": 2.5,
    "PctChange": 0.8,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIVER_API_KEY", "test-key")
    return QuiverClient(api_key="test-key", cache_root=tmp_path / "quiver")


def _ok_resp(data):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


class TestQuiverClientCongress:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([CONGRESS_RECORD])):
            result = client.get_politician_trades(lookback_days=90)
        assert isinstance(result, list)
        assert result[0]["Ticker"] == "NVDA"

    def test_caches_result(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([CONGRESS_RECORD])) as mock_get:
            client.get_politician_trades(lookback_days=90)
            client.get_politician_trades(lookback_days=90)
            assert mock_get.call_count == 1

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_politician_trades(lookback_days=90)
        assert result == []

    def test_no_api_key_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUIVER_API_KEY", raising=False)
        c = QuiverClient(cache_root=tmp_path / "quiver")
        assert c.get_politician_trades() == []


class TestQuiverClientInsider:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD])):
            result = client.get_insider_trades("NVDA")
        assert isinstance(result, list)
        assert result[0]["Ticker"] == "NVDA"

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("network")):
            assert client.get_insider_trades("NVDA") == []

    def test_caches_per_ticker(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD])) as mock_get:
            client.get_insider_trades("NVDA")
            client.get_insider_trades("NVDA")
            assert mock_get.call_count == 1


class TestQuiverClient13F:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([F13_RECORD])):
            result = client.get_13f_summary("NVDA")
        assert isinstance(result, list)
        assert result[0]["Pct"] == pytest.approx(2.5)

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("network")):
            assert client.get_13f_summary("NVDA") == []


class TestQuiverClientLobbying:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([{"Ticker": "NVDA", "Amount": 500000}])):
            result = client.get_lobbying("NVDA")
        assert result[0]["Ticker"] == "NVDA"

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            assert client.get_lobbying("NVDA") == []


class TestQuiverClientGovContracts:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([{"Ticker": "LMT", "Amount": 1000000}])):
            result = client.get_gov_contracts("LMT")
        assert result[0]["Ticker"] == "LMT"

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            assert client.get_gov_contracts("LMT") == []


class TestQuiverClientCongressByTicker:
    def test_congress_by_ticker_aggregates_purchases(self, client):
        records = [
            {"Ticker": "NVDA", "Transaction": "Purchase",
             "TransactionDate": "2026-04-01", "TickerType": "ST",
             "Representative": "Nancy Pelosi"},
            {"Ticker": "NVDA", "Transaction": "Purchase",
             "TransactionDate": "2026-04-10", "TickerType": "ST",
             "Representative": "John Doe"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert result["NVDA"]["purchases"] == 2
        assert result["NVDA"]["sales"] == 0
        assert len(result["NVDA"]["representatives"]) == 2

    def test_congress_by_ticker_aggregates_sales(self, client):
        records = [
            {"Ticker": "MSFT", "Transaction": "Sale",
             "TransactionDate": "2026-04-05", "TickerType": "ST",
             "Representative": "Someone"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert result["MSFT"]["sales"] == 1
        assert result["MSFT"]["net"] == -1

    def test_congress_by_ticker_filters_non_stock(self, client):
        records = [
            {"Ticker": "BTC", "Transaction": "Purchase",
             "TransactionDate": "2026-04-01", "TickerType": "CRYPTO",
             "Representative": "Someone"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert "BTC" not in result

    def test_congress_by_ticker_filters_old_trades(self, client):
        records = [
            {"Ticker": "AAPL", "Transaction": "Purchase",
             "TransactionDate": "2020-01-01", "TickerType": "ST",
             "Representative": "Someone"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert "AAPL" not in result

    def test_congress_by_ticker_skips_invalid_tickers(self, client):
        records = [
            {"Ticker": "N/A", "Transaction": "Purchase",
             "TransactionDate": "2026-04-01", "TickerType": "ST",
             "Representative": "A"},
            {"Ticker": "", "Transaction": "Purchase",
             "TransactionDate": "2026-04-01", "TickerType": "ST",
             "Representative": "B"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert result == {}

    def test_congress_by_ticker_recency_days_populated(self, client):
        records = [
            {"Ticker": "NVDA", "Transaction": "Purchase",
             "TransactionDate": "2026-05-10", "TickerType": "ST",
             "Representative": "Nancy Pelosi"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert result["NVDA"]["recency_days"] < 30


class TestCIIsolation:
    def test_client_is_constructible_in_ci(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUIVER_API_KEY", "test-key")
        monkeypatch.setenv("CI", "1")
        c = QuiverClient(api_key="test-key", cache_root=tmp_path / "quiver")
        assert c is not None
        assert c._api_key == "test-key"
