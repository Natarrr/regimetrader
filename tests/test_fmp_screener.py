"""Unit tests for FMPClient.get_company_screener (stable/company-screener)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.fmp_client import FMPClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_MAX_RPS", "1000")
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


class TestCompanyScreener:
    def test_hits_stable_route_with_liquidity_filters(self, client):
        rows = [{"symbol": "AAA", "marketCap": 5e9},
                {"symbol": "BBB", "marketCap": 3e9}]
        with patch.object(client, "_get", return_value=rows) as m:
            out = client.get_company_screener(
                exchange="NASDAQ",
                market_cap_more_than=2_000_000_000,
                volume_more_than=1_000_000,
                limit=100,
            )
        assert out == rows
        path, params = m.call_args.args[0], m.call_args.args[1]
        assert path == "company-screener"
        assert m.call_args.kwargs.get("bucket") == "screener"
        assert params["exchange"] == "NASDAQ"
        assert params["marketCapMoreThan"] == 2_000_000_000
        assert params["volumeMoreThan"] == 1_000_000
        assert params["limit"] == 100
        # ETFs/funds excluded by default — we screen operating companies only
        assert params["isEtf"] == "false"
        assert params["isFund"] == "false"

    def test_no_api_key_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(cache_root=tmp_path / "fmp")
        assert c.get_company_screener(exchange="NASDAQ") == []

    def test_none_response_returns_empty_list(self, client):
        with patch.object(client, "_get", return_value=None):
            assert client.get_company_screener(exchange="X") == []

    def test_second_call_is_served_from_cache(self, client):
        rows = [{"symbol": "AAA"}]
        with patch.object(client, "_get", return_value=rows) as m:
            first = client.get_company_screener(exchange="NASDAQ")
            second = client.get_company_screener(exchange="NASDAQ")
        assert first == second == rows
        assert m.call_count == 1   # second call hit the screener cache bucket
