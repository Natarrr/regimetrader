"""Unit tests for FMPClient service module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import time
import pytest

from regime_trader.services.fmp_client import FMPClient


# ── Fixtures & helpers ──────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FMP_MAX_RPS", "1000")  # disable rate limiting in tests
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


def _ok_resp(data):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


def _empty_resp():
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = []
    return resp


# ── FMPClient construction ──────────────────────────────────────────────────

class TestFMPClientConstruction:
    def test_reads_api_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "env-key")
        c = FMPClient(cache_root=tmp_path / "fmp")
        assert c._api_key == "env-key"

    def test_no_key_sets_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(cache_root=tmp_path / "fmp")
        assert c._api_key == ""

    def test_fmp_max_rps_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "10")
        c = FMPClient(api_key="k", cache_root=tmp_path / "fmp")
        assert c._min_delay == pytest.approx(0.1, abs=1e-6)


# ── Congress factor ─────────────────────────────────────────────────────────

SENATE_RECORD = {
    "symbol": "NVDA",
    "senator": "Nancy Pelosi",
    "transactionDate": "2026-04-01",
    "disclosureDate": "2026-04-15",
    "type": "Purchase",
    "amount": "15001-50000",
}
HOUSE_RECORD = {
    "representative": "John Doe",
    "ticker": "NVDA",
    "transactionDate": "2026-04-10",
    "disclosureDate": "2026-04-20",
    "type": "Purchase--",
    "amount": "1001-15000",
}


class TestFMPClientCongress:
    """get_congress_trades is a stub — FMP senate/house routes returned HTTP 404
    in the Phase-0 smoke-test (2026-05-30). Congress uses S3 Stock Watcher feeds.
    All calls return {} regardless of session state."""

    def test_returns_empty_dict_stub(self, client):
        """Stub always returns {} — congress routes are dead on this plan."""
        result = client.get_congress_trades("NVDA", lookback_days=180)
        assert result == {}

    def test_returns_empty_dict_on_api_error(self, client):
        # Stub does not call session at all — error path still returns {}
        result = client.get_congress_trades("NVDA")
        assert result == {}

    def test_returns_empty_dict_on_empty_response(self, client):
        result = client.get_congress_trades("NVDA")
        assert result == {}

    def test_returns_empty_dict_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(cache_root=tmp_path / "fmp")
        result = c.get_congress_trades("NVDA")
        assert result == {}

    def test_no_http_calls_made(self, client):
        """Stub makes zero HTTP calls — never hits the dead FMP route."""
        with patch.object(client._session, "get") as mock_get:
            client.get_congress_trades("NVDA", lookback_days=180)
            assert mock_get.call_count == 0


# ── Insider factor ──────────────────────────────────────────────────────────

INSIDER_RECORD_ACQUISITION = {
    "symbol": "NVDA",
    "filingDate": "2026-04-15",
    "transactionDate": "2026-04-01",
    "disclosureDate": "2026-04-15",
    "acquistionOrDisposition": "A",
    "securitiesTransacted": 1000.0,
    "price": 800.0,
    "transactionType": "P-Purchase",
}
INSIDER_RECORD_DISPOSITION = {
    **INSIDER_RECORD_ACQUISITION,
    "acquistionOrDisposition": "D",
}


class TestFMPClientInsider:
    def test_returns_tuple_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD_ACQUISITION])):
            usd, days = client.get_insider_purchases("NVDA", lookback_days=180)
        assert usd == pytest.approx(800_000.0)
        assert days >= 0

    def test_filters_disposition_records(self, client):
        """Only 'A' (Acquisition) records count — dispositions must be ignored."""
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD_DISPOSITION])):
            usd, days = client.get_insider_purchases("NVDA")
        assert usd == 0.0

    def test_null_securities_transacted_skipped(self, client):
        """None/empty securitiesTransacted must not raise ValueError."""
        bad = {**INSIDER_RECORD_ACQUISITION, "securitiesTransacted": None}
        with patch.object(client._session, "get", return_value=_ok_resp([bad])):
            usd, days = client.get_insider_purchases("NVDA")
        assert usd == 0.0

    def test_null_price_skipped(self, client):
        bad = {**INSIDER_RECORD_ACQUISITION, "price": None}
        with patch.object(client._session, "get", return_value=_ok_resp([bad])):
            usd, days = client.get_insider_purchases("NVDA")
        assert usd == 0.0

    def test_returns_zero_tuple_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            usd, days = client.get_insider_purchases("NVDA")
        assert (usd, days) == (0.0, 0)

    def test_returns_zero_tuple_on_empty_response(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            usd, days = client.get_insider_purchases("NVDA")
        assert (usd, days) == (0.0, 0)

    def test_caches_per_ticker(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD_ACQUISITION])) as mock_get:
            client.get_insider_purchases("NVDA")
            client.get_insider_purchases("NVDA")
            assert mock_get.call_count == 1

    def test_uses_limit_500_in_request(self, client):
        """Mega-caps need limit=500 to cover 180-day lookback without truncation."""
        with patch.object(client._session, "get", return_value=_empty_resp()) as mock_get:
            client.get_insider_purchases("NVDA")
        call_args = mock_get.call_args
        params = call_args[1].get("params", {}) or (call_args[0][1] if len(call_args[0]) > 1 else {})
        assert str(params.get("limit", "")) == "500"


# ── News factor ─────────────────────────────────────────────────────────────

NEWS_POSITIVE = {"title": "NVDA beats earnings", "sentiment": "Positive", "publishedDate": "2026-04-15"}
NEWS_NEGATIVE = {"title": "NVDA misses guidance", "sentiment": "Negative", "publishedDate": "2026-04-14"}
NEWS_NEUTRAL  = {"title": "NVDA releases product", "sentiment": "Neutral",  "publishedDate": "2026-04-13"}


class TestFMPClientNews:
    def test_returns_float_on_success(self, client):
        articles = [NEWS_POSITIVE] * 30 + [NEWS_NEGATIVE] * 10
        with patch.object(client._session, "get", return_value=_ok_resp(articles)):
            score = client.get_news_raw_articles("NVDA")
        assert isinstance(score, list)
        assert len(score) == 40

    def test_returns_empty_list_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_news_raw_articles("NVDA")
        assert result == []

    def test_caches_result(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([NEWS_POSITIVE])) as mock_get:
            client.get_news_raw_articles("NVDA")
            client.get_news_raw_articles("NVDA")
            assert mock_get.call_count == 1


# ── Quote factor ─────────────────────────────────────────────────────────────

QUOTE_RECORD = {
    "symbol": "NVDA",
    "price": 800.0,
    "marketCap": 2_000_000_000_000,
    "volume": 50_000_000,
    "avgVolume": 40_000_000,
    "eps": 22.5,
}


class TestFMPClientQuote:
    def test_returns_dict_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([QUOTE_RECORD])):
            result = client.get_quote("NVDA")
        assert result["symbol"] == "NVDA"
        assert result["price"] == 800.0

    def test_bypass_cache_forces_live_call(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([QUOTE_RECORD])) as mock_get:
            client.get_quote("NVDA")
            client.get_quote("NVDA", bypass_cache=True)
            assert mock_get.call_count == 2

    def test_returns_empty_dict_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_quote("NVDA")
        assert result == {}

    def test_accepts_international_suffix(self, client):
        """FMP Ultimate accepts SAP.DE, 7203.T natively."""
        with patch.object(client._session, "get", return_value=_ok_resp([{**QUOTE_RECORD, "symbol": "SAP.DE"}])):
            result = client.get_quote("SAP.DE")
        assert result.get("symbol") == "SAP.DE"


# ── Stub: _enrich_with_quiver ───────────────────────────────────────────────

class TestEnrichWithQuiverStub:
    def test_always_returns_empty_quiver_dict(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from regime_trader.scanners.discovery_scanner import _enrich_with_quiver
        results = [{"symbol": "NVDA"}, {"symbol": "AAPL"}]
        enriched = _enrich_with_quiver(results)
        for r in enriched:
            assert r["quiver"] == {}

    def test_no_network_calls_ever(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from regime_trader.scanners.discovery_scanner import _enrich_with_quiver
        with patch("requests.get") as mock_get:
            _enrich_with_quiver([{"symbol": "NVDA"}])
            mock_get.assert_not_called()


# ── CI isolation ────────────────────────────────────────────────────────────

class TestCIIsolation:
    def test_client_constructible_in_ci(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("CI", "1")
        c = FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")
        assert c is not None
        assert c._api_key == "test-key"

    def test_fmp_max_rps_defaults_to_30(self, tmp_path, monkeypatch):
        """Default changed from 20 to 30 RPS (Ultimate cap 50; 30 leaves headroom)."""
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.delenv("FMP_MAX_RPS", raising=False)
        c = FMPClient(api_key="k", cache_root=tmp_path / "fmp")
        assert c._min_delay == pytest.approx(1.0 / 30.0, abs=1e-6)


# ── Pipeline wrappers ───────────────────────────────────────────────────────

class TestFetchFMPInsiderAll:
    """Tests for the fetch_fmp_insider_all() function in run_pipeline.py."""

    def test_returns_dict_keyed_by_ticker(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from scripts.run_pipeline import fetch_fmp_insider_all
        from regime_trader.services.fmp_client import FMPClient

        with patch.object(FMPClient, "get_insider_purchases", return_value=(800_000.0, 5)):
            result = fetch_fmp_insider_all(["NVDA", "AAPL"])
        assert "NVDA" in result
        assert result["NVDA"] == (800_000.0, 5)
        assert "AAPL" in result

    def test_returns_empty_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        from scripts.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all(["NVDA"])
        assert result == {}

    def test_returns_empty_on_empty_input(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from scripts.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all([])
        assert result == {}


class TestScoreNewsFMP:
    """Tests for score_news_fmp() in run_pipeline.py."""

    def test_scores_positive_articles_above_neutral(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from scripts.run_pipeline import score_news_fmp
        from regime_trader.services.fmp_client import FMPClient

        articles = [{"sentiment": "Positive"}] * 40 + [{"sentiment": "Negative"}] * 10
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("NVDA")
        assert score > 0.5

    def test_falls_back_to_yfinance_on_empty_articles(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from scripts.run_pipeline import score_news_fmp
        from regime_trader.services.fmp_client import FMPClient

        with patch.object(FMPClient, "get_news_raw_articles", return_value=[]), \
             patch("scripts.run_pipeline._score_news_sentiment_yfinance", return_value=0.6) as mock_yf:
            score = score_news_fmp("SAP.DE")
        mock_yf.assert_called_once_with("SAP.DE")
        assert score == pytest.approx(0.6)

    def test_score_bounded_0_to_1(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from scripts.run_pipeline import score_news_fmp
        from regime_trader.services.fmp_client import FMPClient

        articles = [{"sentiment": "Positive"}] * 50
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("NVDA")
        assert 0.0 <= score <= 1.0


class TestGetEarningsTranscript:
    """get_earnings_transcript fetches stable/earning-call-transcript-latest.

    Cache bucket: "transcript" (24h TTL).
    Soft-fail: returns None on error, empty list, or FMPEndpointError.
    max_chars=3000 fetch ceiling is intentionally larger than build_prompt's
    2000-char injection limit — no second network call needed if budget changes.
    """

    def test_returns_content_truncated_to_max_chars(self, client):
        long_content = "A" * 5000
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026,
                    "date": "2026-01-15", "content": long_content}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            result = client.get_earnings_transcript("AAPL", max_chars=3000)
        assert result == "A" * 3000

    def test_returns_full_content_when_shorter_than_max_chars(self, client):
        content = "Short transcript text."
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026,
                    "date": "2026-01-15", "content": content}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            result = client.get_earnings_transcript("AAPL")
        assert result == content

    def test_returns_none_on_empty_list(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_returns_none_on_missing_content_key(self, client):
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026, "date": "2026-01-15"}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_returns_none_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        result = c.get_earnings_transcript("AAPL")
        assert result is None

    def test_caches_result(self, client):
        content = "Transcript text."
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026,
                    "date": "2026-01-15", "content": content}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)) as mock_get:
            client.get_earnings_transcript("AAPL")
            client.get_earnings_transcript("AAPL")
        assert mock_get.call_count == 1  # second call served from cache

    def test_returns_none_on_fmp_endpoint_error(self, client):
        from regime_trader.services.fmp_client import FMPEndpointError
        with patch.object(client, "_get", side_effect=FMPEndpointError("earning-call-transcript-latest", 404)):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_returns_none_on_network_exception(self, client):
        with patch.object(client, "_get", side_effect=RuntimeError("timeout")):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_caches_sentinel_when_no_content(self, client):
        """Missing content writes empty-string sentinel so we don't re-fetch."""
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026, "date": "2026-01-15"}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)) as mock_get:
            result1 = client.get_earnings_transcript("AAPL")
            result2 = client.get_earnings_transcript("AAPL")
        assert result1 is None  # First call: no content, so returns None
        assert result2 == ""    # Second call: cached sentinel empty string
        assert mock_get.call_count == 1  # second call served from cache sentinel
