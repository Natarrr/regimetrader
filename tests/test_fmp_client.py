"""Unit tests for FMPClient service module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import time
import pytest

from src.services.fmp_client import FMPClient


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

    def setup_method(self):
        # Reset the class-level probe flag so each test starts clean.
        FMPClient._fmp_congress_probe_done = True  # skip probe — no real network in tests

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

    def test_insider_statistics_sorted_newest_first(self, client):
        """get_insider_statistics returns quarters newest-first (latest at index 0)."""
        rows = [
            {"symbol": "NVDA", "year": 2025, "quarter": 4,
             "acquiredTransactions": 3, "disposedTransactions": 3},
            {"symbol": "NVDA", "year": 2026, "quarter": 2,
             "acquiredTransactions": 9, "disposedTransactions": 1},
            {"symbol": "NVDA", "year": 2026, "quarter": 1,
             "acquiredTransactions": 5, "disposedTransactions": 5},
        ]
        with patch.object(client._session, "get", return_value=_ok_resp(rows)):
            out = client.get_insider_statistics("NVDA")
        assert [(r["year"], r["quarter"]) for r in out] == [(2026, 2), (2026, 1), (2025, 4)]

    def test_insider_statistics_empty_on_no_data(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            assert client.get_insider_statistics("NVDA") == []

    def test_insider_statistics_no_key_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        from src.services.fmp_client import FMPClient
        c = FMPClient(cache_root=tmp_path)
        assert c.get_insider_statistics("NVDA") == []

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


# ── CI isolation ────────────────────────────────────────────────────────────

class TestCIIsolation:
    def test_client_constructible_in_ci(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("CI", "1")
        c = FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")
        assert c is not None
        assert c._api_key == "test-key"

    def test_fmp_max_rps_defaults_to_50(self, tmp_path, monkeypatch):
        """Default is 50 RPS — saturates the FMP Ultimate cap (3,000 calls/min)."""
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.delenv("FMP_MAX_RPS", raising=False)
        c = FMPClient(api_key="k", cache_root=tmp_path / "fmp")
        assert c._min_delay == pytest.approx(1.0 / 50.0, abs=1e-6)


# ── Pipeline wrappers ───────────────────────────────────────────────────────

class TestFetchFMPInsiderAll:
    """Tests for the fetch_fmp_insider_all() function in run_pipeline.py."""

    def test_returns_dict_keyed_by_ticker(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from src.ingestion.run_pipeline import fetch_fmp_insider_all
        from src.services.fmp_client import FMPClient

        with patch.object(FMPClient, "get_insider_purchases", return_value=(800_000.0, 5)):
            result = fetch_fmp_insider_all(["NVDA", "AAPL"])
        assert "NVDA" in result
        assert result["NVDA"] == (800_000.0, 5)
        assert "AAPL" in result

    def test_returns_empty_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        from src.ingestion.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all(["NVDA"])
        assert result == {}

    def test_returns_empty_on_empty_input(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from src.ingestion.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all([])
        assert result == {}


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
        from src.services.fmp_client import FMPEndpointError
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


class TestGetUpsideToTarget:
    """get_upside_to_target computes score from two already-cached calls.

    Delegates entirely to get_price_target_consensus() and get_quote().
    Writes nothing to cache itself. Returns None on missing/zero data.
    """

    def test_returns_score_when_both_values_present(self, client):
        """25% upside → score 0.75."""
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote",
                              return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result == 0.75

    def test_returns_none_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        result = c.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_target_missing(self, client):
        with patch.object(client, "get_price_target_consensus", return_value={}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_price_missing(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote", return_value={}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_target_is_zero(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 0.0}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_price_is_zero(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote", return_value={"price": 0.0}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_on_exception(self, client):
        with patch.object(client, "get_price_target_consensus",
                          side_effect=RuntimeError("network error")):
            result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_does_not_write_to_cache(self, client, tmp_path):
        """get_upside_to_target is a pure computation wrapper — writes nothing."""
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                with patch.object(client, "_cache_write") as mock_write:
                    client.get_upside_to_target("AAPL")
        mock_write.assert_not_called()

    def test_at_target_scores_0_50(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 100.0}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result == 0.50


class TestGetQualityScore:
    """get_quality_score delegates to get_ratios_ttm() and score_quality_piotroski().

    Returns tuple[float, int] — dead signal is (0.0, 0), not None.
    """

    def _perfect_ratios(self) -> dict:
        return {
            "returnOnAssetsTTM":        0.10,
            "operatingProfitMarginTTM": 0.15,
            "debtEquityRatioTTM":       0.30,
            "currentRatioTTM":          2.0,
            "grossProfitMarginTTM":     0.45,
            "netProfitMarginTTM":       0.08,
        }

    def test_returns_perfect_score_for_quality_ratios(self, client):
        with patch.object(client, "get_ratios_ttm", return_value=self._perfect_ratios()):
            score, raw = client.get_quality_score("AAPL")
        assert score == 1.0
        assert raw == 8

    def test_returns_float_not_optional(self, client):
        with patch.object(client, "get_ratios_ttm", return_value=self._perfect_ratios()):
            score, raw = client.get_quality_score("AAPL")
        assert isinstance(score, float)
        assert isinstance(raw, int)

    def test_returns_0_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        score, raw = c.get_quality_score("AAPL")
        assert score == 0.0
        assert raw == 0

    def test_returns_0_when_ratios_empty(self, client):
        with patch.object(client, "get_ratios_ttm", return_value={}):
            score, raw = client.get_quality_score("AAPL")
        assert score == 0.0
        assert raw == 0

    def test_returns_0_on_exception(self, client):
        with patch.object(client, "get_ratios_ttm", side_effect=RuntimeError("timeout")):
            score, raw = client.get_quality_score("AAPL")
        assert score == 0.0
        assert raw == 0

    def test_partial_quality_ratios(self, client):
        """5 of 8 points passing → 5/8 = 0.625."""
        ratios = {
            "returnOnAssetsTTM":        0.02,   # point 1 only (not > 0.05)
            "operatingProfitMarginTTM": 0.10,   # point 3
            "debtEquityRatioTTM":       0.70,   # point 4 only (not < 0.5)
            "currentRatioTTM":          2.0,    # point 6
            "grossProfitMarginTTM":     0.40,   # point 7
            "netProfitMarginTTM":       0.02,   # fails point 8
        }
        with patch.object(client, "get_ratios_ttm", return_value=ratios):
            score, raw = client.get_quality_score("AAPL")
        assert score == round(5 / 8, 4)
        assert raw == 5


# ── PEAD: get_earnings_surprise (stable/ "earnings") ────────────────────────

class TestGetEarningsSurprise:
    """PEAD surprise from stable/ "earnings" (epsActual vs epsEstimated).

    The legacy "earnings-surprises" route is HTTP 404 on stable/; the earnings
    calendar mixes future scheduled rows (epsActual=None) with past reports.
    """

    @staticmethod
    def _days_ago(n: int) -> str:
        from datetime import datetime, timedelta, timezone
        return (datetime.now(timezone.utc).date() - timedelta(days=n)).isoformat()

    @staticmethod
    def _days_ahead(n: int) -> str:
        from datetime import datetime, timedelta, timezone
        return (datetime.now(timezone.utc).date() + timedelta(days=n)).isoformat()

    def test_beat_returns_positive_pct_and_days(self, client):
        rows = [{"date": self._days_ago(10), "epsActual": 1.15, "epsEstimated": 1.00}]
        with patch.object(client, "_get", return_value=rows):
            pct, days = client.get_earnings_surprise("AAPL")
        assert pct == pytest.approx(0.15, abs=1e-6)
        assert days == 10

    def test_future_scheduled_row_skipped(self, client):
        rows = [
            {"date": self._days_ahead(30), "epsActual": None, "epsEstimated": 1.20},
            {"date": self._days_ago(45), "epsActual": 0.90, "epsEstimated": 1.00},
        ]
        with patch.object(client, "_get", return_value=rows):
            pct, days = client.get_earnings_surprise("AAPL")
        assert pct == pytest.approx(-0.10, abs=1e-6)
        assert days == 45

    def test_miss_returns_negative_pct(self, client):
        rows = [{"date": self._days_ago(5), "epsActual": -0.50, "epsEstimated": 1.00}]
        with patch.object(client, "_get", return_value=rows):
            pct, _ = client.get_earnings_surprise("AAPL")
        assert pct == pytest.approx(-1.50, abs=1e-6)

    def test_zero_estimate_returns_none(self, client):
        """Pre-revenue: |estimate| < 1e-6 → undefined surprise %, no fallthrough."""
        rows = [{"date": self._days_ago(5), "epsActual": 0.10, "epsEstimated": 0.0}]
        with patch.object(client, "_get", return_value=rows):
            assert client.get_earnings_surprise("AAPL") == (None, 0)

    def test_empty_response_returns_none(self, client):
        with patch.object(client, "_get", return_value=[]):
            assert client.get_earnings_surprise("AAPL") == (None, 0)

    def test_result_cached_no_second_get(self, client):
        rows = [{"date": self._days_ago(10), "epsActual": 1.15, "epsEstimated": 1.00}]
        with patch.object(client, "_get", return_value=rows) as mock_get:
            first = client.get_earnings_surprise("AAPL")
            second = client.get_earnings_surprise("AAPL")
        assert mock_get.call_count == 1
        assert first == second

    def test_endpoint_error_soft_fails(self, client):
        from src.services.fmp_client import FMPEndpointError
        with patch.object(client, "_get", side_effect=FMPEndpointError("earnings", 404)):
            assert client.get_earnings_surprise("AAPL") == (None, 0)

    def test_endpoint_not_quarantined(self, client):
        """Regression: "earnings" must never join _DEAD_ENDPOINTS by accident."""
        from src.services.fmp_client import _DEAD_ENDPOINTS
        assert "earnings" not in _DEAD_ENDPOINTS
        assert "earnings-surprises" not in _DEAD_ENDPOINTS  # removed — no longer called


# ── v3.0 client additions ────────────────────────────────────────────────────

def _route(responses):
    """side_effect router: (url_tail, symbol) → payload; everything else []."""
    def _side_effect(url, params=None, timeout=None, **kw):
        params = params or {}
        path = url.split("?")[0]
        for (url_tail, symbol), payload in responses.items():
            if path.endswith(url_tail) and params.get("symbol") == symbol:
                return _ok_resp(payload)
        return _empty_resp()
    return _side_effect


def _fresh_date():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


class TestGetIncomeStatements:
    def test_returns_empty_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        assert c.get_income_statements("7203.T") == []

    def test_fetches_quarterly_with_limit(self, client):
        rows = [{"date": "2026-03-31", "revenue": 100, "operatingIncome": 10}]
        with patch.object(client._session, "get",
                          return_value=_ok_resp(rows)) as mock_get:
            result = client.get_income_statements("7203.T", period="quarter", limit=8)
        assert result == rows
        url = mock_get.call_args[0][0]
        params = mock_get.call_args[1]["params"]
        assert url.split("?")[0].endswith("/income-statement")
        assert params["symbol"] == "7203.T"
        assert params["period"] == "quarter"
        assert params["limit"] == 8

    def test_supports_annual_period(self, client):
        with patch.object(client._session, "get",
                          return_value=_ok_resp([])) as mock_get:
            client.get_income_statements("7203.T", period="annual", limit=2)
        assert mock_get.call_args[1]["params"]["period"] == "annual"

    def test_non_list_response_returns_empty(self, client):
        with patch.object(client._session, "get",
                          return_value=_ok_resp({"error": "x"})):
            assert client.get_income_statements("7203.T") == []


class TestGetBalanceSheet:
    """Accruals (Sloan 1996) totalAssets deflator — stable/balance-sheet-statement."""

    def test_returns_empty_without_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        assert c.get_balance_sheet("AAPL") == []

    def test_fetches_with_route_and_params(self, client):
        rows = [{"date": "2026-03-31", "filingDate": "2026-05-02", "totalAssets": 1.0e12}]
        with patch.object(client._session, "get",
                          return_value=_ok_resp(rows)) as mock_get:
            result = client.get_balance_sheet("AAPL", period="quarter", limit=1)
        assert result == rows
        url = mock_get.call_args[0][0]
        params = mock_get.call_args[1]["params"]
        assert url.split("?")[0].endswith("/balance-sheet-statement")
        assert params["symbol"] == "AAPL"
        assert params["period"] == "quarter"
        assert params["limit"] == 1

    def test_non_list_response_returns_empty(self, client):
        with patch.object(client._session, "get",
                          return_value=_ok_resp({"error": "x"})):
            assert client.get_balance_sheet("AAPL") == []


class TestInstitutionalOwnershipFallback:
    def test_prior_quarter_fallback_when_current_unfiled(self, client):
        # now − 45d frequently lands in the CURRENT, not-yet-ended quarter
        # (e.g., June 11 → Apr 27 → Q2), where no 13F exists yet. The client
        # must retry the previous quarter before giving up.
        calls = []

        def _route_by_quarter(url, params=None, timeout=None, **kw):
            params = params or {}
            calls.append((params.get("year"), params.get("quarter")))
            if len(calls) == 1:
                return _empty_resp()  # current quarter: unfiled
            return _ok_resp([{"symbol": "AAPL", "investorsHolding": 5000}])

        with patch.object(client._session, "get", side_effect=_route_by_quarter):
            result = client.get_institutional_ownership("AAPL")
        assert result.get("investorsHolding") == 5000
        assert len(calls) == 2
        y0, q0 = calls[0]
        y1, q1 = calls[1]
        assert (y1, q1) == ((y0, q0 - 1) if q0 > 1 else (y0 - 1, 4))

    def test_base_symbol_fallback_on_empty(self, client):
        base_row = {"symbol": "ASML", "investorsHolding": 1200,
                    "increasedPositions": 300, "reducedPositions": 200}
        responses = {
            ("/institutional-ownership/symbol-positions-summary", "ASML"): [base_row],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            result = client.get_institutional_ownership("ASML.AS")
        assert result.get("investorsHolding") == 1200

    def test_no_fallback_when_exact_hit(self, client):
        exact_row = {"symbol": "ASML.AS", "investorsHolding": 50}
        responses = {
            ("/institutional-ownership/symbol-positions-summary", "ASML.AS"): [exact_row],
        }
        with patch.object(client._session, "get",
                          side_effect=_route(responses)) as mock_get:
            result = client.get_institutional_ownership("ASML.AS")
        assert result.get("investorsHolding") == 50
        assert mock_get.call_count == 1


class TestUpsideToTargetGuards:
    """Same-symbol pairing + GBX rescue + order-of-magnitude backstop."""

    def test_same_symbol_pairing_on_base_fallback(self, client):
        # PT only exists for the base symbol → the quote MUST also come from
        # the base symbol (USD/USD), never base-target × local-quote (USD/EUR).
        from src.scoring.momentum_signals import score_price_target_upside
        responses = {
            ("/price-target-consensus", "ASML"): [
                {"symbol": "ASML", "targetConsensus": 900.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "ASML"): [{"symbol": "ASML", "price": 850.0}],
            # Local-line quote present but must NOT be used for this pairing:
            ("/quote", "ASML.AS"): [{"symbol": "ASML.AS", "price": 620.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            score = client.get_upside_to_target("ASML.AS")
        assert score == pytest.approx(score_price_target_upside(900.0, 850.0))

    def test_gbx_rescue_for_l_lines(self, client):
        # LSE quote in pence (GBX), consensus target in pounds (GBP):
        # ratio 3/250 = 0.012 ∈ [0.005, 0.02] → target ×100 → 300 vs 250.
        from src.scoring.momentum_signals import score_price_target_upside
        responses = {
            ("/price-target-consensus", "VOD.L"): [
                {"symbol": "VOD.L", "targetConsensus": 3.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "VOD.L"): [{"symbol": "VOD.L", "price": 250.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            score = client.get_upside_to_target("VOD.L")
        assert score == pytest.approx(score_price_target_upside(300.0, 250.0))

    def test_l_line_normal_ratio_not_rescued(self, client):
        from src.scoring.momentum_signals import score_price_target_upside
        responses = {
            ("/price-target-consensus", "VOD.L"): [
                {"symbol": "VOD.L", "targetConsensus": 280.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "VOD.L"): [{"symbol": "VOD.L", "price": 250.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            score = client.get_upside_to_target("VOD.L")
        assert score == pytest.approx(score_price_target_upside(280.0, 250.0))

    def test_order_of_magnitude_backstop_returns_none(self, client):
        # 20× target/price on a non-.L symbol: unrescuable scale mismatch.
        responses = {
            ("/price-target-consensus", "AAPL"): [
                {"symbol": "AAPL", "targetConsensus": 2000.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "AAPL"): [{"symbol": "AAPL", "price": 100.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            assert client.get_upside_to_target("AAPL") is None


class TestPairedTargetAndPrice:
    """_paired_target_and_price is the single source of truth feeding BOTH the
    upside score and the Discord 🎯 displayed level — so they cannot disagree
    (the SHEL.L '$102 (−96.6%)' class of bug)."""

    def test_returns_rescued_same_currency_levels_for_l_line(self, client):
        # LSE pence quote + GBP target: rescue lifts the target ×100 so the
        # DISPLAYED (target, price) are both pence → a sane bounded upside, not
        # the −96.6% garbage the un-paired display path produced.
        responses = {
            ("/price-target-consensus", "VOD.L"): [
                {"symbol": "VOD.L", "targetConsensus": 3.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "VOD.L"): [{"symbol": "VOD.L", "price": 250.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            target_f, price_f, resolved = client._paired_target_and_price("VOD.L")
        assert (target_f, price_f, resolved) == (300.0, 250.0, "VOD.L")
        assert 0.2 < target_f / price_f < 5.0     # same-unit → bounded ratio

    def test_pairs_quote_with_resolved_consensus_symbol(self, client):
        # Consensus only on the base symbol → the price MUST come from the base
        # (USD/USD), never base-target × local quote (USD/EUR).
        responses = {
            ("/price-target-consensus", "ASML"): [
                {"symbol": "ASML", "targetConsensus": 900.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "ASML"): [{"symbol": "ASML", "price": 850.0}],
            ("/quote", "ASML.AS"): [{"symbol": "ASML.AS", "price": 620.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            target_f, price_f, resolved = client._paired_target_and_price("ASML.AS")
        assert (target_f, price_f, resolved) == (900.0, 850.0, "ASML")  # not 620

    def test_none_on_currency_mismatch(self, client):
        # Unrescuable scale mismatch → None → display renders nothing (never a
        # fabricated/garbage level).
        responses = {
            ("/price-target-consensus", "AAPL"): [
                {"symbol": "AAPL", "targetConsensus": 2000.0,
                 "targetConsensusDate": _fresh_date()}],
            ("/quote", "AAPL"): [{"symbol": "AAPL", "price": 100.0}],
        }
        with patch.object(client._session, "get", side_effect=_route(responses)):
            assert client._paired_target_and_price("AAPL") is None


# ── Candidate-factor endpoints (Track A) ────────────────────────────────────

class TestLeveredDcf:
    def test_returns_dcf_value(self, client):
        payload = [{"symbol": "AAPL", "date": "2026-06-01", "dcf": 250.3,
                    "Stock Price": 210.0}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            assert client.get_levered_dcf("AAPL") == pytest.approx(250.3)

    def test_none_when_empty(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            assert client.get_levered_dcf("AAPL") is None

    def test_falls_back_to_base_symbol(self, client):
        payload_base = [{"symbol": "ASML", "dcf": 900.0}]

        def _route(url, params=None, **kw):
            return _ok_resp(payload_base) if params.get("symbol") == "ASML" else _empty_resp()

        with patch.object(client._session, "get", side_effect=_route):
            assert client.get_levered_dcf("ASML.AS") == pytest.approx(900.0)

    def test_no_key_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(cache_root=tmp_path / "fmp")
        assert c.get_levered_dcf("AAPL") is None


class TestSectorPe:
    SNAPSHOT = [
        {"date": "2026-06-19", "sector": "Technology", "exchange": "NASDAQ", "pe": 35.2},
        {"date": "2026-06-19", "sector": "Energy", "exchange": "NASDAQ", "pe": 12.0},
    ]

    def test_returns_matching_sector_pe(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp(self.SNAPSHOT)):
            assert client.get_sector_pe(
                "Technology", exchange="NASDAQ", date="2026-06-19"
            ) == pytest.approx(35.2)

    def test_case_insensitive_match(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp(self.SNAPSHOT)):
            assert client.get_sector_pe(
                "energy", exchange="NASDAQ", date="2026-06-19"
            ) == pytest.approx(12.0)

    def test_none_for_unknown_sector(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp(self.SNAPSHOT)):
            assert client.get_sector_pe(
                "Utilities", exchange="NASDAQ", date="2026-06-19"
            ) is None

    def test_snapshot_cached_one_fetch_per_exchange_date(self, client):
        with patch.object(client._session, "get",
                          return_value=_ok_resp(self.SNAPSHOT)) as mock_get:
            client.get_sector_pe("Technology", exchange="NASDAQ", date="2026-06-19")
            client.get_sector_pe("Energy", exchange="NASDAQ", date="2026-06-19")
        assert mock_get.call_count == 1   # second sector served from cached snapshot

    def test_none_when_empty(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            assert client.get_sector_pe(
                "Technology", exchange="NASDAQ", date="2026-06-19"
            ) is None
