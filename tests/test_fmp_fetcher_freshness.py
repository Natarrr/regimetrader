# Path: tests/test_fmp_fetcher_freshness.py
"""Freshness gates inside FMPFetcher._fetch_all_factors (audit F1/F2).

F2 — a price series whose newest EOD bar is older than the recency threshold is
     stale (halt/delist/feed gap) → the ticker is dropped (return None) before it
     can emit phantom momentum / return_5d into the extension gate.
F1 — a present-but-stale quote during US RTH drops quote-derived FMP factors for
     a US listing (price-only path); INTL listings and after-hours are kept, with
     the age recorded in source_diagnostics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.core.fetchers_base import MarketEnum
from src.ingestion import fmp_fetcher as ff
from src.ingestion.fmp_fetcher import FMPFetcher, _should_reject_stale_quote
from src.services.fmp_client import fmp_prices_to_arrays
from src.scoring.momentum_signals import (
    score_momentum_long, score_volume_attention,
    score_quality_piotroski, score_price_target_upside,
)
from src.scoring.news_signals import score_news_sentiment, score_news_buzz
from src.scoring.insider_signals import (
    score_insider_conviction, score_insider_breadth, orthogonalize_insider_scores,
)
from src.scoring.analyst import _score_record as ac_score_record

_UTC = timezone.utc


def _price_rows(latest_age_days: int, n: int = 12) -> list[dict]:
    """FMP historical-price rows (newest-first). row[0] is `latest_age_days` old."""
    today = datetime.now(_UTC).date()
    rows = []
    for i in range(n):
        d = today - timedelta(days=latest_age_days + i)
        rows.append({
            "symbol": "X", "date": d.isoformat(),
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000,
        })
    return rows


class _FakeClient:
    """Minimal client: only the methods the early gates touch are needed, since a
    stale-series drop returns before quote fetch and an empty quote routes to the
    price-only path before any other endpoint is called."""

    def __init__(self, rows, quote):
        self._rows = rows
        self._quote = quote

    def get_historical_prices(self, ticker, limit=280):
        return self._rows

    def get_quote(self, ticker, *a, **k):
        return self._quote


def _run(fetcher, ticker, client):
    return fetcher._fetch_all_factors(
        ticker, client,
        score_momentum_long, score_volume_attention,
        score_quality_piotroski, score_price_target_upside,
        score_news_sentiment, score_news_buzz,
        score_insider_conviction, score_insider_breadth,
        orthogonalize_insider_scores, ac_score_record,
        fmp_prices_to_arrays,
    )


class TestPriceSeriesRecencyGate:
    def test_stale_series_drops_ticker(self):
        # newest bar 6 days old (> 4-day default) → dropped
        fetcher = FMPFetcher(api_key="k", market=MarketEnum.USA)
        client = _FakeClient(_price_rows(latest_age_days=6), quote={})
        assert _run(fetcher, "AAPL", client) is None

    def test_recent_series_survives_gate(self):
        # newest bar 1 day old → not dropped by recency; empty quote → price-only dict
        fetcher = FMPFetcher(api_key="k", market=MarketEnum.USA)
        client = _FakeClient(_price_rows(latest_age_days=1), quote={})
        rf = _run(fetcher, "AAPL", client)
        assert rf is not None
        assert "momentum_long_score" in rf

    def test_insufficient_history_still_dropped(self):
        # fewer than 5 closes → existing guard still returns None
        fetcher = FMPFetcher(api_key="k", market=MarketEnum.USA)
        client = _FakeClient(_price_rows(latest_age_days=1, n=3), quote={})
        assert _run(fetcher, "AAPL", client) is None


# 2026-06-22 Monday, EDT (UTC-4): 14:00 UTC = 10:00 ET (RTH); 23:00 UTC = 19:00 ET (closed)
_RTH = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
_AFTER_HOURS = datetime(2026, 6, 22, 23, 0, tzinfo=_UTC)


def _quote(now: datetime, age_sec: float) -> dict:
    return {"timestamp": now.timestamp() - age_sec, "marketCap": 1e9, "price": 100.0}


class TestShouldRejectStaleQuote:
    def test_us_listing_stale_during_rth_rejects(self):
        assert _should_reject_stale_quote("AAPL", _quote(_RTH, 20 * 60), now=_RTH) is True

    def test_us_listing_fresh_during_rth_keeps(self):
        assert _should_reject_stale_quote("AAPL", _quote(_RTH, 60), now=_RTH) is False

    def test_us_listing_after_hours_keeps_even_if_stale(self):
        assert _should_reject_stale_quote(
            "AAPL", _quote(_AFTER_HOURS, 20 * 60), now=_AFTER_HOURS) is False

    def test_intl_listing_during_rth_keeps_even_if_stale(self):
        assert _should_reject_stale_quote(
            "ASML.AS", _quote(_RTH, 20 * 60), now=_RTH) is False

    def test_missing_timestamp_keeps(self):
        assert _should_reject_stale_quote("AAPL", {"price": 100.0}, now=_RTH) is False


class TestQuoteFreshnessWiring:
    def test_stale_us_quote_during_rth_routes_to_price_only(self, monkeypatch):
        # Force RTH so the wiring is deterministic regardless of wall clock.
        monkeypatch.setattr(ff, "is_us_rth", lambda now=None: True)
        now = datetime.now(_UTC)
        stale_quote = {"timestamp": now.timestamp() - 3600, "marketCap": 5e9, "price": 100.0}
        fetcher = FMPFetcher(api_key="k", market=MarketEnum.USA)
        client = _FakeClient(_price_rows(latest_age_days=1), quote=stale_quote)
        rf = _run(fetcher, "AAPL", client)
        # Routed to the price-only path: quote-derived factors dropped, mcap zeroed.
        assert rf is not None
        assert rf["market_cap"] == 0.0
        assert rf["price_target_upside_score"] is None
        assert "source_diagnostics" not in rf  # full path would carry this
        assert rf["momentum_long_score"] is not None  # price factors retained
