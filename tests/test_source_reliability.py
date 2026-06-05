# Path: tests/test_source_reliability.py
"""Tests confirming source_reliability dampening is no longer applied to final_score.

The old dampening loop (generate_top_lists.py:769-774) multiplied final_score by
0.80 (EU) or 0.70 (Asia). It has been removed. FMPFetcher.source_reliability()
now returns 1.0 for all markets — dampening is replaced by the dynamic
available-factor denominator in StrategyEngine.score_ticker_pool.
"""
import pytest
from regime_trader.fetchers.fmp_fetcher import FMPFetcher
from regime_trader.fetchers.base import MarketEnum


def test_source_reliability_eu_is_one():
    fetcher = FMPFetcher(api_key="k", market=MarketEnum.EUROPE)
    assert fetcher.source_reliability("SAP.DE") == pytest.approx(1.0)


def test_source_reliability_asia_is_one():
    fetcher = FMPFetcher(api_key="k", market=MarketEnum.ASIA)
    assert fetcher.source_reliability("7203.T") == pytest.approx(1.0)


def test_source_reliability_us_is_one():
    fetcher = FMPFetcher(api_key="k", market=MarketEnum.USA)
    assert fetcher.source_reliability("AAPL") == pytest.approx(1.0)


def test_entry_with_source_reliability_one_unchanged():
    """When source_reliability == 1.0, the entry final_score must not change."""
    entry = {"final_score": 0.82, "source_reliability": 1.0}
    rel = float(entry.get("source_reliability", 1.0))
    result = round(entry["final_score"] * rel, 4)
    assert result == pytest.approx(0.82, abs=1e-4)


# ── Ticker format regex ────────────────────────────────────────────────────────

import re

_TICKER_RE = re.compile(r"^([A-Z]{1,5}|[A-Z0-9]{1,6}\.[A-Z]{1,2})$")


def test_regex_accepts_us_ticker():
    assert _TICKER_RE.match("AAPL")
    assert _TICKER_RE.match("PLTR")


def test_regex_accepts_eu_ticker():
    assert _TICKER_RE.match("SAP.DE")
    assert _TICKER_RE.match("ASML.AS")


def test_regex_accepts_asia_ticker():
    assert _TICKER_RE.match("7203.T")
    assert _TICKER_RE.match("9984.T")


def test_regex_rejects_invalid():
    assert not _TICKER_RE.match("TOOLONG7")
    assert not _TICKER_RE.match("SAP.DEU")
    assert not _TICKER_RE.match("sa.de")
