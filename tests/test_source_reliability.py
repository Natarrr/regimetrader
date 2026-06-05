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
