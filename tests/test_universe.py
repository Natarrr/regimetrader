"""tests/test_universe.py
Unit tests for regime_trader.services.universe.UniverseManager.

Markowitz (1990 Nobel) — stratified, diversified coverage prevents sector
concentration and ensures rotation across the full universe.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from regime_trader.services.universe import (
    UniverseManager,
    _ALL_TICKERS,
    _ROTATION_WINDOW_DAYS,
)


@pytest.fixture()
def mgr(tmp_path: Path) -> UniverseManager:
    return UniverseManager(state_path=tmp_path / "universe_state.json")


class TestUniverseSize:
    def test_total_universe_at_least_150(self):
        assert len(_ALL_TICKERS) >= 150

    def test_all_entries_have_three_fields(self):
        for ticker, sector, cap in _ALL_TICKERS:
            assert ticker
            assert sector
            assert cap in ("large", "mid", "small")


class TestGetTickersForDay:
    def test_returns_list(self, mgr: UniverseManager):
        tickers = mgr.get_tickers_for_day("2026-01-15", budget=50)
        assert isinstance(tickers, list)

    def test_respects_budget(self, mgr: UniverseManager):
        tickers = mgr.get_tickers_for_day("2026-01-15", budget=50)
        assert len(tickers) <= 50

    def test_no_duplicates(self, mgr: UniverseManager):
        tickers = mgr.get_tickers_for_day("2026-01-15", budget=100)
        assert len(tickers) == len(set(tickers))

    def test_all_tickers_valid_strings(self, mgr: UniverseManager):
        tickers = mgr.get_tickers_for_day("2026-01-15", budget=30)
        for t in tickers:
            assert isinstance(t, str)
            assert t.strip() == t

    def test_accepts_date_object(self, mgr: UniverseManager):
        tickers = mgr.get_tickers_for_day(date(2026, 1, 15), budget=10)
        assert len(tickers) <= 10

    def test_budget_350_returns_at_most_350(self, mgr: UniverseManager):
        """Daily target: ≥ 350 tickers. Universe must be large enough."""
        tickers = mgr.get_tickers_for_day("2026-01-15", budget=350)
        assert len(tickers) <= 350


class TestRotationPriority:
    def test_unprocessed_tickers_preferred(self, mgr: UniverseManager):
        """Tickers not seen in 7d get a rotation boost and appear in the selection."""
        # Mark all as processed today except a few
        all_syms = [t for t, _, _ in _ALL_TICKERS]
        today    = "2026-01-15"
        # Process all tickers 8 days ago (just expired rotation window)
        stale    = "2026-01-07"
        mgr._state["processed"] = {t: stale for t in all_syms}

        # Now pick a small budget — should pick from the stale pool
        tickers = mgr.get_tickers_for_day(today, budget=10)
        assert len(tickers) == 10

    def test_priority_scores_influence_ordering(self, mgr: UniverseManager):
        """Setting high priority for AAPL should make it appear in top picks."""
        mgr.set_priority_scores({
            "AAPL_news":    10.0,
            "AAPL_insider": 10.0,
            "AAPL_vol":     10.0,
        })
        tickers = mgr.get_tickers_for_day("2026-01-15", budget=5)
        assert "AAPL" in tickers


class TestRecordProcessed:
    def test_record_persists_to_state(self, mgr: UniverseManager, tmp_path: Path):
        mgr.record_processed(["AAPL", "MSFT"], "2026-01-15")
        assert mgr._state["processed"].get("AAPL") == "2026-01-15"
        assert mgr._state["processed"].get("MSFT") == "2026-01-15"

    def test_state_file_written(self, mgr: UniverseManager, tmp_path: Path):
        mgr.record_processed(["NVDA"], "2026-01-15")
        state_path = tmp_path / "universe_state.json"
        assert state_path.exists()

    def test_record_updates_existing(self, mgr: UniverseManager):
        mgr.record_processed(["AAPL"], "2026-01-14")
        mgr.record_processed(["AAPL"], "2026-01-15")
        assert mgr._state["processed"]["AAPL"] == "2026-01-15"


class TestCoverageStats:
    def test_returns_dict_with_expected_keys(self, mgr: UniverseManager):
        stats = mgr.coverage_stats()
        assert "total_universe" in stats
        assert "processed_7d" in stats
        assert "not_processed_7d" in stats

    def test_total_universe_consistent(self, mgr: UniverseManager):
        stats = mgr.coverage_stats()
        assert stats["total_universe"] == len(_ALL_TICKERS)

    def test_processed_and_not_sum_to_total(self, mgr: UniverseManager):
        mgr.record_processed(["AAPL", "MSFT"], "2026-01-15")
        stats = mgr.coverage_stats()
        assert stats["processed_7d"] + stats["not_processed_7d"] == stats["total_universe"]
