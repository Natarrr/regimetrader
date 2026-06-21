"""tests/mcp/test_artifacts.py — read-only artifact access layer for the MCP server.

The MCP server NEVER calls FMP live (CLAUDE.md §1 — status from artifact state,
not live scraping); it reads the committed pipeline outputs under logs/. These
tests use synthetic fixtures mirroring the real shapes of
intel_source_status.json / top_lists.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mcp.artifacts import ArtifactStore


@pytest.fixture
def logs(tmp_path: Path) -> Path:
    (tmp_path / "intel_source_status.json").write_text(json.dumps({
        "source_meta": {"fmp": {"last_updated": "2026-06-20"}, "edgar": {}, "none": {}},
        "_edgar_meta": {"run_duration": 12.0, "ticker_count": 2, "error_count": 0},
        "results": [
            {"ticker": "AAPL", "sector": "Technology", "cap_tier": "mega",
             "market_cap": 3.2e12, "insider_conviction_score": 0.33,
             "momentum_long_score": 0.71, "news_sentiment_score": 0.5},
            {"ticker": "XOM", "sector": "Energy", "cap_tier": "large",
             "market_cap": 4.5e11, "insider_conviction_score": 0.0,
             "momentum_long_score": 0.4},
        ],
    }), encoding="utf-8")
    (tmp_path / "top_lists.json").write_text(json.dumps({
        "top_buys_usa": [
            {"ticker": "AAPL", "final_score": 0.62, "badge": "buy",
             "market": "USA", "factors": {"momentum_long_score": 0.71}},
        ],
        "top_buys_europe": [
            {"ticker": "SAP.DE", "final_score": 0.55, "badge": "watch",
             "market": "EUROPE", "factors": {}},
        ],
        "top_buys_asia": [],
        "vix": 17.3, "vix_regime": "Normal", "kill_switch": False,
        "ticker_count": 83, "generated_at": "2026-06-21T00:30:00Z",
    }), encoding="utf-8")
    return tmp_path


class TestTickerScore:
    def test_merges_factors_and_final_score(self, logs):
        store = ArtifactStore(logs)
        out = store.ticker_score("AAPL")
        assert out["ticker"] == "AAPL"
        assert out["sector"] == "Technology"
        assert out["final_score"] == 0.62        # joined from top_lists
        assert out["badge"] == "buy"
        assert out["factors"]["momentum_long_score"] == 0.71

    def test_case_insensitive(self, logs):
        assert ArtifactStore(logs).ticker_score("aapl")["ticker"] == "AAPL"

    def test_unknown_ticker_returns_none(self, logs):
        assert ArtifactStore(logs).ticker_score("NVDA") is None

    def test_ticker_without_toplist_has_no_final_score(self, logs):
        out = ArtifactStore(logs).ticker_score("XOM")
        assert out["ticker"] == "XOM"
        assert out.get("final_score") is None


class TestToplists:
    def test_all_markets(self, logs):
        out = ArtifactStore(logs).toplists()
        tickers = {r["ticker"] for r in out["names"]}
        assert tickers == {"AAPL", "SAP.DE"}     # empty asia list contributes none
        assert out["regime"]["vix_regime"] == "Normal"

    def test_filter_by_market(self, logs):
        out = ArtifactStore(logs).toplists(market="europe")
        assert [r["ticker"] for r in out["names"]] == ["SAP.DE"]

    def test_filter_by_badge(self, logs):
        out = ArtifactStore(logs).toplists(badge="buy")
        assert [r["ticker"] for r in out["names"]] == ["AAPL"]


class TestRegime:
    def test_regime_fields(self, logs):
        r = ArtifactStore(logs).regime()
        assert r == {"vix": 17.3, "vix_regime": "Normal", "kill_switch": False}


class TestSourceHealth:
    def test_exposes_source_meta_and_edgar_meta(self, logs):
        h = ArtifactStore(logs).source_health()
        assert "fmp" in h["source_meta"]
        assert h["edgar_meta"]["ticker_count"] == 2


class TestSearchUniverse:
    def test_min_score_filter(self, logs):
        rows = ArtifactStore(logs).search_universe(min_score=0.6)
        assert [r["ticker"] for r in rows] == ["AAPL"]

    def test_market_filter(self, logs):
        rows = ArtifactStore(logs).search_universe(market="europe")
        assert [r["ticker"] for r in rows] == ["SAP.DE"]

    def test_sorted_descending_by_score(self, logs):
        rows = ArtifactStore(logs).search_universe()
        scores = [r["final_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)


class TestMissingArtifacts:
    def test_missing_files_degrade_gracefully(self, tmp_path):
        store = ArtifactStore(tmp_path)   # empty dir
        assert store.ticker_score("AAPL") is None
        assert store.toplists()["names"] == []
        assert store.regime() == {"vix": None, "vix_regime": None, "kill_switch": None}
        assert store.search_universe() == []
