"""tests/mcp/test_tools.py — pydantic-validated tool handlers + handler factory.

Mirrors the external fmp-mcp `createToolHandler` pattern (uniform error handling
+ structured output), with Zod replaced by pydantic. The handlers are pure over
an ArtifactStore, so no MCP SDK is needed to test them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mcp.artifacts import ArtifactStore
from src.mcp.tools import build_tools


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    (tmp_path / "top_lists.json").write_text(json.dumps({
        "top_buys_usa": [{"ticker": "AAPL", "final_score": 0.62, "badge": "buy",
                          "market": "USA", "factors": {}}],
        "top_buys_europe": [], "top_buys_asia": [],
        "vix": 17.3, "vix_regime": "Normal", "kill_switch": False,
    }), encoding="utf-8")
    (tmp_path / "intel_source_status.json").write_text(json.dumps({
        "source_meta": {"fmp": {"last_updated": "2026-06-20"}},
        "_edgar_meta": {"ticker_count": 1},
        "results": [{"ticker": "AAPL", "sector": "Technology",
                     "momentum_long_score": 0.71}],
    }), encoding="utf-8")
    return ArtifactStore(tmp_path)


def _tool(store, name):
    return next(t for t in build_tools(store) if t.name == name)


class TestRegistry:
    def test_exposes_expected_tools(self, store):
        names = {t.name for t in build_tools(store)}
        assert names == {
            "get_ticker_score", "get_toplists", "get_regime",
            "get_source_health", "search_universe",
        }

    def test_each_tool_has_description_and_schema(self, store):
        for t in build_tools(store):
            assert t.description
            assert t.input_schema["type"] == "object"   # JSON-schema for MCP


class TestTickerScoreTool:
    def test_valid_returns_ok_data(self, store):
        out = _tool(store, "get_ticker_score").handler({"ticker": "AAPL"})
        assert out["ok"] is True
        assert out["data"]["ticker"] == "AAPL"

    def test_missing_required_ticker_is_validation_error(self, store):
        out = _tool(store, "get_ticker_score").handler({})
        assert out["ok"] is False
        assert "ticker" in out["error"].lower()

    def test_unknown_ticker_is_ok_with_null_data(self, store):
        out = _tool(store, "get_ticker_score").handler({"ticker": "NVDA"})
        assert out["ok"] is True
        assert out["data"] is None     # absence is valid, not an error


class TestSearchUniverseTool:
    def test_min_score_out_of_range_rejected(self, store):
        out = _tool(store, "search_universe").handler({"min_score": 1.5})
        assert out["ok"] is False

    def test_filters_apply(self, store):
        out = _tool(store, "search_universe").handler({"min_score": 0.6})
        assert [r["ticker"] for r in out["data"]] == ["AAPL"]


class TestNoArgTools:
    def test_regime(self, store):
        out = _tool(store, "get_regime").handler({})
        assert out["ok"] is True
        assert out["data"]["vix_regime"] == "Normal"

    def test_source_health(self, store):
        out = _tool(store, "get_source_health").handler({})
        assert out["ok"] is True
        assert "fmp" in out["data"]["source_meta"]


class TestHandlerIsolatesErrors:
    def test_underlying_exception_becomes_error_dict(self, store, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("disk gone")
        monkeypatch.setattr(store, "regime", _boom)
        out = _tool(store, "get_regime").handler({})
        assert out["ok"] is False
        assert "disk gone" in out["error"]
