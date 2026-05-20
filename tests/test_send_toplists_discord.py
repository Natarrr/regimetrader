"""tests/test_send_toplists_discord.py
Tests for satellite integration in send_toplists_discord.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.send_toplists_discord import _load_satellite, build_payload


# ── Fixture: minimal valid top_lists ─────────────────────────────────────────

def _top_lists() -> dict:
    return {
        "generated_at":  "2026-05-20T08:00:00+00:00",
        "source_run_id": "test-run",
        "ticker_count":  5,
        "weights":       {},
        "kill_switch":   False,
        "vix":           18.0,
        "top_buys":      [
            {"ticker": "PLTR", "final_score": 0.75, "badge": "HIGH BUY",
             "factors": {"edgar": 0.8, "insider": 0.7, "congress": 0.6,
                         "news": 0.65, "momentum": 0.6}, "ceo_buy": False}
        ],
        "mid_caps":   [],
        "small_caps": [],
    }


def _satellite() -> dict:
    return {
        "generated_at": "2026-05-20T08:41:00+00:00",
        "month":        "May",
        "status":       "success",
        "cyclicals": [
            {"ticker": "PLTR", "win_rate": 0.75, "median_return": 0.031, "years": 9}
        ],
        "cannibals": [
            {"ticker": "SQ", "buyback_yield": 0.048, "pe": 18.2, "price_vs_52w_low": 1.18}
        ],
    }


# ── _load_satellite ───────────────────────────────────────────────────────────

class TestLoadSatellite:
    def test_returns_none_on_missing_file(self, tmp_path):
        assert _load_satellite(tmp_path) is None

    def test_returns_none_on_corrupt_json(self, tmp_path):
        (tmp_path / "satellite_insights.json").write_text("not json", encoding="utf-8")
        assert _load_satellite(tmp_path) is None

    def test_returns_none_on_non_dict_json(self, tmp_path):
        (tmp_path / "satellite_insights.json").write_text(
            json.dumps([1, 2, 3]), encoding="utf-8"
        )
        assert _load_satellite(tmp_path) is None

    def test_returns_dict_on_valid_file(self, tmp_path):
        data = _satellite()
        (tmp_path / "satellite_insights.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = _load_satellite(tmp_path)
        assert result == data


# ── build_payload with satellite ──────────────────────────────────────────────

class TestBuildPayloadSatellite:
    def test_without_satellite_embed_unchanged(self):
        """satellite=None → embed fields identical to original 4-field structure."""
        payload = build_payload(_top_lists(), satellite=None)
        fields = payload["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "🌀 Seasonal Cyclicals — May" not in field_names
        assert "🐷 Share Cannibals — Buyback Yield" not in field_names
        assert len(fields) == 4

    def test_with_satellite_adds_cyclical_and_cannibal_fields(self):
        """satellite with non-empty lists → 6 embed fields total."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        fields = payload["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "🌀 Seasonal Cyclicals — May" in field_names
        assert "🐷 Share Cannibals — Buyback Yield" in field_names
        assert len(fields) == 6

    def test_cyclical_field_content(self):
        """Cyclical field renders win-rate and median correctly."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        fields = payload["embeds"][0]["fields"]
        cyclical_field = next(f for f in fields if "Cyclicals" in f["name"])
        assert "PLTR" in cyclical_field["value"]
        assert "75%" in cyclical_field["value"]
        assert "+3.1%" in cyclical_field["value"]

    def test_cannibal_field_content(self):
        """Cannibal field renders yield, P/E, and price ratio correctly."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        fields = payload["embeds"][0]["fields"]
        cannibal_field = next(f for f in fields if "Cannibals" in f["name"])
        assert "SQ" in cannibal_field["value"]
        assert "4.8%" in cannibal_field["value"]
        assert "18.2" in cannibal_field["value"]

    def test_empty_cyclicals_no_cyclical_field(self):
        """If cyclicals is empty, no cyclical embed field added."""
        sat = _satellite()
        sat["cyclicals"] = []
        payload = build_payload(_top_lists(), satellite=sat)
        field_names = [f["name"] for f in payload["embeds"][0]["fields"]]
        assert "🌀 Seasonal Cyclicals — May" not in field_names

    def test_empty_cannibals_no_cannibal_field(self):
        """If cannibals is empty, no cannibal embed field added."""
        sat = _satellite()
        sat["cannibals"] = []
        payload = build_payload(_top_lists(), satellite=sat)
        field_names = [f["name"] for f in payload["embeds"][0]["fields"]]
        assert "🐷 Share Cannibals — Buyback Yield" not in field_names

    def test_factor_legend_always_last(self):
        """Factor Legend must always be the last embed field."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        last_field = payload["embeds"][0]["fields"][-1]
        assert "Factor Legend" in last_field["name"]
