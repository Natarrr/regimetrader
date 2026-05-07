"""tests/test_top_lists.py
Unit tests for generate_top_lists.py and send_toplists_discord.py.

Covers:
  - compute_insider_score: CEO buy bonus, buy/sell ratio, amendment penalty
  - assign_tier: market cap tier assignment within a universe
  - compute_macro_score: pipeline health → macro proxy
  - generate(): end-to-end output structure validation
  - build_payload(): Discord embed structure validation
  - build_alert_payload(): alert embed when file is missing
  - send_toplists_discord.main(): dry-run against sample_top_lists.json

Historical validation anchor:
  CEO open-market purchase (AAPL fixture) should produce final_score > 0.70.
  Mass insider selling (NVDA fixture) should produce final_score < 0.45.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# ── Adjust sys.path to import from project root ────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.market_intel.generate_top_lists import (
    assign_tier,
    compute_insider_score,
    compute_macro_score,
    generate,
)
from scripts.send_toplists_discord import (
    build_alert_payload,
    build_payload,
    _format_ticker_block,
    _section_value,
    main as discord_main,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_EVENTS = Path(__file__).parent / "data" / "sample_marketintel_events.json"
SAMPLE_LISTS  = Path(__file__).parent / "data" / "sample_top_lists.json"

DEFAULT_WEIGHTS = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "macro":    0.10,
}


# ── compute_insider_score ──────────────────────────────────────────────────────

class TestComputeInsiderScore:
    def test_ceo_buy_gives_high_score(self):
        """CEO open-market purchase → strong conviction signal > 0.75."""
        bd = {"ceo_buy": True, "buy_count": 1, "sell_count": 0, "amendment_count": 0}
        score = compute_insider_score(bd)
        assert score > 0.75, f"CEO buy should score > 0.75, got {score}"

    def test_pure_selling_gives_low_score(self):
        """11 sells, 0 buys, no CEO → score well below 0.50."""
        bd = {"ceo_buy": False, "buy_count": 0, "sell_count": 11, "amendment_count": 0}
        score = compute_insider_score(bd)
        assert score < 0.50, f"Pure selling should score < 0.50, got {score}"

    def test_neutral_activity_near_half(self):
        """No buys, no sells → score = 0.50 (neutral)."""
        bd = {"ceo_buy": False, "buy_count": 0, "sell_count": 0, "amendment_count": 0}
        score = compute_insider_score(bd)
        assert abs(score - 0.50) < 0.01, f"Neutral should be ~0.50, got {score}"

    def test_amendment_penalty_applied(self):
        """Amendment count > 0 reduces score slightly."""
        clean  = compute_insider_score({"ceo_buy": False, "buy_count": 1, "sell_count": 1, "amendment_count": 0})
        amend  = compute_insider_score({"ceo_buy": False, "buy_count": 1, "sell_count": 1, "amendment_count": 1})
        assert amend < clean, "Amendment should reduce score"

    def test_score_clamped_0_to_1(self):
        """Score is always in [0, 1] regardless of inputs."""
        bd = {"ceo_buy": True, "buy_count": 100, "sell_count": 0, "amendment_count": 0}
        assert 0.0 <= compute_insider_score(bd) <= 1.0
        bd = {"ceo_buy": False, "buy_count": 0, "sell_count": 100, "amendment_count": 5}
        assert 0.0 <= compute_insider_score(bd) <= 1.0

    def test_buy_sell_ratio_directional(self):
        """More buys than sells → score above 0.50."""
        bd_buy  = {"ceo_buy": False, "buy_count": 4, "sell_count": 1, "amendment_count": 0}
        bd_sell = {"ceo_buy": False, "buy_count": 1, "sell_count": 4, "amendment_count": 0}
        assert compute_insider_score(bd_buy) > 0.50
        assert compute_insider_score(bd_sell) < 0.50


# ── assign_tier ────────────────────────────────────────────────────────────────

class TestAssignTier:
    def test_five_ticker_tiers(self):
        """5 tickers → correct large/mid/small distribution."""
        caps = [
            ("NVDA", 5000e9), ("AAPL", 4000e9),   # top 40% → large (2)
            ("JPM",  800e9), ("DIS", 150e9),        # middle 35% → mid (1-2)
            ("INTC",  90e9),                        # bottom 25% → small (1)
        ]
        tiers = assign_tier(caps)
        assert tiers["NVDA"] == "large"
        assert tiers["AAPL"] == "large"
        assert tiers["INTC"] == "small"

    def test_all_tiers_assigned(self):
        """Every ticker gets a tier."""
        caps = [(f"T{i}", float(i * 1e9)) for i in range(1, 11)]
        tiers = assign_tier(caps)
        assert len(tiers) == 10
        assert all(v in ("large", "mid", "small") for v in tiers.values())

    def test_empty_list_returns_empty(self):
        assert assign_tier([]) == {}

    def test_single_ticker_is_large(self):
        tiers = assign_tier([("AAPL", 4000e9)])
        assert tiers["AAPL"] == "large"


# ── compute_macro_score ────────────────────────────────────────────────────────

class TestComputeMacroScore:
    def test_none_metrics_returns_neutral(self):
        assert compute_macro_score(None) == 0.50

    def test_full_coverage_bonus(self):
        metrics = {"ticker_count": 50, "edgar_count": 50, "error_count": 0}
        score = compute_macro_score(metrics)
        assert score > 0.55, "Full coverage should give bonus"

    def test_low_coverage_penalty(self):
        metrics = {"ticker_count": 50, "edgar_count": 20, "error_count": 0}
        score = compute_macro_score(metrics)
        assert score < 0.55, "Low coverage should reduce score"

    def test_high_error_rate_penalty(self):
        metrics = {"ticker_count": 50, "edgar_count": 45, "error_count": 10}
        clean   = compute_macro_score({"ticker_count": 50, "edgar_count": 45, "error_count": 0})
        errored = compute_macro_score(metrics)
        assert errored < clean, "High error rate should reduce macro score"

    def test_score_always_in_range(self):
        for coverage in [0, 20, 50, 100]:
            m = {"ticker_count": 100, "edgar_count": coverage, "error_count": 0}
            s = compute_macro_score(m)
            assert 0.0 <= s <= 1.0


# ── generate() end-to-end ──────────────────────────────────────────────────────

class TestGenerate:
    def test_output_structure(self, tmp_path):
        """generate() with sample events produces correct top_lists structure."""
        import shutil
        shutil.copy(SAMPLE_EVENTS, tmp_path / "marketintel_events.json")

        result = generate(
            events_path  = tmp_path / "marketintel_events.json",
            metrics_path = tmp_path / "metrics.json",   # doesn't exist — OK
            output_path  = tmp_path / "top_lists.json",
            output_csv   = tmp_path / "top5.csv",
            weights      = DEFAULT_WEIGHTS,
            top_n        = 5,
            force        = True,
        )

        assert "top_buys"   in result
        assert "mid_caps"   in result
        assert "small_caps" in result
        assert "generated_at" in result
        assert "weights" in result

    def test_top_buys_sorted_descending(self, tmp_path):
        """top_buys list is sorted by final_score descending."""
        import shutil
        shutil.copy(SAMPLE_EVENTS, tmp_path / "marketintel_events.json")

        result = generate(
            events_path  = tmp_path / "marketintel_events.json",
            metrics_path = tmp_path / "metrics.json",
            output_path  = tmp_path / "top_lists.json",
            output_csv   = tmp_path / "top5.csv",
            weights      = DEFAULT_WEIGHTS,
            top_n        = 5,
            force        = True,
        )
        scores = [t["final_score"] for t in result["top_buys"]]
        assert scores == sorted(scores, reverse=True), "top_buys must be sorted descending"

    def test_ceo_buy_ticker_ranks_high(self, tmp_path):
        """AAPL (CEO buy in fixture) should be #1 in top_buys."""
        import shutil
        shutil.copy(SAMPLE_EVENTS, tmp_path / "marketintel_events.json")

        result = generate(
            events_path  = tmp_path / "marketintel_events.json",
            metrics_path = tmp_path / "metrics.json",
            output_path  = tmp_path / "top_lists.json",
            output_csv   = tmp_path / "top5.csv",
            weights      = DEFAULT_WEIGHTS,
            top_n        = 5,
            force        = True,
        )
        top_ticker = result["top_buys"][0]["ticker"]
        assert top_ticker == "AAPL", f"Expected AAPL (CEO buy) as #1, got {top_ticker}"

    def test_mass_seller_ranks_last(self, tmp_path):
        """NVDA (mass insider selling) should rank at the bottom."""
        import shutil
        shutil.copy(SAMPLE_EVENTS, tmp_path / "marketintel_events.json")

        result = generate(
            events_path  = tmp_path / "marketintel_events.json",
            metrics_path = tmp_path / "metrics.json",
            output_path  = tmp_path / "top_lists.json",
            output_csv   = tmp_path / "top5.csv",
            weights      = DEFAULT_WEIGHTS,
            top_n        = 5,
            force        = True,
        )
        last_ticker = result["top_buys"][-1]["ticker"]
        assert last_ticker == "NVDA", f"Expected NVDA (mass selling) as last, got {last_ticker}"

    def test_csv_produced(self, tmp_path):
        """top5.csv is written and contains a header + rows."""
        import shutil
        shutil.copy(SAMPLE_EVENTS, tmp_path / "marketintel_events.json")
        csv_path = tmp_path / "top5.csv"

        generate(
            events_path  = tmp_path / "marketintel_events.json",
            metrics_path = tmp_path / "metrics.json",
            output_path  = tmp_path / "top_lists.json",
            output_csv   = csv_path,
            weights      = DEFAULT_WEIGHTS,
            top_n        = 5,
            force        = True,
        )
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 2, "CSV should have header + at least 1 data row"
        assert "ticker" in lines[0]

    def test_freshness_skip(self, tmp_path):
        """Without --force, a recent file is not regenerated."""
        import shutil, json as _json
        shutil.copy(SAMPLE_EVENTS, tmp_path / "marketintel_events.json")
        out = tmp_path / "top_lists.json"

        # First run
        generate(
            events_path=tmp_path/"marketintel_events.json",
            metrics_path=tmp_path/"metrics.json",
            output_path=out,
            output_csv=tmp_path/"top5.csv",
            weights=DEFAULT_WEIGHTS,
            top_n=5,
            force=True,
        )
        mtime_1 = out.stat().st_mtime

        # Second run without force — should skip (file was just written)
        generate(
            events_path=tmp_path/"marketintel_events.json",
            metrics_path=tmp_path/"metrics.json",
            output_path=out,
            output_csv=tmp_path/"top5.csv",
            weights=DEFAULT_WEIGHTS,
            top_n=5,
            force=False,
        )
        mtime_2 = out.stat().st_mtime
        assert mtime_1 == mtime_2, "File should not be rewritten without --force"


# ── Discord payload builders ───────────────────────────────────────────────────

class TestBuildPayload:
    def test_embed_structure(self):
        """build_payload() returns correct Discord embed structure."""
        top_lists = json.loads(SAMPLE_LISTS.read_text())
        payload   = build_payload(top_lists)

        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        embed = payload["embeds"][0]
        assert "title"   in embed
        assert "color"   in embed
        assert "fields"  in embed
        assert len(embed["fields"]) >= 3  # top_buys, mid_caps, small_caps

    def test_field_values_within_discord_limit(self):
        """No field value exceeds Discord's 1024-char hard limit."""
        top_lists = json.loads(SAMPLE_LISTS.read_text())
        payload   = build_payload(top_lists)
        for field in payload["embeds"][0]["fields"]:
            assert len(field["value"]) <= 1024, (
                f"Field '{field['name']}' value too long: {len(field['value'])} chars"
            )

    def test_color_green_for_high_avg_score(self):
        """High-scoring universe → green color."""
        top_lists = json.loads(SAMPLE_LISTS.read_text())
        for entry in top_lists["top_buys"]:
            entry["final_score"] = 0.90
        payload = build_payload(top_lists)
        assert payload["embeds"][0]["color"] == 0x00B37D

    def test_color_red_for_low_avg_score(self):
        """Low-scoring universe → red color."""
        top_lists = json.loads(SAMPLE_LISTS.read_text())
        for entry in top_lists["top_buys"]:
            entry["final_score"] = 0.20
        payload = build_payload(top_lists)
        assert payload["embeds"][0]["color"] == 0xE53935

    def test_alert_payload_has_required_fields(self):
        """build_alert_payload() produces a valid embed."""
        payload = build_alert_payload("File not found: logs/top_lists.json")
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert "DATA UNAVAILABLE" in embed["title"]
        assert "description" in embed

    def test_format_ticker_block_length(self):
        """Single ticker block is under 300 chars."""
        entry = {
            "ticker": "AAPL",
            "final_score": 0.784,
            "badge": "TACTICAL BUY",
            "factors": {"edgar": 0.82, "insider": 0.80, "congress": 0.52,
                        "news": 0.60, "macro": 0.65},
            "ceo_buy": True,
        }
        block = _format_ticker_block(entry, rank=1)
        assert len(block) < 300, f"Ticker block too long: {len(block)} chars"


# ── discord_main dry-run ───────────────────────────────────────────────────────

class TestDiscordMainDryRun:
    def test_dry_run_no_network(self, capsys, tmp_path):
        """--dry-run prints JSON payload without hitting Discord."""
        import shutil
        shutil.copy(SAMPLE_LISTS, tmp_path / "top_lists.json")

        exit_code = discord_main([
            "--input",   str(tmp_path / "top_lists.json"),
            "--log-dir", str(tmp_path),
            "--dry-run",
        ])
        captured = capsys.readouterr()
        output   = captured.out.strip()

        assert exit_code == 0
        parsed = json.loads(output)   # must be valid JSON
        assert "embeds" in parsed

    def test_dry_run_missing_file_sends_alert(self, capsys, tmp_path):
        """--dry-run with missing top_lists.json prints alert embed."""
        exit_code = discord_main([
            "--input",   str(tmp_path / "nonexistent.json"),
            "--log-dir", str(tmp_path),
            "--dry-run",
        ])
        captured = capsys.readouterr()
        output   = captured.out.strip()

        assert exit_code == 0
        parsed = json.loads(output)
        assert "DATA UNAVAILABLE" in parsed["embeds"][0]["title"]
