# Path: tests/test_single_ticker_scoring.py
"""On-demand single-ticker mode for generate_top_lists.generate().

A 1-element universe must never flow through cross-sectional normalization
(n == 1 collapses every non-zero factor to neutral 0.5). Single-ticker mode
scores the raw factor values absolutely, keeps the per-ticker None-weight
redistribution, and silences the universe-level circuit breaker — without
mutating any daily-path behavior when the flag is absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from backend.market_intel.generate_top_lists import generate, PipelineIntegrityError


# ── Helpers (mirrors tests/test_pipeline_integrity.py fixtures) ──────────────

def _raw_row(
    ticker: str = "TSLA",
    edgar: float = 0.70,
    insider: float = 0.80,
    congress: float = 0.60,
    news: float = 0.65,
    momentum: float = 0.55,
    cap_tier: str = "large",
    market_cap: float = 8e11,
    sector: str = "Consumer Discretionary",
    quality_piotroski_raw: int | None = 6,
) -> Dict[str, Any]:
    return {
        "ticker":                    ticker,
        "sector":                    sector,
        "cap_tier":                  cap_tier,
        "market_cap":                market_cap,
        "insider_conviction_score":  edgar,
        "insider_breadth_score":     insider,
        "congress_score":            congress,
        "news_sentiment_score":      news,
        "news_buzz_score":           round(news * 0.85, 4),
        "momentum_long_score":       momentum,
        "volume_attention_score":    round(momentum * 0.90, 4),
        "analyst_consensus_score":   round(news * 0.50, 4),
        "analyst_revision_score":    round(momentum * 0.40, 4),
        "price_target_upside_score": round(momentum * 0.30, 4),
        "quality_piotroski_score":   round(edgar * 0.60, 4),
        "quality_piotroski_raw":     quality_piotroski_raw,
        "transcript_tone_score":     round(news * 0.20, 4),
        "ceo_buy":                   False,
        "form4_count":               3,
        "quiver_evidence":           {},
        "news_source":               "fmp",
        "insider_usd":               50_000.0,
        "momentum_spy_relative":     0.02,
        "volume_spike":              1.5,
        "_edgar_ok":                 True,
        "_scoring_error":            False,
    }


def _status_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"results": rows, "run_id": "test"}


def _generate(rows, tmp_path, vix=None, single_ticker=None):
    status = _status_payload(rows)
    with patch("backend.market_intel.generate_top_lists._read_vix", return_value=vix):
        return generate(status, run_id="single-test", log_dir=tmp_path,
                        single_ticker=single_ticker)


# ── Absolute scoring (the core bypass) ───────────────────────────────────────

class TestAbsoluteFactors:
    def test_raw_factor_values_preserved(self, tmp_path):
        """n=1 absolute mode must keep 0.55 at 0.55 — not collapse to 0.5."""
        out = _generate([_raw_row(momentum=0.55)], tmp_path, single_ticker="TSLA")
        entry = out["top_buys"][0]
        assert entry["factors"]["momentum_long"] == pytest.approx(0.55)
        assert entry["factors"]["insider_breadth"] == pytest.approx(0.80)

    def test_daily_path_at_n1_collapses_to_neutral(self, tmp_path):
        """Contrast pin: WITHOUT the flag, n=1 normalizes to 0.5 — the exact
        behavior single-ticker mode exists to bypass."""
        out = _generate([_raw_row(momentum=0.55)], tmp_path)
        entry = out["top_buys"][0]
        assert entry["factors"]["momentum_long"] == pytest.approx(0.5)

    def test_values_clipped_to_unit_interval(self, tmp_path):
        row = _raw_row()
        row["momentum_long_score"] = 1.7   # defensive: raw feeds assumed [0,1]
        out = _generate([row], tmp_path, single_ticker="TSLA")
        assert out["top_buys"][0]["factors"]["momentum_long"] == 1.0

    def test_none_factor_redistributes_weight_not_bearish(self, tmp_path):
        """SIGNED contract: None (API failure) must score HIGHER than a true
        0.0 signal because its weight leaves the denominator entirely."""
        row_none = _raw_row()
        row_none["analyst_consensus_score"] = None
        row_zero = _raw_row()
        row_zero["analyst_consensus_score"] = 0.0

        score_none = _generate([row_none], tmp_path, single_ticker="TSLA")["top_buys"][0]["final_score"]
        score_zero = _generate([row_zero], tmp_path, single_ticker="TSLA")["top_buys"][0]["final_score"]
        assert score_none > score_zero

    def test_none_factor_displays_as_zero(self, tmp_path):
        row = _raw_row()
        row["analyst_consensus_score"] = None
        out = _generate([row], tmp_path, single_ticker="TSLA")
        assert out["top_buys"][0]["factors"]["analyst_consensus"] == 0.0


# ── Mode mechanics ────────────────────────────────────────────────────────────

class TestSingleTickerMode:
    def test_output_tagged(self, tmp_path):
        out = _generate([_raw_row()], tmp_path, single_ticker="TSLA")
        assert out["single_ticker"] is True
        assert out["scoring_mode"] == "absolute"

    def test_daily_output_not_tagged(self, tmp_path):
        out = _generate([_raw_row(f"T{i}") for i in range(5)], tmp_path)
        assert "single_ticker" not in out
        assert "scoring_mode" not in out

    def test_filters_to_requested_ticker(self, tmp_path):
        rows = [_raw_row("TSLA"), _raw_row("MSFT"), _raw_row("NVDA")]
        out = _generate(rows, tmp_path, single_ticker="MSFT")
        assert out["ticker_count"] == 1
        assert out["top_buys"][0]["ticker"] == "MSFT"

    def test_missing_ticker_raises(self, tmp_path):
        with pytest.raises(PipelineIntegrityError):
            _generate([_raw_row("TSLA")], tmp_path, single_ticker="ZZZZ")

    def test_circuit_breaker_silenced_but_metadata_attached(self, tmp_path):
        """A thin name (>10 dead factors) must still score — flagged, not fatal."""
        row = _raw_row(edgar=0.0, insider=0.0, congress=0.0, news=0.0,
                       momentum=0.3)
        out = _generate([row], tmp_path, single_ticker="TSLA")
        meta = out["top_buys"][0]["validation_metadata"]
        assert meta["is_complete"] is False
        assert "insider_conviction" in meta["missing_sources"]

    def test_mvo_skipped(self, tmp_path):
        out = _generate([_raw_row()], tmp_path, single_ticker="TSLA")
        entry = out["top_buys"][0]
        assert entry["portfolio_weight"] == 0.0
        assert entry["portfolio_weight_method"] == "n/a"

    def test_congress_dead_streak_state_untouched(self, tmp_path, monkeypatch):
        """Single-ticker runs must never write/delete the daily congress
        dead-streak tracker (.cache/congress_dead_since.txt)."""
        monkeypatch.chdir(tmp_path)
        row = _raw_row(congress=0.0)
        _generate([row], tmp_path, single_ticker="TSLA")
        assert not (tmp_path / ".cache" / "congress_dead_since.txt").exists()


# ── Safety overlays still apply ───────────────────────────────────────────────

class TestSafetyOverlays:
    def test_vix_overlay_applied(self, tmp_path):
        score_calm = _generate([_raw_row()], tmp_path, vix=None,
                               single_ticker="TSLA")["top_buys"][0]["final_score"]
        score_panic = _generate([_raw_row()], tmp_path, vix=35.0,
                                single_ticker="TSLA")["top_buys"][0]["final_score"]
        assert score_panic / score_calm == pytest.approx(0.5, abs=0.02)

    def test_kill_switch_set_at_panic_vix(self, tmp_path):
        out = _generate([_raw_row()], tmp_path, vix=35.0, single_ticker="TSLA")
        assert out["kill_switch"] is True
        assert out["vix"] == 35.0

    def test_badge_consistent_with_score(self, tmp_path):
        from src.delivery.audit_payload import _expected_badge
        out = _generate([_raw_row()], tmp_path, single_ticker="TSLA")
        entry = out["top_buys"][0]
        assert entry["badge"] == _expected_badge(entry["final_score"])
