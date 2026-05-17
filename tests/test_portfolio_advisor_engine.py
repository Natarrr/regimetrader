"""tests/test_portfolio_advisor_engine.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from regime_trader.ui.portfolio_advisor_engine import (
    compute_signal,
    compute_health_score,
    find_swap_candidate,
    PositionAdvice,
    _signal_age_days,
)


# ── Signal thresholds ─────────────────────────────────────────────────────────

class TestComputeSignal:
    def test_high_score_is_add(self):
        assert compute_signal(0.70, regime="Bull") == "ADD"

    def test_mid_score_is_hold(self):
        assert compute_signal(0.55, regime="Bull") == "HOLD"

    def test_low_score_is_reduce(self):
        assert compute_signal(0.38, regime="Bull") == "REDUCE"

    def test_very_low_score_is_exit(self):
        assert compute_signal(0.20, regime="Bull") == "EXIT"

    def test_kill_switch_regime_forces_exit(self):
        assert compute_signal(0.80, regime="Crash") == "EXIT"

    def test_boundary_065_is_add(self):
        assert compute_signal(0.65, regime="Neutral") == "ADD"

    def test_boundary_045_is_hold(self):
        assert compute_signal(0.45, regime="Neutral") == "HOLD"

    def test_boundary_030_is_reduce(self):
        assert compute_signal(0.30, regime="Neutral") == "REDUCE"


# ── Portfolio health score ────────────────────────────────────────────────────

class TestComputeHealthScore:
    def test_weighted_average_by_value(self):
        positions = [
            {"ticker": "AAPL", "final_score": 0.80, "market_value": 800.0},
            {"ticker": "COIN", "final_score": 0.20, "market_value": 200.0},
        ]
        score = compute_health_score(positions)
        # 0.80 * 0.8 + 0.20 * 0.2 = 0.64 + 0.04 = 0.68
        assert score == pytest.approx(0.68, abs=1e-4)

    def test_empty_returns_zero(self):
        assert compute_health_score([]) == 0.0

    def test_single_position(self):
        positions = [{"ticker": "X", "final_score": 0.75, "market_value": 1000.0}]
        assert compute_health_score(positions) == pytest.approx(0.75)


# ── Signal age ────────────────────────────────────────────────────────────────

class TestSignalAge:
    def test_recent_run_shows_small_age(self):
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        status = {"computed_at": recent}
        assert _signal_age_days(status) == pytest.approx(3, abs=1)

    def test_missing_computed_at_returns_none(self):
        assert _signal_age_days({}) is None


# ── Swap candidates ───────────────────────────────────────────────────────────

class TestFindSwapCandidate:
    _TOP_LISTS = {
        "top_buys": [
            {"ticker": "NVDA", "sector": "Information Technology", "final_score": 0.90, "badge": "HIGH BUY"},
            {"ticker": "AAPL", "sector": "Information Technology", "final_score": 0.85, "badge": "HIGH BUY"},
        ],
        "mid_caps": [
            {"ticker": "PANW", "sector": "Communication Services", "final_score": 0.80, "badge": "HIGH BUY"},
        ],
        "small_caps": [],
        "sector_picks": {},
    }

    def test_returns_top_unowned_in_same_sector(self):
        held = {"AAPL"}
        result = find_swap_candidate("MSFT", "Information Technology", held, self._TOP_LISTS)
        assert result is not None
        assert result["ticker"] == "NVDA"

    def test_no_swap_when_all_owned(self):
        held = {"NVDA", "AAPL"}
        result = find_swap_candidate("MSFT", "Information Technology", held, self._TOP_LISTS)
        assert result is None

    def test_no_swap_for_unknown_sector(self):
        result = find_swap_candidate("XYZ", "Utilities", set(), self._TOP_LISTS)
        assert result is None

    def test_does_not_suggest_the_reduce_ticker_itself(self):
        held = set()
        result = find_swap_candidate("NVDA", "Information Technology", held, self._TOP_LISTS)
        assert result is not None
        assert result["ticker"] != "NVDA"


# ── build_advice() weights and keys ───────────────────────────────────────────

class TestBuildAdviceWeightsAndKeys:
    """build_advice() must use 28/23/22/15/12 weights and 'momentum' key."""

    def _make_status(self) -> dict:
        return {
            "computed_at": "2026-05-17T10:00:00Z",
            "results": [{
                "ticker":         "AAPL",
                "sector":         "Information Technology",
                "cap_tier":       "large",
                "edgar_score":    1.0,
                "insider_score":  0.0,
                "congress_score": 0.0,
                "news_score":     0.0,
                "momentum_score": 0.0,
            }],
        }

    def test_final_score_uses_correct_weights(self):
        from regime_trader.ui.portfolio_advisor_engine import build_advice

        status = self._make_status()
        with tempfile.TemporaryDirectory() as td:
            status_path = Path(td) / "intel_source_status.json"
            status_path.write_text(json.dumps(status))
            top_lists_path = Path(td) / "top_lists.json"
            top_lists_path.write_text(json.dumps({}))

            with patch("regime_trader.ui.portfolio_advisor_engine._STATUS_PATH", status_path), \
                 patch("regime_trader.ui.portfolio_advisor_engine._TOP_LISTS_PATH", top_lists_path):
                result = build_advice(
                    [{"ticker": "AAPL", "net_qty": 10, "avg_cost": 150.0}],
                    regime="Bull",
                )

        assert len(result) == 1
        adv = result[0]
        # edgar_score=1.0, all others=0 → final_score should be 0.28 (not 0.30)
        assert adv.final_score == pytest.approx(0.28, abs=1e-4)

    def test_factors_dict_has_momentum_not_macro(self):
        from regime_trader.ui.portfolio_advisor_engine import build_advice

        status = self._make_status()
        with tempfile.TemporaryDirectory() as td:
            status_path = Path(td) / "intel_source_status.json"
            status_path.write_text(json.dumps(status))
            top_lists_path = Path(td) / "top_lists.json"
            top_lists_path.write_text(json.dumps({}))

            with patch("regime_trader.ui.portfolio_advisor_engine._STATUS_PATH", status_path), \
                 patch("regime_trader.ui.portfolio_advisor_engine._TOP_LISTS_PATH", top_lists_path):
                result = build_advice(
                    [{"ticker": "AAPL", "net_qty": 10, "avg_cost": 150.0}],
                    regime="Bull",
                )

        adv = result[0]
        assert "momentum" in adv.factors,     "'momentum' key missing from factors"
        assert "macro"    not in adv.factors, "'macro' key must not be in factors"
