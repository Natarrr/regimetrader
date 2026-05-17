"""tests/test_discord_formatter.py
Unit tests for Discord formatter — factor line uses "momentum" key.
"""
from __future__ import annotations
import pytest


class TestFormatFactorLine:
    def test_reads_momentum_key(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors = {
            "edgar": 0.80, "insider": 0.70, "congress": 0.60,
            "news": 0.55, "momentum": 0.65,
        }
        line = _format_factor_line(factors)
        assert "0.65" in line, "momentum value 0.65 not in output"

    def test_momentum_key_present_gives_non_default(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors_with    = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5, "momentum": 0.99}
        factors_without = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5}
        line_with    = _format_factor_line(factors_with)
        line_without = _format_factor_line(factors_without)
        assert "0.99" in line_with,    "momentum 0.99 not rendered"
        assert "0.99" not in line_without

    def test_macro_key_ignored(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors = {
            "edgar": 0.5, "insider": 0.5, "congress": 0.5,
            "news": 0.5, "macro": 0.99,
        }
        line = _format_factor_line(factors)
        assert "0.99" not in line, "'macro' key must not affect output"

    def test_output_contains_all_five_emojis(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5, "momentum": 0.5}
        line = _format_factor_line(factors)
        for emoji in ("📋", "🏦", "🏛️", "📰", "📈"):
            assert emoji in line, f"{emoji} missing from factor line"

    def test_missing_factor_defaults_to_zero_not_neutral(self):
        from scripts.send_toplists_discord import _format_factor_line
        # Missing factors must show 0.00, not 0.50 — dead feed is penalised
        line = _format_factor_line({})
        assert "0.00" in line, "missing factor must default to 0.00, not 0.50"
        assert "0.50" not in line, "missing factor must not default to neutral 0.50"


class TestBuildPayloadWeights:
    def _make_top_lists(self, weights):
        return {
            "generated_at":  "2026-05-17T12:00:00+00:00",
            "source_run_id": "test-run",
            "ticker_count":  10,
            "weights":       weights,
            "kill_switch":   False,
            "top_buys":      [{"ticker": "AAPL", "final_score": 0.70, "badge": "TACTICAL BUY",
                               "factors": {"edgar": 0.7, "insider": 0.6, "congress": 0.5,
                                           "news": 0.6, "momentum": 0.5}, "ceo_buy": False}],
            "mid_caps":      [],
            "small_caps":    [],
        }

    def test_nominal_weights_no_redistribution_label(self):
        from scripts.send_toplists_discord import build_payload
        weights = {"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12}
        payload = build_payload(self._make_top_lists(weights))
        desc = payload["embeds"][0]["description"]
        assert "feed down" not in desc, "nominal weights must not trigger redistribution warning"
        assert "redistributed" not in desc

    def test_redistributed_weights_shows_warning(self):
        from scripts.send_toplists_discord import build_payload
        # Simulate insider feed dead — weight redistributed to other factors
        weights = {"edgar": 0.359, "congress": 0.282, "news": 0.192, "momentum": 0.154}
        payload = build_payload(self._make_top_lists(weights))
        desc = payload["embeds"][0]["description"]
        assert "feed down" in desc or "redistributed" in desc, (
            "redistributed weights must show a warning in description"
        )
