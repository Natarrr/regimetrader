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
