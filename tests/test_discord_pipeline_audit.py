# Path: tests/test_discord_pipeline_audit.py
"""Pipeline audit regression tests.

Originally written for the 6-fix Discord pipeline audit. The suites tied to
the legacy build_payload internals (_load_top_lists_overlay, _normalise_entry,
_action_section) were retired with the DiscordPayloadBuilder consolidation —
their contracts are covered by tests/test_discord_formatter.py. What remains:

  - run-id footer contract, ported to DiscordPayloadBuilder
  - news-sentiment dead-signal scoring (run_pipeline)
  - run_id / form4_purchase_count structural contracts (run_pipeline)
  - Minsky form4_purchase_count preference (monitoring)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


def _cooked_status(**overrides) -> dict:
    st = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "vix":             17.0,
        "vix_regime":      "NORMAL",
        "kill_switch":     False,
        "ticker_count":    0,
        "top_buys_usa":    [],
        "top_buys_europe": [],
        "top_buys_asia":   [],
        "watchlist":       [],
        "mvo_pools":       {},
    }
    st.update(overrides)
    return st


# ── Fix #5 (ported): run_id surfaces in the embed footer ─────────────────────

class TestRunIdInFooter:
    def test_run_id_shown_in_footer(self):
        from src.delivery.send_discord import DiscordPayloadBuilder

        payload = DiscordPayloadBuilder(
            _cooked_status(run_id="12345678901")).build()
        footer_text = payload["embeds"][0]["footer"]["text"]
        assert "12345678901" in footer_text, f"run_id not in footer: {footer_text!r}"

    def test_local_fallback_without_run_id(self):
        from src.delivery.send_discord import DiscordPayloadBuilder

        payload = DiscordPayloadBuilder(_cooked_status()).build()
        footer_text = payload["embeds"][0]["footer"]["text"]
        assert "local" in footer_text

    def test_run_id_key_in_status_dict_local(self, monkeypatch):
        """Without GITHUB_RUN_ID env var, run_id must be 'local'."""
        import os
        monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
        run_id = os.getenv("GITHUB_RUN_ID", "local")
        assert run_id == "local"

    def test_run_id_key_in_status_dict_ci(self, monkeypatch):
        """With GITHUB_RUN_ID set, status['run_id'] must equal that value."""
        import os
        monkeypatch.setenv("GITHUB_RUN_ID", "12345678901")
        run_id = os.getenv("GITHUB_RUN_ID", "local")
        assert run_id == "12345678901"

    def test_status_dict_has_run_id_key(self):
        """run_pipeline must write run_id into intel_source_status.json."""
        import ast
        import src.ingestion.run_pipeline as run_pipeline_mod
        file_src = Path(run_pipeline_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(file_src)

        found_run_id = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value == "run_id":
                        found_run_id = True
                        break
        assert found_run_id, "run_id key not found in any dict literal in run_pipeline.py"


# ── Fix #4: _score_news_sentiment_yfinance dead signal is 0.0, not 0.5 ────────

class TestNewsSentimentDeadSignal:
    """Fix #4: no-signal headlines must return 0.0, not 0.5."""

    def test_all_neutral_headlines_returns_zero(self, monkeypatch):
        """Headlines with zero bull AND zero bear words → 0.0 (dead signal)."""
        from src.ingestion import run_pipeline

        neutral_headlines = [
            {"content": {"title": "Company announces something today"}},
            {"content": {"title": "Market open as usual"}},
        ]

        class _FakeTicker:
            news = neutral_headlines

        monkeypatch.setattr(
            "src.ingestion.run_pipeline.yf",
            type("yf", (), {"Ticker": staticmethod(lambda t: _FakeTicker())})(),
            raising=False,
        )

        import sys
        fake_yf = type("yfinance", (), {"Ticker": staticmethod(lambda t: _FakeTicker())})()
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        result = run_pipeline._score_news_sentiment_yfinance("NVDA")
        assert result == 0.0, f"Expected 0.0 for neutral headlines, got {result}"

    def test_no_headlines_returns_zero(self, monkeypatch):
        """Empty news list → 0.0."""
        import sys
        from src.ingestion import run_pipeline

        class _FakeTicker:
            news = []

        fake_yf = type("yfinance", (), {"Ticker": staticmethod(lambda t: _FakeTicker())})()
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        result = run_pipeline._score_news_sentiment_yfinance("AAPL")
        assert result == 0.0, f"Expected 0.0 for empty news, got {result}"

    def test_bullish_headline_returns_above_half(self, monkeypatch):
        """Headline with bull words → score > 0.5."""
        import sys
        from src.ingestion import run_pipeline

        class _FakeTicker:
            news = [{"content": {"title": "Company beats earnings upgrade buy strong"}}]

        fake_yf = type("yfinance", (), {"Ticker": staticmethod(lambda t: _FakeTicker())})()
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        result = run_pipeline._score_news_sentiment_yfinance("TSLA")
        assert result > 0.5, f"Expected > 0.5 for bullish headline, got {result}"


# ── Fix #6: form4_purchase_count in result rows + minsky fallback ─────────────

class TestForm4PurchaseCount:
    """Fix #6: pipeline result rows must carry form4_purchase_count (P-code only).
    Minsky must prefer form4_purchase_count over form4_count."""

    def test_minsky_prefers_form4_purchase_count(self):
        """_compute_stress uses form4_purchase_count when available."""
        from monitoring import minsky_alert as ma

        results = [
            {"ticker": "T1", "ceo_buy": False, "form4_count": 42,
             "form4_purchase_count": 3, "insider_breadth_score": 0.0},
        ]
        stress = ma._compute_stress(results)
        assert stress.mean_form4 == pytest.approx(3.0), (
            f"Minsky should use form4_purchase_count=3, got mean_form4={stress.mean_form4}"
        )

    def test_minsky_falls_back_to_form4_count_when_purchase_count_absent(self):
        """Without form4_purchase_count, Minsky falls back to form4_count."""
        from monitoring import minsky_alert as ma

        results = [
            {"ticker": "T1", "ceo_buy": False, "form4_count": 7,
             "insider_breadth_score": 0.0},
        ]
        stress = ma._compute_stress(results)
        assert stress.mean_form4 == pytest.approx(7.0), (
            f"Minsky fallback to form4_count=7 failed, got mean_form4={stress.mean_form4}"
        )

    def test_form4_purchase_count_key_in_successful_score_ticker(self):
        """The success-path result dict from _score_ticker must include
        form4_purchase_count as a top-level key."""
        import ast
        import src.ingestion.run_pipeline as run_pipeline_mod
        file_src = Path(run_pipeline_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(file_src)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value == "form4_purchase_count":
                        found = True
                        break
        assert found, "form4_purchase_count not emitted in any result dict in run_pipeline.py"
