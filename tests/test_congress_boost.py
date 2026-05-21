"""tests/test_congress_boost.py
Congress conviction boost integration tests.

Verifies _apply_congress_boost() and its interaction with generate():
  - Boost applied correctly when anomaly report contains CONGRESS_CLUSTER
  - Boost is 0.0 when report absent (dead feed)
  - Shadow score preserved before boost
  - Promoted tickers appear in top_buys after boost
  - congress_boost field always present on every entry
  - Boost coefficient: final_score × (1 + 0.10 × conviction_score)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _entry(ticker: str, final_score: float, cap_tier: str = "large") -> Dict[str, Any]:
    return {
        "ticker":     ticker,
        "final_score": final_score,
        "cap_tier":   cap_tier,
        "sector":     "Information Technology",
        "market_cap": 1e12,
    }


def _write_cluster_report(path: Path, ticker: str, conviction: float) -> None:
    records = [
        {
            "run_id":          "test",
            "ticker":          ticker,
            "timestamp":       "2026-01-01T00:00:00+00:00",
            "flag":            "CONGRESS_CLUSTER",
            "value":           3.0,
            "threshold":       3.0,
            "action":          "boost_candidate",
            "source":          "quiver",
            "conviction_score": conviction,
        }
    ]
    path.write_text(json.dumps(records), encoding="utf-8")


# ── TestCongressBoost ─────────────────────────────────────────────────────────

class TestCongressBoost:

    def _boost(self, entries, tmp_path):
        from backend.market_intel.generate_top_lists import _apply_congress_boost
        return _apply_congress_boost(entries, tmp_path)

    def test_boost_applied_when_cluster_present(self, tmp_path):
        _write_cluster_report(tmp_path / "anomaly_report_latest.json", "NVDA", conviction=1.0)
        entries = [_entry("NVDA", final_score=0.70)]
        self._boost(entries, tmp_path)
        expected = round(0.70 * 1.10, 4)
        assert abs(entries[0]["final_score"] - expected) < 1e-6

    def test_boost_coefficient_scales_with_conviction(self, tmp_path):
        _write_cluster_report(tmp_path / "anomaly_report_latest.json", "AAPL", conviction=0.60)
        entries = [_entry("AAPL", final_score=0.80)]
        self._boost(entries, tmp_path)
        expected = round(0.80 * (1.0 + 0.10 * 0.60), 4)
        assert abs(entries[0]["final_score"] - expected) < 1e-6

    def test_congress_boost_field_present_when_boosted(self, tmp_path):
        _write_cluster_report(tmp_path / "anomaly_report_latest.json", "META", conviction=1.0)
        entries = [_entry("META", final_score=0.75)]
        self._boost(entries, tmp_path)
        assert "congress_boost" in entries[0]
        assert abs(entries[0]["congress_boost"] - 0.10) < 1e-6

    def test_congress_boost_field_zero_when_no_cluster(self, tmp_path):
        # No report file → boost must be 0.0 on every entry
        entries = [_entry("MSFT", final_score=0.65)]
        self._boost(entries, tmp_path)
        assert entries[0]["congress_boost"] == 0.0
        assert abs(entries[0]["final_score"] - 0.65) < 1e-6

    def test_no_crash_when_report_absent(self, tmp_path):
        entries = [_entry("AMZN", final_score=0.60)]
        result = self._boost(entries, tmp_path)
        assert len(result) == 1   # returned without raising

    def test_non_cluster_flags_do_not_boost(self, tmp_path):
        records = [{
            "run_id": "test", "ticker": "TSLA", "timestamp": "2026-01-01T00:00:00+00:00",
            "flag": "VOLUME_SPIKE", "value": 50.0, "threshold": 10.0,
            "action": "flag_only", "source": "quiver",
        }]
        (tmp_path / "anomaly_report_latest.json").write_text(
            json.dumps(records), encoding="utf-8"
        )
        entries = [_entry("TSLA", final_score=0.55)]
        self._boost(entries, tmp_path)
        assert entries[0]["congress_boost"] == 0.0
        assert abs(entries[0]["final_score"] - 0.55) < 1e-6

    def test_boost_does_not_affect_non_cluster_tickers(self, tmp_path):
        _write_cluster_report(tmp_path / "anomaly_report_latest.json", "JPM", conviction=1.0)
        entries = [
            _entry("JPM", final_score=0.70),
            _entry("GS",  final_score=0.68),
        ]
        self._boost(entries, tmp_path)
        assert entries[0]["final_score"] > 0.70   # JPM boosted
        assert abs(entries[1]["final_score"] - 0.68) < 1e-6  # GS untouched
        assert entries[1]["congress_boost"] == 0.0

    def test_boost_can_promote_ticker_above_unboosted_peer(self, tmp_path):
        _write_cluster_report(tmp_path / "anomaly_report_latest.json", "AMD", conviction=1.0)
        entries = [
            _entry("INTC", final_score=0.74),
            _entry("AMD",  final_score=0.70),
        ]
        self._boost(entries, tmp_path)
        amd_score  = next(e["final_score"] for e in entries if e["ticker"] == "AMD")
        intc_score = next(e["final_score"] for e in entries if e["ticker"] == "INTC")
        assert amd_score > intc_score, "boosted AMD should overtake INTC"


# ── TestShadowScore ───────────────────────────────────────────────────────────

class TestShadowScore:
    """Verify that generate() produces shadow_top_buys capturing pre-boost ranking."""

    def _minimal_status(self, tickers: List[str]) -> Dict[str, Any]:
        """Build a minimal intel_source_status dict for generate()."""
        results = []
        for i, ticker in enumerate(tickers):
            score = round(0.80 - i * 0.05, 4)
            results.append({
                "ticker":         ticker,
                "sector":         "Information Technology",
                "cap_tier":       "large",
                "market_cap":     1e12,
                "edgar_score":    score,
                "insider_score":  score,
                "congress_score": score,
                "news_score":     score,
                "momentum_score": score,
                "ceo_buy":        False,
                "form4_count":    5,
                "quiver_evidence": {
                    "congress": {
                        "purchases": 3, "sales": 0, "net": 3,
                        "recency_days": 5,
                        "representatives": ["Rep.0", "Rep.1", "Rep.2"],
                    },
                    "source": "quiver",
                    "insider_source": "quiver",
                },
                "news_source":           "finnhub",
                "insider_usd":           50_000.0,
                "momentum_spy_relative": 0.02,
                "volume_spike":          1.2,
                "insider_source":        "quiver",
                "computed_at":           "2026-05-21T12:00:00+00:00",
            })
        return {
            "_edgar_meta": {"last_run": "2026-05-21T12:00:00+00:00"},
            "source_meta": {
                "quiver": {"last_updated": "2026-05-21T12:00:00+00:00"},
                "finnhub": {"last_updated": "2026-05-21T12:00:00+00:00"},
                "edgar":  {"last_updated": "2026-05-21T12:00:00+00:00"},
                "none":   {"last_updated": "2026-05-21T12:00:00+00:00"},
            },
            "weights": {
                "edgar": 0.28, "insider": 0.23, "congress": 0.22,
                "news": 0.15, "momentum": 0.12,
            },
            "results": results,
            "computed_at": "2026-05-21T12:00:00+00:00",
        }

    def test_shadow_top_buys_present_in_output(self, tmp_path):
        from backend.market_intel.generate_top_lists import generate

        tickers = [f"T{i:02d}" for i in range(8)]
        status = self._minimal_status(tickers)

        with patch("backend.market_intel.generate_top_lists._read_vix", return_value=None):
            result = generate(status, run_id="shadow-test", log_dir=tmp_path)

        assert "shadow_top_buys" in result

    def test_shadow_top_buys_has_five_entries(self, tmp_path):
        from backend.market_intel.generate_top_lists import generate

        tickers = [f"T{i:02d}" for i in range(8)]
        status = self._minimal_status(tickers)

        with patch("backend.market_intel.generate_top_lists._read_vix", return_value=None):
            result = generate(status, run_id="shadow-count", log_dir=tmp_path)

        assert len(result["shadow_top_buys"]) == 5

    def test_shadow_top_buys_unaffected_by_boost(self, tmp_path):
        """shadow_top_buys must reflect pre-boost scores (congress_boost=0 on all entries)."""
        from backend.market_intel.generate_top_lists import generate

        tickers = [f"T{i:02d}" for i in range(8)]
        status = self._minimal_status(tickers)

        with patch("backend.market_intel.generate_top_lists._read_vix", return_value=None):
            result = generate(status, run_id="shadow-unaffected", log_dir=tmp_path)

        # Shadow entries should not carry congress_boost (captured before boost applied)
        for entry in result["shadow_top_buys"]:
            assert "congress_boost" not in entry

    def test_congress_boost_field_on_top_buys(self, tmp_path):
        """Every entry in top_buys must carry congress_boost after generate()."""
        from backend.market_intel.generate_top_lists import generate

        tickers = [f"T{i:02d}" for i in range(8)]
        status = self._minimal_status(tickers)

        with patch("backend.market_intel.generate_top_lists._read_vix", return_value=None):
            result = generate(status, run_id="boost-field", log_dir=tmp_path)

        for entry in result["top_buys"]:
            assert "congress_boost" in entry
