"""tests/test_check_metrics.py — threshold gate truth-table.

Markowitz frame: the canary gate is a binary acceptance test on two axes —
data coverage (analogous to portfolio completeness) and error count (analogous
to realised losses). Both must clear simultaneously, mirroring the joint
constraint structure of mean-variance optimisation.

Coverage gate:  $\\text{coverage} = \\frac{\\text{edgar\\_count}}{\\text{ticker\\_count}} \\geq c_{\\min}$
Error gate:     $\\text{error\\_count} \\leq e_{\\max}$
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from monitoring.evaluate import evaluate
from monitoring.check_metrics import check_score_distribution


def _metrics(ticker: int, edgar: int, errors: int = 0) -> dict:
    return {
        "last_run":             "2026-05-06T00:00:00+00:00",
        "run_duration_seconds": 1.0,
        "ticker_count":         ticker,
        "edgar_count":          edgar,
        "fmp_count":            max(ticker - edgar, 0),
        "error_count":          errors,
    }


# ── happy path ───────────────────────────────────────────────────────────────

def test_evaluate_ok_when_thresholds_met() -> None:
    """Markowitz: 70% coverage with zero errors clears the default 60%/0
    contract — the gate must remain quiet under nominal conditions."""
    ok, reasons = evaluate(_metrics(10, 7, 0))
    assert ok is True
    assert reasons == []


def test_evaluate_ok_at_exact_coverage_boundary() -> None:
    """Engle: gates with strict-inequality thresholds must be tested *at*
    the boundary — equal-to-min must pass (not fail)."""
    ok, reasons = evaluate(_metrics(10, 6, 0), min_coverage=0.6)
    assert ok is True
    assert reasons == []


# ── failure modes ────────────────────────────────────────────────────────────

def test_evaluate_fails_low_coverage() -> None:
    """Markowitz: 30% coverage on a 10-asset universe leaves 70% of the
    target portfolio unobserved — gate must fail with a quantitative reason."""
    ok, reasons = evaluate(_metrics(10, 3, 0))
    assert ok is False
    assert any("EDGAR coverage" in r for r in reasons)


def test_evaluate_fails_with_any_error() -> None:
    """Kahneman: prospect theory motivates strict downside aversion — the
    default `max_errors=0` rejects even a single fault."""
    ok, reasons = evaluate(_metrics(10, 8, errors=1))
    assert ok is False
    assert any("error_count" in r for r in reasons)


def test_evaluate_fails_when_no_tickers_processed() -> None:
    """Fama: zero tickers processed is a degenerate run — coverage is
    undefined, so the gate must fail explicitly rather than divide-by-zero."""
    ok, reasons = evaluate(_metrics(0, 0, 0))
    assert ok is False
    assert any("ticker_count" in r for r in reasons)


def test_evaluate_reports_both_failures_simultaneously() -> None:
    """Markowitz: the joint constraint (coverage ∧ errors) means a run that
    breaks both must surface both reasons — operators need the full diagnosis."""
    ok, reasons = evaluate(_metrics(10, 2, errors=3))
    assert ok is False
    assert len(reasons) == 2
    assert any("EDGAR coverage" in r for r in reasons)
    assert any("error_count"   in r for r in reasons)


# ── threshold tunability ─────────────────────────────────────────────────────

def test_evaluate_respects_relaxed_coverage_threshold() -> None:
    """Engle: regime-dependent thresholds matter — a 40% coverage that
    fails the default 60% gate must pass when the operator dials it to 30%."""
    metrics = _metrics(10, 4, 0)
    assert evaluate(metrics, min_coverage=0.6)[0] is False
    assert evaluate(metrics, min_coverage=0.3)[0] is True


def test_evaluate_respects_relaxed_max_errors() -> None:
    """Kahneman: tolerance for losses is policy, not law — `max_errors=2`
    must accept exactly two errors and reject three."""
    assert evaluate(_metrics(10, 8, errors=2), max_errors=2)[0] is True
    assert evaluate(_metrics(10, 8, errors=3), max_errors=2)[0] is False


# ── check_score_distribution (PATCH 11) ──────────────────────────────────────

def _write_top_lists(tmp: Path, data: dict) -> Path:
    p = tmp / "top_lists.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return tmp


class TestCheckScoreDistribution:
    """Canary gate: score distribution must be non-degenerate (PATCH 11).

    A dead-feed scenario (all factors return 0.0) causes the cross-sectional
    normaliser to set all scores to 0.0, and weight redistribution raises all
    final_scores to the weight-floor artefact (~0.18). The check must detect
    this and return False.
    """

    def test_healthy_distribution_passes(self, tmp_path):
        data = {"top_buys": [
            {"final_score": 0.82}, {"final_score": 0.65}, {"final_score": 0.55},
            {"final_score": 0.42}, {"final_score": 0.30},
        ]}
        _write_top_lists(tmp_path, data)
        assert check_score_distribution(tmp_path) is True

    def test_degenerate_uniform_scores_fail(self, tmp_path):
        """Weight-floor artefact: all tickers at ~0.18 must trigger the gate."""
        data = {"top_buys": [
            {"final_score": 0.181}, {"final_score": 0.183}, {"final_score": 0.179},
            {"final_score": 0.182}, {"final_score": 0.180},
        ]}
        _write_top_lists(tmp_path, data)
        assert check_score_distribution(tmp_path) is False

    def test_low_max_score_fails_even_with_spread(self, tmp_path):
        """If no ticker exceeds min_max_score, the gate fails regardless of stdev."""
        data = {"top_buys": [
            {"final_score": 0.10}, {"final_score": 0.15}, {"final_score": 0.20},
            {"final_score": 0.25}, {"final_score": 0.30},
        ]}
        _write_top_lists(tmp_path, data)
        assert check_score_distribution(tmp_path) is False

    def test_missing_file_is_skipped_not_failed(self, tmp_path):
        """No top_lists.json → canary must not fail (file may not exist yet)."""
        assert check_score_distribution(tmp_path) is True

    def test_scores_across_multiple_buckets_aggregated(self, tmp_path):
        """Scores from top_buys_usa and top_buys_europe are combined for the check."""
        data = {
            "top_buys_usa": [{"final_score": 0.80}, {"final_score": 0.60}],
            "top_buys_europe": [{"final_score": 0.50}, {"final_score": 0.30}],
        }
        _write_top_lists(tmp_path, data)
        assert check_score_distribution(tmp_path) is True

    def test_fewer_than_min_entries_skips_check(self, tmp_path):
        """Only 2 unique scores — below min_entries=3 — must skip gracefully."""
        data = {"top_buys": [{"final_score": 0.80}, {"final_score": 0.20}]}
        _write_top_lists(tmp_path, data)
        assert check_score_distribution(tmp_path) is True

    def test_entries_missing_final_score_key_are_ignored(self, tmp_path):
        """Entries without a final_score key must be skipped, not crash."""
        data = {"top_buys": [
            {"ticker": "X"}, {"final_score": 0.75},
            {"final_score": 0.55}, {"final_score": 0.45},
        ]}
        _write_top_lists(tmp_path, data)
        assert check_score_distribution(tmp_path) is True

    def test_corrupted_json_is_skipped_not_failed(self, tmp_path):
        """Unreadable top_lists.json must not crash the canary."""
        (tmp_path / "top_lists.json").write_text("not-json", encoding="utf-8")
        assert check_score_distribution(tmp_path) is True
