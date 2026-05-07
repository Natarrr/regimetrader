"""tests/test_check_metrics.py — threshold gate truth-table.

Markowitz frame: the canary gate is a binary acceptance test on two axes —
data coverage (analogous to portfolio completeness) and error count (analogous
to realised losses). Both must clear simultaneously, mirroring the joint
constraint structure of mean-variance optimisation.

Coverage gate:  $\\text{coverage} = \\frac{\\text{edgar\\_count}}{\\text{ticker\\_count}} \\geq c_{\\min}$
Error gate:     $\\text{error\\_count} \\leq e_{\\max}$
"""
from __future__ import annotations

from monitoring.evaluate import evaluate


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
