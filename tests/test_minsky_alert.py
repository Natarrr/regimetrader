"""tests/test_minsky_alert.py — Minsky insider-stress detector contract.

Minsky frame: the breadth axis measures the fraction of the universe showing
elevated insider conviction. After the 7-factor migration the pipeline writes
`insider_breadth_score` (orthogonal breadth), not the legacy `insider_score`.
These tests pin the breadth signal to the key the pipeline actually emits so a
silent-zero regression cannot recur.
"""
from __future__ import annotations

from monitoring import minsky_alert as ma


def _row(ticker: str, **factors) -> dict:
    """A pipeline result row with 7-factor keys; overrides via kwargs."""
    base = {
        "ticker":                   ticker,
        "ceo_buy":                  False,
        "form4_count":              0,
        "insider_breadth_score":    0.0,
        "insider_conviction_score": 0.0,
    }
    base.update(factors)
    return base


def test_breadth_reads_insider_breadth_score() -> None:
    """Minsky: every ticker with insider_breadth_score ≥ 0.70 counts toward
    breadth. The 7-factor pipeline emits this key — not legacy insider_score."""
    results = [_row(f"T{i}", insider_breadth_score=0.80) for i in range(10)]
    stress = ma._compute_stress(results)
    assert stress.breadth_ratio == 1.0


def test_breadth_elevated_triggers_axis() -> None:
    """Minsky: ≥50% breadth is one of the three stress preconditions. With all
    three axes lit, the level must escalate to CRITICAL."""
    results = [
        _row(f"T{i}", ceo_buy=True, form4_count=6, insider_breadth_score=0.80)
        for i in range(10)
    ]
    stress = ma._compute_stress(results)
    assert stress.breadth_ratio >= 0.50
    assert stress.conditions_met == 3
    assert stress.level == "CRITICAL"


def test_breadth_below_threshold_does_not_count() -> None:
    """Minsky: insider_breadth_score below 0.70 must not register as breadth —
    the axis is about elevated conviction, not mere presence."""
    results = [_row(f"T{i}", insider_breadth_score=0.40) for i in range(10)]
    stress = ma._compute_stress(results)
    assert stress.breadth_ratio == 0.0
