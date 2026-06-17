"""Tests for src.monitoring.factor_orthogonality — permanent factor monitoring.

Guards the C1 fix: the orthogonality diagnostic must cover the FULL live US
factor set (WEIGHTS_US, v2.2-global), not just the original 7. Otherwise the
consensus/quality signals added in v2.4 (analyst_consensus, quality_piotroski,
transcript_tone, revenue_revision) are invisible to redundancy detection.
"""
from __future__ import annotations

import random

from src.config.weights import WEIGHTS_US
from src.monitoring.factor_orthogonality import (
    FACTORS_TO_MONITOR,
    compute_factor_correlation_matrix,
)

# INTL-only / US-zeroed factors deliberately excluded from the US monitor.
_NON_US_FACTORS = {
    "fcf_yield", "amihud_shock", "pb_value_up", "roic_quality",
    "analyst_revision", "price_target_upside",
}


def test_monitor_covers_v24_consensus_quality_factors():
    """The four v2.4 factors that were previously unmonitored are now included."""
    for factor in (
        "analyst_consensus_score",
        "quality_piotroski_score",
        "transcript_tone_score",
        "revenue_revision_score",
    ):
        assert factor in FACTORS_TO_MONITOR, f"{factor} missing from monitor"


def test_monitor_matches_live_us_weight_keys():
    """Every monitored factor maps to a live WEIGHTS_US factor (no stale keys),
    and every live US factor (minus the structurally-zeroed INTL set) is covered."""
    monitored = {f.removesuffix("_score") for f in FACTORS_TO_MONITOR}
    live_us = {f for f, w in WEIGHTS_US.items() if w > 0} - _NON_US_FACTORS
    assert monitored == live_us, (
        f"monitor/live drift — only_monitored={monitored - live_us}, "
        f"only_live={live_us - monitored}"
    )


def test_correlation_matrix_evaluates_all_monitored_factors():
    """With dense values present, no monitored factor is skipped or sparsity-excluded."""
    rng = random.Random(42)
    rows = []
    for _ in range(40):  # > min_observations (30)
        row = {"market": "USA"}
        for factor in FACTORS_TO_MONITOR:
            row[factor] = rng.uniform(0.1, 0.9)  # dense, non-zero, independent
        rows.append(row)

    report = compute_factor_correlation_matrix(rows, market_filter="US")

    assert "error" not in report, report.get("error")
    assert set(report["factors_evaluated"]) == set(FACTORS_TO_MONITOR)
    assert not report["factors_skipped"]
    # Independent uniforms → no spurious redundancy errors.
    assert report["errors"] == []
