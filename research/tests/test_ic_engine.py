# Path: research/tests/test_ic_engine.py
import numpy as np
import pytest

from research.scripts.ic_engine import (
    rank_ic_per_snapshot,
    weight_recommendation,
    build_ic_report,
    FACTORS,
)


def test_rank_ic_perfect_positive():
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ic = rank_ic_per_snapshot(scores, returns)
    assert abs(ic - 1.0) < 1e-6


def test_rank_ic_perfect_negative():
    scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ic = rank_ic_per_snapshot(scores, returns)
    assert abs(ic + 1.0) < 1e-6


def test_rank_ic_all_identical_scores():
    scores = np.array([0.5, 0.5, 0.5])
    returns = np.array([0.01, -0.01, 0.02])
    ic = rank_ic_per_snapshot(scores, returns)
    assert ic == 0.0


def test_rank_ic_too_few_samples():
    ic = rank_ic_per_snapshot(np.array([0.5, 0.7]), np.array([0.01, 0.02]))
    assert np.isnan(ic) or ic == pytest.approx(0.0)


def test_weight_recommendation_investigate():
    assert weight_recommendation(-0.01, 0.4, 0.55) == "investigate"


def test_weight_recommendation_increase():
    assert weight_recommendation(0.05, 0.6, 0.70) == "increase"


def test_weight_recommendation_hold():
    assert weight_recommendation(0.02, 0.35, 0.55) == "hold"


def test_weight_recommendation_decrease():
    assert weight_recommendation(0.01, 0.2, 0.45) == "decrease"


def test_build_ic_report_all_factors_present():
    import random
    random.seed(42)
    records = []
    for date_offset in range(5):
        snap = f"2025-{6 + date_offset:02d}-06"
        for i in range(10):
            rec = {"snapshot_date": snap, "forward_return_21d": random.gauss(0, 0.02)}
            for f in FACTORS:
                rec[f] = random.uniform(0, 1)
            records.append(rec)

    report = build_ic_report(records)
    assert set(report.keys()) == set(FACTORS)
    for factor, metrics in report.items():
        assert "mean_ic" in metrics
        assert "ic_ir" in metrics
        assert "ic_positive_rate" in metrics
        assert "monthly_ic" in metrics
        assert metrics["weight_recommendation"] in ("increase", "hold", "decrease", "investigate")


def test_build_ic_report_values_in_range():
    import random
    random.seed(0)
    records = []
    for date_offset in range(10):
        snap = f"2025-{6 + date_offset:02d}-01"
        for i in range(20):
            rec = {"snapshot_date": snap, "forward_return_21d": random.gauss(0, 0.02)}
            for f in FACTORS:
                rec[f] = random.uniform(0, 1)
            records.append(rec)
    report = build_ic_report(records)
    for factor, metrics in report.items():
        assert -1.0 <= metrics["mean_ic"] <= 1.0
        assert 0.0 <= metrics["ic_positive_rate"] <= 1.0
