# Path: research/tests/test_ic_metrics.py
"""IC metrics — rank-IC time-series aggregation with de-overlap embargo.

The per-snapshot rank-IC reuses the SSOT spearman in tools.compare_v22_v3
(same implementation monitoring/region_metrics.py uses). What is NEW here and
under test is the *time-series* aggregation across snapshots and the
overlap-embargo correction (López de Prado 2018, ch. 7): daily/weekly
snapshots paired with a 21-day horizon overlap, which inflates the apparent
number of independent observations and therefore the significance of the IR.
"""
from __future__ import annotations

from datetime import date, timedelta

from src.research.ic_metrics import (
    aggregate_ic,
    compute_ic_report,
    effective_breadth,
    factor_ic_series,
    snapshot_ic,
    weight_recommendation,
)


def _snapshot(d, scores_and_returns):
    """Build a snapshot: {date, rows:[{ticker, <factor>, forward_return_21d}]}."""
    rows = [
        {"ticker": t, "momentum_long": s, "forward_return_21d": r}
        for t, s, r in scores_and_returns
    ]
    return {"date": d, "rows": rows}


class TestSnapshotIC:
    def test_perfect_rank_correlation_is_one(self):
        # factor score and forward return share the same ranking → IC == 1.0
        snap = _snapshot("2025-01-31", [
            ("A", 0.1, 0.01), ("B", 0.2, 0.02),
            ("C", 0.3, 0.03), ("D", 0.4, 0.04),
        ])
        assert snapshot_ic(snap, "momentum_long") == 1.0

    def test_perfect_inversion_is_minus_one(self):
        snap = _snapshot("2025-01-31", [
            ("A", 0.1, 0.04), ("B", 0.2, 0.03),
            ("C", 0.3, 0.02), ("D", 0.4, 0.01),
        ])
        assert snapshot_ic(snap, "momentum_long") == -1.0

    def test_too_few_pairs_returns_none(self):
        snap = _snapshot("2025-01-31", [("A", 0.1, 0.01)])
        assert snapshot_ic(snap, "momentum_long") is None


class TestFactorICSeries:
    def test_series_one_ic_per_qualifying_snapshot(self):
        snaps = [
            _snapshot("2025-01-31", [
                ("A", 0.1, 0.01), ("B", 0.2, 0.02), ("C", 0.3, 0.03)]),
            _snapshot("2025-02-28", [
                ("A", 0.3, 0.01), ("B", 0.2, 0.02), ("C", 0.1, 0.03)]),
        ]
        series = factor_ic_series(snaps, "momentum_long")
        assert series == [1.0, -1.0]


class TestEffectiveBreadth:
    def test_daily_snapshots_collapse_to_horizon_spacing(self):
        # 63 consecutive calendar days; 21 business-day (≈ trading-day) embargo
        # → 3 independent windows.
        dates = [date(2025, 1, 1) + timedelta(days=k) for k in range(63)]
        assert effective_breadth(dates, horizon_days=21) == 3

    def test_monthly_firsts_collapse_short_month_overlap(self):
        # The 21-day horizon is in TRADING days (~30 calendar). A calendar-month
        # gap is only ~20–23 business days, so Feb 1 → Mar 1 (20 business days)
        # overlaps and must collapse: five independent windows, not six.
        dates = [date(2025, m, 1) for m in range(1, 7)]
        assert effective_breadth(dates, horizon_days=21) == 5

    def test_trading_day_spaced_snapshots_all_independent(self):
        # Snapshots sampled exactly 21 trading days apart (how the backfill is
        # built) stay fully independent — n_effective == n_snapshots.
        import pandas as pd
        bdays = pd.bdate_range("2025-01-02", periods=200)
        dates = [bdays[i].date() for i in range(0, 200, 21)]
        assert effective_breadth(dates, horizon_days=21) == len(dates)


class TestAggregateIC:
    def test_aggregate_reports_embargo_corrected_breadth(self):
        ic_series = [0.05, 0.03, 0.04, 0.06, 0.02, 0.05]
        dates = [date(2025, 1, 1) + timedelta(days=k) for k in range(6)]
        agg = aggregate_ic(ic_series, dates, horizon_days=21)
        assert agg["n_snapshots"] == 6
        assert agg["n_effective"] == 1          # all 6 inside one 21d window
        assert abs(agg["mean_ic"] - 0.0416667) < 1e-4
        assert agg["ic_positive_rate"] == 1.0
        # t-stat must use the de-overlapped breadth, not the raw count
        assert abs(agg["ic_t_stat"] - agg["ic_ir"] * (1 ** 0.5)) < 1e-9


class TestWeightRecommendation:
    def test_negative_mean_ic_is_investigate(self):
        assert weight_recommendation(
            {"mean_ic": -0.01, "ic_ir": 0.8, "ic_positive_rate": 0.7}
        ) == "investigate"

    def test_strong_signal_is_increase(self):
        assert weight_recommendation(
            {"mean_ic": 0.05, "ic_ir": 0.61, "ic_positive_rate": 0.73}
        ) == "increase"

    def test_moderate_signal_is_hold(self):
        assert weight_recommendation(
            {"mean_ic": 0.02, "ic_ir": 0.35, "ic_positive_rate": 0.55}
        ) == "hold"

    def test_weak_signal_is_decrease(self):
        assert weight_recommendation(
            {"mean_ic": 0.01, "ic_ir": 0.1, "ic_positive_rate": 0.45}
        ) == "decrease"


class TestComputeICReport:
    def test_report_has_one_entry_per_factor_with_full_schema(self):
        snaps = [
            {"date": date(2025, 1, 31), "rows": [
                {"ticker": "A", "momentum_long": 0.1, "quality_piotroski": 0.4,
                 "forward_return_21d": 0.01},
                {"ticker": "B", "momentum_long": 0.2, "quality_piotroski": 0.3,
                 "forward_return_21d": 0.02},
                {"ticker": "C", "momentum_long": 0.3, "quality_piotroski": 0.2,
                 "forward_return_21d": 0.03},
            ]},
            {"date": date(2025, 3, 31), "rows": [
                # momentum ranks high→low A,B,C but returns low→high → IC = -1
                {"ticker": "A", "momentum_long": 0.3, "quality_piotroski": 0.2,
                 "forward_return_21d": 0.01},
                {"ticker": "B", "momentum_long": 0.2, "quality_piotroski": 0.3,
                 "forward_return_21d": 0.02},
                {"ticker": "C", "momentum_long": 0.1, "quality_piotroski": 0.4,
                 "forward_return_21d": 0.03},
            ]},
        ]
        report = compute_ic_report(snaps, ["momentum_long", "quality_piotroski"])
        assert set(report) == {"momentum_long", "quality_piotroski"}
        for stats in report.values():
            assert set(stats) >= {
                "mean_ic", "ic_ir", "ic_positive_rate", "n_snapshots",
                "n_effective", "ic_t_stat", "weight_recommendation",
            }
            assert stats["weight_recommendation"] in (
                "increase", "hold", "decrease", "investigate")
        # momentum_long perfectly predicts in snap1, inverts in snap2 → mean ~0
        assert abs(report["momentum_long"]["mean_ic"]) < 1e-9
