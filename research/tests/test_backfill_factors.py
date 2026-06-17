# Path: research/tests/test_backfill_factors.py
"""Backfill point-in-time reconstruction — pure quant helpers.

These functions carry the look-ahead-bias risk (CLAUDE.md §3): every value at
snapshot date D must be computable from data observable on or before D. The
network fetch loop is a thin shell around them.
"""
from __future__ import annotations

from datetime import date, timedelta

from research.scripts.backfill_factors import (
    anchor_filing,
    forward_excess_return,
    momentum_excess,
    sample_snapshot_indices,
)


class TestSampleSnapshotIndices:
    def test_spacing_at_least_horizon_and_leaves_forward_window(self):
        # 60 trading days, want snapshots 21 apart with a 21d forward window
        idxs = sample_snapshot_indices(n=60, spacing=21, horizon=21, count=52)
        # last usable index is 60-1-21 = 38; step back by 21 → {38, 17}
        assert idxs == [17, 38]
        # every gap respects the horizon (no overlap at the source)
        assert all(b - a >= 21 for a, b in zip(idxs, idxs[1:]))

    def test_count_caps_number_of_snapshots(self):
        idxs = sample_snapshot_indices(n=500, spacing=21, horizon=21, count=3)
        assert len(idxs) == 3


class TestForwardExcessReturn:
    def test_spy_relative_excess(self):
        closes = [100.0, 110.0, 120.0]
        spy = [100.0, 100.0, 100.0]
        # idx 0 → +20% asset, 0% SPY → +20% excess
        assert abs(forward_excess_return(closes, spy, 0, horizon=2) - 0.20) < 1e-9

    def test_none_when_no_forward_data(self):
        closes = [100.0, 110.0, 120.0]
        spy = [100.0, 100.0, 100.0]
        assert forward_excess_return(closes, spy, 2, horizon=2) is None


class TestMomentumExcess:
    def test_twelve_minus_one_spy_relative(self):
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        spy = [10.0, 10.0, 10.0, 10.0, 10.0]
        # idx 4, lookback 4, skip 1 → closes[3]/closes[0]-1 = 0.3, SPY 0 → 0.3
        assert abs(momentum_excess(closes, spy, 4, lookback=4, skip=1) - 0.3) < 1e-9

    def test_none_when_insufficient_history(self):
        closes = [10.0, 11.0, 12.0]
        spy = [10.0, 10.0, 10.0]
        assert momentum_excess(closes, spy, 2, lookback=5, skip=1) is None


class TestAnchorFiling:
    def test_picks_latest_filing_on_or_before_snapshot(self):
        filings = [
            {"filingDate": "2025-01-10", "v": 1},
            {"filingDate": "2025-03-15", "v": 2},
            {"filingDate": "2025-06-01", "v": 3},
        ]
        chosen = anchor_filing(filings, date(2025, 4, 1))
        assert chosen["v"] == 2

    def test_never_returns_future_filing(self):
        # snapshot before every filing → nothing observable yet (no look-ahead)
        filings = [{"filingDate": "2025-03-15", "v": 2}]
        assert anchor_filing(filings, date(2025, 1, 1)) is None

    def test_filters_filingdate_not_period_end(self):
        # a filing whose fiscal period ended before D but was FILED after D
        # must be excluded — anchoring on filingDate, never period end.
        filings = [{"filingDate": "2025-05-20", "date": "2025-03-31", "v": 9}]
        assert anchor_filing(filings, date(2025, 4, 15)) is None
