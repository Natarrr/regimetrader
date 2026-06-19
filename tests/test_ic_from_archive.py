"""tests/test_ic_from_archive.py — archive → snapshot bridge for factor IC.

Covers the pure, network-free snapshot builder. The price fetch + IC math are
exercised by their own modules (backtest_signals / ic_metrics).
"""
from __future__ import annotations

from datetime import date

from scripts.backtest_signals import SignalRecord
from scripts.ic_from_archive import _RETURN_KEY, build_snapshots


def _rec(ticker, d, factors, ret20, entry=100.0):
    r = SignalRecord(
        ticker=ticker, signal_date=d, badge="TACTICAL BUY", cap_tier="large",
        final_score=0.7, factors=factors, weights={}, strategy_era="e",
        source_file="f", entry_next_day=False)
    r.entry_price = entry
    r.returns = {20: ret20}
    return r


def test_groups_by_date_and_attaches_forward_return():
    d1, d2 = date(2026, 5, 28), date(2026, 5, 29)
    recs = [
        _rec("A", d1, {"momentum_long": 0.8}, 0.05),
        _rec("B", d1, {"momentum_long": 0.4}, -0.02),
        _rec("C", d2, {"momentum_long": 0.9}, 0.03),
    ]
    snaps = build_snapshots(recs, horizon=20)
    assert [s["date"] for s in snaps] == ["2026-05-28", "2026-05-29"]
    assert len(snaps[0]["rows"]) == 2
    assert snaps[0]["rows"][0][_RETURN_KEY] == 0.05
    assert "momentum_long" in snaps[0]["rows"][0]


def test_drops_unpriced_and_missing_horizon():
    d = date(2026, 5, 28)
    priced = _rec("A", d, {"x": 0.5}, 0.05)
    unpriced = _rec("B", d, {"x": 0.5}, 0.05)
    unpriced.entry_price = None
    no_fwd = _rec("C", d, {"x": 0.5}, 0.05)
    no_fwd.returns = {20: None}
    snaps = build_snapshots([priced, unpriced, no_fwd], horizon=20)
    assert len(snaps) == 1 and len(snaps[0]["rows"]) == 1
