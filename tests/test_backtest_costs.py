"""tests/test_backtest_costs.py — net-of-cost backtest returns (audit P2.2).

The backtest previously reported GROSS returns; on a ~20%-turnover book that
overstates the edge. compute_returns now subtracts a cap-tier round-trip cost.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest_signals import (
    SignalRecord,
    _roundtrip_cost,
    compute_returns,
)


def test_roundtrip_cost_by_tier():
    assert _roundtrip_cost("large") == pytest.approx(0.0020)
    assert _roundtrip_cost("mid") == pytest.approx(0.0040)
    assert _roundtrip_cost("small") == pytest.approx(0.0060)
    assert _roundtrip_cost("") == pytest.approx(0.0020)       # unknown → large default


def _record(cap_tier: str, signal_ts) -> SignalRecord:
    return SignalRecord(
        ticker="X", signal_date=signal_ts.date(), badge="HIGH BUY",
        cap_tier=cap_tier, final_score=0.9, factors={}, weights={},
        strategy_era="e", source_file="f", entry_next_day=False,
    )


def test_compute_returns_is_net_of_cost():
    idx = pd.bdate_range("2026-01-05", periods=30)
    closes = pd.Series([100.0] * 30, index=idx)
    closes.iloc[5] = closes.iloc[10] = closes.iloc[20] = 110.0   # +10% gross

    rec = _record("large", idx[0])
    compute_returns([rec], {"X": closes})
    # gross +10% minus the 20 bps large-cap round-trip cost
    assert rec.returns[5] == pytest.approx(0.10 - _roundtrip_cost("large"), abs=1e-6)


def test_smaller_caps_pay_more_cost():
    idx = pd.bdate_range("2026-01-05", periods=30)
    closes = pd.Series([100.0] * 30, index=idx)
    closes.iloc[5] = closes.iloc[10] = closes.iloc[20] = 110.0

    big = _record("large", idx[0])
    small = _record("small", idx[0])
    compute_returns([big], {"X": closes})
    compute_returns([small], {"X": closes})
    assert big.returns[5] > small.returns[5]   # small-cap nets less after costs
