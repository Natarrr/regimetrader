"""tests/scoring/test_technical_candidates.py — shadow technical candidates.

Pure scorer math, no network. Covers the price-technical candidates computed
from OHLCV we already pull (zero marginal API cost):
    - score_rsi_reversion  (Wilder RSI → short-term reversal tilt, SIGNED)
    - score_adx_trend      (Wilder ADX → trend strength, UNSIGNED)
Weight-0 until a de-overlapped IC gate justifies a weight.
"""
from __future__ import annotations

import pytest

from src.scoring.momentum_signals import score_adx_trend, score_rsi_reversion


# ── RSI mean-reversion (SIGNED, center 0.5) ───────────────────────────────────

class TestRsiReversion:
    def test_pure_uptrend_is_overbought_low_score(self):
        # Monotone rising closes → RSI 100 (overbought) → reversal-down tilt → 0.0
        closes = [float(x) for x in range(1, 17)]   # 16 closes, period 14
        assert score_rsi_reversion(closes) == 0.0

    def test_pure_downtrend_is_oversold_high_score(self):
        # Monotone falling closes → RSI 0 (oversold) → reversal-up tilt → 1.0
        closes = [float(x) for x in range(16, 0, -1)]
        assert score_rsi_reversion(closes) == 1.0

    def test_flat_series_is_neutral_half(self):
        closes = [25.0] * 16
        assert score_rsi_reversion(closes) == 0.5

    def test_midband_rsi(self):
        # period=2, deltas +1 then -0.5 → RSI = 100-100/3 = 66.67 → 0.5+(50-66.67)/100
        assert score_rsi_reversion([10.0, 11.0, 10.5], period=2) == pytest.approx(
            0.3333, abs=1e-4
        )

    def test_signed_none_when_insufficient_history(self):
        assert score_rsi_reversion([1.0] * 14) is None   # need period+1
        assert score_rsi_reversion([]) is None
        assert score_rsi_reversion(None) is None

    def test_none_when_data_unparseable(self):
        closes = [float(x) for x in range(1, 16)] + [None]
        assert score_rsi_reversion(closes) is None

    def test_bounded_unit_interval(self):
        closes = [10.0, 11.0, 10.0, 12.0, 11.5, 13.0, 12.0, 14.0,
                  13.5, 15.0, 14.0, 16.0, 15.5, 17.0, 16.0, 18.0]
        assert 0.0 <= score_rsi_reversion(closes) <= 1.0


# ── ADX trend strength (UNSIGNED, 0.0 dead) ───────────────────────────────────

def _trend(start, step, n, gap=2.0):
    """n bars of a constant-step linear trend; close at the high."""
    highs = [start + step * i for i in range(n)]
    lows = [h - gap for h in highs]
    closes = list(highs)
    return highs, lows, closes


class TestAdxTrend:
    def test_strong_uptrend_max_strength(self):
        highs, lows, closes = _trend(10.0, 1.0, 29)   # 2*period+1
        assert score_adx_trend(highs, lows, closes) == 1.0

    def test_strong_downtrend_also_max_strength(self):
        # ADX is non-directional: a clean downtrend is just as "trending".
        highs, lows, closes = _trend(100.0, -1.0, 29)
        assert score_adx_trend(highs, lows, closes) == 1.0

    def test_flat_series_dead_zero(self):
        flat = [25.0] * 29
        assert score_adx_trend(flat, flat, flat) == 0.0

    def test_dead_zero_when_insufficient_history(self):
        highs, lows, closes = _trend(10.0, 1.0, 28)   # one short of 2*period+1
        assert score_adx_trend(highs, lows, closes) == 0.0

    def test_dead_zero_on_ragged_inputs(self):
        highs, lows, closes = _trend(10.0, 1.0, 29)
        assert score_adx_trend(highs, lows[:-1], closes) == 0.0
        assert score_adx_trend(None, None, None) == 0.0

    def test_bounded_unit_interval(self):
        highs, lows, closes = _trend(10.0, 0.3, 40)
        assert 0.0 <= score_adx_trend(highs, lows, closes) <= 1.0
