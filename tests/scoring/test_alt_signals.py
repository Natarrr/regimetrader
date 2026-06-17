"""tests/scoring/test_alt_signals.py — v3.0 alternative / micro-structural flow scorers.

Tests:
  1. insider_alpha dead short-circuit (the 0.03 mass-point regression).
  2. insider_alpha composite math, P-code-only C-suite tilt.
  3. congress surge multiplier bounds.
  4. inst_flow_13f / inst_concentration formulas, [0,1] bounds, None on empty.
  5. dividend_sustain payer/non-payer/missing semantics.

No live network calls. Pure scorer math.
"""
from __future__ import annotations

import math

import pytest

from src.scoring.alt_signals import (
    congress_surge_multiplier,
    score_dividend_sustain,
    score_insider_alpha,
    score_insider_npr_spike,
    score_inst_concentration,
    score_inst_flow_13f,
)


class TestInsiderAlpha:
    def test_dead_short_circuit_is_exactly_zero(self):
        # Regression for the 0.03 mass-point: zero P-code purchases across the
        # lookback must yield EXACTLY 0.0 (the zero_is_dead exclusion tier),
        # never the csuite-neutral leak 0.15 × 0.4 × 0.5 = 0.03.
        score = score_insider_alpha(
            conviction=0.0, breadth_residual=0.0,
            p_count_30d=0, p_count_31_180d=0,
            usd_buy_csuite=0.0, usd_buy_total=0.0,
        )
        assert score == 0.0

    def test_active_composite_value(self):
        # velocity = log1p(3/1.0)/log1p(5) = 0.773745; tilt = 0.75
        # micro = 0.6·0.773745 + 0.4·0.75 = 0.764247
        # score = 0.55·0.8 + 0.30·0.4 + 0.15·0.764247 = 0.674637
        score = score_insider_alpha(
            conviction=0.8, breadth_residual=0.4,
            p_count_30d=3, p_count_31_180d=5,
            usd_buy_csuite=500_000.0, usd_buy_total=1_000_000.0,
        )
        assert score == pytest.approx(0.674637, abs=1e-4)

    def test_csuite_neutral_only_when_usd_unparseable(self):
        # Purchases exist (P_count > 0) but USD values unparseable → tilt 0.5.
        # velocity = log1p(2/1)/log1p(5) = 0.613147; micro = 0.567888
        # score = 0.55·0.5 + 0.30·0.2 + 0.15·0.567888 = 0.420183
        score = score_insider_alpha(
            conviction=0.5, breadth_residual=0.2,
            p_count_30d=2, p_count_31_180d=0,
            usd_buy_csuite=0.0, usd_buy_total=0.0,
        )
        assert score == pytest.approx(0.420183, abs=1e-4)

    def test_clipped_to_unit_interval(self):
        score = score_insider_alpha(
            conviction=1.0, breadth_residual=1.0,
            p_count_30d=50, p_count_31_180d=1,
            usd_buy_csuite=1_000_000.0, usd_buy_total=1_000_000.0,
        )
        assert 0.0 <= score <= 1.0

    def test_velocity_capped_at_one(self):
        # Extreme 30d burst: velocity term must saturate at 1.0, score finite.
        score = score_insider_alpha(
            conviction=0.0, breadth_residual=0.0,
            p_count_30d=1000, p_count_31_180d=0,
            usd_buy_csuite=0.0, usd_buy_total=1.0,
        )
        # micro = 0.6·1.0 + 0.4·0.5 = 0.8 → score = 0.15·0.8 = 0.12
        assert score == pytest.approx(0.12, abs=1e-6)


class TestCongressSurgeMultiplier:
    def test_no_surge_is_identity(self):
        assert congress_surge_multiplier(0, 10) == pytest.approx(1.0)

    def test_steady_flow_is_identity(self):
        # nb30 == pro-rata baseline → surge = 1 → no boost
        assert congress_surge_multiplier(2, 12) == pytest.approx(1.0)

    def test_strong_surge_caps_at_1_25(self):
        # baseline = 12·30/180 = 2 → surge = 5 → mult capped at 1.25
        assert congress_surge_multiplier(10, 12) == pytest.approx(1.25)

    def test_mid_surge_value(self):
        # surge = 3/2 = 1.5 → mult = 1 + 0.25·0.25 = 1.0625
        assert congress_surge_multiplier(3, 12) == pytest.approx(1.0625)


class TestInstFlow13F:
    def test_empty_payload_is_none(self):
        assert score_inst_flow_13f({}) is None
        assert score_inst_flow_13f(None) is None

    def test_composite_value(self):
        # 0.5 + 0.2·clip(10·50/1000) + 0.2·(100/500) + 0.1·(2/5)
        #   = 0.5 + 0.1 + 0.04 + 0.04 = 0.68
        row = {"investorsHolding": 1000, "investorsHoldingChange": 50,
               "increasedPositions": 300, "reducedPositions": 200,
               "ownershipPercentChange": 2.0}
        assert score_inst_flow_13f(row) == pytest.approx(0.68, abs=1e-6)

    def test_extreme_outflow_floors_at_zero(self):
        row = {"investorsHolding": 1000, "investorsHoldingChange": -500,
               "increasedPositions": 0, "reducedPositions": 500,
               "ownershipPercentChange": -10.0}
        assert score_inst_flow_13f(row) == pytest.approx(0.0)


class TestInstConcentration:
    def test_empty_payload_is_none(self):
        assert score_inst_concentration({}) is None
        assert score_inst_concentration(None) is None

    def test_composite_value(self):
        # 0.4·(40/80) + 0.3·(log1p(400)/log1p(500)) + 0.3·(0.5 + 0.5·0.2)
        expected = (0.4 * 0.5
                    + 0.3 * (math.log1p(400) / math.log1p(500))
                    + 0.3 * 0.6)
        row = {"ownershipPercent": 40.0, "investorsHolding": 400,
               "increasedPositions": 300, "reducedPositions": 200}
        assert score_inst_concentration(row) == pytest.approx(expected, abs=1e-4)

    def test_mega_roster_bounded_at_one(self):
        # 2,500 holders would push log1p ratio to ≈1.26 without the min(1,·)
        # cap — composite must stay ≤ 1.0 (bounded-[0,1] contract).
        row = {"ownershipPercent": 80.0, "investorsHolding": 2500,
               "increasedPositions": 1000, "reducedPositions": 0}
        score = score_inst_concentration(row)
        assert score == pytest.approx(1.0)
        assert score <= 1.0


class TestDividendSustain:
    def test_non_payer_is_dead_zero(self):
        assert score_dividend_sustain(
            dividend_yield=0.0, payout_ratio=0.0,
            fcf_ttm=500.0, dividends_paid_ttm=0.0,
        ) == 0.0

    def test_payer_composite_value(self):
        # 0.45·(1−0) + 0.35·clip(300/100−1, 0, 2)/2 + 0.20·(0.04/0.08) = 0.90
        score = score_dividend_sustain(
            dividend_yield=0.04, payout_ratio=0.45,
            fcf_ttm=300.0, dividends_paid_ttm=-100.0,
        )
        assert score == pytest.approx(0.90, abs=1e-6)

    def test_payer_with_missing_field_is_none(self):
        assert score_dividend_sustain(
            dividend_yield=0.04, payout_ratio=None,
            fcf_ttm=300.0, dividends_paid_ttm=-100.0,
        ) is None

    def test_all_missing_is_none(self):
        assert score_dividend_sustain(
            dividend_yield=None, payout_ratio=None,
            fcf_ttm=None, dividends_paid_ttm=None,
        ) is None


class TestInsiderNprSpike:
    def test_none_on_empty(self):
        assert score_insider_npr_spike(None) is None
        assert score_insider_npr_spike([]) is None

    def test_none_when_latest_quarter_has_no_transactions(self):
        assert score_insider_npr_spike(
            [{"year": 2026, "quarter": 1,
              "acquiredTransactions": 0, "disposedTransactions": 0}]
        ) is None

    def test_npr_and_spike_vs_baseline(self):
        # Latest quarter: 9 buys / 1 sell → NPR 0.9. Trailing two quarters avg
        # NPR = (0.5 + 0.5)/2 = 0.5 → spike = +0.4 (unusual cluster buying).
        stats = [
            {"year": 2026, "quarter": 2, "acquiredTransactions": 9, "disposedTransactions": 1},
            {"year": 2026, "quarter": 1, "acquiredTransactions": 5, "disposedTransactions": 5},
            {"year": 2025, "quarter": 4, "acquiredTransactions": 3, "disposedTransactions": 3},
        ]
        out = score_insider_npr_spike(stats)
        assert out["npr"] == pytest.approx(0.9)
        assert out["spike"] == pytest.approx(0.4)
        assert out["acquired"] == 9 and out["disposed"] == 1

    def test_single_quarter_spike_is_zero(self):
        # No prior quarters → baseline falls back to the latest NPR → spike 0.0
        # (insufficient history must never fabricate a spike).
        out = score_insider_npr_spike(
            [{"year": 2026, "quarter": 2,
              "acquiredTransactions": 8, "disposedTransactions": 2}])
        assert out["npr"] == pytest.approx(0.8)
        assert out["spike"] == pytest.approx(0.0)
