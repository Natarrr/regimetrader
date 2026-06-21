"""tests/scoring/test_candidate_factors.py — shadow candidate factors (A1 + A2).

Pure scorer math, no network. Covers the valuation-breadth (E/P, EV/EBITDA) and
growth/earnings-quality (revenue/EPS growth, accruals) scorers added to widen the
factor set. These are weight-0 until a de-overlapped IC gate justifies a weight.
"""
from __future__ import annotations

import pytest

from src.scoring.fundamental_signals import (
    score_accruals,
    score_dcf_upside,
    score_earnings_yield,
    score_eps_growth,
    score_ev_ebitda,
    score_revenue_growth,
    score_sector_relative_value,
)


# ── A1 · Earnings yield (E/P) ─────────────────────────────────────────────────

class TestEarningsYield:
    def test_midband_value(self):
        # ey = 10/100 = 0.10 → 0.10/0.15
        assert score_earnings_yield(10.0, 100.0) == pytest.approx(0.6667, abs=1e-4)

    def test_clips_high_yield_to_one(self):
        assert score_earnings_yield(20.0, 100.0) == 1.0   # ey 0.20 → clip 0.15

    def test_loss_is_dead_zero(self):
        assert score_earnings_yield(-5.0, 100.0) == 0.0
        assert score_earnings_yield(0.0, 100.0) == 0.0

    def test_bad_market_cap_dead_zero(self):
        assert score_earnings_yield(10.0, 0.0) == 0.0
        assert score_earnings_yield(10.0, None) == 0.0

    def test_bounded_unit_interval(self):
        assert 0.0 <= score_earnings_yield(7.0, 100.0) <= 1.0


# ── A1 · Enterprise multiple (EV/EBITDA) ──────────────────────────────────────

class TestEvEbitda:
    def test_midband_value(self):
        # ratio = 200/10 = 20 → 1 - (20-5)/30 = 0.5
        assert score_ev_ebitda(200.0, 10.0) == pytest.approx(0.5, abs=1e-4)

    def test_cheap_multiple_scores_high(self):
        assert score_ev_ebitda(40.0, 10.0) == 1.0          # ratio 4 → clip 5 → 1.0

    def test_expensive_multiple_floors_at_zero(self):
        assert score_ev_ebitda(400.0, 10.0) == 0.0         # ratio 40 → clip 35 → 0.0

    def test_negative_ebitda_dead_zero(self):
        assert score_ev_ebitda(120.0, -5.0) == 0.0
        assert score_ev_ebitda(0.0, 10.0) == 0.0
        assert score_ev_ebitda(120.0, None) == 0.0

    def test_cheaper_outranks_pricier(self):
        assert score_ev_ebitda(60.0, 10.0) > score_ev_ebitda(150.0, 10.0)


# ── A2 · TTM YoY growth ───────────────────────────────────────────────────────

def _row(date, filing, **fields):
    return {"date": date, "filingDate": filing, **fields}


def _eight_quarters(field, recent, prior):
    """8 discrete quarters (≈91d gaps): first 4 = `recent`, last 4 = `prior`."""
    ends = ["2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30",
            "2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30"]
    filings = ["2026-05-10", "2026-02-10", "2025-11-10", "2025-08-10",
               "2025-05-10", "2025-02-10", "2024-11-10", "2024-08-10"]
    vals = [recent] * 4 + [prior] * 4
    return [_row(e, f, **{field: v}) for e, f, v in zip(ends, filings, vals)]


class TestRevenueGrowth:
    def test_quarterly_ttm_yoy(self):
        # TTM now 440 vs prior 400 → +10% → 0.5 + 0.10/0.60
        rows = _eight_quarters("revenue", 110.0, 100.0)
        assert score_revenue_growth(rows, []) == pytest.approx(0.6667, abs=1e-4)

    def test_contraction_below_half(self):
        rows = _eight_quarters("revenue", 90.0, 100.0)   # −10%
        assert score_revenue_growth(rows, []) == pytest.approx(0.3333, abs=1e-4)

    def test_growth_clipped_to_band(self):
        rows = _eight_quarters("revenue", 200.0, 100.0)  # +100% → clip +30%
        assert score_revenue_growth(rows, []) == 1.0

    def test_annual_fallback(self):
        annuals = [_row("2026-03-31", "2026-06-20", revenue=440.0),
                   _row("2025-03-31", "2025-06-20", revenue=400.0)]
        assert score_revenue_growth([], annuals) == pytest.approx(0.6667, abs=1e-4)

    def test_none_when_no_data(self):
        assert score_revenue_growth([], []) is None

    def test_none_when_prior_non_positive(self):
        annuals = [_row("2026-03-31", "2026-06-20", revenue=440.0),
                   _row("2025-03-31", "2025-06-20", revenue=0.0)]
        assert score_revenue_growth([], annuals) is None

    def test_filing_date_required(self):
        # Missing filingDate → row dropped (look-ahead guard); too few rows → None
        rows = _eight_quarters("revenue", 110.0, 100.0)
        for r in rows:
            r.pop("filingDate")
        assert score_revenue_growth(rows, []) is None


class TestEpsGrowth:
    def test_quarterly_ttm_yoy(self):
        # TTM now 48 vs prior 40 → +20% → 0.5 + 0.20/1.0 (band 0.50)
        rows = _eight_quarters("netIncome", 12.0, 10.0)
        assert score_eps_growth(rows, []) == pytest.approx(0.70, abs=1e-4)

    def test_none_when_prior_loss(self):
        rows = _eight_quarters("netIncome", 12.0, -10.0)   # prior TTM < 0
        assert score_eps_growth(rows, []) is None

    def test_wider_band_than_revenue(self):
        # +40% earnings growth: EPS band 0.50 keeps it below 1.0 …
        rows = _eight_quarters("netIncome", 14.0, 10.0)    # +40%
        assert score_eps_growth(rows, []) == pytest.approx(0.90, abs=1e-4)


# ── A2 · Accruals (Sloan 1996) ────────────────────────────────────────────────

class TestAccruals:
    def test_midrange_value(self):
        # accr = (10-8)/100 = 0.02 → 1 - (0.02+0.20)/0.40 = 0.45
        assert score_accruals(10.0, 8.0, 100.0) == pytest.approx(0.45, abs=1e-4)

    def test_low_accruals_outranks_high(self):
        cash_backed = score_accruals(10.0, 12.0, 100.0)   # accr −0.02
        accrual_heavy = score_accruals(30.0, 5.0, 100.0)  # accr +0.25 → clip
        assert cash_backed > accrual_heavy

    def test_extreme_accruals_floor_zero(self):
        assert score_accruals(30.0, 5.0, 100.0) == 0.0    # accr 0.25 → clip 0.20

    def test_bad_assets_dead_zero(self):
        assert score_accruals(10.0, 8.0, 0.0) == 0.0
        assert score_accruals(10.0, None, 100.0) == 0.0

    def test_bounded_unit_interval(self):
        assert 0.0 <= score_accruals(5.0, 7.0, 80.0) <= 1.0


# ── A3 · Sector-relative value (own P/E vs sector P/E) ─────────────────────────

class TestSectorRelativeValue:
    def test_parity_with_sector(self):
        # ratio 1.0 → 1 - (1.0-0.5)/1.5
        assert score_sector_relative_value(20.0, 20.0) == pytest.approx(0.6667, abs=1e-4)

    def test_cheap_vs_sector_scores_high(self):
        assert score_sector_relative_value(10.0, 20.0) == 1.0     # ratio 0.5 → clip → 1.0

    def test_expensive_vs_sector_floors_zero(self):
        assert score_sector_relative_value(40.0, 20.0) == 0.0     # ratio 2.0 → clip → 0.0

    def test_loss_making_dead_zero(self):
        assert score_sector_relative_value(-5.0, 20.0) == 0.0
        assert score_sector_relative_value(0.0, 20.0) == 0.0

    def test_bad_sector_pe_dead_zero(self):
        assert score_sector_relative_value(20.0, 0.0) == 0.0
        assert score_sector_relative_value(20.0, None) == 0.0
        assert score_sector_relative_value(20.0, -10.0) == 0.0

    def test_cheaper_outranks_pricier(self):
        assert (score_sector_relative_value(15.0, 20.0)
                > score_sector_relative_value(30.0, 20.0))


# ── A3 · DCF intrinsic-value upside (SIGNED, center 0.5) ───────────────────────

class TestDcfUpside:
    def test_at_fair_value_is_neutral_half(self):
        assert score_dcf_upside(100.0, 100.0) == 0.5

    def test_full_upside_caps_at_one(self):
        assert score_dcf_upside(150.0, 100.0) == 1.0             # +50% → clip → 1.0

    def test_full_downside_floors_zero(self):
        assert score_dcf_upside(50.0, 100.0) == 0.0             # -50% → clip → 0.0

    def test_midband_upside(self):
        assert score_dcf_upside(120.0, 100.0) == pytest.approx(0.70, abs=1e-4)

    def test_signed_none_when_unavailable(self):
        assert score_dcf_upside(None, 100.0) is None
        assert score_dcf_upside(120.0, 0.0) is None
        assert score_dcf_upside(120.0, None) is None

    def test_none_when_dcf_non_positive(self):
        assert score_dcf_upside(0.0, 100.0) is None
        assert score_dcf_upside(-30.0, 100.0) is None
