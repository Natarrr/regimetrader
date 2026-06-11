"""tests/scoring/test_fundamental_v3.py — v3.0 fundamental scorer additions.

Tests:
  1. quality_dupont: negative-range preservation (ROA −0.5% ≠ ROA −45%),
     DuPont fallback, partial-component renormalization.
  2. margin_expansion: TTM-vs-prior-TTM delta on discrete quarters,
     validation-driven annual fallback, None when both tracks unavailable.

No live network calls. Pure scorer math.
"""
from __future__ import annotations

import pytest

from src.scoring.fundamental_signals import (
    score_margin_expansion,
    score_quality_dupont,
)


class TestQualityDupont:
    def test_full_components_value(self):
        # roa_c = (0.10+0.10)/0.30; npm_c = (0.15+0.10)/0.35; lev = 1.0
        # score = 0.5·0.666667 + 0.3·0.714286 + 0.2·1.0 = 0.747619
        score = score_quality_dupont(
            roa=0.10, npm=0.15, asset_turnover=None, debt_to_equity=0.3,
        )
        assert score == pytest.approx(0.747619, abs=1e-4)

    def test_dupont_fallback_when_roa_absent(self):
        # roa_eff = npm × asset_turnover = 0.10 × 1.5 = 0.15
        # D/E missing → renormalize over (0.5, 0.3):
        # (0.5·0.833333 + 0.3·0.571429) / 0.8 = 0.735119
        score = score_quality_dupont(
            roa=None, npm=0.10, asset_turnover=1.5, debt_to_equity=None,
        )
        assert score == pytest.approx(0.735119, abs=1e-4)

    def test_negative_variance_preserved(self):
        # The whole point of the negative-range clip: a minor headwind must
        # outrank catastrophic losses instead of flattening to the same 0.
        mild = score_quality_dupont(
            roa=-0.005, npm=0.05, asset_turnover=None, debt_to_equity=0.3)
        severe = score_quality_dupont(
            roa=-0.45, npm=0.05, asset_turnover=None, debt_to_equity=0.3)
        assert mild > severe

    def test_leverage_tiers(self):
        base = dict(roa=0.10, npm=0.10, asset_turnover=None)
        s_low = score_quality_dupont(**base, debt_to_equity=0.2)
        s_mid = score_quality_dupont(**base, debt_to_equity=0.8)
        s_high = score_quality_dupont(**base, debt_to_equity=1.5)
        s_extreme = score_quality_dupont(**base, debt_to_equity=3.0)
        assert s_low > s_mid > s_high > s_extreme

    def test_all_missing_is_dead_zero(self):
        # Quality data is universal — total absence means broken feed.
        assert score_quality_dupont(
            roa=None, npm=None, asset_turnover=None, debt_to_equity=None,
        ) == 0.0

    def test_bounded_unit_interval(self):
        score = score_quality_dupont(
            roa=0.50, npm=0.60, asset_turnover=None, debt_to_equity=0.1)
        assert 0.0 <= score <= 1.0


def _q(date, filing, rev, op):
    return {"date": date, "filingDate": filing,
            "revenue": rev, "operatingIncome": op}


_QUARTERS_OK = [
    _q("2026-03-31", "2026-05-10", 100.0, 12.0),
    _q("2025-12-31", "2026-02-10", 100.0, 12.0),
    _q("2025-09-30", "2025-11-10", 100.0, 12.0),
    _q("2025-06-30", "2025-08-10", 100.0, 12.0),
    _q("2025-03-31", "2025-05-10", 100.0, 10.0),
    _q("2024-12-31", "2025-02-10", 100.0, 10.0),
    _q("2024-09-30", "2024-11-10", 100.0, 10.0),
    _q("2024-06-30", "2024-08-10", 100.0, 10.0),
]

_ANNUAL_OK = [
    _q("2026-03-31", "2026-06-20", 400.0, 48.0),   # OPM 12%
    _q("2025-03-31", "2025-06-20", 400.0, 40.0),   # OPM 10%
]


class TestMarginExpansion:
    def test_quarterly_ttm_delta(self):
        # OPM now 12% vs prior 10% → Δ +0.02 → 0.5 + 0.02/0.20 = 0.60
        assert score_margin_expansion(_QUARTERS_OK, []) == pytest.approx(0.60)

    def test_contraction_scores_below_half(self):
        # swap halves: now 10%, prior 12% → Δ −0.02 → 0.40
        swapped = (
            [_q(r["date"], r["filingDate"], 100.0, 10.0) for r in _QUARTERS_OK[:4]]
            + [_q(r["date"], r["filingDate"], 100.0, 12.0) for r in _QUARTERS_OK[4:]]
        )
        assert score_margin_expansion(swapped, []) == pytest.approx(0.40)

    def test_annual_fallback_when_too_few_quarters(self):
        assert score_margin_expansion(
            _QUARTERS_OK[:7], _ANNUAL_OK) == pytest.approx(0.60)

    def test_invalid_spacing_falls_to_annual(self):
        # Two rows share a period-end (cumulative/duplicate symptom) →
        # quarterly track rejected → annual track must be used.
        bad = list(_QUARTERS_OK)
        bad[1] = _q("2026-03-31", "2026-02-10", 100.0, 12.0)
        assert score_margin_expansion(bad, _ANNUAL_OK) == pytest.approx(0.60)

    def test_none_when_both_tracks_unavailable(self):
        assert score_margin_expansion(_QUARTERS_OK[:5], []) is None

    def test_delta_clipped_to_band(self):
        # +30pp margin swing clips at +0.10 → score 1.0
        wide = (
            [_q(r["date"], r["filingDate"], 100.0, 40.0) for r in _QUARTERS_OK[:4]]
            + [_q(r["date"], r["filingDate"], 100.0, 10.0) for r in _QUARTERS_OK[4:]]
        )
        assert score_margin_expansion(wide, []) == pytest.approx(1.0)
