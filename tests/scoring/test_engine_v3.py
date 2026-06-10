"""tests/scoring/test_engine_v3.py — v3.0 pillar scoring engine.

Deterministic fixtures: identical raw values across a bucket give std=0 →
sigmoid(0)=0.5 neutrals; a 1-vs-5 split gives the population-z bound ±√5 →
sigmoid(±2.2360680) = 0.903442 / 0.096558 (6dp, as rounded by
neutralize_factors). All expected values below derive from those constants.

Covers: pillar aggregation, None-reweighting (factor and whole-pillar),
one-sided surge bound, Piotroski gate ordering, coverage flags, blackout
telemetry, EU λ=0.
"""
from __future__ import annotations

import pytest

from src.config.factor_matrix import (
    FACTOR_MATRIX_V3,
    SURGE_MAX_BONUS,
)
from src.scoring.engine_v3 import score_universe_v3

_HI = 0.903442   # sigmoid(+sqrt(5)) rounded to 6dp
_LO = 0.096558   # sigmoid(-sqrt(5))

_US_P1 = ("quality_dupont", "fcf_yield", "quality_piotroski")
_US_P3 = ("insider_alpha", "congress", "inst_flow_13f")


def _row(region_factors, ticker, market, value=0.7, overrides=None, raw_pio=8):
    row = {
        "ticker": ticker, "sector": "Technology", "cap_tier": "large",
        "market": market, "quality_piotroski_raw": raw_pio,
    }
    for name in region_factors:
        row[f"{name}_score"] = value
    row.update(overrides or {})
    return row


def _us_universe(n=6, overrides_first=None, raw_pio=8):
    names = list(FACTOR_MATRIX_V3["US"].keys())
    rows = [_row(names, f"T{i}", "USA", raw_pio=raw_pio) for i in range(n)]
    if overrides_first:
        rows[0].update(overrides_first)
    return rows


def _eu_universe(n=6, overrides_first=None):
    names = list(FACTOR_MATRIX_V3["EU"].keys())
    rows = [_row(names, f"E{i}", "EUROPE") for i in range(n)]
    if overrides_first:
        rows[0].update(overrides_first)
    return rows


class TestPillarAggregation:
    def test_uniform_universe_scores_half(self):
        result = score_universe_v3(_us_universe(), "US")
        for r in result:
            assert r["pillar_fundamental_score"] == pytest.approx(0.5)
            assert r["pillar_consensus_score"] == pytest.approx(0.5)
            assert r["pillar_alternative_score"] == pytest.approx(0.5)
            assert r["final_score_v3"] == pytest.approx(0.5)
            assert r["weight_coverage_v3"] == pytest.approx(1.0)
            assert r["_low_coverage_v3"] is False

    def test_none_factor_reweights_within_pillar(self):
        rows = _us_universe()
        for r in rows:
            r["inst_flow_13f_score"] = None
        result = score_universe_v3(rows, "US")
        for r in result:
            # P3 reweights over insider(0.30) + congress(0.05), both 0.5
            assert r["pillar_alternative_score"] == pytest.approx(0.5)
            assert r["final_score_v3"] == pytest.approx(0.5)
            assert r["weight_coverage_v3"] == pytest.approx(0.90)

    def test_whole_pillar_missing_reweights_base(self):
        rows = _us_universe()
        for r in rows:
            for name in ("analyst_revision", "pead_surprise", "price_target_upside"):
                r[f"{name}_score"] = None
        result = score_universe_v3(rows, "US")
        for r in result:
            assert r["pillar_consensus_score"] is None
            # base reweights over P1 (0.30) + P3 (0.45), both 0.5
            assert r["final_score_v3"] == pytest.approx(0.5)
            assert r["weight_coverage_v3"] == pytest.approx(0.75)

    def test_low_coverage_flagged(self):
        names = list(FACTOR_MATRIX_V3["US"].keys())
        rows = []
        for i in range(6):
            row = {"ticker": f"T{i}", "sector": "Technology",
                   "cap_tier": "large", "market": "USA",
                   "quality_piotroski_raw": 8}
            for name in names:
                row[f"{name}_score"] = None
            row["quality_piotroski_score"] = 0.7
            rows.append(row)
        result = score_universe_v3(rows, "US")
        for r in result:
            assert r["weight_coverage_v3"] == pytest.approx(0.08)
            assert r["_low_coverage_v3"] is True

    def test_blackout_telemetry(self):
        rows = _us_universe()
        for r in rows:
            r["pead_surprise_score"] = None
        result = score_universe_v3(rows, "US")
        assert result[0]["_factor_blackout"] == ["pead_surprise"]


class TestSurgeInteraction:
    def _surge_universe(self, fund_value_first=0.9):
        # Row 0: P1 + P3 factors at fund_value_first/0.9, rows 1-5 at 0.1 →
        # row 0 neutrals hit sigmoid(±√5); P2 stays uniform 0.7 → 0.5.
        over = {f"{n}_score": fund_value_first for n in _US_P1}
        over.update({f"{n}_score": 0.9 for n in _US_P3})
        rows = _us_universe(overrides_first=over)
        for r in rows[1:]:
            for n in _US_P1 + _US_P3:
                r[f"{n}_score"] = 0.1
        return rows

    def test_corroborated_surge_adds_bonus(self):
        result = score_universe_v3(self._surge_universe(), "US")
        r0 = result[0]
        base = 0.30 * _HI + 0.25 * 0.5 + 0.45 * _HI            # 0.802581
        bonus = 0.5 * (_HI - 0.80) * (_HI - 0.50)               # 0.020867
        assert r0["pillar_alternative_score"] == pytest.approx(_HI, abs=1e-5)
        assert r0["final_score_v3"] == pytest.approx(base + bonus, abs=5e-4)

    def test_uncorroborated_surge_no_penalty(self):
        # Row 0: alt surges but fundamentals are bucket-worst → conf = 0 →
        # final == base exactly (one-sided: no penalty, no bonus).
        over = {f"{n}_score": 0.1 for n in _US_P1}
        over.update({f"{n}_score": 0.9 for n in _US_P3})
        rows = _us_universe(overrides_first=over)
        for r in rows[1:]:
            for n in _US_P1:
                r[f"{n}_score"] = 0.9
            for n in _US_P3:
                r[f"{n}_score"] = 0.1
        result = score_universe_v3(rows, "US")
        r0 = result[0]
        base = 0.30 * _LO + 0.25 * 0.5 + 0.45 * _HI             # 0.560516
        assert r0["final_score_v3"] == pytest.approx(base, abs=5e-4)

    def test_bonus_within_registered_bound(self):
        result = score_universe_v3(self._surge_universe(), "US")
        r0 = result[0]
        base = 0.30 * _HI + 0.25 * 0.5 + 0.45 * _HI
        bonus = r0["final_score_v3"] - base
        assert bonus <= SURGE_MAX_BONUS + 1e-9

    def test_eu_lambda_zero_no_surge(self):
        # Same alt+fund surge shape on EU: final must equal base (λ=0 ex-US).
        eu_p1 = ("quality_piotroski", "fcf_yield", "pb_value_up")
        eu_p3 = ("inst_concentration", "dividend_sustain", "amihud_shock")
        over = {f"{n}_score": 0.9 for n in eu_p1 + eu_p3}
        rows = _eu_universe(overrides_first=over)
        for r in rows[1:]:
            for n in eu_p1 + eu_p3:
                r[f"{n}_score"] = 0.1
        result = score_universe_v3(rows, "EU")
        r0 = result[0]
        base = 0.45 * _HI + 0.35 * 0.5 + 0.20 * _HI
        assert r0["final_score_v3"] == pytest.approx(base, abs=5e-4)


class TestPiotroskiGateOrdering:
    def test_distress_suppression(self):
        result = score_universe_v3(_us_universe(raw_pio=2), "US")
        assert all(r["final_score_v3"] == pytest.approx(0.0) for r in result)

    def test_discount_tier(self):
        result = score_universe_v3(_us_universe(raw_pio=4), "US")
        assert all(r["final_score_v3"] == pytest.approx(0.30) for r in result)

    def test_missing_raw_uses_conservative_sentinel(self):
        result = score_universe_v3(_us_universe(raw_pio=None), "US")
        # 0.5 × (3/8) = 0.1875
        assert all(r["final_score_v3"] == pytest.approx(0.1875) for r in result)

    def test_gate_applied_after_surge(self):
        # Gate multiplies the post-surge clipped score: with raw=4, the
        # corroborated-surge row must score 0.6 × (base + bonus).
        over = {f"{n}_score": 0.9 for n in _US_P1 + _US_P3}
        rows = _us_universe(overrides_first=over, raw_pio=4)
        for r in rows[1:]:
            for n in _US_P1 + _US_P3:
                r[f"{n}_score"] = 0.1
        result = score_universe_v3(rows, "US")
        base = 0.30 * _HI + 0.25 * 0.5 + 0.45 * _HI
        bonus = 0.5 * (_HI - 0.80) * (_HI - 0.50)
        assert result[0]["final_score_v3"] == pytest.approx(
            0.6 * (base + bonus), abs=5e-4)
