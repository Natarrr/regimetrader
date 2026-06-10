"""tests/scoring/test_neutralization.py
Unit tests for cross-sectional factor neutralization.

Grinold & Kahn ch. 7: IC is only meaningful after removing common-factor
exposures.  These tests verify that sector bias is removed and that the
fallback chain (sector×cap_tier → cap_tier → raw) works correctly.
"""
from __future__ import annotations

import pytest
from src.scoring.neutralization import neutralize_factors


def _make_ticker(
    ticker: str,
    sector: str,
    cap_tier: str,
    congress_score: float,
    market: str = "USA",
) -> dict:
    return {
        "ticker": ticker,
        "sector": sector,
        "cap_tier": cap_tier,
        "market": market,
        "edgar_score": 0.5,
        "insider_score": 0.5,
        "congress_score": congress_score,
        "news_score": 0.5,
        "momentum_score": 0.5,
    }


class TestNeutralizationRemovesSectorBias:
    """
    Congress score: Defense mean=0.8, Tech mean=0.3.
    After neutralization, both sector means should collapse to ~0.5.
    Intra-bucket ranking must be preserved.
    """

    def _build_fixture(self) -> list[dict]:
        rows = []
        # 10 Defense tickers: congress scores 0.70–0.90
        for i in range(10):
            score = 0.70 + i * 0.02  # 0.70, 0.72, ..., 0.88
            rows.append(_make_ticker(f"DEF{i}", "Defense", "large", score))
        # 10 Tech tickers: congress scores 0.20–0.40
        for i in range(10):
            score = 0.20 + i * 0.02  # 0.20, 0.22, ..., 0.38
            rows.append(_make_ticker(f"TECH{i}", "Technology", "large", score))
        return rows

    def test_sector_means_collapse_to_neutral(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)

        defense_neutral = [
            r["congress_score_neutral"]
            for r in result
            if r["sector"] == "Defense"
        ]
        tech_neutral = [
            r["congress_score_neutral"]
            for r in result
            if r["sector"] == "Technology"
        ]

        defense_mean = sum(defense_neutral) / len(defense_neutral)
        tech_mean = sum(tech_neutral) / len(tech_neutral)

        # Both sector means should converge near 0.5 (sigmoid(0) = 0.5)
        assert abs(defense_mean - 0.5) < 0.05, (
            f"Defense mean {defense_mean:.3f} should be ~0.5 after neutralization"
        )
        assert abs(tech_mean - 0.5) < 0.05, (
            f"Tech mean {tech_mean:.3f} should be ~0.5 after neutralization"
        )

    def test_intra_bucket_ranking_preserved(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)

        # Within Defense bucket: higher raw score → higher neutral score
        defense = sorted(
            [r for r in result if r["sector"] == "Defense"],
            key=lambda r: r["congress_score"],
        )
        neutral_vals = [r["congress_score_neutral"] for r in defense]
        assert neutral_vals == sorted(neutral_vals), (
            "Intra-bucket ranking must be preserved (monotone sigmoid)"
        )

    def test_original_scores_preserved(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)
        for original, processed in zip(rows, result):
            assert processed["congress_score"] == original["congress_score"], (
                "Original score keys must not be mutated"
            )

    def test_primary_fallback_flag_set(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)
        for r in result:
            assert r["_neutralization_fallback"] == "sector_cap_tier", (
                f"{r['ticker']}: expected sector_cap_tier, got {r['_neutralization_fallback']}"
            )


class TestNeutralizationFallbackForSmallBucket:
    """
    (Energy, small) has only 2 tickers → fallback to cap_tier.
    Large buckets (>=5) must still receive sector_cap_tier treatment.
    """

    def _build_fixture(self) -> list[dict]:
        rows = []
        # 8 Tech large tickers — primary bucket (>=5)
        for i in range(8):
            rows.append(_make_ticker(f"TECH{i}", "Technology", "large", 0.4 + i * 0.05))
        # 8 Defense large tickers — primary bucket (>=5)
        for i in range(8):
            rows.append(_make_ticker(f"DEF{i}", "Defense", "large", 0.6 + i * 0.03))
        # 2 Energy small tickers — too small for primary, falls back to cap_tier
        rows.append(_make_ticker("ENRG0", "Energy", "small", 0.55))
        rows.append(_make_ticker("ENRG1", "Energy", "small", 0.65))
        return rows

    def test_small_bucket_gets_cap_tier_fallback(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)

        energy_small = [
            r for r in result
            if r["sector"] == "Energy" and r["cap_tier"] == "small"
        ]
        assert len(energy_small) == 2
        for r in energy_small:
            assert r["_neutralization_fallback"] in ("cap_tier", "raw"), (
                f"Energy/small should fall back to cap_tier or raw, got {r['_neutralization_fallback']}"
            )

    def test_large_buckets_get_primary_treatment(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)

        large_buckets = [
            r for r in result
            if r["cap_tier"] == "large"
        ]
        for r in large_buckets:
            assert r["_neutralization_fallback"] == "sector_cap_tier", (
                f"{r['ticker']}: large bucket should get sector_cap_tier, "
                f"got {r['_neutralization_fallback']}"
            )

    def test_neutralized_column_always_present(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)
        for r in result:
            assert "congress_score_neutral" in r, (
                f"{r['ticker']} missing congress_score_neutral"
            )

    def test_zero_scores_preserved_as_zero(self):
        rows = self._build_fixture()
        # Force one ticker to have congress_score=0.0
        rows[0]["congress_score"] = 0.0
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)
        zero_row = next(r for r in result if r["ticker"] == "TECH0")
        assert zero_row["congress_score_neutral"] == 0.0
        assert zero_row["_neutralization_fallback"] == "zero"

    def test_neutral_scores_bounded(self):
        rows = self._build_fixture()
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)
        for r in result:
            val = r["congress_score_neutral"]
            assert 0.0 <= val <= 1.0, (
                f"{r['ticker']}: neutral score {val} out of [0, 1]"
            )

    def test_markets_isolated(self):
        """EU tickers in same sector/cap_tier as US must NOT share a bucket."""
        rows = self._build_fixture()
        # Add 5 EU tickers in same sector/cap_tier as US Tech large
        for i in range(5):
            rows.append(_make_ticker(f"EU{i}", "Technology", "large", 0.9, market="EUROPE"))
        result = neutralize_factors(rows, factors=("congress_score",), min_bucket_size=5)

        us_tech_large = [
            r for r in result
            if r["sector"] == "Technology" and r["cap_tier"] == "large" and r.get("market", "USA") == "USA"
        ]
        eu_tech_large = [
            r for r in result
            if r["sector"] == "Technology" and r["cap_tier"] == "large" and r.get("market") == "EUROPE"
        ]
        # EU tickers all have score 0.9 — within their own bucket they should all be equal (~0.5)
        eu_neutral = [r["congress_score_neutral"] for r in eu_tech_large]
        if len(eu_neutral) > 1:
            assert all(abs(v - eu_neutral[0]) < 0.01 for v in eu_neutral), (
                "EU tickers with identical scores should all map to ~0.5 (std=0 bucket)"
            )
        # US Tech bucket must not be influenced by EU tickers
        us_neutral = [r["congress_score_neutral"] for r in us_tech_large]
        assert max(us_neutral) < 0.99, "US bucket should show spread, not all at ceiling"


class TestV3MissingDataSemantics:
    """v3.0: None passthrough (opt-in) + per-factor zero_is_dead flags.

    None = data unavailable (excluded from stats AND output None);
    0.0 on a signed factor = real observation entering bucket stats.
    Default arguments preserve byte-identical v2.2 behavior.
    """

    def _bucket(self, scores, factor="pead_score", market="USA"):
        return [
            {"ticker": f"T{i}", "sector": "Technology", "cap_tier": "large",
             "market": market, factor: s}
            for i, s in enumerate(scores)
        ]

    def test_none_passes_through_as_none(self):
        rows = self._bucket([0.6, 0.7, 0.8, 0.5, 0.4, None])
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
            none_passthrough=True,
        )
        none_row = next(r for r in result if r["ticker"] == "T5")
        assert none_row["pead_score_neutral"] is None
        assert none_row["_neutralization_fallback"] == "none"

    def test_none_excluded_from_bucket_stats(self):
        # Adding a None row must not perturb the other rows' neutral scores.
        base_scores = [0.6, 0.7, 0.8, 0.5, 0.4]
        without = neutralize_factors(
            self._bucket(base_scores), factors=("pead_score",),
            min_bucket_size=5, none_passthrough=True,
        )
        with_none = neutralize_factors(
            self._bucket(base_scores + [None]), factors=("pead_score",),
            min_bucket_size=5, none_passthrough=True,
        )
        for i in range(5):
            assert (with_none[i]["pead_score_neutral"]
                    == without[i]["pead_score_neutral"]), f"row {i} perturbed"

    def test_default_keeps_v22_none_coercion(self):
        # Without none_passthrough, None must still coerce to dead 0.0 —
        # v2.2 callers stay byte-identical until cutover.
        rows = self._bucket([0.6, 0.7, 0.8, 0.5, 0.4, None])
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
        )
        none_row = next(r for r in result if r["ticker"] == "T5")
        assert none_row["pead_score_neutral"] == 0.0
        assert none_row["_neutralization_fallback"] == "zero"

    def test_signed_zero_enters_bucket_stats(self):
        # zero_is_dead=False: a true 0.0 is a real (worst-in-bucket)
        # observation — sigmoid(z) output, never the dead-zero floor.
        rows = self._bucket([0.0, 0.2, 0.4, 0.6, 0.8])
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
            zero_is_dead={"pead_score": False},
        )
        zero_row = next(r for r in result if r["ticker"] == "T0")
        assert zero_row["_neutralization_fallback"] == "sector_cap_tier"
        assert 0.01 <= zero_row["pead_score_neutral"] < 0.5
        neutrals = [r["pead_score_neutral"] for r in result]
        assert neutrals == sorted(neutrals)  # ordering preserved

    def test_unsigned_zero_still_dead_by_default(self):
        rows = self._bucket([0.0, 0.2, 0.4, 0.6, 0.8])
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
        )
        zero_row = next(r for r in result if r["ticker"] == "T0")
        assert zero_row["pead_score_neutral"] == 0.0
        assert zero_row["_neutralization_fallback"] == "zero"

    def test_all_none_bucket_no_errors(self):
        rows = self._bucket([None] * 6)
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
            none_passthrough=True,
        )
        assert all(r["pead_score_neutral"] is None for r in result)

    def test_all_dead_bucket_no_errors(self):
        rows = self._bucket([0.0] * 6)
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
        )
        assert all(r["pead_score_neutral"] == 0.0 for r in result)

    def test_identical_values_neutral_despite_float_noise(self):
        # 6 × 0.7 doesn't sum exactly in binary floating point, leaving
        # std ≈ 2e-16 — the exact `std == 0.0` check missed it and amplified
        # noise/noise into z ≈ 1 (sigmoid 0.731). Identical buckets must
        # always map to 0.5 (documented degenerate case).
        rows = self._bucket([0.7] * 6)
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
        )
        for r in result:
            assert r["pead_score_neutral"] == pytest.approx(0.5)

    def test_four_identical_plus_one_outlier(self):
        # Population-std bound: |z| ≤ √(n−1) → n=5 gives z_outlier = 2.0
        # (sigmoid ≈ 0.8808), the 4 identical names sit at z = −0.5
        # (sigmoid ≈ 0.3775). Compressed spread, order preserved, no blowup.
        rows = self._bucket([0.5, 0.5, 0.5, 0.5, 0.9])
        result = neutralize_factors(
            rows, factors=("pead_score",), min_bucket_size=5,
            zero_is_dead={"pead_score": False},
        )
        outlier = next(r for r in result if r["ticker"] == "T4")
        others = [r for r in result if r["ticker"] != "T4"]
        assert outlier["pead_score_neutral"] == pytest.approx(0.8808, abs=1e-3)
        for r in others:
            assert r["pead_score_neutral"] == pytest.approx(0.3775, abs=1e-3)
        assert all(0.01 <= r["pead_score_neutral"] <= 0.99 for r in result)
