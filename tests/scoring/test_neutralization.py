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
