"""tests/scoring/test_market_parity.py — Market parity validation.

Tests:
  1. renormalize_weights_for_market produces valid per-market weights (sum=1.0,
     congress=0.0 for EU/Asia, unchanged for US).
  2. MARKET_FACTORS contents

No live network calls. No yfinance I/O.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scoring.market_config import (
    Market,
    MARKET_FACTORS,
    renormalize_weights_for_market,
    LOW_COVERAGE_THRESHOLD,
    market_weight_coverage,
)


# ── Test 1: weight renormalization ───────────────────────────────────────────

# Matches WEIGHTS in scripts/run_pipeline.py exactly.
_BASE_WEIGHTS = {
    "insider_conviction":  0.30,
    "insider_breadth":     0.15,
    "congress":            0.22,
    "news_sentiment":      0.10,
    "news_buzz":           0.05,
    "momentum_long":       0.15,
    "volume_attention":    0.03,
}


class TestWeightsRenormalize:
    def test_us_weights_unchanged(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.US)
        for k, v in _BASE_WEIGHTS.items():
            assert result[k] == pytest.approx(v, abs=1e-6), (
                f"US weight for '{k}' changed: expected {v}, got {result[k]}"
            )

    def test_us_sums_to_one(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.US)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_europe_congress_is_zero(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["congress"] == 0.0

    def test_europe_insider_conviction_positive(self):
        """v2.2-global: FMP insider-trading/search confirmed live for EU (MAR Art.19)."""
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["insider_conviction"] > 0.0

    def test_europe_insider_breadth_positive(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["insider_breadth"] > 0.0

    def test_europe_news_sentiment_positive(self):
        """v2.2-global: FMP news/stock confirmed live for EU."""
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["news_sentiment"] > 0.0

    def test_europe_news_buzz_positive(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["news_buzz"] > 0.0

    def test_europe_sums_to_one(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_europe_momentum_long_positive(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["momentum_long"] > 0.0

    def test_europe_volume_attention_positive(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["volume_attention"] > 0.0

    def test_europe_momentum_volume_ratio_preserved(self):
        """Relative ratio of momentum_long : volume_attention must equal base ratio."""
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        base_ratio = _BASE_WEIGHTS["momentum_long"] / _BASE_WEIGHTS["volume_attention"]
        result_ratio = result["momentum_long"] / result["volume_attention"]
        assert result_ratio == pytest.approx(base_ratio, rel=1e-5)

    def test_asia_congress_is_zero(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.ASIA)
        assert result["congress"] == 0.0

    def test_asia_sums_to_one(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.ASIA)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_markets_sum_to_one(self):
        for market in Market:
            result = renormalize_weights_for_market(_BASE_WEIGHTS, market)
            total = sum(result.values())
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Market {market.value} weights sum to {total}, expected 1.0"
            )

    def test_market_weight_coverage_eu_below_one(self):
        """EU covers all factors except congress — coverage ≈ 0.78 with _BASE_WEIGHTS."""
        cov = market_weight_coverage(Market.EUROPE, _BASE_WEIGHTS)
        assert cov < 1.0
        assert cov > 0.5   # v2.2: EU now covers 6/7 factors (congress absent)

    def test_market_weight_coverage_us_is_one(self):
        cov = market_weight_coverage(Market.US, _BASE_WEIGHTS)
        assert cov == pytest.approx(1.0, abs=1e-6)

    def test_eu_coverage_excludes_only_congress(self):
        """v2.2: EU coverage = total weight minus congress weight."""
        expected = sum(v for k, v in _BASE_WEIGHTS.items() if k != "congress")
        cov = market_weight_coverage(Market.EUROPE, _BASE_WEIGHTS)
        assert cov == pytest.approx(expected, abs=1e-6)


# ── Test 2: MARKET_FACTORS contents ─────────────────────────────────────────

class TestMarketFactors:
    def test_us_has_all_twelve_factors(self):
        assert len(MARKET_FACTORS[Market.US]) == 12

    def test_europe_does_not_have_congress(self):
        assert "congress_score" not in MARKET_FACTORS[Market.EUROPE]

    def test_asia_does_not_have_congress(self):
        assert "congress_score" not in MARKET_FACTORS[Market.ASIA]

    def test_europe_has_momentum_and_volume(self):
        assert "momentum_long_score" in MARKET_FACTORS[Market.EUROPE]
        assert "volume_attention_score" in MARKET_FACTORS[Market.EUROPE]

    def test_europe_has_exactly_two_base_factors(self):
        """Momentum and volume are always present (pre-PATCH 07 baseline)."""
        assert "momentum_long_score" in MARKET_FACTORS[Market.EUROPE]
        assert "volume_attention_score" in MARKET_FACTORS[Market.EUROPE]

    def test_asia_has_exactly_two_base_factors(self):
        """Momentum and volume are always present (pre-PATCH 07 baseline)."""
        assert "momentum_long_score" in MARKET_FACTORS[Market.ASIA]
        assert "volume_attention_score" in MARKET_FACTORS[Market.ASIA]

    def test_europe_has_ten_factors(self):
        """v2.2-global: EU now has 10 factors (FMP Ultimate confirmed globally)."""
        assert len(MARKET_FACTORS[Market.EUROPE]) == 10

    def test_asia_has_ten_factors(self):
        """v2.2-global: Asia now has 10 factors (FMP Ultimate confirmed globally)."""
        assert len(MARKET_FACTORS[Market.ASIA]) == 10

    def test_europe_has_quality_piotroski(self):
        """PATCH 07: FMP ratios-ttm confirmed PASS for SAP.DE (Phase-0 2026-05-30)."""
        assert "quality_piotroski_score" in MARKET_FACTORS[Market.EUROPE]

    def test_asia_has_quality_piotroski(self):
        """PATCH 07: FMP ratios-ttm confirmed PASS for 7203.T (Phase-0 2026-05-30)."""
        assert "quality_piotroski_score" in MARKET_FACTORS[Market.ASIA]

    def test_europe_has_price_target_upside(self):
        """PATCH 07: partial FMP price-target-consensus coverage for EU."""
        assert "price_target_upside_score" in MARKET_FACTORS[Market.EUROPE]

    def test_asia_has_price_target_upside(self):
        """PATCH 07: partial FMP price-target-consensus coverage for Asia."""
        assert "price_target_upside_score" in MARKET_FACTORS[Market.ASIA]

    def test_us_has_twelve_factors(self):
        """PATCH 07: US now has 12 factors (added analyst + quality + transcript)."""
        assert len(MARKET_FACTORS[Market.US]) == 12

    def test_low_coverage_threshold_is_sane(self):
        """LOW_COVERAGE_THRESHOLD must be in (0, 1) — boundary sanity check."""
        assert 0.0 < LOW_COVERAGE_THRESHOLD < 1.0, (
            f"LOW_COVERAGE_THRESHOLD must be in (0, 1), got {LOW_COVERAGE_THRESHOLD}"
        )
        # v2.2: EU renormalized weight for volume_attention is small (~0.04)
        # relative to threshold of 0.50 — correctly flagged as low_coverage
        eu_renorm = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        eu_volume_only = eu_renorm["volume_attention"]
        assert eu_volume_only < LOW_COVERAGE_THRESHOLD, (
            f"A EU ticker with only volume_attention (weight={eu_volume_only:.3f}) "
            f"should be flagged _low_coverage (threshold={LOW_COVERAGE_THRESHOLD})"
        )
