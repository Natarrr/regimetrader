"""tests/scoring/test_market_parity.py — Fix #5 market parity validation.

Two tests verifying that:
  1. renormalize_weights_for_market produces valid per-market weights (sum=1.0,
     congress=0.0 for EU/Asia, unchanged for US).
  2. _score_ticker_international correctly handles a EU ticker with partially
     missing data: momentum computed, insider=None, news=None, congress=None.

No live network calls. No yfinance I/O.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from regime_trader.scoring.market_config import (
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

    def test_europe_insider_conviction_is_zero(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["insider_conviction"] == 0.0

    def test_europe_insider_breadth_is_zero(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["insider_breadth"] == 0.0

    def test_europe_news_sentiment_is_zero(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["news_sentiment"] == 0.0

    def test_europe_news_buzz_is_zero(self):
        result = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        assert result["news_buzz"] == 0.0

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
        """EU covers only 2/7 factors — coverage fraction must be < 1.0."""
        cov = market_weight_coverage(Market.EUROPE, _BASE_WEIGHTS)
        assert cov < 1.0
        assert cov > 0.0

    def test_market_weight_coverage_us_is_one(self):
        cov = market_weight_coverage(Market.US, _BASE_WEIGHTS)
        assert cov == pytest.approx(1.0, abs=1e-6)

    def test_eu_coverage_equals_momentum_plus_volume(self):
        """EU coverage = weight(momentum_long) + weight(volume_attention)."""
        expected = _BASE_WEIGHTS["momentum_long"] + _BASE_WEIGHTS["volume_attention"]
        cov = market_weight_coverage(Market.EUROPE, _BASE_WEIGHTS)
        assert cov == pytest.approx(expected, abs=1e-6)


# ── Test 2: _score_ticker_international with missing data ────────────────────

def _make_eu_entry(return_12_1m=0.15, volume_spike=None):
    """Construct a mock TickerEntry for a EU ticker."""
    from regime_trader.fetchers.base import MarketEnum, TickerEntry
    return TickerEntry(
        ticker="SAP.DE",
        market=MarketEnum.EUROPE,
        sector="Information Technology",
        cap_tier="large",
        source_reliability=0.60,
        raw_factors={
            "return_12_1m": return_12_1m,
            "volume_spike":  volume_spike if volume_spike is not None else 1.5,
            "company_name":  "SAP SE",
        },
    )


class TestInternationalScorer:
    def test_returns_dict_on_valid_entry(self):
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry(return_12_1m=0.15, volume_spike=2.0)
        result = _score_ticker_international(entry, spy_return_baseline=0.05)
        assert isinstance(result, dict)
        assert result["ticker"] == "SAP.DE"
        assert result["market"] == "EUROPE"

    def test_momentum_long_is_computed_not_none(self):
        """return_12_1m=0.15 → momentum_long_score is a float, not None."""
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry(return_12_1m=0.15, volume_spike=2.0)
        result = _score_ticker_international(entry, spy_return_baseline=0.05)
        assert result["momentum_long_score"] is not None
        assert isinstance(result["momentum_long_score"], float)
        assert 0.0 <= result["momentum_long_score"] <= 1.0

    def test_momentum_long_zero_when_no_history(self):
        """return_12_1m=None (insufficient history) → momentum_long_score=0.0 (dead signal)."""
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry(return_12_1m=None, volume_spike=1.0)
        result = _score_ticker_international(entry, spy_return_baseline=0.0)
        assert result["momentum_long_score"] == 0.0

    def test_congress_score_is_none(self):
        """congress_score must be None (structurally absent for EU)."""
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry()
        result = _score_ticker_international(entry)
        assert result["congress_score"] is None

    def test_insider_conviction_is_none(self):
        """insider_conviction_score must be None (FMP 403 for EU — structurally absent)."""
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry()
        result = _score_ticker_international(entry)
        assert result["insider_conviction_score"] is None

    def test_insider_breadth_is_none(self):
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry()
        result = _score_ticker_international(entry)
        assert result["insider_breadth_score"] is None

    def test_news_sentiment_is_none(self):
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry()
        result = _score_ticker_international(entry)
        assert result["news_sentiment_score"] is None

    def test_news_buzz_is_none(self):
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry()
        result = _score_ticker_international(entry)
        assert result["news_buzz_score"] is None

    def test_volume_attention_is_float(self):
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry(volume_spike=3.0)
        result = _score_ticker_international(entry)
        assert isinstance(result["volume_attention_score"], float)
        assert 0.0 <= result["volume_attention_score"] <= 1.0

    def test_source_reliability_is_metadata_not_multiplier(self):
        """source_reliability is preserved in output but does NOT multiply the score."""
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry(return_12_1m=0.30, volume_spike=2.0)
        result = _score_ticker_international(entry)
        # source_reliability should be in the dict as metadata
        assert "source_reliability" in result
        assert result["source_reliability"] == pytest.approx(0.60)
        # final_score is NOT set by this function — computed downstream
        assert "final_score" not in result

    def test_no_final_score_set(self):
        """_score_ticker_international must not set final_score — delegated to pipeline."""
        from scripts.run_pipeline import _score_ticker_international
        entry = _make_eu_entry()
        result = _score_ticker_international(entry)
        assert "final_score" not in result, (
            "final_score must be computed downstream after cross-sectional normalization, "
            "not inside _score_ticker_international"
        )

    def test_asia_entry_works(self):
        from regime_trader.fetchers.base import MarketEnum, TickerEntry
        from scripts.run_pipeline import _score_ticker_international
        entry = TickerEntry(
            ticker="7203.T",
            market=MarketEnum.ASIA,
            sector="Consumer Discretionary",
            cap_tier="large",
            source_reliability=0.60,
            raw_factors={"return_12_1m": 0.268, "volume_spike": 0.67, "company_name": "Toyota"},
        )
        result = _score_ticker_international(entry, spy_return_baseline=0.05)
        assert result["market"] == "ASIA"
        assert result["congress_score"] is None
        assert result["momentum_long_score"] is not None

    def test_returns_none_on_exception(self):
        """Entry with broken raw_factors must return None, not raise."""
        from scripts.run_pipeline import _score_ticker_international
        from regime_trader.fetchers.base import MarketEnum, TickerEntry
        entry = TickerEntry(
            ticker="BAD.DE",
            market=MarketEnum.EUROPE,
            sector="Unknown",
            cap_tier="large",
            source_reliability=0.60,
            raw_factors=None,  # broken — will cause AttributeError on .get()
        )
        # Should not raise; returns None
        result = _score_ticker_international(entry)
        assert result is None


# ── Test 3: MARKET_FACTORS contents ─────────────────────────────────────────

class TestMarketFactors:
    def test_us_has_all_seven_factors(self):
        assert len(MARKET_FACTORS[Market.US]) == 7

    def test_europe_does_not_have_congress(self):
        assert "congress_score" not in MARKET_FACTORS[Market.EUROPE]

    def test_asia_does_not_have_congress(self):
        assert "congress_score" not in MARKET_FACTORS[Market.ASIA]

    def test_europe_has_momentum_and_volume(self):
        assert "momentum_long_score" in MARKET_FACTORS[Market.EUROPE]
        assert "volume_attention_score" in MARKET_FACTORS[Market.EUROPE]

    def test_europe_has_exactly_two_factors(self):
        assert len(MARKET_FACTORS[Market.EUROPE]) == 2

    def test_asia_has_exactly_two_factors(self):
        assert len(MARKET_FACTORS[Market.ASIA]) == 2

    def test_low_coverage_threshold_is_sane(self):
        """LOW_COVERAGE_THRESHOLD is applied in the RENORMALIZED weight space (sum=1.0).

        A EU ticker with both momentum_long + volume_attention present will have
        weight_coverage ≈ 1.0 in the renormalized space (all available factors computed).
        The threshold 0.50 correctly marks a EU ticker where momentum_long is missing
        (renormalized weight ~0.833) as _low_coverage.

        This test checks that the threshold is below 1.0 (so some EU tickers pass)
        and above 0.0 (so it filters something meaningful).
        """
        assert 0.0 < LOW_COVERAGE_THRESHOLD < 1.0, (
            f"LOW_COVERAGE_THRESHOLD must be in (0, 1), got {LOW_COVERAGE_THRESHOLD}"
        )
        # Specifically: a EU ticker with both factors present gets weight_coverage ~1.0
        # in the renormalized space, which must be >= threshold.
        eu_renorm = renormalize_weights_for_market(_BASE_WEIGHTS, Market.EUROPE)
        # Weight of momentum_long in renormalized EU weights (~0.833)
        eu_momentum_weight = eu_renorm["momentum_long"]
        # If only volume_attention is present (momentum missing), weight_coverage = ~0.167
        # That should be < LOW_COVERAGE_THRESHOLD (correctly flagged as low_coverage)
        eu_volume_only = eu_renorm["volume_attention"]
        assert eu_volume_only < LOW_COVERAGE_THRESHOLD, (
            f"A EU ticker with only volume_attention (weight={eu_volume_only:.3f}) "
            f"should be flagged _low_coverage (threshold={LOW_COVERAGE_THRESHOLD})"
        )
