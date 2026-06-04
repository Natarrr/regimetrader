# Path: tests/test_global_scoring_v22.py
"""Tests for the v2.2-global patch.

Validates:
  1. WEIGHTS sums and structure
  2. EU/Asia factor availability in market_config
  3. _score_ticker_international reads raw_factors correctly
  4. _build_intl_entry produces correct factors dict
  5. EU/Asia scores are comparable to US scores (not capped at 0.33)
  6. congress always 0.0 for non-US
"""
import pytest
from unittest.mock import MagicMock, patch


# ── 1. WEIGHTS structure ──────────────────────────────────────────────────────

def test_weights_us_unchanged():
    from regime_trader.config.weights import WEIGHTS_US
    assert WEIGHTS_US["congress"] == 0.22
    assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6


def test_weights_global_sum():
    from regime_trader.config.weights import WEIGHTS_GLOBAL
    assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6


def test_weights_global_congress_zero():
    from regime_trader.config.weights import WEIGHTS_GLOBAL
    assert WEIGHTS_GLOBAL["congress"] == 0.0


def test_weights_global_transcript_zero():
    from regime_trader.config.weights import WEIGHTS_GLOBAL
    assert WEIGHTS_GLOBAL.get("transcript_tone", 0.0) == 0.0


def test_weights_global_insider_unchanged():
    """EU/Asia insider signal has same weight as US — FMP MAR Art.19 parity."""
    from regime_trader.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL
    assert WEIGHTS_GLOBAL["insider_conviction"] == WEIGHTS_US["insider_conviction"]


def test_weights_global_analyst_consensus_positive():
    """analyst_consensus should be > 0 in WEIGHTS_GLOBAL — it's now fetched globally."""
    from regime_trader.config.weights import WEIGHTS_GLOBAL
    assert WEIGHTS_GLOBAL.get("analyst_consensus", 0.0) > 0.0


def test_weights_version_is_v22():
    from regime_trader.config.weights import WEIGHTS_VERSION
    assert "v2.2" in WEIGHTS_VERSION


# ── 2. Market config factor availability ─────────────────────────────────────

def test_eu_factors_include_insider():
    from regime_trader.scoring.market_config import MARKET_FACTORS, Market
    assert "insider_conviction_score" in MARKET_FACTORS[Market.EUROPE]
    assert "insider_breadth_score" in MARKET_FACTORS[Market.EUROPE]


def test_eu_factors_include_news():
    from regime_trader.scoring.market_config import MARKET_FACTORS, Market
    assert "news_sentiment_score" in MARKET_FACTORS[Market.EUROPE]
    assert "news_buzz_score" in MARKET_FACTORS[Market.EUROPE]


def test_eu_factors_include_analyst():
    from regime_trader.scoring.market_config import MARKET_FACTORS, Market
    assert "analyst_consensus_score" in MARKET_FACTORS[Market.EUROPE]


def test_eu_factors_exclude_congress():
    from regime_trader.scoring.market_config import MARKET_FACTORS, Market
    assert "congress_score" not in MARKET_FACTORS[Market.EUROPE]
    assert "congress_score" not in MARKET_FACTORS[Market.ASIA]


def test_asia_factors_include_insider():
    from regime_trader.scoring.market_config import MARKET_FACTORS, Market
    assert "insider_conviction_score" in MARKET_FACTORS[Market.ASIA]


def test_market_weight_coverage_eu_high():
    """EU/Asia should now have ~100% weight coverage (only congress+transcript absent)."""
    from regime_trader.scoring.market_config import market_weight_coverage, Market
    from regime_trader.config.weights import WEIGHTS_GLOBAL
    coverage = market_weight_coverage(Market.EUROPE, WEIGHTS_GLOBAL)
    assert coverage > 0.95, f"Expected >95% coverage, got {coverage:.2%}"


def test_renormalize_eu_sums_to_one():
    from regime_trader.scoring.market_config import renormalize_weights_for_market, Market
    from regime_trader.config.weights import WEIGHTS_GLOBAL
    w = renormalize_weights_for_market(WEIGHTS_GLOBAL, Market.EUROPE)
    assert abs(sum(w.values()) - 1.0) < 1e-6


# ── 3. _score_ticker_international reads raw_factors ─────────────────────────

class _MockEntry:
    def __init__(self, ticker, market_str, raw_factors):
        self.ticker = ticker
        self.market = type("M", (), {"value": market_str})()
        self.sector = "Information Technology"
        self.cap_tier = "large"
        self.source_reliability = 0.80
        self.raw_factors = raw_factors


def test_score_intl_reads_all_factors():
    """_score_ticker_international should propagate all raw_factors, not return None."""
    from scripts.run_pipeline import _score_ticker_international

    rf = {
        "momentum_long_score":       0.72,
        "volume_attention_score":    0.30,
        "news_sentiment_score":      0.65,
        "news_buzz_score":           0.40,
        "insider_conviction_score":  0.55,
        "insider_breadth_score":     0.45,
        "analyst_consensus_score":   0.75,
        "analyst_revision_score":    0.60,
        "quality_piotroski_score":   0.78,
        "price_target_upside_score": 0.68,
        "return_12_1m":              0.18,
        "volume_spike":              2.5,
        "market_cap":                50e9,
    }
    entry = _MockEntry("SAP.DE", "EUROPE", rf)
    result = _score_ticker_international(entry, spy_return_baseline=0.10)

    assert result is not None
    assert result["insider_conviction_score"] == pytest.approx(0.55)
    assert result["news_sentiment_score"] == pytest.approx(0.65)
    assert result["analyst_consensus_score"] == pytest.approx(0.75)
    assert result["congress_score"] == 0.0   # always 0.0 for non-US


def test_score_intl_congress_always_zero():
    from scripts.run_pipeline import _score_ticker_international
    rf = {"momentum_long_score": 0.5, "congress_score": 0.9}  # contaminated input
    entry = _MockEntry("7203.T", "ASIA", rf)
    result = _score_ticker_international(entry)
    assert result is not None
    assert result["congress_score"] == 0.0


def test_score_intl_handles_missing_raw_factors():
    """Missing raw_factors keys should default to 0.0, not crash."""
    from scripts.run_pipeline import _score_ticker_international
    rf = {}  # completely empty
    entry = _MockEntry("ASML.AS", "EUROPE", rf)
    result = _score_ticker_international(entry)
    assert result is not None
    assert result["insider_conviction_score"] == 0.0
    assert result["momentum_long_score"] == 0.0


def test_score_intl_returns_float_not_none_for_quality():
    """quality_piotroski_score must be a float (0.0 if absent), not None."""
    from scripts.run_pipeline import _score_ticker_international
    rf = {"return_12_1m": 0.15, "volume_spike": 2.0}
    entry = _MockEntry("SAP.DE", "EUROPE", rf)
    result = _score_ticker_international(entry)
    assert result is not None
    assert isinstance(result["quality_piotroski_score"], float)
    assert result["quality_piotroski_score"] == 0.0  # absent in raw_factors


# ── 4. _build_intl_entry produces correct factors dict ──────────────────────

def test_build_intl_entry_factors():
    from backend.market_intel._generate_top_lists_intl_patch import _build_intl_entry

    row = {
        "ticker": "SAP.DE",
        "market": "EUROPE",
        "sector": "Information Technology",
        "cap_tier": "large",
        "market_cap": 200e9,
        "final_score": 0.71,
        "insider_conviction_score":  0.55,
        "insider_breadth_score":     0.45,
        "news_sentiment_score":      0.65,
        "news_buzz_score":           0.40,
        "momentum_long_score":       0.72,
        "volume_attention_score":    0.30,
        "analyst_consensus_score":   0.75,
        "analyst_revision_score":    0.60,
        "quality_piotroski_score":   0.78,
        "price_target_upside_score": 0.68,
        "congress_score":            0.0,
    }
    entry = _build_intl_entry(row)

    assert entry["factors"]["insider_conviction"] == pytest.approx(0.55)
    assert entry["factors"]["news_sentiment"] == pytest.approx(0.65)
    assert entry["factors"]["analyst_consensus"] == pytest.approx(0.75)
    assert entry["factors"]["congress"] == 0.0
    assert entry["weights_set"] == "GLOBAL"
    assert entry["region"] == "EU"


def test_build_intl_entry_no_hardcoded_zeros():
    """Non-congress/transcript factors should reflect actual values, not 0.0."""
    from backend.market_intel._generate_top_lists_intl_patch import _build_intl_entry

    row = {
        "ticker": "005930.KS",
        "market": "ASIA",
        "insider_conviction_score": 0.42,
        "news_sentiment_score": 0.58,
        "analyst_consensus_score": 0.70,
    }
    entry = _build_intl_entry(row)

    assert entry["factors"]["insider_conviction"] == pytest.approx(0.42)
    assert entry["factors"]["news_sentiment"] == pytest.approx(0.58)
    assert entry["factors"]["analyst_consensus"] == pytest.approx(0.70)
    assert entry["region"] == "ASIA"


# ── 5. EU/Asia score is now competitive with US ───────────────────────────────

def test_eu_score_uses_global_weights():
    """A strong EU ticker should reach TACTICAL BUY (>=0.55) with v2.2 weights."""
    from backend.market_intel._score_compositor import compute_composite_score

    factors = {
        "insider_conviction": 0.60,
        "insider_breadth":    0.50,
        "congress":           0.00,
        "news_sentiment":     0.70,
        "news_buzz":          0.45,
        "momentum_long":      0.75,
        "volume_attention":   0.35,
        "analyst_consensus":  0.80,
        "quality_piotroski":  0.75,
    }
    score, meta = compute_composite_score("SAP.DE", factors, piotroski_raw=7)

    assert meta["weights_set"] == "GLOBAL"
    assert meta["region"] == "EU"
    assert score >= 0.55, f"Expected score >= 0.55, got {score:.4f}"


def test_eu_score_vs_old_penalty():
    """EU score with WEIGHTS_GLOBAL must exceed naive WEIGHTS_US score (dead congress weight)."""
    from backend.market_intel._score_compositor import compute_composite_score
    from regime_trader.config.weights import WEIGHTS_US

    factors = {k: 0.7 for k in WEIGHTS_US}
    factors["congress"] = 0.0

    score_new, _ = compute_composite_score("SAP.DE", factors)
    score_old = sum(WEIGHTS_US[f] * factors.get(f, 0.0) for f in WEIGHTS_US)

    assert score_new > score_old + 0.05, (
        f"New score {score_new:.4f} should be >0.05 higher than old {score_old:.4f}"
    )


# ── 6. congress contamination guard still works ───────────────────────────────

def test_canary_contamination_guard():
    """Even if upstream passes congress > 0 for EU/Asia, compositor blocks it."""
    from backend.market_intel._score_compositor import compute_composite_score

    contaminated = {k: 0.5 for k in ["insider_conviction", "insider_breadth",
                                       "congress", "news_sentiment", "news_buzz",
                                       "momentum_long", "volume_attention",
                                       "analyst_consensus", "quality_piotroski"]}
    contaminated["congress"] = 0.8

    score, meta = compute_composite_score("SAP.DE", contaminated)
    assert meta["congress_masked"] is True


# ── 7. FMPFetcher source reliability updated ──────────────────────────────────

def test_fmp_fetcher_source_reliability():
    from regime_trader.fetchers.fmp_fetcher import FMPFetcher
    from regime_trader.fetchers.base import MarketEnum

    eu = FMPFetcher(api_key="k", market=MarketEnum.EUROPE)
    asia = FMPFetcher(api_key="k", market=MarketEnum.ASIA)

    assert eu.source_reliability("SAP.DE") >= 0.75
    assert asia.source_reliability("7203.T") >= 0.65
