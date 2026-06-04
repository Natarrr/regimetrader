# Path: regime_trader/scoring/market_config.py
"""Per-market factor availability and weight renormalization.

PATCH v2.2-global (2026-06):
Previous version marked insider/news/analyst as "STRUCTURALLY ABSENT" for
EU/Asia based on Phase-0 smoke tests against retired /api/v3/ routes.
FMP Ultimate stable/ routes confirmed globally available (2026-06-03):

  EU/Asia CONFIRMED LIVE (stable/):
    insider-trading/search        — MAR Art.19 (EU) / EDINET partial (JP)
    news/stock                    — global news corpus
    upgrades-downgrades-consensus-bulk — global analyst coverage
    analyst-estimates             — global (partial coverage mid/small)
    ratios-ttm                    — global (accounting-identity based)
    price-target-consensus        — global large-cap coverage
    historical-price-eod/full     — all listed securities

  CONFIRMED ABSENT (structural — no global equivalent):
    congress_score: US STOCK Act / S3 Stock Watcher only
    transcript_tone_score: FMP earning-call-transcript-latest US-only
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet

# Minimum tickers required for cross-sectional z-score neutralization.
MIN_BUCKET_SIZE: int = 5

# If a ticker's final_score is computed from less than this fraction of the
# market's total available weights, it is marked _low_coverage=True.
LOW_COVERAGE_THRESHOLD: float = 0.50


class Market(str, Enum):
    US = "US"
    EUROPE = "EUROPE"
    ASIA = "ASIA"


PIPELINE_MARKET_MAP: Dict[str, Market] = {
    "USA":    Market.US,
    "US":     Market.US,
    "EUROPE": Market.EUROPE,
    "ASIA":   Market.ASIA,
}

# ── Factor availability per market ───────────────────────────────────────────
#
# CHANGE LOG vs previous version:
#   EUROPE/ASIA: added insider_conviction_score, insider_breadth_score,
#                news_sentiment_score, news_buzz_score,
#                analyst_consensus_score, analyst_revision_score,
#                price_target_upside_score
#   EUROPE/ASIA: removed congress_score (US STOCK Act only)
#   EUROPE/ASIA: removed transcript_tone_score (FMP earning-call US-only)

MARKET_FACTORS: Dict[Market, FrozenSet[str]] = {
    Market.US: frozenset({
        "insider_conviction_score",
        "insider_breadth_score",
        "congress_score",            # US STOCK Act — US only
        "news_sentiment_score",
        "news_buzz_score",
        "momentum_long_score",
        "volume_attention_score",
        "analyst_consensus_score",
        "analyst_revision_score",
        "price_target_upside_score",
        "quality_piotroski_score",
        "transcript_tone_score",     # FMP transcripts — US only
    }),
    Market.EUROPE: frozenset({
        # ── Available via FMP Ultimate stable/ ───────────────────────────
        "insider_conviction_score",   # MAR Art.19 mandatory disclosures
        "insider_breadth_score",      # same source
        "news_sentiment_score",       # news/stock confirmed live for EU
        "news_buzz_score",            # same source
        "momentum_long_score",        # historical-price-eod/full ✓
        "volume_attention_score",     # same source
        "analyst_consensus_score",    # upgrades-downgrades-consensus-bulk ✓
        "analyst_revision_score",     # analyst-estimates ✓ (partial coverage)
        "price_target_upside_score",  # price-target-consensus ✓
        "quality_piotroski_score",    # ratios-ttm ✓
        # ── Structurally absent ──────────────────────────────────────────
        # congress_score:       no STOCK Act equivalent in EU
        # transcript_tone_score: FMP earning-call-transcript-latest US-only
    }),
    Market.ASIA: frozenset({
        # ── Available via FMP Ultimate stable/ ───────────────────────────
        "insider_conviction_score",   # EDINET (JP) partial, KRX partial, HKEX partial
        "insider_breadth_score",      # same source
        "news_sentiment_score",       # news/stock confirmed live for Asia
        "news_buzz_score",            # same source
        "momentum_long_score",        # historical-price-eod/full ✓
        "volume_attention_score",     # same source
        "analyst_consensus_score",    # upgrades-downgrades-consensus-bulk ✓
        "analyst_revision_score",     # analyst-estimates ✓ (partial coverage)
        "price_target_upside_score",  # price-target-consensus ✓
        "quality_piotroski_score",    # ratios-ttm ✓
        # ── Structurally absent ──────────────────────────────────────────
        # congress_score:       no STOCK Act equivalent in Asia
        # transcript_tone_score: FMP earning-call-transcript-latest US-only
    }),
}


def renormalize_weights_for_market(
    base_weights: Dict[str, float],
    market: Market,
) -> Dict[str, float]:
    """Renormalize base_weights for factors available in this market.

    Unavailable factors get weight 0.0. Available factors keep their
    relative proportions from base_weights, scaled to sum to 1.0.

    With WEIGHTS_GLOBAL (congress=0.0 already), EU/Asia renormalization
    has near-100% coverage — only transcript_tone (weight 0.0) is absent.

    Args:
        base_weights: WEIGHTS_US or WEIGHTS_GLOBAL dict (sum must be 1.0).
                      Keys are factor short names WITHOUT "_score" suffix.
        market: Market enum value.

    Returns:
        Dict with same keys as base_weights. Sum = 1.0 for available factors.
        Unavailable factors have value 0.0.
    """
    available = MARKET_FACTORS[market]

    available_weight_sum = sum(
        w for name, w in base_weights.items()
        if f"{name}_score" in available
    )

    if available_weight_sum == 0.0:
        raise ValueError(
            f"No factors from base_weights available for market {market.value}. "
            f"Available: {available}. base_weights keys: {list(base_weights.keys())}."
        )

    result: Dict[str, float] = {}
    for name, w in base_weights.items():
        if f"{name}_score" in available:
            result[name] = w / available_weight_sum
        else:
            result[name] = 0.0

    return result


def market_weight_coverage(market: Market, base_weights: Dict[str, float]) -> float:
    """Return fraction of total base weight covered by this market's factors.

    With WEIGHTS_GLOBAL for EU/Asia:
      congress=0.0, transcript_tone=0.0 absent → coverage ≈ 1.0
    This is a dramatic improvement vs the old 0.33 coverage.
    """
    available = MARKET_FACTORS[market]
    return sum(
        w for name, w in base_weights.items()
        if f"{name}_score" in available
    )
