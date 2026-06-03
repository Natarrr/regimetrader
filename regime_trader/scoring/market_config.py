"""regime_trader.scoring.market_config — per-market factor availability and weight renormalization.

Diagnostic-driven configuration (updated 2026-06-03, 9-factor schema):

Phase-0 smoke-test results (FMP Ultimate, stable/ routes):
  PASS:  EU:quote, EU:ratios-ttm, ASIA:quote, ASIA:ratios-ttm
  PASS:  upgrades-downgrades-consensus-bulk — exchange-agnostic, covers EU/Asia
  EMPTY: EU/ASIA news/stock (returns 0 articles for non-US tickers)
  N/A:   insider/congress — no EU MAR / EDINET / STOCK-Act equivalent via FMP

Schema migration (12 → 9 factors):
  Removed: analyst_revision, price_target_upside, transcript_tone
  Reason:  sell-side triplet correlation risk (Grinold-Kahn 2000); staleness risk
           on quarterly PT targets; transcript tone redundant with news_sentiment.

Therefore:
  US     — 9 factors (full WEIGHTS)
  EUROPE — 5 factors: momentum_long + volume_attention + quality_piotroski
                       + analyst_consensus + news_sentiment + news_buzz
           (analyst_consensus now sourced from upgrades-downgrades-consensus-bulk,
            which covers global exchanges; news remains EMPTY for EU)
  ASIA   — same as EUROPE

Structurally absent factors (no data source available) are excluded from
renormalize_weights_for_market so their weight is redistributed proportionally
among available factors. This preserves the relative ordering of available factors
as validated on the US universe.

Constants named here are the authoritative thresholds — do not hardcode elsewhere.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet

# Minimum tickers required for a cross-sectional bucket to apply z-score neutralization.
MIN_BUCKET_SIZE: int = 5

# If a ticker's final_score is computed from less than this fraction of the market's
# total available weights, it is marked _low_coverage=True and excluded from Top-N.
# Example: EU has 2 available factors summing to 0.18 weights. A ticker where even
# momentum_long is None (recent IPO, no history) would have weight_coverage < 0.15.
LOW_COVERAGE_THRESHOLD: float = 0.50


class Market(str, Enum):
    US = "US"
    EUROPE = "EUROPE"
    ASIA = "ASIA"


# Maps pipeline market strings (from ticker_registry.json and run_pipeline.py) to Market enum.
PIPELINE_MARKET_MAP: Dict[str, Market] = {
    "USA":    Market.US,
    "US":     Market.US,
    "EUROPE": Market.EUROPE,
    "ASIA":   Market.ASIA,
}


# Factors POTENTIALLY available per market.
# "Potentially" = a data source exists; absence for a specific ticker (insufficient
# history, recent IPO) is still possible and handled downstream via None vs 0.0.
#
# EUROPE/ASIA rationale (updated 2026-05-30 Phase-0 smoke-test):
#   FMP stable/ quote + ratios-ttm: PASS for EU/Asia — usable for market cap,
#     P/E, quality factors in future phases.
#   FMP stable/ news/stock: EMPTY for EU/Asia (0 articles for SAP.DE, 7203.T).
#   Congress disclosures: no STOCK Act equivalent outside the US.
#   Insider disclosures: EU MAR, Japan EDINET — no accessible API without
#     institutional-grade subscription (Refinitiv, Bloomberg).
#   Momentum + volume: yfinance provides universal coverage for all major
#     exchange-listed EU/Asia symbols.
#   Factor matrix unchanged — only documentation corrected.
MARKET_FACTORS: Dict[Market, FrozenSet[str]] = {
    Market.US: frozenset({
        "insider_conviction_score",
        "insider_breadth_score",
        "congress_score",
        "news_sentiment_score",
        "news_buzz_score",
        "momentum_long_score",
        "volume_attention_score",
        "analyst_consensus_score",
        "quality_piotroski_score",
    }),
    Market.EUROPE: frozenset({
        # congress_score: STRUCTURALLY ABSENT — no STOCK Act equivalent in EU
        # insider_conviction_score: STRUCTURALLY ABSENT — no EU MAR API integrated
        # insider_breadth_score: STRUCTURALLY ABSENT — same
        # news_sentiment_score: STRUCTURALLY ABSENT — FMP news/stock EMPTY for EU tickers
        # news_buzz_score: STRUCTURALLY ABSENT — same
        "momentum_long_score",       # FMP historical-price-eod/full ✅ (Phase-0 PASS)
        "volume_attention_score",    # FMP historical-price-eod/full ✅
        "quality_piotroski_score",   # FMP financial-scores-bulk ✅ (exchange-agnostic)
        "analyst_consensus_score",   # FMP upgrades-downgrades-consensus-bulk ✅ (global)
    }),
    Market.ASIA: frozenset({
        # congress_score: STRUCTURALLY ABSENT
        # insider_conviction_score: STRUCTURALLY ABSENT — EDINET not integrated
        # insider_breadth_score: STRUCTURALLY ABSENT — same
        # news_sentiment_score: STRUCTURALLY ABSENT — FMP news/stock EMPTY for Asia tickers
        # news_buzz_score: STRUCTURALLY ABSENT — same
        "momentum_long_score",       # FMP historical-price-eod/full ✅ (Phase-0 PASS for 7203.T)
        "volume_attention_score",    # FMP historical-price-eod/full ✅
        "quality_piotroski_score",   # FMP financial-scores-bulk ✅ (exchange-agnostic)
        "analyst_consensus_score",   # FMP upgrades-downgrades-consensus-bulk ✅ (global)
    }),
}


def renormalize_weights_for_market(
    base_weights: Dict[str, float],
    market: Market,
) -> Dict[str, float]:
    """Renormalize WEIGHTS for the factors available in this market.

    Factors unavailable for this market get weight 0.0. Available factors
    keep their *relative proportions* from base_weights (scaled to sum to 1.0).

    This is a conservative renormalization: it does NOT re-weight according to
    a new logic (e.g. "give more to momentum since it's the only factor").
    It assumes the relative IC ratios validated on the US universe are a
    reasonable baseline for the subset of available factors.

    Example for EUROPE (momentum_long + volume_attention + quality_piotroski + analyst_consensus):
        base_weights["momentum_long"]     = 0.15
        base_weights["volume_attention"]  = 0.03
        base_weights["quality_piotroski"] = 0.08
        base_weights["analyst_consensus"] = 0.10
        Sum of available = 0.36
        renormalized["momentum_long"] = 0.15 / 0.36 ≈ 0.417
        renormalized["analyst_consensus"] = 0.10 / 0.36 ≈ 0.278
        ...
        Sum = 1.0 ✅

    Args:
        base_weights: The full WEIGHTS dict from run_pipeline.py (7 factors, sum=1.0).
                      Keys are factor short names WITHOUT "_score" suffix (e.g. "momentum_long").
        market: Market enum value.

    Returns:
        Dict with same keys as base_weights. Unavailable factors have value 0.0.
        sum(values) == 1.0 (within float tolerance).

    Raises:
        ValueError: If no factors from base_weights are available for this market
                    (would produce division by zero).
    """
    available = MARKET_FACTORS[market]

    # Map base_weights keys to factor score names (add _score suffix for lookup).
    # base_weights keys: "momentum_long", "congress", etc.
    # MARKET_FACTORS keys: "momentum_long_score", "congress_score", etc.
    available_weight_sum = sum(
        w for name, w in base_weights.items()
        if f"{name}_score" in available
    )

    if available_weight_sum == 0.0:
        raise ValueError(
            f"No factors from base_weights are available for market {market.value}. "
            f"Available factors: {available}. "
            f"base_weights keys: {list(base_weights.keys())}."
        )

    result: Dict[str, float] = {}
    for name, w in base_weights.items():
        if f"{name}_score" in available:
            result[name] = w / available_weight_sum
        else:
            result[name] = 0.0

    return result


def market_weight_coverage(market: Market, base_weights: Dict[str, float]) -> float:
    """Return the fraction of total base weight covered by this market's available factors.

    Used as metadata: EU/Asia tickers have weight_coverage < 1.0 by construction.
    A ticker-level weight_coverage < LOW_COVERAGE_THRESHOLD triggers _low_coverage=True.

    Example: EUROPE has momentum_long(0.15) + volume_attention(0.03) + quality_piotroski(0.08) + analyst_consensus(0.10) = 0.36 / 1.0 = 36%.
    """
    available = MARKET_FACTORS[market]
    return sum(
        w for name, w in base_weights.items()
        if f"{name}_score" in available
    )
