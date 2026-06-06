# Path: backend/market_intel/_generate_top_lists_intl_patch.py
"""Helper for building EU/Asia entry dicts in generate_top_lists.generate().

PATCH v2.2-global: replaces the hardcoded-zeros intl_results merge block.
Previously the merge set insider/news/analyst to 0.0 unconditionally because
the old FMPFetcher only returned momentum + volume for EU/Asia.

Now FMPFetcher.prepare() populates all 10 global factor scores. _build_intl_entry
reads them from the result row instead of hardcoding zeros.
"""
from __future__ import annotations

from typing import Any, Dict

# Badge thresholds — mirror generate_top_lists._BADGES to avoid circular import
_BADGES = [
    (0.80, "HIGH BUY"),
    (0.60, "TACTICAL BUY"),
    (0.00, "WATCHLIST"),
]


def _badge(score: float) -> str:
    for threshold, label in _BADGES:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _build_intl_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build a scored entry dict from an international (EU/Asia) result row.

    Reads actual factor scores instead of hardcoding zeros.
    All factors available via FMP Ultimate globally are populated.
    congress and transcript_tone are always 0.0 (structural absence).

    Args:
        row: Result dict from _score_ticker_international() — contains
             *_score fields populated by FMPFetcher.prepare().

    Returns:
        Entry dict compatible with the top_lists.json schema.
    """
    def _f(key: str) -> float:
        val = row.get(key)
        if val is None:
            return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    market = row.get("market", "EUROPE")
    region = "EU" if market == "EUROPE" else "ASIA"

    factors = {
        # ── Globally available via FMP Ultimate ──────────────────────────
        "insider_conviction":  _f("insider_conviction_score"),
        "insider_breadth":     _f("insider_breadth_score"),
        "news_sentiment":      _f("news_sentiment_score"),
        "news_buzz":           _f("news_buzz_score"),
        "momentum_long":       _f("momentum_long_score"),
        "volume_attention":    _f("volume_attention_score"),
        "analyst_consensus":   _f("analyst_consensus_score"),
        "analyst_revision":    _f("analyst_revision_score"),
        "quality_piotroski":   _f("quality_piotroski_score"),
        "price_target_upside": _f("price_target_upside_score"),
        # ── Structurally absent ──────────────────────────────────────────
        "congress":            0.0,   # no STOCK Act outside US
        "transcript_tone":     0.0,   # FMP transcripts US-only
    }

    final_score = _f("final_score")

    return {
        "ticker":          row.get("ticker", "?"),
        "company_name":    row.get("company_name", ""),
        "sector":          row.get("sector", "Unknown"),
        "cap_tier":        row.get("cap_tier", "large"),
        "market_cap":      _f("market_cap"),
        "raw_score":       final_score,
        "final_score":     final_score,
        "badge":           _badge(final_score),
        "region":          region,
        "weights_set":     "GLOBAL",
        "ceo_buy":         False,
        "form4_count":     0,
        "factors":         factors,
        "validation_metadata": {
            "is_complete":     True,
            "missing_sources": ["congress", "transcript_tone"],
        },
        "quiver_evidence":           {},
        "news_source":               row.get("news_sentiment_source", "fmp"),
        "insider_usd":               0.0,
        "momentum_spy_relative":     _f("momentum_spy_relative"),
        "volume_spike":              _f("volume_spike"),
        "market":                    market,
        "target_price":              row.get("target_price"),
        "current_price":             row.get("current_price"),
        "analyst_consensus_source":  row.get("analyst_consensus_source", "bulk"),
        "insider_source":            row.get("insider_source", "fmp"),
        # Diagnostic metadata only — FMPFetcher.source_reliability() returns 1.0.
        "source_reliability": _f("source_reliability") or 1.0,
        # Factor score pass-through for Discord formatter
        "insider_conviction_score":  _f("insider_conviction_score"),
        "insider_breadth_score":     _f("insider_breadth_score"),
        "news_sentiment_score":      _f("news_sentiment_score"),
        "news_buzz_score":           _f("news_buzz_score"),
        "analyst_consensus_score":   _f("analyst_consensus_score"),
        "analyst_revision_score":    _f("analyst_revision_score"),
        "quality_piotroski_score":   _f("quality_piotroski_score"),
        "price_target_upside_score": _f("price_target_upside_score"),
        "momentum_long_score":       _f("momentum_long_score"),
        "volume_attention_score":    _f("volume_attention_score"),
    }
