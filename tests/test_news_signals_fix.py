"""tests/test_news_signals_fix.py
Unit tests for news_signals.py fix: all-neutral articles return 0.0 (absent), not 0.5 (neutral).

The bug: when FMP returns articles that ALL have sentiment="Neutral" or missing sentiment,
weighted_sum = 0.0, causing score = (0.0 + 1.0) / 2.0 = 0.5. This is wrong: 0.5 suggests
a genuine neutral signal (equal positive and negative), but we actually have NO directional
content at all. Fix: track has_directional and return 0.0 if no articles had Positive/Negative.
"""
from __future__ import annotations

from regime_trader.scoring.news_signals import score_news_sentiment


def test_all_neutral_articles_returns_zero():
    """All-neutral FMP articles must return 0.0 (absent), not 0.5 (neutral)."""
    articles = [
        {"sentiment": "Neutral", "publishedDate": "2026-06-04"},
        {"sentiment": "Neutral", "publishedDate": "2026-06-03"},
        {"sentiment": "",        "publishedDate": "2026-06-03"},
        {"publishedDate": "2026-06-02"},  # missing sentiment key
    ]
    score = score_news_sentiment(articles)
    assert score == 0.0, f"Expected 0.0 for all-neutral articles, got {score}"


def test_genuinely_balanced_returns_near_half():
    """Equal positive and negative articles should return ~0.5 (genuine neutral)."""
    articles = [
        {"sentiment": "Positive", "publishedDate": "2026-06-04"},
        {"sentiment": "Negative", "publishedDate": "2026-06-04"},
    ]
    score = score_news_sentiment(articles)
    assert 0.48 <= score <= 0.52, f"Expected ~0.5 for balanced articles, got {score}"


def test_all_positive_returns_one():
    articles = [{"sentiment": "Positive", "publishedDate": "2026-06-04"}] * 5
    score = score_news_sentiment(articles)
    assert score == 1.0


def test_all_negative_returns_zero():
    articles = [{"sentiment": "Negative", "publishedDate": "2026-06-04"}] * 5
    score = score_news_sentiment(articles)
    assert score == 0.0


def test_empty_returns_zero():
    assert score_news_sentiment([]) == 0.0


def test_mixed_directional_with_neutrals():
    """Neutral articles should not affect computation when directional articles exist."""
    articles = [
        {"sentiment": "Positive", "publishedDate": "2026-06-04"},
        {"sentiment": "Neutral",  "publishedDate": "2026-06-04"},
        {"sentiment": "Neutral",  "publishedDate": "2026-06-03"},
    ]
    score = score_news_sentiment(articles)
    # One Positive (val=1.0) + two Neutral (val=0.0)
    # raw = (1.0 * w1 + 0 + 0) / (w1 + w2 + w3) where all weights ~1 (same day)
    # raw ≈ 1/3 → score = (1/3 + 1) / 2 ≈ 0.667
    assert 0.60 <= score <= 0.75, f"Expected mixed directional+neutral ~0.67, got {score}"
