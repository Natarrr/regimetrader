"""tests/scoring/test_momentum_news_signals.py
Unit tests for Jegadeesh-Titman momentum and recency-weighted news signals.

References:
    Jegadeesh & Titman (1993), Journal of Finance 48(1)
    Tetlock (2007), Journal of Finance 62(3)
    Barber & Odean (2008), Review of Financial Studies 21(2)
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from regime_trader.scoring.momentum_signals import score_momentum_long, score_volume_attention, score_price_target_upside
from regime_trader.scoring.news_signals import score_news_sentiment, score_news_buzz


def _article(sentiment: str, days_ago: int) -> dict:
    return {
        "sentiment":     sentiment,
        "publishedDate": (date.today() - timedelta(days=days_ago)).isoformat(),
    }


class TestMomentumLong:
    """Jegadeesh-Titman (1993): 12-1m SPY-relative return maps to [0, 1]."""

    def test_distinguishes_winners_from_losers(self):
        # Fixture A: strong winner vs SPY
        score_a = score_momentum_long(return_12_1m=0.40, spy_return_12_1m=0.10)
        # Fixture B: loser vs SPY
        score_b = score_momentum_long(return_12_1m=-0.30, spy_return_12_1m=0.10)
        # Fixture C: market-equal (excess = 0)
        score_c = score_momentum_long(return_12_1m=0.10, spy_return_12_1m=0.10)

        assert score_a >= 0.70, f"Winner +30% excess should score >= 0.70, got {score_a}"
        assert score_b <= 0.30, f"Loser  -40% excess should score <= 0.30, got {score_b}"
        assert abs(score_c - 0.50) < 0.01, f"Market-equal should score ~0.50, got {score_c}"
        assert score_a > score_c > score_b, "Ordering must be winner > neutral > loser"

    def test_none_returns_dead_signal(self):
        assert score_momentum_long(None) == 0.0

    def test_bounded(self):
        assert 0.0 <= score_momentum_long(return_12_1m=99.0) <= 1.0
        assert 0.0 <= score_momentum_long(return_12_1m=-99.0) <= 1.0

    def test_clip_at_sixty_percent(self):
        # ±60% is the hard clip — anything beyond maps to exactly 0.0 or 1.0
        assert score_momentum_long(return_12_1m=0.70, spy_return_12_1m=0.0) == 1.0
        assert score_momentum_long(return_12_1m=-0.70, spy_return_12_1m=0.0) == 0.0


class TestVolumeAttention:
    """Barber & Odean (2008): volume spike is pure attention, not direction."""

    def test_flat_volume_is_dead_signal(self):
        assert score_volume_attention(1.0) == 0.0

    def test_five_x_saturates(self):
        assert score_volume_attention(5.0) == 1.0

    def test_ordering(self):
        assert score_volume_attention(3.0) > score_volume_attention(2.0)

    def test_below_one_is_dead(self):
        assert score_volume_attention(0.5) == 0.0

    def test_bounded(self):
        assert 0.0 <= score_volume_attention(100.0) <= 1.0


class TestNewsSentimentRecency:
    """Tetlock (2007): sentiment effect decays with half-life ~5 days."""

    def test_recent_positive_high_score(self):
        # 3 Positive articles from last 3 days — strong recent signal
        articles = [_article("Positive", d) for d in (1, 2, 3)]
        score = score_news_sentiment(articles)
        assert score > 0.85, f"3 recent Positive articles should score > 0.85, got {score}"

    def test_old_negatives_outweighed_by_recent_positive(self):
        # Recent positive overrides old negatives — recency decay suppresses old sentiment.
        # Without decay: 2 Negative + 1 Positive → raw = (-2+1)/3 = -0.33 → score ≈ 0.33.
        # With decay (half-life=5d): old negatives at J-60/J-70 weigh ~exp(-60×ln2/5)≈0.
        # The single J-1 Positive dominates → score must be > 0.60.
        articles = [
            _article("Positive",  1),
            _article("Negative", 60),
            _article("Negative", 70),
        ]
        score = score_news_sentiment(articles)
        assert score > 0.60, (
            f"Recent Positive should dominate old Negatives after decay, got {score}"
        )

    def test_recency_has_measurable_effect(self):
        # Recency effect is visible when mixing sentiments: recent Positive + old Negative
        # should score higher than old Positive + recent Negative.
        recent_pos = score_news_sentiment([
            _article("Positive",  1),
            _article("Negative", 60),
        ])
        recent_neg = score_news_sentiment([
            _article("Negative",  1),
            _article("Positive", 60),
        ])
        assert recent_pos - recent_neg > 0.15, (
            f"Recency effect must exceed 0.15: recent_pos={recent_pos:.3f} "
            f"recent_neg={recent_neg:.3f} diff={recent_pos - recent_neg:.3f}"
        )

    def test_all_negative_low_score(self):
        articles = [_article("Negative", d) for d in (1, 2, 3)]
        score = score_news_sentiment(articles)
        assert score < 0.15, f"All-Negative recent articles should score < 0.15, got {score}"

    def test_empty_returns_dead_signal(self):
        assert score_news_sentiment([]) == 0.0

    def test_neutral_articles_near_half(self):
        articles = [_article("Neutral", d) for d in (1, 2, 3)]
        score = score_news_sentiment(articles)
        assert abs(score - 0.50) < 0.01, f"All-Neutral should be 0.50, got {score}"

    def test_bounded(self):
        articles = [_article("Positive", 1) for _ in range(100)]
        assert 0.0 <= score_news_sentiment(articles) <= 1.0


class TestNewsBuzz:
    """Barber & Odean (2008): buzz is coverage volume, not direction."""

    def test_no_articles_dead_signal(self):
        assert score_news_buzz([]) == 0.0

    def test_old_articles_ignored(self):
        # Articles 30 days old — outside 7-day buzz window
        articles = [_article("Positive", 30) for _ in range(10)]
        assert score_news_buzz(articles) == 0.0

    def test_recent_articles_score_nonzero(self):
        articles = [_article("Positive", d) for d in range(1, 6)]
        assert score_news_buzz(articles) > 0.0

    def test_more_recent_articles_higher_buzz(self):
        few  = score_news_buzz([_article("Positive", d) for d in range(1, 4)])
        many = score_news_buzz([_article("Positive", d) for d in range(1, 11)])
        assert many > few, "More recent articles must produce higher buzz"

    def test_bounded(self):
        articles = [_article("Positive", 1) for _ in range(100)]
        assert 0.0 <= score_news_buzz(articles) <= 1.0


class TestPriceTargetUpside:
    """Forward-looking analyst price target signal in [0, 1].

    Semantics:
        0.50 = target == current price (no upside/downside)
        0.75 = 25% upside to target
        0.25 = 25% downside to target
        1.00 = 50%+ upside (clipped)
        0.00 = 50%+ downside (clipped) OR dead signal (None/zero input)
    """

    def test_at_target_scores_neutral(self):
        """Target == current → exactly 0.50 (no upside/downside)."""
        assert score_price_target_upside(100.0, 100.0) == 0.5000

    def test_25pct_upside_scores_0_75(self):
        """25% upside → 0.75."""
        assert score_price_target_upside(125.0, 100.0) == 0.7500

    def test_25pct_downside_scores_0_25(self):
        """25% downside → 0.25."""
        assert score_price_target_upside(75.0, 100.0) == 0.2500

    def test_clips_at_50pct_upside(self):
        """70% upside clipped to 50% → 1.00."""
        assert score_price_target_upside(170.0, 100.0) == 1.0000

    def test_clips_at_50pct_downside(self):
        """-70% downside clipped to -50% → 0.00."""
        assert score_price_target_upside(30.0, 100.0) == 0.0000

    def test_exact_50pct_upside_scores_1(self):
        assert score_price_target_upside(150.0, 100.0) == 1.0000

    def test_exact_50pct_downside_scores_0(self):
        assert score_price_target_upside(50.0, 100.0) == 0.0000

    def test_none_target_returns_dead_signal(self):
        assert score_price_target_upside(None, 100.0) == 0.0

    def test_none_current_returns_dead_signal(self):
        assert score_price_target_upside(100.0, None) == 0.0

    def test_zero_current_price_returns_dead_signal(self):
        """Zero current price → division guard → 0.0."""
        assert score_price_target_upside(100.0, 0.0) == 0.0

    def test_zero_target_returns_dead_signal(self):
        """Zero target is a data error → 0.0."""
        assert score_price_target_upside(0.0, 100.0) == 0.0

    def test_returns_float_rounded_to_4dp(self):
        result = score_price_target_upside(110.0, 100.0)
        assert isinstance(result, float)
        assert result == round(result, 4)

    def test_small_upside(self):
        """5% upside → (0.05 + 0.50) / 1.00 = 0.55."""
        assert score_price_target_upside(105.0, 100.0) == 0.5500
