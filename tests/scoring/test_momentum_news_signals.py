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

from src.scoring.momentum_signals import (
    score_momentum_long,
    score_volume_attention,
    score_price_target_upside,
    score_quality_piotroski,
)
from src.scoring.news_signals import score_news_sentiment, score_news_buzz


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

    def test_neutral_articles_are_absent_signal(self):
        """All-Neutral articles have no directional content → return 0.0 (absent), not 0.5."""
        articles = [_article("Neutral", d) for d in (1, 2, 3)]
        score = score_news_sentiment(articles)
        assert score == 0.0, f"All-Neutral should be 0.0 (absent signal), got {score}"

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
        0.00 = 50%+ downside (clipped) — a REAL observation
        None = unavailable (missing/zero/NaN input). SIGNED contract: absence
               must never read as a bearish 0.0 (CLAUDE.md §2).
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

    def test_none_target_returns_unavailable(self):
        """SIGNED contract: missing target is unavailable (None), not bearish 0.0."""
        assert score_price_target_upside(None, 100.0) is None

    def test_none_current_returns_unavailable(self):
        assert score_price_target_upside(100.0, None) is None

    def test_zero_current_price_returns_unavailable(self):
        """Zero current price → division guard → unavailable (None)."""
        assert score_price_target_upside(100.0, 0.0) is None

    def test_zero_target_returns_unavailable(self):
        """Zero target is a data error → unavailable (None), not bearish 0.0."""
        assert score_price_target_upside(0.0, 100.0) is None

    def test_returns_float_rounded_to_4dp(self):
        result = score_price_target_upside(110.0, 100.0)
        assert isinstance(result, float)
        assert result == round(result, 4)

    def test_small_upside(self):
        """5% upside → (0.05 + 0.50) / 1.00 = 0.55."""
        assert score_price_target_upside(105.0, 100.0) == 0.5500

    def test_nan_returns_unavailable(self):
        assert score_price_target_upside(float('nan'), 100.0) is None
        assert score_price_target_upside(100.0, float('nan')) is None


class TestQualityPiotroski:
    """Simplified 8-point Piotroski F-score mapped to [0, 1].

    References:
        Piotroski (2000) JAR — historical financial statements separate winners from losers.
        Novy-Marx (2013) JFE — gross profitability predicts cross-sectional returns.

    Score = points_earned / 8.0. Dead signal (0.0) when ratios is None/empty/all-None.
    """

    def _full_quality_ratios(self) -> dict:
        """Ratios dict where all 8 points pass — perfect score."""
        return {
            "returnOnAssetsTTM":        0.10,   # > 0 (point 1) and > 0.05 (point 2)
            "operatingProfitMarginTTM": 0.15,   # > 0 (point 3)
            "debtEquityRatioTTM":       0.30,   # < 1.0 (point 4) and < 0.5 (point 5)
            "currentRatioTTM":          2.0,    # > 1.5 (point 6)
            "grossProfitMarginTTM":     0.45,   # > 0.30 (point 7)
            "netProfitMarginTTM":       0.08,   # > 0.05 (point 8)
        }

    def test_perfect_score_all_8_points(self):
        score, raw = score_quality_piotroski(self._full_quality_ratios())
        assert score == 1.0000
        assert raw == 8

    def test_zero_score_all_8_points_fail(self):
        ratios = {
            "returnOnAssetsTTM":        -0.05,  # fails points 1 and 2
            "operatingProfitMarginTTM": -0.10,  # fails point 3
            "debtEquityRatioTTM":        2.0,   # fails points 4 and 5
            "currentRatioTTM":           0.8,   # fails point 6
            "grossProfitMarginTTM":      0.10,  # fails point 7
            "netProfitMarginTTM":       -0.02,  # fails point 8
        }
        score, raw = score_quality_piotroski(ratios)
        assert score == 0.0000
        assert raw == 0

    def test_partial_score_5_of_8_points(self):
        """ROA > 0 only (not > 0.05), opMargin OK, D/E < 1 only (not < 0.5),
        currentRatio OK, grossMargin OK, netMargin fails."""
        ratios = {
            "returnOnAssetsTTM":        0.02,   # passes point 1, fails point 2
            "operatingProfitMarginTTM": 0.10,   # passes point 3
            "debtEquityRatioTTM":       0.70,   # passes point 4, fails point 5
            "currentRatioTTM":          2.0,    # passes point 6
            "grossProfitMarginTTM":     0.40,   # passes point 7
            "netProfitMarginTTM":       0.02,   # fails point 8
        }
        score, raw = score_quality_piotroski(ratios)
        assert score == round(5 / 8, 4)
        assert raw == 5

    def test_empty_dict_returns_dead_signal(self):
        score, raw = score_quality_piotroski({})
        assert score == 0.0
        assert raw == 0

    def test_none_returns_dead_signal(self):
        score, raw = score_quality_piotroski(None)
        assert score == 0.0
        assert raw == 0

    def test_all_none_fields_returns_dead_signal(self):
        ratios = {
            "returnOnAssetsTTM":        None,
            "operatingProfitMarginTTM": None,
            "debtEquityRatioTTM":       None,
            "currentRatioTTM":          None,
            "grossProfitMarginTTM":     None,
            "netProfitMarginTTM":       None,
        }
        score, raw = score_quality_piotroski(ratios)
        assert score == 0.0
        assert raw == 0

    def test_missing_individual_fields_score_zero_for_that_point(self):
        """A company with 5 of 8 fields present and all passing scores 5/8."""
        ratios = {
            "returnOnAssetsTTM":        0.10,   # points 1+2 pass
            "operatingProfitMarginTTM": 0.15,   # point 3 passes
            # debtEquityRatioTTM missing — points 4+5 score 0
            "currentRatioTTM":          2.0,    # point 6 passes
            "grossProfitMarginTTM":     0.40,   # point 7 passes
            # netProfitMarginTTM missing — point 8 scores 0
        }
        score, raw = score_quality_piotroski(ratios)
        assert score == round(5 / 8, 4)
        assert raw == 5

    def test_negative_debt_equity_fails_both_leverage_points(self):
        """Negative D/E (negative book equity) is worse than high D/E — fails points 4 and 5."""
        ratios = {**self._full_quality_ratios(), "debtEquityRatioTTM": -0.5}
        # Loses 2 leverage points: 8 - 2 = 6 → 6/8
        score, raw = score_quality_piotroski(ratios)
        assert score == round(6 / 8, 4)
        assert raw == 6

    def test_roa_exactly_at_5pct_threshold(self):
        """ROA == 0.05 fails point 2 (must be strictly greater than 0.05)."""
        ratios = {**self._full_quality_ratios(), "returnOnAssetsTTM": 0.05}
        # Loses point 2: 8 - 1 = 7 → 7/8
        score, raw = score_quality_piotroski(ratios)
        assert score == round(7 / 8, 4)
        assert raw == 7

    def test_gross_margin_exactly_at_threshold(self):
        """grossProfitMarginTTM == 0.30 fails point 7 (must be strictly greater)."""
        ratios = {**self._full_quality_ratios(), "grossProfitMarginTTM": 0.30}
        score, raw = score_quality_piotroski(ratios)
        assert score == round(7 / 8, 4)
        assert raw == 7

    def test_returns_float_in_range_0_to_1(self):
        score, raw = score_quality_piotroski(self._full_quality_ratios())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_rounded_to_4_decimal_places(self):
        ratios = {**self._full_quality_ratios(), "returnOnAssetsTTM": 0.02}
        score, raw = score_quality_piotroski(ratios)
        assert score == round(score, 4)

    def test_non_dict_input_returns_dead_signal(self):
        score1, raw1 = score_quality_piotroski("not a dict")
        score2, raw2 = score_quality_piotroski(42)
        assert score1 == 0.0
        assert score2 == 0.0


class TestScoreQualityPiotroskiLiveFieldShape:
    """Live stable/ratios-ttm + ratios-ttm-bulk shape (verified 2026-06-09):
    leverage field is debtToEquityRatioTTM, and there is NO returnOnAssets*
    field — ROA derives via DuPont (netProfitMargin × assetTurnover).
    Regression for the bug that pinned the whole universe at raw=4.
    """

    def _live_quality_ratios(self) -> dict:
        """Field names exactly as the live FMP payload — all 8 points pass."""
        return {
            "operatingProfitMarginTTM": 0.15,    # point 3
            "debtToEquityRatioTTM":     0.30,    # points 4+5 (live name)
            "currentRatioTTM":          2.0,     # point 6
            "grossProfitMarginTTM":     0.45,    # point 7
            "netProfitMarginTTM":       0.08,    # point 8
            "assetTurnoverTTM":         0.80,    # DuPont: ROA = 0.08*0.8 = 0.064 → points 1+2
        }

    def test_live_shape_reaches_8_of_8(self):
        score, raw = score_quality_piotroski(self._live_quality_ratios())
        assert raw == 8, "live payload must be able to award all 8 points"
        assert score == 1.0

    def test_dupont_roa_derivation(self):
        """npm=0.10 × at=0.8 → ROA=0.08 > 0.05: points 1 and 2 awarded."""
        ratios = {"netProfitMarginTTM": 0.10, "assetTurnoverTTM": 0.80}
        score, raw = score_quality_piotroski(ratios)
        # npm 0.10 > 0.05 (point 8) + derived ROA points 1, 2 = 3 points
        assert raw == 3

    def test_dupont_skipped_when_explicit_roa_present(self):
        """Explicit returnOnAssetsTTM wins over the DuPont derivation."""
        ratios = {
            "returnOnAssetsTTM":  -0.10,   # explicit negative ROA: points 1+2 fail
            "netProfitMarginTTM":  0.10,   # would derive positive ROA if used
            "assetTurnoverTTM":    0.80,
        }
        score, raw = score_quality_piotroski(ratios)
        assert raw == 1  # only point 8 (npm > 0.05)

    def test_unsuffixed_bulk_shape_scores_identically(self):
        """Future-proofing: a bulk snapshot without the TTM suffix scores the same."""
        unsuffixed = {
            "operatingProfitMargin": 0.15,
            "debtToEquityRatio":     0.30,
            "currentRatio":          2.0,
            "grossProfitMargin":     0.45,
            "netProfitMargin":       0.08,
            "assetTurnover":         0.80,
        }
        score, raw = score_quality_piotroski(unsuffixed)
        assert raw == 8
        assert score == 1.0

    def test_live_name_preferred_over_legacy(self):
        """When both leverage names exist the live one wins."""
        ratios = {
            "debtToEquityRatioTTM": 0.30,   # live: passes points 4+5
            "debtEquityRatioTTM":   5.00,   # legacy: would fail both
            "currentRatioTTM":      2.0,
        }
        score, raw = score_quality_piotroski(ratios)
        assert raw == 3  # points 4, 5, 6
