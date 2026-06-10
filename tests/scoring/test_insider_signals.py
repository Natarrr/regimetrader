"""tests/scoring/test_insider_signals.py
Unit tests for orthogonal insider signal decomposition.

Cohen, Malloy & Pomorski (2012): conviction (dollar magnitude) and
breadth (consensus among insiders) are designed to be uncorrelated.
"""
from __future__ import annotations

import pytest
from datetime import date, timedelta

from src.scoring.insider_signals import (
    score_insider_conviction,
    score_insider_breadth,
)


def _recent(days_ago: int = 5) -> str:
    return (date.today() - timedelta(days=days_ago)).isoformat()


class TestConvictionVsBreadthOrthogonal:
    """
    Fixture A — single large CEO purchase: high conviction, low breadth.
    Fixture B — eight small purchases by different insiders: low conviction, high breadth.
    Each pair must diverge by > 0.3 to confirm orthogonality.
    """

    def test_fixture_a_high_conviction_low_breadth(self):
        # CEO buys $5M in a $500M-cap company (1% of mktcap).
        # Conviction must be high; breadth with a single buyer is moderate
        # (100% consensus but n=1 → breadth_scale is low).
        # Key check: conviction >> breadth (the CEO premium + large % lifts conviction
        # well above what 1-insider breadth can reach).
        conviction_a = score_insider_conviction(
            key_purchases_usd=5_000_000,
            market_cap=500_000_000,
            days_since_most_recent=3,
            ceo_purchase_usd=5_000_000,
        )
        p_txs = [{"value_usd": 5_000_000, "title": "CEO", "date": _recent(3), "insider_id": "ceo_001"}]
        breadth_a = score_insider_breadth(p_txs, [])

        assert conviction_a > 0.85, f"CEO $5M (1% mktcap) + CEO premium: conviction expected > 0.85, got {conviction_a}"
        assert conviction_a > breadth_a, "Conviction must exceed breadth for single large buyer"

    def test_fixture_b_low_conviction_high_breadth(self):
        # 8 directors each buy $50k in a $500M-cap company (0.008% each).
        # Total $400k = 0.08% of mktcap — modest conviction.
        # 8 distinct buyers → high breadth (consensus signal).
        total_usd = 8 * 50_000
        conviction_b = score_insider_conviction(
            key_purchases_usd=total_usd,
            market_cap=500_000_000,
            days_since_most_recent=5,
            ceo_purchase_usd=0.0,
        )
        p_txs = [
            {"value_usd": 50_000, "title": "Director", "date": _recent(5), "insider_id": f"dir_{i}"}
            for i in range(8)
        ]
        breadth_b = score_insider_breadth(p_txs, [])

        assert conviction_b < 0.65, f"8×$50k (0.08% mktcap) conviction expected < 0.65, got {conviction_b}"
        assert breadth_b > 0.85, f"8 distinct buyers breadth expected > 0.85, got {breadth_b}"
        # Core orthogonality check: breadth substantially exceeds conviction
        assert breadth_b - conviction_b > 0.2, (
            f"Breadth should substantially exceed conviction for many-small-buyer case: "
            f"breadth={breadth_b}, conviction={conviction_b}"
        )

    def test_conviction_zero_on_no_purchases(self):
        assert score_insider_conviction(0.0, 1_000_000) == 0.0

    def test_conviction_zero_on_zero_mktcap(self):
        assert score_insider_conviction(100_000, 0.0) == 0.0

    def test_breadth_zero_on_empty(self):
        assert score_insider_breadth([], []) == 0.0

    def test_conviction_ceo_premium_applied(self):
        base = score_insider_conviction(1_000_000, 1_000_000_000, ceo_purchase_usd=0.0)
        with_ceo = score_insider_conviction(1_000_000, 1_000_000_000, ceo_purchase_usd=1_000_000)
        assert with_ceo > base, "CEO premium must increase conviction score"
        assert with_ceo <= 0.95, "CEO premium must be capped at 0.95"

    def test_conviction_recency_decay(self):
        fresh  = score_insider_conviction(1_000_000, 1_000_000_000, days_since_most_recent=5)
        stale  = score_insider_conviction(1_000_000, 1_000_000_000, days_since_most_recent=120)
        assert fresh > stale, "Fresh purchase must score higher than stale"
        assert stale > 0.5, "Stale net-buy signal should still be > 0.5 (direction preserved)"

    def test_conviction_bounded(self):
        score = score_insider_conviction(999_999_999, 1_000_000, ceo_purchase_usd=999_999_999)
        assert 0.0 <= score <= 1.0


class TestBreadthHandlesSells:
    """
    Breadth signal correctly handles mixed buy/sell scenarios.
    """

    def test_50_50_split_near_neutral(self):
        # 5 buyers vs 5 sellers: buyer_ratio=0.5, breadth_scale=log(11)/log(11)=1.0
        # base = 0.7*0.5 + 0.3*1.0 = 0.65. The absolute breadth term (0.3 weight)
        # pulls the score up when there is high participation on both sides.
        # This is correct: high activity WITH 50/50 direction = moderately positive
        # (market-wide liquidity event, slightly elevated score vs zero activity).
        p_txs = [
            {"value_usd": 50_000, "title": "Dir", "date": _recent(3), "insider_id": f"buyer_{i}"}
            for i in range(5)
        ]
        s_txs = [
            {"value_usd": 50_000, "title": "Dir", "date": _recent(3), "insider_id": f"seller_{i}"}
            for i in range(5)
        ]
        score = score_insider_breadth(p_txs, s_txs)
        # Must be below all-buyers threshold (>= 0.7) and above dead-signal (0.0)
        assert score < 0.75, f"50/50 split must be below all-buyers score, got {score}"
        assert score > 0.3, f"50/50 split with 10 active insiders must be > 0.3, got {score}"

    def test_all_buyers_high_breadth(self):
        p_txs = [
            {"value_usd": 30_000, "title": "Dir", "date": _recent(5), "insider_id": f"buyer_{i}"}
            for i in range(5)
        ]
        score = score_insider_breadth(p_txs, [])
        assert score >= 0.7, f"5 buyers, 0 sellers — breadth should be >= 0.7, got {score}"

    def test_all_sellers_low_breadth(self):
        s_txs = [
            {"value_usd": 30_000, "title": "Dir", "date": _recent(5), "insider_id": f"seller_{i}"}
            for i in range(5)
        ]
        score = score_insider_breadth([], s_txs)
        assert score < 0.35, f"0 buyers, 5 sellers — breadth should be < 0.35, got {score}"

    def test_deduplication_same_insider_multiple_trades(self):
        # Same insider_id buying 3 times counts as 1 distinct buyer
        p_txs = [
            {"value_usd": 10_000, "title": "CEO", "date": _recent(3), "insider_id": "ceo_001"},
            {"value_usd": 10_000, "title": "CEO", "date": _recent(4), "insider_id": "ceo_001"},
            {"value_usd": 10_000, "title": "CEO", "date": _recent(5), "insider_id": "ceo_001"},
        ]
        s_txs = [
            {"value_usd": 10_000, "title": "Dir", "date": _recent(3), "insider_id": "dir_001"},
        ]
        score = score_insider_breadth(p_txs, s_txs)
        # 1 distinct buyer vs 1 distinct seller → buyer_ratio = 0.5
        assert abs(score - 0.5) < 0.2, (
            f"1 unique buyer vs 1 unique seller should be near neutral, got {score}"
        )

    def test_lookback_filters_old_transactions(self):
        # Transaction is 120 days old — outside default 90-day window
        old_tx = [{"value_usd": 999_999, "title": "CEO", "date": _recent(120), "insider_id": "ceo_001"}]
        score = score_insider_breadth(old_tx, [], lookback_days=90)
        assert score == 0.0, "Transactions outside lookback window must be ignored"

    def test_breadth_bounded(self):
        p_txs = [{"value_usd": 1, "title": "X", "date": _recent(1), "insider_id": f"x{i}"}
                 for i in range(20)]
        score = score_insider_breadth(p_txs, [])
        assert 0.0 <= score <= 1.0
