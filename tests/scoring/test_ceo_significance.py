"""tests/scoring/test_ceo_significance.py — Fix #7: relative CEO purchase significance.

Validates that the CEO conviction multiplier is scale-invariant: the same
dollar amount has 100× more signal strength on a micro-cap than on a mega-cap,
exactly as Cohen, Malloy & Pomorski (2012) observe for opportunistic CEO trades.
"""
from __future__ import annotations

import pytest

from src.scoring.insider_signals import _ceo_purchase_significance, score_insider_conviction


class TestCEOPurchaseSignificance:
    def test_scale_invariance_key_result(self):
        """
        Bug fixed by Fix #7: Cas B (10× the dollar of Cas A) should NOT receive
        more multiplier than Cas A. The absolute $1M purchase is immaterial on
        $100B but material on $1B.

        Cas A: $100k on $1B market cap → 1 bp → substantial tier → 1.10
        Cas B: $1M on $100B market cap → 0.1 bp → below modest → 1.00
        Cas C: $100k on $50M market cap → 20 bps → exceptional tier → 1.15
        """
        mult_a = _ceo_purchase_significance(ceo_purchase_usd=100_000, market_cap=1_000_000_000)
        mult_b = _ceo_purchase_significance(ceo_purchase_usd=1_000_000, market_cap=100_000_000_000)
        mult_c = _ceo_purchase_significance(ceo_purchase_usd=100_000, market_cap=50_000_000)

        # Cas A: 1 bp → substantial tier
        assert mult_a == pytest.approx(1.10), f"Expected 1.10 for 1 bp, got {mult_a}"
        # Cas B: 0.1 bp → below modest threshold → no bonus
        assert mult_b == pytest.approx(1.00), f"Expected 1.00 for 0.1 bp, got {mult_b}"
        # Cas C: 20 bps → exceptional tier
        assert mult_c == pytest.approx(1.15), f"Expected 1.15 for 20 bps, got {mult_c}"

        # The core invariance: Cas B (10× more dollars) must be LESS than Cas A
        assert mult_b < mult_a, (
            f"Scale invariance violated: $1M on $100B (mult={mult_b}) should be "
            f"less than $100k on $1B (mult={mult_a})"
        )

    def test_below_modest_threshold_no_bonus(self):
        """< 0.5 bps → multiplier = 1.00."""
        # 0.4 bps: $40k on $1B
        assert _ceo_purchase_significance(40_000, 1_000_000_000) == pytest.approx(1.00)

    def test_modest_tier(self):
        """0.5–1.0 bps → multiplier = 1.05."""
        # 0.75 bps: $75k on $1B
        assert _ceo_purchase_significance(75_000, 1_000_000_000) == pytest.approx(1.05)

    def test_substantial_tier(self):
        """1.0–5.0 bps → multiplier = 1.10."""
        # 2 bps: $200k on $1B
        assert _ceo_purchase_significance(200_000, 1_000_000_000) == pytest.approx(1.10)

    def test_exceptional_tier(self):
        """≥ 5.0 bps → multiplier = 1.15."""
        # 10 bps: $1M on $1B
        assert _ceo_purchase_significance(1_000_000, 1_000_000_000) == pytest.approx(1.15)

    def test_exceptional_capped_when_below_half_comp(self):
        """≥ 5 bps but purchase < 50% of annual comp → capped at 1.10 (window dressing)."""
        # 10 bps but ceo_annual_comp = $10M and purchase = $1M < $5M (50% of comp)
        mult = _ceo_purchase_significance(
            ceo_purchase_usd=1_000_000,
            market_cap=1_000_000_000,
            ceo_annual_comp=10_000_000,
        )
        assert mult == pytest.approx(1.10)

    def test_exceptional_not_capped_when_above_half_comp(self):
        """≥ 5 bps AND purchase ≥ 50% of annual comp → stays at 1.15."""
        # 10 bps AND purchase ($1M) >= 50% of $1M comp
        mult = _ceo_purchase_significance(
            ceo_purchase_usd=1_000_000,
            market_cap=1_000_000_000,
            ceo_annual_comp=1_500_000,
        )
        assert mult == pytest.approx(1.15)

    def test_zero_ceo_purchase_returns_one(self):
        assert _ceo_purchase_significance(0.0, 1_000_000_000) == pytest.approx(1.00)

    def test_negative_ceo_purchase_returns_one(self):
        assert _ceo_purchase_significance(-5_000, 1_000_000_000) == pytest.approx(1.00)

    def test_fallback_when_market_cap_zero(self):
        """market_cap=0 → fallback absolute logic, no crash."""
        # Below $50k threshold → 1.00
        m_low = _ceo_purchase_significance(30_000, 0.0)
        # Above $50k threshold → 1.10
        m_high = _ceo_purchase_significance(100_000, 0.0)
        assert m_low == pytest.approx(1.00)
        assert m_high == pytest.approx(1.10)

    def test_fallback_when_market_cap_negative(self):
        """market_cap<0 treated same as unavailable."""
        m = _ceo_purchase_significance(100_000, -1_000_000)
        assert m in (1.00, 1.10)  # one of the two valid fallback values

    def test_conviction_score_uses_relative_multiplier(self):
        """score_insider_conviction integrates _ceo_purchase_significance correctly.

        Two CEOs both buy $100k, same key_purchases_usd, but different market caps.
        The micro-cap CEO (20 bps) must score higher than the mega-cap CEO (0.1 bps).
        """
        # Micro-cap: $100k on $50M → 20 bps → exceptional multiplier
        score_micro = score_insider_conviction(
            key_purchases_usd=100_000,
            market_cap=50_000_000,
            ceo_purchase_usd=100_000,
        )
        # Mega-cap: $100k on $100B → 0.001 bps → no CEO bonus
        score_mega = score_insider_conviction(
            key_purchases_usd=100_000,
            market_cap=100_000_000_000,
            ceo_purchase_usd=100_000,
        )
        assert score_micro > score_mega, (
            f"Micro-cap CEO conviction ({score_micro}) should exceed "
            f"mega-cap CEO conviction ({score_mega}) for identical dollar purchases"
        )
