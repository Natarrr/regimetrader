"""WS5 — price_target_upside signed-contract sentinel.

`price_target_upside` ∈ SIGNED_FACTORS (src/config/factor_matrix.py). Per
CLAUDE.md §2, a SIGNED factor's absence must read as None ("unavailable"), never
0.0 — a 0.0 is a *real* observation of 50%+ downside, so coercing absence to 0.0
silently marks every uncovered ticker maximally bearish. This pins the scorer's
None contract and proves the v3 neutralizer reweights None instead of treating it
as a bearish dead 0.0.
"""
from __future__ import annotations

import math

import pytest

from src.config.factor_matrix import SIGNED_FACTORS
from src.scoring.momentum_signals import score_price_target_upside
from src.scoring.neutralization import neutralize_factors


# ── Scorer contract ─────────────────────────────────────────────────────────

class TestScorerSentinel:
    def test_factor_is_registered_signed(self):
        assert "price_target_upside" in SIGNED_FACTORS

    def test_none_target_returns_none(self):
        assert score_price_target_upside(None, 100.0) is None

    def test_none_price_returns_none(self):
        assert score_price_target_upside(120.0, None) is None

    def test_zero_target_returns_none(self):
        assert score_price_target_upside(0.0, 100.0) is None

    def test_zero_price_returns_none(self):
        assert score_price_target_upside(120.0, 0.0) is None

    def test_nan_returns_none(self):
        assert score_price_target_upside(float("nan"), 100.0) is None

    def test_non_numeric_returns_none(self):
        assert score_price_target_upside("n/a", 100.0) is None

    def test_valid_upside_scores_above_half(self):
        # 20% upside → (0.20 + 0.50) = 0.70
        assert score_price_target_upside(120.0, 100.0) == pytest.approx(0.70)

    def test_at_target_scores_half(self):
        assert score_price_target_upside(100.0, 100.0) == pytest.approx(0.50)

    def test_downside_scores_below_half(self):
        # genuine 20% downside is a REAL observation, not absence → 0.30, not None
        s = score_price_target_upside(80.0, 100.0)
        assert s == pytest.approx(0.30)
        assert s is not None


# ── v3 neutralizer regression ───────────────────────────────────────────────

_FACTOR = "price_target_upside_score"


def _peers():
    """Six covered large-cap Tech peers with real upside scores (one bucket)."""
    vals = [0.80, 0.70, 0.75, 0.65, 0.85, 0.60]
    return [
        {"ticker": f"PEER{i}", "market": "USA", "sector": "Tech",
         "cap_tier": "large", _FACTOR: v}
        for i, v in enumerate(vals)
    ]


def _subject(value):
    return {"ticker": "SUBJ", "market": "USA", "sector": "Tech",
            "cap_tier": "large", _FACTOR: value}


def _run(rows):
    return neutralize_factors(
        rows,
        factors=(_FACTOR,),
        min_bucket_size=5,
        none_passthrough=True,
        zero_is_dead={_FACTOR: False},   # signed: matches engine_v3 production wiring
    )


class TestNeutralizerTreatsNoneAsUnavailable:
    def test_missing_pt_is_emitted_as_none_not_bearish(self):
        out = _run(_peers() + [_subject(None)])
        subj = next(r for r in out if r["ticker"] == "SUBJ")
        # Absence → reweighted, never a bearish floor.
        assert subj[f"{_FACTOR}_neutral"] is None
        assert subj["_neutralization_fallback"] == "none"

    def test_genuine_zero_downside_enters_stats_as_bearish(self):
        # The OLD scorer returned 0.0 on absence; this proves that a *real* 0.0
        # is (correctly) bearish — i.e. None and 0.0 are now distinct outcomes.
        out = _run(_peers() + [_subject(0.0)])
        subj = next(r for r in out if r["ticker"] == "SUBJ")
        neutral = subj[f"{_FACTOR}_neutral"]
        assert neutral is not None
        assert neutral < 0.5                      # below peers → bearish
        assert subj["_neutralization_fallback"] != "none"

    def test_none_does_not_perturb_peer_statistics(self):
        """A None subject must be excluded from bucket μ/σ, so peers neutralize
        identically whether the subject is present-as-None or absent entirely."""
        with_none = _run(_peers() + [_subject(None)])
        without = _run(_peers())
        for tkr in (f"PEER{i}" for i in range(6)):
            a = next(r for r in with_none if r["ticker"] == tkr)[f"{_FACTOR}_neutral"]
            b = next(r for r in without if r["ticker"] == tkr)[f"{_FACTOR}_neutral"]
            assert a == pytest.approx(b)
