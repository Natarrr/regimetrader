"""tests/scoring/test_consensus_signals.py — v3.0 consensus momentum scorers.

Tests:
  1. analyst_revision: centered at 0.5, damping pulls toward 0.5 (not 0.0),
     None below the coverage gate.
  2. revision_velocity: second derivative of the revision path, None on
     short/degenerate estimate histories.
  3. pead_surprise: Bernard & Thomas decay at the 59/60/89/90/91-day
     boundaries, None when drift is exhausted.

No live network calls. Pure scorer math.
"""
from __future__ import annotations

import pytest

from src.scoring.consensus_signals import (
    score_analyst_revision,
    score_pead_surprise,
    score_revision_velocity,
)


class TestAnalystRevision:
    def test_none_revision_is_none(self):
        assert score_analyst_revision(None, 10) is None

    def test_below_coverage_gate_is_none(self):
        assert score_analyst_revision(0.10, 2) is None

    def test_zero_revision_is_neutral(self):
        assert score_analyst_revision(0.0, 20) == pytest.approx(0.5)

    def test_full_coverage_full_positive(self):
        # clip(0.30)/0.60 · min(1, 10/10) → 0.5 + 0.5 = 1.0
        assert score_analyst_revision(0.30, 10) == pytest.approx(1.0)

    def test_damping_pulls_toward_half_not_zero(self):
        # n=5 → damp 0.5: 0.5 + 0.5·0.5 = 0.75 (a thin-coverage positive
        # revision must NOT be dragged toward bearish 0.0)
        assert score_analyst_revision(0.30, 5) == pytest.approx(0.75)

    def test_negative_revision_clips_to_zero(self):
        # clip(−0.60 → −0.30)/0.60 → 0.5 − 0.5 = 0.0 (a REAL signed
        # observation, not a dead signal — zero_is_dead=False downstream)
        assert score_analyst_revision(-0.60, 10) == pytest.approx(0.0)


def _est(eps_avg, n=8):
    return {"estimatedEpsAvg": eps_avg, "numberAnalystEstimatedEps": n}


class TestRevisionVelocity:
    def test_short_history_is_none(self):
        assert score_revision_velocity([_est(1.0), _est(1.0), _est(1.0)]) is None

    def test_accelerating_revisions(self):
        # rev_now = (1.2−1.0)/1.0 = +0.2; rev_prev = (1.0−1.25)/1.25 = −0.2
        # vel = +0.4 → clip +0.30 → 0.5 + 0.5·min(1, 8/8) = 1.0
        rows = [_est(1.2, 8), _est(1.0, 8), _est(1.0, 8), _est(1.25, 8)]
        assert score_revision_velocity(rows) == pytest.approx(1.0)

    def test_thin_coverage_damps_toward_half(self):
        rows = [_est(1.2, 4), _est(1.0, 4), _est(1.0, 4), _est(1.25, 4)]
        # damp = min(1, 4/8) = 0.5 → 0.5 + 0.5·0.5 = 0.75
        assert score_revision_velocity(rows) == pytest.approx(0.75)

    def test_zero_base_is_none(self):
        rows = [_est(1.2), _est(1.0), _est(0.0), _est(1.25)]
        assert score_revision_velocity(rows) is None

    def test_flat_path_is_neutral(self):
        rows = [_est(1.0), _est(1.0), _est(1.0), _est(1.0)]
        assert score_revision_velocity(rows) == pytest.approx(0.5)


class TestPeadSurprise:
    def test_missing_surprise_is_none(self):
        assert score_pead_surprise(None, 10) is None

    def test_full_strength_inside_60_days(self):
        # base = clip(0.10) + 0.5 = 0.60; decay = 1 → 0.60
        assert score_pead_surprise(0.10, 30) == pytest.approx(0.60)

    @pytest.mark.parametrize("days,expected", [
        (59, 0.60),                      # full strength
        (60, 0.60),                      # boundary: still full strength
        (89, 0.5 + 0.10 * (1 / 30)),     # decay = (90−89)/30
        (90, 0.50),                      # drift exhausted → neutral
    ])
    def test_decay_boundaries(self, days, expected):
        assert score_pead_surprise(0.10, days) == pytest.approx(expected, abs=1e-6)

    def test_past_90_days_is_none(self):
        # Exhausted drift is "no information", NOT bearish.
        assert score_pead_surprise(0.10, 91) is None

    def test_surprise_clipped_at_half(self):
        # +200% surprise clips to +0.50 → base 1.0 → score 1.0
        assert score_pead_surprise(2.0, 10) == pytest.approx(1.0)

    def test_negative_surprise_symmetric(self):
        # −20% surprise: base = 0.30 → 0.5 + (0.30−0.5)·1 = 0.30
        assert score_pead_surprise(-0.20, 30) == pytest.approx(0.30)
