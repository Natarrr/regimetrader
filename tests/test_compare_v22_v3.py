"""tests/test_compare_v22_v3.py — shadow-window comparator (migration step 5 gates).

Pure-function tests: Spearman rank correlation, top-N overlap, sparsity map,
rank-turnover inflation. The CLI wrapper just orchestrates these.
"""
from __future__ import annotations

import pytest

from tools.compare_v22_v3 import (
    rank_turnover,
    sparsity_map,
    spearman,
    top_n_overlap,
)


class TestSpearman:
    def test_perfect_agreement(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_inversion(self):
        assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_short_input_is_none(self):
        assert spearman([1], [2]) is None

    def test_constant_series_is_none(self):
        assert spearman([1, 1, 1], [1, 2, 3]) is None


class TestTopNOverlap:
    def test_full_overlap(self):
        rows = [{"ticker": f"T{i}", "a": 10 - i, "b": 10 - i} for i in range(10)]
        assert top_n_overlap(rows, "a", "b", n=5) == pytest.approx(1.0)

    def test_disjoint_tops(self):
        rows = [{"ticker": f"T{i}", "a": 10 - i, "b": i} for i in range(10)]
        assert top_n_overlap(rows, "a", "b", n=3) == pytest.approx(0.0)


class TestSparsityMap:
    def test_counts_none_and_dead(self):
        rows = [
            {"market": "USA", "pead_surprise_score": None,
             "insider_alpha_score": 0.0},
            {"market": "USA", "pead_surprise_score": 0.6,
             "insider_alpha_score": 0.4},
        ]
        result = sparsity_map(rows, ["pead_surprise", "insider_alpha"])
        usa = result["USA"]
        assert usa["pead_surprise"]["none_rate"] == pytest.approx(0.5)
        assert usa["insider_alpha"]["dead_rate"] == pytest.approx(0.5)

    def test_prefers_v3_column_for_collision_factors(self):
        # US rows carry BOTH analyst_revision_score (v2.2, dead-coerced 0.0)
        # and analyst_revision_score_v3 (None = unavailable). The shadow
        # gates must read the v3 semantics, not v2.2's downward coercion.
        rows = [
            {"market": "USA", "analyst_revision_score": 0.0,
             "analyst_revision_score_v3": None},
        ]
        result = sparsity_map(rows, ["analyst_revision"])
        usa = result["USA"]
        assert usa["analyst_revision"]["none_rate"] == pytest.approx(1.0)
        assert usa["analyst_revision"]["dead_rate"] == pytest.approx(0.0)


class TestRankTurnover:
    def _rows(self, scores, key="final_score_v3"):
        return [{"ticker": f"T{i}", key: s} for i, s in enumerate(scores)]

    def test_stable_ranks_full_autocorrelation(self):
        prev = self._rows([0.9, 0.8, 0.7, 0.6, 0.5])
        curr = self._rows([0.95, 0.85, 0.75, 0.65, 0.55])
        result = rank_turnover(prev, curr, "final_score_v3", top_n=3)
        assert result["rank_autocorrelation"] == pytest.approx(1.0)
        assert result["top_n_churn"] == pytest.approx(0.0)

    def test_full_reversal_detected(self):
        prev = self._rows([0.9, 0.8, 0.7, 0.6, 0.5])
        curr = self._rows([0.5, 0.6, 0.7, 0.8, 0.9])
        result = rank_turnover(prev, curr, "final_score_v3", top_n=2)
        assert result["rank_autocorrelation"] == pytest.approx(-1.0)
        assert result["top_n_churn"] == pytest.approx(1.0)

    def test_missing_tickers_intersected(self):
        prev = self._rows([0.9, 0.8, 0.7])
        curr = self._rows([0.9, 0.8])  # T2 dropped out
        result = rank_turnover(prev, curr, "final_score_v3", top_n=2)
        assert result["common_tickers"] == 2
