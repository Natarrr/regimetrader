"""tests/test_cross_sectional.py
Unit tests for cross-sectional factor normalization in generate_top_lists.

Markowitz (1990 Nobel) — portfolio construction requires comparable, bounded
signals. Validates that normalization produces peer-relative scores rather
than absolute thresholds, and that uniform factors don't crash or mislead.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.market_intel.generate_top_lists import _cross_sectional_normalize, FACTOR_FIELDS


def _make_results(n: int, overrides: dict | None = None) -> list:
    """Build n neutral result rows, optionally overriding specific fields."""
    base = {
        "edgar_score":    0.50,
        "insider_score":  0.50,
        "congress_score": 0.50,
        "news_score":     0.50,
        "momentum_score": 0.50,
    }
    rows = [{**base} for _ in range(n)]
    if overrides:
        for key, values in overrides.items():
            for i, v in enumerate(values):
                rows[i][key] = v
    return rows


class TestCrossSectionalNormalize:
    def test_higher_raw_score_gives_higher_normalized_score(self):
        results = _make_results(2, {"edgar_score": [0.30, 0.90]})
        normed  = _cross_sectional_normalize(results)
        assert normed[0]["edgar"] < normed[1]["edgar"]

    def test_normalized_scores_bounded_0_to_1(self):
        results = _make_results(10, {
            "edgar_score": np.random.default_rng(42).uniform(0.3, 0.9, 10).tolist()
        })
        normed = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_all_identical_scores_return_half(self):
        """When all tickers have the same raw score, normalized output is 0.5."""
        results = _make_results(5)   # all 0.50 by default
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert v == pytest.approx(0.5, abs=1e-4)

    def test_all_five_factors_present_in_output(self):
        results = _make_results(3)
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            assert set(row.keys()) == {"edgar", "insider", "congress", "news", "macro"}

    def test_output_length_matches_input(self):
        results = _make_results(7)
        normed  = _cross_sectional_normalize(results)
        assert len(normed) == 7

    def test_single_ticker_returns_neutral(self):
        """One ticker — no peer comparison possible — returns 0.5 for all factors."""
        results = _make_results(1, {"edgar_score": [0.90]})
        normed  = _cross_sectional_normalize(results)
        assert normed[0]["edgar"] == pytest.approx(0.5, abs=1e-4)

    def test_congress_factor_key_present(self):
        """FACTOR_FIELDS maps 'congress' → congress_score."""
        assert FACTOR_FIELDS.get("congress") == "congress_score"
        assert "momentum" not in FACTOR_FIELDS

    def test_macro_factor_present(self):
        """FACTOR_FIELDS maps 'macro' → momentum_score (pipeline field name)."""
        assert FACTOR_FIELDS.get("macro") == "momentum_score"

    def test_2008_crash_outlier_does_not_collapse_scores(self):
        """2020 COVID analog: one ticker with extreme momentum → others not collapsed to 0."""
        scores = [0.50] * 49 + [9999.0]   # 1 extreme outlier
        results = _make_results(50, {"momentum_score": scores})
        normed  = _cross_sectional_normalize(results)
        # The 49 normal tickers should not all map to near-zero
        normal_scores = [normed[i]["macro"] for i in range(49)]
        assert max(normal_scores) > 0.30   # not all collapsed
