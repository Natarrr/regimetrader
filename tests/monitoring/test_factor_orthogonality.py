"""tests/monitoring/test_factor_orthogonality.py — Fix #8: factor orthogonality monitoring.

Validates that compute_factor_correlation_matrix() correctly detects redundant
factors (ρ ≈ 1.0 pair) while reporting low correlation for genuinely independent factors.
"""
from __future__ import annotations

import math
import random

import pytest

from regime_trader.monitoring.factor_orthogonality import (
    CORRELATION_ERROR_THRESHOLD,
    CORRELATION_WARN_THRESHOLD,
    compute_factor_correlation_matrix,
)


def _make_results(
    n: int,
    *,
    seed: int = 42,
    include_duplicate_pair: bool = False,
) -> list[dict]:
    """
    Build a synthetic results list with 5 named factors for n tickers.

    If include_duplicate_pair=True, factor_1 and factor_2 are identical (ρ=1.0).
    Factors 3, 4, 5 are independently random (expected |ρ| << 0.3 for n=50).
    All rows carry market="USA" and pre-populated *_score_neutral columns.
    """
    rng = random.Random(seed)

    rows = []
    for i in range(n):
        f3 = rng.random()
        f4 = rng.random()
        f5 = rng.random()
        f1 = rng.random()
        f2 = f1 if include_duplicate_pair else rng.random()  # identical to f1 when flag set

        rows.append({
            "ticker": f"T{i:04d}",
            "market": "USA",
            # Raw scores
            "factor_1_score": f1,
            "factor_2_score": f2,
            "factor_3_score": f3,
            "factor_4_score": f4,
            "factor_5_score": f5,
            # Neutral columns (Fix #1): same values in this synthetic fixture
            "factor_1_score_neutral": f1,
            "factor_2_score_neutral": f2,
            "factor_3_score_neutral": f3,
            "factor_4_score_neutral": f4,
            "factor_5_score_neutral": f5,
        })
    return rows


_TEST_FACTORS = [
    "factor_1_score",
    "factor_2_score",
    "factor_3_score",
    "factor_4_score",
    "factor_5_score",
]


class TestOrthogonalityDetectsRedundantFactors:
    def test_detects_duplicate_pair_as_error(self):
        """factor_1 == factor_2 → Spearman ρ = 1.0 → must appear in errors."""
        rows = _make_results(50, include_duplicate_pair=True)
        report = compute_factor_correlation_matrix(
            rows,
            factors=_TEST_FACTORS,
            use_neutralized=True,
            market_filter="US",
            min_observations=30,
        )
        assert "error" not in report, f"Unexpected failure: {report.get('error')}"
        assert report["max_abs_correlation"] >= 0.95, (
            f"Expected max |ρ| ≥ 0.95 for identical factors, got {report['max_abs_correlation']}"
        )
        assert report["max_pair"] == ["factor_1_score", "factor_2_score"] or \
               report["max_pair"] == ["factor_2_score", "factor_1_score"], (
            f"Expected max_pair to be the duplicate pair, got {report['max_pair']}"
        )
        assert len(report["errors"]) >= 1, "Expected at least one error for ρ=1.0 pair"
        assert any("factor_1_score" in e and "factor_2_score" in e for e in report["errors"]), (
            f"Error message should name both factors; got: {report['errors']}"
        )

    def test_independent_factors_have_low_correlation(self):
        """Factors 3, 4, 5 are random and independent — all pairwise |ρ| should be < 0.3."""
        rows = _make_results(50, include_duplicate_pair=False)
        report = compute_factor_correlation_matrix(
            rows,
            factors=_TEST_FACTORS,
            use_neutralized=True,
            market_filter="US",
            min_observations=30,
        )
        assert "error" not in report
        matrix = report["correlation_matrix"]
        for fi in ["factor_3_score", "factor_4_score", "factor_5_score"]:
            for fj in ["factor_3_score", "factor_4_score", "factor_5_score"]:
                if fi == fj:
                    continue
                rho = matrix[fi][fj]
                assert abs(rho) < 0.3, (
                    f"Independent factors {fi} and {fj} have |ρ|={abs(rho):.3f} ≥ 0.3 "
                    f"— orthogonality not preserved for random factors"
                )

    def test_duplicate_pair_does_not_contaminate_independent_pairs(self):
        """The ρ=1.0 pair should not cause factor 3-5 pairwise ρ to spike."""
        rows = _make_results(50, include_duplicate_pair=True)
        report = compute_factor_correlation_matrix(
            rows,
            factors=_TEST_FACTORS,
            use_neutralized=True,
            market_filter="US",
        )
        matrix = report["correlation_matrix"]
        for fi in ["factor_3_score", "factor_4_score", "factor_5_score"]:
            for fj in ["factor_3_score", "factor_4_score", "factor_5_score"]:
                if fi == fj:
                    continue
                rho = matrix[fi][fj]
                assert abs(rho) < 0.5, (
                    f"Independent pair {fi}↔{fj} incorrectly shows |ρ|={abs(rho):.3f}"
                )

    def test_no_errors_for_fully_independent_factors(self):
        """All independent factors → no errors, no warnings."""
        rows = _make_results(60, include_duplicate_pair=False, seed=123)
        report = compute_factor_correlation_matrix(
            rows,
            factors=_TEST_FACTORS,
            use_neutralized=True,
            market_filter="US",
        )
        assert report["errors"] == [], f"Unexpected errors: {report['errors']}"

    def test_returns_all_required_keys(self):
        """Report dict must contain all documented keys."""
        rows = _make_results(50)
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US"
        )
        required = {
            "computed_at", "market", "n_observations", "factors_evaluated",
            "factors_skipped", "correlation_matrix", "max_abs_correlation",
            "max_pair", "warnings", "errors",
        }
        assert required.issubset(report.keys()), (
            f"Missing keys: {required - report.keys()}"
        )

    def test_diagonal_is_one(self):
        """correlation_matrix[f][f] == 1.0 for all evaluated factors."""
        rows = _make_results(50)
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US"
        )
        for f in report["factors_evaluated"]:
            assert report["correlation_matrix"][f][f] == pytest.approx(1.0), (
                f"Diagonal for {f} should be 1.0"
            )

    def test_returns_error_dict_when_below_min_observations(self):
        """Fewer rows than min_observations → error dict, no crash."""
        rows = _make_results(10)  # only 10 rows
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US", min_observations=30
        )
        assert "error" in report
        assert "computed_at" in report

    def test_non_blocking_on_empty_results(self):
        """Empty results list → error dict, pipeline must not crash."""
        report = compute_factor_correlation_matrix(
            [], factors=_TEST_FACTORS, market_filter="US"
        )
        assert "error" in report

    def test_market_filter_us_accepts_legacy_usa(self):
        """Rows with market='USA' are included when market_filter='US'."""
        rows = _make_results(40, include_duplicate_pair=False)
        # All rows have market='USA' by construction in _make_results
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US", min_observations=30
        )
        assert "error" not in report
        assert report["n_observations"] == 40

    def test_n_observations_matches_filtered_rows(self):
        """n_observations reflects the post-filter count, not total len(results)."""
        rows = _make_results(40)
        # Add some EUROPE rows that should be excluded
        for i in range(10):
            rows.append({
                "ticker": f"EU{i:04d}",
                "market": "EUROPE",
                **{f"{f}_score_neutral": 0.5 for f in ["factor_1_score", "factor_2_score",
                                                         "factor_3_score", "factor_4_score",
                                                         "factor_5_score"]},
                **{f"{f}": 0.5 for f in ["factor_1_score", "factor_2_score",
                                          "factor_3_score", "factor_4_score",
                                          "factor_5_score"]},
            })
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US", min_observations=30
        )
        assert report["n_observations"] == 40
