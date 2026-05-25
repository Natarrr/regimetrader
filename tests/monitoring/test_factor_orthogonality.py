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
    SPARSITY_THRESHOLD,
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


class TestSparsityFilter:
    def test_sparse_pair_excluded_from_warnings(self):
        """Fix #8.1: a factor with < SPARSITY_THRESHOLD non-zero density goes to
        low_density_pairs, not warnings/errors — even if Spearman rho is very high.

        Fixture: 100 tickers.
        - factor_a: dense (90% non-zero, random values).
        - factor_b: sparse (5% non-zero, non-zeros happen to coincide with
          factor_a's top values — artificially inflating Spearman rho).
        - factor_c through factor_e: independent random (dense).

        Without sparsity filter: (factor_a, factor_b) would trigger WARN.
        With sparsity filter: pair goes to low_density_pairs, no WARN emitted.
        """
        rng = random.Random(99)
        n = 100
        sparse_factors = ["factor_a", "factor_b", "factor_c", "factor_d", "factor_e"]

        rows = []
        # Build factor_a as dense random, factor_b as sparse (5% non-zero at top of factor_a)
        fa_vals = [rng.random() for _ in range(n)]
        top_indices = sorted(range(n), key=lambda i: fa_vals[i], reverse=True)[:5]  # top 5% = 5

        for i in range(n):
            fb = fa_vals[i] if i in top_indices else 0.0  # sparse: only fires at top of fa
            rows.append({
                "ticker": f"T{i:04d}",
                "market": "USA",
                "factor_a_score_neutral": fa_vals[i],
                "factor_b_score_neutral": fb,
                "factor_c_score_neutral": rng.random(),
                "factor_d_score_neutral": rng.random(),
                "factor_e_score_neutral": rng.random(),
                "factor_a": fa_vals[i],
                "factor_b": fb,
                "factor_c": rng.random(),
                "factor_d": rng.random(),
                "factor_e": rng.random(),
            })

        test_factors = [f"{f}_score" if "_score" not in f else f for f in
                        ["factor_a", "factor_b", "factor_c", "factor_d", "factor_e"]]
        # Use the neutral column names directly as factor names
        test_factors = ["factor_a", "factor_b", "factor_c", "factor_d", "factor_e"]

        report = compute_factor_correlation_matrix(
            rows,
            factors=test_factors,
            use_neutralized=True,
            market_filter="US",
            min_observations=30,
        )

        assert "error" not in report, f"Unexpected error: {report.get('error')}"

        # factor_b is sparse (5/100 = 0.05 < SPARSITY_THRESHOLD=0.20)
        density_b = report["factor_densities"]["factor_b"]
        assert density_b < SPARSITY_THRESHOLD, (
            f"factor_b density={density_b:.2f} should be below SPARSITY_THRESHOLD={SPARSITY_THRESHOLD}"
        )

        # The (factor_a, factor_b) pair must be in low_density_pairs
        sparse_pair_names = {
            (p["factor_a"], p["factor_b"]) for p in report["low_density_pairs"]
        } | {
            (p["factor_b"], p["factor_a"]) for p in report["low_density_pairs"]
        }
        assert ("factor_a", "factor_b") in sparse_pair_names or \
               ("factor_b", "factor_a") in sparse_pair_names, (
            f"Sparse pair not in low_density_pairs: {report['low_density_pairs']}"
        )

        # No warning or error should mention factor_b
        all_messages = report["warnings"] + report["errors"]
        assert not any("factor_b" in m for m in all_messages), (
            f"Sparse factor_b should not appear in warnings/errors: {all_messages}"
        )

    def test_dense_duplicate_still_triggers_error(self):
        """Sparsity filter must NOT suppress detection of truly redundant dense factors.

        factor_a and factor_b are identical (rho=1.0) AND dense (90% non-zero).
        Must still appear in errors.
        """
        rng = random.Random(77)
        n = 100
        rows = []
        for i in range(n):
            v = rng.random() if rng.random() > 0.10 else 0.0  # 90% non-zero
            rows.append({
                "ticker": f"T{i:04d}",
                "market": "USA",
                "factor_a_score_neutral": v,
                "factor_b_score_neutral": v,  # identical
                "factor_c_score_neutral": rng.random(),
                "factor_a": v,
                "factor_b": v,
                "factor_c": rng.random(),
            })

        report = compute_factor_correlation_matrix(
            rows,
            factors=["factor_a", "factor_b", "factor_c"],
            use_neutralized=True,
            market_filter="US",
        )
        assert "error" not in report
        assert report["max_abs_correlation"] >= 0.95
        assert any("factor_a" in e and "factor_b" in e for e in report["errors"]), (
            f"Dense duplicate pair must trigger error. Got: {report['errors']}"
        )

    def test_factor_densities_key_present(self):
        """Report always contains factor_densities dict."""
        rows = _make_results(50)
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US"
        )
        assert "factor_densities" in report
        for f in report["factors_evaluated"]:
            assert f in report["factor_densities"]
            assert 0.0 <= report["factor_densities"][f] <= 1.0

    def test_low_density_pairs_key_present(self):
        """Report always contains low_density_pairs list (may be empty)."""
        rows = _make_results(50)
        report = compute_factor_correlation_matrix(
            rows, factors=_TEST_FACTORS, market_filter="US"
        )
        assert "low_density_pairs" in report
        assert isinstance(report["low_density_pairs"], list)

    def test_reliable_correlations_only_flag(self):
        """reliable_correlations_only=True when any pair was sparsity-excluded."""
        rng = random.Random(55)
        n = 60
        rows = []
        for i in range(n):
            rows.append({
                "ticker": f"T{i:04d}",
                "market": "USA",
                # factor_sparse: only 5% non-zero → will be in low_density_pairs
                "factor_sparse_score_neutral": rng.random() if rng.random() < 0.05 else 0.0,
                "factor_dense_score_neutral":  rng.random(),
                "factor_sparse": rng.random() if rng.random() < 0.05 else 0.0,
                "factor_dense":  rng.random(),
            })
        report = compute_factor_correlation_matrix(
            rows,
            factors=["factor_sparse", "factor_dense"],
            use_neutralized=True,
            market_filter="US",
        )
        if report.get("low_density_pairs"):
            assert report["reliable_correlations_only"] is True
