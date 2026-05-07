"""backend/tests/test_contagion_web.py
Validates Module D — Contagion Web (Leontief / Tirole).

Mathematical validation: Leontief inverse must satisfy (I-A)(I-A)^{-1} = I.
Applied scenario: Energy shock propagation.
"""
import numpy as np
import pytest

from backend.quant_models.contagion_web import (
    SECTORS,
    build_io_matrix,
    leontief_inverse,
    shock_propagation,
    critical_nodes,
    total_gdp_impact,
)


class TestBuildIOMatrix:
    def test_shape(self):
        A = build_io_matrix()
        assert A.shape == (11, 11)

    def test_all_non_negative(self):
        """Leontief (1973 Nobel) — All technical coefficients must be ≥ 0."""
        A = build_io_matrix()
        assert (A >= 0).all()

    def test_column_sums_lt_one(self):
        """Hawkins-Simon condition: each column (sector's input share) must sum < 1."""
        A = build_io_matrix()
        col_sums = A.sum(axis=0)
        assert (col_sums < 1.0).all(), (
            f"Column sums must be < 1 for productive economy. Got: {col_sums}"
        )


class TestLeontiefInverse:
    def test_identity_property(self):
        """Leontief (1973 Nobel) — Mathematical verification: (I-A)·L = I.

        # $(I - A)(I - A)^{-1} = I$
        """
        A = build_io_matrix()
        L = leontief_inverse(A)
        n = A.shape[0]
        product = (np.eye(n) - A) @ L
        np.testing.assert_allclose(product, np.eye(n), atol=1e-10,
                                   err_msg="Leontief inverse identity check failed")

    def test_all_elements_positive(self):
        """Leontief (1973 Nobel) — All multiplier elements must be > 0 (Perron-Frobenius)."""
        A = build_io_matrix()
        L = leontief_inverse(A)
        assert (L > 0).all(), "All Leontief inverse elements must be positive"

    def test_diagonal_gt_one(self):
        """Each sector's own multiplier must be ≥ 1 (own-shock propagation)."""
        A = build_io_matrix()
        L = leontief_inverse(A)
        assert (np.diag(L) >= 1.0).all()

    def test_singular_matrix_raises(self):
        A_singular = np.eye(11)  # (I - I) = 0 matrix → singular
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            leontief_inverse(A_singular)


class TestShockPropagation:
    def test_energy_shock_negative_impact(self):
        """Tirole (2014 Nobel) — A negative demand shock must reduce output."""
        A = build_io_matrix()
        impacts = shock_propagation(A, {"Energy": -0.20})
        assert impacts["Energy"] < 0, "Energy sector must be negatively impacted"

    def test_shock_spreads_to_other_sectors(self):
        """Tirole (2014 Nobel) — Energy shock must propagate to Industrials, Utilities."""
        A = build_io_matrix()
        impacts = shock_propagation(A, {"Energy": -0.20})
        assert impacts["Industrials"] < 0
        assert impacts["Utilities"] < 0

    def test_total_impact_less_than_direct(self):
        """The aggregate impact must be less than a 100% sector wipeout."""
        A = build_io_matrix()
        impacts = shock_propagation(A, {"IT": -0.50})
        gdp_impact = total_gdp_impact(impacts)
        assert gdp_impact < 0
        assert gdp_impact > -100.0

    def test_unknown_sector_ignored(self):
        A = build_io_matrix()
        impacts = shock_propagation(A, {"Semiconductors": -0.20})
        assert all(v == 0.0 for v in impacts.values())


class TestCriticalNodes:
    def test_returns_top_n(self):
        A = build_io_matrix()
        L = leontief_inverse(A)
        nodes = critical_nodes(L, top_n=3)
        assert len(nodes) == 3

    def test_nodes_are_valid_sectors(self):
        A = build_io_matrix()
        L = leontief_inverse(A)
        nodes = critical_nodes(L, top_n=5)
        for n in nodes:
            assert n in SECTORS

    def test_it_or_industrials_in_top_nodes(self):
        """IT and Industrials are highly interconnected — expect them in top multipliers."""
        A = build_io_matrix()
        L = leontief_inverse(A)
        nodes = critical_nodes(L, top_n=5)
        assert any(s in nodes for s in ("IT", "Industrials", "Financials"))
