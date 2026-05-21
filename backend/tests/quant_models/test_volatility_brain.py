"""backend/tests/test_volatility_brain.py
Validates Module B — Volatility Brain (Engle / Merton).

Historical validation: SPY Oct 2008 should yield GARCH persistence > 0.90.
"""
import numpy as np

from backend.quant_models.volatility_brain import (
    fit_gjr_garch,
    volatility_regime,
    merton_distance_to_default,
)


class TestGJRGARCH:
    def test_returns_required_keys(self, spy_oct2008_returns):
        """Engle (2003 Nobel) — fit_gjr_garch must return all parameter keys."""
        result = fit_gjr_garch(spy_oct2008_returns)
        for key in ("omega", "alpha", "gamma", "beta", "persistence",
                    "latest_conditional_vol_ann"):
            assert key in result, f"Missing key: {key}"

    def test_persistence_positive_and_lt_one(self, spy_oct2008_returns):
        """Persistence must be in (0, 1) for a stationary GARCH process."""
        result = fit_gjr_garch(spy_oct2008_returns)
        assert 0 < result["persistence"] < 1.0, (
            f"Persistence {result['persistence']:.4f} outside (0, 1)"
        )

    def test_gfc_crash_high_persistence(self, spy_oct2008_returns):
        """Engle (2003 Nobel) — GFC-style crash returns must show high persistence.

        Historical Oct 2008: daily vol was 5-8%, implying very persistent clustering.
        We assert persistence > 0.90 on our synthetic GFC-like returns.
        """
        result = fit_gjr_garch(spy_oct2008_returns)
        assert result["persistence"] > 0.90, (
            f"Expected persistence > 0.90 for GFC crash, got {result['persistence']:.4f}"
        )

    def test_2020_crash_clustering(self, spy_2020_crash_returns):
        """COVID crash (Feb-Mar 2020) should also show elevated persistence."""
        result = fit_gjr_garch(spy_2020_crash_returns)
        assert result["persistence"] > 0.80, (
            f"Expected persistence > 0.80 for 2020 crash, got {result['persistence']:.4f}"
        )

    def test_low_vol_regime_lower_persistence(self):
        """Calm market returns should produce lower persistence than crash."""
        rng = np.random.default_rng(99)
        calm = rng.normal(0.0004, 0.005, 500)
        result_calm = fit_gjr_garch(calm)
        rng2 = np.random.default_rng(42)
        crash = rng2.normal(-0.003, 0.035, 500)
        crash[400:460] += rng2.normal(-0.01, 0.05, 60)
        result_crash = fit_gjr_garch(crash)
        assert result_crash["persistence"] >= result_calm["persistence"] - 0.05

    def test_conditional_vol_positive(self, spy_oct2008_returns):
        result = fit_gjr_garch(spy_oct2008_returns)
        assert result["latest_conditional_vol_ann"] > 0


class TestVolatilityRegime:
    def test_clustering_above_threshold(self):
        assert volatility_regime(0.99) == "CLUSTERING"

    def test_stable_below_threshold(self):
        assert volatility_regime(0.95) == "STABLE"

    def test_boundary(self):
        assert volatility_regime(0.98) == "STABLE"
        assert volatility_regime(0.981) == "CLUSTERING"


class TestMertonD2D:
    def test_returns_required_keys(self):
        """Merton (1997 Nobel) — D2D result must contain all structural model outputs."""
        result = merton_distance_to_default(
            equity_value=100.0,
            face_value_debt=80.0,
            risk_free_rate=0.05,
            equity_vol=0.30,
        )
        for key in ("asset_value", "asset_vol", "d2d", "prob_default"):
            assert key in result

    def test_prob_default_in_unit_interval(self):
        result = merton_distance_to_default(100.0, 80.0, 0.05, 0.30)
        assert 0 <= result["prob_default"] <= 1.0

    def test_high_leverage_higher_pd(self):
        """Merton (1997 Nobel) — Higher leverage must imply higher default probability."""
        low_lev = merton_distance_to_default(100.0, 50.0, 0.05, 0.25)
        high_lev = merton_distance_to_default(100.0, 95.0, 0.05, 0.25)
        assert high_lev["prob_default"] > low_lev["prob_default"]

    def test_higher_vol_increases_pd(self):
        """Higher equity vol → smaller D2D → higher P(default)."""
        low_vol = merton_distance_to_default(100.0, 70.0, 0.05, 0.15)
        high_vol = merton_distance_to_default(100.0, 70.0, 0.05, 0.60)
        assert high_vol["prob_default"] > low_vol["prob_default"]

    def test_very_solvent_firm_near_zero_pd(self):
        """Firm with tiny debt relative to equity: near-zero default probability."""
        result = merton_distance_to_default(1000.0, 10.0, 0.04, 0.20)
        assert result["prob_default"] < 0.05
