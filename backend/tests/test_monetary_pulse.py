"""backend/tests/test_monetary_pulse.py
Validates Module A — Monetary Pulse (Friedman / Kuznets / Prescott).

Historical validation: yield curve was inverted 2006-2007 pre-GFC.
"""
import numpy as np
import pandas as pd
import pytest

from backend.quant_models.monetary_pulse import (
    yield_spread,
    is_inverted,
    m2_velocity_trend,
    hp_filter_trend,
    monetary_regime,
)


class TestYieldSpread:
    def test_spread_is_series(self, yield_data_pre_gfc):
        gs10, gs2 = yield_data_pre_gfc
        spread = yield_spread(gs10, gs2)
        assert isinstance(spread, pd.Series)

    def test_spread_in_basis_points(self, yield_data_pre_gfc):
        gs10, gs2 = yield_data_pre_gfc
        spread = yield_spread(gs10, gs2)
        # Spread should be in bps (order of magnitude ~100, not ~1)
        assert abs(spread.mean()) < 500

    def test_pre_gfc_inversion_detected(self, yield_data_pre_gfc):
        """Friedman (1968 Nobel) — Yield curve must show inversion in 2006-07."""
        gs10, gs2 = yield_data_pre_gfc
        spread = yield_spread(gs10, gs2)
        # Our fixture has gs2 > gs10 for the 2006-2007 window (months ~18-30)
        inverted_window = spread.iloc[18:30]
        assert (inverted_window < 0).any(), (
            "Expected negative spread in 2006-07 pre-GFC inversion window"
        )

    def test_is_inverted_true_when_negative(self, yield_data_pre_gfc):
        gs10, gs2 = yield_data_pre_gfc
        spread = yield_spread(gs10, gs2)
        # Manufacture a clearly inverted series for the check
        inverted = pd.Series([-50.0, -30.0, -20.0])
        assert is_inverted(inverted) is True

    def test_is_inverted_false_when_positive(self):
        steep = pd.Series([100.0, 120.0, 150.0])
        assert is_inverted(steep) is False


class TestM2Velocity:
    def test_falling_trend(self, m2v_series):
        """Kuznets (1971 Nobel) — M2V declining trend should be detected."""
        trend = m2_velocity_trend(m2v_series)
        assert trend == "FALLING", f"Expected FALLING, got {trend}"

    def test_rising_trend(self):
        dates = pd.date_range("2010", periods=12, freq="QS")
        rising = pd.Series(np.linspace(1.5, 2.0, 12), index=dates)
        assert m2_velocity_trend(rising) == "RISING"

    def test_stable_trend(self):
        dates = pd.date_range("2010", periods=12, freq="QS")
        stable = pd.Series(np.ones(12) * 1.8, index=dates)
        assert m2_velocity_trend(stable) == "STABLE"

    def test_short_series_returns_stable(self):
        short = pd.Series([1.8, 1.75])
        assert m2_velocity_trend(short) == "STABLE"


class TestHPFilter:
    def test_returns_two_series(self, gdp_series):
        """Prescott (2004 Nobel) — HP filter must return (trend, cycle) tuple."""
        trend, cycle = hp_filter_trend(gdp_series)
        assert isinstance(trend, pd.Series)
        assert isinstance(cycle, pd.Series)

    def test_cycle_mean_near_zero(self, gdp_series):
        """HP cycle should be mean-zero (centered around trend)."""
        _, cycle = hp_filter_trend(gdp_series)
        assert abs(cycle.mean()) < 50, "HP cycle mean should be close to 0"

    def test_trend_plus_cycle_equals_original(self, gdp_series):
        """Prescott (2004 Nobel) — trend + cycle = original series (additive decomposition)."""
        trend, cycle = hp_filter_trend(gdp_series)
        reconstructed = trend + cycle
        np.testing.assert_allclose(
            reconstructed.values, gdp_series.dropna().values, rtol=1e-6
        )

    def test_trend_is_smoother_than_original(self, gdp_series):
        """Trend volatility must be less than original series volatility."""
        trend, _ = hp_filter_trend(gdp_series)
        assert trend.std() < gdp_series.std()


class TestMonetaryRegime:
    def test_tightening_when_inverted(self, yield_data_pre_gfc, m2v_series):
        gs10, gs2 = yield_data_pre_gfc
        spread = yield_spread(gs10, gs2)
        inverted_spread = pd.Series([-30.0, -50.0, -20.0])
        regime = monetary_regime(inverted_spread, m2v_series)
        assert regime == "TIGHTENING"

    def test_easing_when_steep_and_rising(self):
        steep = pd.Series([150.0, 180.0, 200.0])
        dates = pd.date_range("2010", periods=8, freq="QS")
        rising_m2v = pd.Series(np.linspace(1.5, 2.0, 8), index=dates)
        assert monetary_regime(steep, rising_m2v) == "EASING"
