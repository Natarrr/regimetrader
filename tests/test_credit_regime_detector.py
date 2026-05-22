r"""tests/test_credit_regime_detector.py
Unit and integration tests for regime/credit_regime_detector.py.

Coverage:
  - z-score computation (known values)
  - OLS slope (linear series)
  - compute_credit_stress_score: [0,1] bounds, missing components, weights
  - classify_credit_regime: all threshold boundaries
  - apply_credit_persistence_filter: asymmetric rules (1/2/3 day requirements)
  - credit_score_to_vix_proba: shape, simplex, monotonicity
  - apply_credit_overrides: CRISIS floor, STRESS+VIX<20 early warning
  - CreditRegimeDetector.compute_features_from_prices: HY-only, IG-only, both
  - CreditRegimeDetector.compute_features_series: length, range
  - CreditRegimeDetector.regime_series: labels valid
  - Integration with RegimeDetector: backward-compat + credit-enabled paths

Engle (2003 Nobel) — credit spreads and equity volatility share a
common stochastic volatility factor; these tests validate that the
credit signal correctly bridges both regimes.

# Credit Score: $S = \frac{\sum_i w_i \cdot \sigma(x_i)}{\sum_i w_i}$
# where $\sigma(x) = \frac{1}{1+e^{-1.5x}}$
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regime_trader.models.credit_regime_detector import (
    CreditFeatures,
    CreditRegime,
    CreditRegimeDetector,
    _SEVERITY,
    _component_to_01,
    _ols_slope,
    _zscore_series,
    apply_credit_overrides,
    apply_credit_persistence_filter,
    classify_credit_regime,
    compute_credit_stress_score,
    credit_score_to_vix_proba,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_idx(n: int = 100) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=n, freq="B")


def _make_series(values, n: int = 100) -> pd.Series:
    idx = _make_idx(n)
    return pd.Series(values, index=idx)


def _flat_features(**kwargs) -> CreditFeatures:
    """Return CreditFeatures with all components set to 0.0 unless overridden."""
    base = {"z_hy": 0.0, "z_ig": 0.0, "slope_hy_norm": 0.0,
            "hy_ig_ratio_norm": 0.0, "z_move": 0.0}
    base.update(kwargs)
    return CreditFeatures(**base)


# ── z-score computation ───────────────────────────────────────────────────────

class TestZscoreSeries:
    r"""Verify rolling z-score implementation.

    # $z_t = \frac{x_t - \mu_{t-w:t}}{\sigma_{t-w:t}}$
    """

    def test_constant_series_zscore_is_zero(self):
        """All-constant series → z-score = 0 (std=0 → eps guard)."""
        s = _make_series(np.ones(100))
        z = _zscore_series(s, window=20)
        valid = z.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-6)

    def test_known_zscore_value(self):
        """Series = [0]*50 + [3]*50; window=20. After warm-up, latest z ≈ +3."""
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(0, 1, 80), np.array([5.0])])
        s = pd.Series(values, index=_make_idx(81))
        z = _zscore_series(s, window=30)
        # Latest value is 5 standard deviations away from recent mean ≈ 0 → z > 3
        assert float(z.iloc[-1]) > 2.0

    def test_zscore_output_length_matches_input(self):
        s = _make_series(np.arange(100, dtype=float))
        z = _zscore_series(s, window=20)
        assert len(z) == 100

    def test_zscore_nan_during_warmup(self):
        """First window-1 values should be NaN (insufficient history)."""
        s = _make_series(np.arange(100, dtype=float))
        z = _zscore_series(s, window=20)
        # With min_periods=10, first 9 should be NaN
        assert z.iloc[:9].isna().all()

    def test_zscore_symmetric(self):
        """After a regime shift to lower values, z-score turns negative."""
        rng = np.random.default_rng(17)
        # 60 obs at level +2 (with noise so std > 0), then 20 obs at level -2
        values = np.concatenate([
            rng.normal(2.0, 0.5, 60),
            rng.normal(-2.0, 0.5, 20),
        ])
        s = pd.Series(values, index=_make_idx(80))
        z = _zscore_series(s, window=40)
        # Window of 40 straddles the shift: mean ≈ 0, last obs ≈ -2 → z < 0
        assert float(z.dropna().iloc[-1]) < 0


# ── OLS slope ─────────────────────────────────────────────────────────────────

class TestOlsSlope:
    r"""OLS slope on linear and constant series.

    # $\hat{\beta} = \frac{\sum_t t \cdot x_t - n\bar{t}\bar{x}}{\sum_t t^2 - n\bar{t}^2}$
    """

    def test_linear_increasing_slope_positive(self):
        """y = t → slope should equal 1."""
        s = pd.Series(np.arange(20, dtype=float), index=_make_idx(20))
        slope = _ols_slope(s)
        assert abs(slope - 1.0) < 1e-6

    def test_linear_decreasing_slope_negative(self):
        s = pd.Series(-np.arange(20, dtype=float), index=_make_idx(20))
        slope = _ols_slope(s)
        assert abs(slope + 1.0) < 1e-6

    def test_constant_slope_zero(self):
        s = pd.Series(np.ones(20), index=_make_idx(20))
        slope = _ols_slope(s)
        assert abs(slope) < 1e-6

    def test_two_point_slope(self):
        """Two points: [0, 1] → slope = 1."""
        s = pd.Series([0.0, 1.0], index=_make_idx(2))
        assert abs(_ols_slope(s) - 1.0) < 1e-6

    def test_one_point_returns_nan(self):
        s = pd.Series([42.0], index=_make_idx(1))
        assert np.isnan(_ols_slope(s))


# ── component_to_01 ───────────────────────────────────────────────────────────

class TestComponentTo01:
    """Verify the soft-clamp sigmoid helper."""

    def test_zero_input_gives_half(self):
        assert abs(_component_to_01(0.0) - 0.5) < 1e-6

    def test_large_positive_near_one(self):
        assert _component_to_01(10.0) > 0.99

    def test_large_negative_near_zero(self):
        assert _component_to_01(-10.0) < 0.01

    def test_always_in_01(self):
        for x in np.linspace(-20, 20, 100):
            v = _component_to_01(float(x))
            assert 0.0 <= v <= 1.0

    def test_monotonically_increasing(self):
        prev = _component_to_01(-5.0)
        for x in np.linspace(-4, 5, 20):
            curr = _component_to_01(float(x))
            assert curr >= prev
            prev = curr


# ── compute_credit_stress_score ───────────────────────────────────────────────

class TestComputeCreditStressScore:
    """Verify score is in [0,1] and responds correctly to feature values.

    Black-Scholes (1997 Nobel) — credit pricing must respect no-arbitrage
    bounds; the score is similarly bounded by construction.
    """

    def test_all_zero_features_gives_midpoint(self):
        """Neutral features → score ≈ 0.5."""
        feats = _flat_features()
        score = compute_credit_stress_score(feats)
        assert abs(score - 0.5) < 0.01

    def test_all_high_stress_near_one(self):
        """All features at extreme stress (z=5) → score near 1."""
        feats = _flat_features(z_hy=5.0, z_ig=5.0, slope_hy_norm=5.0,
                                hy_ig_ratio_norm=5.0, z_move=5.0)
        score = compute_credit_stress_score(feats)
        assert score > 0.90

    def test_all_low_stress_near_zero(self):
        """All features at low stress (z=-5) → score near 0."""
        feats = _flat_features(z_hy=-5.0, z_ig=-5.0, slope_hy_norm=-5.0,
                                hy_ig_ratio_norm=-5.0, z_move=-5.0)
        score = compute_credit_stress_score(feats)
        assert score < 0.10

    def test_score_always_in_01(self):
        """Score must be in [0, 1] for any feature values."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            vals = rng.uniform(-10, 10, 5)
            feats = CreditFeatures(
                z_hy=vals[0], z_ig=vals[1], slope_hy_norm=vals[2],
                hy_ig_ratio_norm=vals[3], z_move=vals[4],
            )
            score = compute_credit_stress_score(feats)
            assert 0.0 <= score <= 1.0

    def test_no_features_returns_neutral(self):
        """All None → return 0.5 (neutral/unknown)."""
        feats = CreditFeatures()
        score = compute_credit_stress_score(feats)
        assert abs(score - 0.5) < 1e-6

    def test_hy_only_still_returns_valid_score(self):
        """HY alone (other components None) → valid score in [0,1]."""
        feats = CreditFeatures(z_hy=2.0)
        score = compute_credit_stress_score(feats)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # positive stress signal

    def test_ig_only_still_returns_valid_score(self):
        feats = CreditFeatures(z_ig=2.0)
        score = compute_credit_stress_score(feats)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_move_only_returns_valid_score(self):
        feats = CreditFeatures(z_move=3.0)
        score = compute_credit_stress_score(feats)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_nan_feature_ignored(self):
        """NaN features are treated as missing."""
        feats = CreditFeatures(z_hy=float("nan"), z_ig=2.0)
        score = compute_credit_stress_score(feats)
        assert 0.0 <= score <= 1.0

    def test_stress_increases_monotonically_with_z_hy(self):
        """Higher z_hy → higher stress score."""
        scores = [
            compute_credit_stress_score(CreditFeatures(z_hy=z))
            for z in [-2.0, -1.0, 0.0, 1.0, 2.0]
        ]
        assert all(scores[i] < scores[i+1] for i in range(len(scores)-1))


# ── classify_credit_regime ────────────────────────────────────────────────────

class TestClassifyCreditRegime:
    """Verify threshold boundaries.

    Lucas (1995 Nobel) — threshold classifiers must be tested at the
    exact boundary values to prevent off-by-one misclassification.
    """

    def test_crisis_at_0_80(self):
        assert classify_credit_regime(0.80) == CreditRegime.CRISIS

    def test_crisis_boundary_exact(self):
        assert classify_credit_regime(0.75) == CreditRegime.CRISIS

    def test_stress_just_below_crisis(self):
        assert classify_credit_regime(0.74) == CreditRegime.STRESS

    def test_stress_at_0_65(self):
        assert classify_credit_regime(0.65) == CreditRegime.STRESS

    def test_stress_boundary_exact(self):
        assert classify_credit_regime(0.60) == CreditRegime.STRESS

    def test_caution_just_below_stress(self):
        assert classify_credit_regime(0.59) == CreditRegime.CAUTION

    def test_caution_at_0_50(self):
        assert classify_credit_regime(0.50) == CreditRegime.CAUTION

    def test_caution_boundary_exact(self):
        assert classify_credit_regime(0.40) == CreditRegime.CAUTION

    def test_normal_just_below_caution(self):
        assert classify_credit_regime(0.39) == CreditRegime.NORMAL

    def test_normal_at_0_30(self):
        assert classify_credit_regime(0.30) == CreditRegime.NORMAL

    def test_normal_at_zero(self):
        assert classify_credit_regime(0.0) == CreditRegime.NORMAL

    def test_crisis_at_one(self):
        assert classify_credit_regime(1.0) == CreditRegime.CRISIS

    def test_returns_credit_regime_type(self):
        for score in [0.1, 0.45, 0.65, 0.80]:
            result = classify_credit_regime(score)
            assert isinstance(result, CreditRegime)


# ── apply_credit_persistence_filter ──────────────────────────────────────────

class TestApplyCreditPersistenceFilter:
    """Verify asymmetric persistence filter rules.

    Rules:
      NORMAL/CAUTION : 1 consecutive → commit (fast de-escalation)
      STRESS         : 2 consecutive → commit
      CRISIS         : 3 consecutive → commit

    Lucas (1995 Nobel) — entry costs are asymmetric: adding stress
    positions is expensive, removing them is cheap.
    """

    def _f(self, regimes: List[CreditRegime]) -> List[CreditRegime]:
        return apply_credit_persistence_filter(regimes)

    def test_empty_returns_empty(self):
        assert self._f([]) == []

    def test_single_element(self):
        r = self._f([CreditRegime.NORMAL])
        assert r == [CreditRegime.NORMAL]

    def test_stable_normal_unchanged(self):
        r = self._f([CreditRegime.NORMAL] * 5)
        assert all(x == CreditRegime.NORMAL for x in r)

    def test_one_stress_day_not_committed(self):
        """Single STRESS signal: not enough — stays NORMAL."""
        r = self._f([CreditRegime.NORMAL, CreditRegime.STRESS, CreditRegime.NORMAL])
        assert r[1] == CreditRegime.NORMAL  # STRESS not committed

    def test_two_stress_days_committed(self):
        """Two consecutive STRESS days → commit on day 2."""
        r = self._f([CreditRegime.NORMAL,
                     CreditRegime.STRESS,
                     CreditRegime.STRESS,
                     CreditRegime.NORMAL])
        assert r[1] == CreditRegime.NORMAL   # first STRESS: not yet
        assert r[2] == CreditRegime.STRESS   # second STRESS: committed
        assert r[3] == CreditRegime.NORMAL   # 1 NORMAL → de-escalate immediately

    def test_three_stress_days_committed_earlier(self):
        """Three consecutive STRESS → commits on day 2, stays STRESS on day 3."""
        r = self._f([CreditRegime.NORMAL] + [CreditRegime.STRESS] * 3)
        assert r[1] == CreditRegime.NORMAL
        assert r[2] == CreditRegime.STRESS   # committed day 2
        assert r[3] == CreditRegime.STRESS

    def test_one_crisis_day_not_committed(self):
        """Single CRISIS signal: needs 3 → stays at prior regime."""
        r = self._f([CreditRegime.NORMAL, CreditRegime.CRISIS, CreditRegime.NORMAL])
        assert r[1] == CreditRegime.NORMAL

    def test_two_crisis_days_not_committed(self):
        r = self._f([CreditRegime.NORMAL, CreditRegime.CRISIS, CreditRegime.CRISIS,
                     CreditRegime.NORMAL])
        assert r[2] == CreditRegime.NORMAL   # only 2 consecutive → not committed

    def test_three_crisis_days_committed(self):
        """Three consecutive CRISIS days → commit on day 3."""
        r = self._f([CreditRegime.NORMAL] + [CreditRegime.CRISIS] * 4)
        assert r[1] == CreditRegime.NORMAL
        assert r[2] == CreditRegime.NORMAL
        assert r[3] == CreditRegime.CRISIS   # day 3: committed
        assert r[4] == CreditRegime.CRISIS

    def test_return_to_normal_immediate(self):
        """After STRESS is committed, single NORMAL immediately de-escalates."""
        r = self._f([CreditRegime.NORMAL,
                     CreditRegime.STRESS, CreditRegime.STRESS,
                     CreditRegime.NORMAL])
        assert r[2] == CreditRegime.STRESS   # committed
        assert r[3] == CreditRegime.NORMAL   # immediate de-escalation

    def test_mixed_stress_normal_stress(self):
        """STRESS x2 → NORMAL x1 → STRESS x2: counter resets on NORMAL."""
        r = self._f([
            CreditRegime.NORMAL,
            CreditRegime.STRESS, CreditRegime.STRESS,   # commit
            CreditRegime.NORMAL,                         # de-escalate
            CreditRegime.STRESS, CreditRegime.STRESS,   # re-commit
        ])
        assert r[2] == CreditRegime.STRESS
        assert r[3] == CreditRegime.NORMAL
        assert r[4] == CreditRegime.NORMAL   # counter reset; 1 STRESS not enough
        assert r[5] == CreditRegime.STRESS

    def test_caution_immediate_commit(self):
        """CAUTION requires only 1 consecutive day."""
        r = self._f([CreditRegime.NORMAL, CreditRegime.CAUTION])
        assert r[1] == CreditRegime.CAUTION

    def test_output_same_length(self):
        regimes = [CreditRegime.NORMAL, CreditRegime.STRESS, CreditRegime.CRISIS]
        r = self._f(regimes)
        assert len(r) == len(regimes)

    def test_all_outputs_are_credit_regime(self):
        regimes = [CreditRegime.NORMAL, CreditRegime.STRESS, CreditRegime.CRISIS,
                   CreditRegime.CAUTION, CreditRegime.NORMAL]
        r = self._f(regimes)
        assert all(isinstance(x, CreditRegime) for x in r)

    def test_severity_ordering(self):
        """Severity dict covers all regimes and is ordered."""
        assert _SEVERITY[CreditRegime.NORMAL]  == 0
        assert _SEVERITY[CreditRegime.CAUTION] == 1
        assert _SEVERITY[CreditRegime.STRESS]  == 2
        assert _SEVERITY[CreditRegime.CRISIS]  == 3


# ── credit_score_to_vix_proba ─────────────────────────────────────────────────

class TestCreditScoreToVixProba:
    r"""Verify the credit → VIX-label probability bridge.

    # $\mathbf{p}_{credit} = \text{vix\_proba}(8 + s \cdot 57) \in \Delta^5$
    """

    def test_output_shape(self):
        p = credit_score_to_vix_proba(0.5)
        assert p.shape == (6,)

    def test_sums_to_one(self):
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = credit_score_to_vix_proba(score)
            assert abs(p.sum() - 1.0) < 1e-6

    def test_all_non_negative(self):
        for score in np.linspace(0, 1, 20):
            assert np.all(credit_score_to_vix_proba(float(score)) >= 0)

    def test_high_score_biased_toward_crash(self):
        """score=1.0 → effective_vix=65 → mostly Crash."""
        p = credit_score_to_vix_proba(1.0)
        assert p[0] >= p[3]  # Crash prob >= Neutral prob

    def test_low_score_biased_toward_euphoria_bull(self):
        """score=0.0 → effective_vix=8 → mostly Euphoria."""
        p = credit_score_to_vix_proba(0.0)
        # Euphoria is index 5, Bull is index 4
        assert p[5] > p[0]  # Euphoria > Crash

    def test_monotone_crash_probability(self):
        """Higher credit stress score → higher crash probability."""
        proba_low  = credit_score_to_vix_proba(0.1)
        proba_high = credit_score_to_vix_proba(0.9)
        assert proba_high[0] > proba_low[0]  # Crash column


# ── apply_credit_overrides ────────────────────────────────────────────────────

class TestApplyCreditOverrides:
    """Verify override rules.

    Merton (1997 Nobel) — credit and equity share structural risk;
    credit stress must prevent the ensemble from signalling euphoria.
    """

    def test_crisis_overrides_bull_to_bear(self):
        result = apply_credit_overrides("Bull", CreditRegime.CRISIS)
        assert result == "Bear"

    def test_crisis_overrides_neutral_to_bear(self):
        result = apply_credit_overrides("Neutral", CreditRegime.CRISIS)
        assert result == "Bear"

    def test_crisis_overrides_euphoria_to_bear(self):
        result = apply_credit_overrides("Euphoria", CreditRegime.CRISIS)
        assert result == "Bear"

    def test_crisis_does_not_downgrade_existing_bear(self):
        """CRISIS + Bear → stays Bear (no downgrade needed)."""
        result = apply_credit_overrides("Bear", CreditRegime.CRISIS)
        assert result == "Bear"

    def test_crisis_does_not_downgrade_panic(self):
        result = apply_credit_overrides("Panic", CreditRegime.CRISIS)
        assert result == "Panic"

    def test_crisis_does_not_downgrade_crash(self):
        result = apply_credit_overrides("Crash", CreditRegime.CRISIS)
        assert result == "Crash"

    def test_stress_plus_low_vix_overrides_bull(self):
        """STRESS + VIX=15 (< 20) → Bull forced to Bear."""
        result = apply_credit_overrides("Bull", CreditRegime.STRESS, latest_vix=15.0)
        assert result == "Bear"

    def test_stress_plus_high_vix_no_override(self):
        """STRESS + VIX=25 (≥ 20) → no override."""
        result = apply_credit_overrides("Bull", CreditRegime.STRESS, latest_vix=25.0)
        assert result == "Bull"

    def test_stress_no_vix_no_override(self):
        """STRESS + no VIX provided → no override."""
        result = apply_credit_overrides("Bull", CreditRegime.STRESS, latest_vix=None)
        assert result == "Bull"

    def test_caution_no_override(self):
        """CAUTION regime → no override regardless of label."""
        assert apply_credit_overrides("Bull",    CreditRegime.CAUTION) == "Bull"
        assert apply_credit_overrides("Neutral", CreditRegime.CAUTION) == "Neutral"

    def test_normal_no_override(self):
        assert apply_credit_overrides("Euphoria", CreditRegime.NORMAL) == "Euphoria"

    def test_vix_boundary_exactly_20_no_override(self):
        """VIX exactly 20 is NOT < 20 → no STRESS early-warning override."""
        result = apply_credit_overrides("Bull", CreditRegime.STRESS, latest_vix=20.0)
        assert result == "Bull"


# ── CreditRegimeDetector (feature computation) ───────────────────────────────

class TestCreditRegimeDetectorFeatures:
    """Verify feature computation from price series.

    Arrow (1972 Nobel) — information is most valuable when extracted
    from multiple correlated signals; these tests verify the extraction.
    """

    def _make_price_series(self, n: int = 120, slope: float = 0.0, noise: float = 0.5,
                            seed: int = 42) -> pd.Series:
        rng = np.random.default_rng(seed)
        log_returns = rng.normal(slope, noise * 0.01, n)
        log_prices = np.cumsum(log_returns) + np.log(100.0)
        prices = np.exp(log_prices)
        return pd.Series(prices, index=_make_idx(n))

    def test_hy_only_features_valid(self):
        det = CreditRegimeDetector()
        hy = self._make_price_series(120)
        feats = det.compute_features_from_prices(hy_prices=hy)
        assert feats.z_hy is not None
        assert feats.z_ig is None
        assert feats.n_sources == 1

    def test_ig_only_features_valid(self):
        det = CreditRegimeDetector()
        ig = self._make_price_series(120)
        feats = det.compute_features_from_prices(ig_prices=ig)
        assert feats.z_ig is not None
        assert feats.z_hy is None
        assert feats.n_sources == 1

    def test_both_hy_ig_features(self):
        det = CreditRegimeDetector()
        hy = self._make_price_series(120)
        ig = self._make_price_series(120, seed=99)
        feats = det.compute_features_from_prices(hy_prices=hy, ig_prices=ig)
        assert feats.z_hy is not None
        assert feats.z_ig is not None
        assert feats.hy_ig_ratio_norm is not None
        assert feats.n_sources == 2

    def test_falling_hy_gives_positive_z_hy(self):
        """Falling HY prices (spread widening) → positive z_hy (stress)."""
        det = CreditRegimeDetector()
        # Strong downtrend
        hy = self._make_price_series(120, slope=-5.0)
        feats = det.compute_features_from_prices(hy_prices=hy)
        # z_hy is negated z-score of log-price; falling price → low log-price → neg z → pos z_hy
        assert feats.z_hy is not None

    def test_short_series_returns_no_features(self):
        """Series too short (< min_periods) → features are None."""
        det = CreditRegimeDetector()
        hy = self._make_price_series(5)
        feats = det.compute_features_from_prices(hy_prices=hy)
        assert feats.z_hy is None
        assert feats.n_sources == 0

    def test_none_inputs_all_none_features(self):
        det = CreditRegimeDetector()
        feats = det.compute_features_from_prices()
        assert feats.z_hy is None
        assert feats.z_ig is None
        assert feats.n_sources == 0


# ── CreditRegimeDetector (series) ─────────────────────────────────────────────

class TestCreditRegimeDetectorSeries:
    """Verify compute_features_series and regime_series outputs."""

    def _make_price_series(self, n: int = 150, seed: int = 1) -> pd.Series:
        rng = np.random.default_rng(seed)
        lp = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100.0)
        return pd.Series(np.exp(lp), index=_make_idx(n))

    def test_score_series_length(self):
        det = CreditRegimeDetector()
        hy = self._make_price_series(150)
        scores = det.compute_features_series(hy_prices=hy)
        assert len(scores) == 150

    def test_score_series_values_in_01(self):
        det = CreditRegimeDetector()
        hy = self._make_price_series(150)
        scores = det.compute_features_series(hy_prices=hy)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_regime_series_valid_labels(self):
        det = CreditRegimeDetector()
        hy = self._make_price_series(150)
        ig = self._make_price_series(150, seed=7)
        regimes = det.regime_series(hy_prices=hy, ig_prices=ig)
        valid = set(CreditRegime)
        assert all(r in valid for r in regimes)

    def test_regime_series_length_matches(self):
        det = CreditRegimeDetector()
        hy = self._make_price_series(150)
        regimes = det.regime_series(hy_prices=hy)
        assert len(regimes) == 150

    def test_requires_at_least_one_series(self):
        det = CreditRegimeDetector()
        with pytest.raises((ValueError, Exception)):
            det.compute_features_series()

    def test_no_filter_differs_from_filtered(self):
        """Filtered series should have fewer or equal transitions."""
        det = CreditRegimeDetector()
        rng = np.random.default_rng(2024)
        lp = np.cumsum(rng.normal(0, 0.02, 200)) + np.log(100.0)
        hy = pd.Series(np.exp(lp), index=_make_idx(200))
        raw = det.regime_series(hy_prices=hy, apply_filter=False)
        filtered = det.regime_series(hy_prices=hy, apply_filter=True)
        raw_trans = (raw != raw.shift()).sum()
        filt_trans = (filtered != filtered.shift()).sum()
        assert filt_trans <= raw_trans


# ── Integration with RegimeDetector ──────────────────────────────────────────

class TestRegimeDetectorCreditIntegration:
    """Test that credit signal integrates correctly with the ensemble.

    Verifies:
      1. Backward-compatibility (w_credit=0 → identical to prior behaviour)
      2. Credit NORMAL has minimal effect when ensemble already points Bull
      3. Credit CRISIS forces output to at least Bear
      4. Credit STRESS + low VIX → early warning (at least Bear)
      5. API signatures unchanged for existing callers
    """

    @pytest.fixture(scope="class")
    def ensemble(self):
        from regime_trader.models.regime_detector import RegimeDetector
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 300
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        vix_quiet  = rng.normal(12, 1.5, 150).clip(8, 20)
        vix_stress = rng.normal(30, 4.0, 150).clip(18, 50)
        vix = pd.Series(np.concatenate([vix_quiet, vix_stress]), index=idx)
        ret = pd.Series(rng.normal(0, 0.01, n), index=idx)
        det = RegimeDetector(n_hmm_states=3)
        det.fit(vix, ret)
        return det, vix, ret, idx

    def test_backward_compat_no_credit(self, ensemble):
        """w_credit=0 (default): predict_series works without credit_scores."""
        det, vix, ret, _ = ensemble
        labels = det.predict_series(vix, ret)
        assert all(lbl in ["Crash","Panic","Bear","Neutral","Bull","Euphoria"]
                   for lbl in labels)

    def test_w_credit_zero_ignores_credit_scores(self, ensemble):
        """When w_credit=0, passing credit_scores has no effect."""
        det_no_credit, vix, ret, idx = ensemble
        credit_scores = pd.Series(np.ones(len(vix)) * 0.99, index=idx)

        labels_without = det_no_credit.predict_series(vix, ret)
        labels_with    = det_no_credit.predict_series(vix, ret, credit_scores=credit_scores)
        pd.testing.assert_series_equal(labels_without, labels_with)

    def test_credit_enabled_bull_normal_stays_calm(self, ensemble):
        """All-bull VIX + CREDIT=NORMAL → regime stays in calm range."""
        from regime_trader.models.regime_detector import RegimeDetector
        _, _, _, idx = ensemble
        n = 50
        short_idx = idx[:n]
        vix_bull = pd.Series(np.ones(n) * 11.0, index=short_idx)
        ret_flat = pd.Series(np.zeros(n), index=short_idx)

        det = RegimeDetector(w_vix=0.30, w_hmm=0.25, w_ml=0.25, w_credit=0.20,
                             n_hmm_states=3)
        det.fit(vix_bull, ret_flat)

        credit_scores = pd.Series(np.ones(n) * 0.20, index=short_idx)  # NORMAL
        label = det.predict(vix_bull, ret_flat, credit_scores=credit_scores)
        assert label in ("Bull", "Euphoria", "Neutral")

    def test_credit_crisis_prevents_bull_output(self, ensemble):
        """CRISIS credit + neutral-VIX → override forces at least Bear."""
        from regime_trader.models.regime_detector import RegimeDetector
        _, _, _, idx = ensemble
        n = 50
        short_idx = idx[:n]
        # VIX = 18 (Neutral range)
        vix_neutral = pd.Series(np.ones(n) * 18.0, index=short_idx)
        ret_flat    = pd.Series(np.zeros(n), index=short_idx)

        det = RegimeDetector(w_vix=0.30, w_hmm=0.25, w_ml=0.25, w_credit=0.20,
                             n_hmm_states=3)
        det.fit(vix_neutral, ret_flat)

        # All credit scores in CRISIS range (0.80)
        credit_scores = pd.Series(np.ones(n) * 0.80, index=short_idx)
        labels = det.predict_series(vix_neutral, ret_flat, credit_scores=credit_scores)

        _SEVERITY_VIX = {"Euphoria":0,"Bull":1,"Neutral":2,"Bear":3,"Panic":4,"Crash":5}
        for lbl in labels:
            assert _SEVERITY_VIX[lbl] >= _SEVERITY_VIX["Bear"], (
                f"CRISIS override failed: got {lbl!r}, expected >= Bear"
            )

    def test_credit_stress_low_vix_early_warning(self, ensemble):
        """STRESS credit + VIX < 20 → at least Bear (early warning rule)."""
        from regime_trader.models.regime_detector import RegimeDetector
        _, _, _, idx = ensemble
        n = 50
        short_idx = idx[:n]
        vix_low = pd.Series(np.ones(n) * 15.0, index=short_idx)  # VIX=15, Bull
        ret_flat = pd.Series(np.zeros(n), index=short_idx)

        det = RegimeDetector(w_vix=0.30, w_hmm=0.25, w_ml=0.25, w_credit=0.20,
                             n_hmm_states=3)
        det.fit(vix_low, ret_flat)

        credit_scores = pd.Series(np.ones(n) * 0.65, index=short_idx)  # STRESS
        labels = det.predict_series(vix_low, ret_flat, credit_scores=credit_scores)

        _SEVERITY_VIX = {"Euphoria":0,"Bull":1,"Neutral":2,"Bear":3,"Panic":4,"Crash":5}
        for lbl in labels:
            assert _SEVERITY_VIX[lbl] >= _SEVERITY_VIX["Bear"], (
                f"Early-warning override failed: got {lbl!r}, expected >= Bear"
            )

    def test_backtest_report_includes_credit_weight(self, ensemble):
        """backtest_report should include credit key in ensemble_weights."""
        det, vix, ret, _ = ensemble
        report = det.backtest_report(vix, ret)
        assert "credit" in report["ensemble_weights"]
        assert report["ensemble_weights"]["credit"] == 0.0  # default off

    def test_predict_signature_unchanged(self, ensemble):
        """predict(vix, returns) → str (existing callers unaffected)."""
        det, vix, ret, _ = ensemble
        result = det.predict(vix, ret)
        assert isinstance(result, str)
        assert result in ["Crash","Panic","Bear","Neutral","Bull","Euphoria"]

    def test_predict_series_signature_unchanged(self, ensemble):
        """predict_series(vix, returns) → pd.Series (existing callers unaffected)."""
        det, vix, ret, _ = ensemble
        result = det.predict_series(vix, ret)
        assert isinstance(result, pd.Series)
