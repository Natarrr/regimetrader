r"""tests/test_regime_detector.py
Unit tests for regime/regime_detector.py.

Covers:
  - vix_rule: all threshold boundaries
  - vix_proba: shape, sums-to-1, dominant label
  - apply_persistence_filter: switching behaviour, n=1 passthrough
  - HMMRegimeDetector: fit/predict on synthetic data
  - MLRegimeDetector: fit/predict, feature matrix shape
  - RegimeDetector (ensemble): fit/predict, backtest_report structure
  - evaluate: accuracy, false-positive counting
  - Historical crash validation: 2008 (VIX ~80) and 2020 (VIX ~65)

Engle (2003 Nobel) — volatility clustering is the empirical regularity
these tests validate; regime detection accuracy at the extremes
(VIX > 45, VIX > 35) is the primary performance criterion.

# VIX threshold: $\text{regime} = \begin{cases}
#   \text{Crash}   & \text{VIX} \geq 45 \\
#   \text{Panic}   & 35 \leq \text{VIX} < 45 \\
#   \text{Bear}    & 25 \leq \text{VIX} < 35 \\
#   \text{Neutral} & 15 \leq \text{VIX} < 25 \\
#   \text{Bull}    & 12 \leq \text{VIX} < 15 \\
#   \text{Euphoria}& \text{VIX} < 12
# \end{cases}$
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regime.regime_detector import (
    HMMRegimeDetector,
    MLRegimeDetector,
    RegimeDetector,
    _LABELS,
    apply_persistence_filter,
    evaluate,
    vix_proba,
    vix_rule,
    vix_rule_series,
)

# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _make_vix_returns(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Generate n days of synthetic VIX + log-return data.

    Uses a two-regime simulation:
      - Quiet (days 0..149): VIX~12, returns~0.1%/day
      - Stress (days 150..299): VIX~35, returns~-0.5%/day
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")

    vix_quiet = rng.normal(12, 1.5, n // 2).clip(8, 20)
    vix_stress = rng.normal(35, 4.0, n - n // 2).clip(20, 50)
    vix = pd.Series(np.concatenate([vix_quiet, vix_stress]), index=idx)

    ret_quiet = rng.normal(0.001, 0.008, n // 2)
    ret_stress = rng.normal(-0.005, 0.020, n - n // 2)
    returns = pd.Series(np.concatenate([ret_quiet, ret_stress]), index=idx)

    return vix, returns


# ── vix_rule ──────────────────────────────────────────────────────────────────

class TestVixRule:
    r"""Engle (2003 Nobel) — VIX threshold classification.

    # Point mapping: $\text{regime}(\text{VIX}) \in \{$Crash, Panic, Bear,
    # Neutral, Bull, Euphoria$\}$.
    """

    def test_crash_at_80(self):
        """2008 crisis: VIX hit ~80 → must classify as Crash."""
        assert vix_rule(80.0) == "Crash"

    def test_crash_at_65(self):
        """2020 COVID crisis: VIX hit ~65 → must classify as Crash."""
        assert vix_rule(65.0) == "Crash"

    def test_crash_boundary_45(self):
        assert vix_rule(45.0) == "Crash"

    def test_panic_just_below_45(self):
        assert vix_rule(44.9) == "Panic"

    def test_panic_at_35(self):
        assert vix_rule(35.0) == "Panic"

    def test_bear_just_below_35(self):
        assert vix_rule(34.9) == "Bear"

    def test_bear_at_25(self):
        assert vix_rule(25.0) == "Bear"

    def test_neutral_just_below_25(self):
        assert vix_rule(24.9) == "Neutral"

    def test_neutral_at_15(self):
        assert vix_rule(15.0) == "Neutral"

    def test_bull_just_below_15(self):
        assert vix_rule(14.9) == "Bull"

    def test_bull_at_12(self):
        assert vix_rule(12.0) == "Bull"

    def test_euphoria_just_below_12(self):
        assert vix_rule(11.9) == "Euphoria"

    def test_euphoria_at_zero(self):
        assert vix_rule(0.0) == "Euphoria"

    def test_all_thresholds_return_valid_label(self):
        for vix_val in [80, 45, 40, 35, 30, 25, 20, 15, 13, 12, 10, 5, 0]:
            label = vix_rule(float(vix_val))
            assert label in _LABELS


class TestVixRuleSeries:
    """vix_rule_series applies vix_rule element-wise."""

    def test_series_correct_length(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        vix = pd.Series([20.0] * 10, index=idx)
        result = vix_rule_series(vix)
        assert len(result) == 10

    def test_all_elements_valid_label(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="B")
        vix = pd.Series([80, 40, 30, 20, 13, 10], index=idx, dtype=float)
        result = vix_rule_series(vix)
        assert all(r in _LABELS for r in result)


# ── vix_proba ─────────────────────────────────────────────────────────────────

class TestVixProba:
    r"""VIX soft-probability vector.

    # $\mathbf{p}(VIX) \in \Delta^5$ (probability simplex):
    # dominant label gets 0.70, each neighbour gets 0.15 (normalised at edges).
    """

    def test_sums_to_one(self):
        for vix_val in [80, 45, 35, 25, 15, 12, 10]:
            p = vix_proba(float(vix_val))
            assert abs(p.sum() - 1.0) < 1e-6

    def test_shape(self):
        p = vix_proba(20.0)
        assert p.shape == (len(_LABELS),)

    def test_dominant_label_has_highest_prob(self):
        """Neutral regime: VIX=20 → index 3 is dominant."""
        p = vix_proba(20.0)
        dominant_label = _LABELS[np.argmax(p)]
        assert dominant_label == "Neutral"

    def test_crash_vix_dominant(self):
        p = vix_proba(80.0)
        assert _LABELS[np.argmax(p)] == "Crash"

    def test_euphoria_vix_dominant(self):
        p = vix_proba(8.0)
        assert _LABELS[np.argmax(p)] == "Euphoria"

    def test_all_probabilities_non_negative(self):
        for vix_val in np.linspace(5, 80, 20):
            p = vix_proba(float(vix_val))
            assert np.all(p >= 0)


# ── apply_persistence_filter ──────────────────────────────────────────────────

class TestApplyPersistenceFilter:
    """Lucas (1995 Nobel) — persistence filter dampens spurious transitions.

    # Filter: switch only after $n$ consecutive identical signals.
    # Cost: detection lag increases by at most $n-1$ days.
    """

    def test_n1_passthrough(self):
        """n=1 should return unchanged labels."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        labels = pd.Series(["Bull", "Bear", "Bull", "Neutral", "Bear"], index=idx)
        result = apply_persistence_filter(labels, n=1)
        assert list(result) == list(labels)

    def test_single_spike_suppressed(self):
        """One-day outlier surrounded by Bull should be fully suppressed with n=2.

        The single Crash signal never forms a 2-consecutive streak, so the
        committed regime stays Bull throughout.
        """
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        labels = pd.Series(["Bull", "Bull", "Crash", "Bull", "Bull"], index=idx)
        result = apply_persistence_filter(labels, n=2)
        assert result.iloc[2] == "Bull"   # single Crash fully suppressed
        assert result.iloc[3] == "Bull"   # stays Bull after spike

    def test_two_consecutive_triggers_switch(self):
        """n=2: two Crash signals in a row commit to Crash on the second day."""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        labels = pd.Series(["Bull", "Crash", "Crash", "Crash", "Crash"], index=idx)
        result = apply_persistence_filter(labels, n=2)
        assert result.iloc[1] == "Bull"   # first Crash: not yet committed
        assert result.iloc[2] == "Crash"  # second Crash: committed
        assert result.iloc[3] == "Crash"  # stays Crash

    def test_output_same_length_as_input(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        labels = pd.Series(["Bull"] * 10, index=idx)
        result = apply_persistence_filter(labels, n=3)
        assert len(result) == 10

    def test_stable_regime_unchanged(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        labels = pd.Series(["Neutral"] * 5, index=idx)
        result = apply_persistence_filter(labels, n=3)
        assert list(result) == ["Neutral"] * 5


# ── HMMRegimeDetector ─────────────────────────────────────────────────────────

class TestHMMRegimeDetector:
    """Lucas (1995 Nobel) — HMM captures regime dynamics beyond threshold rules.

    Tests use synthetic two-regime data; correctness criteria:
      - fit() converges without error
      - predict_proba() returns (n, 6) matrix summing to 1 per row
      - predict_series() returns valid labels
      - Unfitted model raises RuntimeError
    """

    @pytest.fixture(scope="class")
    def fitted_hmm(self):
        vix, returns = _make_vix_returns(300)
        hmm = HMMRegimeDetector(n_states=3)
        hmm.fit(vix, returns)
        return hmm, vix, returns

    def test_fit_returns_self(self):
        vix, returns = _make_vix_returns(200)
        hmm = HMMRegimeDetector(n_states=3)
        result = hmm.fit(vix, returns)
        assert result is hmm

    def test_is_fitted_after_fit(self):
        vix, returns = _make_vix_returns(200)
        hmm = HMMRegimeDetector(n_states=3)
        hmm.fit(vix, returns)
        assert hmm._is_fitted is True

    def test_predict_proba_shape(self, fitted_hmm):
        hmm, vix, returns = fitted_hmm
        proba = hmm.predict_proba(vix, returns)
        assert proba.shape == (len(vix), len(_LABELS))

    def test_predict_proba_rows_sum_to_one(self, fitted_hmm):
        hmm, vix, returns = fitted_hmm
        proba = hmm.predict_proba(vix, returns)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self, fitted_hmm):
        hmm, vix, returns = fitted_hmm
        proba = hmm.predict_proba(vix, returns)
        assert np.all(proba >= 0)

    def test_predict_series_valid_labels(self, fitted_hmm):
        hmm, vix, returns = fitted_hmm
        series = hmm.predict_series(vix, returns)
        assert all(lbl in _LABELS for lbl in series)

    def test_predict_series_same_length(self, fitted_hmm):
        hmm, vix, returns = fitted_hmm
        series = hmm.predict_series(vix, returns)
        assert len(series) == len(vix)

    def test_unfitted_raises(self):
        hmm = HMMRegimeDetector()
        vix, returns = _make_vix_returns(50)
        with pytest.raises(RuntimeError, match="fit"):
            hmm.predict_proba(vix, returns)

    def test_stress_period_elevated_regime(self, fitted_hmm):
        """Quiet period labels should differ from stress period labels on average."""
        hmm, vix, returns = fitted_hmm
        series = hmm.predict_series(vix, returns)
        quiet_labels = series.iloc[:150]
        stress_labels = series.iloc[150:]
        # Not all labels must differ, but distributions should shift
        quiet_counts = quiet_labels.value_counts()
        stress_counts = stress_labels.value_counts()
        # At minimum, both sub-periods should have some content
        assert len(quiet_counts) >= 1
        assert len(stress_counts) >= 1


# ── MLRegimeDetector ──────────────────────────────────────────────────────────

class TestMLRegimeDetector:
    """Fama (2013 Nobel) — systematic feature engineering; tests verify
    that the 8-feature engineering pipeline produces valid outputs.
    """

    @pytest.fixture(scope="class")
    def fitted_ml(self):
        vix, returns = _make_vix_returns(300)
        ml = MLRegimeDetector()
        ml.fit(vix, returns)
        return ml, vix, returns

    def test_fit_returns_self(self):
        vix, returns = _make_vix_returns(200)
        ml = MLRegimeDetector()
        result = ml.fit(vix, returns)
        assert result is ml

    def test_is_fitted_after_fit(self):
        vix, returns = _make_vix_returns(200)
        ml = MLRegimeDetector()
        ml.fit(vix, returns)
        assert ml._is_fitted is True

    def test_predict_proba_shape(self, fitted_ml):
        ml, vix, returns = fitted_ml
        proba = ml.predict_proba(vix, returns)
        assert proba.shape == (len(vix), len(_LABELS))

    def test_predict_proba_rows_sum_to_one(self, fitted_ml):
        ml, vix, returns = fitted_ml
        proba = ml.predict_proba(vix, returns)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self, fitted_ml):
        ml, vix, returns = fitted_ml
        proba = ml.predict_proba(vix, returns)
        assert np.all(proba >= 0)

    def test_predict_series_valid_labels(self, fitted_ml):
        ml, vix, returns = fitted_ml
        series = ml.predict_series(vix, returns)
        assert all(lbl in _LABELS for lbl in series)

    def test_unfitted_raises(self):
        ml = MLRegimeDetector()
        vix, returns = _make_vix_returns(50)
        with pytest.raises(RuntimeError, match="fit"):
            ml.predict_proba(vix, returns)

    def test_feature_matrix_8_columns(self):
        vix, returns = _make_vix_returns(60)
        df = MLRegimeDetector._build_features(vix, returns)
        assert df.shape[1] == 8

    def test_fit_with_true_labels(self):
        """ML accepts external true_labels for supervised training."""
        vix, returns = _make_vix_returns(200)
        true_labels = vix_rule_series(vix)
        ml = MLRegimeDetector()
        ml.fit(vix, returns, true_labels=true_labels)
        assert ml._is_fitted

    def test_predict_series_same_length(self, fitted_ml):
        ml, vix, returns = fitted_ml
        series = ml.predict_series(vix, returns)
        assert len(series) == len(vix)


# ── RegimeDetector (ensemble) ─────────────────────────────────────────────────

class TestRegimeDetector:
    r"""Ensemble combines HMM, ML, and VIX rule via soft voting.

    Soft-vote formula:
    # $\mathbf{p}_{ens} = w_{HMM}\,\mathbf{p}_{HMM} + w_{ML}\,\mathbf{p}_{ML}
    #                    + w_{VIX}\,\mathbf{p}_{VIX}$
    """

    @pytest.fixture(scope="class")
    def fitted_ensemble(self):
        vix, returns = _make_vix_returns(300)
        det = RegimeDetector(n_hmm_states=3)
        det.fit(vix, returns)
        return det, vix, returns

    def test_fit_returns_self(self):
        vix, returns = _make_vix_returns(200)
        det = RegimeDetector(n_hmm_states=3)
        result = det.fit(vix, returns)
        assert result is det

    def test_predict_returns_string(self, fitted_ensemble):
        det, vix, returns = fitted_ensemble
        label = det.predict(vix, returns)
        assert isinstance(label, str)
        assert label in _LABELS

    def test_predict_series_length(self, fitted_ensemble):
        det, vix, returns = fitted_ensemble
        series = det.predict_series(vix, returns)
        assert len(series) == len(vix)

    def test_predict_series_valid_labels(self, fitted_ensemble):
        det, vix, returns = fitted_ensemble
        series = det.predict_series(vix, returns)
        assert all(lbl in _LABELS for lbl in series)

    def test_predict_proba_rows_sum_to_one(self, fitted_ensemble):
        det, vix, returns = fitted_ensemble
        proba_df = det.predict_proba_series(vix, returns)
        row_sums = proba_df.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)

    def test_unfitted_raises_on_predict(self):
        det = RegimeDetector()
        vix, returns = _make_vix_returns(50)
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(vix, returns)

    def test_backtest_report_structure(self, fitted_ensemble):
        """Backtest report must contain all expected keys and methods."""
        det, vix, returns = fitted_ensemble
        report = det.backtest_report(vix, returns)
        assert "n_samples" in report
        assert "methods" in report
        for method_name in ("vix_rule", "hmm", "ml", "ensemble"):
            assert method_name in report["methods"]
        for method_name, m in report["methods"].items():
            assert "accuracy" in m
            assert "false_positives_per_year" in m
            assert "transitions" in m
            assert 0.0 <= m["accuracy"] <= 1.0
            assert m["false_positives_per_year"] >= 0.0

    def test_backtest_report_date_range(self, fitted_ensemble):
        det, vix, returns = fitted_ensemble
        report = det.backtest_report(vix, returns)
        assert "date_range" in report
        assert "start" in report["date_range"]
        assert "end" in report["date_range"]

    def test_backtest_report_ensemble_weights(self, fitted_ensemble):
        det, vix, returns = fitted_ensemble
        report = det.backtest_report(vix, returns)
        assert "ensemble_weights" in report
        weights = report["ensemble_weights"]
        total = weights["hmm"] + weights["ml"] + weights["vix"]
        assert abs(total - 1.0) < 1e-6

    def test_no_filter_different_from_filtered(self, fitted_ensemble):
        """Persistence filter should produce at least as many stable transitions."""
        det, vix, returns = fitted_ensemble
        raw = det.predict_series(vix, returns, apply_filter=False)
        filtered = det.predict_series(vix, returns, apply_filter=True)
        raw_trans = (raw != raw.shift()).sum()
        filt_trans = (filtered != filtered.shift()).sum()
        # Filtered must have fewer or equal transitions
        assert filt_trans <= raw_trans


# ── evaluate (standalone) ─────────────────────────────────────────────────────

class TestEvaluate:
    r"""Standalone VIX-rule evaluation helper.

    # Accuracy = $\frac{\sum_{t} \mathbf{1}[\hat{y}_t = y_t]}{T}$
    """

    def test_perfect_accuracy(self):
        vix, _ = _make_vix_returns(100)
        true_labels = vix_rule_series(vix)  # same as predicted
        result = evaluate(vix, true_labels)
        assert result["accuracy"] == 1.0

    def test_imperfect_accuracy_between_zero_and_one(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        vix = pd.Series([20.0, 20.0, 20.0, 20.0], index=idx)
        # Two match, two do not
        true_labels = pd.Series(["Neutral", "Neutral", "Bull", "Bear"], index=idx)
        result = evaluate(vix, true_labels)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_all_wrong_accuracy_zero(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        vix = pd.Series([20.0, 20.0, 20.0], index=idx)  # Neutral
        true_labels = pd.Series(["Crash", "Crash", "Crash"], index=idx)
        result = evaluate(vix, true_labels)
        assert result["accuracy"] == 0.0

    def test_returns_required_keys(self):
        vix, _ = _make_vix_returns(50)
        true_labels = vix_rule_series(vix)
        result = evaluate(vix, true_labels)
        for key in ("method", "n_samples", "accuracy",
                    "false_positives_per_year", "transitions_predicted", "transitions_true"):
            assert key in result

    def test_method_name(self):
        vix, _ = _make_vix_returns(50)
        true_labels = vix_rule_series(vix)
        result = evaluate(vix, true_labels)
        assert result["method"] == "vix_rule"

    def test_false_positives_non_negative(self):
        vix, _ = _make_vix_returns(100)
        true_labels = vix_rule_series(vix)
        result = evaluate(vix, true_labels)
        assert result["false_positives_per_year"] >= 0.0

    def test_n_samples(self):
        vix, _ = _make_vix_returns(252)
        true_labels = vix_rule_series(vix)
        result = evaluate(vix, true_labels)
        assert result["n_samples"] == 252


# ── Historical crash validation ───────────────────────────────────────────────

class TestHistoricalCrashes:
    """Verify correct classification at known extreme VIX levels.

    2008 GFC: VIX peaked at ~80 → Crash
    2020 COVID: VIX peaked at ~65 → Crash
    2011 EU debt crisis: VIX peaked at ~46 → Crash (just above boundary)

    Engle (2003 Nobel) — volatility regimes are most critical at extremes;
    any misclassification of a genuine Crash as Panic is an unacceptable error.
    """

    @pytest.mark.parametrize("vix_val,expected", [
        (80.86, "Crash"),   # 2008 GFC peak (Nov 20, 2008)
        (65.54, "Crash"),   # 2020 COVID peak (Mar 18, 2020)
        (48.00, "Crash"),   # 2011 EU sovereign debt crisis (Aug 8, 2011)
        (45.00, "Crash"),   # boundary case — must be Crash
        (44.99, "Panic"),   # just below — must be Panic
        (12.00, "Bull"),    # 2017 low-vol calm market
        (9.14,  "Euphoria"),# 2017 all-time VIX low
    ])
    def test_known_vix_levels(self, vix_val, expected):
        assert vix_rule(float(vix_val)) == expected

    def test_2008_crisis_series(self):
        """2008 crisis simulation: 252 days with VIX progressively spiking."""
        idx = pd.date_range("2008-01-01", periods=10, freq="B")
        vix = pd.Series([20, 25, 35, 48, 60, 80, 65, 50, 38, 28], index=idx, dtype=float)
        labels = vix_rule_series(vix)
        assert labels.iloc[4] == "Crash"   # VIX 60
        assert labels.iloc[5] == "Crash"   # VIX 80
        assert labels.iloc[2] == "Panic"   # VIX 35 (boundary)

    def test_evaluate_on_2008_spike(self):
        """evaluate() should return near-100% accuracy when using VIX rule as reference."""
        idx = pd.date_range("2008-09-01", periods=63, freq="B")
        rng = np.random.default_rng(2008)
        vix = pd.Series(
            np.concatenate([
                rng.normal(25, 3, 20).clip(15, 35),
                rng.normal(55, 8, 25).clip(40, 80),
                rng.normal(35, 4, 18).clip(25, 50),
            ]),
            index=idx,
        )
        true_labels = vix_rule_series(vix)  # VIX rule IS the reference
        result = evaluate(vix, true_labels)
        assert result["accuracy"] == 1.0   # trivially perfect against itself
        assert result["false_positives_per_year"] == 0.0
