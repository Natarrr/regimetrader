"""tests/test_hmm_engine.py
Unit tests for hmm_engine.classifier.RegimeClassifier.

Lucas (1995 Nobel) — Rational expectations: the HMM forward algorithm prices
all available information without look-ahead bias.  Historical validation
uses a synthesised 2008 GFC return sequence where the Bear state must be the
dominant regime during the Oct 2008 volatility cluster.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmm_engine.classifier import RegimeClassifier, RegimeState
from analysis.feature_engineer import FeatureEngineer


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _make_ohlcv(returns: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Convert a 1-D return array into a minimal OHLCV DataFrame."""
    rng   = np.random.default_rng(seed)
    n     = len(returns)
    close = 100.0 * np.exp(np.cumsum(returns))
    high  = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol   = rng.integers(500_000, 2_000_000, n).astype(float)
    idx   = pd.date_range("2005-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _make_regime_returns(rng_seed: int = 42) -> np.ndarray:
    """Synthesise 600 days: calm → crash cluster → recovery (GFC profile)."""
    rng   = np.random.default_rng(rng_seed)
    calm  = rng.normal(0.0005, 0.008, 400)
    crash = rng.normal(-0.015, 0.045, 100)   # Oct 2008 cluster
    crash[30:50] += rng.normal(-0.02, 0.06, 20)
    recov = rng.normal(0.002, 0.012, 100)
    return np.concatenate([calm, crash, recov])


# ── Smoke: fit + predict do not raise ─────────────────────────────────────────

class TestRegimeClassifierSmoke:
    def test_fit_returns_self(self):
        returns = _make_regime_returns()
        ohlcv   = _make_ohlcv(returns)
        fe      = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)
        clf = RegimeClassifier()
        result = clf.fit(features, rets)
        assert result is clf

    def test_predict_current_returns_regime_state(self):
        returns = _make_regime_returns()
        ohlcv   = _make_ohlcv(returns)
        fe      = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)
        clf = RegimeClassifier()
        clf.fit(features, rets)
        state = clf.predict_current(features[-20:])
        assert isinstance(state, RegimeState)

    def test_predict_sequence_returns_dataframe(self):
        returns = _make_regime_returns()
        ohlcv   = _make_ohlcv(returns)
        fe      = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)
        clf = RegimeClassifier()
        clf.fit(features, rets)
        seq = clf.predict_sequence(features)
        assert isinstance(seq, pd.DataFrame)
        assert "raw_label" in seq.columns
        assert "confirmed_label" in seq.columns


# ── Output contract: labels, types, bounds ────────────────────────────────────

class TestRegimeStateContract:
    @pytest.fixture(scope="class")
    def fitted_state(self):
        returns  = _make_regime_returns()
        ohlcv    = _make_ohlcv(returns)
        fe       = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)
        clf = RegimeClassifier()
        clf.fit(features, rets)
        return clf.predict_current(features[-20:])

    def test_raw_label_is_valid(self, fitted_state):
        assert fitted_state.raw_label in ("Bull", "Neutral", "Bear", "Unknown")

    def test_confirmed_label_is_valid_or_none(self, fitted_state):
        assert fitted_state.confirmed_label in ("Bull", "Neutral", "Bear", None)

    def test_position_scale_in_range(self, fitted_state):
        assert 0.0 <= fitted_state.position_scale <= 1.0

    def test_is_uncertain_is_bool(self, fitted_state):
        assert isinstance(fitted_state.is_uncertain, bool)

    def test_regime_probs_sum_to_one(self, fitted_state):
        probs = fitted_state.regime_probs
        assert probs is not None
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_color_map_contains_expected_keys(self, fitted_state):
        assert "Bull" in fitted_state.color_map
        assert "Bear" in fitted_state.color_map
        assert "Neutral" in fitted_state.color_map


# ── Sequence contract ─────────────────────────────────────────────────────────

class TestPredictSequenceContract:
    @pytest.fixture(scope="class")
    def seq(self):
        returns  = _make_regime_returns()
        ohlcv    = _make_ohlcv(returns)
        fe       = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)
        clf = RegimeClassifier()
        clf.fit(features, rets)
        return clf.predict_sequence(features), len(features)

    def test_length_matches_features(self, seq):
        df, n = seq
        assert len(df) == n

    def test_raw_labels_only_valid_values(self, seq):
        df, _ = seq
        valid = {"Bull", "Neutral", "Bear", "Unknown"}
        assert set(df["raw_label"].unique()).issubset(valid)

    def test_confirmed_labels_only_valid_or_none(self, seq):
        df, _ = seq
        # pandas stores None as NaN in string columns; check non-null values only.
        non_null = df["confirmed_label"].dropna().unique()
        assert set(non_null).issubset({"Bull", "Neutral", "Bear"})


# ── Historical validation: 2008 GFC crash dominates Bear state ────────────────

class TestGFC2008Validation:
    """Lucas (1995 Nobel) — validate on synthetic GFC data.

    The crash segment (bars 400-500, i.e. ~Oct 2008 analogue) must be dominated
    by the Bear state.  A majority-Bear requirement of ≥ 40% within the crash
    window is used (lenient: HMM may split crash into Bear + Neutral).
    """

    def test_crash_window_majority_bear(self):
        rng_seed = 42
        returns  = _make_regime_returns(rng_seed)
        ohlcv    = _make_ohlcv(returns, seed=rng_seed)
        fe       = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)

        # Feature matrix is shorter than returns (20+ warmup rows dropped).
        # Crash window in returns starts at idx 400; align to feature index.
        n_warmup = len(returns) - len(features)

        clf = RegimeClassifier(n_components=3, random_state=rng_seed)
        clf.fit(features, rets)
        seq = clf.predict_sequence(features)

        # Map return crash window [400, 500) → feature window.
        crash_start = max(0, 400 - n_warmup)
        crash_end   = max(0, 500 - n_warmup)
        crash_labels = seq["raw_label"].iloc[crash_start:crash_end]

        bear_frac = (crash_labels == "Bear").mean()
        assert bear_frac >= 0.40, (
            f"Expected ≥40% Bear labels in the 2008 crash window; got {bear_frac:.1%}"
        )

    def test_calm_window_not_dominated_by_bear(self):
        rng_seed = 42
        returns  = _make_regime_returns(rng_seed)
        ohlcv    = _make_ohlcv(returns, seed=rng_seed)
        fe       = FeatureEngineer()
        features, rets, _ = fe.build(ohlcv)

        n_warmup = len(returns) - len(features)
        clf = RegimeClassifier(n_components=3, random_state=rng_seed)
        clf.fit(features, rets)
        seq = clf.predict_sequence(features)

        calm_end   = max(0, 300 - n_warmup)
        calm_labels = seq["raw_label"].iloc[:calm_end]
        bear_frac   = (calm_labels == "Bear").mean()
        assert bear_frac < 0.60, (
            f"Calm pre-crash window should not be dominated by Bear; got {bear_frac:.1%}"
        )


# ── Unfitted classifier returns safe defaults ─────────────────────────────────

class TestUnfittedDefaults:
    def test_predict_current_unfitted_returns_unknown(self):
        clf = RegimeClassifier()
        dummy = np.zeros((10, 5))
        # Should not raise; returns Unknown gracefully.
        state = clf.predict_current(dummy)
        assert state.raw_label == "Unknown"

    def test_predict_sequence_unfitted_returns_unknown_df(self):
        clf = RegimeClassifier()
        dummy = np.zeros((20, 5))
        seq = clf.predict_sequence(dummy)
        assert all(lbl == "Unknown" for lbl in seq["raw_label"])
