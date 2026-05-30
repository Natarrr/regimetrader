"""regime_trader/models/regime_detector.py
Market regime detection — four methods + ensemble.

Methods:
  1. VIX threshold rule   — instantaneous, zero latency, interpretable
  2. HMM (hmmlearn)       — hidden Markov model on (VIX, log-returns) features
  3. ML classifier        — RandomForest on 8 technical/volatility features
  4. Credit Stress        — JNK/LQD/MOVE proxy signal (opt-in, w_credit > 0)

Ensemble:
  - Soft voting: weighted average of method probabilities per class
  - Persistence filter: require N_PERSIST consecutive signals before switching
  - Default weights (3-signal): HMM 0.45, ML 0.35, VIX 0.20
  - With credit (4-signal): VIX 0.30, HMM 0.25, ML 0.25, CREDIT 0.20
  - Credit override rules applied after soft-vote argmax (before persistence)

RegimeLabel mapping (matches core.models.RegimeLabel):
  0 → Crash    VIX > 45
  1 → Panic    VIX 35–45
  2 → Bear     VIX 25–35
  3 → Neutral  VIX 15–25
  4 → Bull     VIX 12–15
  5 → Euphoria VIX < 12

Evaluation:
  evaluate(vix_series, true_labels) → metrics dict
  backtest_comparison(vix_series, returns) → dict with per-method metrics

Public API:
  RegimeDetector.fit(vix_series, returns)
  RegimeDetector.predict(vix_series, returns) → str  (single latest label)
  RegimeDetector.predict_series(vix_df, returns_df) → pd.Series
  RegimeDetector.backtest_report(vix_series, returns) → dict
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Ensemble weights (env-tunable) ────────────────────────────────────────────
_W_HMM = float(os.getenv("REGIME_W_HMM", "0.45"))
_W_ML  = float(os.getenv("REGIME_W_ML",  "0.35"))
_W_VIX = float(os.getenv("REGIME_W_VIX", "0.20"))

# Persistence: require N consecutive same-label signals before committing
N_PERSIST = int(os.getenv("REGIME_PERSIST", "2"))

# Number of HMM hidden states
N_HMM_STATES = int(os.getenv("REGIME_HMM_STATES", "5"))

# ── VIX thresholds → regime ────────────────────────────────────────────────────
# Fixed VIX cutoffs. Volatility clusters and is persistent (Engle 1982, ARCH),
# but this is a static threshold rule, not a GARCH estimate — the HMM/ML
# detectors below capture the dynamics; this rule is a fast, interpretable baseline.
_VIX_THRESHOLDS: List[Tuple[float, str]] = [
    (45.0, "Crash"),
    (35.0, "Panic"),
    (25.0, "Bear"),
    (15.0, "Neutral"),
    (12.0, "Bull"),
    (0.0,  "Euphoria"),
]

# Ordered label list for probability arrays
_LABELS = ["Crash", "Panic", "Bear", "Neutral", "Bull", "Euphoria"]
_LABEL_IDX = {lbl: i for i, lbl in enumerate(_LABELS)}


# ── Method 1: VIX threshold rule ─────────────────────────────────────────────

def vix_rule(vix: float) -> str:
    """Classify regime from a single VIX value using fixed thresholds.

    VIX (implied volatility) is the primary regime indicator here.

    Args:
        vix: CBOE VIX level.

    Returns:
        RegimeLabel string.
    """
    for threshold, label in _VIX_THRESHOLDS:
        if vix >= threshold:
            return label
    return "Euphoria"


def vix_rule_series(vix_series: pd.Series) -> pd.Series:
    """Apply vix_rule to every element of a Series."""
    return vix_series.map(vix_rule)


def vix_proba(vix: float) -> np.ndarray:
    """Soft-vote probability vector from VIX value.

    Returns a probability distribution over _LABELS (sums to 1.0).
    The point-classification label gets 0.70; adjacent labels share 0.30.
    """
    label = vix_rule(vix)
    idx = _LABEL_IDX[label]
    proba = np.zeros(len(_LABELS))
    proba[idx] = 0.70
    # Distribute remaining 0.30 to neighbours
    for neighbour in [idx - 1, idx + 1]:
        if 0 <= neighbour < len(_LABELS):
            proba[neighbour] += 0.15
    # Normalise (edge labels have only one neighbour)
    proba /= proba.sum()
    return proba


# ── Method 2: HMM ─────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """Hidden Markov Model on (normalised VIX, log-returns).

    Regime-switching captures non-linear volatility dynamics that fixed
    thresholds miss (Hamilton 1989, "A New Approach to the Economic Analysis of
    Nonstationary Time Series"; HMM estimation per Rabiner 1989).

    Implementation:
        GaussianHMM from hmmlearn with N_HMM_STATES hidden states.
        States are labelled post-hoc by mean VIX: highest VIX → "Crash".
    """

    def __init__(self, n_states: int = N_HMM_STATES) -> None:
        self.n_states = n_states
        self._model: Any = None
        self._state_to_label: Dict[int, str] = {}
        self._is_fitted = False

    def _features(self, vix: pd.Series, returns: pd.Series) -> np.ndarray:
        """Build feature matrix: [normalised VIX, log-return]."""
        v = vix.values.reshape(-1, 1)
        r = returns.values.reshape(-1, 1)
        X = np.hstack([v / 30.0, r * 100])  # scale to similar magnitude
        return np.nan_to_num(X, nan=0.0)

    def fit(self, vix: pd.Series, returns: pd.Series) -> "HMMRegimeDetector":
        """Fit GaussianHMM on (VIX, returns) features.

        Args:
            vix:     Daily VIX close series (aligned with returns).
            returns: Daily log-return series.

        Returns:
            self
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn required: pip install hmmlearn>=0.3.0")

        X = self._features(vix, returns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
            )
            model.fit(X)

        self._model = model
        self._map_states_to_labels(vix)
        self._is_fitted = True
        log.info("[HMM] Fitted: %d states, converged=%s", self.n_states, model.monitor_.converged)
        return self

    def _map_states_to_labels(self, vix: pd.Series) -> None:
        """Map HMM states to regime labels via mean VIX per state."""
        X = self._features(vix, pd.Series(np.zeros(len(vix))))
        states = self._model.predict(X)
        vix_arr = vix.values

        state_means: Dict[int, float] = {}
        for s in range(self.n_states):
            mask = states == s
            state_means[s] = float(vix_arr[mask].mean()) if mask.any() else 0.0

        # Sort states by mean VIX descending → Crash first
        sorted_states = sorted(state_means.items(), key=lambda x: x[1], reverse=True)

        # Map to labels — compress if fewer states than labels
        label_pool = _LABELS[:]
        self._state_to_label = {}
        for rank, (state, _) in enumerate(sorted_states):
            label_idx = min(rank, len(label_pool) - 1)
            self._state_to_label[state] = label_pool[label_idx]

    def predict_proba(self, vix: pd.Series, returns: pd.Series) -> np.ndarray:
        """Return posterior probability matrix (n_samples, n_labels)."""
        if not self._is_fitted:
            raise RuntimeError("HMMRegimeDetector.fit() must be called first")
        X = self._features(vix, returns)
        state_proba = self._model.predict_proba(X)  # (n, n_states)

        label_proba = np.zeros((len(vix), len(_LABELS)))
        for state, label in self._state_to_label.items():
            label_proba[:, _LABEL_IDX[label]] += state_proba[:, state]

        # Normalise rows
        row_sums = label_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return label_proba / row_sums

    def predict_series(self, vix: pd.Series, returns: pd.Series) -> pd.Series:
        proba = self.predict_proba(vix, returns)
        labels = [_LABELS[np.argmax(row)] for row in proba]
        return pd.Series(labels, index=vix.index)


# ── Method 3: ML Classifier ───────────────────────────────────────────────────

class MLRegimeDetector:
    """RandomForest classifier on 8 volatility/trend features.

    Supervised regime classification from engineered VIX/return features
    (Breiman 2001, "Random Forests", Machine Learning 45(1)).

    Features (all from VIX + returns):
        vix, vix_20d_ma, vix_5d_chg, vix_20d_std,
        ret_5d, ret_20d, ret_5d_std, vix_above_ma
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._is_fitted = False

    @staticmethod
    def _build_features(vix: pd.Series, returns: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"vix": vix, "ret": returns})
        df["vix_20d_ma"] = df["vix"].rolling(20).mean()
        df["vix_5d_chg"] = df["vix"].pct_change(5)
        df["vix_20d_std"] = df["vix"].rolling(20).std()
        df["ret_5d"] = df["ret"].rolling(5).sum()
        df["ret_20d"] = df["ret"].rolling(20).sum()
        df["ret_5d_std"] = df["ret"].rolling(5).std()
        df["vix_above_ma"] = (df["vix"] > df["vix_20d_ma"]).astype(float)
        return df[["vix", "vix_20d_ma", "vix_5d_chg", "vix_20d_std",
                   "ret_5d", "ret_20d", "ret_5d_std", "vix_above_ma"]].fillna(0)

    def fit(
        self,
        vix: pd.Series,
        returns: pd.Series,
        true_labels: Optional[pd.Series] = None,
    ) -> "MLRegimeDetector":
        """Fit RandomForestClassifier.

        If true_labels is None, uses VIX rule as the training target.

        Args:
            vix:         Daily VIX close series.
            returns:     Daily log-return series.
            true_labels: Optional ground-truth labels (pd.Series of strings).

        Returns:
            self
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn>=1.4.0")

        X = self._build_features(vix, returns)
        if true_labels is not None:
            y = true_labels.values
        else:
            y = vix_rule_series(vix).values

        mask = ~np.isnan(X.values).any(axis=1)
        X_clean, y_clean = X.values[mask], y[mask]

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_clean, y_clean)
        self._model = clf
        self._is_fitted = True
        log.info("[ML] RandomForest fitted on %d samples, %d features", len(X_clean), X.shape[1])
        return self

    def predict_proba(self, vix: pd.Series, returns: pd.Series) -> np.ndarray:
        """Return probability matrix aligned to _LABELS order."""
        if not self._is_fitted:
            raise RuntimeError("MLRegimeDetector.fit() must be called first")
        X = self._build_features(vix, returns)
        raw_proba = self._model.predict_proba(X.values)  # (n, n_classes)
        classes = list(self._model.classes_)

        # Re-order columns to match _LABELS
        label_proba = np.zeros((len(vix), len(_LABELS)))
        for cls_idx, cls in enumerate(classes):
            if cls in _LABEL_IDX:
                label_proba[:, _LABEL_IDX[cls]] = raw_proba[:, cls_idx]

        row_sums = label_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return label_proba / row_sums

    def predict_series(self, vix: pd.Series, returns: pd.Series) -> pd.Series:
        proba = self.predict_proba(vix, returns)
        labels = [_LABELS[np.argmax(row)] for row in proba]
        return pd.Series(labels, index=vix.index)


# ── Persistence filter ────────────────────────────────────────────────────────

def apply_persistence_filter(labels: pd.Series, n: int = N_PERSIST) -> pd.Series:
    """Smooth regime sequence: only switch after N consecutive same-label.

    Debounce filter: regime transitions should be persistent, not noise-driven.
    Dampens false positives at the cost of switch latency.

    # Debounce rule: commit to new regime only after n consecutive identical
    # candidate signals (persistence check before committing).

    Args:
        labels: Raw predicted regime labels.
        n:      Number of consecutive signals required before switching.

    Returns:
        Smoothed series with fewer spurious transitions.
    """
    smoothed = labels.copy()
    current = labels.iloc[0]   # committed (output) regime
    candidate = labels.iloc[0] # streak-in-progress regime
    count = 1

    for i in range(1, len(labels)):
        if labels.iloc[i] == candidate:
            count += 1
        else:
            candidate = labels.iloc[i]
            count = 1

        if count >= n:
            current = candidate
        smoothed.iloc[i] = current

    return smoothed


# ── Ensemble ───────────────────────────────────────────────────────────────────

class RegimeDetector:
    """Ensemble regime detector: VIX rule + HMM + ML + optional Credit signal.

    The ensemble is the recommended production configuration.
    VIX rule provides an anchor; HMM captures regime dynamics;
    ML provides non-linear feature interactions.
    Credit signal (optional, w_credit > 0) adds cross-asset early warning.

    Persistence filter (N_PERSIST) is applied after soft voting.
    Credit override rules are applied between argmax and persistence filter.

    Backward-compatible: w_credit=0.0 by default → identical to prior behaviour.
    """

    def __init__(
        self,
        w_hmm: float = _W_HMM,
        w_ml: float = _W_ML,
        w_vix: float = _W_VIX,
        w_credit: float = 0.0,          # opt-in credit signal (default: disabled)
        n_persist: int = N_PERSIST,
        n_hmm_states: int = N_HMM_STATES,
    ) -> None:
        self.w_hmm    = w_hmm
        self.w_ml     = w_ml
        self.w_vix    = w_vix
        self.w_credit = w_credit
        self.n_persist = n_persist

        self.hmm = HMMRegimeDetector(n_states=n_hmm_states)
        self.ml  = MLRegimeDetector()
        self._is_fitted = False

    def fit(
        self,
        vix: pd.Series,
        returns: pd.Series,
        true_labels: Optional[pd.Series] = None,
    ) -> "RegimeDetector":
        """Fit HMM and ML sub-detectors.

        Args:
            vix:         Daily VIX close (pd.Series, DatetimeIndex).
            returns:     Daily log-returns (same index as vix).
            true_labels: Optional ground-truth for ML supervised training.

        Returns:
            self
        """
        log.info("[ENSEMBLE] Fitting HMM…")
        self.hmm.fit(vix, returns)
        log.info("[ENSEMBLE] Fitting ML…")
        self.ml.fit(vix, returns, true_labels)
        self._is_fitted = True
        return self

    def predict_proba_series(
        self,
        vix: pd.Series,
        returns: pd.Series,
        credit_scores: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Return soft-vote probability DataFrame (index=vix.index, cols=_LABELS).

        Args:
            vix:           Daily VIX close series.
            returns:       Daily log-returns.
            credit_scores: Optional daily credit stress scores ∈ [0, 1].
                           Used only when self.w_credit > 0.

        Returns:
            DataFrame (n_days × 6), columns = _LABELS, rows sum to 1.0.
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeDetector.fit() must be called first")

        vix_proba_mat = np.array([vix_proba(v) for v in vix.values])
        hmm_proba = self.hmm.predict_proba(vix, returns)
        ml_proba  = self.ml.predict_proba(vix, returns)

        if self.w_credit > 0 and credit_scores is not None:
            from regime_trader.models.credit_regime_detector import credit_score_to_vix_proba as _cs2p
            # Align credit_scores to vix index; fill gaps with 0.5 (neutral)
            aligned = credit_scores.reindex(vix.index).fillna(0.5)
            credit_proba = np.array([_cs2p(float(s)) for s in aligned.values])

            w_sum = self.w_vix + self.w_hmm + self.w_ml + self.w_credit
            ensemble = (
                self.w_vix    * vix_proba_mat
                + self.w_hmm  * hmm_proba
                + self.w_ml   * ml_proba
                + self.w_credit * credit_proba
            ) / w_sum
        else:
            w_sum = self.w_vix + self.w_hmm + self.w_ml
            ensemble = (
                self.w_vix  * vix_proba_mat
                + self.w_hmm * hmm_proba
                + self.w_ml  * ml_proba
            ) / w_sum

        row_sums = ensemble.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        ensemble /= row_sums

        return pd.DataFrame(ensemble, index=vix.index, columns=_LABELS)

    def predict_series(
        self,
        vix: pd.Series,
        returns: pd.Series,
        apply_filter: bool = True,
        credit_scores: Optional[pd.Series] = None,
        credit_regimes: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Return regime label series, optionally persistence-filtered.

        Credit override rules are applied per-day between argmax and the
        persistence filter (so overrides are themselves smoothed).

        Args:
            vix:            Daily VIX close series.
            returns:        Daily log-returns.
            apply_filter:   Apply persistence filter (default True).
            credit_scores:  Optional credit stress scores ∈ [0, 1].
            credit_regimes: Optional CreditRegime series (for override rules).
                            If None but credit_scores provided, derived automatically.

        Returns:
            pd.Series of regime label strings.
        """
        proba_df = self.predict_proba_series(vix, returns, credit_scores)
        raw_labels: List[str] = [_LABELS[np.argmax(row)] for row in proba_df.values]

        # Apply credit override rules if credit signal is active
        if self.w_credit > 0 and credit_scores is not None:
            from regime_trader.models.credit_regime_detector import (
                classify_credit_regime as _classify,
                apply_credit_overrides as _override,
            )
            aligned_scores = credit_scores.reindex(vix.index).fillna(0.5)
            if credit_regimes is not None:
                aligned_regimes = credit_regimes.reindex(vix.index)
            else:
                aligned_regimes = aligned_scores.map(_classify)

            for i, (label, cr, vx) in enumerate(
                zip(raw_labels, aligned_regimes.values, vix.values)
            ):
                raw_labels[i] = _override(label, cr, float(vx))

        raw = pd.Series(raw_labels, index=vix.index)

        if apply_filter and self.n_persist > 1:
            return apply_persistence_filter(raw, self.n_persist)
        return raw

    def predict(
        self,
        vix: pd.Series,
        returns: pd.Series,
        credit_scores: Optional[pd.Series] = None,
        credit_regimes: Optional[pd.Series] = None,
    ) -> str:
        """Return the single latest regime label.

        Args:
            vix:            VIX series (at least 20 days for ML features).
            returns:        Returns series (same length).
            credit_scores:  Optional credit stress scores (enables credit signal).
            credit_regimes: Optional CreditRegime series (for overrides).

        Returns:
            Latest regime label string.
        """
        series = self.predict_series(
            vix, returns,
            credit_scores=credit_scores,
            credit_regimes=credit_regimes,
        )
        return str(series.iloc[-1])

    # ── Evaluation / backtest ─────────────────────────────────────────────────

    def backtest_report(
        self,
        vix: pd.Series,
        returns: pd.Series,
        true_labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Compare all three methods and ensemble on historical data.

        Metrics per method:
          - accuracy        : fraction of correctly labelled days (vs VIX rule baseline)
          - detection_lag   : mean days from regime change to first correct detection
          - false_positives : spurious transitions per year
          - transitions     : total label changes

        Args:
            vix:         Historical VIX series.
            returns:     Historical returns series.
            true_labels: Ground truth (if None, uses VIX rule as reference).

        Returns:
            Dict with 'methods' key containing per-method metrics,
            and 'ensemble' key for ensemble-specific stats.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before backtest_report()")

        baseline = vix_rule_series(vix)
        reference = true_labels if true_labels is not None else baseline

        vix_labels = baseline
        hmm_labels = self.hmm.predict_series(vix, returns)
        ml_labels = self.ml.predict_series(vix, returns)
        ens_labels = self.predict_series(vix, returns)

        def _metrics(predicted: pd.Series, ref: pd.Series) -> Dict[str, Any]:
            aligned = predicted.align(ref, join="inner")
            pred_a, ref_a = aligned[0], aligned[1]
            accuracy = float((pred_a == ref_a).mean())

            # Detection lag: days from first regime change in ref to correct detection
            lags: List[float] = []
            in_lag = False
            lag_start = 0
            for i in range(1, len(ref_a)):
                if ref_a.iloc[i] != ref_a.iloc[i - 1]:
                    in_lag = True
                    lag_start = i
                if in_lag and pred_a.iloc[i] == ref_a.iloc[i]:
                    lags.append(float(i - lag_start))
                    in_lag = False
            det_lag = float(np.mean(lags)) if lags else float("nan")

            # False positives: transitions in predicted that are NOT in reference
            pred_trans = (pred_a != pred_a.shift()).sum() - 1
            ref_trans = (ref_a != ref_a.shift()).sum() - 1
            n_years = max(len(ref_a) / 252, 1)
            fp_per_year = max(0, pred_trans - ref_trans) / n_years

            return {
                "accuracy": round(accuracy, 4),
                "detection_lag_days": round(det_lag, 2) if not np.isnan(det_lag) else None,
                "false_positives_per_year": round(fp_per_year, 2),
                "transitions": int(pred_trans),
            }

        report = {
            "n_samples": len(vix),
            "date_range": {
                "start": str(vix.index[0])[:10] if hasattr(vix.index[0], "__str__") else "",
                "end": str(vix.index[-1])[:10] if hasattr(vix.index[-1], "__str__") else "",
            },
            "methods": {
                "vix_rule":  _metrics(vix_labels, reference),
                "hmm":       _metrics(hmm_labels, reference),
                "ml":        _metrics(ml_labels, reference),
                "ensemble":  _metrics(ens_labels, reference),
            },
            "ensemble_weights": {
                "hmm":    self.w_hmm,
                "ml":     self.w_ml,
                "vix":    self.w_vix,
                "credit": self.w_credit,   # 0.0 when disabled (backward-compat)
            },
            "persistence_n": self.n_persist,
        }

        # Log summary
        for method, m in report["methods"].items():
            log.info(
                "[BACKTEST] %-10s acc=%.2f%%  lag=%-5s  fp/yr=%.1f  transitions=%d",
                method,
                m["accuracy"] * 100,
                str(m["detection_lag_days"]),
                m["false_positives_per_year"],
                m["transitions"],
            )

        return report


# ── Standalone evaluation helper (for CI/testing) ────────────────────────────

def evaluate(
    vix_series: pd.Series,
    true_labels: pd.Series,
) -> Dict[str, Any]:
    """Run a quick VIX-rule-only evaluation without fitting ML/HMM.

    Useful for unit tests and smoke checks.

    Args:
        vix_series:  Daily VIX close values.
        true_labels: Ground-truth regime labels.

    Returns:
        Metrics dict: accuracy, detection_lag_days, false_positives_per_year.
    """
    predicted = vix_rule_series(vix_series)
    correct = (predicted == true_labels).sum()
    accuracy = float(correct / len(true_labels)) if len(true_labels) > 0 else 0.0

    transitions_pred = int((predicted != predicted.shift()).sum() - 1)
    transitions_true = int((true_labels != true_labels.shift()).sum() - 1)
    n_years = max(len(true_labels) / 252, 1)
    fp_yr = max(0, transitions_pred - transitions_true) / n_years

    return {
        "method": "vix_rule",
        "n_samples": len(vix_series),
        "accuracy": round(accuracy, 4),
        "false_positives_per_year": round(fp_yr, 2),
        "transitions_predicted": transitions_pred,
        "transitions_true": transitions_true,
    }
