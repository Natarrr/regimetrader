"""
classifier.py
─────────────
RegimeClassifier — the single public entry-point for the hmm_engine.

Wires together the four sub-modules into one coherent interface:

    ┌─────────────────────────────────────────────────────────────┐
    │                     RegimeClassifier                        │
    │                                                             │
    │  .fit()                                                     │
    │   ├─ model_selector  → auto-selects n_states ∈ {3 … 7}     │
    │   ├─ regime_labeler  → names states by mean return rank     │
    │   └─ StabilityFilter → initialised with the label map       │
    │                                                             │
    │  .predict_current(window)  ← live / streaming bar           │
    │   └─ forward_filter → last-bar posterior → stability gate   │
    │                                                             │
    │  .predict_sequence(history) ← bias-free backtesting         │
    │   └─ forward_filter → all bars → stability gate → DataFrame │
    └─────────────────────────────────────────────────────────────┘

MANDATORY TRAINING RULE
────────────────────────
fit() enforces a minimum of 2 × 252 = 504 bars (~two trading years) of
daily data.  Shorter windows are rejected with an informative ValueError.

NO LOOK-AHEAD BIAS GUARANTEE
──────────────────────────────
Both prediction methods use forward_filter() exclusively.  The Viterbi-
based model.predict() is never called anywhere in this module.

THREAD SAFETY
─────────────
The StabilityFilter is stateful and NOT thread-safe.  In a multi-asset
live system, instantiate one RegimeClassifier per asset or protect calls
with a lock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm

from .forward_algorithm import forward_filter
from .model_selector import SelectionResult, select_best_model
from .regime_labeler import REGIME_COLORS, label_regimes, regime_summary
from .stability_filter import FilterResult, StabilityFilter

logger = logging.getLogger(__name__)

# Minimum bars for training: 2 years × 252 trading days per year
_MIN_TRAIN_BARS: int = 2 * 252


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RegimeState:
    """
    Snapshot of the classifier output for a single bar.

    Fields
    ------
    raw_regime        : HMM state index before any filtering.
    raw_label         : Label for raw_regime.
    confirmed_regime  : State index after 3-bar persistence gate;
                        None until at least one regime is confirmed.
    confirmed_label   : Label for confirmed_regime, or None.
    regime_probs      : np.ndarray (n_states,) — causal forward posteriors.
                        Index matches state indices, NOT label rank.
    is_uncertain      : True when flicker_count > threshold in last 20 bars.
    flicker_count     : Transitions in the last 20-bar rolling window.
    position_scale    : Multiply all position targets by this (0.5 if uncertain).
    streak            : Consecutive bars the current candidate has held.
    n_regimes         : Number of HMM states in the fitted model.
    label_map         : {state_index → label} for this classifier.
    """
    raw_regime:       int
    raw_label:        str
    confirmed_regime: Optional[int]
    confirmed_label:  Optional[str]
    regime_probs:     np.ndarray
    is_uncertain:     bool
    flicker_count:    int
    position_scale:   float
    streak:           int
    n_regimes:        int
    label_map:        Dict[int, str]
    color_map:        Dict[str, str] = field(default_factory=dict)


# ── Classifier ────────────────────────────────────────────────────────────────


class RegimeClassifier:
    """
    HMM-based volatility / regime classifier.

    Quick-start
    ───────────
    >>> from hmm_engine import RegimeClassifier
    >>> clf = RegimeClassifier()
    >>> clf.fit(features, returns)           # train on 2+ years of daily data
    >>> state = clf.predict_current(window)  # live bar — causal forward pass
    >>> df    = clf.predict_sequence(hist)   # backtest — full history, no bias

    Parameters
    ----------
    covariance_type      : GaussianHMM covariance structure — "full" recommended.
    n_iter               : Maximum EM iterations per model fit.
    random_state         : Base seed for reproducible model selection.
    returns_feature_index: Column index in the feature matrix that holds
                           log-returns — used for regime labelling by mean return.
    persistence_bars     : Bars a raw regime must hold before being confirmed (≥1).
    flicker_window       : Rolling window for flicker-count computation (bars).
    flicker_threshold    : Max transitions before declaring uncertainty.
    uncertain_scale      : Position multiplier applied when is_uncertain is True.
    """

    def __init__(
        self,
        covariance_type:       str   = "full",
        n_iter:                int   = 200,
        random_state:          int   = 42,
        returns_feature_index: int   = 0,
        persistence_bars:      int   = 3,
        flicker_window:        int   = 20,
        flicker_threshold:     int   = 4,
        uncertain_scale:       float = 0.5,
    ) -> None:
        self._covariance_type       = covariance_type
        self._n_iter                = n_iter
        self._random_state          = random_state
        self._returns_feature_index = returns_feature_index
        self._persistence_bars      = persistence_bars
        self._flicker_window        = flicker_window
        self._flicker_threshold     = flicker_threshold
        self._uncertain_scale       = uncertain_scale

        self._selection: Optional[SelectionResult] = None
        self._label_map: Dict[int, str]            = {}
        self._color_map: Dict[str, str]            = {}
        self._filter:    Optional[StabilityFilter] = None
        self._is_fitted: bool                      = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "RegimeClassifier":
        """
        Train the HMM classifier on historical data.

        Parameters
        ----------
        features : np.ndarray, shape (T, n_features)
            Scaled observation matrix.  Column 0 (or returns_feature_index)
            must be log-returns; other columns may include realised volatility,
            volume ratio, RSI, etc.  Must have ≥ 504 rows (2 trading years).
        returns  : np.ndarray, shape (T,)
            Raw log-returns — only used for labelling; may equal features[:, 0].

        Returns
        -------
        self  (for method chaining)

        Raises
        ------
        ValueError  : if features has fewer than 504 rows.
        RuntimeError: if all model fits fail to converge.
        """
        self._validate_training_data(features)

        # ── 1. Select optimal n_states via BIC ────────────────────────────────
        logger.info(
            "RegimeClassifier.fit() | T=%d bars, F=%d features",
            len(features),
            features.shape[1] if features.ndim > 1 else 1,
        )
        self._selection = select_best_model(
            features,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
        )

        # ── 2. Label states by mean return rank ───────────────────────────────
        self._label_map = label_regimes(
            self._selection.best_model,
            returns_feature_index=self._returns_feature_index,
        )
        self._color_map = {
            lbl: REGIME_COLORS[lbl]
            for lbl in self._label_map.values()
            if lbl in REGIME_COLORS
        }

        # ── 3. Log summary table ──────────────────────────────────────────────
        summary = regime_summary(self._selection.best_model, self._label_map)
        logger.info("Regime statistics:\n%s", summary)

        # ── 4. Initialise stability filter ────────────────────────────────────
        self._filter = StabilityFilter(
            label_map         = self._label_map,
            persistence       = self._persistence_bars,
            flicker_window    = self._flicker_window,
            flicker_threshold = self._flicker_threshold,
            uncertain_scale   = self._uncertain_scale,
        )

        self._is_fitted = True
        logger.info(
            "RegimeClassifier ready | n_regimes=%d | labels=%s",
            self._selection.best_n,
            {i: l for i, l in sorted(self._label_map.items())},
        )
        return self

    # ── Live inference ────────────────────────────────────────────────────────

    def predict_current(self, features_window: np.ndarray) -> RegimeState:
        """
        Classify the regime for the most recent (last) bar in a feature window.

        Uses ONLY the forward algorithm — no future data, no look-ahead.

        Parameters
        ----------
        features_window : np.ndarray, shape (T, n_features)
            Feature matrix ending at the current bar.  T should be at least
            a few bars so the forward pass has meaningful context.  In
            production, pass all bars since the last model retrain.

        Returns
        -------
        RegimeState for the latest bar (features_window[-1]).
        """
        self._assert_fitted()

        posteriors = forward_filter(self._selection.best_model, features_window)
        last_probs = posteriors[-1]                 # (n_states,) for current bar
        raw_regime = int(np.argmax(last_probs))

        result: FilterResult = self._filter.update(raw_regime)
        return self._build_state(result, last_probs)

    # ── Backtesting ───────────────────────────────────────────────────────────

    def predict_sequence(self, features: np.ndarray) -> pd.DataFrame:
        """
        Classify every bar in a historical feature array without look-ahead.

        The stability filter is reset before the run to prevent state leakage
        from prior calls.

        Parameters
        ----------
        features : np.ndarray, shape (T, n_features)

        Returns
        -------
        pd.DataFrame with one row per bar and columns:
            raw_regime, raw_label,
            confirmed_regime, confirmed_label,
            is_uncertain, flicker_count, position_scale, streak,
            prob_<Label>  ×  n_regimes   (causal forward posteriors)

        Notes
        -----
        - Only forward_filter() is used — no model.predict(), no Viterbi.
        - The stability filter sees bars in chronological order, so its
          state at bar t depends only on bars 0 … t-1 — still causal.
        """
        self._assert_fitted()
        self._filter.reset()

        posteriors  = forward_filter(self._selection.best_model, features)  # (T, S)
        raw_regimes = np.argmax(posteriors, axis=1)                          # (T,)

        records: List[dict] = []
        for t, raw in enumerate(raw_regimes):
            result = self._filter.update(int(raw))
            row: dict = {
                "raw_regime":       result.raw_regime,
                "raw_label":        result.raw_label,
                "confirmed_regime": result.confirmed_regime,
                "confirmed_label":  result.confirmed_label,
                "is_uncertain":     result.is_uncertain,
                "flicker_count":    result.flicker_count,
                "position_scale":   result.position_scale,
                "streak":           result.streak,
            }
            # Append per-regime forward probabilities for analysis
            for state_idx, label in self._label_map.items():
                row[f"prob_{label}"] = float(posteriors[t, state_idx])
            records.append(row)

        return pd.DataFrame(records)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_regimes(self) -> int:
        self._assert_fitted()
        return self._selection.best_n

    @property
    def label_map(self) -> Dict[int, str]:
        self._assert_fitted()
        return dict(self._label_map)

    @property
    def color_map(self) -> Dict[str, str]:
        return dict(self._color_map)

    @property
    def bic_scores(self) -> Dict[int, float]:
        self._assert_fitted()
        return dict(self._selection.bic_scores)

    @property
    def model(self) -> hmm.GaussianHMM:
        self._assert_fitted()
        return self._selection.best_model

    @property
    def filter(self) -> StabilityFilter:
        self._assert_fitted()
        return self._filter

    # ── Private ───────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "RegimeClassifier is not fitted.  Call .fit(features, returns) first."
            )

    @staticmethod
    def _validate_training_data(features: np.ndarray) -> None:
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        n_bars = len(features)
        if n_bars < _MIN_TRAIN_BARS:
            raise ValueError(
                f"RegimeClassifier requires at least {_MIN_TRAIN_BARS} bars "
                f"(2 trading years × 252 days/year) to train.  "
                f"Received {n_bars} bars."
            )
        if np.isnan(features).any():
            bad = int(np.isnan(features).any(axis=1).sum())
            raise ValueError(
                f"features contains NaN values in {bad} row(s).  "
                "Impute or drop missing data before calling .fit()."
            )

    def _build_state(
        self,
        result: FilterResult,
        probs: np.ndarray,
    ) -> RegimeState:
        return RegimeState(
            raw_regime       = result.raw_regime,
            raw_label        = result.raw_label,
            confirmed_regime = result.confirmed_regime,
            confirmed_label  = result.confirmed_label,
            regime_probs     = probs,
            is_uncertain     = result.is_uncertain,
            flicker_count    = result.flicker_count,
            position_scale   = result.position_scale,
            streak           = result.streak,
            n_regimes        = self._selection.best_n,
            label_map        = dict(self._label_map),
            color_map        = dict(self._color_map),
        )
