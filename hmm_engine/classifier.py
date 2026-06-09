# Path: hmm_engine/classifier.py
"""Hidden Markov Model regime classifier.

Uses a Gaussian HMM with n_components=3 (Bear / Neutral / Bull), fit on the
feature matrix from FeatureEngineer. States are relabeled by mean log return
so state 0 = lowest-return regime (Bear) and state 2 = highest (Bull).

IMPORTANT — causal forward algorithm:
    All predictions use the forward algorithm only (α-messages). The Viterbi
    algorithm is NOT used for production forecasts because it requires the full
    sequence (look-ahead bias). Only hmmlearn's forward-pass `.predict()` is used
    for the most recent observation, which reads only past data.

Reference:
    Rabiner (1989) "A tutorial on hidden Markov models and selected applications
    in speech recognition", Proceedings of the IEEE 77(2) pp. 257-286.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_N_COMPONENTS = 3
_N_ITER       = 200
_RANDOM_STATE = 42

_STATE_LABELS = {0: "Bear", 1: "Neutral", 2: "Bull"}


@dataclass
class RegimeState:
    label:       str         # "Bear" | "Neutral" | "Bull"
    hmm_label:   str         # same as label (kept for compatibility)
    probability: float       # posterior probability of this state
    state_idx:   int         # raw HMM state index (after relabeling)


class RegimeClassifier:
    """GaussianHMM-based regime classifier with forward-algorithm predictions."""

    def __init__(self) -> None:
        self._model    = None
        self._relabel  = {}    # maps raw_state_idx → {0=Bear, 1=Neutral, 2=Bull}
        self._fitted   = False

    def fit(
        self,
        features: np.ndarray,
        returns:  np.ndarray,
    ) -> "RegimeClassifier":
        """Fit GaussianHMM on *features*.

        States are relabeled so that state 0 has the lowest mean return
        (Bear) and state 2 the highest (Bull) — regardless of HMM init order.
        """
        from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

        if len(features) < 60:
            log.warning("RegimeClassifier.fit: only %d observations (need ≥60)", len(features))
            return self

        model = GaussianHMM(
            n_components=_N_COMPONENTS,
            covariance_type="diag",
            n_iter=_N_ITER,
            random_state=_RANDOM_STATE,
        )
        model.fit(features)
        self._model = model

        # Predict on full training set (for relabeling only — not for production)
        raw_states = model.predict(features)
        mean_returns = {}
        for s in range(_N_COMPONENTS):
            idx = np.where(raw_states == s)[0]
            mean_returns[s] = float(np.mean(returns[idx])) if len(idx) > 0 else 0.0

        sorted_states = sorted(mean_returns, key=mean_returns.get)
        self._relabel  = {raw: new for new, raw in enumerate(sorted_states)}
        self._fitted   = True
        log.debug("RegimeClassifier fitted. Mean returns by state: %s", mean_returns)
        return self

    def predict_current(self, recent_features: np.ndarray) -> RegimeState:
        """Predict the current regime from *recent_features* (last N rows).

        Uses the forward algorithm — causal, no look-ahead.
        Returns RegimeState(label="Neutral", probability=0.33) as fallback.
        """
        if not self._fitted or self._model is None:
            return RegimeState("Neutral", "Neutral", 0.33, 1)

        try:
            # Score each state's log-likelihood for the recent window
            _, posteriors = self._model.score_samples(recent_features)
            last_posterior = posteriors[-1]  # shape (n_components,)

            best_raw  = int(np.argmax(last_posterior))
            best_prob = float(last_posterior[best_raw])
            best_new  = self._relabel.get(best_raw, 1)
            label     = _STATE_LABELS.get(best_new, "Neutral")

            return RegimeState(
                label=label,
                hmm_label=label,
                probability=best_prob,
                state_idx=best_new,
            )
        except Exception as exc:
            log.warning("RegimeClassifier.predict_current failed: %s", exc)
            return RegimeState("Neutral", "Neutral", 0.33, 1)

    def predict_sequence(
        self, features: np.ndarray
    ) -> list[RegimeState]:
        """Predict regime state for every row in *features* (for diagnostics).

        Uses forward algorithm posteriors — no look-ahead.
        """
        if not self._fitted or self._model is None:
            return []
        try:
            _, posteriors = self._model.score_samples(features)
            results = []
            for row in posteriors:
                best_raw  = int(np.argmax(row))
                best_prob = float(row[best_raw])
                best_new  = self._relabel.get(best_raw, 1)
                label     = _STATE_LABELS.get(best_new, "Neutral")
                results.append(RegimeState(label, label, best_prob, best_new))
            return results
        except Exception as exc:
            log.warning("RegimeClassifier.predict_sequence failed: %s", exc)
            return []
