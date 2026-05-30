"""hmm_engine/classifier.py
GaussianHMM regime classifier.

Reference: Rabiner (1989), "A Tutorial on Hidden Markov Models", Proc. IEEE 77(2);
parameters fit via Baum-Welch EM (Baum et al. 1970).

predict_current() uses the forward algorithm only (no Viterbi decoding), so the
current-bar posterior depends only on past observations — backtests are free of
look-ahead / future-data contamination.

$q_t \\sim \\text{HMM}(\\pi, A, B)$ — forward filter only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ── State label maps ───────────────────────────────────────────────────────────

_LABEL_MAP = {0: "Bear", 1: "Neutral", 2: "Bull"}
_COLOR_MAP  = {"Bull": "#00FFA3", "Neutral": "#60A5FA", "Bear": "#FF6B6B",
               "Unknown": "#888888"}

# Position scales per raw state: scale exposure down as the regime deteriorates
# (volatility-targeting heuristic — Bear/Crash regimes carry higher tail risk).
_SCALE_MAP = {"Bull": 1.0, "Neutral": 0.6, "Bear": 0.2, "Unknown": 1.0}

# A state is "confirmed" when the last _CONFIRM_WINDOW bars agree.
_CONFIRM_WINDOW = 5

# A state is "uncertain" when its max posterior < _UNCERTAIN_THRESHOLD.
_UNCERTAIN_THRESHOLD = 0.60


@dataclass
class RegimeState:
    """Snapshot of the HMM classifier output for the current bar."""

    raw_label:       Optional[str]
    confirmed_label: Optional[str]
    position_scale:  float
    is_uncertain:    bool
    regime_probs:    Optional[List[float]]
    color_map:       Dict[str, str] = field(default_factory=lambda: dict(_COLOR_MAP))


class RegimeClassifier:
    """Three-state Gaussian HMM regime classifier (Rabiner 1989).

    State ordering is normalised post-fit by mean return so that state 0 = Bear,
    state 1 = Neutral, state 2 = Bull regardless of HMM initialisation order.

    Args:
        n_components: Number of HMM hidden states (default 3).
        n_iter:       EM iterations (default 100).
        covariance_type: GaussianHMM covariance ('full' | 'diag' | 'tied').
        returns_feature_index: Column index of log-returns in the feature matrix
            (used for state ordering by mean return). Default 0 matches
            FeatureEngineer's column layout.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 100,
        covariance_type: str = "full",
        returns_feature_index: int = 0,
        random_state: int = 42,
    ) -> None:
        self._n      = n_components
        self._ridx   = returns_feature_index
        self._model  = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self._order: Optional[np.ndarray] = None  # maps raw state → sorted state

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, returns: np.ndarray) -> "RegimeClassifier":
        """Fit GaussianHMM via Baum-Welch EM (Baum et al. 1970).

        States are reordered post-fit by mean return so that state indices are
        stable: 0 = Bear, 1 = Neutral, 2 = Bull.

        $\\hat{\\theta} = \\arg\\max_{\\theta} \\log P(X_{1:T} | \\theta)$

        Args:
            features: (T, F) float64 array from FeatureEngineer.build().
            returns:  (T,) float64 log-returns (used only for state ordering).

        Returns:
            self (for chaining).
        """
        self._model.fit(features)

        # Decode training sequence with Viterbi — used ONLY for post-fit
        # state ordering, never exposed as a prediction output.
        raw_states = self._model.predict(features)

        # Order states by mean return: 0→Bear (lowest), 2→Bull (highest).
        mean_ret_by_state = np.array([
            returns[raw_states == s].mean() if (raw_states == s).any() else 0.0
            for s in range(self._n)
        ])
        self._order = np.argsort(mean_ret_by_state)  # ascending: low → high

        return self

    # ── _state_to_label ───────────────────────────────────────────────────────

    def _state_to_label(self, raw_state: int) -> str:
        """Map a raw HMM state index to a named label via the ordered mapping."""
        if self._order is None:
            return "Unknown"
        rank = int(np.where(self._order == raw_state)[0][0])
        return _LABEL_MAP.get(rank, "Unknown")

    # ── predict_current ───────────────────────────────────────────────────────

    def predict_current(self, features_window: np.ndarray) -> RegimeState:
        """Forward-filter only: no Viterbi, no look-ahead (Rabiner 1989 §III).

        Uses the HMM posterior (forward algorithm) to estimate the current
        regime probability. The «confirmed» label requires _CONFIRM_WINDOW
        consecutive bars to agree.

        $\\alpha_t(i) = P(X_{1:t}, q_t = i | \\theta)$

        Args:
            features_window: (W, F) array — the last W bars of features.
                Recommended W ≥ _CONFIRM_WINDOW (≥5). Caller typically passes
                features[-20:] for a 20-bar confirmation window.

        Returns:
            RegimeState with raw/confirmed labels, position scale, and state probs.
        """
        if self._order is None:
            return RegimeState(
                raw_label="Unknown", confirmed_label=None,
                position_scale=1.0, is_uncertain=False,
                regime_probs=None,
            )

        # Forward algorithm: posteriors for each timestep in the window.
        # posteriors shape: (W, n_components)
        log_prob, posteriors = self._model.score_samples(features_window)

        # Current-bar posterior (last row).
        current_probs = posteriors[-1]
        raw_state_idx = int(np.argmax(current_probs))
        raw_label     = self._state_to_label(raw_state_idx)
        max_prob      = float(current_probs[raw_state_idx])
        is_uncertain  = max_prob < _UNCERTAIN_THRESHOLD

        # Confirmation: derive the MAP state for each bar in the window,
        # then check that the last _CONFIRM_WINDOW bars all agree.
        window_states  = np.argmax(posteriors, axis=1)
        window_labels  = [self._state_to_label(int(s)) for s in window_states]
        recent         = window_labels[-_CONFIRM_WINDOW:]
        confirmed_label: Optional[str] = (
            recent[-1] if len(set(recent)) == 1 else None
        )

        # Reorder probs to match the Bear/Neutral/Bull rank order.
        ordered_probs = [
            float(current_probs[int(raw)]) for raw in self._order
        ]

        return RegimeState(
            raw_label=raw_label,
            confirmed_label=confirmed_label,
            position_scale=_SCALE_MAP.get(raw_label, 1.0),
            is_uncertain=is_uncertain,
            regime_probs=ordered_probs,
            color_map=dict(_COLOR_MAP),
        )

    # ── predict_sequence ──────────────────────────────────────────────────────

    def predict_sequence(self, features: np.ndarray) -> pd.DataFrame:
        """Annotate the full history with regime labels (MAP state per bar).

        Uses the MAP state per bar (argmax posterior).  The «confirmed» column
        marks bars where the surrounding _CONFIRM_WINDOW bars all agreed — a
        signal that the regime is stable rather than transitioning.

        Args:
            features: (T, F) float64 — full training/inference feature matrix.

        Returns:
            DataFrame with columns:
              - raw_label:       MAP label for each bar.
              - confirmed_label: Label if confirmed, else None.
        """
        if self._order is None:
            return pd.DataFrame({
                "raw_label":       ["Unknown"] * len(features),
                "confirmed_label": [None]      * len(features),
            })

        _, posteriors  = self._model.score_samples(features)
        map_states     = np.argmax(posteriors, axis=1)
        raw_labels     = [self._state_to_label(int(s)) for s in map_states]

        # Sliding window confirmation: a bar is confirmed when all neighbours
        # in [t - W//2, t + W//2] agree (centred window for the history view).
        half = _CONFIRM_WINDOW // 2
        confirmed_labels: List[Optional[str]] = []
        for i, lbl in enumerate(raw_labels):
            lo = max(0, i - half)
            hi = min(len(raw_labels), i + half + 1)
            window_slice = raw_labels[lo:hi]
            confirmed_labels.append(lbl if len(set(window_slice)) == 1 else None)

        return pd.DataFrame({
            "raw_label":       raw_labels,
            "confirmed_label": confirmed_labels,
        })
