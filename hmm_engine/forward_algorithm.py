"""
forward_algorithm.py
────────────────────
Causal (no look-ahead) regime inference via the HMM forward algorithm.

WHY NOT model.predict()
───────────────────────
hmmlearn's model.predict() runs the Viterbi algorithm over the *full*
observation sequence.  Viterbi is a two-pass algorithm (forward + backward
traceback), meaning the state assigned to bar t depends on observations
AFTER bar t.  In a live trading context that is a look-ahead bias that
inflates backtest performance but is impossible to replicate in production.

THE FORWARD ALGORITHM
─────────────────────
The forward variable α_t(i) = P(o₁, …, o_t, q_t = i | λ) is defined
recursively using only observations up to and including time t:

  Initialisation   α₁(i)  = π_i · P(o₁ | q₁ = i)

  Recursion        α_t(j)  = P(o_t | q_t = j) · Σᵢ [α_{t-1}(i) · A_{ij}]

Normalising gives the causal posterior P(q_t = i | o₁, …, o_t), which can
be computed bar-by-bar using only information available at that moment.

NUMERICAL STABILITY
───────────────────
All computation is performed in log-space using scipy.special.logsumexp to
avoid floating-point underflow on long sequences.  The vectorised recursion
processes all states simultaneously at each time step.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from hmmlearn import hmm


# ── Public API ────────────────────────────────────────────────────────────────


def forward_filter(
    model: hmm.GaussianHMM,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Compute causal posterior state probabilities for every bar.

    Parameters
    ----------
    model        : A fitted hmmlearn GaussianHMM instance.
    observations : np.ndarray of shape (T, n_features).

    Returns
    -------
    posteriors : np.ndarray of shape (T, n_states)
        posteriors[t, i]  =  P(q_t = i | o₁, …, o_t)

        Each row sums to 1.  Row t uses ONLY information from bars 0 … t,
        making it safe for both live inference and bias-free backtesting.

    Notes
    -----
    - Uses hmmlearn's internal _compute_log_likelihood for emission probs.
    - Vectorised over states at each time step; loops only over T.
    """
    n_states: int = model.n_components
    T: int = len(observations)

    if T == 0:
        raise ValueError(
            "forward_filter received an empty observation sequence (T=0). "
            "Ensure the feature matrix has at least one row before calling predict."
        )

    # ── Emission log-probabilities: log P(o_t | q_t = j)  shape (T, n_states)
    log_emission: np.ndarray = model._compute_log_likelihood(observations)

    # ── Transition log-matrix: log_transmat[i, j] = log P(q_t = j | q_{t-1} = i)
    log_transmat: np.ndarray = np.log(model.transmat_ + 1e-300)  # (n_states, n_states)

    # ── Allocate log-alpha array
    log_alpha = np.empty((T, n_states), dtype=np.float64)

    # ── Initialisation ────────────────────────────────────────────────────────
    log_alpha[0] = np.log(model.startprob_ + 1e-300) + log_emission[0]

    # ── Recursion  (vectorised over states) ───────────────────────────────────
    #
    #   log α_t(j) = log P(o_t | j)
    #              + logsumexp_i( log α_{t-1}(i) + log A_{i→j} )
    #
    #   Broadcasting:
    #     log_alpha[t-1][:, np.newaxis]  →  (n_states, 1)   = α_{t-1}(i)
    #     log_transmat                   →  (n_states, n_states)  A_{i→j}
    #     sum along axis-0              →  (n_states,)       = Σᵢ for each j
    #
    for t in range(1, T):
        transition_scores = log_alpha[t - 1, :, np.newaxis] + log_transmat  # (S, S)
        log_alpha[t] = logsumexp(transition_scores, axis=0) + log_emission[t]

    # ── Normalise to causal posterior  P(q_t = i | o₁…t) ─────────────────────
    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)          # (T, 1)
    log_posteriors = log_alpha - log_norm                           # (T, S)

    return np.exp(log_posteriors)


def decode_sequence_causal(
    model: hmm.GaussianHMM,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Return the most-probable state index at every bar using causal posteriors.

    Equivalent to argmax over the forward-filter output.  No look-ahead.

    Parameters
    ----------
    model        : fitted GaussianHMM
    observations : np.ndarray of shape (T, n_features)

    Returns
    -------
    states : np.ndarray of shape (T,), dtype int
    """
    return np.argmax(forward_filter(model, observations), axis=1)
