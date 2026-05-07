"""
model_selector.py
─────────────────
Automatically selects the optimal number of HMM hidden states (regimes)
by fitting models for n_states ∈ {3, 4, 5, 6, 7} and comparing them on
the Bayesian Information Criterion (BIC).

BIC FORMULA
───────────
    BIC = −2·ℓ + k·ln(n)

where
  ℓ  = log-likelihood of the fitted model on the training data
  k  = number of free parameters
  n  = number of observations

Lower BIC = better model after penalising complexity.  BIC penalises extra
parameters more strongly than AIC for n > 7 (always true here), making it
the preferred criterion for selecting parsimonious regime models.

FREE-PARAMETER COUNT  (GaussianHMM, full covariance)
─────────────────────
  start probs          : S − 1              (sum-to-1 constraint)
  transition matrix    : S·(S − 1)         (each row sums to 1)
  means                : S·F
  full covariances     : S·F·(F+1)/2       (symmetric matrix)

where S = n_states, F = n_features.  Other covariance types are also
supported and counted correctly.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)

MIN_REGIMES: int = 3
MAX_REGIMES: int = 7

# Number of EM restarts per (n_states, seed) combination.
# More restarts ↑ chance of finding the global optimum; slower training.
N_INIT: int = 5


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ModelFitResult:
    """Diagnostics for a single (n_states) fit."""
    n_states: int
    model: Optional[hmm.GaussianHMM]
    log_likelihood: float
    n_params: int
    bic: float
    converged: bool


@dataclass
class SelectionResult:
    """Output of select_best_model()."""
    best_n: int
    best_model: hmm.GaussianHMM
    bic_scores: Dict[int, float]              = field(default_factory=dict)
    log_likelihoods: Dict[int, float]         = field(default_factory=dict)
    fit_results: Dict[int, ModelFitResult]    = field(default_factory=dict)


# ── Parameter counting ───────────────────────────────────────────────────────


def _count_free_parameters(
    n_states: int,
    n_features: int,
    covariance_type: str = "full",
) -> int:
    """Return the number of free parameters for a GaussianHMM configuration."""
    n_start  = n_states - 1
    n_trans  = n_states * (n_states - 1)
    n_means  = n_states * n_features

    if covariance_type == "full":
        n_cov = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        n_cov = n_states * n_features
    elif covariance_type == "tied":
        n_cov = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        n_cov = n_states
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type!r}")

    return n_start + n_trans + n_means + n_cov


# ── Model fitting ─────────────────────────────────────────────────────────────


def _fit_single(
    observations: np.ndarray,
    n_states: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
) -> ModelFitResult:
    """
    Fit one GaussianHMM with N_INIT random restarts, keep the best.

    Multiple restarts help escape local optima in the EM algorithm.
    """
    n_obs, n_features = observations.shape
    best_ll: float = -np.inf
    best_model: Optional[hmm.GaussianHMM] = None

    for seed in range(random_state, random_state + N_INIT):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=seed,
                tol=1e-4,
            )
            try:
                candidate.fit(observations)
                ll = candidate.score(observations)
                if ll > best_ll:
                    best_ll = ll
                    best_model = candidate
            except Exception:
                continue  # degenerate init; try next seed

    if best_model is None:
        return ModelFitResult(
            n_states=n_states,
            model=None,
            log_likelihood=-np.inf,
            n_params=0,
            bic=np.inf,
            converged=False,
        )

    n_params = _count_free_parameters(n_states, n_features, covariance_type)
    bic = -2.0 * best_ll + n_params * np.log(n_obs)

    return ModelFitResult(
        n_states=n_states,
        model=best_model,
        log_likelihood=best_ll,
        n_params=n_params,
        bic=bic,
        converged=True,
    )


# ── Main selector ─────────────────────────────────────────────────────────────


def select_best_model(
    observations: np.ndarray,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
) -> SelectionResult:
    """
    Fit GaussianHMM for n_states ∈ {3 … 7} and return the best by BIC.

    Parameters
    ----------
    observations    : np.ndarray of shape (T, n_features)
                      Pre-scaled feature matrix (e.g. from feature_engineering).
    covariance_type : hmmlearn covariance type — "full" recommended.
    n_iter          : Maximum EM iterations per fit.
    random_state    : Base seed; N_INIT restarts use seeds [base, base+N_INIT).

    Returns
    -------
    SelectionResult with best model, BIC table, and per-n diagnostics.
    """
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)

    logger.info(
        "Model selection: fitting GaussianHMM for n_states ∈ {%s} "
        "(covariance=%s, n_iter=%d, %d restarts each)",
        ", ".join(str(n) for n in range(MIN_REGIMES, MAX_REGIMES + 1)),
        covariance_type,
        n_iter,
        N_INIT,
    )

    fit_results: Dict[int, ModelFitResult] = {}

    for n in range(MIN_REGIMES, MAX_REGIMES + 1):
        result = _fit_single(observations, n, covariance_type, n_iter, random_state)
        fit_results[n] = result

        status = f"BIC={result.bic:.1f}" if result.converged else "FAILED"
        logger.info("  n_states=%d  LL=%+.1f  k=%d  %s",
                    n, result.log_likelihood, result.n_params, status)

    # ── Select lowest BIC among converged models ──────────────────────────────
    valid = {n: r for n, r in fit_results.items() if r.converged}
    if not valid:
        raise RuntimeError(
            "All HMM fits failed to converge.  "
            "Check that your feature data has no NaNs or constant columns."
        )

    best_n = min(valid, key=lambda n: valid[n].bic)
    logger.info(
        "Selected n_states=%d  (BIC=%.1f)",
        best_n,
        fit_results[best_n].bic,
    )

    return SelectionResult(
        best_n=best_n,
        best_model=fit_results[best_n].model,
        bic_scores={n: r.bic for n, r in fit_results.items()},
        log_likelihoods={n: r.log_likelihood for n, r in fit_results.items()},
        fit_results=fit_results,
    )
