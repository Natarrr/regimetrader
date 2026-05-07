"""
regime_labeler.py
─────────────────
Labels HMM hidden states with economically meaningful regime names by
ranking states from lowest to highest mean return.

LABELLING LOGIC
───────────────
After the HMM is fitted, each state j has a mean vector μ_j.  The first
element (index 0 by convention) is the log-return feature.  States are
ranked by μ_j[returns_feature_index] in ascending order and assigned names
from the label set appropriate for the chosen number of states.

    rank 0  →  most negative average return  →  leftmost label (e.g. "Crash")
    rank S-1 →  most positive average return  →  rightmost label (e.g. "Euphoria")

This approach is model-agnostic: the integer state indices produced by
hmmlearn have no inherent economic meaning, but after label_regimes() they
map to a stable, human-readable vocabulary.

LABEL SETS  (3 – 7 states)
──────────────────────────
  States   Labels (ascending mean return)
  ───────  ───────────────────────────────────────────────────
    3      Bear · Neutral · Bull
    4      Crash · Bear · Bull · Euphoria
    5      Crash · Bear · Neutral · Bull · Euphoria
    6      Crash · Bear · Neutral · Bull · Euphoria · Mania
    7      Crash · Panic · Bear · Neutral · Bull · Euphoria · Mania
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from hmmlearn import hmm


# ── Label vocabulary ──────────────────────────────────────────────────────────

# Ordered from most bearish/low-return to most bullish/high-return.
# The subset used depends on n_states (see _LABEL_SETS).
LABEL_SETS: Dict[int, List[str]] = {
    3: ["Bear",  "Neutral", "Bull"],
    4: ["Crash", "Bear",    "Bull",     "Euphoria"],
    5: ["Crash", "Bear",    "Neutral",  "Bull",     "Euphoria"],
    6: ["Crash", "Bear",    "Neutral",  "Bull",     "Euphoria", "Mania"],
    7: ["Crash", "Panic",   "Bear",     "Neutral",  "Bull",     "Euphoria", "Mania"],
}

# Dashboard colour palette — one colour per label name.
REGIME_COLORS: Dict[str, str] = {
    "Crash":    "#67000D",  # deep crimson
    "Panic":    "#CB181D",  # vivid red
    "Bear":     "#EF3B2C",  # red-orange
    "Neutral":  "#969696",  # mid-grey
    "Bull":     "#74C476",  # medium green
    "Euphoria": "#238B45",  # forest green
    "Mania":    "#00441B",  # dark green
}

# Volatility characteristic per label (informational, used by risk_manager).
REGIME_VOLATILITY: Dict[str, str] = {
    "Crash":    "extreme",
    "Panic":    "very_high",
    "Bear":     "high",
    "Neutral":  "medium",
    "Bull":     "medium",
    "Euphoria": "low",
    "Mania":    "low",
}


# ── Public API ────────────────────────────────────────────────────────────────


def label_regimes(
    model: hmm.GaussianHMM,
    returns_feature_index: int = 0,
) -> Dict[int, str]:
    """
    Map HMM state indices to regime label strings.

    Parameters
    ----------
    model                 : A fitted hmmlearn GaussianHMM instance.
    returns_feature_index : Column in model.means_ that represents log-returns.
                            Defaults to 0 (first feature).

    Returns
    -------
    label_map : dict  {state_index (int) → label (str)}

    Example
    -------
    >>> label_map = label_regimes(fitted_model)
    >>> label_map
    {2: 'Crash', 0: 'Bear', 4: 'Neutral', 1: 'Bull', 3: 'Euphoria'}
    """
    n_states = model.n_components
    labels = LABEL_SETS.get(n_states)

    if labels is None:
        supported = sorted(LABEL_SETS.keys())
        raise ValueError(
            f"label_regimes: no label set for n_states={n_states}.  "
            f"Supported values: {supported}."
        )

    # Rank states by ascending mean return (lowest → rank 0 → leftmost label)
    mean_returns: np.ndarray = model.means_[:, returns_feature_index]
    sorted_states: np.ndarray = np.argsort(mean_returns)  # (n_states,)

    # sorted_states[rank] = state_index with that rank
    # → state_index : label at that rank
    return {int(state): labels[rank] for rank, state in enumerate(sorted_states)}


def regime_summary(
    model: hmm.GaussianHMM,
    label_map: Dict[int, str],
    feature_names: List[str] | None = None,
) -> str:
    """
    Return a human-readable table of per-regime statistics.

    Parameters
    ----------
    model        : fitted GaussianHMM
    label_map    : output of label_regimes()
    feature_names: optional list of feature column names for the header

    Returns
    -------
    Multi-line string suitable for logging or printing.
    """
    n_states = model.n_components
    n_features = model.means_.shape[1]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    lines = [
        f"{'State':>6}  {'Label':>10}  {'MeanRet':>9}  "
        + "  ".join(f"{fn:>10}" for fn in feature_names)
        + "  {'Volatility':>12}"
    ]

    for state_idx in range(n_states):
        label = label_map.get(state_idx, str(state_idx))
        means = model.means_[state_idx]
        mean_ret = means[0]
        lines.append(
            f"{state_idx:>6}  {label:>10}  {mean_ret:>+9.5f}  "
            + "  ".join(f"{m:>10.5f}" for m in means)
            + f"  {REGIME_VOLATILITY.get(label, 'unknown'):>12}"
        )

    return "\n".join(lines)
