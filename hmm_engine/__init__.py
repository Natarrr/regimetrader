"""
hmm_engine
──────────
Hidden Markov Model volatility classifier for regime_trader.

Architecture
────────────
                          ┌─────────────────────┐
                          │   RegimeClassifier   │  ← primary public API
                          └──────────┬──────────┘
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
            model_selector    regime_labeler    StabilityFilter
          (BIC, 3–7 states)  (mean-ret rank)  (persist + flicker)
                    │
                    └──── forward_algorithm  (α-recursion, no look-ahead)

Quick-start
───────────
    from hmm_engine import RegimeClassifier

    clf = RegimeClassifier()
    clf.fit(features, returns)           # ≥ 504 bars (2 trading years)
    state = clf.predict_current(window)  # live bar — causal, no look-ahead
    df    = clf.predict_sequence(hist)   # backtest — full history, no bias

Regime labels (ascending mean return, scaled to n_states)
──────────────────────────────────────────────────────────
    3 states → Bear · Neutral · Bull
    4 states → Crash · Bear · Bull · Euphoria
    5 states → Crash · Bear · Neutral · Bull · Euphoria
    6 states → Crash · Bear · Neutral · Bull · Euphoria · Mania
    7 states → Crash · Panic · Bear · Neutral · Bull · Euphoria · Mania
"""

from .classifier import RegimeClassifier, RegimeState
from .stability_filter import StabilityFilter, FilterResult
from .regime_labeler import label_regimes, REGIME_COLORS, LABEL_SETS
from .forward_algorithm import forward_filter, decode_sequence_causal
from .model_selector import select_best_model, SelectionResult, ModelFitResult

__all__ = [
    # Primary interface
    "RegimeClassifier",
    "RegimeState",
    # Stability filter
    "StabilityFilter",
    "FilterResult",
    # Labelling
    "label_regimes",
    "REGIME_COLORS",
    "LABEL_SETS",
    # Forward algorithm (low-level)
    "forward_filter",
    "decode_sequence_causal",
    # Model selection (low-level)
    "select_best_model",
    "SelectionResult",
    "ModelFitResult",
]
