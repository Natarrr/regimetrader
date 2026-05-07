"""
stability_filter.py
───────────────────
Prevents the system from acting on noisy or short-lived regime signals
through two complementary mechanisms.

RULE 1 — PERSISTENCE GATE  (3-bar minimum)
───────────────────────────────────────────
A raw regime classification must be sustained for at least PERSISTENCE_BARS
(default 3) consecutive bars before it becomes the *confirmed* regime that
downstream modules (order_executor, risk_manager) act on.

    bar 0   raw=Bull  → candidate=Bull (streak=1) — not confirmed yet
    bar 1   raw=Bull  → candidate=Bull (streak=2) — not confirmed yet
    bar 2   raw=Bull  → candidate=Bull (streak=3) → CONFIRMED as Bull ✓
    bar 3   raw=Bear  → candidate=Bear (streak=1) — prior confirmed still active

This prevents single-bar "blips" in the HMM output from triggering trades.

RULE 2 — FLICKER DETECTOR  (>4 changes in 20 bars)
────────────────────────────────────────────────────
At every bar, count the number of raw-regime *transitions* in the rolling
FLICKER_WINDOW (default 20) bars.  If that count exceeds FLICKER_THRESHOLD
(default 4), the filter raises the `is_uncertain` flag and sets
`position_scale` to `uncertain_scale` (default 0.5 = half-size positions).

The log message includes all diagnostic values so that post-hoc review of
logs can reveal which periods were inherently ambiguous for the HMM.

STATE MACHINE
─────────────
    ┌──────────────────────────────────────────┐
    │  _candidate   : the currently "auditioned" regime                │
    │  _streak      : consecutive bars candidate has held              │
    │  _confirmed   : last confirmed regime (None until first confirm) │
    │  _raw_history : deque of last FLICKER_WINDOW raw regimes         │
    └──────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Tuneable constants (overridable via constructor) ──────────────────────────

PERSISTENCE_BARS: int   = 3    # consecutive bars before confirming a change
FLICKER_WINDOW: int     = 20   # rolling window for transition counting
FLICKER_THRESHOLD: int  = 4    # max allowed transitions inside that window
UNCERTAIN_SCALE: float  = 0.5  # position-size multiplier when uncertain


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class FilterResult:
    """
    Output of StabilityFilter.update() for a single bar.

    Attributes
    ----------
    raw_regime        : HMM state index BEFORE any filtering.
    raw_label         : Human-readable label for raw_regime.
    confirmed_regime  : State index after persistence gate; None if the
                        filter has not yet confirmed any regime.
    confirmed_label   : Label for confirmed_regime, or None.
    is_uncertain      : True when the flicker count exceeds the threshold.
    flicker_count     : Number of raw transitions in the last FLICKER_WINDOW bars.
    position_scale    : 1.0 under normal conditions; UNCERTAIN_SCALE when
                        is_uncertain is True.  Multiply target sizes by this.
    streak            : Current candidate-regime consecutive-bar count.
    """
    raw_regime:       int
    raw_label:        str
    confirmed_regime: Optional[int]
    confirmed_label:  Optional[str]
    is_uncertain:     bool
    flicker_count:    int
    position_scale:   float
    streak:           int


# ── Filter class ──────────────────────────────────────────────────────────────


class StabilityFilter:
    """
    Stateful, bar-by-bar stability filter for HMM regime signals.

    Designed to be used in two modes:

    Live trading
    ─────────────
        filt = StabilityFilter(label_map)
        …
        result = filt.update(raw_regime_int)
        if result.confirmed_regime is not None:
            trade(result.confirmed_regime, size * result.position_scale)

    Backtesting  (call reset() before each new run)
    ────────────
        filt.reset()
        for raw in raw_regime_array:
            result = filt.update(raw)
            record(result)
    """

    def __init__(
        self,
        label_map: Dict[int, str],
        persistence: int     = PERSISTENCE_BARS,
        flicker_window: int  = FLICKER_WINDOW,
        flicker_threshold: int = FLICKER_THRESHOLD,
        uncertain_scale: float = UNCERTAIN_SCALE,
    ) -> None:
        """
        Parameters
        ----------
        label_map         : {state_index → label string} from regime_labeler.
        persistence       : Bars required before confirming a regime (default 3).
        flicker_window    : Rolling window size for flicker detection (default 20).
        flicker_threshold : Max tolerated transitions per window (default 4).
        uncertain_scale   : Position multiplier under uncertainty (default 0.5).
        """
        self._label_map         = label_map
        self._persistence       = persistence
        self._flicker_window    = flicker_window
        self._flicker_threshold = flicker_threshold
        self._uncertain_scale   = uncertain_scale

        # Mutable state — reset via reset()
        self._raw_history: deque[int] = deque(maxlen=flicker_window)
        self._candidate:   Optional[int] = None
        self._streak:      int           = 0
        self._confirmed:   Optional[int] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, raw_regime: int) -> FilterResult:
        """
        Ingest one bar's raw regime and return the filtered result.

        Parameters
        ----------
        raw_regime : int — HMM state index for the current bar.

        Returns
        -------
        FilterResult describing the current filter state.

        Side-effects
        ------------
        - Updates _raw_history, _candidate, _streak, _confirmed in-place.
        - Emits a WARNING log when the flicker threshold is exceeded.
        """
        # 1. Record raw observation
        self._raw_history.append(raw_regime)

        # 2. Update persistence gate
        self._update_persistence(raw_regime)

        # 3. Flicker detection
        flicker_count = self._count_flickers()
        is_uncertain  = flicker_count > self._flicker_threshold

        if is_uncertain:
            logger.warning(
                "REGIME UNCERTAINTY | flicker_count=%d > threshold=%d "
                "in the last %d bars (streak=%d, candidate=%s, confirmed=%s). "
                "Position sizes scaled to %.0f%%.",
                flicker_count,
                self._flicker_threshold,
                self._flicker_window,
                self._streak,
                self._label_map.get(self._candidate, str(self._candidate)),
                self._label_map.get(self._confirmed, "none"),
                self._uncertain_scale * 100,
            )

        return FilterResult(
            raw_regime       = raw_regime,
            raw_label        = self._label_map.get(raw_regime, str(raw_regime)),
            confirmed_regime = self._confirmed,
            confirmed_label  = (
                self._label_map.get(self._confirmed)
                if self._confirmed is not None else None
            ),
            is_uncertain     = is_uncertain,
            flicker_count    = flicker_count,
            position_scale   = self._uncertain_scale if is_uncertain else 1.0,
            streak           = self._streak,
        )

    def reset(self) -> None:
        """
        Reset all internal state.  Call before each new backtest episode
        to prevent state leakage between independent simulation runs.
        """
        self._raw_history.clear()
        self._candidate = None
        self._streak    = 0
        self._confirmed = None

    # ── Properties (read-only snapshots) ─────────────────────────────────────

    @property
    def confirmed_regime(self) -> Optional[int]:
        return self._confirmed

    @property
    def confirmed_label(self) -> Optional[str]:
        return self._label_map.get(self._confirmed) if self._confirmed is not None else None

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def current_flicker_count(self) -> int:
        return self._count_flickers()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_persistence(self, raw_regime: int) -> None:
        """
        Advance the persistence gate state machine.

        A new regime must be the *only* observed raw regime for
        `_persistence` consecutive bars in a row.  Any interruption
        resets the streak counter.
        """
        if raw_regime == self._candidate:
            self._streak += 1
        else:
            # Candidate changed — restart streak
            self._candidate = raw_regime
            self._streak    = 1

        # Promote candidate to confirmed once persistence threshold is met
        if self._streak >= self._persistence:
            if self._confirmed != self._candidate:
                logger.info(
                    "Regime CONFIRMED: %s → %s  (after %d consecutive bars)",
                    self._label_map.get(self._confirmed, "none"),
                    self._label_map.get(self._candidate, str(self._candidate)),
                    self._streak,
                )
            self._confirmed = self._candidate

    def _count_flickers(self) -> int:
        """
        Count the number of raw-regime transitions in the rolling history.

        A transition is any bar where raw_history[t] != raw_history[t-1].
        Returns 0 if fewer than 2 bars have been observed.
        """
        hist = list(self._raw_history)
        return sum(1 for i in range(1, len(hist)) if hist[i] != hist[i - 1])
