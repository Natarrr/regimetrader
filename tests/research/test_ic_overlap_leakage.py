"""tests/research/test_ic_overlap_leakage.py

López de Prado (2018) AFML ch. 7 — overlapping forward-return windows make
per-snapshot IC observations serially dependent. Treating daily snapshots with
a 21-day forward horizon as independent inflates the effective sample size,
which deflates p-values and narrows confidence intervals (spurious significance).

The fix: purged_kfold_ic must not count more IC observations than there are
NON-OVERLAPPING forward-return windows over the snapshot span. With daily
snapshots and horizon_days=21, ~21 consecutive snapshots collapse to ONE
independent observation.

These tests pin that contract so the nominal-embargo regression cannot recur.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from regime_trader.research.ic_metrics import purged_kfold_ic


_RNG = np.random.default_rng(2026_05_30)
_N_TICKERS = 50
_TICKERS = [f"T{i:03d}" for i in range(_N_TICKERS)]


def _daily_overlapping_snapshots(
    n_days: int,
    alpha: float,
    noise_std: float = 0.15,
    factor_name: str = "test_factor_score",
):
    """Build n_days DAILY snapshots (spacing = 1 day << horizon → overlapping)."""
    base = date(2025, 1, 1)
    snapshots = []
    fwd_map = {}
    for i in range(n_days):
        d = base + timedelta(days=i)  # daily cadence → overlapping 21d windows
        scores = _RNG.uniform(0.0, 1.0, _N_TICKERS)
        rets = alpha * scores + _RNG.normal(0.0, noise_std, _N_TICKERS)
        snapshots.append((d, [
            {"ticker": _TICKERS[j], factor_name: float(scores[j])}
            for j in range(_N_TICKERS)
        ]))
        fwd_map[d] = {_TICKERS[j]: float(rets[j]) for j in range(_N_TICKERS)}
    return snapshots, fwd_map


def test_overlapping_snapshots_capped_at_independent_window_count():
    """With 63 daily snapshots and a 21-day horizon, at most ceil(63/21)=3
    independent IC observations may inform the t-test. The current code counts
    all 63 (one per snapshot), treating dependent observations as independent."""
    horizon = 21
    n_days = 63
    snapshots, fwd_map = _daily_overlapping_snapshots(n_days, alpha=0.0)

    result = purged_kfold_ic(
        snapshots=snapshots,
        factor_name="test_factor_score",
        forward_return_map=fwd_map,
        n_folds=3,
        embargo_days=horizon,
    )

    max_independent = math.ceil(n_days / horizon)  # = 3
    assert result.n_snapshots <= max_independent, (
        f"n_snapshots={result.n_snapshots} exceeds the {max_independent} "
        f"non-overlapping {horizon}-day windows — overlapping observations are "
        f"being double-counted (López de Prado AFML ch. 7 leakage)."
    )


def test_embargo_zero_keeps_all_snapshots():
    """Sanity guard: with embargo_days=0 the de-overlapping is disabled, so every
    snapshot is retained. Ensures the cap is driven by the embargo, not a bug."""
    snapshots, fwd_map = _daily_overlapping_snapshots(30, alpha=0.0)
    result = purged_kfold_ic(
        snapshots=snapshots,
        factor_name="test_factor_score",
        forward_return_map=fwd_map,
        n_folds=3,
        embargo_days=0,
    )
    assert result.n_snapshots == 30
