"""regime_trader.research.ic_metrics — Information Coefficient computation.

Implements:
  - Spearman rank IC with bootstrap 95% CI and p-value
  - Purged k-fold IC per López de Prado (2018) AFML ch. 7
    (embargo_days prevents leakage from overlapping forward-return windows)

References:
  Grinold & Kahn (2000) Active Portfolio Management ch. 6 — IC and IR definitions
  López de Prado (2018) AFML ch. 7 — purged k-fold CV for financial data
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import NamedTuple, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_BOOTSTRAP_SAMPLES = 1_000
_BOOTSTRAP_RNG_SEED = 42


class ICResult(NamedTuple):
    factor_name: str
    ic_mean: float           # mean Spearman IC across cross-sections
    ic_std: float            # std dev of IC time-series
    ir: float                # IC / IC_std (information ratio)
    p_value: float           # from t-test: H0 = IC mean == 0
    ci_lower: float          # bootstrap 95% CI lower bound
    ci_upper: float          # bootstrap 95% CI upper bound
    n_snapshots: int         # number of cross-sections used
    n_tickers_avg: float     # average tickers per cross-section
    schema_warning: str      # non-empty if field had schema migration caveats


def compute_ic(
    factor_scores: Sequence[float],
    forward_returns: Sequence[float],
) -> tuple[float, float]:
    """Compute cross-sectional Spearman IC for one snapshot.

    Returns (ic, p_value). Returns (nan, nan) if fewer than 5 pairs.
    """
    xs = np.asarray(factor_scores, dtype=float)
    ys = np.asarray(forward_returns, dtype=float)

    # Drop any NaN pairs (dead signal = 0.0 is valid; None was excluded upstream)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]

    if len(xs) < 5:
        return float("nan"), float("nan")

    ic, p_val = stats.spearmanr(xs, ys)
    return float(ic), float(p_val)


def _bootstrap_ci(ic_series: np.ndarray, n_samples: int = _BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of ic_series."""
    rng = np.random.default_rng(_BOOTSTRAP_RNG_SEED)
    means = np.array([
        rng.choice(ic_series, size=len(ic_series), replace=True).mean()
        for _ in range(n_samples)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def purged_kfold_ic(
    snapshots: list[tuple[date, list[dict]]],
    factor_name: str,
    forward_return_map: dict[date, dict[str, float]],
    n_folds: int = 5,
    embargo_days: int = 5,
    schema_warning: str = "",
) -> ICResult:
    """Compute purged k-fold IC for factor_name across all snapshots.

    Purging: test-fold snapshots within embargo_days of a training fold boundary
    are excluded to prevent leakage from overlapping 21-day forward-return windows.

    Args:
        snapshots: Ordered list of (date, rows) from load_historical_snapshots().
        factor_name: Key in rows dict to use as factor score.
        forward_return_map: {date: {ticker: forward_return}} from fetch_forward_returns.
        n_folds: Number of purged k-folds.
        embargo_days: Days of embargo around fold boundaries.
        schema_warning: Propagated to ICResult.schema_warning.

    Returns:
        ICResult with mean/std/IR/p-value/CI across all qualifying cross-sections.
    """
    dates = [d for d, _ in snapshots]
    n = len(dates)

    if n < n_folds:
        raise ValueError(
            f"purged_kfold_ic: {n} snapshots < n_folds={n_folds}. "
            f"Cannot construct {n_folds} folds."
        )

    fold_size = n // n_folds
    ic_values: list[float] = []
    ticker_counts: list[int] = []

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_folds - 1 else n

        # Embargo: exclude snapshots within embargo_days of fold boundaries
        embargo_before = dates[test_start] - timedelta(days=embargo_days)
        embargo_after  = dates[test_end - 1] + timedelta(days=embargo_days)

        for snap_idx in range(test_start, test_end):
            snap_date, rows = snapshots[snap_idx]
            if snap_date < embargo_before or snap_date > embargo_after:
                continue  # outside embargo (shouldn't happen within test fold, but guard)

            fwd = forward_return_map.get(snap_date, {})
            if not fwd:
                logger.debug("purged_kfold_ic: no forward returns for %s — skip", snap_date)
                continue

            scores, returns = [], []
            for row in rows:
                ticker = row.get("ticker", "")
                score = row.get(factor_name)
                fwd_ret = fwd.get(ticker)
                if score is None or fwd_ret is None:
                    continue
                try:
                    scores.append(float(score))
                    returns.append(float(fwd_ret))
                except (TypeError, ValueError):
                    continue

            ic, _ = compute_ic(scores, returns)
            if math.isnan(ic):
                continue

            ic_values.append(ic)
            ticker_counts.append(len(scores))

    if not ic_values:
        return ICResult(
            factor_name=factor_name,
            ic_mean=float("nan"),
            ic_std=float("nan"),
            ir=float("nan"),
            p_value=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_snapshots=0,
            n_tickers_avg=0.0,
            schema_warning=schema_warning,
        )

    arr = np.array(ic_values)
    ic_mean = float(arr.mean())
    ic_std  = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    ir      = ic_mean / ic_std if (ic_std and not math.isnan(ic_std)) else float("nan")

    # Two-sided t-test: H0 = IC mean == 0
    if len(arr) > 1:
        t_stat = ic_mean / (ic_std / math.sqrt(len(arr)))
        p_value = float(2 * stats.t.sf(abs(t_stat), df=len(arr) - 1))
    else:
        p_value = float("nan")

    ci_lower, ci_upper = _bootstrap_ci(arr) if len(arr) >= 5 else (float("nan"), float("nan"))

    return ICResult(
        factor_name=factor_name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_snapshots=len(ic_values),
        n_tickers_avg=float(np.mean(ticker_counts)) if ticker_counts else 0.0,
        schema_warning=schema_warning,
    )
