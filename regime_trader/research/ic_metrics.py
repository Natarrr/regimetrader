"""regime_trader.research.ic_metrics — Information Coefficient computation.

Implements:
  - Spearman rank IC with bootstrap 95% CI and p-value
  - De-overlapped IC time series: retains only snapshots whose forward-return
    windows are disjoint (spacing >= embargo_days), so the t-test and bootstrap
    see independent observations rather than serially-dependent overlapping ones.

Note: this is an embargo / de-overlapping procedure in the spirit of López de
Prado's purged CV, NOT train/test purging — IC is a per-cross-section statistic,
so there is no fitted model whose training labels need purging. The leakage we
remove is the serial dependence of overlapping label windows.

References:
  Grinold & Kahn (2000) Active Portfolio Management ch. 6 — IC and IR definitions
  López de Prado (2018) AFML ch. 7 — overlapping labels inflate effective sample size
"""
from __future__ import annotations

import logging
import math
from datetime import date
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


def _select_non_overlapping(dates: list[date], embargo_days: int) -> list[int]:
    """Return indices of a non-overlapping subset of `dates` (assumed sorted).

    López de Prado (2018) AFML ch. 7: when forward-return windows overlap, the
    per-snapshot IC observations are serially dependent and must not be treated
    as independent draws. Greedily keep the earliest snapshot, then skip every
    later one whose date falls within `embargo_days` of the last kept snapshot —
    i.e. enforce a minimum spacing equal to the forward-return horizon so each
    retained observation has a disjoint label window.

    embargo_days <= 0 disables de-overlapping (every index retained), preserving
    backward-compatible behaviour for callers that pass non-overlapping data.
    """
    if embargo_days <= 0 or not dates:
        return list(range(len(dates)))

    kept: list[int] = []
    last_kept: date | None = None
    for i, d in enumerate(dates):
        if last_kept is None or (d - last_kept).days >= embargo_days:
            kept.append(i)
            last_kept = d
    return kept


def purged_kfold_ic(
    snapshots: list[tuple[date, list[dict]]],
    factor_name: str,
    forward_return_map: dict[date, dict[str, float]],
    n_folds: int = 5,
    embargo_days: int = 5,
    schema_warning: str = "",
) -> ICResult:
    """Compute IC across non-overlapping cross-sections for factor_name.

    Overlapping forward-return windows make per-snapshot IC observations serially
    dependent; counting them all inflates the effective sample size and produces
    spuriously small p-values / narrow CIs. To prevent this leakage we first
    select a non-overlapping subset of snapshots spaced >= embargo_days apart
    (López de Prado 2018 AFML ch. 7), then compute the cross-sectional Spearman
    IC on each retained snapshot.

    Note: this is a de-overlapping (embargo) procedure, not train/test purging —
    IC is a per-cross-section statistic, so there is no fitted model to purge.
    Set embargo_days = horizon_days so retained windows are disjoint.

    Args:
        snapshots: Ordered list of (date, rows) from load_historical_snapshots().
        factor_name: Key in rows dict to use as factor score.
        forward_return_map: {date: {ticker: forward_return}} from fetch_forward_returns.
        n_folds: Retained for API compatibility / sufficiency check (>= n_folds
                 non-overlapping snapshots required for a stable estimate).
        embargo_days: Minimum spacing (days) between retained snapshots. Should
                      equal the forward-return horizon. 0 disables de-overlapping.
        schema_warning: Propagated to ICResult.schema_warning.

    Returns:
        ICResult with mean/std/IR/p-value/CI across non-overlapping cross-sections.
    """
    dates = [d for d, _ in snapshots]
    n = len(dates)

    if n < n_folds:
        raise ValueError(
            f"purged_kfold_ic: {n} snapshots < n_folds={n_folds}. "
            f"Cannot construct {n_folds} folds."
        )

    # De-overlap: keep only snapshots whose label windows are disjoint so the
    # downstream t-test / bootstrap see independent observations.
    keep_indices = _select_non_overlapping(dates, embargo_days)
    if embargo_days > 0 and len(keep_indices) < len(dates):
        logger.info(
            "purged_kfold_ic[%s]: de-overlapped %d→%d snapshots "
            "(embargo=%dd) to remove forward-window leakage",
            factor_name, len(dates), len(keep_indices), embargo_days,
        )

    ic_values: list[float] = []
    ticker_counts: list[int] = []

    for snap_idx in keep_indices:
        snap_date, rows = snapshots[snap_idx]

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
