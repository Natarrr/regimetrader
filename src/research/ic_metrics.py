"""src.research.ic_metrics — Information Coefficient time-series aggregation.

Per-snapshot rank-IC reuses the single spearman implementation in
``tools.compare_v22_v3`` (the same SSOT monitoring/region_metrics.py consumes).
What this module adds is the *cross-snapshot* aggregation used to judge whether
a factor earns its weight:

    Rank IC      Spearman(factor_score, forward_return) within one snapshot
    Mean IC      average rank-IC across snapshots
    IC IR        mean_ic / std_ic                    (Grinold & Kahn 2000)
    IC t-stat    IC IR * sqrt(n_effective)           ← embargo-corrected
    Pos rate     fraction of snapshots with IC > 0

**Overlap embargo (López de Prado 2018, ch. 7).** Snapshots sampled more
frequently than the forward-return horizon overlap, so the raw snapshot count
overstates the number of *independent* observations and inflates the IR
t-statistic. ``effective_breadth`` collapses the snapshot dates to a maximal
non-overlapping subset (gaps >= horizon) and the reported ``ic_t_stat`` uses
that de-overlapped breadth — never the raw count.

This is a research/advisory tool. It never mutates WEIGHTS; weight changes are
human decisions (see src/research/__init__.py).
"""
from __future__ import annotations

import statistics
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.compare_v22_v3 import spearman  # SSOT rank correlation

_RETURN_KEY = "forward_return_21d"


def snapshot_ic(
    snapshot: Dict[str, Any],
    factor: str,
    return_key: str = _RETURN_KEY,
) -> Optional[float]:
    """Rank-IC of one factor against the forward return within one snapshot.

    Pairs are dropped when either the factor score or the forward return is
    missing/None. Returns None when fewer than 2 usable pairs remain (spearman
    is undefined), so a thin snapshot never poisons the series with a spurious 0.
    """
    scores: List[float] = []
    returns: List[float] = []
    for row in snapshot.get("rows", []):
        s, r = row.get(factor), row.get(return_key)
        if s is None or r is None:
            continue
        scores.append(float(s))
        returns.append(float(r))
    return spearman(scores, returns)


def _factor_ic_pairs(
    snapshots: Sequence[Dict[str, Any]],
    factor: str,
    return_key: str = _RETURN_KEY,
) -> List[Tuple[date, float]]:
    """(snapshot_date, ic) for every snapshot whose IC is defined."""
    pairs: List[Tuple[date, float]] = []
    for snap in snapshots:
        ic = snapshot_ic(snap, factor, return_key)
        if ic is None:
            continue
        pairs.append((_as_date(snap.get("date")), ic))
    return pairs


def factor_ic_series(
    snapshots: Sequence[Dict[str, Any]],
    factor: str,
    return_key: str = _RETURN_KEY,
) -> List[float]:
    """Per-snapshot rank-IC list for a factor (defined snapshots only)."""
    return [ic for _, ic in _factor_ic_pairs(snapshots, factor, return_key)]


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def effective_breadth(dates: Sequence[date], horizon_days: int = 21) -> int:
    """Count of non-overlapping forward-return windows among the snapshot dates.

    Greedy maximal subset whose consecutive gaps span >= ``horizon_days``
    *trading* sessions. The forward-return label covers ``horizon_days`` TRADING
    days, so the embargo is measured in business days (``np.busday_count``), not
    calendar days: a calendar-day gap would let a 21-trading-day (~30 calendar)
    window masquerade as independent after only ~15 sessions, inflating the IR
    t-statistic (López de Prado 2018, ch. 7). Business days approximate trading
    days up to exchange holidays (~9/yr), a far smaller error than the ~30%
    calendar-vs-trading mismatch it replaces.
    """
    ordered = sorted(_as_date(d) for d in dates)
    if not ordered:
        return 0
    kept = 1
    anchor = ordered[0]
    for d in ordered[1:]:
        if np.busday_count(np.datetime64(anchor), np.datetime64(d)) >= horizon_days:
            kept += 1
            anchor = d
    return kept


def aggregate_ic(
    ic_series: Sequence[float],
    dates: Sequence[date],
    horizon_days: int = 21,
) -> Dict[str, Any]:
    """Aggregate an IC series into mean/IR/significance with embargo correction.

    ``ic_series`` and ``dates`` must be aligned (one date per IC). ``ic_t_stat``
    uses the de-overlapped ``n_effective`` so significance is not inflated by
    overlapping snapshots.
    """
    n = len(ic_series)
    if n == 0:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
                "ic_positive_rate": 0.0, "n_snapshots": 0,
                "n_effective": 0, "ic_t_stat": 0.0}

    mean_ic = statistics.mean(ic_series)
    ic_std = statistics.stdev(ic_series) if n > 1 else 0.0
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
    pos_rate = sum(1 for ic in ic_series if ic > 0) / n
    n_eff = effective_breadth(dates, horizon_days)
    ic_t_stat = ic_ir * (n_eff ** 0.5)

    return {
        "mean_ic": round(mean_ic, 6),
        "ic_std": round(ic_std, 6),
        "ic_ir": round(ic_ir, 6),
        "ic_positive_rate": round(pos_rate, 4),
        "n_snapshots": n,
        "n_effective": n_eff,
        "ic_t_stat": round(ic_t_stat, 6),
    }


def weight_recommendation(stats: Dict[str, Any]) -> str:
    """Map aggregated IC stats to one of four advisory actions (spec §Phase 2).

    A negative mean IC means the signal may be inverting — investigate before
    trusting any reweight. Otherwise reward consistency (IR) and hit-rate.
    """
    mean_ic = stats.get("mean_ic", 0.0)
    ic_ir = stats.get("ic_ir", 0.0)
    pos_rate = stats.get("ic_positive_rate", 0.0)
    if mean_ic < 0:
        return "investigate"
    if ic_ir > 0.5 and pos_rate >= 0.60:
        return "increase"
    if ic_ir >= 0.3:
        return "hold"
    return "decrease"


def compute_ic_report(
    snapshots: Sequence[Dict[str, Any]],
    factors: Sequence[str],
    return_key: str = _RETURN_KEY,
    horizon_days: int = 21,
) -> Dict[str, Dict[str, Any]]:
    """Full per-factor IC report across the snapshot series."""
    report: Dict[str, Dict[str, Any]] = {}
    for factor in factors:
        pairs = _factor_ic_pairs(snapshots, factor, return_key)
        dates = [d for d, _ in pairs]
        ics = [ic for _, ic in pairs]
        agg = aggregate_ic(ics, dates, horizon_days)
        agg["weight_recommendation"] = (
            "investigate" if agg["n_snapshots"] == 0
            else weight_recommendation(agg))
        report[factor] = agg
    return report
