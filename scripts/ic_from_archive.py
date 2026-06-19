"""scripts/ic_from_archive.py
Per-factor de-overlapped Information Coefficient from the live signal archive.

Bridges logs/archive/*top_lists.json → realized forward returns (via the backtest
price fetch) → Spearman rank-IC per factor (src.research.ic_metrics), applying the
López de Prado overlap embargo so the reported t-stat reflects INDEPENDENT
observations, not the inflated raw snapshot count.

This is the "IC de-overlappé" pre-launch checklist item (docs/BILAN_PRE_LANCEMENT).
Re-run as snapshots accumulate — early reads are NOISE: on a ~3-week archive the
effective breadth is ≈ 1–4, so no factor can be significant regardless of its
point IC. A trustworthy read needs months of snapshots.

Usage:
  python scripts/ic_from_archive.py --horizon 5
  python scripts/ic_from_archive.py --horizon 20 --archive-dir logs/archive
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from scripts.backtest_signals import (
    SignalRecord,
    _ARCHIVE_DIR,
    _CANONICAL_FACTORS,
    compute_returns,
    fetch_prices,
    load_all_signals,
)
from src.research.ic_metrics import compute_ic_report

log = logging.getLogger("ic_from_archive")

_RETURN_KEY = "forward_return"


def build_snapshots(records: List[SignalRecord], horizon: int) -> List[Dict[str, Any]]:
    """Group priced records by signal date into ic_metrics-shaped snapshots.

    Each row carries the record's factor scores plus the realized forward return
    at `horizon` trading days under _RETURN_KEY. Records without an entry price or
    without a forward return at `horizon` are dropped (no fabricated 0.0). Pure —
    no network — so it is unit-testable on synthetic records.
    """
    by_date: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        fwd = r.returns.get(horizon)
        if r.entry_price is None or fwd is None:
            continue
        by_date[r.signal_date].append({**r.factors, _RETURN_KEY: float(fwd)})
    return [
        {"date": d.isoformat(), "rows": rows}
        for d, rows in sorted(by_date.items())
    ]


def factor_ic_report(records: List[SignalRecord], horizon: int) -> Dict[str, Dict[str, Any]]:
    """Per-factor de-overlapped IC report over the archive at `horizon` days."""
    snapshots = build_snapshots(records, horizon)
    factors = [
        f for f in _CANONICAL_FACTORS
        if any(f in row for s in snapshots for row in s["rows"])
    ]
    return compute_ic_report(
        snapshots, factors, return_key=_RETURN_KEY, horizon_days=horizon)


def _print_report(report: Dict[str, Dict[str, Any]], snapshots_n: int, rows_n: int,
                  horizon: int) -> None:
    print(f"\nPer-factor de-overlapped IC — T+{horizon} horizon  "
          f"(snapshots={snapshots_n}, priced rows={rows_n})")
    print(f"{'factor':<22}{'meanIC':>8}{'IC_IR':>8}{'t-stat':>8}"
          f"{'n_eff':>6}{'nsnap':>6}{'pos%':>7}  rec")
    for f, s in sorted(report.items(), key=lambda kv: -(kv[1]["mean_ic"] or 0)):
        print(f"{f:<22}{s['mean_ic']:>8.3f}{s['ic_ir']:>8.2f}{s['ic_t_stat']:>8.2f}"
              f"{s['n_effective']:>6}{s['n_snapshots']:>6}"
              f"{s['ic_positive_rate'] * 100:>6.0f}%  {s['weight_recommendation']}")
    max_neff = max((s["n_effective"] for s in report.values()), default=0)
    if max_neff < 8:
        print(f"\n⚠  max n_effective = {max_neff} (< 8) — NOT statistically reliable. "
              f"Every IC above is noise; accumulate more snapshots before trusting it.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward horizon in trading days (5/10/20). Default 5.")
    parser.add_argument("--archive-dir", type=Path, default=_ARCHIVE_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s", stream=sys.stdout)

    records = load_all_signals(args.archive_dir)
    if not records:
        print(f"No signals in {args.archive_dir}")
        return 0
    prices = fetch_prices(
        list({r.ticker for r in records}),
        min(r.signal_date for r in records),
        max(r.signal_date for r in records),
        dry_run=False,
    )
    compute_returns(records, prices)
    snapshots = build_snapshots(records, args.horizon)
    rows_n = sum(len(s["rows"]) for s in snapshots)
    report = factor_ic_report(records, args.horizon)
    _print_report(report, len(snapshots), rows_n, args.horizon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
