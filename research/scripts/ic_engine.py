# Path: research/scripts/ic_engine.py
"""IC engine — turn backfilled factor history into research/ic_report.json.

Pipeline:
    backfill NDJSON  (one record per ticker × snapshot_date, with raw + SPY
                      forward returns)
      → records_to_snapshots()   group by date, derive SPY-relative excess label
      → ic_metrics.compute_ic_report()   per-factor rank-IC, IR, embargo-corrected
      → research/ic_report.json   advisory report consumed by portfolio_optimizer

This is a research tool (local-only, CLAUDE.md §2). It never edits WEIGHTS;
``weight_recommendation`` is advisory only.

Run:
    python -m research.scripts.ic_engine                 # v2.2 factors → ic_report.json
    python -m research.scripts.ic_engine --engine v3     # v3 factors  → ic_report_v3.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.research.ic_metrics import compute_ic_report

_DEFAULT_BACKFILL = Path("research/data/backfill/factor_scores.ndjson")
_DEFAULT_OUT = Path("research/ic_report.json")
_HORIZON_DAYS = 21
_EXCESS_KEY = "excess_return_21d"

# Weight-0 candidate factors that are point-in-time reconstructable in the
# backfill (price-derived). OFF by default so the optimizer's report input is
# unchanged; enable with --candidates to inspect their de-overlapped IC.
_CANDIDATE_TECHNICAL_FACTORS = ("rsi_reversion", "adx_trend")


def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    """Read one JSON object per non-blank line."""
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def records_to_snapshots(
    records: Sequence[Dict[str, Any]],
    return_key: str = _EXCESS_KEY,
    date_key: str = "snapshot_date",
    fwd_key: str = "forward_return_21d",
    spy_key: str = "spy_return_21d",
) -> List[Dict[str, Any]]:
    """Group flat (ticker, date) records into date-ordered snapshots.

    The IC label is the SPY-relative excess forward return
    (forward_return - spy_return), consistent with how momentum_long is
    constructed (SPY-relative); a row missing either leg gets None and is
    dropped pairwise by snapshot_ic. The raw factor columns are preserved.
    """
    by_date: Dict[date, List[Dict[str, Any]]] = {}
    for rec in records:
        d = date.fromisoformat(str(rec[date_key]))
        row = dict(rec)
        fwd, spy = rec.get(fwd_key), rec.get(spy_key)
        row[return_key] = (None if fwd is None or spy is None
                           else float(fwd) - float(spy))
        by_date.setdefault(d, []).append(row)
    return [{"date": d, "rows": by_date[d]} for d in sorted(by_date)]


def run(
    backfill_path: Path,
    out_path: Path,
    factors: Sequence[str],
    return_key: str = _EXCESS_KEY,
    horizon_days: int = _HORIZON_DAYS,
) -> Dict[str, Any]:
    """Compute the IC report from a backfill file and persist it as JSON."""
    records = load_ndjson(backfill_path)
    snapshots = records_to_snapshots(records, return_key=return_key)
    report = compute_ic_report(
        snapshots, factors, return_key=return_key, horizon_days=horizon_days)

    # Schema: factor dicts at the TOP LEVEL (spec §Phase 2) so the existing
    # consumer portfolio_optimizer._ic_estimate() — which iterates values() for
    # dicts carrying "mean_ic" — reads them directly. Metadata lives under _meta
    # (a dict without "mean_ic", so the consumer skips it).
    payload: Dict[str, Any] = dict(report)
    payload["_meta"] = {
        "generated_at": date.today().isoformat(),
        "horizon_days": horizon_days,
        "return_label": return_key,
        "n_snapshots": len(snapshots),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _default_factors(engine: str) -> List[str]:
    if engine == "v3":
        from src.config.factor_matrix import FACTOR_MATRIX_V3
        names: List[str] = []
        for region in FACTOR_MATRIX_V3.values():
            for pillar in region.get("pillars", region).values():
                factors = pillar.get("factors", pillar) if isinstance(pillar, dict) else pillar
                names.extend(factors)
        return sorted(set(names))
    from src.config.weights import WEIGHTS_US
    return list(WEIGHTS_US.keys())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backfill", type=Path, default=_DEFAULT_BACKFILL)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: ic_report.json (v2.2) / ic_report_v3.json (v3)")
    parser.add_argument("--engine", choices=["v2.2", "v3"], default="v2.2")
    parser.add_argument("--horizon", type=int, default=_HORIZON_DAYS)
    parser.add_argument("--candidates", action="store_true",
                        help="also measure weight-0 technical candidates "
                             "(rsi_reversion, adx_trend); does not change WEIGHTS")
    args = parser.parse_args(argv)

    if not args.backfill.exists():
        parser.error(
            f"backfill not found: {args.backfill}\n"
            "Run: python -m research.scripts.backfill_factors")

    out = args.out or (Path("research/ic_report_v3.json")
                       if args.engine == "v3" else _DEFAULT_OUT)
    factors = _default_factors(args.engine)
    if args.candidates:
        factors = factors + [f for f in _CANDIDATE_TECHNICAL_FACTORS
                             if f not in factors]
    payload = run(args.backfill, out, factors, horizon_days=args.horizon)

    print(f"IC report ({args.engine}) -> {out}  "
          f"[{payload['_meta']['n_snapshots']} snapshots, {len(factors)} factors]")
    factor_rows = {k: v for k, v in payload.items() if k != "_meta"}
    for name, st in sorted(factor_rows.items(),
                           key=lambda kv: kv[1]["mean_ic"], reverse=True):
        print(f"  {name:24s} meanIC={st['mean_ic']:+.4f}  IR={st['ic_ir']:+.3f}"
              f"  t={st['ic_t_stat']:+.2f}  n_eff={st['n_effective']:3d}"
              f"  -> {st['weight_recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
