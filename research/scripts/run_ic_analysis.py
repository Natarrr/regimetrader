# Path: research/scripts/run_ic_analysis.py
"""Run IC analysis on backfill data → research/ic_report.json.

Run from repo root after build_qlib_dataset.py completes:
    python research/scripts/run_ic_analysis.py

Output: research/ic_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from research.scripts.build_qlib_dataset import load_ndjson
from research.scripts.ic_engine import build_ic_report, ACADEMIC_WEIGHTS_US, FACTORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_ic_analysis")

_IN = Path("research/data/backfill/factor_scores.ndjson")
_OUT = Path("research/ic_report.json")


def main() -> None:
    if not _IN.exists():
        raise FileNotFoundError(f"{_IN} not found — run backfill_factors.py first")

    log.info("Loading factor scores from %s...", _IN)
    df = load_ndjson(_IN)
    records = df.to_dict("records")
    log.info("Loaded %d records across %d tickers", len(records), df["ticker"].nunique())

    log.info("Computing IC for %d factors...", len(FACTORS))
    report = build_ic_report(records)

    # Annotate with academic weight for comparison
    for factor, metrics in report.items():
        metrics["academic_weight"] = ACADEMIC_WEIGHTS_US.get(factor, 0.0)

    _OUT.write_text(json.dumps(report, indent=2))
    log.info("IC report written to %s", _OUT)

    # Print summary table
    print("\n── IC Report Summary ──────────────────────────────────────")
    print(f"{'Factor':<22} {'Mean IC':>8} {'IC IR':>7} {'IC>0':>6} {'Acad.W':>7} {'Rec':>12}")
    print("-" * 68)
    for factor, m in report.items():
        print(
            f"{factor:<22} {m['mean_ic']:>8.4f} {m['ic_ir']:>7.3f} "
            f"{m['ic_positive_rate']:>6.2%} {m['academic_weight']:>7.2f} "
            f"{m['weight_recommendation']:>12}"
        )
    print("─" * 68)


if __name__ == "__main__":
    main()
