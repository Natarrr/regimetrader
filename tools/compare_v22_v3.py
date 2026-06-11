# Path: tools/compare_v22_v3.py
"""Shadow-window comparator: v2.2 vs v3.0 (migration step 5 acceptance gates).

Reads pipeline result snapshots where rows carry BOTH `final_score` (v2.2)
and the shadow `final_score_v3` columns, and reports:

  - Spearman rank correlation v2.2 ↔ v3 and top-20 overlap
  - weight_coverage_v3 distribution per region (gates: >=0.60 US /
    0.45 EU / 0.40 APAC)
  - per-region × per-factor sparsity map (None/dead rates — watch APAC
    margin_expansion and where pro-rata reweighting concentrates)
  - day-over-day rank turnover vs the previous snapshot, for v3 AND for
    v2.2's own baseline (gate: v3 turnover inflation <= +15%)
  - per-bucket σ of final_score_v3 (spread-collapse detector)

Run:
    python tools/compare_v22_v3.py logs/intel_source_status.json
        [--prev logs/archive/2026-06-10_intel_source_status.json]
        [--json-out logs/v3_shadow_report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.factor_matrix import FACTOR_MATRIX_V3  # noqa: E402

_V3_FACTORS = sorted({name for m in FACTOR_MATRIX_V3.values() for name in m})


# ── Pure metrics ──────────────────────────────────────────────────────────────

def _ranks(values: List[float]) -> List[float]:
    """Average ranks (ties shared), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: List[float], b: List[float]) -> Optional[float]:
    """Spearman rank correlation; None when undefined (<2 pts or 0 variance)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _ranks(list(a)), _ranks(list(b))
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra)
                    * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else None


def top_n_overlap(rows: List[Dict], key_a: str, key_b: str, n: int = 20
                  ) -> Optional[float]:
    """Jaccard-free overlap: |topN(a) ∩ topN(b)| / n."""
    scored = [r for r in rows
              if r.get(key_a) is not None and r.get(key_b) is not None]
    if len(scored) < n:
        return None
    top_a = {r["ticker"] for r in sorted(
        scored, key=lambda r: r[key_a], reverse=True)[:n]}
    top_b = {r["ticker"] for r in sorted(
        scored, key=lambda r: r[key_b], reverse=True)[:n]}
    return len(top_a & top_b) / n


def sparsity_map(rows: List[Dict], factors: List[str]) -> Dict[str, Dict]:
    """Per-market None/dead rates on the raw v3 factor columns."""
    by_market: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_market[str(r.get("market", "USA"))].append(r)
    out: Dict[str, Dict] = {}
    for market, group in by_market.items():
        n = len(group)
        stats: Dict[str, Dict] = {}
        for factor in factors:
            # Collision factors (analyst_revision, congress, PT) carry BOTH
            # a v2.2 column (dead-coerced 0.0) and a _v3 column (None =
            # unavailable). Shadow gates must read the v3 semantics.
            v3_key, base_key = f"{factor}_score_v3", f"{factor}_score"
            values = [
                r[v3_key] if v3_key in r else r[base_key]
                for r in group
                if v3_key in r or base_key in r
            ]
            if not values:
                continue
            nones = sum(1 for v in values if v is None)
            deads = sum(1 for v in values
                        if v is not None and float(v) == 0.0)
            stats[factor] = {
                "none_rate": round(nones / n, 4),
                "dead_rate": round(deads / n, 4),
            }
        out[market] = stats
    return out


def rank_turnover(prev_rows: List[Dict], curr_rows: List[Dict],
                  score_key: str, top_n: int = 20) -> Dict[str, Any]:
    """Day-over-day rank autocorrelation + top-N churn for one score column."""
    prev = {r["ticker"]: r.get(score_key) for r in prev_rows
            if r.get(score_key) is not None}
    curr = {r["ticker"]: r.get(score_key) for r in curr_rows
            if r.get(score_key) is not None}
    common = sorted(set(prev) & set(curr))
    auto = spearman([prev[t] for t in common], [curr[t] for t in common])
    k = min(top_n, len(common))
    churn = None
    if k > 0:
        top_prev = set(sorted(common, key=lambda t: prev[t], reverse=True)[:k])
        top_curr = set(sorted(common, key=lambda t: curr[t], reverse=True)[:k])
        churn = 1.0 - len(top_prev & top_curr) / k
    return {
        "common_tickers": len(common),
        "rank_autocorrelation": auto,
        "top_n_churn": churn,
    }


def bucket_sigma(rows: List[Dict], score_key: str = "final_score_v3"
                 ) -> Dict[str, float]:
    """σ of the v3 score per (market, sector, cap_tier) — spread-collapse map."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.get(score_key) is None:
            continue
        key = "|".join([str(r.get("market", "USA")),
                        str(r.get("sector", "Unknown")),
                        str(r.get("cap_tier", "large"))])
        buckets[key].append(float(r[score_key]))
    return {k: round(statistics.pstdev(v), 6)
            for k, v in sorted(buckets.items()) if len(v) >= 2}


# ── Snapshot I/O + report ─────────────────────────────────────────────────────

def load_rows(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("results", "rankings", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"{path}: no results/rankings/rows list found")
    return data


def build_report(rows: List[Dict],
                 prev_rows: Optional[List[Dict]] = None) -> Dict[str, Any]:
    paired = [r for r in rows if r.get("final_score") is not None
              and r.get("final_score_v3") is not None]
    coverage = [r["weight_coverage_v3"] for r in rows
                if r.get("weight_coverage_v3") is not None]
    report: Dict[str, Any] = {
        "rows": len(rows),
        "rows_with_both_scores": len(paired),
        "spearman_v22_v3": spearman(
            [r["final_score"] for r in paired],
            [r["final_score_v3"] for r in paired]),
        "top20_overlap": top_n_overlap(paired, "final_score", "final_score_v3"),
        "coverage_v3": {
            "mean": round(statistics.mean(coverage), 4) if coverage else None,
            "min": min(coverage) if coverage else None,
        },
        "sparsity": sparsity_map(rows, _V3_FACTORS),
        "bucket_sigma_v3": bucket_sigma(rows),
        "blackouts": sorted({f for r in rows
                             for f in (r.get("_factor_blackout") or [])}),
    }
    if prev_rows is not None:
        v3 = rank_turnover(prev_rows, rows, "final_score_v3")
        v22 = rank_turnover(prev_rows, rows, "final_score")
        inflation = None
        if v3["top_n_churn"] is not None and v22["top_n_churn"]:
            inflation = round(
                (v3["top_n_churn"] - v22["top_n_churn"]) / v22["top_n_churn"], 4)
        report["turnover"] = {
            "v3": v3, "v22": v22,
            "churn_inflation_vs_v22": inflation,
            "gate_max_inflation": 0.15,
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--prev", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load_rows(args.snapshot)
    prev_rows = load_rows(args.prev) if args.prev else None
    report = build_report(rows, prev_rows)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
        print(f"\nwritten: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
