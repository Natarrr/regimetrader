# Path: monitoring/region_metrics.py
"""Per-region v3.0 monitoring metrics → logs/scoring_v3_metrics.json.

Backtest-target formulation (plan Deliverable 3):
  - IC = Spearman(final_score_v3, T+20 forward excess return vs the
    LOCAL-CURRENCY benchmark) — FX-neutral ranking by construction.
    Targets: US >= 0.03, EU >= 0.025, APAC >= 0.02.
  - Quintile (not decile — universes ~100–160 names) top-minus-bottom
    spread; positive in >= 60% of weekly snapshots.
  - Top-20 currency mix (suffix-derived) — FX-exposure visibility.
  - Coverage gates: mean weight_coverage_v3 >= 0.60 US / 0.45 EU / 0.40 APAC.

Forward returns arrive ex-post (T+20); pre-cutover snapshots therefore have
ic_spearman=None, which evaluate_region() treats as "not yet measurable",
never as a failure. evaluate.py integrates evaluate_region() at cutover
(migration step 6) alongside its existing coverage/error gates.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_IC_TARGET = {"USA": 0.03, "US": 0.03,
              "EUROPE": 0.025, "EU": 0.025,
              "ASIA": 0.02}
_COVERAGE_GATE = {"USA": 0.60, "US": 0.60,
                  "EUROPE": 0.45, "EU": 0.45,
                  "ASIA": 0.40}

_SUFFIX_CCY = {
    ".L": "GBp", ".PA": "EUR", ".DE": "EUR", ".AS": "EUR", ".MI": "EUR",
    ".MC": "EUR", ".BR": "EUR", ".F": "EUR", ".BE": "EUR", ".LS": "EUR",
    ".HE": "EUR", ".VX": "CHF", ".OL": "NOK", ".ST": "SEK", ".CO": "DKK",
    ".T": "JPY", ".HK": "HKD", ".KS": "KRW", ".KQ": "KRW",
    ".SS": "CNY", ".SZ": "CNY", ".NS": "INR", ".BO": "INR",
    ".SI": "SGD", ".BK": "THB", ".JK": "IDR",
}


def currency_of(ticker: str) -> str:
    upper = (ticker or "").upper()
    dot = upper.rfind(".")
    if dot == -1:
        return "USD"
    return _SUFFIX_CCY.get(upper[dot:], "USD")


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    from tools.compare_v22_v3 import spearman  # single implementation (SSOT)
    return spearman(a, b)


def compute_region_metrics(
    rows: List[Dict[str, Any]],
    forward_returns: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-market metrics for rows carrying final_score_v3.

    forward_returns: optional {ticker: T+20 forward excess return vs the
    region's local-currency benchmark}; absent → IC/spread are None/level.
    """
    by_market: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if r.get("final_score_v3") is None:
            continue
        by_market[str(r.get("market", "USA"))].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for market, group in by_market.items():
        scores = [float(r["final_score_v3"]) for r in group]
        coverage = [float(r.get("weight_coverage_v3") or 0.0) for r in group]
        ranked = sorted(group, key=lambda r: r["final_score_v3"], reverse=True)
        top20 = ranked[:20]

        ic = None
        spread = None
        if forward_returns:
            paired = [(float(r["final_score_v3"]), forward_returns[r["ticker"]])
                      for r in group if r["ticker"] in forward_returns]
            if len(paired) >= 5:
                ic = _spearman([p[0] for p in paired], [p[1] for p in paired])
                # Quintile top-minus-bottom on forward returns
                paired.sort(key=lambda p: p[0], reverse=True)
                q = max(1, len(paired) // 5)
                top_ret = statistics.mean(p[1] for p in paired[:q])
                bot_ret = statistics.mean(p[1] for p in paired[-q:])
                spread = round(top_ret - bot_ret, 6)

        out[market] = {
            "n": len(group),
            "score_mean": round(statistics.mean(scores), 6),
            "score_sigma": round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0,
            "coverage_mean": round(statistics.mean(coverage), 4) if coverage else None,
            "ic_spearman": round(ic, 6) if ic is not None else None,
            "quintile_spread": spread,
            "top20_currency_mix": dict(Counter(
                currency_of(r["ticker"]) for r in top20)),
            "low_coverage_count": sum(
                1 for r in group if r.get("_low_coverage_v3")),
        }
    return out


def evaluate_region(market: str, metrics: Dict[str, Any]) -> List[str]:
    """Gate check for one region's metrics. Returns failure strings ([]=pass).

    Missing IC (pre-cutover, no T+20 returns yet) is NOT a failure —
    absence of evidence is not evidence of degradation.
    """
    failures: List[str] = []
    ic = metrics.get("ic_spearman")
    target = _IC_TARGET.get(market)
    if ic is not None and target is not None and ic < target:
        failures.append(
            f"{market}: IC {ic:.4f} below target {target:.3f}")
    coverage = metrics.get("coverage_mean")
    gate = _COVERAGE_GATE.get(market)
    if coverage is not None and gate is not None and coverage < gate:
        failures.append(
            f"{market}: mean coverage {coverage:.2f} below gate {gate:.2f}")
    return failures


def write_metrics(
    rows: List[Dict[str, Any]],
    out_path: Path,
    forward_returns: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute, gate, and persist logs/scoring_v3_metrics.json."""
    metrics = compute_region_metrics(rows, forward_returns=forward_returns)
    failures = [f for market, m in metrics.items()
                for f in evaluate_region(market, m)]
    payload = {"regions": metrics, "gate_failures": failures}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if failures:
        log.error("region metrics gate failures: %s", failures)
    else:
        log.info("region metrics: all gates pass (%d regions)", len(metrics))
    return payload
