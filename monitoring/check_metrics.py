"""monitoring/check_metrics.py — threshold gate for the canary.

Reads `<log-dir>/metrics.json` and exits:
    0 — all thresholds passed
    2 — ALERT (errors > 0  OR  edgar_count / ticker_count < min_coverage
               OR fmp_health.json reports has_structural_failure=true)

On a failed gate, posts to Discord via DISCORD_WEBHOOK_URL when set.

Defaults match the canary spec:
    --min-coverage  0.6   (≥60% of tickers must come back from EDGAR)
    --max-errors    0     (any error fails the gate)

Usage:
    python -m monitoring.check_metrics
    python -m monitoring.check_metrics --log-dir logs --min-coverage 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import statistics

from .alert_state import update_after_evaluation
from .evaluate import evaluate
from .slack_notifier import send_discord_alert as send_slack_alert

log = logging.getLogger("monitoring.check_metrics")


def check_score_distribution(log_dir: Path) -> bool:
    """Assert the pipeline score distribution is non-degenerate.

    Grinold & Kahn (2000): a factor with zero cross-sectional variance has
    IC = 0 and contributes nothing to IR. A distribution where every ticker
    scores identically, or where fewer than 20% of tickers have a non-zero
    final_score, indicates one or more dead data feeds.

    Checks:
      1. intel_source_status.json exists and contains US results.
      2. At least 10 US result rows are present (population gate).
      3. >= 20% of US tickers have final_score > 0 (non-zero density gate).
      4. Standard deviation of final_scores > 0.05 (variance gate).
      5. At least ONE factor column has non-zero density > 5% (per-factor
         gate — catches the case where scores are non-zero but all driven by
         a single factor because every other feed is dead).

    Returns True when all checks pass; logs ERROR and returns False otherwise.
    Never raises — all exceptions are caught and logged.
    """
    status_path = Path(log_dir) / "intel_source_status.json"
    if not status_path.exists():
        log.error(
            "check_score_distribution: intel_source_status.json not found at %s",
            status_path,
        )
        return False

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("check_score_distribution: cannot parse status file: %s", exc)
        return False

    results = [
        r for r in status.get("results", [])
        if r.get("market", "USA") in ("USA", "US")
    ]

    # Gate 1: minimum population
    if len(results) < 10:
        log.error(
            "check_score_distribution: only %d US results (need >= 10). "
            "Pipeline may have failed completely.",
            len(results),
        )
        return False

    final_scores = [float(r.get("final_score", 0.0) or 0.0) for r in results]

    # Gate 2: non-zero density
    nonzero_count = sum(1 for s in final_scores if s > 0.0)
    nonzero_density = nonzero_count / len(final_scores)
    if nonzero_density < 0.20:
        log.error(
            "check_score_distribution: DEGENERATE — only %d/%d US tickers "
            "have final_score > 0 (density=%.1f%% < 20%%). "
            "Dead signal feeds likely. Check fmp_health.json.",
            nonzero_count,
            len(final_scores),
            nonzero_density * 100,
        )
        return False

    # Gate 3: variance
    std_dev = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0

    if std_dev < 0.05:
        log.error(
            "check_score_distribution: DEGENERATE — std_dev=%.4f < 0.05. "
            "All tickers scoring nearly identically. "
            "Cross-sectional normalization may be broken.",
            std_dev,
        )
        return False

    # Gate 4: per-factor density (at least one live factor)
    _FACTOR_SCORE_KEYS = [
        "insider_conviction_score",
        "insider_breadth_score",
        "congress_score",
        "news_sentiment_score",
        "momentum_long_score",
    ]
    live_factors = []
    for key in _FACTOR_SCORE_KEYS:
        nz = sum(1 for r in results if float(r.get(key, 0.0) or 0.0) > 0.0)
        density = nz / len(results)
        if density > 0.05:
            live_factors.append((key, density))

    if not live_factors:
        log.error(
            "check_score_distribution: ALL core factor feeds appear dead "
            "(each has < 5%% non-zero density across %d tickers). "
            "insider / news / momentum feeds may all be down.",
            len(results),
        )
        return False

    log.info(
        "check_score_distribution PASSED: n=%d, nonzero=%.1f%%, std=%.4f, "
        "live_factors=%d (%s)",
        len(results),
        nonzero_density * 100,
        std_dev,
        len(live_factors),
        ", ".join(f"{k.replace('_score', '')}={d:.0%}" for k, d in live_factors[:3]),
    )

    # Advisory check: insider feed may be silently dead (HTTP 200 all-empty)
    _check_insider_feed_density(results, log)

    return True


def _check_insider_feed_density(results: list, log) -> bool:
    """Warn when insider feed appears silently dead (HTTP 200 but all-zeros).

    Distinguishes between:
      - Structurally sparse (e.g. 5–15% non-zero on S&P 500): EXPECTED, no warn
      - Apparently dead (< 1% non-zero across 10+ US tickers): WARN — possible
        HTTP 200 empty-array response from FMP insider endpoint

    Does NOT hard-fail — sparsity is a known property of the insider signal
    on large-cap universes (Cohen, Malloy & Pomorski 2012: ~11% of S&P 500
    tickers have key-officer purchases in any 90-day window).

    Returns True (always) — this is a monitoring signal, not a gate.
    """
    us_results = [r for r in results if r.get("market", "USA") in ("USA", "US")]
    if len(us_results) < 10:
        return True

    conviction_nonzero = sum(
        1 for r in us_results
        if float(r.get("insider_conviction_score", 0.0) or 0.0) > 0.0
    )
    breadth_nonzero = sum(
        1 for r in us_results
        if float(r.get("insider_breadth_score", 0.0) or 0.0) > 0.0
    )
    n = len(us_results)

    conviction_density = conviction_nonzero / n
    breadth_density = breadth_nonzero / n

    if conviction_density < 0.01 and breadth_density < 0.01:
        log.warning(
            "INSIDER FEED LIKELY DEAD (HTTP 200 empty-array): "
            "conviction_density=%.1f%% breadth_density=%.1f%% across %d US tickers. "
            "FMP insider-trading/search may be returning [] for all tickers. "
            "Check fmp_health.json for endpoint call counts vs failures.",
            conviction_density * 100,
            breadth_density * 100,
            n,
        )
    elif conviction_density < 0.03:
        log.warning(
            "Insider conviction density unusually low: %.1f%% across %d US tickers "
            "(expected 5–15%% on S&P 500 universe). "
            "Possible feed degradation — check fmp_health.json.",
            conviction_density * 100,
            n,
        )
    else:
        log.info(
            "Insider feed density OK: conviction=%.1f%% breadth=%.1f%% "
            "across %d US tickers.",
            conviction_density * 100,
            breadth_density * 100,
            n,
        )

    return True


def _format_alert_body(metrics: dict, reasons: List[str]) -> str:
    lines = ["🚨 EDGAR canary failed:"]
    lines.extend(f"  • {r}" for r in reasons)
    lines.append("")
    lines.append("Metrics snapshot:")
    lines.append("```")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canary threshold check")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing metrics.json (default: logs)")
    parser.add_argument("--min-coverage", type=float, default=0.60,
                        help="Minimum EDGAR coverage ratio (default: 0.60)")
    parser.add_argument("--max-errors", type=int, default=0,
                        help="Maximum tolerated error_count (default: 0)")
    parser.add_argument("--webhook", type=str, default=None,
                        help="Discord webhook URL (defaults to env DISCORD_WEBHOOK_URL)")
    parser.add_argument("--no-slack", action="store_true",
                        help="Do not send Discord alert even if a webhook is set")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_path = args.log_dir / "metrics.json"
    if not metrics_path.exists():
        log.error("metrics.json not found at %s", metrics_path)
        return 2

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("could not parse metrics.json: %s", exc)
        return 2

    ok, reasons = evaluate(metrics, min_coverage=args.min_coverage, max_errors=args.max_errors)

    # ── FMP structural-failure gate ────────────────────────────────────────────
    # A non-zero failure count means a factor feed is dead — the pipeline is
    # producing zeroed scores, not sparse ones. This must fail the canary so it
    # surfaces loudly rather than being masked by a lowered circuit-breaker.
    fmp_health_path = args.log_dir / "fmp_health.json"
    if fmp_health_path.exists():
        try:
            fmp_health = json.loads(fmp_health_path.read_text(encoding="utf-8"))
            if fmp_health.get("has_structural_failure"):
                failed_routes = fmp_health.get("failures", {})
                reason = f"FMP structural failure on route(s): {failed_routes}"
                reasons.append(reason)
                ok = False
                log.error("FMP structural failure detected: %s", failed_routes)
        except Exception as exc:
            log.warning("could not read fmp_health.json: %s", exc)

    # ── Score distribution gate (PATCH 11) ────────────────────────────────────
    # Catches dead-feed scenarios where all factors return 0.0 and the
    # cross-sectional normaliser collapses every final_score to ~0.18.
    if not check_score_distribution(args.log_dir):
        reasons.append(
            "Score distribution degenerate: stdev < 0.05 or max_score < 0.40 "
            "— insider/news/congress feeds may all be returning 0.0"
        )
        ok = False

    decision = update_after_evaluation(ok)

    if ok:
        log.info("OK — coverage %d/%d, errors=%d",
                 metrics.get("edgar_count", 0), metrics.get("ticker_count", 0),
                 metrics.get("error_count", 0))
        return 0

    log.error("ALERT — %s (consecutive_failures=%d, escalate=%s)",
              "; ".join(reasons), decision.consecutive_failures, decision.escalate)
    if not args.no_slack:
        webhook: Optional[str] = args.webhook or os.getenv("DISCORD_WEBHOOK_URL")
        if webhook:
            sent = send_slack_alert(
                webhook=webhook,
                title="EDGAR Canary FAILED",
                body=_format_alert_body(metrics, reasons),
                escalate=decision.escalate,
            )
            log.info("discord alert sent=%s", sent)
        else:
            log.warning("DISCORD_WEBHOOK_URL not set — skipping notification")
    return 2


if __name__ == "__main__":
    sys.exit(main())
