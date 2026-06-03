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

_SCORE_BUCKETS = (
    "top_buys", "top_buys_usa", "top_buys_europe",
    "top_buys_asia", "mid_caps", "small_caps",
)


def check_score_distribution(
    log_dir: Path,
    min_stdev: float = 0.05,
    min_max_score: float = 0.40,
    min_entries: int = 3,
) -> bool:
    """Assert that top_lists.json score distribution is non-degenerate.

    Detects dead-feed scenarios where all tickers score near the weight-floor
    value (~0.18 when all factors return 0.0). Returns True when healthy,
    False when degenerate. Missing or unreadable files are treated as skips
    (return True) so canary does not fail on a missing artifact.
    """
    tl_path = Path(log_dir) / "top_lists.json"
    if not tl_path.exists():
        log.warning("check_score_distribution: top_lists.json not found — skipping")
        return True

    try:
        data = json.loads(tl_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("check_score_distribution: cannot parse top_lists.json — %s", exc)
        return True

    scores: list[float] = []
    for bucket in _SCORE_BUCKETS:
        for entry in data.get(bucket, []):
            s = entry.get("final_score")
            if s is not None:
                scores.append(float(s))

    scores = list(set(scores))  # deduplicate same ticker across lists

    if len(scores) < min_entries:
        log.warning(
            "check_score_distribution: only %d unique scores (need >= %d) — skipping",
            len(scores), min_entries,
        )
        return True

    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    max_score = max(scores)
    mean_score = statistics.mean(scores)
    degenerate = False

    if stdev < min_stdev:
        log.error(
            "::error::SCORE DISTRIBUTION DEGENERATE: stdev=%.4f < %.4f threshold. "
            "All tickers scoring near %.4f — data feeds may be dead. "
            "Check FMP endpoint health (fmp_health.json).",
            stdev, min_stdev, mean_score,
        )
        degenerate = True

    if max_score < min_max_score:
        log.error(
            "::error::DEAD SIGNAL DETECTED: max_score=%.4f < %.4f threshold. "
            "No ticker reached the minimum actionable score — "
            "insider/news/congress feeds may all be returning 0.0.",
            max_score, min_max_score,
        )
        degenerate = True

    if not degenerate:
        log.info(
            "check_score_distribution: OK — stdev=%.4f max=%.4f mean=%.4f (%d entries)",
            stdev, max_score, mean_score, len(scores),
        )

    return not degenerate


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
