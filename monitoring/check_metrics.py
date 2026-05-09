"""monitoring/check_metrics.py — threshold gate for the canary.

Reads `<log-dir>/metrics.json` and exits:
    0 — all thresholds passed
    2 — ALERT (errors > 0  OR  edgar_count / ticker_count < min_coverage)

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

from .alert_state import update_after_evaluation
from .evaluate import evaluate
from .slack_notifier import send_discord_alert as send_slack_alert

log = logging.getLogger("monitoring.check_metrics")


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
