"""monitoring/minsky_alert.py — Minsky insider-stress detector.

Hyman Minsky (Financial Instability Hypothesis): prolonged stability breeds
fragility as agents take on excess risk. Applied to insider data: when a large
fraction of the universe shows simultaneous CEO-level buying, the pipeline is
seeing the accumulation phase of a potential Minsky cycle.

Reads intel_source_status.json (written by run_pipeline.py) and computes three
stress signals:
    1. CEO buy ratio   — fraction of tickers with key-exec purchases
    2. Filing velocity — mean Form 4 count across the universe
    3. Insider breadth — fraction of tickers with elevated insider_breadth_score (≥ 0.70)

Stress levels:
    CLEAR    — all signals below watch thresholds
    WATCH    — one signal elevated
    WARNING  — two signals elevated
    CRITICAL — all three signals elevated simultaneously (Minsky trigger)

Always exits 0 — this step is observational, not a gate. Failures loading
the source file are logged as warnings and the module exits cleanly.

Usage:
    python -m monitoring.minsky_alert
    python -m monitoring.minsky_alert --log-dir logs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

from .slack_notifier import send_discord_alert

log = logging.getLogger("monitoring.minsky_alert")

# ── Thresholds ─────────────────────────────────────────────────────────────────
_CEO_BUY_WATCH    = 0.20   # ≥20% of tickers have CEO purchases
_CEO_BUY_ELEVATED = 0.40   # ≥40% — elevated conviction
_CEO_BUY_CRITICAL = 0.60   # ≥60% — mass executive accumulation

_FILING_WATCH    = 3.0    # mean Form 4 count per ticker
_FILING_ELEVATED = 5.0

_BREADTH_WATCH    = 0.25   # fraction with insider_score ≥ 0.70
_BREADTH_ELEVATED = 0.50


class _StressResult(NamedTuple):
    level: str            # CLEAR | WATCH | WARNING | CRITICAL
    conditions_met: int   # 0–3
    ceo_buy_ratio: float
    mean_form4: float
    breadth_ratio: float
    narrative: str


def _compute_stress(results: list) -> _StressResult:
    """Minsky: derive stress level from the pipeline's per-ticker result rows."""
    n = len(results)
    if n == 0:
        return _StressResult("CLEAR", 0, 0.0, 0.0, 0.0, "No tickers to evaluate.")

    ceo_buys   = sum(1 for r in results if r.get("ceo_buy", False))
    mean_form4 = sum(r.get("form4_count", 0) for r in results) / n
    # 7-factor pipeline emits `insider_breadth_score`; fall back to the legacy
    # `insider_score` so historical snapshots still score correctly.
    breadth    = sum(
        1 for r in results
        if (r.get("insider_breadth_score") or r.get("insider_score") or 0) >= 0.70
    )

    ceo_ratio     = ceo_buys / n
    breadth_ratio = breadth / n

    flags: List[str] = []
    if ceo_ratio >= _CEO_BUY_ELEVATED:
        flags.append(f"CEO buy ratio {ceo_ratio:.0%} ≥ {_CEO_BUY_ELEVATED:.0%}")
    if mean_form4 >= _FILING_ELEVATED:
        flags.append(f"mean Form 4 filings {mean_form4:.1f} ≥ {_FILING_ELEVATED:.0f}")
    if breadth_ratio >= _BREADTH_ELEVATED:
        flags.append(f"insider breadth {breadth_ratio:.0%} ≥ {_BREADTH_ELEVATED:.0%}")

    conditions_met = len(flags)

    # Escalate into WATCH if any single threshold is breached at the lower level
    watch_flags = 0
    if ceo_ratio >= _CEO_BUY_WATCH:
        watch_flags += 1
    if mean_form4 >= _FILING_WATCH:
        watch_flags += 1
    if breadth_ratio >= _BREADTH_WATCH:
        watch_flags += 1

    if conditions_met == 3:
        level = "CRITICAL"
        narrative = (
            "MINSKY MOMENT — All 3 insider-stress preconditions breached. "
            "Mass executive accumulation detected: " + " | ".join(flags)
        )
    elif conditions_met == 2:
        level = "WARNING"
        narrative = "Insider stress elevated on 2 axes: " + " | ".join(flags)
    elif conditions_met == 1 or watch_flags >= 1:
        level = "WATCH"
        parts = flags or [
            f"CEO buy ratio {ceo_ratio:.0%}",
            f"mean Form 4 {mean_form4:.1f}",
            f"insider breadth {breadth_ratio:.0%}",
        ]
        narrative = "Insider stress at watch level: " + " | ".join(parts[:2])
    else:
        level = "CLEAR"
        narrative = (
            f"No Minsky insider stress. "
            f"CEO buy ratio {ceo_ratio:.0%}, "
            f"mean Form 4 {mean_form4:.1f}, "
            f"insider breadth {breadth_ratio:.0%}."
        )

    return _StressResult(level, conditions_met, ceo_ratio, mean_form4, breadth_ratio, narrative)


def _format_discord_body(stress: _StressResult, ticker_count: int) -> str:
    icons = {"CRITICAL": "🚨", "WARNING": "⚠️", "WATCH": "👁️", "CLEAR": "✅"}
    icon  = icons.get(stress.level, "ℹ️")
    lines = [
        f"{icon} **{stress.level}** ({stress.conditions_met}/3 conditions met)",
        "",
        stress.narrative,
        "",
        "```",
        f"Universe:       {ticker_count} tickers",
        f"CEO buy ratio:  {stress.ceo_buy_ratio:.1%}",
        f"Mean Form 4:    {stress.mean_form4:.2f}",
        f"Insider breadth:{stress.breadth_ratio:.1%}",
        "```",
    ]
    return "\n".join(lines)


def run(log_dir: Path, webhook: Optional[str] = None, no_alert: bool = False) -> int:
    """Load pipeline results, compute stress, optionally alert. Returns 0 always."""
    src = log_dir / "intel_source_status.json"
    if not src.exists():
        log.warning("intel_source_status.json not found at %s — skipping Minsky check", src)
        return 0

    try:
        raw     = json.loads(src.read_text(encoding="utf-8"))
        results = raw.get("results", [])
        meta    = raw.get("_edgar_meta", {})
    except Exception as exc:
        log.warning("Could not parse %s: %s — skipping Minsky check", src, exc)
        return 0

    stress       = _compute_stress(results)
    ticker_count = int(meta.get("ticker_count", len(results)))

    log.info(
        "Minsky stress: level=%s conditions=%d/3 ceo_ratio=%.1f%% mean_form4=%.2f breadth=%.1f%%",
        stress.level, stress.conditions_met,
        stress.ceo_buy_ratio * 100, stress.mean_form4, stress.breadth_ratio * 100,
    )
    log.info(stress.narrative)

    should_alert = stress.level in ("WARNING", "CRITICAL") and not no_alert
    if should_alert:
        wh = webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
        if wh:
            sent = send_discord_alert(
                webhook=wh,
                title=f"Minsky Insider Stress: {stress.level}",
                body=_format_discord_body(stress, ticker_count),
                escalate=(stress.level == "CRITICAL"),
            )
            log.info("discord alert sent=%s", sent)
        else:
            log.warning("DISCORD_WEBHOOK_URL not set — skipping notification")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minsky insider-stress detector")
    parser.add_argument("--log-dir",  type=Path, default=Path("logs"),
                        help="Directory containing intel_source_status.json (default: logs)")
    parser.add_argument("--webhook",  type=str,  default=None,
                        help="Discord webhook URL (defaults to env DISCORD_WEBHOOK_URL)")
    parser.add_argument("--no-alert", action="store_true",
                        help="Compute stress but do not send Discord notification")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args.log_dir, webhook=args.webhook, no_alert=args.no_alert)


if __name__ == "__main__":
    sys.exit(main())
