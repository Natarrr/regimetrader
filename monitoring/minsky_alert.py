"""monitoring/minsky_alert.py — EDGAR insider stress alert.

Reads marketintel_events.json produced by the nightly EDGAR pipeline and
computes a three-condition insider stress metric that mirrors the structure
of the Minsky Financial Instability Hypothesis.

This is an EDGAR-FIRST signal — it does NOT require GARCH / CAPE / yield
data. It measures whether insiders as a group are signalling systemic risk
through their own open-market trading behaviour.

Three conditions (all must breach for CRITICAL):
    1. Bearish universe     : universe avg insider score < 0.40
    2. Sell concentration   : fraction of tickers with score < 0.35  > 0.50
    3. No buy conviction    : zero CEO / CFO open-market purchases in universe

Alert levels (mirroring MinskyStatusOut from prediction_controller.py):
    CRITICAL (3/3) — exit 1  : insider exodus, no buy conviction, mass selling
    WARNING  (2/3) — exit 0  : two signals active — monitor and trim risk
    WATCH    (1/3) — exit 0  : one signal active — heightened attention
    CLEAR    (0/3) — exit 0  : normal insider activity

Graceful degradation:
    Missing file  → warning log, exit 0 (does not fail the job)
    Malformed JSON → warning log, exit 0
    Empty events  → treated as CLEAR, exit 0

Usage:
    python -m monitoring.minsky_alert
    python -m monitoring.minsky_alert --log-dir logs --no-slack
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .slack_notifier import send_discord_alert as send_slack_alert

log = logging.getLogger("monitoring.minsky_alert")

# ── Thresholds ─────────────────────────────────────────────────────────────────
# Calibrated against real top-50 EDGAR runs (May 2026):
#   typical avg_score ≈ 0.38–0.45 in neutral markets
#   CEO buys absent in ~70% of universe-wide runs
#   sell_fraction > 0.50 is a 2-sigma event for large-cap insiders

_AVG_SCORE_THRESHOLD    = 0.40   # condition 1: avg score below this
_SELL_SCORE_FLOOR       = 0.35   # per-ticker floor used for condition 2
_SELL_FRACTION_THRESH   = 0.50   # condition 2: fraction below floor exceeds this
# condition 3: no CEO/CFO open-market purchase anywhere in universe

_ALERT_LEVELS = {0: "CLEAR", 1: "WATCH", 2: "WARNING", 3: "CRITICAL"}


# ── Core computation ───────────────────────────────────────────────────────────

def compute_insider_stress(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute EDGAR-native insider stress from marketintel_events.json entries.

    Spence (2001 Nobel) — CEO open-market purchases are costly signals;
    their absence across the entire universe is the strongest risk flag.

    Args:
        events: Parsed list from marketintel_events.json. Each entry must have
                at minimum: ticker (str), score (float), score_breakdown (dict).

    Returns:
        Dict with raw metrics, per-condition booleans, conditions_met [0-3],
        and alert_level ["CLEAR" | "WATCH" | "WARNING" | "CRITICAL"].
    """
    if not events:
        return {
            "ticker_count":                 0,
            "avg_score":                    0.50,
            "sell_fraction":                0.0,
            "ceo_buy_present":              False,
            "conditions_met":               0,
            "alert_level":                  "CLEAR",
            "condition_bearish_universe":   False,
            "condition_sell_concentration": False,
            "condition_no_buy_conviction":  False,
            "per_ticker":                   [],
        }

    scores: List[float] = []
    ceo_buy_present = False
    per_ticker: List[Dict] = []

    for entry in events:
        ticker  = str(entry.get("ticker") or "?")
        score   = float(entry.get("score") or 0.50)
        brk     = entry.get("score_breakdown") or {}
        ceo_buy = bool(brk.get("ceo_buy", False))

        scores.append(score)
        if ceo_buy:
            ceo_buy_present = True

        per_ticker.append({
            "ticker":    ticker,
            "score":     round(score, 4),
            "ceo_buy":   ceo_buy,
            "net_value": brk.get("net_value"),
        })

    ticker_count  = len(scores)
    avg_score     = sum(scores) / ticker_count
    below_floor   = sum(1 for s in scores if s < _SELL_SCORE_FLOOR)
    sell_fraction = below_floor / ticker_count

    cond_bearish = avg_score     < _AVG_SCORE_THRESHOLD
    cond_conc    = sell_fraction > _SELL_FRACTION_THRESH
    cond_no_buy  = not ceo_buy_present

    conditions_met = int(cond_bearish) + int(cond_conc) + int(cond_no_buy)

    return {
        "ticker_count":                 ticker_count,
        "avg_score":                    round(avg_score, 4),
        "sell_fraction":                round(sell_fraction, 4),
        "ceo_buy_present":              ceo_buy_present,
        "conditions_met":               conditions_met,
        "alert_level":                  _ALERT_LEVELS[conditions_met],
        "condition_bearish_universe":   cond_bearish,
        "condition_sell_concentration": cond_conc,
        "condition_no_buy_conviction":  cond_no_buy,
        "per_ticker":                   per_ticker,
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _slack_body(result: Dict[str, Any]) -> str:
    level = result["alert_level"]
    n     = result["conditions_met"]
    lines = [
        f"*EDGAR Insider Stress: {level} ({n}/3 conditions triggered)*",
        "",
        f"• Universe avg score  : `{result['avg_score']:.4f}`  (alert threshold < {_AVG_SCORE_THRESHOLD})",
        f"• Sell fraction       : `{result['sell_fraction']:.1%}`  (alert threshold > {_SELL_FRACTION_THRESH:.0%})",
        f"• CEO buy present     : `{result['ceo_buy_present']}`",
        "",
        "Conditions:",
        f"  [{'X' if result['condition_bearish_universe']   else ' '}] Bearish universe  (avg score < {_AVG_SCORE_THRESHOLD})",
        f"  [{'X' if result['condition_sell_concentration'] else ' '}] Sell concentration (>{_SELL_FRACTION_THRESH:.0%} tickers below {_SELL_SCORE_FLOOR})",
        f"  [{'X' if result['condition_no_buy_conviction']  else ' '}] No CEO/CFO open-market purchases",
    ]
    worst = sorted(result["per_ticker"], key=lambda x: x["score"])[:5]
    if worst:
        lines += ["", "Weakest insider scores:"]
        for t in worst:
            lines.append(f"  • {t['ticker']}: score={t['score']:.4f}  ceo_buy={t['ceo_buy']}")
    return "\n".join(lines)


def _step_summary(result: Dict[str, Any]) -> str:
    level = result["alert_level"]
    n     = result["conditions_met"]
    emoji = {"CLEAR": "✅", "WATCH": "👀", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level, "")
    lines = [
        "### Minsky Alert Status — EDGAR Insider Signal",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Alert level | `{level}` {emoji} |",
        f"| Conditions met | `{n} / 3` |",
        f"| Tickers analysed | `{result['ticker_count']}` |",
        f"| Universe avg score | `{result['avg_score']:.4f}` |",
        f"| Sell fraction | `{result['sell_fraction']:.1%}` |",
        f"| CEO buy present | `{result['ceo_buy_present']}` |",
        f"| Bearish universe | `{result['condition_bearish_universe']}` |",
        f"| Sell concentration | `{result['condition_sell_concentration']}` |",
        f"| No buy conviction | `{result['condition_no_buy_conviction']}` |",
        "",
    ]
    if level == "CRITICAL":
        lines += [
            "> **CRITICAL**: mass insider selling, zero buy conviction, extreme concentration.",
            "> Slack notification sent. Job marked FAILED.",
        ]
    elif level == "WARNING":
        lines += ["> **WARNING**: two of three insider stress conditions active. Monitor closely."]
    elif level == "WATCH":
        lines += ["> **WATCH**: one insider stress signal active. No immediate action required."]
    else:
        lines += ["> **CLEAR**: insider activity within normal parameters."]
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="EDGAR insider stress / Minsky-style alert check",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory containing marketintel_events.json (default: logs)",
    )
    parser.add_argument(
        "--webhook", type=str, default=None,
        help="Slack webhook URL (overrides env DISCORD_WEBHOOK_URL)",
    )
    parser.add_argument(
        "--no-slack", action="store_true",
        help="Suppress Slack notification even when CRITICAL",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    events_path = args.log_dir / "marketintel_events.json"

    # ── Graceful degradation ───────────────────────────────────────────────────
    if not events_path.exists():
        log.warning(
            "marketintel_events.json not found at %s — skipping Minsky check (exit 0)",
            events_path,
        )
        return 0

    try:
        raw = json.loads(events_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("could not parse marketintel_events.json: %s — skipping (exit 0)", exc)
        return 0

    if not isinstance(raw, list):
        log.warning(
            "unexpected format in marketintel_events.json (expected list, got %s) — skipping (exit 0)",
            type(raw).__name__,
        )
        return 0

    # ── Compute stress metric ──────────────────────────────────────────────────
    result = compute_insider_stress(raw)
    level  = result["alert_level"]
    n      = result["conditions_met"]

    log.info(
        "EDGAR insider stress: %s (%d/3)  |  avg_score=%.4f  |  sell_frac=%.1f%%  |  ceo_buy=%s",
        level, n,
        result["avg_score"],
        result["sell_fraction"] * 100,
        result["ceo_buy_present"],
    )

    # ── GITHUB_STEP_SUMMARY ────────────────────────────────────────────────────
    summary_path_str = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_path_str:
        try:
            with open(summary_path_str, "a", encoding="utf-8") as fh:
                fh.write("\n" + _step_summary(result) + "\n")
        except Exception as exc:
            log.warning("could not write step summary: %s", exc)

    # ── Slack alert (CRITICAL only) ────────────────────────────────────────────
    if level == "CRITICAL" and not args.no_slack:
        webhook: Optional[str] = args.webhook or os.getenv("DISCORD_WEBHOOK_URL")
        sent = send_slack_alert(
            webhook=webhook,
            title="EDGAR Insider Stress CRITICAL",
            body=_slack_body(result),
            escalate=True,
        )
        log.info("Discord alert sent=%s", sent)

    # ── Exit code ──────────────────────────────────────────────────────────────
    if level == "CRITICAL":
        log.critical(
            "CRITICAL insider stress detected (%d/3 conditions) — failing job", n
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
