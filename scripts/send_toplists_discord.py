"""scripts/send_toplists_discord.py
Send the daily market checkup to Discord via Embed webhook.

Reads logs/top_lists.json (or --input path) and formats a rich Discord
embed with three sections: Top 5 Buys, Top 5 Mid Caps, Top 5 Small Caps.
Each ticker shows its 5-factor breakdown.

Discord embed limits:
  title:       256 chars
  description: 4096 chars
  field name:  256 chars
  field value: 1024 chars (hard limit — truncated if exceeded)
  fields:      25 max
  total:       6000 chars

Color coding:
  GREEN  (#00b37d) — overall minsky alert CLEAR / avg score ≥ 0.65
  ORANGE (#ff9800) — WATCH / WARNING     / avg score 0.45–0.65
  RED    (#e53935) — CRITICAL            / avg score < 0.45

Retry policy: 3 attempts with 30s / 60s backoff (same as monitoring/).

Usage:
  python scripts/send_toplists_discord.py
  python scripts/send_toplists_discord.py --input logs/top_lists.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

try:
    from regime_trader.utils.formatting import score_bar as _score_bar_util
    _HAS_SCORE_BAR = True
except ImportError:
    _HAS_SCORE_BAR = False

log = logging.getLogger("discord.send_toplists")

# ── Color palette ──────────────────────────────────────────────────────────────
_COLOR_GREEN  = 0x00B37D   # strong buy universe
_COLOR_ORANGE = 0xFF9800   # caution
_COLOR_RED    = 0xE53935   # stress / CRITICAL

# ── Factor emoji map ───────────────────────────────────────────────────────────
_FACTOR_EMOJI = {
    "edgar":    "📋",
    "insider":  "🏦",
    "congress": "🏛️",
    "news":     "📰",
    "momentum": "📈",
}

_BADGE_EMOJI = {
    "HIGH BUY":     "🟢",
    "TACTICAL BUY": "🟡",
    "WATCHLIST":    "⚪",
}


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    """Compact ASCII progress bar: ████░░░░"""
    if _HAS_SCORE_BAR:
        return _score_bar_util(score, width)
    filled = min(width, max(0, round(score * width)))
    return "█" * filled + "░" * (width - filled)


def _format_factor_line(factors: Dict[str, float]) -> str:
    """One compact line: 📋0.72 🏦0.90 🏛️0.50 📰0.65 📈0.58"""
    parts = []
    for key in ("edgar", "insider", "congress", "news", "momentum"):
        v = factors.get(key, 0.0)
        parts.append(f"{_FACTOR_EMOJI[key]}`{v:.2f}`")
    return "  ".join(parts)


def _format_ticker_block(entry: Dict[str, Any], rank: Optional[int] = None) -> str:
    """Format one ticker for a Discord embed field value.

    Output (fits within ~200 chars):
        **1. AAPL** 🟡  `0.7341`  TACTICAL BUY
        📋`0.40` 🏦`0.50` 🏛️`0.50` 📰`0.60` 📈`0.65`  ████░░░░
    """
    ticker   = entry.get("ticker", "?")
    score    = float(entry.get("final_score", 0))
    badge    = entry.get("badge", "WATCHLIST")
    factors  = entry.get("factors", {})
    ceo_buy  = entry.get("ceo_buy", False)

    prefix   = f"**{rank}. " if rank else "**"
    suffix   = " (CEO BUY ⚡)" if ceo_buy else ""
    emoji    = _BADGE_EMOJI.get(badge, "⚪")
    bar      = _score_bar(score)
    line1    = f"{prefix}{ticker}**{suffix}  {emoji}  `{score:.4f}`  {badge}"
    line2    = f"{_format_factor_line(factors)}  {bar}"
    return f"{line1}\n{line2}"


def _truncate(text: str, max_chars: int = 1024) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "…"


def _pick_color(top_lists: Dict[str, Any]) -> int:
    """Color based on average final_score across all top_buys."""
    entries = top_lists.get("top_buys") or []
    if not entries:
        return _COLOR_ORANGE
    avg = sum(float(e.get("final_score", 0)) for e in entries) / len(entries)
    if avg >= 0.65:
        return _COLOR_GREEN
    if avg >= 0.45:
        return _COLOR_ORANGE
    return _COLOR_RED


def _section_value(entries: List[Dict], max_chars: int = 1020) -> str:
    """Format up to 5 tickers into one embed field value."""
    if not entries:
        return "_No data available for this universe tier._"
    blocks = []
    for i, entry in enumerate(entries, 1):
        blocks.append(_format_ticker_block(entry, rank=i))
    return _truncate("\n".join(blocks), max_chars)


_STALE_HOURS = 25   # warn in Discord if top_lists.json is older than this


def _data_age_hours(generated_at: str) -> Optional[float]:
    """Return age of top_lists.json in hours, or None if unparseable."""
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    except Exception:
        return None


def _load_satellite(log_dir: Path) -> dict | None:
    """Load satellite_insights.json if present. Returns None on any failure."""
    path = log_dir / "satellite_insights.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        log.warning("satellite_insights.json unreadable: %s", exc)
        return None


def build_payload(top_lists: Dict[str, Any], satellite: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the Discord webhook JSON payload with embeds."""
    generated_at = top_lists.get("generated_at", "")
    run_id       = top_lists.get("source_run_id", "")
    ticker_count = top_lists.get("ticker_count", 0)
    weights      = top_lists.get("weights", {})

    age_h = _data_age_hours(generated_at)
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        date_str = generated_at[:16] or "—"

    color = _pick_color(top_lists)

    _NOMINAL_WEIGHTS = {
        "edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12,
    }
    weights_redistributed = bool(weights) and any(
        abs(weights.get(k, 0) - _NOMINAL_WEIGHTS.get(k, 0)) > 0.001
        for k in _NOMINAL_WEIGHTS
    )
    if weights:
        weight_str = " · ".join(f"{k}={v:.0%}" for k, v in weights.items())
        if weights_redistributed:
            weight_str += " ⚠️ _(feed down — redistributed)_"
    else:
        weight_str = "default"

    stale_warning = ""
    if age_h is not None and age_h > _STALE_HOURS:
        stale_warning = (
            f"\n⚠️ **DATA IS {age_h:.0f}h OLD** — pipeline may have failed. "
            "Check the `edgar_3x` workflow on GitHub Actions."
        )
        color = _COLOR_RED

    # Macro kill-switch — override color and warn when VIX >= 30
    kill_switch = top_lists.get("kill_switch", False)
    if kill_switch:
        color = _COLOR_RED
        vix_val  = top_lists.get("vix")
        vix_mult = top_lists.get("vix_multiplier", 1.0)
        vix_note = f" VIX {vix_val:.1f} ·" if vix_val is not None else ""
        stale_warning += (
            f"\n⛔ **MACRO KILL-SWITCH ACTIVE** —{vix_note} "
            f"all scores dampened ×{vix_mult:.2f}. "
            "Do NOT act on HIGH BUY signals."
        )

    description = (
        f"**Universe:** {ticker_count} tickers  |  "
        f"**Run:** `{run_id}`\n"
        f"**Weights:** {weight_str}\n"
        f"*Tiers by market cap: Large ≥$10B · Mid $2–10B · Small <$2B*"
        f"{stale_warning}"
    )

    # Build embed fields
    fields = [
        {
            "name":   "🏆 Top 5 Buys",
            "value":  _section_value(top_lists.get("top_buys") or []),
            "inline": False,
        },
        {
            "name":   "📈 Top 5 Mid Caps ($2B–$10B)",
            "value":  _section_value(top_lists.get("mid_caps") or []),
            "inline": False,
        },
        {
            "name":   "🔬 Top 5 Small Caps (<$2B)",
            "value":  _section_value(top_lists.get("small_caps") or []),
            "inline": False,
        },
    ]

    # ── Satellite fields (optional — wrapped so never crashes embed) ──────
    try:
        if satellite and isinstance(satellite, dict):
            month_label = satellite.get("month", "")
            cyclicals = satellite.get("cyclicals") or []
            cannibals = satellite.get("cannibals") or []

            if cyclicals:
                lines = []
                for i, c in enumerate(cyclicals, 1):
                    wr   = f"{c['win_rate']:.0%}"
                    med  = f"{c['median_return']:+.1%}"
                    yr   = c.get("years", "?")
                    lines.append(f"{i}. {c['ticker']}  Win-rate: {wr}  Median: {med}  ({yr} yr)")
                fields.append({
                    "name":   f"🌀 Seasonal Cyclicals — {month_label}",
                    "value":  "\n".join(lines),
                    "inline": False,
                })

            if cannibals:
                lines = []
                for i, c in enumerate(cannibals, 1):
                    yld  = f"{c['buyback_yield']:.1%}"
                    pe   = f"{c['pe']:.1f}"
                    pvl  = f"{c['price_vs_52w_low']:.2f}"
                    lines.append(f"{i}. {c['ticker']}  Yield: {yld}  P/E: {pe}  Price/52wLow: {pvl}×")
                fields.append({
                    "name":   "🐷 Share Cannibals — Buyback Yield",
                    "value":  "\n".join(lines),
                    "inline": False,
                })
    except Exception as exc:
        log.warning("satellite embed fields skipped due to error: %s", exc)

    fields += [
        {
            "name":   "📊 Factor Legend",
            "value":  (
                "📋 EDGAR  🏦 Insider  🏛️ Congress/Inst  📰 News  📈 Momentum\n"
                "Scores ∈ [0,1] · 0.50 = neutral · >0.65 = strong buy signal"
            ),
            "inline": False,
        },
    ]

    embed = {
        "title":       f"📊 Daily Market Checkup — {date_str}",
        "description": description,
        "color":       color,
        "fields":      fields,
        "footer":      {
            "text": "regime_trader · EDGAR-first pipeline · regime_trader/regimetrader",
        },
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    return {"embeds": [embed]}


def build_alert_payload(reason: str) -> Dict[str, Any]:
    """Minimal embed when top_lists.json is missing or unreadable."""
    return {
        "embeds": [{
            "title":       "⚠️ Daily Market Checkup — DATA UNAVAILABLE",
            "description": (
                f"**Reason:** {reason}\n\n"
                "The EDGAR pipeline may not have completed its last run.\n"
                "Check the `edgar_3x` workflow on GitHub Actions."
            ),
            "color":       _COLOR_RED,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "footer":      {"text": "regime_trader · EDGAR-first pipeline"},
        }]
    }


# ── HTTP send with retry ───────────────────────────────────────────────────────

def send_to_discord(
    webhook: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    backoff_base_s: float = 30.0,
) -> bool:
    """POST payload to Discord webhook with exponential-like backoff.

    Backoff schedule: 30s, 60s (attempt 1 is immediate).
    Never raises — returns True on 2xx, False otherwise.
    """
    if not _HAS_REQUESTS:
        log.error("'requests' not installed — cannot send to Discord")
        return False
    if not webhook:
        log.warning("DISCORD_WEBHOOK_URL is empty — skipping (no-op)")
        return False

    for attempt in range(max_retries):
        if attempt > 0:
            wait = backoff_base_s * attempt          # 30s, 60s
            log.warning("Retry %d/%d in %.0fs …", attempt + 1, max_retries, wait)
            time.sleep(wait)
        try:
            resp = requests.post(webhook, json=payload, timeout=15.0)
            if 200 <= resp.status_code < 300:
                log.info("Discord message sent (status=%d)", resp.status_code)
                return True
            # 429 = rate-limited: prefer JSON body retry_after (Discord standard),
            # fall back to the Retry-After header, then a 30s default.
            if resp.status_code == 429:
                try:
                    retry_after = float(resp.json().get("retry_after", 0))
                except Exception:
                    retry_after = 0.0
                if not retry_after:
                    retry_after = float(resp.headers.get("Retry-After", 30))
                log.warning("Discord rate-limited — waiting %.1fs", retry_after)
                time.sleep(retry_after)
                continue
            log.warning(
                "Discord non-2xx status=%d body=%r (attempt=%d/%d)",
                resp.status_code, resp.text[:200], attempt + 1, max_retries,
            )
        except Exception as exc:
            log.warning(
                "Discord POST failed (attempt=%d/%d): %s",
                attempt + 1, max_retries, str(exc)[:200],
            )
    return False


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Send daily market checkup to Discord")
    parser.add_argument(
        "--input", type=Path, default=Path("logs/top_lists.json"),
        help="Path to top_lists.json (default: logs/top_lists.json)",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory for discord_send.log output",
    )
    parser.add_argument(
        "--webhook", type=str, default=None,
        help="Discord webhook URL (overrides env DISCORD_WEBHOOK_URL)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print payload JSON without sending",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    args.log_dir.mkdir(parents=True, exist_ok=True)
    discord_log = args.log_dir / "discord_send.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(discord_log, encoding="utf-8"),
        ],
    )

    webhook: str = args.webhook or os.getenv("DISCORD_WEBHOOK_URL", "")

    # ── Load top_lists.json ────────────────────────────────────────────────────
    if not args.input.exists():
        log.warning("top_lists.json not found at %s — sending alert", args.input)
        payload = build_alert_payload(f"File not found: {args.input}")
        if args.dry_run:
            sys.stdout.buffer.write((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
            return 0
        ok = send_to_discord(webhook, payload)
        return 0 if ok else 1

    try:
        top_lists = json.loads(args.input.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("could not parse top_lists.json: %s", exc)
        payload = build_alert_payload(f"JSON parse error: {exc}")
        if args.dry_run:
            sys.stdout.buffer.write((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
            return 0
        send_to_discord(webhook, payload)
        return 1

    if not isinstance(top_lists, dict) or "top_buys" not in top_lists:
        log.error("top_lists.json has unexpected structure")
        payload = build_alert_payload("Unexpected JSON structure in top_lists.json")
        if args.dry_run:
            sys.stdout.buffer.write((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
            return 0
        send_to_discord(webhook, payload)
        return 1

    # ── Build and send ─────────────────────────────────────────────────────────
    satellite = _load_satellite(args.log_dir)
    payload   = build_payload(top_lists, satellite=satellite)

    if args.dry_run:
        out = json.dumps(payload, indent=2, ensure_ascii=False)
        sys.stdout.buffer.write((out + "\n").encode("utf-8"))
        return 0

    ok = send_to_discord(webhook, payload)
    if not ok:
        log.error("All Discord send attempts failed")
        return 1

    log.info("Daily market checkup sent successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
