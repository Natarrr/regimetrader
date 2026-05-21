"""scripts/send_toplists_discord.py
Send the daily market report to Discord via Embed webhook.

Reads logs/top_lists.json (or --input path) and formats a structured
Discord embed designed as an institutional terminal report:

  [REGIME TRADER] Daily Market Report  — date-stamped title
  Top Conviction   — top-3 tickers with score + ⚡ signal emoji
  Fundamentals     — EDGAR / Insider code-block table (inline)
  Sentiment        — Congress / News code-block table (inline)
  Mid Caps         — compact opportunities list (when present)
  Seasonal Cyclicals — satellite seasonal data (when present)

Discord embed limits:
  title:       256 chars
  description: 4096 chars
  field name:  256 chars
  field value: 1024 chars (hard limit — truncated if exceeded)
  fields:      25 max
  total:       6000 chars

Color coding (driven by top conviction pick, not average):
  GREEN  (0x2ecc71) — top pick score ≥ 0.70
  BLUE   (0x3498db) — top pick score < 0.70  or  no data
  RED    (0xe53935) — kill-switch active or stale data (>25h)

Retry policy: 3 attempts with 30s / 60s backoff (same as monitoring/).

Usage:
  python scripts/send_toplists_discord.py
  python scripts/send_toplists_discord.py --input logs/top_lists.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
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
_COLOR_GREEN  = 0x2ECC71   # success — top pick ≥ 0.70
_COLOR_BLUE   = 0x3498DB   # info — top pick < 0.70
_COLOR_RED    = 0xE53935   # stress / CRITICAL — kill-switch or stale
# Legacy aliases used in alert builder
_COLOR_ORANGE = 0xFF9800

# ── Badge / signal config ──────────────────────────────────────────────────────
_BADGE_EMOJI = {
    "HIGH BUY":     "🟢",
    "TACTICAL BUY": "🟡",
    "WATCHLIST":    "⚪",
}

# Factor key order for grouped display
_FUNDAMENTAL = ("edgar", "insider")
_SENTIMENT   = ("congress", "news")
_TECHNICAL   = ("macro",)

_FACTOR_LABEL = {
    "edgar":    "EDGAR",
    "insider":  "Insider",
    "congress": "Congress",
    "news":     "News",
    "macro":    "Macro",
    "momentum": "Momentum",
}


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    """Compact progress bar using solid/light block chars: ▓▓▓▓░░░░"""
    if _HAS_SCORE_BAR:
        return _score_bar_util(score, width)
    filled = min(width, max(0, round(score * width)))
    return "▓" * filled + "░" * (width - filled)


def _fmt_cap(market_cap: float) -> str:
    """Format market cap as $300B / $4.2B / $850M."""
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.1f}T"
    if market_cap >= 1e9:
        v = market_cap / 1e9
        return f"${v:.0f}B" if v >= 10 else f"${v:.1f}B"
    return f"${market_cap/1e6:.0f}M"


def _factor_group(factors: Dict[str, float], keys: tuple) -> str:
    """Render a subset of factors as  KEY `0.72`  KEY `0.90` """
    parts = []
    for k in keys:
        v = factors.get(k)
        if v is not None:
            label = _FACTOR_LABEL.get(k, k.upper())
            parts.append(f"{label} `{v:.2f}`")
    return "  ·  ".join(parts) if parts else "—"


def _truncate(text: str, max_chars: int = 1024) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "…"


def _pick_color(top_lists: Dict[str, Any]) -> int:
    """Color driven by the single highest-conviction pick (top_buys[0]).

    0x2ecc71 (Success Green) if top score > 0.70, else 0x3498db (Info Blue).
    Overridden to RED by callers on kill-switch / stale data.
    """
    entries = top_lists.get("top_buys") or []
    if not entries:
        return _COLOR_BLUE
    top_score = float(entries[0].get("final_score", 0))
    return _COLOR_GREEN if top_score >= 0.70 else _COLOR_BLUE


# ── Field builders ─────────────────────────────────────────────────────────────

def _top_conviction_field(entries: List[Dict]) -> Dict[str, Any]:
    """Top Conviction: top-3 tickers with score and ⚡ signal emoji — non-inline."""
    lines = []
    for e in entries[:3]:
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        badge  = e.get("badge", "WATCHLIST")
        ceo    = " ⚡" if e.get("ceo_buy") else ""
        emoji  = _BADGE_EMOJI.get(badge, "⚪")
        lines.append(f"⚡ **{ticker}**{ceo}  {emoji}  `{score:.4f}`  —  {badge}")
    return {
        "name":   "Top Conviction",
        "value":  _truncate("\n".join(lines) or "—"),
        "inline": False,
    }


def _fundamentals_field(entries: List[Dict]) -> Dict[str, Any]:
    """Fundamentals: Factor | Score code-block table — inline."""
    top = entries[0] if entries else {}
    factors = top.get("factors", {})
    rows = ["```", f"{'Factor':<12} Score", "─" * 20]
    for k in _FUNDAMENTAL:
        v = factors.get(k)
        if v is not None:
            label = _FACTOR_LABEL.get(k, k.upper())
            rows.append(f"{label:<12} {v:.4f}")
    rows.append("```")
    return {
        "name":   "Fundamentals",
        "value":  _truncate("\n".join(rows), 1020),
        "inline": True,
    }


def _sentiment_field(entries: List[Dict]) -> Dict[str, Any]:
    """Sentiment: Factor | Score code-block table — inline."""
    top = entries[0] if entries else {}
    factors = top.get("factors", {})
    rows = ["```", f"{'Factor':<12} Score", "─" * 20]
    for k in _SENTIMENT:
        v = factors.get(k)
        if v is not None:
            label = _FACTOR_LABEL.get(k, k.upper())
            rows.append(f"{label:<12} {v:.4f}")
    # Also include technical / macro in this table if present (keeps inline pair tight)
    for k in _TECHNICAL:
        v = factors.get(k)
        if v is not None:
            label = _FACTOR_LABEL.get(k, k.upper())
            rows.append(f"{label:<12} {v:.4f}")
    rows.append("```")
    return {
        "name":   "Sentiment",
        "value":  _truncate("\n".join(rows), 1020),
        "inline": True,
    }


# Keep snapshot/buy-list/factor-inline helpers for backward-compat with tests
def _snapshot_field(entries: List[Dict]) -> Dict[str, Any]:
    """Top-3 TL;DR in a fixed-width code block."""
    lines = ["```", f"{'#':<3} {'TICKER':<7} {'SCORE':<8} {'BAR':<10} SIGNAL"]
    lines.append("─" * 42)
    for i, e in enumerate(entries[:3], 1):
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        badge  = e.get("badge", "WATCHLIST")
        bar    = _score_bar(score, width=10)
        ceo    = " ⚡" if e.get("ceo_buy") else ""
        lines.append(f"{i:<3} {ticker:<7} {score:.4f}   {bar}  {badge}{ceo}")
    lines.append("```")
    return {
        "name":   "⚡  SNAPSHOT — Top 3 Today",
        "value":  "\n".join(lines),
        "inline": False,
    }


def _conviction_field(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Full-detail block for the #1 conviction pick (used by tests)."""
    ticker  = entry.get("ticker", "?")
    score   = float(entry.get("final_score", 0))
    badge   = entry.get("badge", "WATCHLIST")
    factors = entry.get("factors", {})
    sector  = entry.get("sector", "")
    cap     = entry.get("market_cap", 0)
    cap_str = f"  ·  {_fmt_cap(cap)}" if cap else ""
    ceo_tag = "  ·  **CEO BUY ⚡**" if entry.get("ceo_buy") else ""
    emoji   = _BADGE_EMOJI.get(badge, "⚪")
    bar     = _score_bar(score, width=10)

    lines = [
        f"**{ticker}**  {emoji}  `{score:.4f}`  —  {badge}{ceo_tag}",
        f"_{sector}{cap_str}_",
        f"`{bar}`",
        "",
        f"🏦  **Fundamental** — {_factor_group(factors, _FUNDAMENTAL)}",
        f"🌀  **Sentiment**   — {_factor_group(factors, _SENTIMENT)}",
        f"🐷  **Technical**   — {_factor_group(factors, _TECHNICAL)}",
    ]
    return {
        "name":   "🏆  TOP CONVICTION",
        "value":  _truncate("\n".join(lines)),
        "inline": False,
    }


def _buy_list_field(entries: List[Dict]) -> Dict[str, Any]:
    """Ranks 2-5 as a compact fixed-width code block table (used by tests)."""
    rows = ["```", f"{'#':<3} {'TICKER':<7} {'SCORE':<8} {'BAR':<10} SIGNAL"]
    rows.append("─" * 42)
    for i, e in enumerate(entries[1:5], 2):
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        badge  = e.get("badge", "WATCHLIST")
        bar    = _score_bar(score, width=10)
        ceo    = " ⚡" if e.get("ceo_buy") else ""
        rows.append(f"{i:<3} {ticker:<7} {score:.4f}   {bar}  {badge}{ceo}")
    rows.append("```")
    return {
        "name":   "📋  BUY LIST — Ranks 2–5",
        "value":  _truncate("\n".join(rows)),
        "inline": False,
    }


def _factor_inline_fields(entries: List[Dict]) -> List[Dict[str, Any]]:
    """Three inline fields showing grouped factor scores (used by tests)."""
    tickers = [e.get("ticker", "?") for e in entries[:5]]
    header  = "  ".join(f"{t:<6}" for t in tickers)

    def _col(key: str) -> str:
        label = _FACTOR_LABEL.get(key, key.upper())
        vals  = "  ".join(
            f"{float(e.get('factors', {}).get(key, 0)):.2f} " for e in entries[:5]
        )
        return f"`{label:<9}` {vals}"

    fundamental_lines = ["```", header, "─" * 38]
    for k in _FUNDAMENTAL:
        if any(k in (e.get("factors") or {}) for e in entries[:5]):
            fundamental_lines.append(_col(k))
    fundamental_lines.append("```")

    sentiment_lines = ["```", header, "─" * 38]
    for k in _SENTIMENT:
        if any(k in (e.get("factors") or {}) for e in entries[:5]):
            sentiment_lines.append(_col(k))
    sentiment_lines.append("```")

    technical_lines = ["```", header, "─" * 38]
    for k in _TECHNICAL:
        if any(k in (e.get("factors") or {}) for e in entries[:5]):
            technical_lines.append(_col(k))
    technical_lines.append("```")

    return [
        {"name": "🏦  FUNDAMENTAL", "value": _truncate("\n".join(fundamental_lines), 1020), "inline": True},
        {"name": "🌀  SENTIMENT",   "value": _truncate("\n".join(sentiment_lines),   1020), "inline": True},
        {"name": "🐷  TECHNICAL",   "value": _truncate("\n".join(technical_lines),   1020), "inline": True},
    ]


def _cap_tier_field(name: str, entries: List[Dict]) -> Dict[str, Any]:
    """Compact code-block table for a cap tier (mid or small)."""
    if not entries:
        return {"name": name, "value": "_No data._", "inline": True}
    rows = ["```", f"{'#':<3} {'TICKER':<7} SCORE"]
    rows.append("─" * 22)
    for i, e in enumerate(entries[:5], 1):
        score = float(e.get("final_score", 0))
        ceo   = "⚡" if e.get("ceo_buy") else " "
        rows.append(f"{i:<3} {e.get('ticker','?'):<7} {score:.4f} {ceo}")
    rows.append("```")
    return {
        "name":   name,
        "value":  _truncate("\n".join(rows), 1020),
        "inline": True,
    }


def _section_value(entries: List[Dict], max_chars: int = 1020) -> str:
    """Legacy helper — kept for satellite fields which use a different format."""
    if not entries:
        return "_No data available for this universe tier._"
    blocks = []
    for i, entry in enumerate(entries, 1):
        ticker  = entry.get("ticker", "?")
        score   = float(entry.get("final_score", 0))
        badge   = entry.get("badge", "WATCHLIST")
        emoji   = _BADGE_EMOJI.get(badge, "⚪")
        ceo_tag = " ⚡" if entry.get("ceo_buy") else ""
        blocks.append(f"**{i}. {ticker}**{ceo_tag}  {emoji}  `{score:.4f}`")
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
    run_id       = top_lists.get("source_run_id", top_lists.get("run_id", ""))
    ticker_count = top_lists.get("ticker_count", 0)
    weights      = top_lists.get("weights", {})
    top_buys     = top_lists.get("top_buys") or []

    age_h = _data_age_hours(generated_at)
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%B %d, %Y")
    except Exception:
        date_str = generated_at[:10] or "—"

    color = _pick_color(top_lists)

    # ── Weight redistribution check ────────────────────────────────────────
    _NOMINAL_WEIGHTS = {
        "edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12,
    }
    weights_redistributed = bool(weights) and any(
        abs(weights.get(k, 0) - _NOMINAL_WEIGHTS.get(k, 0)) > 0.001
        for k in _NOMINAL_WEIGHTS
    )

    # ── VIX line ───────────────────────────────────────────────────────────
    vix_val = top_lists.get("vix")
    vix_str = f"  ·  VIX `{vix_val:.1f}`" if vix_val is not None else ""

    # ── Alerts (stale data / kill-switch) ─────────────────────────────────
    alerts: List[str] = []
    if age_h is not None and age_h > _STALE_HOURS:
        color = _COLOR_RED
        alerts.append(
            f"⚠️  DATA IS {age_h:.0f}h OLD — pipeline may have failed. "
            "Check edgar_3x on GitHub Actions."
        )
    kill_switch = top_lists.get("kill_switch", False)
    if kill_switch:
        color = _COLOR_RED
        vix_mult = top_lists.get("vix_multiplier", 1.0)
        vix_note = f"VIX {vix_val:.1f}  ·  " if vix_val is not None else ""
        alerts.append(
            f"⛔  MACRO KILL-SWITCH ACTIVE  —  {vix_note}"
            f"scores dampened ×{vix_mult:.2f}.  Do NOT act on HIGH BUY signals."
        )

    alert_block = ("\n" + "\n".join(f"```diff\n- {a}\n```" for a in alerts)) if alerts else ""

    # ── Description: run metadata ──────────────────────────────────────────
    feed_note = "  ⚠️ feed down — redistributed" if weights_redistributed else ""
    description = (
        f"**{ticker_count} tickers scored**{vix_str}\n"
        f"Pipeline: EDGAR-first{feed_note}"
        f"{alert_block}"
    )

    # ── Fields: Signal → Conviction → Evidence ─────────────────────────────
    fields: List[Dict[str, Any]] = []

    if top_buys:
        # 1. Top Conviction — non-inline, sits above everything
        fields.append(_top_conviction_field(top_buys))
        # 2. Evidence Matrix — Fundamentals + Sentiment inline side-by-side
        fields.append(_fundamentals_field(top_buys))
        fields.append(_sentiment_field(top_buys))

    # 3. Opportunities — Mid Caps (non-inline for readability on mobile)
    mid_caps   = top_lists.get("mid_caps") or []
    small_caps = top_lists.get("small_caps") or []
    if mid_caps:
        fields.append(_cap_tier_field("Mid Caps  ($2B–$10B)", mid_caps))
    if small_caps:
        fields.append(_cap_tier_field("Small Caps  (<$2B)", small_caps))

    # 4. Satellite fields (optional)
    try:
        if satellite and isinstance(satellite, dict):
            month_label = satellite.get("month", "")
            cyclicals   = satellite.get("cyclicals") or []
            cannibals   = satellite.get("cannibals") or []

            if cyclicals:
                rows = ["```", f"{'#':<3} {'TICKER':<7} {'WIN%':<7} {'MEDIAN':>7}  YRS"]
                rows.append("─" * 34)
                for i, c in enumerate(cyclicals, 1):
                    wr  = f"{c['win_rate']:.0%}"
                    med = f"{c['median_return']:+.1%}"
                    yr  = c.get("years", "?")
                    rows.append(f"{i:<3} {c['ticker']:<7} {wr:<7} {med:>7}  {yr}")
                rows.append("```")
                fields.append({
                    "name":   f"Seasonal Cyclicals — {month_label}",
                    "value":  _truncate("\n".join(rows)),
                    "inline": True,
                })

            if cannibals:
                rows = ["```", f"{'#':<3} {'TICKER':<7} {'YIELD':<7} {'P/E':>5}  P/52wL"]
                rows.append("─" * 34)
                for i, c in enumerate(cannibals, 1):
                    yld = f"{c.get('buyback_yield', 0):.1%}"
                    pe  = f"{c.get('pe', 0):.1f}"
                    pvl = f"{c.get('price_vs_52w_low', 0):.2f}×"
                    rows.append(f"{i:<3} {c['ticker']:<7} {yld:<7} {pe:>5}  {pvl}")
                rows.append("```")
                fields.append({
                    "name":   "Share Cannibals — Buyback Yield",
                    "value":  _truncate("\n".join(rows)),
                    "inline": True,
                })
    except Exception as exc:
        log.warning("satellite embed fields skipped due to error: %s", exc)

    embed = {
        "title":       f"[REGIME TRADER] Daily Market Report — {date_str}",
        "description": description,
        "color":       color,
        "fields":      fields,
        "footer":      {
            "text": f"Run: {run_id}  |  Pipeline: EDGAR-first  |  Scores: [0,1]",
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
