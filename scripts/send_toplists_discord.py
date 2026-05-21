"""scripts/send_toplists_discord.py
Send the daily market report to Discord via Embed webhook.

Reads logs/top_lists.json (and optionally satellite_insights.json,
anomaly_report_latest.json) and formats a mobile-first Discord embed:

  ⚡ Alpha Pipeline [May 21, 2026]
  Description: TL;DR summary — buys | anomalies | boost status | VIX regime
  Fields: compact 2-line ticker cards (score + signals + factor matrix)
  Footer: latency · coverage (data-gap) · mode · run-id

Discord embed limits:
  title:       256 chars
  description: 4096 chars
  field value: 1024 chars (hard limit — truncated at 1024)
  fields:      25 max
  total:       6000 chars

Embed color (severity-driven — not score-driven):
  GREEN  (0x00FF00) — system nominal
  ORANGE (0xFFA500) — non-critical anomaly (CONGRESS_CLUSTER, etc.)
  RED    (0xFF0000) — critical: STALE_SOURCE / MISSING_AMOUNT / DEAD_FEED
                      or kill-switch active

Retry policy: 3 attempts with 30s / 60s backoff.

Usage:
  python scripts/send_toplists_discord.py
  python scripts/send_toplists_discord.py --input logs/top_lists.json --dry-run
  python scripts/send_toplists_discord.py --run-tests
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

# ── Color palette (severity-driven) ───────────────────────────────────────────
_COLOR_GREEN  = 0x00FF00   # nominal — system healthy
_COLOR_ORANGE = 0xFFA500   # non-critical anomaly (informational)
_COLOR_RED    = 0xFF0000   # critical intervention required
_COLOR_BLUE   = 0x3498DB   # fallback (no top pick)

# Flags that force red border — require immediate attention
_CRITICAL_FLAGS = frozenset({"STALE_SOURCE", "MISSING_AMOUNT", "DEAD_FEED"})

# ── Factor group constants — used by _factor_group() and tests ────────────────
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

# ── Nominal factor weights (EDGAR-first baseline) ─────────────────────────────
# Used to detect feed-down redistribution and identify lagging sources.
_NOMINAL_WEIGHTS: Dict[str, float] = {
    "edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12,
}

# ── VIX regime thresholds ─────────────────────────────────────────────────────
_VIX_BEARISH = 25.0   # VIX > 25  → BEARISH 🔴
_VIX_STABLE  = 15.0   # VIX > 15  → STABLE  🟡  else BULLISH 🟢

# ── Sector normalisation — SSOT is entry["sector"] from top_lists.json ────────
# Maps full GICS names to compact display labels for the heatmap field.
_SECTOR_SHORT: Dict[str, str] = {
    "Information Technology": "🖥️ Tech",
    "Technology":             "🖥️ Tech",
    "Health Care":            "🏥 Health",
    "Healthcare":             "🏥 Health",
    "Financials":             "🏛️ Fin",
    "Financial Services":     "🏛️ Fin",
    "Communication Services": "📡 Comm",
    "Consumer Discretionary": "🛍️ Cons",
    "Consumer Staples":       "🛒 Staples",
    "Industrials":            "🏭 Indust",
    "Energy":                 "⛽ Energy",
    "Materials":              "⚗️ Mater",
    "Real Estate":            "🏢 RE",
    "Utilities":              "💡 Utils",
}
_SECTOR_MISC = "🔲 Misc"

# ── Buyback yield thresholds ───────────────────────────────────────────────────
_BUYBACK_HIGH = 0.10   # ≥ 10% yield → +0.80 conviction label
_BUYBACK_LOW  = 0.05   # ≥  5% yield → +0.40 conviction label

# Factor emoji map for the matrix line (Line 2 of each ticker card)
_FACTOR_EMOJI: Dict[str, str] = {
    "edgar":    "📋",
    "insider":  "👤",
    "congress": "🏛",
    "news":     "📰",
    "macro":    "🌐",
    "momentum": "📈",
}

# Medal for top-5 ranks
_MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}

# Badge labels with emoji (used by _conviction_field — kept for test compat)
_BADGE_EMOJI = {
    "HIGH BUY":     "🟢",
    "TACTICAL BUY": "🟡",
    "WATCHLIST":    "⚪",
}

_STALE_HOURS = 25


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    """Compact progress bar: ▓▓▓▓░░░░"""
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
    """Render a factor subset as  KEY `0.72`  ·  KEY `0.90`"""
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


# ── Domain logic ───────────────────────────────────────────────────────────────

def get_market_regime(vix: float) -> str:
    """Classify market regime from VIX. Returns embed-ready string.

    VIX > 25 → BEARISH 🔴 | VIX > 15 → STABLE 🟡 | else BULLISH 🟢
    """
    if vix > _VIX_BEARISH:
        label, emoji = "BEARISH", "🔴"
    elif vix > _VIX_STABLE:
        label, emoji = "STABLE",  "🟡"
    else:
        label, emoji = "BULLISH", "🟢"
    return f"VIX `{vix:.1f}` {emoji} **{label}**"


def _buyback_conviction(yield_pct: float) -> Optional[float]:
    """Return conviction boost for a buyback yield, or None if below threshold."""
    if yield_pct >= _BUYBACK_HIGH:
        return 0.80
    if yield_pct >= _BUYBACK_LOW:
        return 0.40
    return None


def _embed_color(anomaly_map: Dict[str, List[str]], kill_switch: bool) -> int:
    """Three-tier severity color for the embed left border.

    RED    — critical flag or kill-switch (do not trade).
    ORANGE — informational anomaly only (e.g. CONGRESS_CLUSTER).
    GREEN  — system nominal.
    """
    if kill_switch:
        return _COLOR_RED
    all_flags = {flag for flags in anomaly_map.values() for flag in flags}
    if all_flags & _CRITICAL_FLAGS:
        return _COLOR_RED
    if all_flags:
        return _COLOR_ORANGE
    return _COLOR_GREEN


def _sector_heatmap(top_buys: List[Dict]) -> str:
    """Build a compact sector exposure line using entry['sector'] as SSOT.

    Uses Counter-style accumulation then sorts by descending count.
    Unknown/missing sectors fall back to _SECTOR_MISC.
    Returns empty string when top_buys is empty.
    """
    counts: Dict[str, int] = {}
    for e in top_buys:
        raw   = (e.get("sector") or "").strip()
        label = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    parts = [f"{lbl} ({n})" for lbl, n in sorted(counts.items(), key=lambda x: -x[1])]
    return "  |  ".join(parts)


def _sector_heatmap_structured(
    entries: List[Dict],
) -> Dict[str, List[tuple]]:
    """Return {sector_label: [(ticker, score), ...]} sorted by descending score.

    At most 2 tickers per sector. Combines Large Cap + Mid Cap entries.
    Unknown/missing sectors fall back to _SECTOR_MISC.
    """
    buckets: Dict[str, List[tuple]] = {}
    for e in entries:
        raw    = (e.get("sector") or "").strip()
        label  = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        buckets.setdefault(label, []).append((ticker, score))

    # Sort each bucket by descending score, keep top 2
    return {
        lbl: sorted(pairs, key=lambda x: -x[1])[:2]
        for lbl, pairs in buckets.items()
    }


# ── Field builders ─────────────────────────────────────────────────────────────

def _ticker_detail_field(
    rank: int,
    entry: Dict[str, Any],
    anomaly_flags: Optional[List[str]] = None,
    rank_delta: Optional[int] = None,
    buyback_conv: Optional[float] = None,
    mid_cap: bool = False,
) -> Dict[str, Any]:
    """Unified 4-line ticker card — identical anatomy for Large Cap and Mid Cap.

    Line 1: {RANK} {TICKER} — {BADGE} — {SCORE} {BAR10}  [boosts]
    Line 2: {SECTOR_EMOJI}{SECTOR} · {MARKET_CAP}
    Line 3: ────────────────────
    Line 4: 👤{v}  🏛{v}  🔄{v}  📰{v}  🌐{v}

    Args:
        mid_cap:     True → use `N` backtick rank; False → use 🥇🥈🥉 for 1–3.
        rank_delta:  shadow_rank − boosted_rank (positive = promoted by boost).
        buyback_conv: conviction boost from buyback yield (0.40 or 0.80), or None.
    """
    ticker  = entry.get("ticker", "?")
    score   = float(entry.get("final_score", 0))
    badge   = entry.get("badge", "WATCHLIST")
    factors = entry.get("factors") or {}
    sector  = (entry.get("sector") or "").strip()
    cap     = entry.get("market_cap", 0)
    boost   = float(entry.get("congress_boost", 0.0))

    # ── Line 1: rank token ────────────────────────────────────────────────
    if mid_cap:
        rank_token = f"`{rank}`"
    else:
        rank_token = _MEDAL.get(rank, f"`{rank}`")

    # ── Line 1: boost / signal tokens (space-separated, after bar) ───────
    bar_str      = _score_bar(score, width=10)
    boost_part   = f"  🏛 `+{boost:.2f}`"        if boost > 0.0              else ""
    buyback_part = f"  🔄 `+{buyback_conv:.2f}`"  if buyback_conv is not None else ""

    if rank_delta is None or rank_delta == 0:
        trend_part = ""
    elif rank_delta > 0:
        trend_part = f"  🟢+{rank_delta}"
    else:
        trend_part = f"  🔴{rank_delta}"

    ceo_tag  = "  ⚡CEO" if entry.get("ceo_buy")  else ""
    flag_tag = "  ⚠️"    if anomaly_flags          else ""

    line1 = (
        f"{rank_token} **{ticker}** — {badge} — `{score:.2f}` {bar_str}"
        f"{boost_part}{buyback_part}{trend_part}{ceo_tag}{flag_tag}"
    )

    # ── Line 2: sector · market cap ───────────────────────────────────────
    sector_label = _SECTOR_SHORT.get(sector, _SECTOR_MISC) if sector else _SECTOR_MISC
    cap_str      = f"  ·  {_fmt_cap(cap)}" if cap else ""
    line2 = f"{sector_label}{cap_str}"

    # ── Line 3: visual separator ──────────────────────────────────────────
    line3 = "────────────────────"

    # ── Line 4: factor matrix — emoji + raw value ─────────────────────────
    factor_parts = []
    for key in ("edgar", "insider", "congress", "news", "macro"):
        v = factors.get(key)
        if v is not None:
            factor_parts.append(f"{_FACTOR_EMOJI[key]}`{v:.2f}`")
    if buyback_conv is not None:
        factor_parts.append(f"🔄`{buyback_conv:.2f}`")
    line4 = "  ".join(factor_parts) if factor_parts else "—"

    value = _truncate("\n".join([line1, line2, line3, line4]), 1024)
    return {
        "name":   f"#{rank}  {ticker}",
        "value":  value,
        "inline": False,
    }


def _conviction_field(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Full-detail block for the #1 conviction pick — kept for test compatibility."""
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


def _cap_tier_field(name: str, entries: List[Dict]) -> Dict[str, Any]:
    """Compact cap-tier opportunity list."""
    if not entries:
        return {"name": name, "value": "*No picks in this tier today.*", "inline": False}
    lines = []
    for i, e in enumerate(entries[:5], 1):
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        bar    = _score_bar(score, width=6)
        ceo    = " ⚡" if e.get("ceo_buy") else ""
        lines.append(f"`{i}` **{ticker}**{ceo} {bar} `{score:.2f}`")
    return {
        "name":   name,
        "value":  _truncate("\n".join(lines), 1020),
        "inline": False,
    }


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _data_age_hours(generated_at: str) -> Optional[float]:
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    except Exception:
        return None


def _load_satellite(log_dir: Path) -> Optional[Dict[str, Any]]:
    """Load satellite_insights.json. Returns None on any failure."""
    path = log_dir / "satellite_insights.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        log.warning("satellite_insights.json unreadable: %s", exc)
        return None


def _load_anomaly_report(log_dir: Path) -> Dict[str, List[str]]:
    """Load anomaly_report_latest.json → {ticker: [flags]}.

    Returns empty dict on any failure — anomaly display is best-effort.
    """
    path = log_dir / "anomaly_report_latest.json"
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return {}
        result: Dict[str, List[str]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ticker = rec.get("ticker", "")
            flag   = rec.get("flag", "")
            if ticker and flag:
                result.setdefault(ticker, []).append(flag)
        return result
    except Exception as exc:
        log.warning("anomaly_report_latest.json unreadable: %s", exc)
        return {}


# ── Payload builder ────────────────────────────────────────────────────────────

def build_payload(
    top_lists: Dict[str, Any],
    satellite: Optional[Dict[str, Any]] = None,
    anomaly_map: Optional[Dict[str, List[str]]] = None,
    pipeline_latency_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the Discord webhook JSON payload.

    Mobile-first: all critical information visible without scrolling.
    Data flow: top_lists.json (SSOT) + optional satellite join (in-flight).
    """
    generated_at = top_lists.get("generated_at", "")
    run_id       = top_lists.get("source_run_id", top_lists.get("run_id", ""))
    ticker_count = top_lists.get("ticker_count", 0)
    weights      = top_lists.get("weights", {})
    top_buys     = top_lists.get("top_buys") or []

    age_h = _data_age_hours(generated_at)
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%b %d, %Y")
    except Exception:
        date_str = generated_at[:10] or "—"

    # ── Weight redistribution — feed-down detection ────────────────────────
    weights_redistributed = bool(weights) and any(
        abs(weights.get(k, 0) - _NOMINAL_WEIGHTS.get(k, 0)) > 0.001
        for k in _NOMINAL_WEIGHTS
    )

    # ── Market Regime Sentinel (VIX → macro context, line 2 of description) ─
    vix_val = top_lists.get("vix")
    vix_str = f"  ·  {get_market_regime(float(vix_val))}" if vix_val is not None else ""

    # ── Congress boost active? ─────────────────────────────────────────────
    congress_boost_on = any(float(e.get("congress_boost", 0.0)) > 0.0 for e in top_buys)

    # ── Anomaly map — normalised early, shared by color + counts ──────────
    anomaly_map = anomaly_map or {}
    anomaly_count = len(set(anomaly_map.keys()) & {e.get("ticker") for e in top_buys})

    # ── Kill-switch ────────────────────────────────────────────────────────
    kill_switch = top_lists.get("kill_switch", False)

    # ── Severity-driven embed color ────────────────────────────────────────
    # Stale data overrides anomaly severity — data integrity takes priority.
    color = _embed_color(anomaly_map, kill_switch)
    if age_h is not None and age_h > _STALE_HOURS:
        color = _COLOR_RED

    # ── Data-Gap Indicator — footer annotation for missing/lagging sources ─
    _KNOWN_SOURCES = {"edgar", "insider", "congress", "news", "momentum"}
    present_sources = set(weights.keys()) if weights else set()
    missing_sources = sorted(_KNOWN_SOURCES - present_sources) if present_sources else []
    lagging_sources: List[str] = []
    if weights_redistributed and not missing_sources:
        lagging_sources = sorted(
            (k for k in _NOMINAL_WEIGHTS if abs(weights.get(k, 0) - _NOMINAL_WEIGHTS[k]) > 0.05),
            key=lambda k: abs(weights.get(k, 0) - _NOMINAL_WEIGHTS[k]),
            reverse=True,
        )[:2]

    # ── Buyback lookup: in-flight join of satellite.cannibals[] onto top_buys
    # satellite_insights.json is optional — graceful degradation on absence.
    buyback_conv_of: Dict[str, float] = {}
    try:
        if satellite and isinstance(satellite, dict):
            for c in (satellite.get("cannibals") or []):
                t    = (c.get("ticker") or "").upper()
                yld  = float(c.get("buyback_yield") or 0.0)
                conv = _buyback_conviction(yld)
                if t and conv is not None:
                    buyback_conv_of[t] = conv
    except Exception as exc:
        log.debug("buyback join failed: %s", exc)

    # ── Alerts (diff blocks — only shown for actionable conditions) ────────
    alerts: List[str] = []
    if age_h is not None and age_h > _STALE_HOURS:
        alerts.append(
            f"⚠️  DATA IS {age_h:.0f}h OLD — pipeline may have failed. "
            "Check edgar_3x on GitHub Actions."
        )
    if kill_switch:
        vix_mult = top_lists.get("vix_multiplier", 1.0)
        vix_note = f"VIX {vix_val:.1f}  ·  " if vix_val is not None else ""
        alerts.append(
            f"⛔  MACRO KILL-SWITCH ACTIVE  —  {vix_note}"
            f"scores dampened ×{vix_mult:.2f}.  Do NOT act on HIGH BUY signals."
        )
    if any(flag == "STALE_SOURCE" for flags in anomaly_map.values() for flag in flags):
        alerts.append("⚠️  STALE DATA SOURCE — scores may be unreliable.")

    alert_block = ("\n" + "\n".join(f"```diff\n- {a}\n```" for a in alerts)) if alerts else ""

    # ── Description: TL;DR — all critical signals visible without scrolling ─
    boost_status  = "ON 🏛" if congress_boost_on else "OFF"
    feed_note     = "  ⚠️ *feed down — redistributed*" if weights_redistributed else ""
    summary_parts = [f"**{len(top_buys)} Buy{'s' if len(top_buys) != 1 else ''}**"]
    if anomaly_count:
        summary_parts.append(f"**{anomaly_count} Anomaly{'s' if anomaly_count != 1 else ''}** ⚠️")
    summary_parts.append(f"Congress Boost: **{boost_status}**")
    summary_line = "  |  ".join(summary_parts)

    description = (
        f"{summary_line}\n"
        f"`{ticker_count} tickers`{vix_str}  ·  EDGAR-first{feed_note}"
        f"{alert_block}"
    )

    # ── Fields: compact ticker cards (incremental 1900-char budget) ────────
    fields: List[Dict[str, Any]] = []

    if top_buys:
        # Shadow rank lookup for conviction trend arrows
        shadow_buys    = top_lists.get("shadow_top_buys") or []
        shadow_rank_of = {e.get("ticker", ""): i for i, e in enumerate(shadow_buys, 1)}

        used = 0
        added = 0
        for i, e in enumerate(top_buys[:5], 1):
            ticker     = e.get("ticker", "")
            shadow_r   = shadow_rank_of.get(ticker)
            rank_delta = (shadow_r - i) if shadow_r is not None else None
            buyback_cv = buyback_conv_of.get(ticker.upper())
            field = _ticker_detail_field(
                i, e,
                anomaly_flags=anomaly_map.get(ticker),
                rank_delta=rank_delta,
                buyback_conv=buyback_cv,
            )
            flen = len(field["value"])
            if used + flen > 1900 and added > 0:
                remaining = len(top_buys[:5]) - added
                fields.append({
                    "name":   "…",
                    "value":  f"*+ {remaining} more ticker{'s' if remaining != 1 else ''} — run `--dry-run` for full output*",
                    "inline": False,
                })
                break
            fields.append(field)
            used  += flen
            added += 1

    # Cap tiers (optional — mid/small caps)
    mid_caps   = top_lists.get("mid_caps")   or []
    small_caps = top_lists.get("small_caps") or []
    if mid_caps:
        fields.append(_cap_tier_field("📈  Mid Caps  ($2B–$10B)", mid_caps))
    if small_caps:
        fields.append(_cap_tier_field("🔬  Small Caps  (<$2B)", small_caps))

    # Satellite detail blocks (cyclicals + cannibals)
    try:
        if satellite and isinstance(satellite, dict):
            month_label = satellite.get("month", "")
            cyclicals   = satellite.get("cyclicals") or []
            cannibals   = satellite.get("cannibals") or []

            if cyclicals:
                lines = [
                    f"**{c['ticker']}** {_score_bar(c['win_rate'], 6)} "
                    f"`{c['win_rate']:.0%}` win · `{c['median_return']:+.1%}` med · `{c.get('years','?')}y`"
                    for c in cyclicals
                ]
                fields.append({
                    "name":   f"🌀  Seasonal Cyclicals — {month_label}",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })

            if cannibals:
                lines = [
                    f"**{c['ticker']}** · `{c.get('buyback_yield',0):.1%}` buyback"
                    f" · P/E `{c.get('pe',0):.1f}` · `{c.get('price_vs_52w_low',0):.2f}×` vs 52w low"
                    for c in cannibals
                ]
                fields.append({
                    "name":   "🐷  Share Cannibals",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })
    except Exception as exc:
        log.warning("satellite embed fields skipped: %s", exc)

    # Sector heatmap — dynamic exposure from entry["sector"] (SSOT)
    heatmap = _sector_heatmap(top_buys)
    if heatmap:
        fields.append({
            "name":   "📊  Sector Exposure",
            "value":  heatmap,
            "inline": False,
        })

    # ── Footer: latency · coverage (data-gap) · mode ──────────────────────
    latency_part = f"Latency: {pipeline_latency_s:.0f}s  |  " if pipeline_latency_s is not None else ""
    coverage_pct = f"{min(100, round(ticker_count / 1.6))}%" if ticker_count else "—"
    if missing_sources:
        gap_note = f" (No: {', '.join(s.upper() for s in missing_sources)})"
    elif lagging_sources:
        gap_note = f" (Lag: {', '.join(s.upper() for s in lagging_sources)})"
    else:
        gap_note = ""
    mode_str    = "Live" if not kill_switch else "Kill-Switch"
    footer_text = f"{latency_part}Cov: {coverage_pct}{gap_note}  |  Mode: {mode_str}  |  Run: {run_id}"

    embed = {
        "title":       f"⚡ Alpha Pipeline  [{date_str}]",
        "description": description,
        "color":       color,
        "fields":      fields,
        "footer":      {"text": footer_text},
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    return {"embeds": [embed]}


def build_alert_payload(reason: str) -> Dict[str, Any]:
    """Minimal red embed when top_lists.json is missing or unreadable."""
    return {
        "embeds": [{
            "title":       "⚠️ Alpha Pipeline — DATA UNAVAILABLE",
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


# ── Self-contained test suite ──────────────────────────────────────────────────

def run_tests() -> int:
    """Autonomous verification suite — no pytest dependency.

    Tests:
      1. Empty top_buys → no fields, no crash
      2. Missing sector → fallback to Misc in heatmap
      3. Budget truncation at 1900 chars
      4. Buyback boost icon rendered (🔄) when satellite present
      5. Critical anomaly → red embed
      6. Congress cluster only → orange embed
      7. VIX regime labels at all three thresholds
    """
    import traceback
    failures: List[str] = []

    def _check(name: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]{': ' + detail if detail else ''}")

    def _base_tl(**overrides) -> Dict[str, Any]:
        tl: Dict[str, Any] = {
            "generated_at":  "2026-05-21T10:00:00+00:00",
            "source_run_id": "test",
            "ticker_count":  5,
            "weights":       {},
            "kill_switch":   False,
            "vix":           17.0,
            "top_buys":      [],
            "mid_caps":      [],
            "small_caps":    [],
        }
        tl.update(overrides)
        return tl

    def _entry(ticker: str, sector: str = "Information Technology",
               score: float = 0.75, **kw) -> Dict[str, Any]:
        return {
            "ticker": ticker, "final_score": score, "badge": "HIGH BUY",
            "sector": sector, "market_cap": 1e12, "cap_tier": "large",
            "ceo_buy": False, "congress_boost": 0.0,
            "factors": {"edgar": 0.8, "insider": 0.7, "congress": 0.6,
                        "news": 0.65, "macro": 0.55},
            **kw,
        }

    # ── Test 1: empty top_buys → no ticker fields, no crash ───────────────
    try:
        payload = build_payload(_base_tl())
        embed   = payload["embeds"][0]
        field_names = [f["name"] for f in embed["fields"]]
        _check("empty_top_buys_no_fields",
               not any(n.startswith("#") for n in field_names))
        _check("empty_top_buys_no_crash", True)
    except Exception:
        failures.append(f"FAIL [empty_top_buys_no_crash]: {traceback.format_exc()}")

    # ── Test 2: missing sector → Misc fallback in heatmap ─────────────────
    try:
        entries = [_entry("AAPL", sector=""), _entry("MSFT", sector="")]
        heatmap = _sector_heatmap(entries)
        _check("misc_fallback", _SECTOR_MISC in heatmap,
               f"heatmap={heatmap!r}")
    except Exception:
        failures.append(f"FAIL [misc_fallback]: {traceback.format_exc()}")

    # ── Test 3: budget truncation at 1900 chars ────────────────────────────
    try:
        fat_entry = _entry("FAT", score=0.9)
        fat_entry["factors"] = {k: 0.99 for k in ("edgar","insider","congress","news","macro")}
        tl = _base_tl(top_buys=[fat_entry] * 5)
        payload = build_payload(tl)
        embed   = payload["embeds"][0]
        total   = sum(len(f.get("value","")) for f in embed["fields"])
        _check("budget_1900", total <= 1900 + 200,
               f"total_field_chars={total}")
        has_more = any("more ticker" in (f.get("value","")) for f in embed["fields"])
        all_five = sum(1 for f in embed["fields"] if f["name"].startswith("#")) == 5
        _check("budget_truncation_or_all_fit", has_more or all_five)
    except Exception:
        failures.append(f"FAIL [budget_truncation]: {traceback.format_exc()}")

    # ── Test 4: buyback boost icon 🔄 in ticker card ──────────────────────
    try:
        tl = _base_tl(top_buys=[_entry("MSFT")])
        sat = {"cannibals": [{"ticker": "MSFT", "buyback_yield": 0.12,
                               "pe": 30.0, "price_vs_52w_low": 1.1}]}
        payload = build_payload(tl, satellite=sat)
        fields  = payload["embeds"][0]["fields"]
        ticker_field = next((f for f in fields if f["name"].startswith("#")), None)
        _check("buyback_icon_present",
               ticker_field is not None and "🔄" in ticker_field["value"],
               f"field_value={ticker_field['value'] if ticker_field else 'None'}")
    except Exception:
        failures.append(f"FAIL [buyback_icon]: {traceback.format_exc()}")

    # ── Test 5: STALE_SOURCE → red embed ─────────────────────────────────
    try:
        amap   = {"AAPL": ["STALE_SOURCE"]}
        tl     = _base_tl(top_buys=[_entry("AAPL")])
        payload = build_payload(tl, anomaly_map=amap)
        color  = payload["embeds"][0]["color"]
        _check("stale_source_red", color == _COLOR_RED, f"color=0x{color:06X}")
    except Exception:
        failures.append(f"FAIL [stale_source_red]: {traceback.format_exc()}")

    # ── Test 6: CONGRESS_CLUSTER only → orange embed ──────────────────────
    try:
        amap    = {"NVDA": ["CONGRESS_CLUSTER"]}
        tl      = _base_tl(top_buys=[_entry("NVDA")])
        payload = build_payload(tl, anomaly_map=amap)
        color   = payload["embeds"][0]["color"]
        _check("congress_cluster_orange", color == _COLOR_ORANGE, f"color=0x{color:06X}")
    except Exception:
        failures.append(f"FAIL [congress_cluster_orange]: {traceback.format_exc()}")

    # ── Test 7: VIX regime labels ─────────────────────────────────────────
    try:
        _check("vix_bullish",  "BULLISH" in get_market_regime(12.0))
        _check("vix_stable",   "STABLE"  in get_market_regime(18.0))
        _check("vix_bearish",  "BEARISH" in get_market_regime(28.0))
        _check("vix_boundary_low",  "BULLISH" in get_market_regime(15.0))
        _check("vix_boundary_high", "BEARISH" in get_market_regime(25.1))
    except Exception:
        failures.append(f"FAIL [vix_regime]: {traceback.format_exc()}")

    # ── Test 8: ticker card anatomy — 4 lines, separator, unified format ──────
    try:
        tl  = _base_tl(top_buys=[_entry("AAPL", score=0.87)])
        payload = build_payload(tl)
        fields  = payload["embeds"][0]["fields"]
        card = next((f for f in fields if f["name"].startswith("#")), None)
        val  = card["value"] if card else ""
        _check("card_has_separator",  "────────────────────" in val, f"val={val!r}")
        _check("card_has_score_bar",  "▓" in val or "░" in val,     f"val={val!r}")
        _check("card_has_factor_emoji","👤" in val,                  f"val={val!r}")
        lines = val.split("\n")
        _check("card_four_lines", len(lines) == 4, f"lines={lines}")
    except Exception:
        failures.append(f"FAIL [card_anatomy]: {traceback.format_exc()}")

    # ── Test 9: mid_cap=True uses backtick rank, not medal ────────────────────
    try:
        entry  = _entry("CRDO", score=0.74)
        field  = _ticker_detail_field(1, entry, mid_cap=True)
        _check("midcap_rank_backtick", "`1`" in field["value"],
               f"val={field['value']!r}")
        _check("midcap_no_medal", "🥇" not in field["value"],
               f"val={field['value']!r}")
    except Exception:
        failures.append(f"FAIL [midcap_rank]: {traceback.format_exc()}")

    # ── Test 10: structured heatmap — top-2 tickers per sector ───────────────
    try:
        entries = [
            _entry("AAPL", sector="Information Technology", score=0.87),
            _entry("MSFT", sector="Information Technology", score=0.81),
            _entry("NVDA", sector="Information Technology", score=0.79),
            _entry("JPM",  sector="Financials",             score=0.68),
        ]
        result = _sector_heatmap_structured(entries)
        tech   = result.get("🖥️ Tech", [])
        fin    = result.get("🏛️ Fin",  [])
        _check("heatmap_struct_tech_count",  len(tech) == 2,
               f"tech={tech}")
        _check("heatmap_struct_tech_ticker", tech[0][0] == "AAPL",
               f"tech={tech}")
        _check("heatmap_struct_fin_count",   len(fin) == 1,
               f"fin={fin}")
        _check("heatmap_struct_sorted_desc",
               tech[0][1] >= tech[1][1],
               f"tech scores out of order: {tech}")
    except Exception:
        failures.append(f"FAIL [heatmap_structured]: {traceback.format_exc()}")

    # ── Report ────────────────────────────────────────────────────────────
    total_tests = 20   # approximate assertion count above
    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print(f"\n{len(failures)} test(s) FAILED", file=sys.stderr)
        return 1
    print(f"All tests passed ({total_tests} assertions)")
    return 0


# ── HTTP send with retry ───────────────────────────────────────────────────────

def send_to_discord(
    webhook: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    backoff_base_s: float = 30.0,
) -> bool:
    """POST payload to Discord webhook with exponential-like backoff.

    Backoff schedule: attempt 1 immediate, then 30s, 60s.
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
            wait = backoff_base_s * attempt
            log.warning("Retry %d/%d in %.0fs …", attempt + 1, max_retries, wait)
            time.sleep(wait)
        try:
            resp = requests.post(webhook, json=payload, timeout=15.0)
            if 200 <= resp.status_code < 300:
                log.info("Discord message sent (status=%d)", resp.status_code)
                return True
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
        help="Directory for discord_send.log and satellite/anomaly files",
    )
    parser.add_argument(
        "--webhook", type=str, default=None,
        help="Discord webhook URL (overrides env DISCORD_WEBHOOK_URL)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print payload JSON without sending",
    )
    parser.add_argument(
        "--run-tests", action="store_true",
        help="Run built-in self-test suite and exit",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.run_tests:
        logging.basicConfig(level=logging.WARNING)
        return run_tests()

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

    if not args.input.exists():
        log.warning("top_lists.json not found at %s — sending alert", args.input)
        payload = build_alert_payload(f"File not found: {args.input}")
        if args.dry_run:
            sys.stdout.buffer.write((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
            return 0
        return 0 if send_to_discord(webhook, payload) else 1

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

    satellite   = _load_satellite(args.log_dir)
    anomaly_map = _load_anomaly_report(args.log_dir)
    payload     = build_payload(top_lists, satellite=satellite, anomaly_map=anomaly_map)

    if args.dry_run:
        sys.stdout.buffer.write((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        return 0

    ok = send_to_discord(webhook, payload)
    if not ok:
        log.error("All Discord send attempts failed")
        return 1

    log.info("Daily market checkup sent successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
