"""scripts/send_toplists_discord.py
Send the daily market report to Discord via Embed webhook.

Reads logs/intel_source_status.json (7-factor schema, SSOT) and formats
a mobile-first Discord embed with institutional desk format:

  ⚡ Alpha Pipeline [May 25, 2026]
  Description: TL;DR — VIX regime | alerts
  Fields: 7-factor ticker cards (score + bar + percentile + badge + catalyst)
          ACTION TODAY (buy/watch recommendations)
          PIPELINE HEALTH (orthogonality + dead factors + latency)

Discord embed limits:
  title:       256 chars
  description: 4096 chars
  field value: 1024 chars (hard limit — truncated at 1024)
  fields:      25 max
  total:       6000 chars

Embed color (severity-driven):
  GREEN  (0x00FF00) — system nominal
  ORANGE (0xFFA500) — non-critical anomaly
  RED    (0xFF0000) — critical: stale data / kill-switch / dead feeds

Retry policy: 3 attempts with 30s / 60s backoff.

Usage:
  python scripts/send_toplists_discord.py
  python scripts/send_toplists_discord.py --input logs/intel_source_status.json --dry-run
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
_COLOR_GREEN  = 0x00FF00
_COLOR_ORANGE = 0xFFA500
_COLOR_RED    = 0xFF0000
_COLOR_BLUE   = 0x3498DB

_CRITICAL_FLAGS = frozenset({"STALE_SOURCE", "MISSING_AMOUNT", "DEAD_FEED"})

# ── 7-factor display config — single SSOT replacing legacy _FACTOR_LABEL/_FACTOR_EMOJI ──
# Each entry: (field_key_in_factors_dict, short_label_for_matrix)
_FACTOR_DISPLAY: List[tuple] = [
    ("insider_conviction", "IC"),
    ("insider_breadth",    "IB"),
    ("congress",           "CG"),
    ("news_sentiment",     "NS"),
    ("news_buzz",          "NB"),
    ("momentum_long",      "MO"),
    ("volume_attention",   "VA"),
]

# ── VIX regime thresholds ─────────────────────────────────────────────────────
_VIX_BEARISH = 25.0
_VIX_STABLE  = 15.0

# ── Sector normalisation ───────────────────────────────────────────────────────
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
_BUYBACK_HIGH = 0.10
_BUYBACK_LOW  = 0.05

_MEDAL: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}
_MARKET_FLAGS: Dict[str, str] = {"USA": "🇺🇸", "US": "🇺🇸", "EUROPE": "🇪🇺", "ASIA": "🇯🇵"}

_STALE_HOURS = 25


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    if _HAS_SCORE_BAR:
        return _score_bar_util(score, width)
    filled = min(width, max(0, round(score * width)))
    return "▓" * filled + "░" * (width - filled)


def _fmt_cap(market_cap: float) -> str:
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.1f}T"
    if market_cap >= 1e9:
        v = market_cap / 1e9
        return f"${v:.0f}B" if v >= 10 else f"${v:.1f}B"
    return f"${market_cap/1e6:.0f}M"


def _truncate(text: str, max_chars: int = 1024) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "…"


# ── Domain logic ───────────────────────────────────────────────────────────────

def get_market_regime(vix: float) -> str:
    if vix > _VIX_BEARISH:
        label, emoji = "BEARISH", "🔴"
    elif vix > _VIX_STABLE:
        label, emoji = "STABLE",  "🟡"
    else:
        label, emoji = "BULLISH", "🟢"
    return f"VIX `{vix:.1f}` {emoji} **{label}**"


def _buyback_conviction(yield_pct: float) -> Optional[float]:
    if yield_pct >= _BUYBACK_HIGH:
        return 0.80
    if yield_pct >= _BUYBACK_LOW:
        return 0.40
    return None


def _embed_color(anomaly_map: Dict[str, List[str]], kill_switch: bool) -> int:
    if kill_switch:
        return _COLOR_RED
    all_flags = {flag for flags in anomaly_map.values() for flag in flags}
    if all_flags & _CRITICAL_FLAGS:
        return _COLOR_RED
    if all_flags:
        return _COLOR_ORANGE
    return _COLOR_GREEN


def _data_age_hours(generated_at: str) -> Optional[float]:
    try:
        ts = datetime.fromisoformat((generated_at or "").replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    except Exception:
        return None


def _timestamp_from_status(status: Dict[str, Any]) -> str:
    """Extract the pipeline timestamp from intel_source_status or top_lists schema.

    intel_source_status.json uses 'computed_at' (top-level) and '_edgar_meta.last_run'.
    top_lists.json uses 'generated_at'.
    Falls back through all three.
    """
    return (
        status.get("generated_at")
        or status.get("computed_at")
        or (status.get("_edgar_meta") or {}).get("last_run")
        or ""
    )


def _compute_percentile(score: float, all_scores: List[float]) -> int:
    """Cross-sectional percentile rank of score within all_scores list."""
    if not all_scores:
        return 0
    below = sum(1 for s in all_scores if s <= score)
    return int(below / len(all_scores) * 100)


def _compute_catalyst(entry: Dict[str, Any]) -> str:
    """Return a one-line catalyst string from the top 2 non-zero factors."""
    factors = entry.get("factors") or {}
    scored = [
        (label, float(factors.get(key, 0) or 0))
        for key, label in _FACTOR_DISPLAY
    ]
    active = [(lbl, v) for lbl, v in scored if v >= 0.05]
    active.sort(key=lambda x: -x[1])
    if not active:
        return "no primary catalyst"
    parts = [f"{lbl}: {v:.2f}" for lbl, v in active[:2]]
    return "driven by " + " + ".join(parts)


def _sector_heatmap_structured(entries: List[Dict]) -> Dict[str, List[tuple]]:
    buckets: Dict[str, List[tuple]] = {}
    for e in entries:
        raw    = (e.get("sector") or "").strip()
        label  = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        buckets.setdefault(label, []).append((ticker, score))
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
    all_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Institutional 3-line ticker card.

    Line 1: {score:.4f}  {bar8}  p{pct}  {badge}  [CEO tag]
    Line 2: IC:{v}  IB:{v}  CG:{v}  NS:{v}  NB:{v}  MO:{v}  VA:{v}
             (values < 0.05 display as — )
    Line 3: {catalyst}
    ─────────────────────
    """
    ticker       = entry.get("ticker", "?")
    company_name = (entry.get("company_name") or "").strip()
    score        = float(entry.get("final_score", 0))
    badge        = entry.get("badge", "WATCHLIST")
    factors      = entry.get("factors") or {}
    market       = entry.get("market", "USA")

    # Percentile
    pct = _compute_percentile(score, all_scores) if all_scores else 0

    # Bar
    bar_str = _score_bar(score, width=8)

    # CEO tag
    ceo_tier = entry.get("ceo_conviction_tier", "none")
    if ceo_tier and ceo_tier != "none":
        ceo_tag = f"  | CEO {ceo_tier.upper()}"
    elif entry.get("ceo_buy"):
        ceo_tag = "  | CEO BUY"
    else:
        ceo_tag = ""

    # Anomaly / boost tags
    flag_tag    = "  ⚠️" if anomaly_flags else ""
    buyback_tag = f"  🔄+{buyback_conv:.2f}" if buyback_conv is not None else ""
    boost       = float(entry.get("congress_boost", 0.0))
    boost_tag   = f"  🏛+{boost:.2f}" if boost > 0.0 else ""

    if rank_delta and rank_delta > 0:
        trend_tag = f"  🟢+{rank_delta}"
    elif rank_delta and rank_delta < 0:
        trend_tag = f"  🔴{rank_delta}"
    else:
        trend_tag = ""

    line1 = f"{score:.4f}  {bar_str}  p{pct}  {badge}{ceo_tag}{boost_tag}{buyback_tag}{trend_tag}{flag_tag}"

    # Line 2: 7-factor matrix — zeros as —
    def _fmt(key: str) -> str:
        v = float(factors.get(key, 0) or 0)
        return "—" if v < 0.05 else f"{v:.2f}"

    parts = [f"{lbl}:{_fmt(key)}" for key, lbl in _FACTOR_DISPLAY]
    line2 = "  ".join(parts)

    # Line 3: catalyst
    line3 = _compute_catalyst(entry)

    flag  = _MARKET_FLAGS.get(market, "🌐")
    value = _truncate(f"{line1}\n{line2}\n{line3}\n─────────────────────", 1020)

    name_base = f"#{rank} {flag} {ticker}"
    if company_name:
        available = 256 - len(name_base) - 3
        safe_co   = company_name[:available] if len(company_name) > available else company_name
        name = f"{name_base} | {safe_co}"
    else:
        name = name_base

    return {"name": name, "value": value, "inline": False}


def _action_section(entries: List[Dict[str, Any]], all_scores: List[float]) -> Optional[Dict[str, Any]]:
    """Top-3 actionable recommendations: BUY (p95+) or WATCH (p80+)."""
    if not entries or not all_scores:
        return None

    lines = []
    for e in entries[:3]:
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        pct    = _compute_percentile(score, all_scores)
        cat    = _compute_catalyst(e)
        ceo_tier = e.get("ceo_conviction_tier", "none")
        ceo_note = f" · CEO {ceo_tier}" if ceo_tier and ceo_tier != "none" else ""
        if pct >= 95:
            verb = "**BUY**  "
        elif pct >= 80:
            verb = "**BUY**  "
        else:
            verb = "**HOLD** "
        lines.append(f"{verb} `{ticker}` — p{pct}{ceo_note} · {cat}")

    if not lines:
        return None

    return {
        "name":   "⚡ ACTION TODAY",
        "value":  _truncate("\n".join(lines), 1024),
        "inline": False,
    }


def _health_field(status: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline health summary from intel_source_status.json top-level fields."""
    meta   = status.get("_edgar_meta", {})
    orth   = status.get("factor_orthogonality", {})

    # Latency
    generated_at = status.get("generated_at") or meta.get("last_run", "")
    age_h = _data_age_hours(generated_at)
    age_str = f"{age_h:.1f}h" if age_h is not None else "?"

    # Orthogonality
    max_rho  = orth.get("max_abs_correlation", 0.0)
    max_pair = orth.get("max_pair", [])
    ldp      = orth.get("low_density_pairs", [])
    if max_pair and len(max_pair) == 2:
        pair_str = (
            f"{max_pair[0].replace('_score','').replace('_','.')}"
            f"<->{max_pair[1].replace('_score','').replace('_','.')}"
        )
        orth_line = f"Orthogonality: max rho={max_rho:.3f} ({pair_str})"
    else:
        orth_line = f"Orthogonality: max rho={max_rho:.3f}"
    if ldp:
        orth_line += f" | {len(ldp)} sparse pairs excluded"

    # Dead factors (density < 0.05)
    densities = orth.get("factor_densities", {})
    dead = [f.replace("_score", "") for f, d in densities.items() if isinstance(d, float) and d < 0.05]
    dead_str = ", ".join(dead) if dead else "none"

    # CEO tiers
    results = status.get("results", [])
    tier_counts: Dict[str, int] = {}
    for r in results:
        t = r.get("ceo_conviction_tier", "none") or "none"
        tier_counts[t] = tier_counts.get(t, 0) + 1
    tier_parts = [f"{n}x {t}" for t, n in sorted(tier_counts.items()) if t != "none" and n > 0]
    ceo_str = ", ".join(tier_parts) if tier_parts else "none"

    tickers  = meta.get("ticker_count", len(results))
    errors   = meta.get("error_count", 0)
    quarantine = meta.get("quarantine_count", 0)

    lines = [
        orth_line,
        f"Dead factors: {dead_str}",
        f"CEO tiers: {ceo_str}",
        f"Latency: {age_str}  |  Tickers: {tickers}  |  Errors: {errors}  |  Quarantine: {quarantine}",
    ]

    return {
        "name":   "🔬 PIPELINE HEALTH",
        "value":  _truncate("\n".join(lines), 1024),
        "inline": False,
    }


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _load_satellite(log_dir: Path) -> Optional[Dict[str, Any]]:
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


def _normalise_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map intel_source_status.json result row → ticker card entry.

    Builds the 'factors' dict from *_score_neutral fields (post-neutralization),
    falling back to raw *_score if neutral is absent.
    """
    def _get(key: str) -> float:
        neutral = raw.get(f"{key}_score_neutral")
        if neutral is not None:
            return float(neutral)
        return float(raw.get(f"{key}_score", 0) or 0)

    factors = {key: _get(key) for key, _ in _FACTOR_DISPLAY}

    return {
        "ticker":              raw.get("ticker", "?"),
        "sector":              raw.get("sector", "Unknown"),
        "cap_tier":            raw.get("cap_tier", "large"),
        "market_cap":          float(raw.get("market_cap", 0) or 0),
        "final_score":         float(raw.get("final_score", 0) or 0),
        "badge":               raw.get("badge") or _badge_from_score(float(raw.get("final_score", 0) or 0)),
        "ceo_buy":             bool(raw.get("ceo_buy", False)),
        "ceo_conviction_tier": raw.get("ceo_conviction_tier", "none"),
        "ceo_purchase_bps":    raw.get("ceo_purchase_bps"),
        "congress_boost":      float(raw.get("congress_boost", 0.0) or 0),
        "market":              raw.get("market", "USA"),
        "factors":             factors,
        "company_name":        raw.get("company_name", ""),
    }


def _badge_from_score(score: float) -> str:
    if score >= 0.80:
        return "HIGH BUY"
    if score >= 0.60:
        return "TACTICAL BUY"
    return "WATCHLIST"


# ── Payload builder ────────────────────────────────────────────────────────────

def build_payload(
    status: Dict[str, Any],
    satellite: Optional[Dict[str, Any]] = None,
    anomaly_map: Optional[Dict[str, List[str]]] = None,
    pipeline_latency_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the Discord webhook JSON payload from intel_source_status.json.

    Accepts both old top_lists.json schema (has 'top_buys' key) and new
    intel_source_status.json schema (has 'top_by_market' + 'results').
    """
    anomaly_map = anomaly_map or {}

    # ── Detect schema: intel_source_status vs legacy top_lists ───────────────
    is_status_schema = "top_by_market" in status

    if is_status_schema:
        generated_at = _timestamp_from_status(status)
        run_id   = status.get("run_id", "")
        weights  = status.get("weights", {})
        vix_val  = status.get("vix")
        kill_switch = status.get("kill_switch", False)

        # Build per-market top-5 from top_by_market (already sorted by final_score)
        tbm = status.get("top_by_market", {})
        # top_by_market values are full result dicts
        us_entries   = [_normalise_entry(e) for e in (tbm.get("US") or tbm.get("USA") or [])[:5]]
        eu_entries   = [_normalise_entry(e) for e in (tbm.get("EUROPE") or [])[:5]]
        asia_entries = [_normalise_entry(e) for e in (tbm.get("ASIA") or [])[:5]]

        # All results for percentile calculation
        all_results  = status.get("results", [])
        all_scores   = sorted([float(r.get("final_score", 0) or 0) for r in all_results if r.get("final_score")])

        # Mid caps: non-top-5 entries with cap_tier == "mid", cross-market
        top_tickers = {e["ticker"] for e in us_entries + eu_entries + asia_entries}
        mid_caps = sorted(
            [_normalise_entry(r) for r in all_results
             if r.get("cap_tier") == "mid" and r.get("ticker") not in top_tickers],
            key=lambda e: -e["final_score"]
        )[:5]

    else:
        # Legacy top_lists.json schema — graceful degradation
        generated_at = _timestamp_from_status(status)
        run_id   = status.get("source_run_id", status.get("run_id", ""))
        weights  = status.get("weights", {})
        vix_val  = status.get("vix")
        kill_switch = status.get("kill_switch", False)

        top_buys = status.get("top_buys") or []
        us_entries   = top_buys[:5]
        eu_entries   = list(status.get("top_buys_europe") or [])[:5]
        asia_entries = list(status.get("top_buys_asia") or [])[:5]
        all_scores   = sorted([float(e.get("final_score", 0)) for e in top_buys if e.get("final_score")])
        mid_caps     = list(status.get("mid_caps") or [])[:5]

    # ── Timing ───────────────────────────────────────────────────────────────
    age_h = _data_age_hours(generated_at)
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%b %d, %Y")
    except Exception:
        date_str = generated_at[:10] or "—"

    # ── Color ────────────────────────────────────────────────────────────────
    color = _embed_color(anomaly_map, kill_switch)
    if age_h is not None and age_h > _STALE_HOURS:
        color = _COLOR_RED

    # ── Buyback join (satellite) ──────────────────────────────────────────────
    buyback_conv_of: Dict[str, float] = {}
    try:
        if satellite:
            for c in (satellite.get("cannibals") or []):
                t    = (c.get("ticker") or "").upper()
                yld  = float(c.get("buyback_yield") or 0.0)
                conv = _buyback_conviction(yld)
                if t and conv is not None:
                    buyback_conv_of[t] = conv
    except Exception as exc:
        log.debug("buyback join failed: %s", exc)

    # ── Alerts ───────────────────────────────────────────────────────────────
    alerts: List[str] = []
    if age_h is not None and age_h > _STALE_HOURS:
        alerts.append(
            f"DATA IS {age_h:.0f}h OLD — pipeline may have failed. "
            "Check edgar_3x on GitHub Actions."
        )
    if kill_switch:
        vix_mult = status.get("vix_multiplier", 1.0)
        vix_note = f"VIX {vix_val:.1f}  |  " if vix_val is not None else ""
        alerts.append(
            f"MACRO KILL-SWITCH ACTIVE  —  {vix_note}"
            f"scores dampened x{vix_mult:.2f}.  Do NOT act on HIGH BUY signals."
        )
    if any(flag == "STALE_SOURCE" for flags in anomaly_map.values() for flag in flags):
        alerts.append("STALE DATA SOURCE — scores may be unreliable.")

    alert_block = (
        "\n" + "\n".join(f"```diff\n- {a}\n```" for a in alerts)
    ) if alerts else ""

    # ── Description ──────────────────────────────────────────────────────────
    vix_regime = get_market_regime(float(vix_val)) if vix_val is not None else "VIX —"
    age_note   = f"  |  Data: {age_h:.1f}h ago" if age_h is not None else ""
    description = (
        f"**[REGIME TRADER]** Daily Market Report — **{date_str}**\n"
        f"{vix_regime}{age_note}"
        f"{alert_block}"
    )

    # ── Shadow rank map (for trend arrow) ────────────────────────────────────
    shadow_buys    = status.get("shadow_top_buys") or []
    shadow_rank_of = {e.get("ticker", ""): i for i, e in enumerate(shadow_buys, 1)}

    def _ticker_fields(entries: List[Dict], max_n: int, budget: int) -> List[Dict]:
        result = []
        used   = 0
        added  = 0
        for i, e in enumerate(entries[:max_n], 1):
            ticker_   = e.get("ticker", "")
            shadow_r  = shadow_rank_of.get(ticker_)
            rank_delta = (shadow_r - i) if shadow_r is not None else None
            buyback_cv = buyback_conv_of.get(ticker_.upper())
            field = _ticker_detail_field(
                i, e,
                anomaly_flags=anomaly_map.get(ticker_),
                rank_delta=rank_delta,
                buyback_conv=buyback_cv,
                mid_cap=False,
                all_scores=all_scores,
            )
            flen = len(field["value"])
            if used + flen > budget and added > 0:
                result.append({
                    "name":   "…",
                    "value":  f"... [{added}/{min(max_n, len(entries))}] shown — full report in logs",
                    "inline": False,
                })
                break
            result.append(field)
            used  += flen
            added += 1
        return result

    # ── Fields ────────────────────────────────────────────────────────────────
    fields: List[Dict[str, Any]] = []

    _MARKET_SECTIONS = [
        ("🇺🇸 Top 5 — USA",    us_entries),
        ("🇪🇺 Top 5 — Europe", eu_entries),
        ("🇯🇵 Top 5 — Asia",   asia_entries),
    ]
    for section_name, section_entries in _MARKET_SECTIONS:
        if not section_entries:
            continue
        fields.append({"name": section_name, "value": "​", "inline": False})
        fields.extend(_ticker_fields(section_entries, max_n=5, budget=1800))

    # ── Mid caps ─────────────────────────────────────────────────────────────
    if mid_caps:
        fields.append({"name": "📈 Mid Caps — Top 5 (All Markets)", "value": "​", "inline": False})
        fields.extend(_ticker_fields(mid_caps, max_n=5, budget=1800))

    # ── Action today (before satellite — high priority) ───────────────────────
    all_top = us_entries + eu_entries + asia_entries
    action = _action_section(all_top, all_scores)
    if action:
        fields.append(action)

    # ── Sector exposure ───────────────────────────────────────────────────────
    all_entries = all_top + mid_caps[:5]
    structured  = _sector_heatmap_structured(all_entries)
    if structured:
        sorted_sectors = sorted(
            structured.items(),
            key=lambda kv: (-len(kv[1]), -(kv[1][0][1] if kv[1] else 0)),
        )
        sector_lines = []
        for lbl, pairs in sorted_sectors:
            total_in_sector = sum(
                1 for e in all_entries
                if _SECTOR_SHORT.get((e.get("sector") or "").strip(), _SECTOR_MISC) == lbl
            )
            chips = "  ".join(f"`{t}` {s:.2f}" for t, s in pairs)
            sector_lines.append(f"{lbl} ({total_in_sector})  {chips}")
        fields.append({
            "name":   "📊  Sector Exposure",
            "value":  _truncate("\n".join(sector_lines), 1024),
            "inline": False,
        })

    # ── Pipeline health (before satellite — critical visibility) ──────────────
    if is_status_schema:
        fields.append(_health_field(status))

    # ── Satellite (cyclicals + cannibals) — optional, appended last ───────────
    # Placed after Action/Sector/Health so Discord's 25-field limit drops
    # satellite before it drops analytical fields.
    try:
        if satellite:
            month_label = satellite.get("month", "")
            cyclicals   = satellite.get("cyclicals") or []
            cannibals   = satellite.get("cannibals") or []
            if cyclicals and len(fields) < 24:
                lines = [
                    f"**{c['ticker']}** {_score_bar(c['win_rate'], 6)} "
                    f"`{c['win_rate']:.0%}` win  |  `{c['median_return']:+.1%}` med  |  `{c.get('years','?')}y`"
                    for c in cyclicals
                ]
                fields.append({
                    "name":   f"🌀  Seasonal Cyclicals — {month_label}",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })
            if cannibals and len(fields) < 24:
                lines = [
                    f"**{c['ticker']}**  `{c.get('buyback_yield',0):.1%}` buyback"
                    f"  |  P/E `{c.get('pe',0):.1f}`  |  `{c.get('price_vs_52w_low',0):.2f}x` vs 52w low"
                    for c in cannibals
                ]
                fields.append({
                    "name":   "🐷  Share Cannibals",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })
    except Exception as exc:
        log.warning("satellite embed fields skipped: %s", exc)

    # ── Discord 25-field hard limit guard ─────────────────────────────────────
    if len(fields) > 25:
        log.warning("Discord 25-field limit exceeded (%d fields) — truncating to 25", len(fields))
        fields = fields[:25]

    # ── Footer ────────────────────────────────────────────────────────────────
    footer_text = f"Run: {run_id}  |  Pipeline: EDGAR-first"

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
    import traceback
    failures: List[str] = []

    def _check(name: str, cond: bool, detail: str = "") -> None:
        if not cond:
            failures.append(f"FAIL [{name}]{': ' + detail if detail else ''}")

    def _base_status(**overrides) -> Dict[str, Any]:
        st: Dict[str, Any] = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "run_id":        "test",
            "vix":           17.0,
            "kill_switch":   False,
            "weights":       {k: v for k, v in [
                ("insider_conviction", 0.30), ("insider_breadth", 0.15),
                ("congress", 0.22), ("news_sentiment", 0.10),
                ("news_buzz", 0.05), ("momentum_long", 0.15), ("volume_attention", 0.03),
            ]},
            "top_by_market": {"US": [], "EUROPE": [], "ASIA": []},
            "results":       [],
            "_edgar_meta":   {"ticker_count": 0, "error_count": 0, "quarantine_count": 0},
            "factor_orthogonality": {
                "max_abs_correlation": 0.35,
                "max_pair": ["momentum_long_score", "news_buzz_score"],
                "low_density_pairs": [],
                "factor_densities":  {"insider_conviction_score": 0.11, "congress_score": 0.0},
                "errors": [], "warnings": [],
            },
        }
        st.update(overrides)
        return st

    def _entry(ticker: str, sector: str = "Information Technology",
               score: float = 0.45, **kw) -> Dict[str, Any]:
        factors = {
            "insider_conviction": 0.50, "insider_breadth": 0.70,
            "congress": 0.0, "news_sentiment": 0.40,
            "news_buzz": 0.50, "momentum_long": 0.30, "volume_attention": 0.0,
        }
        factors.update(kw.pop("factors", {}))
        return {
            "ticker": ticker, "final_score": score, "badge": _badge_from_score(score),
            "sector": sector, "market_cap": 1e11, "cap_tier": "large",
            "ceo_buy": False, "ceo_conviction_tier": "none",
            "congress_boost": 0.0, "factors": factors, "market": "USA",
            **kw,
        }

    # ── Test 1: factor matrix renders 7 factors ───────────────────────────────
    try:
        e = _entry("AAPL", score=0.45)
        field = _ticker_detail_field(1, e, all_scores=[0.45])
        val = field["value"]
        for _, lbl in _FACTOR_DISPLAY:
            _check(f"factor_matrix_{lbl}", lbl in val, f"lbl={lbl!r} not in val={val!r}")
    except Exception:
        failures.append(f"FAIL [factor_matrix_7]: {traceback.format_exc()}")

    # ── Test 2: zeros render as — ─────────────────────────────────────────────
    try:
        e = _entry("AAPL", score=0.45)
        e["factors"]["congress"]           = 0.0
        e["factors"]["volume_attention"]   = 0.0
        field = _ticker_detail_field(1, e, all_scores=[0.45])
        val = field["value"]
        # congress and volume_attention should show — not 0.00
        lines = val.split("\n")
        matrix_line = lines[1] if len(lines) > 1 else ""
        _check("zero_congress_is_dash",         "CG:—" in matrix_line, f"matrix={matrix_line!r}")
        _check("zero_volume_attention_is_dash",  "VA:—" in matrix_line, f"matrix={matrix_line!r}")
        # non-zero factors must NOT be dash
        _check("nonzero_ic_not_dash", "IC:—" not in matrix_line, f"matrix={matrix_line!r}")
    except Exception:
        failures.append(f"FAIL [zeros_as_dash]: {traceback.format_exc()}")

    # ── Test 3: action section picks top 3 ───────────────────────────────────
    try:
        entries = [
            _entry("CHTR", score=0.49),
            _entry("NKE",  score=0.42),
            _entry("PSX",  score=0.40),
            _entry("ETN",  score=0.38),
        ]
        all_sc = [0.10, 0.20, 0.30, 0.35, 0.38, 0.40, 0.42, 0.49]
        action = _action_section(entries, all_sc)
        _check("action_not_none",      action is not None)
        _check("action_has_chtr",      action is not None and "CHTR" in action["value"])
        _check("action_has_nke",       action is not None and "NKE"  in action["value"])
        _check("action_has_psx",       action is not None and "PSX"  in action["value"])
        _check("action_not_etn",       action is not None and "ETN"  not in action["value"])
    except Exception:
        failures.append(f"FAIL [action_section]: {traceback.format_exc()}")

    # ── Test 4: health field includes orthogonality ───────────────────────────
    try:
        st = _base_status()
        health = _health_field(st)
        val = health["value"]
        _check("health_has_rho",    "rho=" in val,          f"val={val!r}")
        _check("health_has_latency", "Latency" in val,       f"val={val!r}")
        _check("health_has_ceo",     "CEO tiers" in val,     f"val={val!r}")
        _check("health_has_dead",    "Dead factors" in val,  f"val={val!r}")
    except Exception:
        failures.append(f"FAIL [health_field]: {traceback.format_exc()}")

    # ── Test 5: build_payload with status schema — no crash ───────────────────
    try:
        e = _entry("CHTR", score=0.49)
        st = _base_status()
        st["top_by_market"] = {"US": [e]}
        st["results"] = [e]
        payload = build_payload(st)
        embed   = payload["embeds"][0]
        _check("payload_has_title",   "Alpha Pipeline" in embed.get("title", ""))
        _check("payload_has_fields",  len(embed.get("fields", [])) > 0)
        _check("payload_has_health",  any("PIPELINE HEALTH" in f["name"] for f in embed["fields"]))
    except Exception:
        failures.append(f"FAIL [build_payload_status]: {traceback.format_exc()}")

    # ── Test 6: build_payload with legacy top_lists schema — no crash ─────────
    try:
        tl = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "source_run_id": "test-legacy",
            "vix":           17.0,
            "kill_switch":   False,
            "weights":       {},
            "top_buys":      [_entry("AAPL", score=0.45)],
            "mid_caps":      [],
        }
        payload = build_payload(tl)
        embed   = payload["embeds"][0]
        _check("legacy_payload_no_crash", True)
        _check("legacy_has_usa_section",
               any("USA" in f["name"] or "Top 5" in f["name"] for f in embed["fields"]))
    except Exception:
        failures.append(f"FAIL [build_payload_legacy]: {traceback.format_exc()}")

    # ── Test 7: empty top_buys → no ticker fields ─────────────────────────────
    try:
        st = _base_status()
        payload = build_payload(st)
        embed   = payload["embeds"][0]
        field_names = [f["name"] for f in embed["fields"]]
        _check("empty_no_ticker_fields", not any(n.startswith("#") for n in field_names))
    except Exception:
        failures.append(f"FAIL [empty_no_ticker_fields]: {traceback.format_exc()}")

    # ── Test 8: missing sector → Misc in heatmap ──────────────────────────────
    try:
        entries = [_entry("AAPL", sector=""), _entry("MSFT", sector="")]
        result = _sector_heatmap_structured(entries)
        _check("misc_fallback", _SECTOR_MISC in result, f"result={result!r}")
    except Exception:
        failures.append(f"FAIL [misc_fallback]: {traceback.format_exc()}")

    # ── Test 9: VIX regime labels ─────────────────────────────────────────────
    try:
        _check("vix_bullish",  "BULLISH" in get_market_regime(12.0))
        _check("vix_stable",   "STABLE"  in get_market_regime(18.0))
        _check("vix_bearish",  "BEARISH" in get_market_regime(28.0))
    except Exception:
        failures.append(f"FAIL [vix_regime]: {traceback.format_exc()}")

    # ── Test 10: EU / Asia sections only appear when non-empty ────────────────
    try:
        eu = _entry("SAP.DE", sector="Information Technology", score=0.42)
        eu["market"] = "EUROPE"
        st = _base_status()
        st["top_by_market"] = {"US": [], "EUROPE": [eu], "ASIA": []}
        st["results"] = [eu]
        payload = build_payload(st)
        names = [f["name"] for f in payload["embeds"][0]["fields"]]
        _check("europe_section_present", any("Europe" in n for n in names), f"names={names}")
        _check("usa_section_absent",     not any("USA" in n for n in names), f"names={names}")
        _check("asia_section_absent",    not any("Asia" in n for n in names),  f"names={names}")
    except Exception:
        failures.append(f"FAIL [market_sections]: {traceback.format_exc()}")

    # ── Test 11: percentile badge on LINE 1 ───────────────────────────────────
    try:
        e = _entry("CHTR", score=0.49)
        field = _ticker_detail_field(1, e, all_scores=[0.10, 0.20, 0.30, 0.40, 0.49])
        line1 = field["value"].split("\n")[0]
        _check("line1_has_percentile", "p" in line1 and any(c.isdigit() for c in line1), f"line1={line1!r}")
        _check("line1_has_score",      "0.4900" in line1, f"line1={line1!r}")
        _check("line1_has_badge",      "WATCHLIST" in line1 or "BUY" in line1, f"line1={line1!r}")
    except Exception:
        failures.append(f"FAIL [line1_format]: {traceback.format_exc()}")

    # ── Test 12: catalyst line present ───────────────────────────────────────
    try:
        e = _entry("CHTR", score=0.49)
        field = _ticker_detail_field(1, e, all_scores=[0.49])
        lines = field["value"].split("\n")
        _check("has_three_lines_plus_sep", len(lines) >= 3, f"lines={lines}")
        _check("catalyst_line_present", any("driven by" in l or "no primary" in l for l in lines), f"lines={lines}")
    except Exception:
        failures.append(f"FAIL [catalyst_line]: {traceback.format_exc()}")

    # ── Report ────────────────────────────────────────────────────────────────
    total_assertions = 32
    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print(f"\n{len(failures)} test(s) FAILED", file=sys.stderr)
        return 1
    print(f"All tests passed ({total_assertions} assertions)")
    return 0


# ── HTTP send with retry ───────────────────────────────────────────────────────

def send_to_discord(
    webhook: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    backoff_base_s: float = 30.0,
) -> bool:
    if not _HAS_REQUESTS:
        log.error("'requests' not installed — cannot send to Discord")
        return False
    if not webhook:
        log.warning("DISCORD_WEBHOOK_URL is empty — skipping (no-op)")
        return False

    for attempt in range(max_retries):
        if attempt > 0:
            wait = backoff_base_s * attempt
            log.warning("Retry %d/%d in %.0fs ...", attempt + 1, max_retries, wait)
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
        "--input", type=Path, default=Path("logs/intel_source_status.json"),
        help="Path to intel_source_status.json (default: logs/intel_source_status.json)",
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

    # Try intel_source_status.json first; fall back to top_lists.json
    input_path = args.input
    if not input_path.exists() and input_path.name == "intel_source_status.json":
        fallback = args.log_dir / "top_lists.json"
        if fallback.exists():
            log.warning("intel_source_status.json not found — falling back to top_lists.json")
            input_path = fallback

    if not input_path.exists():
        log.warning("%s not found — sending alert", input_path)
        payload = build_alert_payload(f"File not found: {input_path}")
        if args.dry_run:
            sys.stdout.buffer.write(
                (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            )
            return 0
        return 0 if send_to_discord(webhook, payload) else 1

    try:
        status = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("could not parse %s: %s", input_path.name, exc)
        payload = build_alert_payload(f"JSON parse error: {exc}")
        if args.dry_run:
            sys.stdout.buffer.write(
                (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            )
            return 0
        send_to_discord(webhook, payload)
        return 1

    satellite   = _load_satellite(args.log_dir)
    anomaly_map = _load_anomaly_report(args.log_dir)
    payload     = build_payload(status, satellite=satellite, anomaly_map=anomaly_map)

    if args.dry_run:
        sys.stdout.buffer.write(
            (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        )
        return 0

    ok = send_to_discord(webhook, payload)
    if not ok:
        log.error("All Discord send attempts failed")
        return 1

    log.info("Daily market checkup sent successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
