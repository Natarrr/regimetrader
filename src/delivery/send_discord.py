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
_COLOR_GREEN = 0x00FF00
_COLOR_ORANGE = 0xFFA500
_COLOR_RED = 0xFF0000
_COLOR_BLUE = 0x3498DB

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
    ("analyst_consensus",  "AC"),
    ("analyst_revision",   "AR"),
]

# ── VIX regime thresholds ─────────────────────────────────────────────────────
_VIX_BEARISH = 25.0
_VIX_STABLE = 15.0

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
_BUYBACK_LOW = 0.05

_MEDAL: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}
_MARKET_FLAGS: Dict[str, str] = {"USA": "🇺🇸",
                                 "US": "🇺🇸", "EUROPE": "🇪🇺", "ASIA": "🇯🇵"}

_STALE_HOURS = 25
_NO_CATALYST = "no primary catalyst detected"


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


def _fmt_usd(usd: float) -> str:
    if not usd or usd <= 0:
        return "$0"
    if usd >= 100_000:
        return f"${usd/1000:.0f}k"
    value = usd / 1000.0
    formatted = f"${value:.1f}k"
    return formatted.rstrip("0").rstrip(".")


def _fmt_insider_badge(entry: Dict[str, Any]) -> Optional[str]:
    usd = float(entry.get("insider_usd", 0) or 0)
    if usd <= 0:
        return None
    label = f"Insider {_fmt_usd(usd)}"
    ceo_tier = (entry.get("ceo_conviction_tier") or "").strip()
    if ceo_tier and ceo_tier.lower() != "none":
        return f"{label} CEO"
    form4_count = int(entry.get("form4_count", 0) or 0)
    if form4_count > 0:
        return f"{label} · {form4_count} filings"
    return label


def _fmt_eps_badge(entry: Dict[str, Any]) -> str:
    pct = entry.get("earnings_surprise_pct")
    days = int(entry.get("earnings_surprise_days") or 0)
    if pct is None or days > 90:
        return "EPS —"
    pct_value = float(pct)
    if pct_value < 0:
        return f"EPS miss {abs(pct_value * 100):.1f}%"
    return f"EPS +{pct_value * 100:.1f}% · {days}d ago"


def _fmt_pt_badge(entry: Dict[str, Any]) -> str:
    """Format price target badge using actual stored prices (not score inversion).

    Uses target_price + current_price stored in result dict by run_pipeline.py.
    Falls back to qualitative label if raw prices unavailable.
    """
    target = entry.get("target_price") or entry.get("targetConsensus")
    price  = entry.get("current_price") or entry.get("price")
    try:
        target_val = float(target) if target is not None else 0.0
        price_val  = float(price)  if price  is not None else 0.0
    except (TypeError, ValueError):
        target_val = price_val = 0.0

    if target_val > 0 and price_val > 0:
        upside = (target_val - price_val) / price_val * 100
        sign   = "+" if upside >= 0 else ""
        return f"PT ${target_val:.0f} ({sign}{upside:.0f}%)"

    # No raw prices — qualitative fallback only (no fake % from score)
    score_val = float(entry.get("price_target_upside_score") or 0.0)
    if score_val >= 0.60:
        return "PT ↑ above target"
    if score_val >= 0.40:
        return "PT → at target"
    if score_val > 0:
        return "PT ↓ below target"
    return "PT —"


def _fmt_analyst_badge(entry: Dict[str, Any]) -> str:
    source = (entry.get("analyst_consensus_source") or "").strip()
    label = source if source else "Consensus —"
    n = int(entry.get("analyst_revision_n_analysts") or 0)
    if n > 0:
        label += f" · {n} upgrades"
    try:
        rev_score = float(entry.get("analyst_revision_score") or 0.0)
    except Exception:
        rev_score = 0.0
    if rev_score > 0.6:
        label += " · rev ↑"
    elif rev_score < 0.4:
        label += " · rev ↓"
    return label


def _fmt_transcript_badge(entry: Dict[str, Any]) -> str:
    signals = entry.get("transcript_signals")
    if not isinstance(signals, dict) or not signals:
        return "No transcript"
    tone = (signals.get("guidance_tone") or "").strip().lower()
    tone_map = {
        "raised":    "Guidance ↑",
        "maintained": "Guidance →",
        "lowered":   "Guidance ↓",
    }
    guidance = tone_map.get(tone, "Guidance")
    extras: List[str] = []
    if signals.get("buyback_mentioned"):
        extras.append("Buyback")
    elif (signals.get("management_confidence") or "").strip().lower() == "high":
        extras.append("conf. high")
    return " · ".join([guidance] + extras)


def _fmt_factor_matrix(entry: Dict[str, Any], market: str = "US") -> str:
    market_norm = (market or "US").upper()
    factors = entry.get("factors") or {}
    raw_values: Dict[str, float] = {
        "analyst_revision": float(entry.get("analyst_revision_score") or 0.0),
        "price_target_upside": float(entry.get("price_target_upside_score") or 0.0),
    }
    labels = [
        ("insider_conviction", "IC"),
        ("insider_breadth", "IB"),
        ("congress", "CG"),
        ("news_sentiment", "NS"),
        ("news_buzz", "NB"),
        ("momentum_long", "MO"),
        ("volume_attention", "VA"),
        ("analyst_revision", "AR"),
        ("price_target_upside", "PT"),
    ]
    parts: List[str] = []
    for key, label in labels:
        # v2.2-global: FMP Ultimate confirmed for EU/Asia — read factors dict as-is.
        # congress (0.0) and transcript_tone (0.0) render as "—" via value <= 0 below.
        value = raw_values.get(key) if key in raw_values else float(
            factors.get(key, 0) or 0)
        parts.append(f"{label}:—" if value <= 0 else f"{label}:{value:.2f}")
    qf = float(entry.get("quality_piotroski_score") or 0.0)
    if qf > 0:
        parts.append(f"QF:{qf:.2f}")
    suffix = ""
    if market_norm.startswith("EU"):
        suffix = " [EU]"
    elif market_norm == "ASIA":
        suffix = " [Asia]"
    return " ".join(parts) + suffix


def _build_ticker_card(
    rank: int,
    entry: Dict[str, Any],
    market: str = "US",
    kill_switch: bool = False,
    mid_cap: bool = False,
    held_context: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    ticker = entry.get("ticker", "?")
    score = float(entry.get("final_score", 0) or 0)
    pct = int(entry.get("percentile", 0) or 0)
    badge = _badge_from_score(score)
    if kill_switch:
        badge = f"[DAMPENED] {badge}"
    mktcap_str = ""
    if mid_cap:
        market_cap = float(entry.get("market_cap", 0) or 0)
        if market_cap > 0:
            mktcap_str = f" · ${market_cap / 1e9:.1f}B"
    factor_matrix = _fmt_factor_matrix(entry, market)
    badge_texts = [
        _fmt_insider_badge(entry),
        _fmt_eps_badge(entry),
        _fmt_pt_badge(entry),
        _fmt_analyst_badge(entry),
        _fmt_transcript_badge(entry),
    ]
    badge_line = " ".join([b for b in badge_texts if b])
    name = f"#{rank} {ticker} | {badge} | p{pct} | {score:.4f}{mktcap_str}"
    if entry.get("esg_flag"):
        name = f"{name} | ESG!"
    # Add position annotation for Revolut context
    if held_context is not None:
        avg_cost = held_context.get(ticker)
        if avg_cost is not None:
            pos_label = f" [HOLD @{avg_cost:.0f}]" if avg_cost > 0 else " [HOLD]"
        else:
            pos_label = " [NEW]"
        name = f"{name}{pos_label}"

    value = _truncate(f"`{factor_matrix}`\n{badge_line}", 1024)
    return {
        "name": name,
        "value": value,
        "inline": False,
    }


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
        ts = datetime.fromisoformat(
            (generated_at or "").replace("Z", "+00:00"))
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
    """Evidence-first catalyst narrative, max 80 chars, signals separated by ' · '.

    Priority order (max 3 signals emitted):
      1. INSIDER: insider_usd > 0 → "Insider $Xk [CEO]"
      2. EPS: earnings_surprise_pct → "EPS +X% · Nd ago" / "EPS miss X%"
      3. CONGRESS: quiver_evidence.congress.purchases > 0 → "Nx congress buy · rep"
      4. MOMENTUM: |momentum_spy_relative| > 0.05 → "±X% vs SPY 12m"
      5. ANALYST REVISION (fallback only): analyst_revision_n_analysts ≥ 5
    """
    signals: list[str] = []

    usd = float(entry.get("insider_usd", 0) or 0)
    if usd > 0:
        usd_label = _fmt_usd(usd)
        ceo_tier = (entry.get("ceo_conviction_tier") or "").strip()
        if ceo_tier and ceo_tier.lower() != "none":
            signals.append(f"Insider {usd_label} [{ceo_tier}]")
        else:
            signals.append(f"Insider {usd_label}")

    eps_pct = entry.get("earnings_surprise_pct")
    eps_days = int(entry.get("earnings_surprise_days") or 0)
    if eps_pct is not None and eps_days <= 90:
        pct = float(eps_pct)
        if pct < 0:
            signals.append(f"EPS miss {abs(pct * 100):.1f}%")
        else:
            signals.append(f"EPS +{pct * 100:.1f}% · {eps_days}d ago")

    # Analyst rating change (most recent upgrade/downgrade within 7 days)
    upg_raw = entry.get("recent_upgrade_downgrade") or {}
    if isinstance(upg_raw, dict) and upg_raw.get("action") in ("upgrade", "downgrade"):
        action_str = upg_raw["action"].upper()
        firm_raw   = upg_raw.get("analyst_firm") or ""
        firm_str   = f" {firm_raw[:12]}" if firm_raw else ""
        days_upg   = int(upg_raw.get("days_ago") or 0)
        signals.append(f"{action_str}{firm_str} {days_upg}d")

    congress = (entry.get("quiver_evidence") or {}).get("congress", {})
    cg_buys = int(congress.get("purchases", 0) or 0)
    if cg_buys > 0:
        reps = congress.get("representatives") or []
        rep_str = reps[0][:12] if reps else "members"
        signals.append(f"{cg_buys}x congress buy · {rep_str}")

    rel = float(entry.get("momentum_spy_relative", 0) or 0)
    if abs(rel) > 0.05:
        sign = "+" if rel >= 0 else ""
        signals.append(f"{sign}{rel * 100:.1f}% vs SPY 12m")

    if not signals:
        n = int(entry.get("analyst_revision_n_analysts",
                entry.get("analyst_revision_n", 0)) or 0)
        if n >= 5:
            signals.append(f"analyst revision ({n} analysts)")

    result = " · ".join(signals[:3])
    catalyst_str = (result or _NO_CATALYST)[:80]

    # PATCH 09: Append double-count warning when both insider and congress fire.
    # Grinold & Kahn (2000): correlated signals overstate effective signal strength.
    if entry.get("_correlated_signal_flag"):
        warning = " ⚠double-signal"
        if len(catalyst_str) + len(warning) <= 80:
            catalyst_str = catalyst_str + warning
        else:
            catalyst_str = catalyst_str[:80 - len(warning)] + warning

    # PATCH 12B: Advisory risk flags for negative signals.
    risk_flags: list[str] = []

    mom = float(entry.get("momentum_spy_relative", 0) or 0)
    if mom < -0.10:
        risk_flags.append(f"⚠ mom {mom*100:.0f}%")

    rev_score = float(entry.get("analyst_revision_score") or 0.0)
    if 0.0 < rev_score < 0.35:
        risk_flags.append("⚠ rev↓")

    quality = float(entry.get("quality_piotroski_score") or 0.0)
    if 0.0 < quality < 0.30:
        risk_flags.append("⚠ F-Score↓")

    if risk_flags:
        risk_str = " ".join(risk_flags[:2])
        combined = f"{catalyst_str} | {risk_str}"
        catalyst_str = combined[:80]

    # F6.2: Append top-3 weighted factor contribution line.
    # Format: "Top: momentum_long(0.19) analyst_consensus(0.08) insider_conviction(0.05)"
    # Only appended when entry carries normalized factor scores from top_lists.json.
    factor_line = _factor_contribution_line(entry)
    if factor_line:
        catalyst_str = f"{catalyst_str}\n{factor_line}"

    return catalyst_str


def _weights_for_entry(entry: dict) -> dict:
    """Return the correct weight dict based on ticker region."""
    from regime_trader.config.weights import get_weights as _gw  # noqa: PLC0415
    ticker = entry.get("ticker", "")
    return _gw(ticker)


def _factor_contribution_line(entry: Dict[str, Any]) -> str:
    """Return top-3 weighted factor contributions as a compact string.

    Contribution = weight × normalized_score.  Pulls weights from the
    'weights' key in the entry (written by generate_top_lists) or falls back
    to the correct regional weight set based on entry pipeline metadata.
    Returns empty string when no factor data is present.
    """
    factors = entry.get("factors")
    if not factors:
        return ""
    weights = entry.get("weights") or _weights_for_entry(entry)
    contributions = {
        k: round(float(weights.get(k, 0.0)) * float(factors.get(k, 0.0)), 4)
        for k in factors
        if float(factors.get(k, 0.0) or 0.0) > 0.0
    }
    if not contributions:
        return ""
    top3 = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    parts = [f"{k.replace('_', ' ')}({v:.2f})" for k, v in top3]
    return "Top: " + " · ".join(parts)


def _sector_heatmap_structured(entries: List[Dict]) -> Dict[str, List[tuple]]:
    buckets: Dict[str, List[tuple]] = {}
    for e in entries:
        raw = (e.get("sector") or "").strip()
        label = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        ticker = e.get("ticker", "?")
        score = float(e.get("final_score", 0))
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
    score_delta: Optional[float] = None,
    buyback_conv: Optional[float] = None,
    mid_cap: bool = False,
    all_scores: Optional[List[float]] = None,
    kill_switch: bool = False,
    held_context: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if all_scores is not None:
        entry["percentile"] = _compute_percentile(
            float(entry.get("final_score", 0) or 0), all_scores)
    field = _build_ticker_card(
        rank,
        entry,
        market=entry.get("market", "US"),
        kill_switch=kill_switch,
        mid_cap=mid_cap,
        held_context=held_context,
    )
    if anomaly_flags:
        field["name"] = f"{field['name']} ⚠️"
    if score_delta is not None:
        try:
            current_score = float(entry.get("final_score", 0))
            diff = current_score - float(score_delta)
            if abs(diff) >= 0.01:
                arrow = "▲" if diff > 0 else "▼"
                delta_str = f" {arrow}{diff:+.3f}"
            else:
                delta_str = ""
        except Exception:
            delta_str = ""
        field["name"] = f"{field['name']}{delta_str}"
    return field


def _action_section(entries: List[Dict[str, Any]], all_scores: List[float]) -> Optional[Dict[str, Any]]:
    """Top-3 actionable recommendations gated on BADGE THRESHOLDS, not just percentile.

    A WATCHLIST ticker (score < 0.60) must not be labelled BUY — that violates
    the badge system the rest of the embed uses. Score thresholds match
    _badge_from_score() exactly:
      score >= 0.60  → BUY  (TACTICAL BUY or HIGH BUY band)
      score <  0.60  → WATCH (WATCHLIST — relative rank only, no action)
    """
    if not entries or not all_scores:
        return None

    lines = []
    for e in entries[:3]:
        ticker = e.get("ticker", "?")
        score = float(e.get("final_score", 0))
        pct = _compute_percentile(score, all_scores)
        cat = _compute_catalyst(e)
        # Fix #3: verb gated on score (badge thresholds), NOT percentile alone
        if score >= 0.60:
            verb = "**BUY**   "
        else:
            verb = "**WATCH** "
        lines.append(f"{verb} `{ticker}` — p{pct} · {cat}")

    if not lines:
        return None

    return {
        "name":   "⚡ ACTION TODAY",
        "value":  _truncate("\n".join(lines), 1024),
        "inline": False,
    }


def _dead_factor_lines(top_buys_data: list, weights: dict) -> list:
    """Return signal-health warning lines for the Discord health section.

    Detects dead (all 0.0) and flat (all identical non-boundary value) factors
    from top_buys entries. Returns [] when everything looks healthy.
    """
    if not top_buys_data or not weights:
        return []

    factor_scores: dict = {}
    for entry in top_buys_data:
        for k, v in entry.get("factors", {}).items():
            factor_scores.setdefault(k, []).append(float(v or 0.0))

    dead: list = []
    flat: list = []
    for factor, scores in factor_scores.items():
        if all(s == 0.0 for s in scores):
            dead.append(factor)
        elif len(set(round(s, 2) for s in scores)) == 1 and scores[0] not in (0.0, 1.0):
            flat.append(f"{factor}={scores[0]:.2f}")

    lines: list = []
    if dead or flat:
        lines.append("⚠️ **Signal health:**")
        if dead:
            lines.append(f"  Dead (0.0): `{'`, `'.join(dead)}`")
        if flat:
            lines.append(f"  Flat (no discrimination): `{'`, `'.join(flat)}`")

    all_dead_flat = dead + [x.split("=")[0] for x in flat]
    total_weight = sum(weights.values()) or 1.0
    active_weight = sum(w for f, w in weights.items() if f not in all_dead_flat)
    dead_weight = total_weight - active_weight
    if all_dead_flat and dead_weight > 0.05 * total_weight:
        lines.append(
            f"  Effective weight: **{active_weight/total_weight*100:.0f}%** "
            f"({dead_weight/total_weight*100:.0f}% in dead/flat factors)"
        )

    return lines


def _health_field(status: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline health summary from intel_source_status.json top-level fields."""
    meta = status.get("_edgar_meta", {})
    orth = status.get("factor_orthogonality", {})

    # Latency
    generated_at = status.get("generated_at") or meta.get("last_run", "")
    age_h = _data_age_hours(generated_at)
    age_str = f"{age_h:.1f}h" if age_h is not None else "?"

    # Orthogonality
    max_rho = orth.get("max_abs_correlation", 0.0)
    max_pair = orth.get("max_pair", [])
    ldp = orth.get("low_density_pairs", [])
    if max_pair and len(max_pair) == 2:
        pair_str = (
            f"{max_pair[0].replace('_score', '').replace('_', '.')}"
            f"<->{max_pair[1].replace('_score', '').replace('_', '.')}"
        )
        orth_line = f"Orthogonality: max rho={max_rho:.3f} ({pair_str})"
    else:
        orth_line = f"Orthogonality: max rho={max_rho:.3f}"
    if ldp:
        orth_line += f" | {len(ldp)} sparse pairs excluded"

    # Dead factors (density < 0.05)
    densities = orth.get("factor_densities", {})
    dead = [f.replace("_score", "") for f, d in densities.items()
            if isinstance(d, float) and d < 0.05]
    dead_str = ", ".join(dead) if dead else "none"

    # CEO tiers
    results = status.get("results", [])
    tier_counts: Dict[str, int] = {}
    for r in results:
        t = r.get("ceo_conviction_tier", "none") or "none"
        tier_counts[t] = tier_counts.get(t, 0) + 1
    tier_parts = [f"{n}x {t}" for t, n in sorted(
        tier_counts.items()) if t != "none" and n > 0]
    ceo_str = ", ".join(tier_parts) if tier_parts else "none"

    tickers = meta.get("ticker_count", len(results))
    errors = meta.get("error_count", 0)
    quarantine = meta.get("quarantine_count", 0)

    # Dead/flat factor warnings — support both intel_source_status.json (results list,
    # _score suffix) and top_lists.json (top_buys list, nested factors dict).
    top_buys_data = status.get("top_buys")
    if not top_buys_data:
        # intel_source_status.json path: synthesise factor dicts from results rows
        raw_results = status.get("results", [])[:50]
        top_buys_data = [
            {
                "factors": {
                    k.replace("_score", ""): float(v or 0.0)
                    for k, v in r.items()
                    if k.endswith("_score") and isinstance(v, (int, float, type(None)))
                }
            }
            for r in raw_results
        ]
    weights_data = status.get("weights", {})
    df_lines = _dead_factor_lines(top_buys_data, weights_data)

    # Congress dead-days (from dead_factors_detail if present)
    congress_detail = (status.get("dead_factors_detail") or {}).get("congress", {})
    if congress_detail.get("dead") and congress_detail.get("dead_days", 0) > 0:
        dead_str = f"congress (dead {congress_detail['dead_days']}d)"

    lines = df_lines + [
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
            flag = rec.get("flag", "")
            if ticker and flag:
                result.setdefault(ticker, []).append(flag)
        return result
    except Exception as exc:
        log.warning("anomaly_report_latest.json unreadable: %s", exc)
        return {}


def _load_top_lists_overlay(log_dir: Path, input_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load vix / kill_switch / vix_multiplier from top_lists.json.

    intel_source_status.json does not carry the macro overlay — that lives in
    top_lists.json next to it in the artifact.  Returns {} if the file is absent
    or unreadable; caller treats missing keys as "no overlay".

    Fallback: if top_lists.json is not found in log_dir, try the same directory
    as input_path (sibling artifact pattern used in CI).
    """
    path = log_dir / "top_lists.json"
    if not path.exists() and input_path is not None:
        path = Path(input_path).parent / "top_lists.json"
    if not path.exists():
        log.warning(
            "top_lists.json not found at %s — VIX overlay will be missing from embed", path
        )
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "vix":             data.get("vix"),
            "kill_switch":     data.get("kill_switch", False),
            "vix_multiplier":  data.get("vix_multiplier", 1.0),
            "shadow_top_buys": data.get("shadow_top_buys", []),
        }
    except Exception as exc:
        log.warning("top_lists.json unreadable: %s", exc)
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

    eps_pct = raw.get("earnings_surprise_pct")   # float | None
    eps_days = int(raw.get("earnings_surprise_days") or 0)

    entry = {
        "ticker":                    raw.get("ticker", "?"),
        "sector":                    raw.get("sector", "Unknown"),
        "cap_tier":                  raw.get("cap_tier", "large"),
        "market_cap":                float(raw.get("market_cap", 0) or 0),
        "final_score":               float(raw.get("final_score", 0) or 0),
        "badge":                     raw.get("badge") or _badge_from_score(float(raw.get("final_score", 0) or 0)),
        "ceo_buy":                   bool(raw.get("ceo_buy", False)),
        "ceo_conviction_tier":       raw.get("ceo_conviction_tier", "none"),
        "ceo_purchase_bps":          raw.get("ceo_purchase_bps"),
        "congress_boost":            float(raw.get("congress_boost", 0.0) or 0),
        "market":                    raw.get("market", "USA"),
        "factors":                   factors,
        # Fix #2: analyst + quality fields read by _fmt_factor_matrix / _fmt_analyst_badge / _fmt_pt_badge
        "analyst_consensus_score":   float(raw.get("analyst_consensus_score") or 0.0),
        "analyst_consensus_source":  raw.get("analyst_consensus_source", "none"),
        "analyst_revision_score":    float(raw.get("analyst_revision_score") or 0.0),
        "analyst_revision_n_analysts": int(raw.get("analyst_revision_n") or 0),
        "price_target_upside_score": float(raw.get("price_target_upside_score") or 0.0),
        "quality_piotroski_score":   float(raw.get("quality_piotroski_score") or 0.0),
        "company_name":              raw.get("company_name", ""),
        "earnings_surprise_pct":     eps_pct,
        "earnings_surprise_days":    eps_days,
        # Catalyst / evidence pass-through
        "insider_usd":               float(raw.get("insider_usd", 0.0) or 0),
        "form4_count":               int(raw.get("form4_count", 0) or 0),
        "quiver_evidence":           raw.get("quiver_evidence", {}),
        "momentum_spy_relative":     float(raw.get("momentum_spy_relative", 0.0) or 0),
        "transcript_tone_score":     float(raw.get("transcript_tone_score") or 0.0),
        "transcript_tone_source":    raw.get("transcript_tone_source", "none"),
        "recent_upgrade_downgrade":  raw.get("recent_upgrade_downgrade", {}),
        "target_price":              raw.get("target_price"),
        "current_price":             raw.get("current_price"),
    }

    for key in ("esg_score", "esg_e_score", "esg_flag"):
        if key in raw:
            entry[key] = raw.get(key)

    return entry


def _badge_from_score(score: float) -> str:
    if score >= 0.80:
        return "HIGH BUY"
    if score >= 0.60:
        return "TACTICAL BUY"
    return "WATCHLIST"


# ── Regional embed builder (v2.1-global) ──────────────────────────────────────
# Lightweight alternative to build_payload for top_lists.json consumers.
# Produces one Discord embed per non-empty region (US / EU / Asia).
# Falls back to unified top_buys when regional keys are absent.

_REGION_CONFIG: Dict[str, Dict[str, str]] = {
    "US":   {"emoji": "🇺🇸", "label": "US",   "weight_note": "9-factor (congress included)"},
    "EU":   {"emoji": "🇪🇺", "label": "EU",   "weight_note": "8-factor (congress absent — weight redistributed)"},
    "ASIA": {"emoji": "🌏", "label": "Asia", "weight_note": "8-factor (congress absent — weight redistributed)"},
}

_FACTOR_SHORT_REGIONAL: Dict[str, str] = {
    "insider_conviction": "IC",
    "insider_breadth":    "IB",
    "congress":           "CON",
    "news_sentiment":     "NS",
    "news_buzz":          "NB",
    "momentum_long":      "MOM",
    "volume_attention":   "VOL",
    "analyst_consensus":  "AC",
    "quality_piotroski":  "PIO",
}


def _fmt_factor_bar_regional(score: float, width: int = 8) -> str:
    filled = min(width, max(0, round(score * width)))
    return "▓" * filled + "░" * (width - filled)


def _build_regional_ticker_line(entry: Dict[str, Any], show_congress: bool = True) -> str:
    ticker  = entry.get("ticker", "?")
    score   = entry.get("final_score", 0.0)
    factors = entry.get("factors", {})
    badge   = entry.get("badge", "")
    region  = entry.get("region", "US")

    core = ["insider_conviction", "news_sentiment", "momentum_long", "analyst_consensus"]
    if show_congress and region == "US":
        core.insert(2, "congress")

    parts = [f"**{ticker}** `{score:.3f}`"]
    if badge:
        parts.append(f"_{badge}_")

    factor_parts = []
    for f in core:
        s = factors.get(f, 0.0)
        if s > 0.01:
            factor_parts.append(f"{_FACTOR_SHORT_REGIONAL[f]}{_fmt_factor_bar_regional(s, 5)}")
    if factor_parts:
        parts.append("  ".join(factor_parts))

    return "  ".join(parts)


def _region_embed_color(region_code: str) -> int:
    return {
        "US":   0x2ECC71,
        "EU":   0x3498DB,
        "ASIA": 0x9B59B6,
    }.get(region_code, 0x95A5A6)


def build_regional_embeds(
    top_lists: Dict[str, Any],
    max_per_region: int = 5,
    vix: float = 0.0,
    kill_switch: bool = False,
) -> List[Dict[str, Any]]:
    """Build one Discord embed per non-empty region from top_lists.json.

    Reads top_buys_us / top_buys_eu / top_buys_asia when present (post-v2.1
    regional keys); falls back to splitting unified top_buys by region field.
    Returns a list of embed dicts ready to POST as ``{"embeds": [...]}`` payload.
    """
    embeds: List[Dict[str, Any]] = []

    has_regional = any(k in top_lists for k in ("top_buys_us", "top_buys_eu", "top_buys_asia"))

    if has_regional:
        region_keys = [
            ("US",   top_lists.get("top_buys_us",   [])),
            ("EU",   top_lists.get("top_buys_eu",   [])),
            ("ASIA", top_lists.get("top_buys_asia", [])),
        ]
    else:
        top_buys = top_lists.get("top_buys", [])
        us   = [e for e in top_buys if e.get("region", "US") == "US"]
        eu   = [e for e in top_buys if e.get("region") == "EU"]
        asia = [e for e in top_buys if e.get("region") == "ASIA"]
        if eu or asia:
            region_keys = [("US", us), ("EU", eu), ("ASIA", asia)]
        else:
            region_keys = [("US", top_buys)]

    for region_code, entries in region_keys:
        if not entries:
            continue

        cfg   = _REGION_CONFIG[region_code]
        top_n = entries[:max_per_region]

        if kill_switch and region_code == "US":
            description = (
                "⚠️ **Kill-switch active** (VIX ≥ 30) — BUY signals suppressed.\n"
                "SELL signals remain live. No new positions."
            )
        else:
            lines = [_build_regional_ticker_line(e, show_congress=(region_code == "US")) for e in top_n]
            description = "\n".join(lines) if lines else "_No tickers in this region today._"

        embeds.append({
            "title":       f"{cfg['emoji']} Top Buys — {cfg['label']}",
            "description": description,
            "color":       _region_embed_color(region_code),
            "footer":      {
                "text": (
                    f"{cfg['weight_note']}  •  VIX {vix:.1f}"
                    + ("  •  🔴 Kill-switch" if kill_switch else "")
                )
            },
        })

    return embeds


# ── Institutional block (v22 — Batch Floor + GTC desk notice) ─────────────────

def _fmt_region_block(label: str, model_name: str, entries: list, max_entries: int = 3) -> str:
    try:
        from regime_trader.risk.exit_rules import format_card_line  # noqa: PLC0415
        _has_exit = True
    except ImportError:
        _has_exit = False

    lines = [f"[{label} - {model_name}]"]
    for rank, e in enumerate(entries[:max_entries], 1):
        ticker = e.get("ticker", "???")
        score  = e.get("final_score", 0.0)
        badge  = e.get("badge", "WATCHLIST")
        cap    = " [CAPITULATION SURVIVOR]" if e.get("_capitulation_survivor") else ""
        lines.append(f"  {rank}. {ticker:<8}| SCORE: {score:.4f} | {badge}{cap}")
        if _has_exit:
            lines.append(f"     " + format_card_line(e))
    if not entries[:max_entries]:
        lines.append("  [NO QUALIFYING ASSETS IN CURRENT REGIME]")
    return "\n".join(lines)


def _fmt_mvo_pool(label: str, pool: dict) -> str:
    try:
        from regime_trader.risk.exit_rules import format_card_line  # noqa: PLC0415
        _has_exit = True
    except ImportError:
        _has_exit = False

    lines = [f"[{label}]"]
    for pos in pool.get("positions", [])[:6]:
        ticker = pos.get("ticker", "???")
        alloc  = (pos.get("allocation") or 0) * 100
        score  = pos.get("final_score", 0)
        stage  = "ENTER POSITION" if score >= 0.80 else "ASYMMETRIC LONG"
        lines.append(f"  - {ticker:<6} (ALLOC: {alloc:5.2f}%) | STAGE: {stage}")
        if _has_exit:
            lines.append(f"    " + format_card_line(pos))
    return "\n".join(lines)


def _fmt_sector_concentration(all_entries: list) -> str:
    """Rigid syntax: Theme Sector Exposure: Sector (N) Ticker Weight | Sector (N) ..."""
    sector_map: dict = {}
    for e in all_entries:
        raw   = (e.get("factors", {}).get("sector") or e.get("sector") or "Other").strip()
        short = _SECTOR_SHORT.get(raw, raw[:5])
        score = e.get("final_score", 0.0)
        sector_map.setdefault(short, []).append((e.get("ticker", "?"), score))

    parts = []
    for sector in sorted(sector_map):
        tickers = sector_map[sector]
        count   = len(tickers)
        top2    = sorted(tickers, key=lambda x: x[1], reverse=True)[:2]
        ticker_str = " ".join(f"{t} {w:.2f}" for t, w in top2)
        parts.append(f"{sector} ({count}) {ticker_str}")

    return "Theme Sector Exposure: " + " | ".join(parts) if parts else "Theme Sector Exposure: N/A"


def build_institutional_payload(top_lists: dict) -> list:
    """Build two-embed institutional monospaced block with GTC desk notice.

    Embed 1: Regime header + regional equity sections with Batch Floor card lines.
    Embed 2: GTC desk notice + MVO pool sections + sector concentration.
    """
    vix         = top_lists.get("vix", 0.0)
    vix_regime  = (top_lists.get("vix_regime") or "UNKNOWN").upper()
    kill_switch = top_lists.get("kill_switch", False)
    gen_at      = (top_lists.get("generated_at") or "")[:16]

    if kill_switch or vix >= 30:
        strategy = "CAPITULATION DISTRESSED REGIME / HIGH-QUALITY ANCHORS ONLY"
    elif vix >= 25:
        strategy = "DEFENSIVE / GRADUATED POSITIONING ACTIVATED"
    else:
        strategy = "NORMAL / FULL POSITIONING"

    SEP  = "=" * 72
    THIN = "-" * 72

    block1 = "\n".join([
        SEP,
        "INSTITUTIONAL RISK & ALPHA DISPATCH",
        SEP,
        f"[REGIME STATUS] DETECTED REGIME: {vix_regime} (VIX: {vix:.2f})",
        f"[RISK OVERLAY]  STRATEGY: {strategy}",
        SEP,
        "",
        "TOP REGIONAL EQUITIES",
        THIN,
        _fmt_region_block("US MARKET", "INSIDER INFILTRATION MODEL",
                          top_lists.get("top_buys_usa", [])),
        "",
        _fmt_region_block("ASIAN MARKET", "LIQUIDITY SENTIMENT MODEL",
                          top_lists.get("top_buys_asia", [])),
        "",
        _fmt_region_block("EUROPEAN MARKET", "BALANCE SHEET QUALITY MODEL",
                          top_lists.get("top_buys_europe", [])),
    ])

    mvo_pools = top_lists.get("mvo_pools", {})
    mvo_lines = [
        "[DESK NOTICE: PLACE GTC BROKERAGE STOPS AT BATCH FLOOR PRICES IMMEDIATELY]",
        "",
        "OPTIMIZED PORTFOLIO POOLS (MEAN-VARIANCE ENGINE)",
        THIN,
    ]

    anchors = mvo_pools.get("large_cap_anchors", {})
    if anchors.get("positions"):
        mvo_lines.append(_fmt_mvo_pool("STRUCTURAL CORE ANCHORS (>$10B - EQUAL WEIGHT)", anchors))
        mvo_lines.append("")

    mid = mvo_pools.get("mid_cap", {})
    if mid.get("positions"):
        mvo_lines.append(_fmt_mvo_pool("MID-CAP SHARPE MAXIMIZER ($2B-$10B)", mid))
        mvo_lines.append("")

    small = mvo_pools.get("small_cap", {})
    if small.get("positions"):
        mvo_lines.append(_fmt_mvo_pool("SMALL-CAP MIN VARIANCE ($300M-$2B | ADV-GATED)", small))

    all_entries = (
        top_lists.get("top_buys_usa", []) +
        top_lists.get("top_buys_europe", []) +
        top_lists.get("top_buys_asia", [])
    )
    block2 = "\n".join(mvo_lines + [
        "",
        "GLOBAL THEMATIC SECTOR CONCENTRATION",
        THIN,
        _fmt_sector_concentration(all_entries),
        "",
        SEP,
        f"[PIPELINE STATUS: NOMINAL | GENERATED: {gen_at}]",
        SEP,
    ])

    color = _COLOR_RED if (kill_switch or vix >= 30) else (
        _COLOR_ORANGE if vix >= 25 else _COLOR_GREEN
    )

    def _wrap_code(text: str) -> str:
        wrapped = "```\n" + text + "\n```"
        return wrapped[:4096]

    return [
        {"embeds": [{"description": _wrap_code(block1), "color": color}]},
        {"embeds": [{"description": _wrap_code(block2), "color": color}]},
    ]


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
        run_id = status.get("run_id", "")
        vix_val = status.get("vix")
        kill_switch = status.get("kill_switch", False)

        # Build per-market top-5 from top_by_market (already sorted by final_score)
        tbm = status.get("top_by_market", {})
        # top_by_market values are full result dicts
        us_entries = [_normalise_entry(e) for e in (
            tbm.get("US") or tbm.get("USA") or [])[:5]]
        eu_entries = [_normalise_entry(e)
                      for e in (tbm.get("EUROPE") or [])[:5]]
        asia_entries = [_normalise_entry(e)
                        for e in (tbm.get("ASIA") or [])[:5]]

        # All results for percentile calculation
        all_results = status.get("results", [])
        all_scores = sorted([
            float(r.get("final_score", 0))
            for r in all_results
            if r.get("final_score") is not None
        ])

        # Mid caps: non-top-5 entries with cap_tier == "mid", cross-market
        top_tickers = {e["ticker"]
                       for e in us_entries + eu_entries + asia_entries}
        mid_caps = sorted(
            [_normalise_entry(r) for r in all_results
             if r.get("cap_tier") == "mid" and r.get("ticker") not in top_tickers],
            key=lambda e: -e["final_score"]
        )[:5]

    else:
        # Legacy top_lists.json schema — graceful degradation
        generated_at = _timestamp_from_status(status)
        run_id = status.get("source_run_id", status.get("run_id", ""))
        vix_val = status.get("vix")
        kill_switch = status.get("kill_switch", False)

        top_buys = status.get("top_buys") or []
        us_entries = top_buys[:5]
        eu_entries = list(status.get("top_buys_europe") or [])[:5]
        asia_entries = list(status.get("top_buys_asia") or [])[:5]
        all_scores = sorted([
            float(e.get("final_score", 0))
            for e in top_buys
            if e.get("final_score") is not None
        ])
        mid_caps = list(status.get("mid_caps") or [])[:5]

    # ── Timing ───────────────────────────────────────────────────────────────
    age_h = _data_age_hours(generated_at)
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = ts.strftime("%b %d %H:%M UTC")
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
                t = (c.get("ticker") or "").upper()
                yld = float(c.get("buyback_yield") or 0.0)
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
    vix_regime = get_market_regime(
        float(vix_val)) if vix_val is not None else "VIX —"
    age_note = f"  |  Data: {age_h:.1f}h ago" if age_h is not None else ""
    description = (
        f"**[REGIME TRADER]** Daily Market Report — **{date_str}**\n"
        f"{vix_regime}{age_note}"
        f"{alert_block}"
    )

    # Load yesterday's scores from archive for real score-delta display.
    # Reads all regional lists so EU/Asia tickers (never in unified top_buys)
    # get a prior-day score and show ▲/▼ instead of always showing [NEW].
    _yesterday_scores: Dict[str, float] = {}
    try:
        _archive_root = Path(__file__).resolve().parent.parent / "logs" / "archive"
        _archive_files = sorted(_archive_root.glob("*_top_lists.json"))
        if len(_archive_files) >= 2:
            _prev_data = json.loads(_archive_files[-2].read_text(encoding="utf-8"))
            _all_prev = (
                _prev_data.get("top_buys", [])
                + _prev_data.get("top_buys_europe", [])
                + _prev_data.get("top_buys_asia", [])
                + _prev_data.get("mid_caps", [])
            )
            for _prev_e in _all_prev:
                _prev_t = _prev_e.get("ticker", "")
                if _prev_t and _prev_t not in _yesterday_scores:
                    _yesterday_scores[_prev_t] = float(_prev_e.get("final_score", 0))
        else:
            log.debug(
                "Only %d archive file(s) found — score delta not available", len(_archive_files)
            )
    except Exception:
        pass  # archive not available — score delta not shown

    # Load Revolut positions for hold/add context
    _held_tickers: set[str] = set()
    _held_avg_cost: Dict[str, float] = {}
    try:
        _rev_path = Path("data/revolut_portfolio.json")
        if _rev_path.exists():
            _rev_data = json.loads(_rev_path.read_text(encoding="utf-8"))
            for _pos in _rev_data.get("positions", []):
                _t = _pos.get("ticker", "")
                if _t:
                    _held_tickers.add(_t)
                    _held_avg_cost[_t] = float(_pos.get("avg_cost", 0.0))
    except Exception:
        pass  # portfolio file not available

    def _ticker_fields(
        entries: List[Dict],
        max_n: int,
        budget: int,
        all_scores: Optional[List[float]] = None,
        mid_cap: bool = False,
    ) -> List[Dict]:
        result = []
        used = 0
        added = 0
        for i, e in enumerate(entries[:max_n], 1):
            ticker_ = e.get("ticker", "")
            score_delta = _yesterday_scores.get(ticker_)
            buyback_cv = buyback_conv_of.get(ticker_.upper())
            field = _ticker_detail_field(
                i,
                e,
                anomaly_flags=anomaly_map.get(ticker_),
                score_delta=score_delta,
                buyback_conv=buyback_cv,
                mid_cap=mid_cap,
                all_scores=all_scores,
                kill_switch=kill_switch,
                held_context=_held_avg_cost,
            )
            flen = len(field["value"])
            if used + flen > budget and added > 0:
                result.append({
                    "name": "…",
                    "value": f"... [{added}/{min(max_n, len(entries))}] shown — full report in logs",
                    "inline": False,
                })
                break
            result.append(field)
            used += flen
            added += 1
        return result

    # ── Fields ────────────────────────────────────────────────────────────────
    fields: List[Dict[str, Any]] = []
    overlay = f"x{float(status.get('vix_multiplier', 1.0)):.2f}"
    kill_state = "ACTIVE" if kill_switch else "NORMAL"
    fields.extend([
        {"name": "VIX", "value": f"`{vix_val:.1f}`" if vix_val is not None else "`—`", "inline": True},
        {"name": "Regime", "value": f"`{vix_regime}`", "inline": True},
        {"name": "Overlay", "value": f"`{overlay}`", "inline": True},
        {"name": "Kill switch", "value": f"`{kill_state}`", "inline": True},
    ])
    fields.append({
        "name": "⚡ ACTION BAR",
        "value": (
            "🚫 NO TRADES — MACRO KILL SWITCH ACTIVE (VIX ≥ 30)"
            if kill_switch else "✅ MARKET OPEN — signals are actionable"
        ),
        "inline": False,
    })

    _MARKET_SECTIONS = [
        ("🇺🇸 Top 3 — USA", us_entries),
        ("🇪🇺 Top 3 — Europe", eu_entries),
        ("🇯🇵 Top 3 — Asia", asia_entries),
    ]
    for section_name, section_entries in _MARKET_SECTIONS:
        if not section_entries:
            continue
        fields.append({"name": section_name, "value": "​", "inline": False})
        fields.extend(_ticker_fields(section_entries, max_n=3,
                      budget=1800, all_scores=all_scores, mid_cap=False))

    if mid_caps:
        mid_scores = sorted([float(e.get("final_score", 0) or 0)
                            for e in mid_caps])
        fields.append(
            {"name": "📈 Mid-cap catalyst watch — top 3 cross-market", "value": "​", "inline": False})
        fields.extend(_ticker_fields(mid_caps, max_n=3,
                      budget=1800, all_scores=mid_scores, mid_cap=True))

    # ── Action today (before satellite — high priority) ───────────────────────
    all_top = us_entries + eu_entries + asia_entries
    action = _action_section(all_top, all_scores)
    if action:
        fields.append(action)

    # ── Sector exposure ───────────────────────────────────────────────────────
    all_entries = all_top + mid_caps[:5]
    structured = _sector_heatmap_structured(all_entries)
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
            cyclicals = satellite.get("cyclicals") or []
            cannibals = satellite.get("cannibals") or []
            if cyclicals and len(fields) < 24:
                lines = [
                    f"**{c['ticker']}** {_score_bar(c['win_rate'], 6)} "
                    f"`{c['win_rate']:.0%}` win  |  `{c['median_return']:+.1%}` med  |  `{c.get('years', '?')}y`"
                    for c in cyclicals
                ]
                fields.append({
                    "name":   f"🌀  Seasonal Cyclicals — {month_label}",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })
            if cannibals and len(fields) < 24:
                lines = [
                    f"**{c['ticker']}**  `{c.get('buyback_yield', 0):.1%}` buyback"
                    f"  |  P/E `{c.get('pe', 0):.1f}`  |  `{c.get('price_vs_52w_low', 0):.2f}x` vs 52w low"
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
        log.warning(
            "Discord 25-field limit exceeded (%d fields) — truncating to 25", len(fields))
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
                ("news_buzz", 0.05), ("momentum_long",
                                      0.15), ("volume_attention", 0.03),
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

    # ── Test 1: factor matrix renders expected labels ────────────────────────
    try:
        e = _entry("AAPL", score=0.45)
        field = _ticker_detail_field(1, e, all_scores=[0.45])
        val = field["value"]
        expected_labels = ["IC", "IB", "CG",
                           "NS", "NB", "MO", "VA", "AR", "PT"]
        for lbl in expected_labels:
            _check(f"factor_matrix_{lbl}", lbl in val,
                   f"lbl={lbl!r} not in val={val!r}")
    except Exception:
        failures.append(
            f"FAIL [factor_matrix_labels]: {traceback.format_exc()}")

    # ── Test 2: zeros render as — ─────────────────────────────────────────────
    try:
        e = _entry("AAPL", score=0.45)
        e["factors"]["congress"] = 0.0
        e["factors"]["volume_attention"] = 0.0
        field = _ticker_detail_field(1, e, all_scores=[0.45])
        val = field["value"]
        # congress and volume_attention should show — not 0.00
        lines = val.split("\n")
        matrix_line = lines[0] if len(lines) > 0 else ""
        _check("zero_congress_is_dash",
               "CG:—" in matrix_line, f"matrix={matrix_line!r}")
        _check("zero_volume_attention_is_dash",
               "VA:—" in matrix_line, f"matrix={matrix_line!r}")
        # non-zero factors must NOT be dash
        _check("nonzero_ic_not_dash", "IC:—" not in matrix_line,
               f"matrix={matrix_line!r}")
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
        _check("action_has_chtr",
               action is not None and "CHTR" in action["value"])
        _check("action_has_nke",
               action is not None and "NKE" in action["value"])
        _check("action_has_psx",
               action is not None and "PSX" in action["value"])
        _check("action_not_etn",
               action is not None and "ETN" not in action["value"])
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
        embed = payload["embeds"][0]
        _check("payload_has_title",   "Alpha Pipeline" in embed.get("title", ""))
        _check("payload_has_fields",  len(embed.get("fields", [])) > 0)
        _check("payload_has_health",  any(
            "PIPELINE HEALTH" in f["name"] for f in embed["fields"]))
    except Exception:
        failures.append(
            f"FAIL [build_payload_status]: {traceback.format_exc()}")

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
        embed = payload["embeds"][0]
        _check("legacy_payload_no_crash", True)
        _check("legacy_has_usa_section",
               any("USA" in f["name"] or "Top 5" in f["name"] for f in embed["fields"]))
    except Exception:
        failures.append(
            f"FAIL [build_payload_legacy]: {traceback.format_exc()}")

    # ── Test 7: empty top_buys → no ticker fields ─────────────────────────────
    try:
        st = _base_status()
        payload = build_payload(st)
        embed = payload["embeds"][0]
        field_names = [f["name"] for f in embed["fields"]]
        _check("empty_no_ticker_fields", not any(n.startswith("#")
               for n in field_names))
    except Exception:
        failures.append(
            f"FAIL [empty_no_ticker_fields]: {traceback.format_exc()}")

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
        _check("vix_stable",   "STABLE" in get_market_regime(18.0))
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
        _check("europe_section_present", any(
            "Europe" in n for n in names), f"names={names}")
        _check("usa_section_absent", not any(
            "USA" in n for n in names), f"names={names}")
        _check("asia_section_absent", not any(
            "Asia" in n for n in names),  f"names={names}")
    except Exception:
        failures.append(f"FAIL [market_sections]: {traceback.format_exc()}")

    # ── Test 11: percentile badge on LINE 1 ───────────────────────────────────
    try:
        e = _entry("CHTR", score=0.49)
        field = _ticker_detail_field(
            1, e, all_scores=[0.10, 0.20, 0.30, 0.40, 0.49])
        line1 = field["name"]
        _check("line1_has_percentile", "p" in line1 and any(c.isdigit()
               for c in line1), f"line1={line1!r}")
        _check("line1_has_score",      "0.4900" in line1, f"line1={line1!r}")
        _check("line1_has_badge",
               "WATCHLIST" in line1 or "BUY" in line1, f"line1={line1!r}")
    except Exception:
        failures.append(f"FAIL [line1_format]: {traceback.format_exc()}")

    # ── Test 12: catalyst line present ───────────────────────────────────────
    try:
        e = _entry("CHTR", score=0.49)
        field = _ticker_detail_field(1, e, all_scores=[0.49])
        lines = field["value"].split("\n")
        _check("has_two_lines", len(lines) >= 2, f"lines={lines}")
        _check(
            "catalyst_line_present",
            any(kw in field["value"] for kw in ["Insider",
                "EPS", "congress", "vs SPY", "no primary"]),
            f"Catalyst line missing expected pattern: value={field['value']!r}",
        )

        # Zero-signal entry → _NO_CATALYST
        e_zero = _entry("ZERO", score=0.10)
        e_zero["insider_usd"] = 0.0
        e_zero["earnings_surprise_pct"] = None
        cat_zero = _compute_catalyst(e_zero)
        _check("zero_signal_fallback", cat_zero.startswith(_NO_CATALYST),
               f"cat_zero={cat_zero!r}")
    except Exception:
        failures.append(f"FAIL [catalyst_line]: {traceback.format_exc()}")

    # ── Test 13: EPS surprise appended to catalyst when within 90-day window ──
    try:
        e = _entry("NVDA", score=0.72)
        e["earnings_surprise_pct"] = 0.153   # +15.3% beat
        e["earnings_surprise_days"] = 8
        # +20% vs SPY — triggers second signal → · separator
        e["momentum_spy_relative"] = 0.20
        cat = _compute_catalyst(e)
        _check("eps_in_catalyst_beat",  "EPS +15.3%" in cat, f"cat={cat!r}")
        _check("eps_days_in_catalyst",  "8d ago" in cat, f"cat={cat!r}")
        _check("eps_separator",         "·" in cat, f"cat={cat!r}")

        # Negative surprise
        e2 = _entry("INTC", score=0.30)
        e2["earnings_surprise_pct"] = -0.087
        e2["earnings_surprise_days"] = 45
        cat2 = _compute_catalyst(e2)
        _check("eps_in_catalyst_miss",
               "EPS miss 8.7%" in cat2, f"cat2={cat2!r}")
        _check("eps_days_miss",
               "45d ago" not in cat2, f"cat2={cat2!r}")

        # Outside 90-day window → no EPS fragment
        e3 = _entry("AAPL", score=0.55)
        e3["earnings_surprise_pct"] = 0.20
        e3["earnings_surprise_days"] = 95
        cat3 = _compute_catalyst(e3)
        _check("eps_absent_outside_window",
               "EPS" not in cat3, f"cat3={cat3!r}")

        # None surprise → no EPS fragment
        e4 = _entry("MSFT", score=0.60)
        e4["earnings_surprise_pct"] = None
        e4["earnings_surprise_days"] = 0
        cat4 = _compute_catalyst(e4)
        _check("eps_absent_when_none", "EPS" not in cat4, f"cat4={cat4!r}")
    except Exception:
        failures.append(f"FAIL [eps_catalyst]: {traceback.format_exc()}")

    # ── Report ────────────────────────────────────────────────────────────────
    total_assertions = 40
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
            log.warning("Retry %d/%d in %.0fs ...",
                        attempt + 1, max_retries, wait)
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
                log.warning(
                    "Discord rate-limited — waiting %.1fs", retry_after)
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
    parser = argparse.ArgumentParser(
        description="Send daily market checkup to Discord")
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
            log.warning(
                "intel_source_status.json not found — falling back to top_lists.json")
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

    # Side-load macro overlay (VIX, kill_switch) from top_lists.json — these
    # live in the sibling artifact, not in intel_source_status.json.
    if input_path.name == "intel_source_status.json":
        overlay = _load_top_lists_overlay(args.log_dir, input_path=input_path)
        for k, v in overlay.items():
            status.setdefault(k, v)
        log.info(
            "Loaded macro overlay: vix=%s kill_switch=%s vix_multiplier=%s",
            status.get("vix"), status.get("kill_switch"), status.get("vix_multiplier"),
        )

    satellite = _load_satellite(args.log_dir)
    anomaly_map = _load_anomaly_report(args.log_dir)
    payload = build_payload(status, satellite=satellite,
                            anomaly_map=anomaly_map)

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
