# Path: src/delivery/send_discord.py
"""Institutional daily-brief Discord delivery for the cooked top_lists.json.

Executed by daily_trading_pipeline.yml (cook_and_notify, 3x/day at
00:30 / 08:30 / 16:30 UTC) as:

  python src/delivery/send_discord.py --input logs/top_lists.json --log-dir logs

Layout (single webhook POST, one embed — DiscordPayloadBuilder):
  description  ```ansi action bar (regime glyph + VIX + overlay + status)
               + Section 1 MACRO RISK & REGIME (markdown, mobile-safe)
  fields       Section 2 ALPHA DESK — USA / EUROPE / ASIA
               Section 3 FACTOR MATRIX (plain code block, ASCII-only)
               Section 4 PORTFOLIO CONSTRUCTION (MVO pools + sectors)
               LEGEND (always last)
  CAPITULATION theme replaces desks/portfolio with STRUCTURAL ANCHORS
  (watchlist, force-WATCHLIST) — BUY suppressed, SELL/exit signals live.

ANSI caveat: Discord renders ```ansi colors on desktop/web only; mobile
clients show plain text. Escape codes are therefore confined to the leading
action-bar block, reset before the closing fence, and the block terminates
with a blank line so mobile clients exit the terminal context cleanly.

Discord embed limits (enforced structurally, never by slicing fenced text):
  title 256 · description 4096 · field value 1024 · fields 25 · total 6000

DATA UNAVAILABLE contract: on missing/corrupt/invalid input the script sends
a red alert embed whose title contains "DATA UNAVAILABLE"
(asserted by .github/workflows/test_daily_toplists_absence.yml).

Exit codes: 0 sent (or alert sent) · 1 send/parse/audit failure · 2 no webhook.

Retry policy: 3 attempts with 30s / 60s backoff, 429-aware.
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
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

from src.risk.regime import (  # noqa: E402
    RiskRegime,
    get_regime,
    score_multiplier,
    vix_multiplier,
    strategy_label,
    classify_market_regime,
    market_regime_label,
)
from src.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL  # noqa: E402

log = logging.getLogger("discord.send_toplists")

# ── Color palette (severity-driven) ───────────────────────────────────────────
_COLOR_GREEN = 0x00FF00
_COLOR_ORANGE = 0xFFA500
_COLOR_RED = 0xFF0000

# ── ANSI (action bar only — see module docstring) ─────────────────────────────
_ESC = "\x1b"
_ANSI_RESET = _ESC + "[0m"
_ANSI_GREEN = _ESC + "[1;32m"
_ANSI_YELLOW = _ESC + "[1;33m"
_ANSI_RED = _ESC + "[1;31m"
_ANSI_DIM = _ESC + "[2;37m"

_REGIME_STYLE: Dict[RiskRegime, Tuple[str, str, str, int]] = {
    # regime → (ansi color, glyph, action text, embed color)
    RiskRegime.NORMAL: (
        _ANSI_GREEN, "●", "ALL SIGNALS ACTIONABLE", _COLOR_GREEN),
    RiskRegime.BEAR: (
        _ANSI_YELLOW, "◆", "GRADUATED CAUTION", _COLOR_ORANGE),
    RiskRegime.CAPITULATION: (
        _ANSI_RED, "■", "BUY SUPPRESSED · SELLS LIVE", _COLOR_RED),
}

# ── Embed budget (Discord hard limits) ────────────────────────────────────────
_LIMIT_DESC = 4096
_LIMIT_FIELD = 1024
_LIMIT_TOTAL = 6000

_STALE_HOURS = 25
_DELTA_STALE_HOURS = 48.0
_NO_CATALYST = "no primary catalyst detected"

# ── Freshness / extension gate ────────────────────────────────────────────────
# A name already up this much over the last 5 sessions is a chase, not an entry:
# it moves off the actionable ALPHA DESK into the ⏱ EXTENDED / ALREADY-MOVED
# (WATCH) section. This is a DISPLAY/RISK gate only — it never touches
# final_score, weights or factor math. Cap-tier-aware because small/mid-caps
# routinely run further before mean-reverting; a flat threshold would dump the
# whole SMID sleeve into "extended". Env-overridable for tuning without a deploy.
_EXTENSION_PCT_LARGE = float(os.getenv("EXTENSION_PCT_LARGE", "0.10"))
_EXTENSION_PCT_SMID  = float(os.getenv("EXTENSION_PCT_SMID",  "0.18"))

# ── Factor matrix columns (9-factor US schema; intl swaps CG/NB/VA/QF for
#    the EU/Asia value & liquidity factors — congress structurally absent) ────
_MATRIX_US_COLS: List[Tuple[str, str]] = [
    ("insider_conviction", "IC"),
    ("insider_breadth",    "IB"),
    ("congress",           "CG"),
    ("news_sentiment",     "NS"),
    ("news_buzz",          "NB"),
    ("momentum_long",      "MO"),
    ("volume_attention",   "VA"),
    ("analyst_consensus",  "AC"),
    ("quality_piotroski",  "QF"),
]
_MATRIX_INTL_COLS: List[Tuple[str, str]] = [
    ("insider_conviction", "IC"),
    ("insider_breadth",    "IB"),
    ("news_sentiment",     "NS"),
    ("momentum_long",      "MO"),
    ("analyst_consensus",  "AC"),
    ("fcf_yield",          "FCF"),
    ("amihud_shock",       "AMH"),
    ("pb_value_up",        "PB"),
    ("roic_quality",       "ROI"),
]
_MATRIX_CELL_W = 5
_MATRIX_TICKER_W_MAX = 9

# ── On-demand factor audit (ChatOps single-ticker) ────────────────────────────
# Fixed-width vertical stack: LABEL(17) + VALUE(6) + " " + NOTE(14) = 38 chars
# per row — equal-length invariant, ASCII-only inside the fence (same rules as
# the SMID desk and factor matrix).
_OD_TITLE_FMT = "── 📊 ON-DEMAND FACTOR AUDIT: {ticker} ──"
_OD_LABEL_W = 17
_OD_VALUE_W = 6
_OD_NOTE_W = 14
_OD_FACTOR_LABELS: Dict[str, str] = {
    "insider_conviction": "Insider Conv",
    "insider_breadth":    "Insider Breadth",
    "congress":           "Congress",
    "news_sentiment":     "News Sentiment",
    "news_buzz":          "News Buzz",
    "momentum_long":      "Momentum 12-1m",
    "volume_attention":   "Volume Attn",
    "analyst_consensus":  "Analyst Cons",
    "analyst_revision":   "EPS Rev",
    "revenue_revision":   "Rev Rev",
    "price_target_upside": "PT Upside",
    "transcript_tone":    "Transcript",
    "quality_piotroski":  "Quality F-Score",
    "fcf_yield":          "FCF Yield",
    "amihud_shock":       "Amihud Liquid",
    "pb_value_up":        "P/B Value",
    "roic_quality":       "ROIC Quality",
    "inst_flow_13f":      "13F Whale Flow",
}

# ── Cognitive factor block (emoji heat in markdown — NEVER inside a fence) ─────
# Heat is keyed on the factor SCORE; ⬜ is reserved for data gaps (signed-None /
# thin coverage) so absence never reads as a weak/bearish 🟥 (CLAUDE.md §2).
_HEAT_STRONG = 0.66
_HEAT_MID = 0.40
_FACTOR_SHORT_LABEL: Dict[str, str] = {
    "insider_conviction": "Insider", "insider_breadth": "InsBrd",
    "congress": "Congress", "news_sentiment": "News", "news_buzz": "Buzz",
    "momentum_long": "Mom", "volume_attention": "Vol", "analyst_consensus": "Analyst",
    "analyst_revision": "EPSrev", "revenue_revision": "RevRev",
    "price_target_upside": "PTUp", "transcript_tone": "Tone",
    "quality_piotroski": "Quality", "fcf_yield": "FCF", "amihud_shock": "Amihud",
    "pb_value_up": "P/B", "roic_quality": "ROIC", "inst_flow_13f": "13F",
}
# [NICHE ALPHA] — low-weight alt-data velocity signals (institutional 13F /
# insider acquired-vs-disposed) surfaced with a 🐋 glyph instead of a heat dot.
_NICHE_ALPHA_FACTORS = frozenset({"inst_flow_13f"})
# Whale-accumulation badge triggers (display only; thresholds on the normalized
# factor + the raw NPR spike). inst_flow_13f is cross-sectionally normalized, so
# ≥ .80 means top-decile institutional inflow this quarter.
_WHALE_FLOW_MIN = 0.80
_WHALE_NPR_SPIKE_MIN = 0.30
# Display-only signal-decay half-lives (days) keyed by dominant catalyst. Mirrors
# the scoring constants (PEAD t½ [Bernard & Thomas, 1989]; insider/congress
# recency decay) — surfaced so a trader sees the signal's expected shelf-life.
_DECAY_HALF_LIFE: Dict[str, int] = {
    "PEAD": 20, "insider": 90, "congress": 90, "momentum": 252,
}

_REGION_KEYS: List[Tuple[str, str, str]] = [
    # (top_lists key, flag, label)
    ("top_buys_usa",    "🇺🇸", "USA"),
    ("top_buys_europe", "🇪🇺", "EUROPE"),
    ("top_buys_asia",   "🌏", "ASIA"),
]
_MEDAL: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}

# ── SMID leverage desk (plain ASCII code block — ANSI stays in the action bar) ─
_SMID_TICKER_W = 9   # ticker.ljust(9) — matches _MATRIX_TICKER_W_MAX
_SMID_LEV_W    = 6   # "0.0000".."1.1000"; leverage_score may exceed 1.0 (ranking key)
_SMID_MOM_W    = 8   # "+999.9%" worst case after the ±9.999 display clamp
_SMID_FLAG_W   = 8   # "E60d F8" worst case
_SMID_HEADER = (
    "TICKER".ljust(_SMID_TICKER_W) + " "
    + "LEV".rjust(_SMID_LEV_W) + " "
    + "vsSPY".rjust(_SMID_MOM_W) + " "
    + "FLAGS".ljust(_SMID_FLAG_W)
)
# Mirrors cook_toplists._SMID_PEAD_WINDOW_DAYS — the flag must agree with the
# PEAD boost window [Bernard & Thomas, 1989].
_SMID_PEAD_WINDOW_DAYS = 60

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

# ── Ticker display names (EU/Asia from registry) ──────────────────────────────
_TICKER_NAMES: Dict[str, str] = {}
try:
    _reg_path = Path(__file__).resolve().parent.parent.parent / "config" / "ticker_registry.json"
    _reg_data = json.loads(_reg_path.read_text(encoding="utf-8"))
    for _reg_entry in _reg_data.get("europe", []) + _reg_data.get("asia", []):
        if _reg_entry.get("ticker") and _reg_entry.get("name"):
            _TICKER_NAMES[_reg_entry["ticker"]] = _reg_entry["name"]
except Exception as _reg_exc:  # registry optional — names are cosmetic
    log.debug("ticker_registry.json not loaded: %s", _reg_exc)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Cast val to float, returning default for None / NaN / Inf / non-numeric."""
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def _fmt_usd(usd: float) -> str:
    if not usd or usd <= 0:
        return "$0"
    if usd >= 100_000:
        return f"${usd/1000:.0f}k"
    value = usd / 1000.0
    formatted = f"${value:.1f}k"
    return formatted.rstrip("0").rstrip(".")


def _truncate(text: str, max_chars: int = _LIMIT_FIELD) -> str:
    """Plain-prose truncation. MUST NOT be used on text containing code
    fences — slicing through ``` orphans the closing fence and spills the
    rest of the embed into terminal context (use structural row-dropping)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


def _data_age_hours(generated_at: str) -> Optional[float]:
    try:
        ts = datetime.fromisoformat((generated_at or "").replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    except ValueError:
        return None


def _compute_percentile(score: float, all_scores: List[float]) -> int:
    """Cross-sectional percentile rank of score within all_scores list."""
    if not all_scores:
        return 0
    below = sum(1 for s in all_scores if s <= score)
    return int(below / len(all_scores) * 100)


def _badge_from_score(score: float) -> str:
    if score >= 0.80:
        return "HIGH BUY"
    if score >= 0.60:
        return "TACTICAL BUY"
    return "WATCHLIST"


def _extension_pct(entry: Dict[str, Any]) -> Optional[float]:
    """Recent run-up (5-session return) for the freshness gate.

    None = unknown (absent / short history / uniquely-stale tape upstream). A
    missing value must NEVER read as a real 0% move — that would wrongly keep an
    unscored-for-freshness name on the actionable desk (CLAUDE.md §2)."""
    val = entry.get("return_5d")
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _extension_threshold(entry: Dict[str, Any]) -> float:
    """Cap-tier-aware run-up gate: wider band for the more-volatile small/mid."""
    tier = (entry.get("cap_tier") or "large").strip().lower()
    return _EXTENSION_PCT_SMID if tier in ("small", "mid") else _EXTENSION_PCT_LARGE


def _is_extended(entry: Dict[str, Any]) -> bool:
    """True when the name has already run past its cap-tier threshold and is a
    chase. Missing data ⇒ not extended (absence is not evidence of a top — keep
    it on the actionable desk rather than hiding it)."""
    ext = _extension_pct(entry)
    return ext is not None and ext >= _extension_threshold(entry)


# ── Target-already-passed gate ────────────────────────────────────────────────
# A name whose analyst consensus target sits at/below the current price has no
# remaining upside — surfacing it as an actionable BUY is incoherent (the desk
# would be buying a name the Street already calls fully valued). Like the
# extension gate this is DISPLAY/RISK only — it never touches final_score,
# weights or factor math; it moves the name off the actionable desk into the
# WATCH section. MIN_TARGET_UPSIDE (default 0.0) is the consensus headroom
# required to STAY actionable: 0.0 gates only already-passed targets; 0.05 would
# require ≥ +5% upside to remain on the desk.
_MIN_TARGET_UPSIDE = float(os.getenv("MIN_TARGET_UPSIDE", "0.0"))


def _target_passed(entry: Dict[str, Any]) -> bool:
    """True when the consensus target is already passed: current ≥ target ×
    (1 + MIN_TARGET_UPSIDE). Missing/zero target or price ⇒ False (absence is not
    evidence of a passed target — never gate on unknown; CLAUDE.md §2)."""
    tgt = _safe_float(entry.get("target_price"))
    cur = _safe_float(entry.get("current_price"))
    if tgt <= 0 or cur <= 0:
        return False
    return tgt < cur * (1.0 + _MIN_TARGET_UPSIDE)


# ── Stale-catalyst (PEAD already played out) gate ─────────────────────────────
# A name carried by a POSITIVE earnings surprise whose recency is now beyond the
# post-earnings-drift window [Bernard & Thomas, 1989] is resting on a SPENT
# catalyst — the drift it was selected for has already happened. Like the other
# gates this is DISPLAY/RISK only; it moves the name off the actionable desk into
# the WATCH section. Names with no positive surprise, or a still-fresh one, are
# NOT gated. PEAD_STALE_DAYS (default = the PEAD window) is env-overridable.
_PEAD_STALE_DAYS = int(os.getenv("PEAD_STALE_DAYS", str(_SMID_PEAD_WINDOW_DAYS)))


def _catalyst_stale(entry: Dict[str, Any]) -> bool:
    """True when the earnings/PEAD catalyst has already played out — a positive
    earnings surprise whose recency exceeds the PEAD drift window. Absent or
    still-fresh surprise ⇒ False (never gate on unknown; CLAUDE.md §2)."""
    pct = entry.get("earnings_surprise_pct")
    if pct is None or _safe_float(pct) <= 0:
        return False
    return int(entry.get("earnings_surprise_days") or 0) > _PEAD_STALE_DAYS


def _off_actionable_desk(entry: Dict[str, Any]) -> bool:
    """Name removed from the actionable buy desk → WATCH section when any of:
    already extended (recent 5d run-up), analyst target already passed (no upside
    left), OR its earnings/PEAD catalyst is spent (drift window elapsed).
    Display/risk gate only — never mutates the score."""
    return _is_extended(entry) or _target_passed(entry) or _catalyst_stale(entry)


def _freshness_token(entry: Dict[str, Any]) -> str:
    """Compact ` · ±X.X% 5d` recency tag for an actionable desk line (empty when
    the recent return is unknown). Negative reads as a pullback — useful entry
    context, so it is shown, not suppressed."""
    ext = _extension_pct(entry)
    if ext is None:
        return ""
    return f" · {ext * 100:+.1f}% 5d"


def _compute_catalyst(entry: Dict[str, Any]) -> str:
    """Evidence-first catalyst narrative, max 80 chars, signals joined by ' · '.

    Priority order (max 3 signals emitted):
      1. INSIDER: insider_usd > 0 → "Insider $Xk [CEO]"
      2. EPS: earnings_surprise_pct → "EPS +X% · Nd ago" / "EPS miss X%"
      3. UPGRADE/DOWNGRADE: recent rating change
      4. CONGRESS: quiver_evidence.congress.purchases > 0
      5. MOMENTUM: |momentum_spy_relative| > 0.05 → "±X% vs SPY 12m"
      6. ANALYST REVISION (fallback only): n_analysts ≥ 5
    """
    signals: List[str] = []

    usd = _safe_float(entry.get("insider_usd"))
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

    upg_raw = entry.get("recent_upgrade_downgrade") or {}
    if isinstance(upg_raw, dict) and upg_raw.get("action") in ("upgrade", "downgrade"):
        action_str = upg_raw["action"].upper()
        firm_raw = upg_raw.get("analyst_firm") or ""
        firm_str = f" {firm_raw[:12]}" if firm_raw else ""
        days_upg = int(upg_raw.get("days_ago") or 0)
        signals.append(f"{action_str}{firm_str} {days_upg}d")

    congress = (entry.get("quiver_evidence") or {}).get("congress", {})
    cg_buys = int(congress.get("purchases", 0) or 0)
    if cg_buys > 0:
        reps = congress.get("representatives") or []
        rep_str = reps[0][:12] if reps else "members"
        signals.append(f"{cg_buys}x congress buy · {rep_str}")

    # INTL entries carry no SPY-relative momentum (cook forwards the absolute
    # listing-market 12-1m return instead) — label honestly, never "vs SPY".
    rel = _safe_float(entry.get("momentum_spy_relative"))
    mom_label = "vs SPY 12m"
    if not rel and entry.get("pipeline") == "INTL":
        rel = _safe_float(entry.get("return_12_1m"))
        mom_label = "12-1m abs"
    if abs(rel) > 0.05:
        sign = "+" if rel >= 0 else ""
        signals.append(f"{sign}{rel * 100:.1f}% {mom_label}")

    if not signals:
        n = int(entry.get("analyst_revision_n_analysts",
                          entry.get("analyst_revision_n", 0)) or 0)
        if n >= 5:
            signals.append(f"analyst revision ({n} analysts)")

    catalyst_str = (" · ".join(signals[:3]) or _NO_CATALYST)[:80]

    # Correlated-signal advisory (Grinold & Kahn 2000: correlated signals
    # overstate effective signal strength).
    if entry.get("_correlated_signal_flag"):
        warning = " ⚠double-signal"
        if len(catalyst_str) + len(warning) <= 80:
            catalyst_str += warning
        else:
            catalyst_str = catalyst_str[:80 - len(warning)] + warning

    # Advisory risk flags for negative signals.
    risk_flags: List[str] = []
    if rel < -0.10:
        risk_flags.append(f"⚠ mom {rel*100:.0f}%")
    rev_score = _safe_float(entry.get("analyst_revision_score"))
    if 0.0 < rev_score < 0.35:
        risk_flags.append("⚠ rev↓")
    quality = _safe_float(entry.get("quality_piotroski_score"))
    if 0.0 < quality < 0.30:
        risk_flags.append("⚠ F-Score↓")
    if risk_flags:
        catalyst_str = f"{catalyst_str} | {' '.join(risk_flags[:2])}"[:80]

    return catalyst_str


def _smid_flags(entry: Dict[str, Any]) -> str:
    """Compact display-meta flags: PEAD recency + Piotroski points.

    'E{days}d' iff positive surprise within the PEAD window (same affirmative-
    evidence rule as the cook boost — absence is NOT bearish, it renders '-').
    'F{n}' recovers the raw 0-8 point count from the normalized points/8
    quality_piotroski_score (see momentum_signals.score_quality_piotroski)."""
    bits: List[str] = []
    pct = entry.get("earnings_surprise_pct")
    days = int(entry.get("earnings_surprise_days") or 0)
    if pct is not None and _safe_float(pct) > 0 and 0 < days <= _SMID_PEAD_WINDOW_DAYS:
        bits.append(f"E{days}d")
    qps = entry.get("quality_piotroski_score")
    if qps is not None and _safe_float(qps) > 0:
        bits.append(f"F{round(_safe_float(qps) * 8):d}")
    return (" ".join(bits) or "-")[:_SMID_FLAG_W]


def _smid_row(entry: Dict[str, Any]) -> str:
    """One fixed-width SMID desk row — every row is the same length as
    _SMID_HEADER by construction (equal-length invariant)."""
    ticker = str(entry.get("ticker") or "?")[:_SMID_TICKER_W].ljust(_SMID_TICKER_W)
    lev = f"{_safe_float(entry.get('leverage_score')):.4f}".rjust(_SMID_LEV_W)
    # Display clamp only (±999.9%) — keeps the column width invariant; the
    # underlying datum is never modified.
    mom = max(-9.999, min(9.999, _safe_float(entry.get("momentum_spy_relative"))))
    mom_s = f"{mom:+.1%}".rjust(_SMID_MOM_W)
    return f"{ticker} {lev} {mom_s} {_smid_flags(entry).ljust(_SMID_FLAG_W)}"


def _market_pulse() -> Dict[str, Any]:
    """Live SPY + QQQ snapshot incl. numeric 63-day returns (fractions), or {}.

    The 63d return feeds the market-regime nowcast (_regime_banner); 1d/12m feed
    the cosmetic price line. Network/data failure degrades to {} (logged, never
    raised) — the brief renders without the snapshot."""
    try:
        from src.data.market_service import MarketData  # noqa: PLC0415

        def _snap(bars) -> dict:
            if bars is None or getattr(bars, "empty", True) or len(bars) < 2:
                return {}
            closes = bars["Close"].dropna()
            if len(closes) < 2:
                return {}
            return {
                "price": float(closes.iloc[-1]),
                "pct_1d": round((closes.iloc[-1] / closes.iloc[-2] - 1) * 100, 2),
                "pct_12m": (round((closes.iloc[-1] / closes.iloc[-252] - 1) * 100, 1)
                            if len(closes) >= 252 else None),
                "ret_63d": (float(closes.iloc[-1] / closes.iloc[-64] - 1)
                            if len(closes) >= 64 else None),
            }

        spy = _snap(MarketData.get_historical_bars("SPY", years_back=2))
        if not spy:
            return {}
        qqq: dict = {}
        try:
            qqq = _snap(MarketData.get_historical_bars("QQQ", years_back=2))
        except Exception:  # QQQ is a confirmation leg — SPY alone is sufficient
            pass
        return {
            "spy_price": spy.get("price"), "spy_1d": spy.get("pct_1d"),
            "spy_12m": spy.get("pct_12m"), "spy_63d": spy.get("ret_63d"),
            "qqq_price": qqq.get("price"), "qqq_1d": qqq.get("pct_1d"),
            "qqq_12m": qqq.get("pct_12m"), "qqq_63d": qqq.get("ret_63d"),
        }
    except Exception as exc:
        log.debug("_market_pulse failed: %s", exc)
        return {}


def _spy_qqq_snapshot(pulse: Optional[Dict[str, Any]] = None) -> str:
    """One-line SPY + QQQ price snapshot, or '' when unavailable."""
    p = pulse if pulse is not None else _market_pulse()
    if not p.get("spy_price"):
        return ""

    def _fmt(v):
        return "—" if v is None else f"{'▲' if v >= 0 else '▼'}{abs(v):.1f}%"

    spy_str = (f"SPY `${p['spy_price']:.0f}` {_fmt(p.get('spy_1d'))} 1d · "
               f"{_fmt(p.get('spy_12m'))} 12m")
    qqq_str = (f"  ·  QQQ `${p['qqq_price']:.0f}` {_fmt(p.get('qqq_1d'))} 1d · "
               f"{_fmt(p.get('qqq_12m'))} 12m" if p.get("qqq_price") else "")
    return f"📈 {spy_str}{qqq_str}"


def _regime_banner(data: Dict[str, Any], pulse: Optional[Dict[str, Any]] = None) -> str:
    """Zone-1 market-regime nowcast (Bull/Euphoria/Bear) from VIX + SPY/QQQ 63d.

    DISPLAY-ONLY (classify_market_regime never re-scales alpha). Prefers the
    plumbed `spy_return_63d`; the live QQQ leg confirms the froth signature."""
    p = pulse if pulse is not None else _market_pulse()
    vix_raw = data.get("vix")
    vix = _safe_float(vix_raw, default=float("nan"))
    spy63 = data.get("spy_return_63d")
    if spy63 is None:
        spy63 = p.get("spy_63d")
    qqq63 = p.get("qqq_63d")
    regime = classify_market_regime(
        vix if vix_raw is not None else float("nan"), spy63, qqq63)
    emoji, blurb = market_regime_label(regime)

    def _pct(x):
        return f"{x * 100:+.1f}%" if isinstance(x, (int, float)) else "—"

    vix_str = f"`{vix:.1f}`" if not math.isnan(vix) else "`—`"
    line1 = f"🧭 **MARKET REGIME — {regime.value}** {emoji} · {blurb}"
    line2 = f"SPY {_pct(spy63)} / QQQ {_pct(qqq63)} (63d) · VIX {vix_str}"
    return f"{line1}\n{line2}"


def _telemetry_line(data: Dict[str, Any]) -> str:
    """FMP bulk-coverage telemetry. '' when the field is absent (pre-plumb
    artifacts); ⚠ banner when coverage < 75% (snapshot likely stale)."""
    cov = data.get("bulk_coverage")
    if cov is None:
        return ""
    frac = _safe_float(cov)
    pct = frac * 100
    if frac < 0.75:
        return (f"🛰 **TELEMETRY ⚠ bulk-cov {pct:.0f}%** — snapshot may be stale; "
                "scores degraded")
    return f"🛰 Telemetry: bulk-cov {pct:.0f}% ✓"


def _factor_heat(score: float, *, unavailable: bool) -> str:
    """Monochrome heat dot. ⬜ = data gap (never bearish); 🟩/🟨/🟥 by strength."""
    if unavailable:
        return "⬜"
    if score >= _HEAT_STRONG:
        return "🟩"
    if score >= _HEAT_MID:
        return "🟨"
    return "🟥"


def _resolve_factor_weights(data: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, float]:
    """Weights for attribution: the payload's own (renormalized) US weights when
    present — so the display matches the engine actually used — else the static
    WEIGHTS_US / WEIGHTS_GLOBAL by market."""
    market = (entry.get("market") or "USA").upper()
    if market in ("EUROPE", "ASIA", "EU"):
        return WEIGHTS_GLOBAL
    w = data.get("weights")
    return w if isinstance(w, dict) and w else WEIGHTS_US


def _missing_factor_set(entry: Dict[str, Any]) -> set:
    """Factor names with absent sources (handles 'factor:reason' encoding)."""
    raw = (entry.get("validation_metadata") or {}).get("missing_sources") or []
    return {str(m).split(":", 1)[0] for m in raw}


def _factor_contributions(
    entry: Dict[str, Any], weights: Dict[str, float], missing: set,
) -> List[Tuple[str, float, float, bool]]:
    """[(factor_key, score, weight*score, unavailable)] over weighted factors.

    weight*score is the pre-overlay attribution (Σ ≈ raw_score). A factor is
    `unavailable` (data gap) when its source is missing or its value is None —
    distinct from a genuine 0.0 dead signal."""
    factors = entry.get("factors") or {}
    out: List[Tuple[str, float, float, bool]] = []
    for key, w in weights.items():
        if not w or w <= 0:
            continue
        raw = factors.get(key)
        unavailable = key in missing or raw is None
        score = _safe_float(raw) if raw is not None else 0.0
        out.append((key, score, w * score, unavailable))
    return out


def _driver_strip(
    entry: Dict[str, Any], weights: Dict[str, float], missing: set, top_n: int = 4,
) -> str:
    """Emoji-heat attribution line: top-N drivers by contribution, plus up to two
    signed-None data gaps surfaced explicitly (⬜ … n/a)."""
    contribs = _factor_contributions(entry, weights, missing)
    if not contribs:
        return ""
    available = sorted((c for c in contribs if not c[3]), key=lambda c: -c[2])
    gaps = sorted((c for c in contribs if c[3]),
                  key=lambda c: -weights.get(c[0], 0.0))
    bits: List[str] = []
    for key, score, _contrib, _ in available[:top_n]:
        label = _FACTOR_SHORT_LABEL.get(key, key[:8])
        # Niche-alpha velocity signals get a 🐋 marker rather than a heat dot.
        glyph = "🐋" if key in _NICHE_ALPHA_FACTORS else _factor_heat(score, unavailable=False)
        bits.append(f"{glyph} {label} {score:.2f}")
    for key, _score, _contrib, _ in gaps[:2]:
        label = _FACTOR_SHORT_LABEL.get(key, key[:8])
        bits.append(f"⬜ {label} n/a")
    return " · ".join(bits)


def _overlay_tag(entry: Dict[str, Any], overlay_mult: float) -> str:
    """`raw 0.91→×0.80` — pre-overlay alpha and the VIX dampening (US only;
    INTL carries no raw_score → '')."""
    raw = entry.get("raw_score")
    if raw is None:
        return ""
    r = _safe_float(raw)
    if r <= 0:
        return ""
    return f"raw {r:.2f}→×{overlay_mult:.2f}"


def _decay_note(entry: Dict[str, Any]) -> str:
    """Dominant-catalyst signal half-life (display-only)."""
    eps_pct = entry.get("earnings_surprise_pct")
    eps_days = int(entry.get("earnings_surprise_days") or 0)
    if eps_pct is not None and _safe_float(eps_pct) > 0 and 0 < eps_days <= 90:
        return f"⏳ PEAD t½≈{_DECAY_HALF_LIFE['PEAD']}d"
    if _safe_float(entry.get("insider_usd")) > 0:
        return f"⏳ insider t½≈{_DECAY_HALF_LIFE['insider']}d"
    if int((entry.get("quiver_evidence") or {}).get("congress", {}).get("purchases", 0) or 0) > 0:
        return f"⏳ congress t½≈{_DECAY_HALF_LIFE['congress']}d"
    return ""


# Home-market currency symbol by exchange suffix — the 🎯 target level renders in
# the currency of the market each name trades on (no FX infra upstream; the % is
# unit-invariant after FMPClient._paired_target_and_price). '.L' (LSE) quotes in
# pence → mapped to '£' and divided by 100 in _target_token (conventional pounds).
# '.SS'/'.SZ' use 'CN¥' to disambiguate the yuan from the Japanese yen ('¥', '.T').
_CCY_SUFFIX: Dict[str, str] = {
    ".PA": "€", ".AS": "€", ".DE": "€", ".F": "€", ".MC": "€", ".MI": "€",
    ".BR": "€", ".HE": "€", ".LS": "€", ".VI": "€",
    ".SW": "CHF",                    # SIX Swiss → Swiss franc
    ".L":  "£",                      # LSE pence → pounds (÷100 in _target_token)
    ".T":  "¥",                      # Tokyo → yen
    ".SS": "CN¥", ".SZ": "CN¥",      # Shanghai / Shenzhen → yuan (≠ JP ¥)
    ".HK": "HK$",                    # Hong Kong dollar
    ".KS": "₩",                      # Korea → won
}


def _ccy_prefix(ticker: str) -> str:
    """Home-market currency symbol from the exchange suffix ('$' default).
    '.L' → '£' (pence→pounds conversion is applied by _target_token)."""
    up = (ticker or "").upper()
    for suf, sym in _CCY_SUFFIX.items():
        if up.endswith(suf):
            return sym
    return "$"


def _target_token(entry: Dict[str, Any]) -> str:
    """🎯 analyst target price + upside — the exit anchor ("know when to sell").

    Empty when either the consensus target or the current price is absent (never
    fabricate a level). target_price / current_price are currency-paired upstream
    (FMPClient._paired_target_and_price), so the % is always sound and the level
    renders in the instrument's home-market currency. ⚠ tgt<px flags a consensus
    target below spot (e.g. consensus lagging a fast momentum name)."""
    tgt = _safe_float(entry.get("target_price"))
    cur = _safe_float(entry.get("current_price"))
    if tgt <= 0 or cur <= 0:
        return ""
    ticker = entry.get("ticker") or ""
    pct = (tgt / cur - 1) * 100
    if ticker.upper().endswith(".L"):
        level = f"£{tgt / 100:.2f}"    # LSE pence (GBX) → conventional pounds
    else:
        level = f"{_ccy_prefix(ticker)}{tgt:.0f}"
    warn = " · ⚠ tgt<px" if tgt < cur else ""
    return f"🎯 tgt {level} ({pct:+.1f}%){warn}"


def _pb_token(entry: Dict[str, Any]) -> str:
    """P/B raw ratio — book-value cheapness alongside the exit target. Omitted
    when price-to-book is absent (never renders 'None')."""
    pb = _safe_float(entry.get("price_to_book"))
    if pb > 0:
        return f"P/B {pb:.1f}×"
    return ""


def _whale_signal(entry: Dict[str, Any]) -> bool:
    """True when alt-data flags extreme whale accumulation — top-decile 13F
    institutional inflow OR an unusual insider acquired-vs-disposed spike."""
    flow = _safe_float((entry.get("factors") or {}).get("inst_flow_13f"))
    if flow >= _WHALE_FLOW_MIN:
        return True
    npr = entry.get("insider_npr") or {}
    return _safe_float(npr.get("spike")) >= _WHALE_NPR_SPIKE_MIN


def _whale_badge(entry: Dict[str, Any]) -> str:
    """🐋 header badge — fires only for a niche small/mid-cap entering on extreme
    alternative-data flow (the rotation signal the desk is built to surface)."""
    cap = (entry.get("cap_tier") or "").lower()
    if cap in ("small", "mid") and _whale_signal(entry):
        return " · 🐋 WHALE ACCUMULATION"
    return ""


def _niche_alpha_strip(entry: Dict[str, Any]) -> str:
    """[NICHE ALPHA] evidence line — the institutional 13F flow + insider
    acquired/disposed velocity behind a whale trigger (CLAUDE.md §5 evidence-first)."""
    bits: List[str] = []
    flow = (entry.get("factors") or {}).get("inst_flow_13f")
    if flow is not None:
        ev = entry.get("inst_13f_evidence") or {}
        dih = ev.get("investors_holding_change")
        tail = f" ({int(dih):+d} holders)" if isinstance(dih, (int, float)) and dih else ""
        bits.append(f"🐋 13F {_safe_float(flow):.2f}{tail}")
    npr = entry.get("insider_npr") or {}
    if npr:
        spike = _safe_float(npr.get("spike"))
        arrow = " ▲" if spike >= _WHALE_NPR_SPIKE_MIN else ""
        bits.append(
            f"insider {int(npr.get('acquired', 0))}B/{int(npr.get('disposed', 0))}S "
            f"NPR {_safe_float(npr.get('npr')):.2f}{arrow}")
    if not bits:
        return ""
    return "[NICHE ALPHA] " + " · ".join(bits)


def _lifecycle_line(entry: Dict[str, Any]) -> str:
    """Zone-3 execution + life-cycle: target upside, relative volume, crowding,
    signal decay. Each token is omitted when its source field is absent."""
    bits: List[str] = []
    tgt = _target_token(entry)
    if tgt:
        bits.append(tgt)
    pb = _pb_token(entry)
    if pb:
        bits.append(pb)
    relvol = _safe_float(entry.get("volume_spike"))
    if relvol > 0:
        bits.append(f"relVol {relvol:.1f}×")
    else:
        va = _safe_float((entry.get("factors") or {}).get("volume_attention"))
        if va > 0:
            bits.append(f"vol-attn {va:.2f}")
    if entry.get("_correlated_signal_flag"):
        bits.append("⚠ crowded (double-signal)")
    decay = _decay_note(entry)
    if decay:
        bits.append(decay)
    return " · ".join(bits)


def _best_per_sector(entries: List[Dict]) -> Dict[str, Tuple[str, float]]:
    """{sector_label: (ticker, final_score)} — the single highest-scoring name
    per sector across the given entries."""
    best: Dict[str, Tuple[str, float]] = {}
    for e in entries:
        raw = (e.get("sector") or (e.get("factors") or {}).get("sector") or "").strip()
        label = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        score = _safe_float(e.get("final_score"))
        cur = best.get(label)
        if cur is None or score > cur[1]:
            best[label] = (e.get("ticker", "?"), score)
    return best


def _sector_heatmap_structured(entries: List[Dict]) -> Dict[str, List[tuple]]:
    buckets: Dict[str, List[tuple]] = {}
    for e in entries:
        raw = (e.get("sector") or e.get("factors", {}).get("sector") or "").strip()
        label = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        ticker = e.get("ticker", "?")
        score = _safe_float(e.get("final_score"))
        buckets.setdefault(label, []).append((ticker, score))
    return {
        lbl: sorted(pairs, key=lambda x: -x[1])[:2]
        for lbl, pairs in buckets.items()
    }


# ── Archive delta loader (lazy, O(1) file reads) ───────────────────────────────

def _load_yesterday_scores(
    archive_root: Path,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, float], Optional[float]]:
    """Load prior-day ticker scores from logs/archive/ for ▲/▼/[NEW] deltas.

    Filenames are YYYY-MM-DD_top_lists.json (name sort == chronological), so
    the loader iterates the reverse-sorted glob lazily and reads ONLY the
    first snapshot dated before today (UTC) — never the whole history.

    Returns (scores, snapshot_age_hours). Missing directory, today-only
    archives, or fully unreadable history → ({}, None). Corrupt candidates
    are logged and skipped in favour of the next older file.
    """
    now = now or datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    root = Path(archive_root)
    if not root.exists():
        return {}, None

    for path in sorted(root.glob("*_top_lists.json"), reverse=True):
        day = path.name[:10]
        if day >= today:
            continue  # today's (or malformed/future-dated) snapshot
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Archive snapshot %s unreadable (%s) — trying older",
                        path.name, exc)
            continue

        scores: Dict[str, float] = {}
        for key in ("top_buys_usa", "top_buys_europe", "top_buys_asia",
                    "watchlist", "top_buys"):
            for e in blob.get(key) or []:
                t = e.get("ticker")
                if t and t not in scores:
                    scores[t] = _safe_float(e.get("final_score"))

        try:
            snap_dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_h: Optional[float] = (now - snap_dt).total_seconds() / 3600
        except ValueError:
            age_h = None
        return scores, age_h

    return {}, None


# ── Payload builder ────────────────────────────────────────────────────────────

class DiscordPayloadBuilder:
    """Builds the institutional daily-brief embed from cooked top_lists.json.

    Themes: NORMAL / BEAR share the standard layout (differing action bar +
    color); CAPITULATION (or kill_switch) swaps desks/portfolio for the
    STRUCTURAL ANCHORS watchlist view. All regime math comes from
    src.risk.regime — thresholds are never hardcoded here.
    """

    MAX_DESK_ENTRIES = 3
    MAX_MATRIX_ROWS = 3  # per region sub-table
    MAX_SMID_ENTRIES = 3  # SMID leverage desk depth

    def __init__(
        self,
        data: Dict[str, Any],
        *,
        yesterday_scores: Optional[Dict[str, float]] = None,
        yesterday_age_h: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.data = data
        self.yesterday_scores = yesterday_scores or {}
        self.yesterday_age_h = yesterday_age_h
        self.now = now or datetime.now(timezone.utc)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return fatal schema problems; empty list means buildable."""
        problems: List[str] = []
        vix = self.data.get("vix")
        if (not isinstance(vix, (int, float)) or isinstance(vix, bool)
                or math.isnan(float(vix)) or float(vix) < 0):
            problems.append(f"vix missing or non-numeric: {vix!r}")
        if not (self.data.get("generated_at") or "").strip():
            problems.append("generated_at missing")
        if "on_demand_ticker" in self.data:
            # ChatOps single-ticker payload — regional keys are absent by
            # design (bulk consumers must never mistake it for a daily brief).
            block = self.data.get("on_demand_ticker")
            entry = block.get("entry") if isinstance(block, dict) else None
            if not isinstance(entry, dict) or not entry.get("ticker"):
                problems.append("on_demand_ticker.entry.ticker missing")
            return problems
        if not any(k in self.data for k in (
                "top_buys_usa", "top_buys_europe", "top_buys_asia", "watchlist")):
            problems.append(
                "no regional top-lists keys present (top_buys_usa / "
                "top_buys_europe / top_buys_asia / watchlist)")
        return problems

    # ── Regime / theme ────────────────────────────────────────────────────────

    @property
    def regime(self) -> RiskRegime:
        if self.data.get("kill_switch"):
            return RiskRegime.CAPITULATION
        return get_regime(float(self.data["vix"]))

    @property
    def multiplier(self) -> float:
        return score_multiplier(self.regime)

    @property
    def overlay_multiplier(self) -> float:
        """VIX macro-overlay actually applied to scores upstream (raw→final).
        4-tier (incl. Crash), unlike the 3-tier display `multiplier`."""
        try:
            return vix_multiplier(float(self.data["vix"]))
        except (KeyError, TypeError, ValueError):
            return 1.0

    @property
    def is_panic(self) -> bool:
        return self.regime == RiskRegime.CAPITULATION

    @property
    def _age_hours(self) -> Optional[float]:
        # Age of the underlying market DATA, not the cook timestamp.
        # `generated_at` is stamped milliseconds before send, so it always
        # reads ~0.0h and can never detect staleness; `data_as_of` (oldest
        # input leg, set by cook_toplists) is the truthful anchor. Fall back
        # to `generated_at` only for legacy artifacts lacking `data_as_of`.
        stamp = self.data.get("data_as_of") or self.data.get("generated_at", "")
        return _data_age_hours(stamp)

    @property
    def _is_stale(self) -> bool:
        age = self._age_hours
        return age is not None and age > _STALE_HOURS

    # ── Data access ───────────────────────────────────────────────────────────

    def _region_entries(self, key: str) -> List[Dict[str, Any]]:
        return list(self.data.get(key) or [])

    def _fresh_entries(self, key: str) -> List[Dict[str, Any]]:
        """Region entries still actionable to enter — NOT already run-up past the
        freshness gate AND NOT already past their analyst target (no upside)."""
        return [e for e in self._region_entries(key) if not _off_actionable_desk(e)]

    def _extended_entries(self, key: str) -> List[Dict[str, Any]]:
        return [e for e in self._region_entries(key) if _off_actionable_desk(e)]

    def _population(self) -> List[float]:
        """All scores in the artifact — denominator for percentile ranks."""
        keys = ("top_buys_usa", "top_buys_europe", "top_buys_asia",
                "usa_overflow", "eu_overflow", "asia_overflow", "watchlist",
                "eu_mid_small", "asia_mid_small")
        return [
            _safe_float(e.get("final_score"))
            for k in keys for e in (self.data.get(k) or [])
        ]

    # ── Section renderers ─────────────────────────────────────────────────────

    def _ansi_bar(self) -> str:
        """Leading ```ansi action bar. ANSI is confined to this block; the
        reset lands before the closing fence and the block terminates with a
        blank line so mobile clients exit terminal context cleanly."""
        regime = self.regime
        color, glyph, action, _ = _REGIME_STYLE[regime]
        vix = float(self.data["vix"])
        age = self._age_hours
        age_str = f"{age:.1f}h" if age is not None else "—"
        tickers = self.data.get("ticker_count", "?")
        line1 = (f"{color}{glyph} {regime.value}{_ANSI_RESET}   "
                 f"VIX {vix:.1f}   OVERLAY ×{self.multiplier:.2f}   {action}")
        line2 = f"{_ANSI_DIM}DATA {age_str} · TICKERS {tickers}{_ANSI_RESET}"
        return f"```ansi\n{line1}\n{line2}\n```\n\n"

    def _macro_description(self) -> str:
        regime = self.regime
        vix = float(self.data["vix"])
        overlay = self.overlay_multiplier
        pulse = _market_pulse()  # one fetch shared by banner + price snapshot
        scaled = (1.0 - overlay) * 100
        scaled_str = ("0%" if scaled <= 0.05
                      else f"−{scaled:.0f}% (overlay ×{overlay:.2f})")
        # Market-regime nowcast leads the readable brief — placed immediately
        # AFTER the ANSI action bar so the description still opens with the
        # ```ansi block (ESC-confinement / mobile-safety contract preserved).
        lines = [
            _regime_banner(self.data, pulse),
            "### 📊 1 · MACRO RISK & REGIME",
            (f"• VIX `{vix:.1f}` — **{regime.value}** · "
             f"{strategy_label(regime)}"),
            (f"• Book alpha scaled {scaled_str} · "
             "Gates: TACTICAL ≥0.60 · HIGH BUY ≥0.80"),
        ]
        snapshot = _spy_qqq_snapshot(pulse)
        if snapshot:
            lines.append(snapshot)
        telemetry = _telemetry_line(self.data)
        if telemetry:
            lines.append(telemetry)
        if self._is_stale:
            lines.append(
                f"⚠️ **DATA STALE — {self._age_hours:.0f}h old** · "
                "check edgar_3x on GitHub Actions")
        if self.is_panic:
            lines.append(
                "🔴 **KILL-SWITCH ACTIVE — BUY SIGNALS SUPPRESSED · "
                "SELL/EXIT SIGNALS REMAIN LIVE**")
        return self._ansi_bar() + "\n".join(lines)

    def _delta_tag(self, ticker: str, score: float) -> str:
        if not self.yesterday_scores:
            return ""
        prev = self.yesterday_scores.get(ticker)
        if prev is None:
            return " [NEW]"
        diff = score - prev
        if abs(diff) < 0.005:
            return ""
        if (self.yesterday_age_h is not None
                and self.yesterday_age_h > _DELTA_STALE_HOURS):
            # Weekend/holiday/outage gap: an arrow over a stale interval is a
            # false momentum signal — tag the interval explicitly instead.
            return f" Δ{diff:+.3f} (>48h)"
        arrow = "▲" if diff > 0 else "▼"
        return f" {arrow}{diff:+.3f}"

    def _desk_lines(self, entry: Dict[str, Any], rank: int,
                    population: List[float], *,
                    show_drivers: bool = True,
                    show_lifecycle: bool = True) -> List[str]:
        ticker = entry.get("ticker", "?")
        score = _safe_float(entry.get("final_score"))
        badge = entry.get("badge") or _badge_from_score(score)
        pct = _compute_percentile(score, population)
        medal = _MEDAL.get(rank, f"#{rank}")
        name = ""
        if ticker in _TICKER_NAMES:
            name = f" ({_TICKER_NAMES[ticker][:14]})"

        cov_warn = ""
        if (entry.get("market") or "").upper() in ("EUROPE", "ASIA", "EU"):
            cov = _safe_float(entry.get("weight_coverage"), 1.0)
            if cov < 0.70:
                cov_warn = f" ⚠COV:{cov:.0%}"

        overlay_tag = _overlay_tag(entry, self.overlay_multiplier)
        overlay_str = f" · {overlay_tag}" if overlay_tag else ""
        whale = _whale_badge(entry)
        head = (f"{medal} **{ticker}**{name} — {badge} · `{score:.4f}` · "
                f"p{pct}{self._delta_tag(ticker, score)}{overlay_str}"
                f"{_freshness_token(entry)}{whale}")
        lines = [head]
        # Zone 2 — cognitive factor attribution (emoji heat, markdown only).
        if show_drivers:
            strip = _driver_strip(
                entry, _resolve_factor_weights(self.data, entry),
                _missing_factor_set(entry))
            if strip:
                lines.append(f"└ {strip}")
        lines.append(f"└ {_compute_catalyst(entry)}{cov_warn}")
        # [NICHE ALPHA] evidence — surfaced only when the whale badge fired,
        # keeping the low-weight alt-data line off every routine large-cap pick.
        if whale:
            niche = _niche_alpha_strip(entry)
            if niche:
                lines.append(f"└ {niche}")
        # Zone 3 — execution + life-cycle. Top pick gets the full line; every
        # other pick still gets the 🎯 target (exit anchor) — "know when to sell".
        if show_lifecycle and rank == 1:
            life = _lifecycle_line(entry)
            if life:
                lines.append(f"└ {life}")
        elif show_lifecycle:
            anchor = " · ".join(t for t in (_target_token(entry), _pb_token(entry)) if t)
            if anchor:
                lines.append(f"└ {anchor}")
        return lines

    def _desk_field(self, key: str, flag: str, label: str,
                    population: List[float],
                    max_entries: int, *,
                    show_drivers: bool = True,
                    show_lifecycle: bool = True) -> Optional[Dict[str, Any]]:
        # Already-moved names are split off to the ⏱ EXTENDED section — the desk
        # carries only still-actionable (fresh) entries.
        entries = self._fresh_entries(key)
        if not entries:
            return None
        lines: List[str] = []
        for rank, entry in enumerate(entries[:max_entries], 1):
            lines.extend(self._desk_lines(
                entry, rank, population,
                show_drivers=show_drivers, show_lifecycle=show_lifecycle))
        return {
            "name":   f"{flag} 2 · ALPHA DESK — {label}",
            "value":  _truncate("\n".join(lines)),
            "inline": False,
        }

    def _smid_field(self, max_entries: int) -> Optional[Dict[str, Any]]:
        """SMID leverage desk — plain ``` block, graceful None when the key
        is absent/empty (backward compat with pre-SMID artifacts)."""
        entries = self._fresh_entries("top_buys_smid")
        if not entries:
            return None
        rows = [_SMID_HEADER] + [_smid_row(e) for e in entries[:max_entries]]
        # Structural fit: drop whole rows, never slice a fenced block.
        while len(rows) > 1 and len("\n".join(rows)) + 8 > _LIMIT_FIELD:
            rows.pop()
        if len(rows) <= 1:
            return None
        # 🐋 marker (outside the ASCII fence) when a sleeve name shows extreme
        # institutional/insider accumulation — the rotation signal at a glance.
        whale_mark = " 🐋" if any(_whale_signal(e) for e in entries[:max_entries]) else ""
        return {
            "name":   f"🚀 2b · SMALL-CAP / MID-CAP LEVERAGE DESK{whale_mark}",
            "value":  "```\n" + "\n".join(rows) + "\n```",
            "inline": False,
        }

    @staticmethod
    def _matrix_cell(factors: Dict[str, Any], factor_key: str) -> str:
        raw = factors.get(factor_key)
        val = _safe_float(raw) if raw is not None else 0.0
        if val <= 0:
            return "-".rjust(_MATRIX_CELL_W)
        return f"{val:.2f}".rjust(_MATRIX_CELL_W)

    def _matrix_rows(self, max_rows: int) -> List[str]:
        """Aligned ASCII rows (no fences — caller wraps exactly once).

        US and EU/ASIA sub-tables both carry 9 fixed-width columns; the
        ticker column is left-justified to one shared width so every line in
        the block has identical length (edge case 6)."""
        us = self._region_entries("top_buys_usa")[:max_rows]
        intl = (self._region_entries("top_buys_europe")[:max_rows]
                + self._region_entries("top_buys_asia")[:max_rows])
        if self.is_panic:
            wl = self._region_entries("watchlist")
            us = [e for e in wl if (e.get("market") or "USA").upper()
                  in ("USA", "US")][:max_rows]
            intl = [e for e in wl if (e.get("market") or "").upper()
                    in ("EUROPE", "ASIA", "EU")][:max_rows]
        if not us and not intl:
            return []

        names = ["USA", "EU/ASIA"] + [e.get("ticker", "?") for e in us + intl]
        width = min(_MATRIX_TICKER_W_MAX, max(len(n) for n in names))

        def _row(name: str, cells: List[str]) -> str:
            return name[:width].ljust(width) + "".join(cells)

        rows: List[str] = []
        if us:
            rows.append(_row("USA", [lbl.rjust(_MATRIX_CELL_W)
                                     for _, lbl in _MATRIX_US_COLS]))
            for e in us:
                rows.append(_row(e.get("ticker", "?"), [
                    self._matrix_cell(e.get("factors") or {}, k)
                    for k, _ in _MATRIX_US_COLS]))
        if intl:
            rows.append(_row("EU/ASIA", [lbl.rjust(_MATRIX_CELL_W)
                                         for _, lbl in _MATRIX_INTL_COLS]))
            for e in intl:
                rows.append(_row(e.get("ticker", "?"), [
                    self._matrix_cell(e.get("factors") or {}, k)
                    for k, _ in _MATRIX_INTL_COLS]))
        return rows

    def _matrix_field(self, max_rows: int) -> Optional[Dict[str, Any]]:
        rows = self._matrix_rows(max_rows)
        if not rows:
            return None
        # Structural budget fit: drop whole rows BEFORE fencing — never slice
        # a rendered code block (edge case 5).
        while rows and len("\n".join(rows)) + 8 > _LIMIT_FIELD:
            rows.pop()
        if not rows:
            return None
        return {
            "name":   "🧬 3 · FACTOR MATRIX",
            "value":  "```\n" + "\n".join(rows) + "\n```",
            "inline": False,
        }

    def _portfolio_field(self) -> Optional[Dict[str, Any]]:
        pools = self.data.get("mvo_pools") or {}
        pool_cfg = [
            ("large_cap_anchors", "🔵", "LARGE-CAP ANCHORS"),
            ("mid_cap",           "🟡", "MID-CAPS"),
            ("small_cap",         "🔴", "SMALL-CAPS"),
        ]
        lines: List[str] = []
        for key, dot, label in pool_cfg:
            pool = pools.get(key) or {}
            positions = pool.get("positions") or []
            if not positions:
                continue
            method = pool.get("method", "equal-weight")
            lines.append(f"{dot} **{label}** · {method} · n={len(positions)}")
            for p in positions[:6]:
                alloc = _safe_float(p.get("allocation")) * 100
                if alloc < 0.1:
                    continue
                floor = _safe_float((p.get("exit_anchors") or {}).get("batch_floor"))
                floor_str = f" · floor ${floor:.0f}" if floor > 0 else ""
                lines.append(f"`{p.get('ticker', '?')}` {alloc:.1f}%{floor_str}")

        shown_entries = [
            e for key, _, _ in _REGION_KEYS
            for e in self._region_entries(key)[:self.MAX_DESK_ENTRIES]
        ]
        heatmap = _sector_heatmap_structured(shown_entries)
        if heatmap:
            sector_bits = [f"{lbl} ({len(pairs)})" for lbl, pairs
                           in sorted(heatmap.items(), key=lambda kv: -len(kv[1]))]
            lines.append("🗺 Sectors: " + " · ".join(sector_bits))

        if not lines:
            return None
        return {
            "name":   "⚖️ 4 · PORTFOLIO CONSTRUCTION",
            "value":  _truncate("\n".join(lines)),
            "inline": False,
        }

    # Region → pools to scan for sector coverage (top buys + sector-cap overflow
    # + the EU/Asia mid/small sleeves) so every sector with exposure is covered.
    _SECTOR_REGION_POOLS: List[Tuple[str, List[str]]] = [
        ("🇺🇸", ["top_buys_usa", "usa_overflow"]),
        ("🇪🇺", ["top_buys_europe", "eu_overflow", "eu_mid_small"]),
        ("🌏", ["top_buys_asia", "asia_overflow", "asia_mid_small"]),
    ]

    def _sector_exposure_field(self) -> Optional[Dict[str, Any]]:
        """One ticker per sector of exposure, within each region (US/EU/Asia).

        Compact emoji-tagged tokens (markdown, not fenced); a region line is
        omitted when its pools are empty (e.g. EU/Asia under thin sessions)."""
        lines: List[str] = []
        for flag, keys in self._SECTOR_REGION_POOLS:
            entries = [e for k in keys for e in self._region_entries(k)]
            best = _best_per_sector(entries)
            if not best:
                continue
            toks = [f"{lbl.split()[0]}{tic} {sc:.2f}"
                    for lbl, (tic, sc) in sorted(best.items(),
                                                 key=lambda kv: -kv[1][1])]
            lines.append(f"{flag} " + " · ".join(toks))
        if not lines:
            return None
        return {
            "name":   "🗺 5 · SECTOR EXPOSURE — best per sector / region",
            "value":  _truncate("\n".join(lines)),
            "inline": False,
        }

    def _anchors_field(self, population: List[float]) -> Dict[str, Any]:
        watchlist = self._region_entries("watchlist")
        if not watchlist:
            return {
                "name":  "🛡 STRUCTURAL ANCHORS",
                "value": ("0 assets met defensive survival thresholds — "
                          "cash allocation 100%"),
                "inline": False,
            }
        lines: List[str] = [
            f"×{self.multiplier:.2f} dampened · force-badged WATCHLIST",
        ]
        for rank, entry in enumerate(watchlist[:self.MAX_DESK_ENTRIES * 2], 1):
            lines.extend(self._desk_lines(entry, rank, population))
        return {
            "name":   "🛡 STRUCTURAL ANCHORS (WATCHLIST)",
            "value":  _truncate("\n".join(lines)),
            "inline": False,
        }

    # Regions scanned for already-moved names, aggregated into one WATCH block.
    _EXTENDED_KEYS: Tuple[str, ...] = (
        "top_buys_usa", "top_buys_europe", "top_buys_asia", "top_buys_smid",
    )
    _MAX_EXTENDED_ENTRIES = 6

    def _extended_field(self) -> Optional[Dict[str, Any]]:
        """⏱ EXTENDED · 🎯 PAST-TARGET (WATCH): names moved off the actionable
        desk — either already ran past the freshness gate OR already past their
        analyst target (no upside). Surfaced (not hidden) so the reader sees
        what's already gone vs chaseable. Aggregated across US/EU/Asia/SMID,
        de-duplicated. Evidence-first (CLAUDE.md §5): ticker · reason · why."""
        seen: set = set()
        entries: List[Dict[str, Any]] = []
        for key in self._EXTENDED_KEYS:
            for e in self._extended_entries(key):
                tic = e.get("ticker")
                if tic in seen:
                    continue
                seen.add(tic)
                entries.append(e)
        if not entries:
            return None
        entries.sort(key=lambda e: -(_extension_pct(e) or 0.0))
        lines: List[str] = []
        for e in entries[:self._MAX_EXTENDED_ENTRIES]:
            tic = e.get("ticker", "?")
            score = _safe_float(e.get("final_score"))
            # Reason precedence: no-upside (target passed) > spent catalyst
            # (PEAD elapsed) > short-term run-up (extended).
            if _target_passed(e):
                tgt = _safe_float(e.get("target_price"))
                cur = _safe_float(e.get("current_price"))
                up = (tgt / cur - 1.0) * 100 if cur > 0 else 0.0
                reason = f"🎯 past target ({up:+.1f}%)"
            elif _catalyst_stale(e):
                days = int(e.get("earnings_surprise_days") or 0)
                reason = f"🍂 PEAD spent ({days}d ago)"
            else:
                reason = f"⏱ already {(_extension_pct(e) or 0.0) * 100:+.1f}% 5d"
            lines.append(
                f"`{tic}` {reason} · `{score:.3f}` — wait for pullback")
            lines.append(f"└ {_compute_catalyst(e)}")
        return {
            "name":   "⏱ EXTENDED · 🎯 PAST-TARGET · 🍂 STALE (WATCH)",
            "value":  _truncate("\n".join(lines)),
            "inline": False,
        }

    @staticmethod
    def _legend_field() -> Dict[str, Any]:
        return {
            "name": "📖 LEGEND",
            "value": (
                "🟩 strong ≥.66 · 🟨 mid · 🟥 weak <.40 · ⬜ data gap (not bearish) · "
                "🐋 [NICHE ALPHA] institutional/insider velocity · "
                "`raw→×` pre-overlay alpha × VIX dampening · "
                "🎯 tgt analyst target in listing ccy (⚠ tgt<px = target below spot) · "
                "`P/B n×` raw price-to-book\n"
                "*IC Insider Conviction · IB Insider Breadth · "
                "CG Congress (US-only) · NS News Sentiment · NB News Buzz · "
                "MO Momentum 12-1m · VA Volume Attention · "
                "AC Analyst Consensus · QF Piotroski Quality · 13F Whale Flow (QoQ) · "
                "FCF Free Cash Flow Yield · AMH Amihud Liquidity · "
                "PB Price-to-Book (matrix score) · ROI ROIC Quality (intl only)*"
            ),
            "inline": False,
        }

    def _footer(self) -> Dict[str, str]:
        run_id = (self.data.get("run_id")
                  or self.data.get("source_run_id") or "local")
        gen = (self.data.get("generated_at") or "")[:16]
        return {"text": f"regime_trader · cook→audit→send · gen {gen} · run {run_id}"}

    def _title(self) -> str:
        try:
            ts = datetime.fromisoformat(
                self.data.get("generated_at", "").replace("Z", "+00:00"))
            stamp = ts.strftime("%b %d %H:%M UTC")
        except ValueError:
            stamp = self.data.get("generated_at", "")[:10] or "—"
        return f"⚡ REGIME TRADER — DAILY BRIEF · {stamp}"

    # ── Assembly ──────────────────────────────────────────────────────────────

    @staticmethod
    def _embed_size(embed: Dict[str, Any]) -> int:
        return (len(embed.get("title", ""))
                + len(embed.get("description", ""))
                + sum(len(f["name"]) + len(f["value"])
                      for f in embed.get("fields", []))
                + len((embed.get("footer") or {}).get("text", "")))

    def _render(self, matrix_rows: int, desk_n: int, *,
                show_drivers: bool = True,
                show_lifecycle: bool = True) -> Dict[str, Any]:
        population = self._population()
        color = _COLOR_RED if (self.is_panic or self._is_stale) \
            else _REGIME_STYLE[self.regime][3]

        fields: List[Dict[str, Any]] = []
        if self.is_panic:
            fields.append(self._anchors_field(population))
            if self._region_entries("watchlist") and matrix_rows:
                matrix = self._matrix_field(matrix_rows)
                if matrix:
                    fields.append(matrix)
        else:
            for key, flag, label in _REGION_KEYS:
                field = self._desk_field(
                    key, flag, label, population, desk_n,
                    show_drivers=show_drivers, show_lifecycle=show_lifecycle)
                if field:
                    fields.append(field)
            # SMID leverage desk — never rendered under CAPITULATION (this is
            # the non-panic branch; cook also empties the pool — defense in
            # depth). Always the full sleeve depth (protected, not tied to the
            # desk-degradation ladder) so the small/mid picks are guaranteed.
            smid = self._smid_field(self.MAX_SMID_ENTRIES)
            if smid:
                fields.append(smid)
            # Already-moved names split off the actionable desks → WATCH block.
            extended = self._extended_field()
            if extended:
                fields.append(extended)
            sector = self._sector_exposure_field()
            if sector:
                fields.append(sector)
            if matrix_rows:
                matrix = self._matrix_field(matrix_rows)
                if matrix:
                    fields.append(matrix)
            portfolio = self._portfolio_field()
            if portfolio:
                fields.append(portfolio)
        fields.append(self._legend_field())

        description = self._macro_description()
        if len(description) > _LIMIT_DESC:  # prose after the ANSI block only
            head, _, tail = description.partition("```\n\n")
            description = head + "```\n\n" + _truncate(
                tail, _LIMIT_DESC - len(head) - 5)

        return {
            "title":       self._title(),
            "description": description,
            "color":       color,
            "fields":      fields[:25],
            "footer":      self._footer(),
            "timestamp":   self.now.isoformat(),
        }

    def build(self) -> Dict[str, Any]:
        """Render the embed, degrading structurally until the 6000-char budget
        holds. Degradation order protects the guaranteed composition: the
        per-pick lifecycle line drops first, then the factor matrix, then the
        driver strips — desk depth (3/region), the SMID sleeve, the sector
        exposure block, the regime banner and the legend are never trimmed."""
        # (matrix_rows, desk_n, show_drivers, show_lifecycle)
        attempts = [
            (self.MAX_MATRIX_ROWS, self.MAX_DESK_ENTRIES, True,  True),
            (self.MAX_MATRIX_ROWS, self.MAX_DESK_ENTRIES, True,  False),
            (2,                    self.MAX_DESK_ENTRIES, True,  False),
            (1,                    self.MAX_DESK_ENTRIES, True,  False),
            (0,                    self.MAX_DESK_ENTRIES, True,  False),
            (0,                    self.MAX_DESK_ENTRIES, False, False),
            (0,                    2,                     False, False),
            (0,                    1,                     False, False),
        ]
        embed = self._render(*attempts[0][:2], show_drivers=attempts[0][2],
                             show_lifecycle=attempts[0][3])
        for matrix_rows, desk_n, drivers, lifecycle in attempts:
            embed = self._render(matrix_rows, desk_n,
                                 show_drivers=drivers, show_lifecycle=lifecycle)
            if self._embed_size(embed) <= _LIMIT_TOTAL:
                break
        return {"embeds": [embed]}

    # ── On-demand single-ticker audit (ChatOps) ───────────────────────────────

    @staticmethod
    def _on_demand_stack_rows(entry: Dict[str, Any]) -> List[str]:
        """Fixed-width vertical factor stack — every row 38 chars (equal-length
        invariant), ASCII-only, '-' for zero/absent values (matrix convention),
        'missing' note for schema-gate flagged sources."""
        def _row(label: str, value: str, note: str = "") -> str:
            return (label[:_OD_LABEL_W].ljust(_OD_LABEL_W)
                    + value[:_OD_VALUE_W].rjust(_OD_VALUE_W) + " "
                    + note[:_OD_NOTE_W].ljust(_OD_NOTE_W))

        score = _safe_float(entry.get("final_score"))
        badge = entry.get("badge") or _badge_from_score(score)
        is_intl = (entry.get("market") or "USA").upper() in ("EUROPE", "ASIA", "EU")
        cols = _MATRIX_INTL_COLS if is_intl else _MATRIX_US_COLS
        missing = set(
            (entry.get("validation_metadata") or {}).get("missing_sources") or []
        )
        factors = entry.get("factors") or {}

        rows = [
            _row("FACTOR", "VALUE", "NOTE"),
            _row("FINAL SCORE", f"{score:.4f}", badge),
        ]
        raw = entry.get("raw_score")
        if raw is not None:
            rows.append(_row("ALPHA (RAW)", f"{_safe_float(raw):.4f}", "pre-overlay"))
        for key, _lbl in cols:
            val = factors.get(key)
            v = _safe_float(val) if val is not None else 0.0
            rows.append(_row(
                _OD_FACTOR_LABELS.get(key, key),
                "-" if v <= 0 else f"{v:.2f}",
                "missing" if key in missing else "",
            ))
        return rows

    def build_on_demand(self) -> Dict[str, Any]:
        """Render the on-demand single-ticker factor audit embed.

        Layout contract: title `── 📊 ON-DEMAND FACTOR AUDIT: {ticker} ──`,
        ANSI confined to the leading action bar (reset before the closing
        fence), plain-ASCII fixed-width factor stack, catalyst + disclosure
        + legend fields. Kill-switch/regime styling comes from the same
        properties as the daily brief — never bypassed.
        """
        block = self.data.get("on_demand_ticker") or {}
        entry = block.get("entry") or {}
        ticker = block.get("ticker") or entry.get("ticker", "?")
        score = _safe_float(entry.get("final_score"))
        badge = entry.get("badge") or _badge_from_score(score)
        regime = self.regime
        color = _COLOR_RED if (self.is_panic or self._is_stale) \
            else _REGIME_STYLE[regime][3]

        name = ""
        if ticker in _TICKER_NAMES:
            name = f" ({_TICKER_NAMES[ticker][:14]})"
        verdict = [
            (f"**{ticker}**{name} — **{badge}** · `{score:.4f}` · "
             f"{block.get('pipeline', '?')} pipeline"),
            (f"• Regime **{regime.value}** · VIX `{float(self.data['vix']):.1f}` "
             f"· overlay `×{self.multiplier:.2f}` · {strategy_label(regime)}"),
        ]
        if self._is_stale:
            verdict.append(f"⚠️ **DATA STALE — {self._age_hours:.0f}h old**")
        if self.is_panic:
            verdict.append(
                "🔴 **KILL-SWITCH ACTIVE — BUY SIGNALS SUPPRESSED · "
                "SELL/EXIT SIGNALS REMAIN LIVE**")
        description = self._ansi_bar() + "\n".join(verdict)

        # Structural budget fit: drop whole factor rows, never slice a fence.
        rows = self._on_demand_stack_rows(entry)
        while len(rows) > 2 and len("\n".join(rows)) + 8 > _LIMIT_FIELD:
            rows.pop()
        fields: List[Dict[str, Any]] = [
            {
                "name":   "🧬 FACTOR STACK",
                "value":  "```\n" + "\n".join(rows) + "\n```",
                "inline": False,
            },
            {
                "name":   "🎯 CATALYST",
                "value":  _truncate(f"└ {_compute_catalyst(entry)}"),
                "inline": False,
            },
        ]

        disclosure = [
            (f"• Scoring: **{block.get('scoring_mode', 'absolute')}** "
             "(single ticker — no peer-group normalization)"),
        ]
        cov = entry.get("weight_coverage")
        if cov is not None:
            cov_f = _safe_float(cov, 1.0)
            warn = " ⚠COV" if cov_f < 0.70 else ""
            disclosure.append(f"• Weight coverage: `{cov_f:.0%}`{warn}")
        missing = (entry.get("validation_metadata") or {}).get("missing_sources") or []
        if missing:
            _no_cov = [m.split(":")[0] for m in missing if m.endswith(":no_coverage")]
            _api_err = [m.split(":")[0] for m in missing if m.endswith(":api_error")]
            _plain = [m for m in missing if ":" not in m]
            if _no_cov:
                disclosure.append("• No coverage: " + ", ".join(_no_cov[:8]))
            if _api_err:
                disclosure.append("• API errors: " + ", ".join(_api_err[:8]))
            if _plain:
                disclosure.append("• Missing sources: " + ", ".join(_plain[:8]))
        fields.append({
            "name":   "⚠️ DATA DISCLOSURE",
            "value":  _truncate("\n".join(disclosure)),
            "inline": False,
        })
        fields.append(self._legend_field())

        embed = {
            "title":       _OD_TITLE_FMT.format(ticker=ticker),
            "description": description,
            "color":       color,
            "fields":      fields,
            "footer":      self._footer(),
            "timestamp":   self.now.isoformat(),
        }
        return {"embeds": [embed]}

    # ── Alert (DATA UNAVAILABLE contract) ─────────────────────────────────────

    @classmethod
    def build_alert(cls, reason: str) -> Dict[str, Any]:
        """High-visibility alert when input is missing/corrupt/invalid.

        Title MUST contain "DATA UNAVAILABLE" — asserted by
        .github/workflows/test_daily_toplists_absence.yml on embeds[0].title.
        """
        return {
            "embeds": [{
                "title":       "⚠️ Alpha Pipeline — DATA UNAVAILABLE",
                "description": (
                    f"**Reason:** {reason}\n\n"
                    "The pipeline may not have completed its last run.\n"
                    "Check `edgar_3x` / `daily_trading_pipeline` on GitHub Actions."
                ),
                "color":       _COLOR_RED,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "footer":      {"text": "regime_trader · cook→audit→send"},
            }]
        }


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

def _print_payload(payload: Dict[str, Any]) -> None:
    sys.stdout.buffer.write(
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send the institutional daily brief to Discord")
    parser.add_argument(
        "--input", type=Path, default=Path("logs/top_lists.json"),
        help="Path to cooked top_lists.json (default: logs/top_lists.json)",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="Directory for discord_send.log and the archive/ delta snapshots",
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
        "--max-data-age-hours", type=float, default=float(_STALE_HOURS),
        help="If the brief's underlying data is older than this, still send "
             "(the DATA STALE banner is in the embed) but exit non-zero so a "
             f"local scheduler can flag the run (default: {_STALE_HOURS}).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_dir / "discord_send.log", encoding="utf-8"),
        ],
    )

    webhook: str = args.webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook and not args.dry_run:
        log.critical(
            "DISCORD_WEBHOOK_URL is not set and --webhook was not supplied. "
            "Cannot deliver output. Exiting with code 2."
        )
        return 2  # config error — distinct from data errors

    if not args.input.exists():
        log.warning("%s not found — sending alert", args.input)
        payload = DiscordPayloadBuilder.build_alert(f"File not found: {args.input}")
        if args.dry_run:
            _print_payload(payload)
            return 0
        return 0 if send_to_discord(webhook, payload) else 1

    try:
        status = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("could not parse %s: %s", args.input.name, exc)
        payload = DiscordPayloadBuilder.build_alert(f"JSON parse error: {exc}")
        if args.dry_run:
            _print_payload(payload)
            return 0
        send_to_discord(webhook, payload)
        return 1

    # Pre-flight schema audit — catches score-range violations, badge
    # mismatches, sort-order errors and cross-contamination before Discord.
    try:
        from src.delivery.audit_payload import (  # noqa: PLC0415
            audit as _audit,
            PipelineAuditError as _PAE,
        )
        try:
            _audit(str(args.input))
        except _PAE as _schema_exc:
            log.error("Pre-flight audit FAILED: %s", _schema_exc)
            alert = DiscordPayloadBuilder.build_alert(
                f"AUDIT GATE FAILED: {_schema_exc}")
            if not args.dry_run:
                send_to_discord(webhook, alert)
            return 1
    except ImportError as _imp_exc:
        log.warning("audit_payload not importable — skipping pre-flight: %s",
                    _imp_exc)

    yesterday_scores, yesterday_age_h = _load_yesterday_scores(
        args.log_dir / "archive")
    builder = DiscordPayloadBuilder(
        status,
        yesterday_scores=yesterday_scores,
        yesterday_age_h=yesterday_age_h,
    )

    problems = builder.validate()
    if problems:
        log.error("Schema validation failed: %s", "; ".join(problems))
        payload = DiscordPayloadBuilder.build_alert(
            "Schema validation failed: " + "; ".join(problems))
        if args.dry_run:
            _print_payload(payload)
            return 0
        send_to_discord(webhook, payload)
        return 1

    # On-demand single-ticker payloads render the factor-audit layout; the
    # daily brief path is untouched (key-presence dispatch).
    if "on_demand_ticker" in status:
        payload = builder.build_on_demand()
    else:
        payload = builder.build()

    # Stale-data guard: the brief is delivered regardless (the DATA STALE
    # banner is already in the embed), but a too-old run returns a distinct
    # non-zero exit so the local scheduler flags it instead of trusting
    # silently-old data. Exit code 3 — distinct from send (1) / config (2).
    age = builder._age_hours
    stale_exit = age is not None and age > args.max_data_age_hours
    if stale_exit:
        log.error(
            "Data is %.1fh old (> --max-data-age-hours %.1f): sending brief "
            "with STALE banner and exiting non-zero.",
            age, args.max_data_age_hours,
        )

    if args.dry_run:
        _print_payload(payload)
        return 3 if stale_exit else 0

    if not send_to_discord(webhook, payload):
        log.error("All Discord send attempts failed")
        return 1
    log.info("Daily brief sent successfully")
    return 3 if stale_exit else 0


if __name__ == "__main__":
    sys.exit(main())
