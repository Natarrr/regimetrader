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
    strategy_label,
)

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


def _spy_qqq_snapshot() -> str:
    """Return a one-line market snapshot for SPY + QQQ, or '' on failure."""
    try:
        from backend.data.market_service import MarketData  # noqa: PLC0415

        def _snap(bars) -> dict:
            if bars is None or getattr(bars, "empty", True) or len(bars) < 2:
                return {}
            closes = bars["Close"].dropna()
            if len(closes) < 2:
                return {}
            price = float(closes.iloc[-1])
            pct_1d = round((closes.iloc[-1] / closes.iloc[-2] - 1) * 100, 2)
            pct_12m = (round((closes.iloc[-1] / closes.iloc[-252] - 1) * 100, 1)
                       if len(closes) >= 252 else None)
            return {"price": price, "pct_1d": pct_1d, "pct_12m": pct_12m}

        def _fmt(v):
            if v is None:
                return "—"
            return f"{'▲' if v >= 0 else '▼'}{abs(v):.1f}%"

        spy = _snap(MarketData.get_historical_bars("SPY", years_back=2))
        if not spy:
            return ""
        qqq: dict = {}
        try:
            qqq = _snap(MarketData.get_historical_bars("QQQ", years_back=2))
        except Exception:
            pass

        spy_str = (f"SPY `${spy['price']:.0f}` {_fmt(spy.get('pct_1d'))} 1d · "
                   f"{_fmt(spy.get('pct_12m'))} 12m")
        qqq_str = (f"  ·  QQQ `${qqq['price']:.0f}` {_fmt(qqq.get('pct_1d'))} 1d · "
                   f"{_fmt(qqq.get('pct_12m'))} 12m" if qqq else "")
        return f"📈 {spy_str}{qqq_str}"
    except Exception as exc:
        log.debug("_spy_qqq_snapshot failed: %s", exc)
        return ""


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
    def is_panic(self) -> bool:
        return self.regime == RiskRegime.CAPITULATION

    @property
    def _age_hours(self) -> Optional[float]:
        return _data_age_hours(self.data.get("generated_at", ""))

    @property
    def _is_stale(self) -> bool:
        age = self._age_hours
        return age is not None and age > _STALE_HOURS

    # ── Data access ───────────────────────────────────────────────────────────

    def _region_entries(self, key: str) -> List[Dict[str, Any]]:
        return list(self.data.get(key) or [])

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
        lines = [
            "### 📊 1 · MACRO RISK & REGIME",
            (f"• VIX `{vix:.1f}` — **{regime.value}** · "
             f"{strategy_label(regime)}"),
            (f"• Risk multiplier `×{self.multiplier:.2f}` · "
             "Action gates: TACTICAL ≥0.60 · HIGH BUY ≥0.80"),
        ]
        snapshot = _spy_qqq_snapshot()
        if snapshot:
            lines.append(snapshot)
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
                    population: List[float]) -> List[str]:
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

        head = (f"{medal} **{ticker}**{name} — {badge} · `{score:.4f}` · "
                f"p{pct}{self._delta_tag(ticker, score)}")
        detail = f"└ {_compute_catalyst(entry)}{cov_warn}"
        return [head, detail]

    def _desk_field(self, key: str, flag: str, label: str,
                    population: List[float],
                    max_entries: int) -> Optional[Dict[str, Any]]:
        entries = self._region_entries(key)
        if not entries:
            return None
        lines: List[str] = []
        for rank, entry in enumerate(entries[:max_entries], 1):
            lines.extend(self._desk_lines(entry, rank, population))
        return {
            "name":   f"{flag} 2 · ALPHA DESK — {label}",
            "value":  _truncate("\n".join(lines)),
            "inline": False,
        }

    def _smid_field(self, max_entries: int) -> Optional[Dict[str, Any]]:
        """SMID leverage desk — plain ``` block, graceful None when the key
        is absent/empty (backward compat with pre-SMID artifacts)."""
        entries = self._region_entries("top_buys_smid")
        if not entries:
            return None
        rows = [_SMID_HEADER] + [_smid_row(e) for e in entries[:max_entries]]
        # Structural fit: drop whole rows, never slice a fenced block.
        while len(rows) > 1 and len("\n".join(rows)) + 8 > _LIMIT_FIELD:
            rows.pop()
        if len(rows) <= 1:
            return None
        return {
            "name":   "🚀 2b · SMALL-CAP / MID-CAP LEVERAGE DESK",
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

    @staticmethod
    def _legend_field() -> Dict[str, Any]:
        return {
            "name": "📖 LEGEND",
            "value": (
                "*IC Insider Conviction · IB Insider Breadth · "
                "CG Congress (US-only) · NS News Sentiment · NB News Buzz · "
                "MO Momentum 12-1m · VA Volume Attention · "
                "AC Analyst Consensus · QF Piotroski Quality · "
                "FCF Free Cash Flow Yield · AMH Amihud Liquidity · "
                "PB Price-to-Book · ROI ROIC Quality (intl only)*"
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

    def _render(self, matrix_rows: int, desk_n: int) -> Dict[str, Any]:
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
                field = self._desk_field(key, flag, label, population, desk_n)
                if field:
                    fields.append(field)
            # SMID leverage desk — never rendered under CAPITULATION (this is
            # the non-panic branch; cook also empties the pool — defense in
            # depth). min(·, desk_n) keeps the build() ladder monotone.
            smid = self._smid_field(min(self.MAX_SMID_ENTRIES, desk_n))
            if smid:
                fields.append(smid)
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
        """Render the embed, degrading structurally until the 6000-char
        message budget holds (matrix rows first, then desk depth — the macro
        section and legend are never trimmed)."""
        attempts = [
            (self.MAX_MATRIX_ROWS, self.MAX_DESK_ENTRIES),
            (2, self.MAX_DESK_ENTRIES),
            (1, self.MAX_DESK_ENTRIES),
            (0, self.MAX_DESK_ENTRIES),
            (0, 2),
            (0, 1),
        ]
        embed = self._render(*attempts[0])
        for matrix_rows, desk_n in attempts:
            embed = self._render(matrix_rows, desk_n)
            if self._embed_size(embed) <= _LIMIT_TOTAL:
                break
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

    payload = builder.build()
    if args.dry_run:
        _print_payload(payload)
        return 0

    if not send_to_discord(webhook, payload):
        log.error("All Discord send attempts failed")
        return 1
    log.info("Daily brief sent successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
