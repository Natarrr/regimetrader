"""
streamlit_app.py
────────────────
Streamlit web dashboard for regime_trader.

TABS
────
  📊 Live Monitor    — Regime · Portfolio · Positions · Risk · System
  🧠 Market Intel    — Composite scores · Source breakdown · Decisions
  📋 Trade Log       — Parsed trades.log with P&L chart and filters
  📈 Regime History  — Regime transitions and probability timeline
  🔄 Portfolio Sync  — Upload brokerage CSV and preview/execute sync

DATA SOURCES (in priority order)
─────────────────────────────────
  1. Alpaca API        (live equity, positions — requires credentials in .env)
  2. monitoring logs   (NDJSON files in LOG_DIR)
  3. Demo stub data    (if no logs and no API credentials)

RUN
───
  cd regime_trader
  streamlit run streamlit_app.py

  # Or from the project root:
  streamlit run regime_trader/streamlit_app.py
"""

from __future__ import annotations
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta, timezone

import asyncio
import json
import logging
import math as _math
import os
import subprocess
import sys
import time

_dm_logger = logging.getLogger("decision_matrix")


# ── Bootstrap: load .env from project root ────────────────────────────────────

_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env", override=False)

# ══════════════════════════════════════════════════════════════════════════════
# DISCOVERY SCANNER  (inlined from discovery_scanner.py)
# ══════════════════════════════════════════════════════════════════════════════

_DISC_CACHE_FILE = _HERE / "logs" / "discovery_cache.json"
_DISC_CACHE_TTL = 6 * 3600   # seconds
_PRESORT_LIMIT = 20

_CAPS: Dict[str, tuple] = {
    "mid_cap":   (2_000_000_000,  10_000_000_000),
    "small_cap": (300_000_000,   2_000_000_000),
}

_FALLBACK: Dict[str, List[str]] = {
    "mid_cap": [
        "PINS", "SNAP", "DKNG", "RBLX", "PATH", "SMAR", "ZI",
        "AFRM", "UPST", "SOFI", "WBA", "M",   "KSS",  "GPS",
        "OPEN", "HCP", "ESTC", "NEOG", "FFIV", "AVLR",
    ],
    "small_cap": [
        "JOBY", "ACHR", "IONQ", "QUBT", "RGTI", "QBTS",
        "ACMR", "FORM", "WOLF", "SMMT", "PTCT", "FOLD",
        "PRTA", "ACAD", "ITCI", "RGEN", "VCNX", "NKLA",
        "BLNK", "CHPT",
    ],
}


def _disc_get_json(url: str, params: Dict, timeout: int = 15) -> Optional[Any]:
    try:
        import requests as _req
        resp = _req.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[discovery] HTTP error — {url}: {exc}")
        return None


def _fmp_screener(cap_min: int, cap_max: int, limit: int = 200) -> List[Dict]:
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return []
    data = _disc_get_json(
        "https://financialmodelingprep.com/api/v3/stock-screener",
        params={
            "marketCapMoreThan": cap_min,
            "marketCapLessThan": cap_max,
            "volumeMoreThan":    100_000,
            "isEtf":             "false",
            "isActivelyTrading": "true",
            "country":           "US",
            "limit":             limit,
            "apikey":            api_key,
        },
    )
    return data if isinstance(data, list) else []


def _fmp_insider_buys(lookback_days: int = 30, limit: int = 200) -> Dict[str, float]:
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return {}
    data = _disc_get_json(
        "https://financialmodelingprep.com/api/v4/insider-trading",
        params={"limit": limit, "apikey": api_key},
    )
    if not isinstance(data, list) or not data:
        return {}
    cutoff = datetime.now() - timedelta(days=lookback_days)
    buy_value: Dict[str, float] = {}
    for tx in data:
        tx_type = str(tx.get("transactionType", "")).upper()
        date_str = str(tx.get("transactionDate", tx.get("date", "")) or "")
        reporting_name = str(tx.get("reportingName", "")).upper()
        if not ("PURCHASE" in tx_type or "P-PURCHASE" in tx_type):
            continue
        if not any(keyword in reporting_name for keyword in ["CEO", "CFO", "DIRECTOR", "CHIEF", "PRESIDENT"]):
            continue
        try:
            tx_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if tx_date < cutoff:
            continue
        sym = str(tx.get("symbol", "")).strip().upper()
        shares = float(tx.get("securitiesTransacted", 0) or 0)
        price = float(tx.get("price", 0) or 0)
        if sym:
            buy_value[sym] = buy_value.get(sym, 0.0) + abs(shares * price)
    print(
        f"[discovery] Insider pre-screener: {len(buy_value)} symbols with buys in last {lookback_days}d")
    return buy_value


def _fmp_profile(sym: str) -> Optional[Dict]:
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return None
    try:
        data = _disc_get_json(
            f"https://financialmodelingprep.com/api/v3/profile/{sym}",
            params={"apikey": api_key},
        )
        return data[0] if isinstance(data, list) and data else None
    except Exception:
        return None


def _fmp_institutional_accumulation(limit: int = 200) -> Dict[str, float]:
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return {}
    data = _disc_get_json(
        "https://financialmodelingprep.com/api/v3/institutional-ownership/list",
        params={"limit": limit, "apikey": api_key},
    )
    if not isinstance(data, list):
        return {}
    ownership = {}
    for item in data:
        sym = str(item.get("symbol", "")).upper()
        total_shares = float(item.get("totalShares", 0) or 0)
        inst_shares = float(item.get("totalInstitutionalShares", 0) or 0)
        pct = inst_shares / total_shares if total_shares > 0 else 0
        ownership[sym] = pct
    print(f"[discovery] Institutional ownership: {len(ownership)} symbols")
    return ownership


def _liquidity_filter(tickers: List[Dict], min_dollar_vol: float = 1_000_000) -> List[Dict]:
    return [
        t for t in tickers
        if float(t.get("price", 0) or 0) > 0.50
        and float(t.get("price", 0) or 0) * float(t.get("volume", 0) or 0) >= min_dollar_vol
    ]


def _enrich_with_momentum(candidates: List[Dict]) -> List[Dict]:
    import concurrent.futures as _cf
    import yfinance as _yf

    syms = [str(r.get("symbol", "")) for r in candidates if r.get("symbol")]
    if not syms:
        return candidates

    def _fetch_mom(sym: str):
        try:
            import pandas as _pd
            df = _yf.download(sym, period="25d", interval="1d",
                              progress=False, auto_adjust=True)
            if df.empty or len(df) < 6:
                return sym, 1.0, 0.0
            if isinstance(df.columns, _pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            close = df["Close"].squeeze().astype(float)
            volume = df["Volume"].squeeze().astype(float)
            avg_vol = float(volume.iloc[:-1].mean()
                            ) if len(volume) > 1 else 1.0
            last_vol = float(volume.iloc[-1])
            spike = round(last_vol / max(avg_vol, 1), 2)
            price_chg = round((float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 2) \
                if len(close) >= 6 else 0.0
            return sym, spike, price_chg
        except Exception:
            return sym, 1.0, 0.0

    with _cf.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_mom, s): s for s in syms}
        enrichment: Dict[str, tuple] = {}
        for fut in _cf.as_completed(futures):
            sym, spike, pchg = fut.result()
            enrichment[sym] = (spike, pchg)

    for r in candidates:
        sym = str(r.get("symbol", ""))
        spike, pchg = enrichment.get(sym, (1.0, 0.0))
        r["volume_spike"] = spike
        r["price_change_pct"] = pchg
    return candidates


def _select_candidates(
    screener_rows: List[Dict],
    insider_buys:  Dict[str, float],
    institutional: Dict[str, float],
    cap_min: int,
    cap_max: int,
    n: int = _PRESORT_LIMIT,
) -> List[Dict]:
    screener_map = {str(r.get("symbol", "")).upper(): r for r in screener_rows}
    candidates = []
    chosen_syms = set()

    # First, add top insider buys
    for sym, value in sorted(insider_buys.items(), key=lambda x: x[1], reverse=True):
        if sym in chosen_syms:
            continue
        if sym in screener_map:
            row = screener_map[sym].copy()
        else:
            profile = _fmp_profile(sym)
            if not profile:
                continue
            cap = float(profile.get("mktCap", 0) or 0)
            if not (cap_min <= cap <= cap_max):
                continue
            row = {
                "symbol": sym,
                "marketCap": cap,
                "price": float(profile.get("price", 0) or 0),
                "volume": float(profile.get("volAvg", 0) or 0),
            }
        cap = float(row.get("marketCap", 0) or 0)
        norm_insider = value / cap if cap > 0 else value
        inst_score = institutional.get(sym, 0.0)
        smart_score = norm_insider + inst_score
        row["smart_score"] = smart_score
        candidates.append(row)
        chosen_syms.add(sym)
        if len(candidates) >= n:
            break

    # Then, add top institutional if not already included
    for sym, score in sorted(institutional.items(), key=lambda x: x[1], reverse=True):
        if sym in chosen_syms:
            continue
        if sym in screener_map:
            row = screener_map[sym].copy()
        else:
            profile = _fmp_profile(sym)
            if not profile:
                continue
            cap = float(profile.get("mktCap", 0) or 0)
            if not (cap_min <= cap <= cap_max):
                continue
            row = {
                "symbol": sym,
                "marketCap": cap,
                "price": float(profile.get("price", 0) or 0),
                "volume": float(profile.get("volAvg", 0) or 0),
            }
        cap = float(row.get("marketCap", 0) or 0)
        norm_insider = insider_buys.get(
            sym, 0.0) / cap if cap > 0 else insider_buys.get(sym, 0.0)
        inst_score = score
        smart_score = norm_insider + inst_score
        row["smart_score"] = smart_score
        candidates.append(row)
        chosen_syms.add(sym)
        if len(candidates) >= n:
            break

    # Sort by smart_score descending
    candidates.sort(key=lambda r: r.get("smart_score", 0), reverse=True)
    chosen = candidates[:n]
    chosen_syms = {r.get("symbol", "") for r in chosen}

    # Fill remaining slots with top volume from screener not already chosen
    remaining_slots = n - len(chosen)
    if remaining_slots > 0:
        rest = sorted(
            [r for r in screener_rows if str(
                r.get("symbol", "")).upper() not in chosen_syms],
            key=lambda r: float(r.get("price", 1) or 1) *
            float(r.get("volume", 0) or 0),
            reverse=True,
        )
        for r in rest[:remaining_slots]:
            r_copy = r.copy()
            r_copy["smart_score"] = 0.0  # or calculate if needed
            chosen.append(r_copy)

    insider_count = len([r for r in chosen if insider_buys.get(
        str(r.get("symbol", "")).upper(), 0) > 0])
    inst_count = len([r for r in chosen if institutional.get(
        str(r.get("symbol", "")).upper(), 0) > 0])
    screener_count = len(chosen) - insider_count - inst_count + len([r for r in chosen if insider_buys.get(str(r.get(
        # adjust for overlap
        "symbol", "")).upper(), 0) > 0 and institutional.get(str(r.get("symbol", "")).upper(), 0) > 0])
    print(
        f"[discovery] Candidates: {len(chosen)} total ({insider_count} insider, {inst_count} institutional, {screener_count} screener-only)")
    return chosen


def _point_fort(ps: Dict[str, float], direction: str) -> str:
    # Handle None values - treat as missing data, not as 0.5
    inst = ps.get("inst")
    ins = ps.get("ins")
    sent = ps.get("sent")
    news = ps.get("news")
    mac = ps.get("macro")

    # Helper to check if value exists and meets threshold
    def has_value(val, threshold):
        return val is not None and val < threshold

    def has_value_high(val, threshold):
        return val is not None and val >= threshold

    if direction == "short":
        if has_value(inst, 0.40) and has_value(ins, 0.40):
            return "Institutions + Insiders Selling"
        if has_value(inst, 0.35):
            return "Institutional Distribution"
        if has_value(ins, 0.35):
            return "Insider Selling Pressure"
        if has_value(sent, 0.30):
            return "Negative Sentiment Spike"
        return "Bearish Multi-Pillar Alignment"

    # Long direction
    if has_value_high(ins, 0.80):
        return "Massive Insider Buying"
    if has_value_high(inst, 0.75):
        return "Institutional Whale Accumulation"
    if inst is not None and ins is not None and inst > 0.65 and ins > 0.65:
        return "Smart Money Convergence"
    if sent is not None and news is not None and sent > 0.70 and news > 0.65:
        return "Social + News Momentum"
    if mac is not None and inst is not None and mac > 0.70 and inst > 0.55:
        return "Macro Tailwind"
    if has_value_high(sent, 0.75):
        return "High Social Momentum"
    if has_value_high(news, 0.70):
        return "Strong News Flow"
    if has_value_high(ins, 0.60):
        return "Insider Buying Activity"
    if inst is not None and sent is not None and inst > 0.60 and sent > 0.60:
        return "Institutional + Sentiment Alignment"
    if has_value_high(inst, 0.70):
        return "Institutional Accumulation"
    return "Multi-Pillar Alpha Signal"


async def _scan_category(
    syms:      List[str],
    price_map: Dict[str, float],
    cap_map:   Dict[str, float],
    limit:     int = 5,
) -> List[Dict]:
    from intelligence.engine import score_tickers_batch
    results = await score_tickers_batch(syms)
    scanned_at = datetime.now(tz=timezone.utc).isoformat()
    scored: List[Dict] = []
    for sym in syms:
        if sym not in results:
            continue
        signal, intel, details = results[sym]

        # Handle None values - distinguish missing data from neutral (0.5)
        ps = {}
        for pillar in ("sent", "ins", "inst", "news", "macro"):
            raw_score = details.get(pillar, {}).get("score")
            # Keep None for missing data, use 0.5 only for display purposes
            ps[pillar] = raw_score if raw_score is not None else None

        # Filter out tickers where ALL pillars have no data (None)
        # Only skip if ALL of ins, inst, sent, news are None
        missing_pillars = [p for p in (
            "ins", "inst", "sent", "news") if ps[p] is None]
        if len(missing_pillars) >= 4:
            # All 4 main pillars are missing - skip this ticker
            print(
                f"[discovery] Skipping {sym} - all pillars missing: {missing_pillars}")
            continue

        scored.append({
            "symbol":         sym,
            "price":          price_map.get(sym, 0.0),
            "market_cap":     cap_map.get(sym, 0.0),
            "conviction":     round(intel.final_conviction, 4),
            "conviction_pct": round(intel.final_conviction * 100, 1),
            "direction":      signal.direction,
            "label":          intel.label,
            "point_fort":     _point_fort(ps, signal.direction),
            "confidence":     round(intel.confidence_level, 4),
            "pillar_scores":  ps,
            "scanned_at":     scanned_at,
        })
    if not scored:
        print(
            "[discovery] WARNING: 0 tickers with real signal data — check API keys & limits.")
    scored.sort(key=lambda s: abs(s["conviction"] - 0.5), reverse=True)
    return scored[:limit]


async def _run_scan(limit: int = 5) -> Dict[str, List[Dict]]:
    insider_buys = _fmp_insider_buys(lookback_days=30, limit=200)
    institutional = _fmp_institutional_accumulation(limit=200)
    results: Dict[str, List[Dict]] = {}

    async def _one_cat(cat: str, cap_min: int, cap_max: int, insider_buys: Dict[str, float], institutional: Dict[str, float]) -> None:
        screener_rows = _fmp_screener(cap_min, cap_max)
        if screener_rows:
            liquid = _liquidity_filter(screener_rows)
            candidates = _select_candidates(
                liquid, insider_buys, institutional, cap_min, cap_max, n=_PRESORT_LIMIT)
            candidates = _enrich_with_momentum(candidates)
            # Calculate final smart score with momentum
            for r in candidates:
                sym = str(r.get("symbol", "")).upper()
                cap = float(r.get("marketCap", 0) or 0)
                insider_val = insider_buys.get(sym, 0.0)
                norm_insider = insider_val / cap if cap > 0 else insider_val
                inst_score = institutional.get(sym, 0.0)
                volume_spike = r.get("volume_spike", 1.0)
                price_change_pct = r.get("price_change_pct", 0.0)
                momentum_score = volume_spike * \
                    (1 + abs(price_change_pct) / 100)
                final_smart_score = 0.4 * norm_insider + \
                    0.4 * inst_score + 0.2 * momentum_score
                r["final_smart_score"] = final_smart_score
            candidates.sort(key=lambda r: r.get(
                "final_smart_score", 0), reverse=True)
            price_map = {str(r.get("symbol", "")): float(
                r.get("price",     0) or 0) for r in candidates}
            cap_map = {str(r.get("symbol", "")): float(
                r.get("marketCap", 0) or 0) for r in candidates}
            syms = [str(r.get("symbol", "")) for r in candidates]
        else:
            syms, price_map, cap_map = _FALLBACK[cat], {}, {}
            print(
                f"[discovery] {cat}: using static fallback ({len(syms)} tickers)")
        print(f"[discovery] {cat}: scoring {len(syms)} tickers")
        results[cat] = await _scan_category(syms, price_map, cap_map, limit=limit)

    await asyncio.gather(
        _one_cat("mid_cap",   *_CAPS["mid_cap"], insider_buys, institutional),
        _one_cat("small_cap", *_CAPS["small_cap"],
                 insider_buys, institutional),
    )
    return results


async def get_top_alpha_picks(limit: int = 5) -> Dict[str, Any]:
    cached = _load_disc_cache()
    if cached is not None:
        return cached
    data = await _run_scan(limit=limit)
    payload = _build_disc_payload(data)
    _save_disc_cache(payload)
    return payload


def get_top_alpha_picks_sync(limit: int = 5) -> Dict[str, Any]:
    try:
        return asyncio.run(get_top_alpha_picks(limit=limit))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(get_top_alpha_picks(limit=limit))
        finally:
            loop.close()


def force_refresh_sync(limit: int = 5) -> Dict[str, Any]:
    _DISC_CACHE_FILE.unlink(missing_ok=True)
    return get_top_alpha_picks_sync(limit=limit)


def _build_disc_payload(data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    now = time.time()
    expires_at = now + _DISC_CACHE_TTL
    return {
        "mid_cap":          data.get("mid_cap",   []),
        "small_cap":        data.get("small_cap", []),
        "cached":           False,
        "computed_at":      datetime.fromtimestamp(now,        tz=timezone.utc).isoformat(),
        "cache_expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "_expires_at":      expires_at,
    }


def _load_disc_cache() -> Optional[Dict[str, Any]]:
    try:
        if not _DISC_CACHE_FILE.exists():
            return None
        raw = json.loads(_DISC_CACHE_FILE.read_text(encoding="utf-8"))
        if time.time() > float(raw.get("_expires_at", 0)):
            return None
        out = {k: v for k, v in raw.items() if not k.startswith("_")}
        out["cached"] = True
        return out
    except Exception:
        return None


def _save_disc_cache(payload: Dict[str, Any]) -> None:
    _DISC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DISC_CACHE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MARKET INTEL MACRO  (inlined from market_intel_macro.py)
# ══════════════════════════════════════════════════════════════════════════════
COMMODITY_UNIVERSE: List[Dict[str, Any]] = [
    {"name": "Crude Oil",   "ticker": "CL=F", "etf": "USO",
        "sector": "Energy",      "unit": "$/bbl"},
    {"name": "Brent Crude", "ticker": "BZ=F", "etf": "BNO",
        "sector": "Energy",      "unit": "$/bbl"},
    {"name": "Natural Gas", "ticker": "NG=F", "etf": "UNG",
        "sector": "Energy",      "unit": "$/MMBtu"},
    {"name": "Gold",        "ticker": "GC=F", "etf": "GLD",
        "sector": "Metals",      "unit": "$/oz"},
    {"name": "Silver",      "ticker": "SI=F", "etf": "SLV",
        "sector": "Metals",      "unit": "$/oz"},
    {"name": "Copper",      "ticker": "HG=F", "etf": "CPER",
        "sector": "Metals",      "unit": "$/lb"},
    {"name": "Wheat",       "ticker": "ZW=F", "etf": "WEAT",
        "sector": "Agriculture", "unit": "¢/bu"},
    {"name": "Corn",        "ticker": "ZC=F", "etf": "CORN",
        "sector": "Agriculture", "unit": "¢/bu"},
    {"name": "Soybeans",    "ticker": "ZS=F", "etf": "SOYB",
        "sector": "Agriculture", "unit": "¢/bu"},
]

MACRO_INDICATORS: List[Dict[str, str]] = [
    {"name": "US 10Y Yield",  "ticker": "^TNX",     "unit": "%"},
    {"name": "Dollar Index",  "ticker": "DX-Y.NYB", "unit": "pts"},
    {"name": "VIX",           "ticker": "^VIX",     "unit": "pts"},
    {"name": "3M T-Bill",     "ticker": "^IRX",     "unit": "%"},
]

FUTURES_TO_ETF: Dict[str, str] = {
    c["ticker"]: c["etf"] for c in COMMODITY_UNIVERSE
}

SECTOR_STOCKS: Dict[str, List[Dict[str, str]]] = {
    "Energy": [
        {"ticker": "XOM",  "name": "Exxon Mobil"},
        {"ticker": "CVX",  "name": "Chevron"},
        {"ticker": "PXD",  "name": "Pioneer Natural Resources"},
    ],
    "Metals": [
        {"ticker": "GOLD", "name": "Barrick Gold"},
        {"ticker": "NEM",  "name": "Newmont"},
        {"ticker": "FCX",  "name": "Freeport-McMoRan"},
    ],
    "Agriculture": [
        {"ticker": "ADM",  "name": "Archer-Daniels-Midland"},
        {"ticker": "DE",   "name": "John Deere"},
        {"ticker": "CTVA", "name": "Corteva"},
    ],
}

SECTOR_FALLBACKS: Dict[str, str] = {
    "XOM": "SHEL", "CVX": "SHEL", "PXD": "SHEL",
    "GOLD": "KL",  "NEM": "KL",   "FCX": "KL",
    "ADM": "BG",   "DE": "AGCO",  "CTVA": "MOS",
}

SECTOR_COMMODITY_MAP: Dict[str, List[str]] = {
    "Energy":      ["CL=F", "BZ=F"],
    "Metals":      ["GC=F", "HG=F"],
    "Agriculture": ["ZW=F", "ZC=F"],
}

STOCK_REGIME_MULT: Dict[str, float] = {
    "Mania":    1.00,
    "Euphoria": 1.15,
    "Bull":     1.20,
    "Neutral":  1.00,
    "Unknown":  0.90,
    "Bear":     0.65,
    "Panic":    0.40,
    "Crash":    0.00,
}

_INSTITUTIONAL_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}
_INSTITUTIONAL_TTL = 86_400.0

_BULL_WORDS: frozenset = frozenset([
    "beat", "beats", "exceed", "exceeds", "exceeded", "upgrade", "upgrades",
    "upgraded", "buy", "outperform", "outperforms", "strong", "record", "rally",
    "rallies", "surge", "surges", "surged", "gain", "gains", "growth", "bullish",
    "profit", "profits", "revenue", "raise", "raises", "raised", "tops",
    "topped", "jump", "jumps", "jumped", "soar", "soars", "soared", "boom",
    "positive", "breakthrough", "approval", "approved", "expands", "expansion",
])
_BEAR_WORDS: frozenset = frozenset([
    "miss", "misses", "missed", "downgrade", "downgrades", "downgraded",
    "sell", "underperform", "underperforms", "concern", "concerns", "decline",
    "declines", "declined", "weak", "weakness", "loss", "losses", "cut",
    "cuts", "cutting", "fall", "falls", "fell", "drop", "drops", "dropped",
    "recession", "layoff", "layoffs", "lawsuit", "fine", "fined", "warning",
    "warn", "warns", "risk", "risks", "volatile", "volatility", "below",
    "disappoints", "disappointing", "disappointed", "halt", "halted",
    "investigation", "probe", "fraud", "bankruptcy", "default",
])

_APE_CACHE: Dict[str, Any] = {"rows": {}, "ts": 0.0}
_APE_TTL = 300.0


def _safe_download(ticker: str, period: str = "1y") -> Optional[Any]:
    try:
        import yfinance as _yf
        import pandas as _pd
        df = _yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
        if df.empty or len(df) < 10:
            return None
        return df
    except Exception:
        return None


def fetch_commodity_prices(commodity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    import pandas as _pd
    for ticker, source in [(commodity["ticker"], "futures"), (commodity["etf"], "etf")]:
        if not ticker:
            continue
        df = _safe_download(ticker)
        if df is None:
            continue
        close = df["Close"].squeeze()
        if isinstance(close, _pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 20:
            continue
        price = float(close.iloc[-1])
        prev1 = float(close.iloc[-2]) if len(close) >= 2 else price
        prev5 = float(close.iloc[-6]) if len(close) >= 6 else price
        prev20 = float(close.iloc[-21]) if len(close) >= 21 else price
        ret_1d = (price / prev1 - 1) if prev1 > 0 else 0.0
        ret_5d = (price / prev5 - 1) if prev5 > 0 else 0.0
        ret_20d = (price / prev20 - 1) if prev20 > 0 else 0.0
        window = min(252, len(close))
        high_52 = float(close.rolling(window).max().iloc[-1])
        low_52 = float(close.rolling(window).min().iloc[-1])
        pct_52 = (price - low_52) / max(high_52 - low_52, 1e-9)
        sma20 = float(close.rolling(20).mean(
        ).iloc[-1]) if len(close) >= 20 else price
        sma50 = float(close.rolling(50).mean(
        ).iloc[-1]) if len(close) >= 50 else price
        sma200 = float(close.rolling(200).mean(
        ).iloc[-1]) if len(close) >= 200 else price
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi14 = float(max(0.0, min(100.0, (100 - 100 / (1 + rs)).iloc[-1])))
        try:
            hi = df["High"].squeeze()
            lo = df["Low"].squeeze()
            if isinstance(hi, _pd.DataFrame):
                hi = hi.iloc[:, 0]
            if isinstance(lo, _pd.DataFrame):
                lo = lo.iloc[:, 0]
            tr = _pd.concat([(hi - lo), (hi - close.shift()).abs(),
                             (lo - close.shift()).abs()], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
        except Exception:
            atr14 = price * 0.015
        return {
            "name":    commodity["name"],
            "ticker":  commodity["ticker"],
            "etf":     commodity["etf"],
            "sector":  commodity["sector"],
            "unit":    commodity["unit"],
            "source":  source,
            "price":   round(price, 4),
            "ret_1d":  round(ret_1d, 6),
            "ret_5d":  round(ret_5d, 6),
            "ret_20d": round(ret_20d, 6),
            "high_52": round(high_52, 4),
            "low_52":  round(low_52, 4),
            "pct_52":  round(pct_52, 4),
            "sma20":   round(sma20, 4),
            "sma50":   round(sma50, 4),
            "sma200":  round(sma200, 4),
            "rsi14":   round(rsi14, 2),
            "atr14":   round(atr14, 4),
            "_close":  close,
        }
    return None


def fetch_macro_indicator(ticker: str) -> Optional[Dict[str, Any]]:
    import pandas as _pd
    df = _safe_download(ticker, period="60d")
    if df is None:
        return None
    close = df["Close"].squeeze()
    if isinstance(close, _pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return None
    price = float(close.iloc[-1])
    prev1 = float(close.iloc[-2]) if len(close) >= 2 else price
    prev5 = float(close.iloc[-6]) if len(close) >= 6 else price
    prev20 = float(close.iloc[-21]) if len(close) >= 21 else price
    high_52 = float(close.rolling(min(252, len(close))).max().iloc[-1])
    low_52 = float(close.rolling(min(252, len(close))).min().iloc[-1])
    return {
        "price":   round(price, 4),
        "ret_1d":  round((price / prev1 - 1) if prev1 > 0 else 0, 6),
        "ret_5d":  round((price / prev5 - 1) if prev5 > 0 else 0, 6),
        "ret_20d": round((price / prev20 - 1) if prev20 > 0 else 0, 6),
        "high_52": round(high_52, 4),
        "low_52":  round(low_52, 4),
        "_close":  close,
    }


def calc_term_structure_score(data: Dict[str, Any]) -> tuple:
    ret_5d = data.get("ret_5d",  0.0)
    ret_20d = data.get("ret_20d", 0.0)
    price = data.get("price",   1.0)
    sma50 = data.get("sma50",   price)
    sma200 = data.get("sma200",  price)
    daily_5d = ret_5d / 5
    daily_20d = ret_20d / 20 if ret_20d != 0 else 0.0
    acceleration = daily_5d - daily_20d
    accel_score = max(0.0, min(1.0, 0.5 + acceleration * 40))
    if price > sma200 and sma50 > sma200:
        trend_score = 0.82
    elif price > sma200:
        trend_score = 0.63
    elif price > sma50:
        trend_score = 0.42
    else:
        trend_score = 0.18
    score = round(0.50 * accel_score + 0.50 * trend_score, 4)
    if score >= 0.72:
        label = "Backwardation ↑↑"
    elif score >= 0.58:
        label = "Mild Backwardation ↑"
    elif score >= 0.42:
        label = "Flat / Neutral"
    elif score >= 0.28:
        label = "Mild Contango ↓"
    else:
        label = "Contango ↓↓"
    return score, label


def calc_cot_proxy_score(data: Dict[str, Any]) -> tuple:
    pct_52 = data.get("pct_52", 0.5)
    rsi14 = data.get("rsi14",  50.0)
    ret_5d = data.get("ret_5d", 0.0)
    base = 1.0 - pct_52
    if pct_52 < 0.15 and rsi14 > 32:
        score = min(0.95, base + 0.18)
        label = "STRONGLY BULLISH — Insider Accumulation"
    elif pct_52 < 0.25:
        score = min(0.85, base + 0.08)
        label = "Bullish — Commercial Buying"
    elif pct_52 < 0.45:
        score = base + 0.03
        label = "Mildly Bullish"
    elif pct_52 < 0.60:
        score = base
        label = "Neutral"
    elif pct_52 < 0.78:
        score = max(0.15, base - 0.05)
        label = "Bearish — Commercial Hedging"
    else:
        score = max(0.05, base - 0.12)
        label = "STRONGLY BEARISH — Commercial Selling"
    score += ret_5d * 0.25
    return round(max(0.0, min(1.0, score)), 4), label


def calc_sentiment_score(etf: str, sentiment_map: Dict[str, float]) -> tuple:
    raw = sentiment_map.get(etf, 0.5)
    score = round(1.0 - 0.80 * raw, 4)
    if raw >= 0.80:
        label = f"⚠ Extreme Retail Bullish ({raw:.0%}) — Contrarian Sell"
    elif raw >= 0.65:
        label = f"Retail Bullish ({raw:.0%}) — Caution"
    elif raw >= 0.45:
        label = f"Neutral ({raw:.0%})"
    elif raw >= 0.30:
        label = f"Retail Bearish ({raw:.0%}) — Contrarian Buy"
    else:
        label = f"Extreme Retail Bearish ({raw:.0%}) — Strong Buy Signal"
    return score, label


def calc_trend_score(data: Dict[str, Any]) -> tuple:
    price = data.get("price",  1.0)
    sma50 = data.get("sma50",  price)
    sma200 = data.get("sma200", price)
    rsi14 = data.get("rsi14",  50.0)
    if price > sma200 and sma50 > sma200:
        tc, tl = 0.88, "Golden Cross ▲"
    elif price > sma200:
        tc, tl = 0.65, "Above 200MA"
    elif price > sma50:
        tc, tl = 0.40, "Between MAs"
    else:
        tc, tl = 0.15, "Death Cross ▼"
    if rsi14 < 30:
        rc, rl = 0.90, f"Oversold ({rsi14:.0f})"
    elif rsi14 < 45:
        rc, rl = 0.72, f"Recovering ({rsi14:.0f})"
    elif rsi14 < 60:
        rc, rl = 0.55, f"Neutral ({rsi14:.0f})"
    elif rsi14 < 70:
        rc, rl = 0.33, f"Extended ({rsi14:.0f})"
    else:
        rc, rl = 0.12, f"Overbought ({rsi14:.0f})"
    score = round(0.60 * tc + 0.40 * rc, 4)
    label = f"{tl} · RSI {rl}"
    return score, label


def calc_macro_conviction(
    price_data:    Dict[str, Any],
    sentiment_map: Dict[str, float],
) -> Dict[str, Any]:
    ts_s,   ts_l = calc_term_structure_score(price_data)
    cot_s,  cot_l = calc_cot_proxy_score(price_data)
    etf = price_data.get("etf", "")
    sent_s, sent_l = calc_sentiment_score(etf, sentiment_map)
    tr_s,   tr_l = calc_trend_score(price_data)
    composite = round(
        0.30 * ts_s + 0.30 * cot_s + 0.20 * sent_s + 0.20 * tr_s, 4,
    )
    composite = max(0.0, min(1.0, composite))
    if composite >= 0.72:
        cv_lbl, cv_clr = "Strong Buy",  "#00c851"
    elif composite >= 0.58:
        cv_lbl, cv_clr = "Buy",         "#7cb342"
    elif composite >= 0.42:
        cv_lbl, cv_clr = "Neutral",     "#9e9e9e"
    elif composite >= 0.28:
        cv_lbl, cv_clr = "Reduce",      "#ff8800"
    else:
        cv_lbl, cv_clr = "Avoid",        "#ff4444"
    return {
        "composite":        composite,
        "conviction_label": cv_lbl,
        "conviction_clr":   cv_clr,
        "ts_score":   ts_s,   "ts_label":   ts_l,
        "cot_score":  cot_s,  "cot_label":  cot_l,
        "sent_score": sent_s, "sent_label": sent_l,
        "tr_score":   tr_s,   "tr_label":   tr_l,
    }


def check_macro_shocks(
    prices: Dict[str, Optional[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    crude = prices.get("CL=F") or {}
    wheat = prices.get("ZW=F") or {}
    gold = prices.get("GC=F") or {}
    copper = prices.get("HG=F") or {}
    ng = prices.get("NG=F") or {}
    oil_5d = crude.get("ret_5d", 0.0)
    if oil_5d >= 0.05:
        alerts.append({"level": "error", "icon": "🔥",
                       "message": (f"Macro Shock — Energy Inflation: Crude Oil {oil_5d:+.1%} in 5 days. "
                                   "Equity margins at risk. Reduce high-beta long exposure.")})
    elif oil_5d >= 0.03:
        alerts.append({"level": "warning", "icon": "⚠",
                       "message": (f"Energy Watch: Crude Oil {oil_5d:+.1%} in 5 days. "
                                   "Monitor margin compression in consumer/industrial sectors.")})
    wheat_5d = wheat.get("ret_5d", 0.0)
    if wheat_5d >= 0.10:
        alerts.append({"level": "error", "icon": "🌾",
                       "message": (f"Macro Shock — Food Inflation: Wheat {wheat_5d:+.1%} in 5 days. "
                                   "Consumer staples margins at risk. Rotate to agricultural producers.")})
    elif wheat_5d >= 0.05:
        alerts.append({"level": "warning", "icon": "⚠",
                       "message": (f"Food Inflation Watch: Wheat {wheat_5d:+.1%} in 5 days. "
                                   "Monitor consumer staples guidance.")})
    gold_5d = gold.get("ret_5d", 0.0)
    copper_5d = copper.get("ret_5d", 0.0)
    if gold_5d > 0.02 and copper_5d < -0.03:
        alerts.append({"level": "error", "icon": "📉",
                       "message": (f"Recession Warning — Copper/Gold ratio crashing. "
                                   f"Gold {gold_5d:+.1%} · Copper {copper_5d:+.1%} (5d). "
                                   "Defensive posture recommended. Reduce cyclical exposure.")})
    elif gold_5d > 0.01 and copper_5d < -0.01:
        alerts.append({"level": "warning", "icon": "🔶",
                       "message": (f"Liquidity Watch: Gold outperforming Copper — flight-to-safety pattern. "
                                   f"Gold {gold_5d:+.1%} · Copper {copper_5d:+.1%} (5d).")})
    ng_5d = ng.get("ret_5d", 0.0)
    if ng_5d >= 0.12:
        alerts.append({"level": "warning", "icon": "⚡",
                       "message": (f"Natural Gas {ng_5d:+.1%} in 5 days. "
                                   "Utility cost pressures rising. Monitor industrial sector margins.")})
    return alerts


def generate_macro_synthesis(
    prices:      Dict[str, Optional[Dict[str, Any]]],
    convictions: Dict[str, Dict[str, Any]],
    indicators:  Dict[str, Optional[Dict[str, Any]]],
) -> List[str]:
    paras: List[str] = []
    crude = prices.get("CL=F") or {}
    brent = prices.get("BZ=F") or {}
    gold = prices.get("GC=F") or {}
    copper = prices.get("HG=F") or {}
    wheat = prices.get("ZW=F") or {}
    corn = prices.get("ZC=F") or {}
    crude_cv = convictions.get("CL=F", {})
    gold_cv = convictions.get("GC=F", {})
    copper_cv = convictions.get("HG=F", {})
    tnx = indicators.get("^TNX") or {}
    dxy = indicators.get("DX-Y.NYB") or {}
    vix = indicators.get("^VIX") or {}
    if crude:
        ts_l = crude_cv.get("ts_label", "")
        cot_l = crude_cv.get("cot_label", "")
        conv_l = crude_cv.get("conviction_label", "Neutral")
        oil_5d = crude.get("ret_5d", 0.0)
        rsi = crude.get("rsi14", 50.0)
        if "Backwardation" in ts_l and "Bullish" in cot_l:
            paras.append(
                f"CRUDE OIL [{conv_l}] — Steep backwardation with commercial accumulation confirmed. "
                f"Physical supply is dangerously tight ({oil_5d:+.1%} 5-day · RSI {rsi:.0f}). "
                f"Energy equities (XLE, XOP) expected to outperform. "
                f"Term structure signals near-term squeeze over forward delivery."
            )
        elif "Contango" in ts_l:
            paras.append(
                f"CRUDE OIL [{conv_l}] — Market in contango — structural oversupply signal. "
                f"Negative roll yield for long ETF holders (USO). "
                f"Energy sector faces headwinds; prefer E&P names with strong balance sheets."
            )
        else:
            paras.append(
                f"CRUDE OIL [{conv_l}] — Mixed signals. Structure: {ts_l}. "
                f"COT proxy: {cot_l}. 5-day return {oil_5d:+.1%}."
            )
    if gold and copper:
        gold_5d = gold.get("ret_5d", 0.0)
        copper_5d = copper.get("ret_5d", 0.0)
        gold_p = gold.get("price", 0.0)
        copper_p = copper.get("price", 0.0)
        cu_au = round(copper_p / gold_p * 1000, 2) if gold_p > 0 else 0
        if gold_5d > 0.015 and copper_5d < -0.02:
            paras.append(
                f"METALS [DEFENSIVE] — Classic flight-to-safety divergence. "
                f"Gold {gold_5d:+.1%} vs Copper {copper_5d:+.1%} (5d). "
                f"Cu/Au ratio: {cu_au:.2f} (×1000). Dr. Copper signalling economic contraction. "
                f"Increase defensive allocation: GLD, TLT. Reduce materials/industrials."
            )
        elif copper_5d > 0.02:
            paras.append(
                f"METALS [RISK-ON] — Copper strength ({copper_5d:+.1%} 5d) signals "
                f"industrial demand recovery. Cu/Au ratio: {cu_au:.2f} (×1000). "
                f"Bullish for materials (XLB), industrials (XLI), and broad risk assets."
            )
        else:
            gold_conv = gold_cv.get("conviction_label", "Neutral")
            paras.append(
                f"METALS [NEUTRAL] — Gold {gold_5d:+.1%} · Copper {copper_5d:+.1%} (5d). "
                f"Cu/Au ratio {cu_au:.2f} (×1000). No strong directional signal. "
                f"Gold conviction: {gold_conv}."
            )
    if wheat or corn:
        w5 = wheat.get("ret_5d", 0.0) if wheat else 0.0
        c5 = corn.get("ret_5d",  0.0) if corn else 0.0
        if abs(w5) > 0.04 or abs(c5) > 0.04:
            direction = "surging" if (w5 + c5) > 0 else "collapsing"
            impact = "food cost pressures rising" if (
                w5 + c5) > 0 else "food cost pressures easing"
            paras.append(
                f"AGRICULTURE [{direction.upper()}] — Wheat {w5:+.1%} · Corn {c5:+.1%} (5d). "
                f"{impact.capitalize()}. "
                + ("Monitor consumer staples margins and emerging market sovereign risk."
                   if (w5 + c5) > 0 else
                   "Positive for restaurant chains, food manufacturers, EM importers.")
            )
    notes: List[str] = []
    if tnx:
        tnx_val = tnx.get("price", 0.0)
        tnx_5d = tnx.get("ret_5d", 0.0)
        if tnx_val > 4.5:
            notes.append(
                f"10Y yield at {tnx_val:.2f}% — deeply restrictive; credit spreads at risk")
        elif tnx_5d > 0.02:
            notes.append(
                f"10Y rising ({tnx_val:.2f}%, {tnx_5d:+.1%}) — growth/inflation expectations lifting")
        else:
            notes.append(f"10Y at {tnx_val:.2f}%")
    if dxy:
        dxy_val = dxy.get("price", 0.0)
        dxy_5d = dxy.get("ret_5d", 0.0)
        if dxy_5d > 0.01:
            notes.append(
                f"USD strengthening ({dxy_val:.1f}, {dxy_5d:+.1%}) — headwind for commodities")
        elif dxy_5d < -0.01:
            notes.append(
                f"USD weakening ({dxy_val:.1f}, {dxy_5d:+.1%}) — commodity tailwind")
        else:
            notes.append(f"USD stable ({dxy_val:.1f})")
    if vix:
        vix_val = vix.get("price", 0.0)
        if vix_val > 30:
            notes.append(
                f"VIX {vix_val:.1f} — extreme fear; commodity volatility elevated")
        elif vix_val > 20:
            notes.append(f"VIX {vix_val:.1f} — elevated uncertainty")
        else:
            notes.append(f"VIX {vix_val:.1f} — calm conditions")
    if notes:
        paras.append("MACRO BACKDROP — " + " · ".join(notes) + ".")
    if not paras:
        paras.append("Insufficient data to generate macro synthesis. "
                     "Click Refresh Macro Data to fetch live prices.")
    return paras


def fetch_stock_pick_data(ticker: str, fallback: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        import yfinance as _yf
    except ImportError:
        return None
    import math as _m
    for tk in ([ticker] + ([fallback] if fallback else [])):
        try:
            _t = _yf.Ticker(tk)
            hist = _t.history(period="1y", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 30:
                continue
            close = hist["Close"].dropna()
            if len(close) < 2:
                continue
            price = float(close.iloc[-1])
            prev1 = float(close.iloc[-2])
            ret_1d = (price / prev1 - 1.0) if prev1 > 0 else 0.0
            _sma_raw = close.rolling(200).mean(
            ).iloc[-1] if len(close) >= 200 else float("nan")
            sma200 = None if (not _m.isfinite(_sma_raw)) else float(_sma_raw)
            above_sma200 = bool(
                price > sma200) if sma200 is not None else False
            close_30d = close.iloc[-30:]
            info: Dict[str, Any] = {}
            try:
                info = _t.info or {}
            except Exception:
                pass
            company_name = info.get("shortName", info.get("longName", tk))
            de_raw = info.get("debtToEquity")
            de_ratio = de_raw / 100.0 if de_raw is not None else None
            net_mg_f = info.get("profitMargins")
            net_margin = net_mg_f * 100.0 if net_mg_f is not None else None
            momentum_c = 0.85 if above_sma200 else 0.25
            de_c = (0.50 if de_ratio is None else
                    0.50 if de_ratio < 0.50 else
                    0.35 if de_ratio < 1.00 else 0.10)
            mg_c = (0.50 if net_margin is None else
                    0.50 if net_margin > 15 else
                    0.38 if net_margin > 5 else
                    0.20 if net_margin > 0 else 0.05)
            quality_c = de_c + mg_c
            stock_score = round(0.50 * momentum_c + 0.50 * quality_c, 4)
            return {
                "ticker":       tk,
                "original":     ticker,
                "name":         company_name,
                "price":        round(price, 2),
                "ret_1d":       round(ret_1d, 6),
                "sma200":       round(sma200, 2) if sma200 is not None else None,
                "above_sma200": above_sma200,
                "de_ratio":     round(de_ratio, 2) if de_ratio is not None else None,
                "net_margin":   round(net_margin, 1) if net_margin is not None else None,
                "stock_score":  stock_score,
                "close_30d":    close_30d,
                "is_fallback":  tk != ticker,
            }
        except Exception:
            continue
    return None


def _trade_reason(
    category:            str,
    ticker:              str,
    data:                Dict[str, Any],
    macro_score:         float,
    regime_lbl:          str = "Neutral",
    institutional_score: float = 0.50,
    insider_score:       float = 0.50,
) -> str:
    macro_adj = (
        "Bullish commodity regime" if macro_score >= 0.72 else
        "Improving commodity backdrop" if macro_score >= 0.60 else
        "Neutral commodity environment"
    )
    sma_adj = "above SMA200" if data.get("above_sma200") else "below SMA200"
    de = data.get("de_ratio")
    mg = data.get("net_margin")
    qual = ""
    if de is not None and de < 0.5 and mg is not None and mg > 10:
        qual = f", D/E {de:.1f}x, margin {mg:.0f}%"
    elif mg is not None and mg > 0:
        qual = f", margin {mg:.0f}%"
    regime_adj = ""
    if regime_lbl in ("Bull", "Euphoria"):
        regime_adj = f" · HMM {regime_lbl} regime adds tailwind"
    elif regime_lbl in ("Bear", "Panic"):
        regime_adj = f" · HMM {regime_lbl} regime — conviction reduced"
    elif regime_lbl == "Crash":
        regime_adj = " · HMM Crash — hard block on all longs"
    institutional_adj = ""
    if institutional_score >= 0.70:
        institutional_adj = " · Institutions accumulating"
    elif institutional_score <= 0.35:
        institutional_adj = " · Institutions reducing — caution"
    insider_adj = ""
    if insider_score >= 0.85:
        insider_adj = " · CEO/CFO buying"
    elif insider_score >= 0.70:
        insider_adj = " · Insider accumulation"
    return f"{macro_adj} + {ticker} {sma_adj}{qual}{regime_adj}{institutional_adj}{insider_adj}."


def get_top_sector_picks(
    category:             str,
    prices:               Dict[str, Optional[Dict[str, Any]]],
    convictions:          Dict[str, Dict[str, Any]],
    regime_lbl:           str = "Neutral",
    institutional_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if institutional_scores is None:
        institutional_scores = {}
    ct_list = SECTOR_COMMODITY_MAP.get(category, [])
    c_scores = [convictions[ct]["composite"] for ct in ct_list
                if ct in convictions and convictions[ct]]
    macro_score = round(sum(c_scores) / len(c_scores), 4) if c_scores else 0.50
    macro_ok = macro_score > 0.60
    regime_mult = STOCK_REGIME_MULT.get(regime_lbl, 1.00)
    picks: List[Dict[str, Any]] = []
    for stock_def in SECTOR_STOCKS.get(category, []):
        tk = stock_def["ticker"]
        fb = SECTOR_FALLBACKS.get(tk)
        dat = fetch_stock_pick_data(tk, fb)
        if dat is None:
            continue
        inst_c = institutional_scores.get(dat["ticker"],
                                          institutional_scores.get(tk, 0.50))
        dat["institutional_score"] = round(inst_c, 4)
        if regime_lbl not in ("Crash", "Panic"):
            insider_c = fetch_insider_data(dat["ticker"])
            news_c = fetch_news_sentiment(dat["ticker"])
        else:
            insider_c = 0.50
            news_c = 0.50
        dat["insider_score"] = round(insider_c, 4)
        dat["news_score"] = round(news_c,    4)
        dat["macro_score"] = round(macro_score, 4)
        momentum_c = 0.85 if dat.get("above_sma200") else 0.25
        de = dat.get("de_ratio")
        mg = dat.get("net_margin")
        de_c = (0.50 if de is None else
                0.50 if de < 0.50 else
                0.35 if de < 1.00 else 0.10)
        mg_c = (0.50 if mg is None else
                0.50 if mg > 15 else
                0.38 if mg > 5 else
                0.20 if mg > 0 else 0.05)
        quality_c = de_c + mg_c
        # Five explicit factors — weights sum to 1.0
        # momentum 0.30 + quality 0.25 + institutional 0.15 + insider 0.15 + news 0.15
        stock_score = round(
            0.30 * momentum_c + 0.25 * quality_c
            + 0.15 * inst_c + 0.15 * insider_c + 0.15 * news_c, 4
        )
        dat["stock_score"] = stock_score
        macro_mult = (macro_score / 0.60) if macro_ok else 0.50
        final_score = round(
            min(1.0, stock_score * macro_mult * regime_mult), 4)
        if final_score >= 0.80:
            badge, badge_clr = "HIGH BUY",     "#00c851"
        elif final_score >= 0.60:
            badge, badge_clr = "TACTICAL BUY", "#ffbb33"
        else:
            badge, badge_clr = "WATCHLIST",    "#9e9e9e"
        dat["final_score"] = final_score
        dat["badge"] = badge
        dat["badge_clr"] = badge_clr
        dat["regime_mult"] = regime_mult
        dat["macro_mult"] = round(macro_mult, 4)
        dat["score_breakdown"] = {
            "momentum":      round(0.30 * momentum_c,  4),
            "quality":       round(0.25 * quality_c,   4),
            "institutional": round(0.15 * inst_c,      4),
            "insider":       round(0.15 * insider_c,   4),
            "news":          round(0.15 * news_c,      4),
            "macro_mult":    round(macro_mult,          4),
            "regime_mult":   round(regime_mult,         4),
        }
        dat["reason"] = _trade_reason(
            category, dat["ticker"], dat, macro_score, regime_lbl, inst_c, insider_c
        )
        picks.append(dat)
    picks.sort(key=lambda x: -x["final_score"])
    return {
        "category":    category,
        "macro_score": macro_score,
        "macro_ok":    macro_ok,
        "regime_lbl":  regime_lbl,
        "regime_mult": regime_mult,
        "picks":       picks,
    }


def _headline_sentiment(title: str) -> float:
    words = set(title.lower().split())
    bull = len(words & _BULL_WORDS)
    bear = len(words & _BEAR_WORDS)
    if bull == 0 and bear == 0:
        return 0.50
    return round(max(0.10, min(0.90, 0.50 + 0.20 * (bull - bear))), 4)


def fetch_news_sentiment(ticker: str) -> float:
    try:
        import yfinance as _yf
        news = _yf.Ticker(ticker).news
        if not news:
            return 0.50
        scores = []
        for item in news[:10]:
            title = ""
            content = item.get("content", {})
            if isinstance(content, dict):
                title = content.get("title", "")
            else:
                title = item.get("title", "")
            if title:
                scores.append(_headline_sentiment(title))
        return round(sum(scores) / len(scores), 4) if scores else 0.50
    except Exception:
        return 0.50


def fetch_institutional_data(ticker: str, _meta: Optional[Dict] = None) -> float:
    import logging as _log
    import yfinance as _yf
    import pandas as _pd
    _logger = _log.getLogger(__name__)

    # Layer 1 — institutional_holders (case-insensitive column matching)
    try:
        ih = _yf.Ticker(ticker).institutional_holders
        if ih is not None and not ih.empty:
            ih.columns = [str(c).strip() for c in ih.columns]
            _cols = {c.lower(): c for c in ih.columns}
            pct_held_col = next((
                _cols[k] for k in ("pctheld", "% out", "pct held", "pctout") if k in _cols
            ), None)
            pct_change_col = next((
                _cols[k] for k in ("pctchange", "% change", "pct change") if k in _cols
            ), None)
            if pct_held_col is not None:
                if _meta is not None:
                    _meta["has_data"] = True
                    _meta["source"] = "yf_institutional_holders"
                top = ih.head(20)
                pct_held = _pd.to_numeric(
                    top[pct_held_col], errors="coerce").fillna(0.0)
                if pct_change_col is not None:
                    pct_change = _pd.to_numeric(
                        top[pct_change_col], errors="coerce").fillna(0.0)
                    net_flow = float((pct_held * pct_change).sum())
                    return round(max(0.20, min(0.90, 0.55 + 5.0 * net_flow)), 4)
                else:
                    total_held = float(pct_held.sum())
                    return round(max(0.20, min(0.90, 0.30 + total_held)), 4)
            _logger.warning(
                "[INST %s] institutional_holders columns not usable: %s", ticker, list(ih.columns))
        else:
            _logger.warning(
                "[INST %s] institutional_holders empty, trying major_holders", ticker)
    except Exception as _e:
        _logger.warning(
            "[INST %s] institutional_holders error: %s", ticker, _e)

    # Layer 2 — major_holders aggregate (institutionsPercentHeld)
    try:
        mh = _yf.Ticker(ticker).major_holders
        if mh is not None and not mh.empty:
            mh.index = [str(i).strip() for i in mh.index]
            _mh_idx = {i.lower(): i for i in mh.index}
            for _key in ("institutionspercentheld", "% held by institutions", "institutions percent held"):
                if _key in _mh_idx:
                    if _meta is not None:
                        _meta["has_data"] = True
                        _meta["source"] = "yf_major_holders"
                    inst_pct = float(_pd.to_numeric(
                        mh.loc[_mh_idx[_key]].iloc[0], errors="coerce") or 0.0)
                    score = round(
                        max(0.20, min(0.90, 0.30 + inst_pct * 0.65)), 4)
                    _logger.warning(
                        "[INST %s] used major_holders fallback, score=%s", ticker, score)
                    return score
        _logger.warning(
            "[INST %s] major_holders empty or key missing, trying FMP", ticker)
    except Exception as _e:
        _logger.warning(
            "[INST %s] major_holders error: %s, trying FMP", ticker, _e)

    # Layer 3 — FMP API
    try:
        score = fetch_fmp_institutional_score(ticker)
        if score != 0.50 and _meta is not None:
            _meta["has_data"] = True
            _meta["source"] = "fmp_fallback"
        return score
    except Exception as _e:
        _logger.warning("[INST %s] FMP fallback error: %s", ticker, _e)
        return 0.50


def fetch_institutional_bulk(tickers: list) -> Dict[str, float]:
    global _INSTITUTIONAL_CACHE
    if (_INSTITUTIONAL_CACHE["data"] is not None
            and (time.time() - _INSTITUTIONAL_CACHE["ts"]) < _INSTITUTIONAL_TTL):
        return _INSTITUTIONAL_CACHE["data"]
    results: Dict[str, float] = {}
    for ticker in tickers:
        results[ticker] = fetch_institutional_data(ticker)
        time.sleep(0.15)
    if results:
        _INSTITUTIONAL_CACHE["data"] = results
        _INSTITUTIONAL_CACHE["ts"] = time.time()
    return results


def reset_institutional_cache() -> None:
    global _INSTITUTIONAL_CACHE
    _INSTITUTIONAL_CACHE["data"] = None
    _INSTITUTIONAL_CACHE["ts"] = 0.0


def fetch_insider_data(ticker: str) -> float:
    import logging as _log
    import pandas as _pd
    import yfinance as _yf
    _logger = _log.getLogger(__name__)

    # Layer 1 — yfinance insider_transactions
    try:
        txns = _yf.Ticker(ticker).insider_transactions
        if txns is not None and not txns.empty:
            _cols = {str(c).strip().lower(): str(c).strip()
                     for c in txns.columns}
            text_col = next(
                (_cols[k] for k in ("text", "transaction") if k in _cols), None)
            pos_col = next((_cols[k]
                           for k in ("position",) if k in _cols), None)
            val_col = next((_cols[k] for k in ("value",) if k in _cols), None)

            buy_count = sell_count = 0
            ceo_cfo_buy = 0.0
            for _, row in txns.iterrows():
                raw_text = row.get(text_col, "") if text_col else ""
                raw_pos = row.get(pos_col,  "") if pos_col else ""
                raw_val = row.get(val_col,   0) if val_col else 0

                text = "" if _pd.isna(raw_text) else str(
                    raw_text).upper().strip()
                pos = "" if _pd.isna(raw_pos) else str(raw_pos).upper().strip()
                val = 0.0
                try:
                    val = abs(float(raw_val or 0))
                except (TypeError, ValueError):
                    pass

                is_buy = "PURCHASE" in text or "ACQUI" in text or text.startswith(
                    "BUY")
                is_sell = "SALE" in text or "SELL" in text

                if is_buy:
                    buy_count += 1
                    if any(k in pos for k in ("CEO", "CFO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL")):
                        ceo_cfo_buy += val
                elif is_sell:
                    sell_count += 1

            total = buy_count + sell_count
            if total > 0:
                buy_frac = buy_count / total
                score = round(0.30 + 0.60 * buy_frac, 4)
                if ceo_cfo_buy > 0:
                    score = max(score, 0.85)
                _logger.warning("[INS %s] yf score=%s buy=%d sell=%d",
                                ticker, score, buy_count, sell_count)
                return score
            _logger.warning(
                "[INS %s] no buy/sell parsed (rows=%d), trying FMP", ticker, len(txns))
        else:
            _logger.warning(
                "[INS %s] insider_transactions empty, trying FMP", ticker)
    except Exception as _e:
        _logger.warning("[INS %s] yfinance error: %s, trying FMP", ticker, _e)

    # Layer 2 — FMP / OpenInsider fallback
    try:
        return fetch_fmp_insider_score(ticker)
    except Exception as _e:
        _logger.warning("[INS %s] FMP fallback error: %s", ticker, _e)
        return 0.50


def fetch_social_sentiment() -> Dict[str, float]:
    scores: Dict[str, float] = {}
    try:
        import requests as _req
        r = _req.get(
            "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
            timeout=10,
            headers={"User-Agent": "regime-trader/1.0"},
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return scores
        max_mentions = max((t.get("mentions", 1) for t in results), default=1)
        for t in results[:50]:
            sym = str(t.get("ticker", "")).strip().upper()
            if not sym or len(sym) > 5 or not sym.isalpha():
                continue
            base = 0.50 + 0.40 * (t.get("mentions", 0) / max(max_mentions, 1))
            rank_now = t.get("rank", 999)
            rank_24h = t.get("rank_24h") or 999
            if rank_now < rank_24h:
                base = min(0.92, base + 0.05)
            scores[sym] = round(min(0.92, base), 4)
    except Exception:
        pass
    return scores


def compute_enhanced_macro_score(
    vix: float,
    dxy: Optional[float] = None,
    spread: Optional[float] = None,
) -> Dict[str, float]:
    import math as _m

    def _sig(val, fn) -> float:
        if val is None:
            return 0.5
        try:
            v = float(val)
            if _m.isnan(v):
                return 0.5
            return round(max(0.0, min(1.0, fn(v))), 4)
        except Exception:
            return 0.5

    vix_score = _sig(vix, lambda v: 1.0 / (1.0 + _m.exp((v - 22.0) / 4.0)))
    dxy_raw = dxy
    spread_raw = spread

    if dxy_raw is None or spread_raw is None:
        try:
            import yfinance as _yf
            if dxy_raw is None:
                _dh = _yf.Ticker("DX-Y.NYB").history(period="5d")
                if not _dh.empty:
                    dxy_raw = float(_dh["Close"].iloc[-1])
            if spread_raw is None:
                _tnx_h = _yf.Ticker("^TNX").history(period="5d")
                _irx_h = _yf.Ticker("^IRX").history(period="5d")
                if not _tnx_h.empty and not _irx_h.empty:
                    spread_raw = round(
                        float(_tnx_h["Close"].iloc[-1]) -
                        float(_irx_h["Close"].iloc[-1]), 4
                    )
        except Exception:
            pass

    dxy_score = _sig(dxy_raw, lambda v: 1.0 /
                     (1.0 + _m.exp((v - 104.0) / 2.0)))
    spread_score = _sig(spread_raw, lambda v: 1.0 / (1.0 + _m.exp(-v * 10.0)))
    composite = round(vix_score * 0.40 + dxy_score *
                      0.30 + spread_score * 0.30, 4)

    if composite >= 0.80:
        regime_label = "🟢 Ultra Bullish (High Liquidity)"
    elif composite >= 0.60:
        regime_label = "🟡 Bullish (Stable)"
    elif composite >= 0.40:
        regime_label = "🟠 Neutral / Transition"
    else:
        regime_label = "🔴 Warning (Risk-Off / Tightening)"

    return {
        "composite":    composite,
        "vix_score":    vix_score,
        "dxy_score":    dxy_score,
        "spread_score": spread_score,
        "yield_score":  spread_score,
        "regime_label": regime_label,
        "dxy":          dxy_raw,
        "spread":       spread_raw,
        "vix":          float(vix) if vix is not None else None,
    }


def _refresh_ape_cache() -> Dict[str, Any]:
    import requests as _req
    try:
        r = _req.get(
            "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        if r.status_code == 200:
            _APE_CACHE["rows"] = {
                str(item.get("ticker", "")).upper(): item
                for item in r.json().get("results", [])
            }
            _APE_CACHE["ts"] = time.monotonic()
    except Exception:
        pass
    return _APE_CACHE["rows"]


def fetch_stocktwits_sentiment(ticker: str) -> Dict[str, float]:
    now = time.monotonic()
    rows = _APE_CACHE["rows"]
    if not rows or now - _APE_CACHE["ts"] > _APE_TTL:
        rows = _refresh_ape_cache()
    item = rows.get(ticker.upper())
    if not item:
        return {}
    rank = item.get("rank", 100)
    upvotes = item.get("upvotes", 0)
    rank_score = max(0.0, (100 - rank) / 99.0)
    upvote_bonus = min(0.10, upvotes / 1000.0)
    score = round(
        max(0.1, min(0.9, 0.50 + 0.35 * rank_score + upvote_bonus)), 4)
    return {ticker.upper(): score}


def fetch_fmp_insider_score(ticker: str) -> float:
    import requests as _req
    try:
        from intelligence.sources import InsiderFetcher
        score, meta = InsiderFetcher().fetch(ticker)
        if score != 0.50 or meta.get("total", 0) > 0:
            return score
    except Exception:
        pass
    api_key = os.getenv("FMP_API_KEY", "")
    if api_key:
        try:
            r = _req.get(
                f"https://financialmodelingprep.com/stable/grades"
                f"?symbol={ticker}&limit=20&apikey={api_key}",
                timeout=10,
            )
            data = r.json() if r.status_code == 200 else []
            if isinstance(data, list) and data:
                upgrades = sum(1 for g in data if str(
                    g.get("action", "")).lower() == "upgrade")
                downgrades = sum(1 for g in data if str(
                    g.get("action", "")).lower() == "downgrade")
                total = upgrades + downgrades
                if total:
                    return round(0.30 + 0.60 * (upgrades / total), 4)
        except Exception:
            pass
    return 0.50


def fetch_fmp_institutional_score(ticker: str) -> float:
    import requests as _req
    api_key = os.getenv("FMP_API_KEY", "")
    if api_key:
        try:
            r = _req.get(
                f"https://financialmodelingprep.com/stable/price-target-consensus"
                f"?symbol={ticker}&apikey={api_key}",
                timeout=10,
            )
            data = r.json() if r.status_code == 200 else []
            if isinstance(data, list) and data:
                rec = data[0]
                target = float(rec.get("targetConsensus")
                               or rec.get("targetMedian") or 0)
                rq = _req.get(
                    f"https://financialmodelingprep.com/stable/quote"
                    f"?symbol={ticker}&apikey={api_key}",
                    timeout=10,
                )
                qdata = rq.json() if rq.status_code == 200 else []
                price = float(qdata[0].get("price", 0)) if isinstance(
                    qdata, list) and qdata else 0
                if target > 0 and price > 0:
                    upside = (target - price) / price
                    score = 0.50 + min(0.40, max(-0.40, upside * 2))
                    return round(max(0.10, min(0.90, score)), 4)
        except Exception:
            pass
        try:
            r = _req.get(
                f"https://financialmodelingprep.com/stable/grades"
                f"?symbol={ticker}&limit=10&apikey={api_key}",
                timeout=10,
            )
            data = r.json() if r.status_code == 200 else []
            if isinstance(data, list) and data:
                upgrades = sum(1 for g in data if str(
                    g.get("action", "")).lower() == "upgrade")
                downgrades = sum(1 for g in data if str(
                    g.get("action", "")).lower() == "downgrade")
                total = upgrades + downgrades
                if total:
                    return round(0.30 + 0.60 * (upgrades / total), 4)
        except Exception:
            pass
    try:
        from free_market_intel import get_congress_score
        return get_congress_score(ticker)
    except Exception:
        return 0.50


def fetch_finnhub_news_sentiment(ticker: str) -> float:
    import requests as _req
    from datetime import date
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return 0.50
    try:
        today = date.today()
        week_ago = today - timedelta(days=7)
        r = _req.get(
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}"
            f"&from={week_ago.isoformat()}"
            f"&to={today.isoformat()}"
            f"&token={api_key}",
            timeout=10,
        )
        if r.status_code != 200:
            return 0.50
        articles = r.json()
        if not isinstance(articles, list) or not articles:
            return 0.50
        import nltk
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _sia = SentimentIntensityAnalyzer()
        compounds = [
            _sia.polarity_scores(a.get("headline", ""))["compound"]
            for a in articles[:30]
        ]
        if not compounds:
            return 0.50
        mean_compound = sum(compounds) / len(compounds)
        return round(max(0.0, min(1.0, (mean_compound + 1) / 2)), 4)
    except Exception:
        return 0.50


def fetch_finnhub_recommendation_score(ticker: str) -> float:
    import requests as _req
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return 0.50
    try:
        r = _req.get(
            f"https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={api_key}",
            timeout=10,
        )
        if r.status_code != 200:
            return 0.50
        data = r.json()
        if not isinstance(data, list) or not data:
            return 0.50
        rec = data[0]
        sb = int(rec.get("strongBuy",  0) or 0)
        b = int(rec.get("buy",        0) or 0)
        h = int(rec.get("hold",       0) or 0)
        s = int(rec.get("sell",       0) or 0)
        ss = int(rec.get("strongSell", 0) or 0)
        total = sb + b + h + s + ss
        if total == 0:
            return 0.50
        weighted = sb * 1.00 + b * 0.75 + h * 0.50 + s * 0.25 + ss * 0.00
        return round(max(0.10, min(0.90, weighted / total)), 4)
    except Exception:
        return 0.50


# ── Page config (MUST be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Alpha Terminal · Regime Trader",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "**Regime Trader** — Institutional Alpha Terminal",
    },
)

# ── Custom CSS (Institutional Alpha Terminal — terminal-density dark theme) ───

st.markdown(
    """
    <style>
    /* ── Design tokens — single source of truth for colours, spacing, typography ── */
    :root {
        /* Regime palette */
        --c-bull:     #00c851;
        --c-euphoria: #00e676;
        --c-mania:    #69f0ae;
        --c-neutral:  #9e9e9e;
        --c-unknown:  #757575;
        --c-bear:     #ff8800;
        --c-panic:    #ff4444;
        --c-crash:    #b71c1c;
        /* Grade palette */
        --c-grade-a:  #00c851;
        --c-grade-b:  #ffbb33;
        --c-grade-c:  #ff4444;
        /* Background scale (darkest → lightest) */
        --bg-0: #040404;
        --bg-1: #0a0a0a;
        --bg-2: #111111;
        --bg-3: #1a1a1a;
        --bg-4: #222222;
        /* Text scale (darkest → brightest) — dark-mode calibrated */
        --tx-0: #888888;
        --tx-1: #888888;
        --tx-2: #999999;
        --tx-3: #AAAAAA;
        --tx-4: #BBBBBB;
        --tx-5: #CCCCCC;
        --tx-6: #D8D8D8;
        --tx-7: #E0E0E0;
        /* 8 px spacing grid */
        --sp-1: 4px;
        --sp-2: 8px;
        --sp-3: 12px;
        --sp-4: 16px;
        --sp-5: 24px;
        /* Typography */
        --font-sans: 'Inter', system-ui, sans-serif;
        --font-mono: 'JetBrains Mono', 'Consolas', monospace;
        /* Status */
        --c-ok:   #00c851;
        --c-warn: #ffbb33;
        --c-err:  #ff4444;
    }

    /* ── Base ── */
    html, body, [class*="css"] { font-family: var(--font-sans); }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; }

    /* ── Monospace for all price / data values ── */
    .mono { font-family: var(--font-mono); }

    /* ── Metrics ── */
    [data-testid="stMetricLabel"]  { font-size: 0.68rem; color: var(--tx-3); letter-spacing:.08em; text-transform:uppercase; }
    [data-testid="stMetricValue"]  { font-size: 1.45rem; font-weight: 800; line-height: 1.1;
                                     font-family: var(--font-mono); }
    [data-testid="stMetricDelta"]  { font-size: 0.76rem; font-weight: 600; }

    /* ── Tabs ── */
    [data-testid="stTabs"] button[role="tab"] p,
    [data-testid="stTabs"] button[role="tab"] span {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--tx-5) !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] span {
        color: var(--c-ok) !important;
        font-weight: 700 !important;
    }

    /* ── Dividers ── */
    hr { margin: var(--sp-1) 0 !important; border-color: var(--bg-2) !important; }

    /* ── Bloomberg-style Risk Ribbon ── */
    .risk-ribbon {
        display: flex; background: var(--bg-0);
        border: 1px solid #181818; border-radius: 5px;
        overflow: hidden; margin-bottom: var(--sp-2);
    }
    .ribbon-block {
        flex: 1; padding: var(--sp-2) var(--sp-3);
        border-right: 1px solid var(--bg-2);
        display: flex; flex-direction: column; gap: var(--sp-1);
    }
    .ribbon-block:last-child { border-right: none; max-width: 90px; }
    .ribbon-label {
        font-size: 0.53rem; color: var(--tx-1); font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
    }
    .ribbon-value {
        font-size: 1.22rem; font-weight: 700; line-height: 1.1; color: var(--tx-6);
        font-family: var(--font-mono);
    }
    .ribbon-sub {
        font-size: 0.57rem; color: var(--tx-1);
        font-family: var(--font-mono);
    }

    /* ── Conviction grade badges A / B / C ── */
    .grade-badge {
        display: inline-block; padding: var(--sp-1) 7px; border-radius: 3px;
        font-size: 0.72rem; font-weight: 800; letter-spacing: 0.04em;
        font-family: var(--font-mono);
    }
    .grade-A { background:#00c85114; color: var(--c-grade-a); border:1px solid #00c85130; }
    .grade-B { background:#ffbb3314; color: var(--c-grade-b); border:1px solid #ffbb3330; }
    .grade-C { background:#ff444414; color: var(--c-grade-c); border:1px solid #ff444430; }

    /* ── Senior Trader's Brief ── */
    .brief-item {
        display: flex; align-items: flex-start; gap: var(--sp-3);
        padding: var(--sp-2) var(--sp-3); margin: var(--sp-1) 0;
        border-radius: 4px; font-size: 0.78rem; border-left: 3px solid;
    }
    .brief-badge {
        font-weight: 900; font-size: 0.62rem;
        padding: var(--sp-1) var(--sp-2); border-radius: 3px;
        letter-spacing: 0.08em; white-space: nowrap; margin-top: 2px;
        font-family: var(--font-mono);
    }

    /* ── Alert row (circuit breakers + brief items unified) ── */
    .alert-row {
        display: flex; align-items: flex-start; gap: var(--sp-3);
        padding: var(--sp-2) var(--sp-3); margin: var(--sp-1) 0;
        border-radius: 4px; font-size: 0.76rem;
    }
    .alert-dot {
        width: 8px; height: 8px; border-radius: 50%;
        flex-shrink: 0; margin-top: 4px;
    }
    .alert-label { font-weight: 700; font-family: var(--font-mono); font-size: 0.70rem; }
    .alert-desc  { font-size: 0.66rem; color: var(--tx-4); margin-top: 2px; }

    /* ── Regime badge ── */
    .regime-badge {
        display: inline-block; padding: var(--sp-1) var(--sp-4); border-radius: 18px;
        font-size: 1.1rem; font-weight: 800; letter-spacing: 0.05em;
    }

    /* ── Section micro-headers ── */
    .section-header {
        font-size: 0.60rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: var(--tx-2); margin: 0 0 var(--sp-1) 0;
    }

    /* ── Status dots ── */
    .dot-ok   { color: var(--c-ok); }
    .dot-err  { color: var(--c-err); }
    .dot-warn { color: var(--c-warn); }

    /* ── KPI strip (kept for Market Intel tab) ── */
    .kpi-strip {
        display: flex; gap: 0; background: var(--bg-1);
        border-radius: 7px; border: 1px solid #181818;
        overflow: hidden; margin-bottom: var(--sp-3);
    }
    .kpi-cell { flex: 1; padding: var(--sp-3) var(--sp-3); border-right: 1px solid #161616; }
    .kpi-cell:last-child { border-right: none; }
    .kpi-label { font-size: 0.60rem; color: var(--tx-2); font-weight:700; text-transform:uppercase; letter-spacing:.10em; margin-bottom:2px; }
    .kpi-value { font-size: 1.42rem; font-weight: 800; line-height: 1; color: var(--tx-7);
                  font-family: var(--font-mono); }
    .kpi-delta { font-size: 0.70rem; font-weight: 600; margin-top: 2px;
                  font-family: var(--font-mono); }

    /* ── Signal card ── */
    .sig-card { border-radius: 8px; padding: var(--sp-3) var(--sp-3); height: 100%; }

    /* ── Bloomberg section title bar ── */
    .section-bar {
        display: flex; align-items: center; gap: var(--sp-3);
        margin: var(--sp-3) 0 var(--sp-2) 0; padding: 0;
    }
    .section-bar-badge {
        font-size: 0.58rem; font-weight: 900; letter-spacing: 0.14em;
        text-transform: uppercase; padding: 3px 9px; border-radius: 3px;
        background: #0d0d0d; border: 1px solid #444444; color: #FFFFFF;
        font-family: var(--font-mono); white-space: nowrap;
    }
    .section-bar-line { flex: 1; height: 1px; background: #2A2A2A; }
    .section-bar-num  { font-size: 0.55rem; color: var(--tx-4); font-family: var(--font-mono); white-space: nowrap; }

    /* ── Macro strip cells in the header ── */
    .mstrip-cell {
        display: flex; flex-direction: column; gap: 0;
        padding: 0 14px; border-right: 1px solid #181818;
        justify-content: center;
    }
    .mstrip-cell:last-child { border-right: none; }
    .mstrip-label {
        font-size: 0.42rem; color: #AAAAAA; letter-spacing: 0.14em;
        text-transform: uppercase; line-height: 1;
    }
    .mstrip-val {
        font-size: 0.72rem; font-weight: 800; font-family: var(--font-mono);
        line-height: 1.3; color: #888;
    }
    .mstrip-up   { color: #00c851 !important; }
    .mstrip-down { color: #ff4444 !important; }
    .mstrip-warn { color: #ffbb33 !important; }

    /* ── Pillar Intelligence Panel ── */
    .pillar-panel {
        background: var(--bg-1); border: 1px solid #181818;
        border-radius: 6px; padding: 12px 14px; margin-bottom: 12px;
    }
    .pillar-row {
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 6px;
    }
    .pillar-row:last-child { margin-bottom: 0; }
    .pillar-lbl {
        font-size: 0.58rem; font-weight: 700; letter-spacing: 0.10em;
        text-transform: uppercase; color: #AAAAAA; width: 36px; text-align: right;
        font-family: var(--font-mono); flex-shrink: 0;
    }
    .pillar-track {
        flex: 1; height: 6px; background: #111; border-radius: 3px; overflow: hidden;
    }
    .pillar-fill {
        height: 6px; border-radius: 3px;
        transition: width 0.4s cubic-bezier(.4,0,.2,1);
    }
    .pillar-pct {
        font-size: 0.62rem; font-weight: 800; font-family: var(--font-mono);
        width: 32px; text-align: right; flex-shrink: 0;
    }

    /* ── Ticker chip selector ── */
    .ticker-chip button {
        background: #111 !important;
        border: 1px solid #222 !important;
        color: #AAAAAA !important;
        font-size: 0.60rem !important;
        font-family: var(--font-mono) !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        padding: 2px 8px !important;
        border-radius: 3px !important;
        height: auto !important;
        min-height: 24px !important;
        line-height: 1 !important;
        cursor: pointer !important;
        transition: border-color 0.15s, color 0.15s !important;
    }
    .ticker-chip button:hover {
        border-color: #00c851 !important;
        color: #00c851 !important;
    }

    /* ── Volume spike badge ── */
    .vol-badge {
        display: inline-block;
        padding: 1px 6px; border-radius: 2px;
        font-size: 0.50rem; font-weight: 900;
        letter-spacing: 0.08em; text-transform: uppercase;
        font-family: var(--font-mono);
    }
    .vol-high  { background: #ff880015; color: #ff8800; border: 1px solid #ff880030; }
    .vol-spike { background: #ff444415; color: #ff4444; border: 1px solid #ff444430; }
    .vol-norm  { background: #22222250; color: #AAAAAA;  border: 1px solid #55555550; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Fixed Alpha Terminal Header ─────────────────────────────── */

    /* Push the main content block below the fixed header (48px bar) */
    .block-container { padding-top: 3.8rem !important; }

    /* The header itself — spans full viewport width */
    #alpha-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 48px;
        z-index: 9999;
        display: flex;
        align-items: center;
        /* Left half: sidebar colour; right half: header colour
           Streamlit sidebar is exactly 336px wide in wide-layout.
           The gradient creates a hard colour-stop at that point. */
        background: linear-gradient(
            to right,
            #0a0a0a 0px,
            #0a0a0a 336px,
            #050505 336px,
            #050505 100%
        );
        border-bottom: 1px solid var(--bg-3);
        box-shadow: 0 2px 12px rgba(0,0,0,0.5);
        font-family: var(--font-mono);
    }

    /* Vertical separator that lines up with the sidebar edge */
    #alpha-header::after {
        content: '';
        position: absolute;
        left: 336px;
        top: 12px;
        bottom: 12px;
        width: 1px;
        background: #1e1e1e;
    }

    /* Left cell — logo area (sits over the sidebar) */
    #alpha-header-logo {
        width: 336px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 20px;
    }
    #alpha-header-logo .ah-icon {
        font-size: 1.1rem;
        line-height: 1;
    }
    #alpha-header-logo .ah-title {
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0.18em;
        color: #00c851;
        text-transform: uppercase;
    }
    #alpha-header-logo .ah-version {
        font-size: 0.52rem;
        color: #888888;
        letter-spacing: 0.08em;
        margin-top: 1px;
        align-self: flex-end;
        padding-bottom: 2px;
    }

    /* Right cell — timestamp + market status (sits over main content) */
    #alpha-header-right {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 18px;
        padding: 0 24px;
    }

    /* "LAST UPDATED" label + value pair */
    .ah-ts-block {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0;
    }
    .ah-ts-label {
        font-size: 0.46rem;
        color: #333;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        line-height: 1;
    }
    .ah-ts-value {
        font-size: 0.72rem;
        font-weight: 700;
        color: #888;
        letter-spacing: 0.06em;
        line-height: 1.3;
    }

    /* Live / Stale / Offline pill */
    .ah-status-pill {
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.60rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .ah-status-dot {
        width: 6px; height: 6px; border-radius: 50%;
    }
    .ah-status-live  { background:#00c85115; color:#00c851; border:1px solid #00c85130; }
    .ah-status-stale { background:#ffbb3315; color:#ffbb33; border:1px solid #ffbb3330; }
    .ah-status-off   { background:#ff444415; color:#ff4444; border:1px solid #ff444430; }

    /* Regime pill in header */
    .ah-regime-pill {
        padding: 3px 12px;
        border-radius: 3px;
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* Blinking dot animation for live status */
    @keyframes blink-dot {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.2; }
    }
    .blink { animation: blink-dot 2s ease-in-out infinite; }

    /* Make sure the Streamlit sidebar renders *above* the gradient left cell */
    [data-testid="stSidebar"] { z-index: 9998 !important; }

    /* Prevent the Streamlit top header chrome from overlapping */
    [data-testid="stHeader"] { display: none !important; }

    /* ── Mobile responsive (≤ 768 px) ─────────────────────────────────────── */
    @media (max-width: 768px) {

        /* Shrink the fixed header */
        #alpha-header          { height: 36px; }
        #alpha-header-logo     { width: auto; padding: 0 10px; gap: 6px; }
        #alpha-header-logo span:first-child { font-size: 0.9rem; }
        #alpha-header-right    { padding: 0 10px; gap: 6px; }
        .block-container       { padding-top: 3.0rem !important; }

        /* Ribbon scrolls horizontally instead of squashing */
        .risk-ribbon           { overflow-x: auto; -webkit-overflow-scrolling: touch; }
        .ribbon-block          { min-width: 76px; flex-shrink: 0; padding: 6px 8px; }
        .ribbon-value          { font-size: 0.95rem; }
        .ribbon-label          { font-size: 0.46rem; letter-spacing: 0.10em; }
        .ribbon-sub            { display: none; }   /* hide subtext on small screen */

        /* KPI strip */
        .kpi-strip             { overflow-x: auto; }
        .kpi-cell              { min-width: 72px; flex-shrink: 0; padding: 8px 10px; }
        .kpi-value             { font-size: 1.1rem; }
        .kpi-label             { font-size: 0.52rem; }

        /* Section bars */
        .section-bar-badge     { font-size: 0.52rem; padding: 2px 7px; }

        /* Ensure minimum touch target size for all buttons */
        button                 { min-height: 44px !important; }

        /* Grade badges slightly larger for tap targets */
        .grade-badge           { padding: 4px 10px; font-size: 0.74rem; }

        /* Brief / alert rows: larger on mobile */
        .brief-item, .alert-row { padding: 8px 12px; font-size: 0.80rem; }

        /* Plotly charts: allow horizontal scroll on narrow screens */
        .js-plotly-plot        { max-width: 100%; overflow-x: auto; }

        /* Tabs: let them scroll if too many */
        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            flex-wrap: nowrap;
        }
        [data-testid="stTabs"] button[role="tab"] p,
        [data-testid="stTabs"] button[role="tab"] span {
            font-size: 0.85rem !important;
            white-space: nowrap;
        }
    }

    /* ── iPhone notch / safe area support ─────────────────────────────────── */
    @supports (padding-top: env(safe-area-inset-top)) {
        #alpha-header {
            padding-top: env(safe-area-inset-top);
            height: calc(48px + env(safe-area-inset-top));
        }
        .block-container {
            padding-left:  env(safe-area-inset-left)  !important;
            padding-right: env(safe-area-inset-right) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── PWA / Home Screen meta tags ───────────────────────────────────────────────
# These tags allow iOS (Safari) and Android (Chrome) to install the dashboard
# as a standalone app icon on the home screen. No manifest file needed.
st.markdown(
    """
    <meta name="apple-mobile-web-app-capable"          content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title"            content="Alpha Terminal">
    <meta name="mobile-web-app-capable"                content="yes">
    <meta name="theme-color"                           content="#050505">
    <meta name="viewport"
          content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ── Constants & helpers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Shared Plotly layout — Bloomberg terminal palette, apply with fig.update_layout(**_PLOTLY_LAYOUT)
_PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0",
              family="Courier New, JetBrains Mono, monospace", size=10),
    margin=dict(t=12, b=12, l=8, r=8),
    xaxis=dict(gridcolor="#1E1E1E", zerolinecolor="#1E1E1E"),
    yaxis=dict(gridcolor="#1E1E1E", zerolinecolor="#1E1E1E"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1, font=dict(size=9),
        bgcolor="rgba(0,0,0,0)",
    ),
)

# ── Chart utilities (Bloomberg pro theme) ─────────────────────────────────────
_utils_p = str(_HERE / "utils")
if _utils_p not in sys.path:
    sys.path.insert(0, _utils_p)
try:
    from charting import apply_pro_theme, macro_heatmap_fig, dcf_waterfall_fig
except ImportError:
    def apply_pro_theme(fig, **_kw): return fig  # noqa: E306
    def macro_heatmap_fig(_d): return go.Figure()  # noqa: E306
    def dcf_waterfall_fig(_r): return go.Figure()  # noqa: E306

# Scan universe for intel fetchers — fallback when morning_report cannot be imported
# (Streamlit Cloud: morning_report imports log_manager which is a local-only package)
try:
    from morning_report import SCAN_UNIVERSE as _SCAN_UNIVERSE
except Exception:
    _SCAN_UNIVERSE: List[str] = [
        "NVDA", "MSFT", "AAPL", "GOOGL", "GOOG", "META", "AMZN", "TSLA", "AMD",
        "AVGO", "CRM", "ORCL", "NFLX", "PYPL", "COIN", "SQ", "PLTR", "SNOW",
        "SPY", "QQQ", "GLD", "TLT", "XLE", "XLK", "XLF", "XLV", "XLP", "XLU",
        "RTX", "LMT", "NOC", "XOM", "CVX", "NEM", "FCX", "VALE", "GDX",
        "JPM", "BAC", "GS", "V", "MA", "UNH", "JNJ", "PFE", "ABBV",
    ]

_REGIME_COLORS: Dict[str, str] = {
    "Bull":     "#00c851",
    "Euphoria": "#00e676",
    "Mania":    "#69f0ae",
    "Neutral":  "#9e9e9e",
    "Unknown":  "#757575",
    "Bear":     "#ff8800",
    "Panic":    "#ff4444",
    "Crash":    "#b71c1c",
}

# Regime multiplier for Unified Conviction Engine
# Bull/Euphoria → risk-on (+20%);  Bear/Panic → defensive (−50%);  Crash → hard halt (0)
_REGIME_MULT: Dict[str, float] = {
    "Bull":     1.2,
    # blow-off top — reduce exposure (matches circuit breaker 0.7× sz)
    "Euphoria": 0.7,
    # blow-off top — reduce exposure (matches circuit breaker 0.7× sz)
    "Mania":    0.7,
    "Neutral":  1.0,
    "Unknown":  0.9,
    "Bear":     0.5,
    "Panic":    0.5,
    "Crash":    0.0,
}

_SCORE_COLORS = [
    (0.78, "#00c851"),
    (0.63, "#7cb342"),
    (0.50, "#9e9e9e"),
    (0.37, "#ff8800"),
    (0.22, "#ff4444"),
    (0.00, "#b71c1c"),
]


def _hex_to_rgba(hex_c: str, alpha: float) -> str:
    """Convert a 6-digit hex color to rgba() for Plotly 6 compatibility."""
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _get_last_updated() -> Tuple[str, str, str]:
    """
    Derive 'Last Updated' from the most recent regime.log entry.

    Returns (timestamp_str, status_class, status_label).
    status_class: "ah-status-live" | "ah-status-stale" | "ah-status-off"
    """
    _log = Path(str(_HERE / "logs" / "regime.log"))
    if not _log.exists():
        return "No data yet", "ah-status-off", "OFFLINE"

    # Read all lines, walk backwards to find the last valid JSON with a timestamp
    try:
        with _log.open("r", encoding="utf-8") as fh:
            all_lines = fh.readlines()
    except OSError:
        return "Log unreadable", "ah-status-off", "OFFLINE"

    ts_raw = ""
    for raw_line in reversed(all_lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            rec = json.loads(raw_line)
            ts_raw = str(rec.get("timestamp", ""))
            if ts_raw:
                break
        except (json.JSONDecodeError, ValueError):
            continue

    if not ts_raw:
        return "No timestamp found", "ah-status-off", "OFFLINE"

    try:
        ts_dt = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if pd.isna(ts_dt):
            return ts_raw[:19], "ah-status-stale", "STALE"

        age_secs = (datetime.now(timezone.utc) -
                    ts_dt.to_pydatetime()).total_seconds()
        ts_str = ts_dt.strftime("%Y-%m-%d  %H:%M:%S UTC")

        if age_secs < 120:
            return ts_str, "ah-status-live", "LIVE"
        elif age_secs < 3600:
            mins = int(age_secs // 60)
            return f"{ts_str}  ({mins}m ago)", "ah-status-stale", "STALE"
        else:
            hrs = int(age_secs // 3600)
            return f"{ts_str}  ({hrs}h ago)", "ah-status-off", "OFFLINE"
    except Exception:
        return ts_raw[:19], "ah-status-off", "OFFLINE"


def _section_bar(label: str, num: str = "") -> None:
    """Render a Bloomberg-style section title bar."""
    st.markdown(
        f'<div class="section-bar">'
        f'<span class="section-bar-badge">{label}</span>'
        f'<div class="section-bar-line"></div>'
        + (f'<span class="section-bar-num">{num}</span>' if num else "")
        + f'</div>',
        unsafe_allow_html=True,
    )


def _alert_row_html(
    label:       str,
    active:      bool,
    description: str,
    color:       str = "#ff4444",
    warn_color:  str = "#ff4444",
) -> str:
    """
    Unified alert/status row used by both Senior Trader's Brief and
    the Circuit Breaker panel.  Returns an HTML string to pass to
    st.markdown(..., unsafe_allow_html=True).

    active=True  → coloured dot + bold label + red/amber border
    active=False → dark dot + dim label + invisible border
    """
    dot_clr = warn_color if active else "#1e1e1e"
    lbl_clr = warn_color if active else "#2a2a2a"
    bg_clr = f"{warn_color}08" if active else "transparent"
    border_clr = f"{warn_color}22" if active else "#111"
    desc_clr = "#555" if active else "#2a2a2a"
    return (
        f'<div class="alert-row" style="background:{bg_clr};border:1px solid {border_clr};border-radius:4px;">'
        f'<div class="alert-dot" style="background:{dot_clr};"></div>'
        f'<div>'
        f'<div class="alert-label" style="color:{lbl_clr}">{label}</div>'
        f'<div class="alert-desc" style="color:{desc_clr}">{description}</div>'
        f'</div>'
        f'</div>'
    )


def regime_color(label: str) -> str:
    return _REGIME_COLORS.get(label, "#9e9e9e")


def score_color(s: float) -> str:
    try:
        f = float(s)
    except (TypeError, ValueError):
        return "#555555"
    if not _math.isfinite(f):
        return "#555555"   # NaN / inf → neutral grey, not crash-red
    for threshold, color in _SCORE_COLORS:
        if f >= threshold:
            return color
    return "#b71c1c"


def fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


def fmt_pct(v: float, plus: bool = True) -> str:
    sign = "+" if plus and v > 0 else ""
    return f"{sign}{v:.2%}"


def pnl_delta_color(v: float) -> str:
    return "normal" if v >= 0 else "inverse"


def _label_for_score(s: float) -> str:
    if s >= 0.78:
        return "Strong Bullish"
    if s >= 0.63:
        return "Moderate Bullish"
    if s >= 0.58:
        return "Mildly Bullish"   # aligned with direction="long" threshold
    if s > 0.42:
        return "Neutral"           # matches direction="neutral" band
    if s >= 0.37:
        return "Mildly Bearish"   # aligned with direction="short" threshold
    if s >= 0.22:
        return "Moderate Bearish"
    return "Strong Bearish"


# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Session clock ─────────────────────────────────────────────────────────
    _now_utc = datetime.now(timezone.utc)
    _h, _m = _now_utc.hour, _now_utc.minute
    _mins_since_mid = _h * 60 + _m
    _is_weekday = _now_utc.weekday() < 5
    _market_open = _is_weekday and 810 <= _mins_since_mid < 1200
    _pre_market = _is_weekday and 570 <= _mins_since_mid < 810
    if _market_open:
        _mkt_label, _mkt_clr = "MARKET OPEN",   "#00c851"
    elif _pre_market:
        _mkt_label, _mkt_clr = "PRE-MARKET",    "#ffbb33"
    else:
        _mkt_label, _mkt_clr = "MARKET CLOSED", "#444"

    _mkt_dot_anim = "animation:blink-dot 1.5s infinite;" if _market_open else ""

    # ── Sidebar top spacer (aligns with fixed header) ─────────────────────────
    st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)

    # ── Market status pill ────────────────────────────────────────────────────
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;"
        "padding:8px 0 6px 0;margin-bottom:4px;'>"
        "<span style='width:8px;height:8px;border-radius:50%;"
        "background:" + _mkt_clr + ";display:inline-block;" + _mkt_dot_anim + "'></span>"
        "<span style='font-size:0.68rem;font-weight:800;color:" + _mkt_clr + ";"
        "letter-spacing:0.10em;font-family:monospace'>" + _mkt_label + "</span>"
        "<span style='font-size:0.60rem;color:#AAAAAA;margin-left:auto;"
        "font-family:monospace'>" + _now_utc.strftime("%H:%M") + " UTC</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_secs = st.select_slider(
        "Interval",
        options=[15, 30, 60, 120, 300],
        value=30,
        disabled=not auto_refresh,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m",
    )

    paper_override = st.selectbox(
        "Alpaca mode",
        ["From .env", "Paper", "Live"],
        index=0,
    )

    if st.button("⚡ SYNC ALPACA", use_container_width=True, type="primary",
                 help="Force-refresh all Alpaca data and cached signals"):
        st.cache_data.clear()
        st.rerun()

    # On Streamlit Community Cloud the project root is read-only.
    # Detect cloud deployment and redirect all log/cache writes to /tmp,
    # which is the only writable directory available.
    # Streamlit Cloud mounts repos under /mount/src
    _IS_CLOUD = Path("/mount/src").exists()
    if _IS_CLOUD:
        _LOG_DIR = Path("/tmp/regime_trader_logs")
    else:
        _LOG_DIR = Path(str(_HERE / "logs"))
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    st.divider()

    # ── Tab guide (expandable) ────────────────────────────────────────────────
    _TAB_GUIDE = [
        ("📊", "Decision Matrix", [
            "01 · Market Status",
            "02 · Trader's Brief",
            "03–04 · Regime Analysis",
            "05 · Positions & Actions",
            "06 · Correlation Guard",
            "07 · Conviction Scoreboard",
            "08 · System Status",
        ]),
        ("🧠", "Market Intel", [
            "Top Buy Signals",
            "Multi-Timeframe Trends",
            "Radar · Symbol Compare",
            "Portfolio Scores",
            "Fundamental Filter",
        ]),
        ("📋", "Trade Log", [
            "Full trade history",
            "P&L by symbol",
        ]),
        ("📈", "Regime History", [
            "Regime timeline chart",
            "Confidence & drawdown",
        ]),
        ("🔄", "Portfolio Sync", [
            "Alpaca sync controls",
            "JSON import/export",
        ]),
        ("🌍", "Macro Center", [
            "Yield Curve · VIX · DXY",
            "Sector heatmap",
            "Commodity prices",
        ]),
    ]
    for _tg_icon, _tg_name, _tg_sections in _TAB_GUIDE:
        with st.expander(f"{_tg_icon}  {_tg_name}", expanded=False):
            for _s in _tg_sections:
                st.markdown(
                    f"<div style='font-size:0.63rem;color:#AAAAAA;font-family:monospace;"
                    f"padding:2px 0 2px 4px;border-left:1px solid #1a1a1a'>→ {_s}</div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Advanced / log dir ────────────────────────────────────────────────────
    with st.expander("Advanced"):
        log_dir_override = st.text_input("Log dir", value=str(_LOG_DIR))
        _LOG_DIR = Path(log_dir_override)

    st.markdown(
        "<div style='font-size:0.52rem;color:#AAAAAA;font-family:monospace;"
        "padding:4px 0;letter-spacing:0.06em;'>v1.0 · "
        + _now_utc.strftime("%Y-%m-%d") +
        "</div>",
        unsafe_allow_html=True,
    )

# Auto-refresh via meta-refresh (no extra package needed)
if auto_refresh:
    st.components.v1.html(
        f'<meta http-equiv="refresh" content="{refresh_secs}">',
        height=0,
    )

# ══════════════════════════════════════════════════════════════════════════════
# ── Data loading ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=15)
def _read_ndjson(path: Path, tail: int = 2000) -> pd.DataFrame:
    """
    Read the last `tail` lines of an NDJSON log file into a DataFrame.
    Returns an empty DataFrame if the file doesn't exist.
    """
    if not path.exists():
        return pd.DataFrame()
    lines: List[str] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return pd.DataFrame()

    lines = lines[-tail:]
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=15)
def _fetch_alpaca_data(api_key: str, secret: str, is_paper: bool, base_url: str) -> Dict[str, Any]:
    """
    Cached inner function: fetches raw Alpaca account + positions data.
    Must not touch st.session_state or any Streamlit widget state.
    Raises on failure so the non-cached wrapper can handle error reporting.
    """
    from alpaca.trading.client import TradingClient  # type: ignore
    client = TradingClient(
        api_key, secret, paper=is_paper, url_override=base_url)
    acct = client.get_account()
    raw_positions = client.get_all_positions()

    positions = []
    for p in raw_positions:
        positions.append({
            "symbol":             p.symbol,
            "qty":                float(p.qty),
            "avg_cost":           float(p.avg_entry_price or 0),
            "current_price":      float(p.current_price or 0),
            "market_value":       float(p.market_value or 0),
            "unrealized_pnl":     float(p.unrealized_pl or 0),
            "unrealized_pnl_pct": float(p.unrealized_plpc or 0),
            "side":               str(p.side),
        })

    equity_val = float(acct.equity or 0)
    prev_close = float(getattr(acct, "equity_previous_close", None) or
                       getattr(acct, "last_equity", None) or equity_val)
    daily_pnl = equity_val - prev_close

    return {
        "equity":       equity_val,
        "cash":         float(acct.cash or 0),
        "buying_power": float(acct.buying_power or 0),
        "daily_pnl":    daily_pnl,
        "positions":    positions,
        "paper_mode":   is_paper,
        "source":       "alpaca",
    }


def _get_alpaca_portfolio() -> Optional[Dict[str, Any]]:
    """
    Fetch live account equity + open positions from Alpaca.
    Returns None if credentials are missing or alpaca-py is not installed.
    This wrapper is intentionally NOT cached so it can safely access
    st.session_state and the paper_override widget value.
    """
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or api_key.startswith("PK"):
        # placeholder value — skip
        if api_key == "PKXXXXXXXXXXXXXXXXXXXXXXXX" or not api_key:
            return None

    env_val = os.getenv("ENV", "paper").lower()
    if paper_override == "Paper":
        is_paper = True
    elif paper_override == "Live":
        is_paper = False
    else:
        is_paper = env_val != "live"

    base_url = (
        os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
        if is_paper
        else os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
    )

    try:
        return _fetch_alpaca_data(api_key, secret, is_paper, base_url)
    except Exception as _exc:
        # Surface the error so the dashboard shows it instead of silently failing.
        # Safe here because this function is NOT inside a @st.cache_data decorator.
        st.session_state["_alpaca_error"] = str(_exc)
        return None


def _latest_row(df: pd.DataFrame) -> Optional[Dict]:
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def _get_regime_state() -> Dict[str, Any]:
    """Latest regime state: session-state live calc first (if < 15 min), then regime.log fallback."""
    # Priority 1: live HMM result stored by _run_live_regime() — expire after 15 minutes
    if "live_regime_state" in st.session_state:
        _live = st.session_state["live_regime_state"]
        _ts = _live.get("_live_calc_ts")
        if _ts:
            try:
                _age_s = (
                    datetime.now(tz=timezone.utc)
                    - datetime.fromisoformat(_ts)
                ).total_seconds()
                if _age_s < 900:   # 15-minute TTL
                    return _live
                # Stale — fall through to log so collect_data.py results take over
            except Exception:
                return _live   # unparseable timestamp: keep showing it
        else:
            return _live   # no timestamp: assume fresh

    # Priority 2: log file written by the background engine
    df = _read_ndjson(_LOG_DIR / "regime.log", tail=500)
    if df.empty:
        return {
            "label": "Unknown", "confidence": 0.0,
            "stability_bars": 0, "flicker_count": 0,
            "probs": {}, "is_uncertain": False,
        }
    row = _latest_row(df)
    probs = row.get("probs", {}) or {}
    return {
        "label":          row.get("new_regime", row.get("regime", "Unknown")),
        "confidence":     float(row.get("confidence", row.get("probability", 0.0))),
        "stability_bars": int(row.get("stability_bars", 0)),
        "flicker_count":  int(row.get("flicker_count", 0)),
        "probs":          probs,
        "is_uncertain":   bool(row.get("is_uncertain", False)),
    }


def _fetch_macro_df(years: int = 5) -> pd.DataFrame:
    """
    Download VIX, 10Y yield, 3M yield, and DXY from yfinance.
    Returns a DataFrame with columns: vix, tnx, irx, dxy  (date-indexed, daily).
    Falls back gracefully — missing tickers return NaN columns.
    """
    try:
        import yfinance as _yf
        tickers = {"^VIX": "vix", "^TNX": "tnx",
                   "^IRX": "irx", "DX-Y.NYB": "dxy"}
        frames = {}
        for yf_sym, col in tickers.items():
            try:
                raw = _yf.download(
                    yf_sym, period=f"{years}y", interval="1d",
                    progress=False, auto_adjust=True,
                )
                if not raw.empty:
                    close = raw["Close"].squeeze()
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    frames[col] = close
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        macro = pd.DataFrame(frames)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        return macro
    except Exception:
        return pd.DataFrame()


@st.cache_resource(ttl=3600)
def _get_fitted_classifier(years_back: int = 3):
    """
    Fit the HMM classifier on SPY and cache the result for 1 hour.
    Using cache_resource so the heavy clf.fit() (BIC selection over n_states 3-7)
    runs at most once per hour across all reruns and user interactions.
    Returns (clf, features, ohlcv).
    """
    from data.market_data import MarketData
    from feature_engineering.feature_engineering import FeatureEngineer
    from hmm_engine.classifier import RegimeClassifier

    ohlcv = MarketData().get_historical_bars("SPY", years_back=years_back)
    eng = FeatureEngineer()
    features, returns, _ = eng.build(ohlcv, fit_scaler=True)
    clf = RegimeClassifier()
    clf.fit(features, returns)
    return clf, features, ohlcv


def _run_live_regime() -> Dict[str, Any]:
    """
    Run the HMM classifier on SPY live and store result in session state.
    Returns the regime dict on success, or last known on failure.
    """
    try:
        clf, features, _ = _get_fitted_classifier(years_back=3)
        clf._filter.reset()   # prevent stale filter state across cached clf reuse
        result = clf.predict_current(features)

        label = result.confirmed_label or result.raw_label
        raw_probs = {
            lbl: float(result.regime_probs[idx])
            for idx, lbl in result.label_map.items()
        }
        # Apply 2% floor + renormalise so no state displays as 100% or 0%.
        # The HMM forward filter naturally concentrates after 700+ observations;
        # the raw output is mathematically correct but visually misleading.
        import numpy as _np_hmm
        _eps = 0.02
        _vals = _np_hmm.array(list(raw_probs.values()))
        _vals = _np_hmm.maximum(_vals, _eps)
        _vals /= _vals.sum()
        probs_dict = {lbl: round(float(_vals[i]), 4)
                      for i, lbl in enumerate(raw_probs.keys())}
        # smoothed confidence for the winning state
        confidence = probs_dict[label]

        state = {
            "label":          label,
            "confidence":     confidence,
            "stability_bars": result.streak,
            "flicker_count":  result.flicker_count,
            "probs":          probs_dict,
            "is_uncertain":   result.is_uncertain,
            "n_states":       result.n_regimes,
            "_live_calc_ts":  datetime.now(tz=timezone.utc).isoformat(),
        }
        st.session_state["live_regime_state"] = state
        return state
    except Exception as exc:
        # Always surface the error — merge into old state so the caller can
        # show it AND still display stale values instead of a blank panel.
        old = dict(st.session_state.get("live_regime_state", {
            "label": "Unknown", "confidence": 0.0, "stability_bars": 0,
            "flicker_count": 0, "probs": {}, "is_uncertain": False,
        }))
        old["_error"] = str(exc)
        return old


def _run_regime_backfill(years: int = 5) -> Optional[pd.DataFrame]:
    """
    Run predict_sequence() on full historical SPY bars (with macro features).
    Returns a DataFrame with columns matching regime.log format so the
    Regime History tab can render it identically.
    Stores result in st.session_state['regime_backfill_df'].
    """
    try:
        clf, features, ohlcv = _get_fitted_classifier(years_back=years)
        macro_df = _fetch_macro_df(years=years)
        seq_df = clf.predict_sequence(features)   # one row per bar

        # Align dates: feature matrix drops NaN warm-up rows from the front.
        # Normalise to tz-naive to avoid datetime64[us,UTC] vs datetime64[s] errors.
        n_dropped = len(ohlcv) - len(features)
        raw_dates = ohlcv.index[n_dropped:]
        if raw_dates.tz is not None:
            raw_dates = raw_dates.tz_convert(None)
        # date-only, no time component
        dates = raw_dates.normalize()[:len(seq_df)]

        # Reset seq_df to RangeIndex before building out — critical to avoid
        # index-alignment NaN when seq_df has a DatetimeIndex and out uses RangeIndex.
        seq_df = seq_df.reset_index(drop=True)

        prob_cols_seq = [c for c in seq_df.columns if c.startswith("prob_")]

        # Ensure regime label is always a plain non-null string
        new_regime_ser = seq_df["confirmed_label"].where(
            seq_df["confirmed_label"].notna(), seq_df["raw_label"]
        ).fillna("Unknown").astype(str)

        confidence_ser = (
            seq_df[prob_cols_seq].max(axis=1)
            if prob_cols_seq
            else pd.Series(0.5, index=seq_df.index)
        )
        out = pd.DataFrame({
            "timestamp":      pd.Series(dates[:len(seq_df)]),
            "new_regime":     new_regime_ser,
            "confidence":     confidence_ser,
            "flicker_count":  seq_df["flicker_count"],
            "stability_bars": seq_df["streak"],
        })
        out["old_regime"] = out["new_regime"].shift(1).fillna("Unknown")
        out["transition"] = out["new_regime"] != out["old_regime"]

        # Prob columns per regime label
        for col in prob_cols_seq:
            # strip "prob_", use .values to avoid index issues
            out[col[5:]] = seq_df[col].values

        out = out[out["new_regime"] != "Unknown"].reset_index(drop=True)
        st.session_state["regime_backfill_df"] = out
        st.session_state["regime_backfill_years"] = years
        st.session_state["regime_backfill_ts"] = datetime.now(
            tz=timezone.utc).isoformat()
        # Store macro separately for overlay charts (not used in HMM)
        if not macro_df.empty:
            macro_df.index = pd.to_datetime(
                macro_df.index).tz_localize(None).normalize()
            st.session_state["regime_backfill_macro"] = macro_df
        return out

    except Exception as exc:
        st.session_state["regime_backfill_error"] = str(exc)
        return None


def _get_portfolio_state() -> Dict[str, Any]:
    """Portfolio state: Alpaca API first, then main.log context fallback."""
    alpaca = _get_alpaca_portfolio()
    if alpaca:
        return alpaca

    df = _read_ndjson(_LOG_DIR / "main.log", tail=200)
    if df.empty:
        return {"equity": 0.0, "cash": 0.0, "daily_pnl": 0.0,
                "positions": [], "paper_mode": True, "source": "none"}
    row = _latest_row(df)
    return {
        "equity":    float(row.get("equity", 0.0)),
        "cash":      0.0,
        "daily_pnl": float(row.get("daily_pnl", 0.0)),
        "positions": row.get("positions", []),
        "paper_mode": True,
        "source":    "log",
    }


def _get_risk_state() -> Dict[str, Any]:
    """Latest risk metrics from alerts.log."""
    df = _read_ndjson(_LOG_DIR / "alerts.log", tail=100)
    default = {
        "daily_dd": 0.0, "daily_dd_limit": -0.03,
        "peak_dd":  0.0, "peak_dd_limit": -0.10,
        "cb_level": "NONE",
    }
    if df.empty:
        return default
    cb_rows = df[df.get("alert_type", pd.Series(dtype=str)) == "circuit_breaker"] \
        if "alert_type" in df.columns else pd.DataFrame()
    if cb_rows.empty:
        return default
    row = cb_rows.iloc[-1].to_dict()
    return {
        "daily_dd":       float(row.get("drawdown", 0.0)),
        "daily_dd_limit": -0.03,
        "peak_dd":        float(row.get("drawdown", 0.0)),
        "peak_dd_limit": -0.10,
        "cb_level":       str(row.get("cb_level", "NONE")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Multi-timeframe technical analysis ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def _multi_tf_signals(symbols: List[str]) -> pd.DataFrame:
    """
    Compute daily / weekly / monthly technical signals via yfinance.
    Returns a DataFrame per symbol with trend, RSI, and composite tf_score.
    """
    try:
        import yfinance as _yf
    except ImportError:
        return pd.DataFrame()

    def _tf_score(prices: "pd.Series") -> Tuple[float, float, float]:
        """(trend 0-1, rsi 0-100, score 0-1) for a price series."""
        if len(prices) < 14:
            return 0.5, 50.0, 0.5
        sma20 = prices.rolling(min(20, len(prices))).mean().iloc[-1]
        sma50 = prices.rolling(min(50, len(prices))).mean().iloc[-1]
        last = float(prices.iloc[-1])
        if last > sma20 and sma20 > sma50:
            trend = 1.0
        elif last < sma20 and sma20 < sma50:
            trend = 0.0
        else:
            trend = 0.5
        delta = prices.diff().dropna()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_raw = float((100 - 100 / (1 + rs)).iloc[-1])
        rsi_raw = max(0.0, min(100.0, rsi_raw))
        rsi_score = max(0.0, min(1.0, (rsi_raw - 20) / 60))  # 20→0, 80→1
        score = round(0.55 * trend + 0.45 * rsi_score, 4)
        return trend, round(rsi_raw, 1), score

    rows = []
    for sym in symbols:
        try:
            raw = _yf.download(sym, period="2y", interval="1d",
                               progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 30:
                continue
            close = raw["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            d_trend, d_rsi, d_score = _tf_score(close)
            w_prices = close.resample("W").last().dropna()
            w_trend, w_rsi, w_score = _tf_score(w_prices) if len(
                w_prices) >= 14 else (0.5, 50.0, 0.5)
            m_prices = close.resample("ME").last().dropna()
            m_trend, m_rsi, m_score = _tf_score(m_prices) if len(
                m_prices) >= 6 else (0.5, 50.0, 0.5)

            # Weighted: monthly confirms direction, daily for timing
            tf_score = round(0.25 * d_score + 0.35 *
                             w_score + 0.40 * m_score, 4)

            rows.append({
                "symbol":        sym,
                "daily_trend":   d_trend,   "daily_rsi":   d_rsi,   "daily_score":   d_score,
                "weekly_trend":  w_trend,   "weekly_rsi":  w_rsi,   "weekly_score":  w_score,
                "monthly_trend": m_trend,   "monthly_rsi": m_rsi,   "monthly_score": m_score,
                "tf_score":      tf_score,
            })
        except Exception as _te:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Technical signals failed for %s: %s", sym, _te)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("tf_score", ascending=False)


def _action_rating(
    intel_score: float,
    unreal_pnl_pct: float,
    regime_lbl: str,
    tf_score: float = 0.5,
) -> Tuple[str, str, str]:
    """
    Returns (action_label, badge_color, reason) for a held position.

    Decision colors: BUY=#00C851 (Neon Green), TRIM=#FFBB33 (Amber), SELL=#FF4444 (Ruby Red).
    No BUY signals are generated when the regime is Bear, Panic, or Crash.
    """
    bull_regime = regime_lbl in ("Bull", "Euphoria", "Mania")
    bear_regime = regime_lbl in ("Bear", "Panic", "Crash")
    has_intel = abs(intel_score - 0.50) > 0.04

    pnl_contrib = max(-0.15, min(0.15, unreal_pnl_pct))
    combined = 0.50 * intel_score + 0.30 * \
        tf_score + 0.20 * (0.5 + pnl_contrib)
    combined = max(0.0, min(1.0, combined))

    # Hard regime overrides — cascade from most severe
    if regime_lbl == "Crash":
        return "SELL", "#ff4444", "Crash regime — capital preservation, exit all longs"
    if regime_lbl == "Panic":
        return "TRIM", "#ffbb33", "Panic regime — reduce all exposure immediately"

    # P&L stop-loss trigger
    if unreal_pnl_pct < -0.12:
        return "TRIM", "#ffbb33", f"Down {unreal_pnl_pct*100:.1f}% — review stop"

    # Bear regime: maximum signal is TRIM — no new buys permitted
    if bear_regime:
        if combined >= 0.55:
            return "TRIM", "#ffbb33", "Bear regime — hold but do not add"
        return "SELL", "#ff4444", "Weak conviction in bear regime — exit"

    # Bull / Neutral regime logic
    if combined >= 0.70 and bull_regime:
        return "BUY MORE", "#00c851", "Strong bull + high conviction"
    if combined >= 0.62 and not bear_regime and has_intel:
        return "ADD", "#7cb342", "Positive signals — consider adding"
    if combined >= 0.44:
        return "HOLD", "#9e9e9e", "Neutral — hold current size"
    if combined >= 0.32:
        return "TRIM", "#ffbb33", "Weak signals — reduce position"
    return "SELL", "#ff4444", "Strong sell signal"


def _portfolio_risk_score(
    positions: List[Dict],
    regime_lbl: str,
    scores_df: "Optional[pd.DataFrame]",
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a 0-100 portfolio risk score and a breakdown by component.
    Higher = riskier.
    """
    components: Dict[str, float] = {}

    # 1. Regime risk (0-40)
    regime_risk_map = {
        "Bull": 10, "Euphoria": 20, "Mania": 35,
        "Neutral": 20,
        "Bear": 35,  "Panic": 40,  "Crash": 40,
        "Unknown": 25,
    }
    components["Regime"] = regime_risk_map.get(regime_lbl, 25)

    # 2. Concentration risk (0-30): top position > 25% of portfolio
    if positions:
        total_mv = sum(float(p.get("market_value", 0)) for p in positions)
        if total_mv > 0:
            top_weight = max(float(p.get("market_value", 0)) /
                             total_mv for p in positions)
            components["Concentration"] = min(30, top_weight * 80)
        else:
            components["Concentration"] = 0.0
        # 3. Intel risk (0-30): avg intel score < 0.5 = risky
        if scores_df is not None and not scores_df.empty and "symbol" in scores_df.columns:
            held_syms = [p["symbol"] for p in positions]
            overlap = scores_df[scores_df["symbol"].isin(held_syms)]
            if not overlap.empty:
                avg_intel = float(overlap["composite"].mean())
                components["Intel"] = round((1.0 - avg_intel) * 30, 1)
            else:
                components["Intel"] = 15.0
        else:
            components["Intel"] = 15.0
    else:
        components["Concentration"] = 0.0
        components["Intel"] = 0.0

    total = round(sum(components.values()), 1)
    return min(100.0, total), components


# ══════════════════════════════════════════════════════════════════════════════
# ── Technical Signals (SMA50/200, RSI14, ATR, Beta) ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=1800)
def _get_technical_signals(symbols: Tuple) -> pd.DataFrame:
    """
    Per-symbol professional technical analysis — cached 30 min.

    Computes:
    - SMA50 / SMA200 crossover → Trend Status + color
    - RSI(14) with Overbought (>70) / Oversold (<30) labels
    - ATR(14) vs 30-day average → Volatility Alert if >20% above
    - Beta vs SPY (market-value-weighted portfolio beta computed by caller)
    - Daily Action signal: BUY / HOLD / TRIM / SELL based on crossover + RSI

    Parameters
    ----------
    symbols : tuple of ticker strings (tuple so it is hashable for st.cache_data)

    Returns
    -------
    pd.DataFrame  one row per symbol
    """
    try:
        import yfinance as _yf
    except ImportError:
        return pd.DataFrame()

    if not symbols:
        return pd.DataFrame()

    # Fetch SPY once for Beta calculation
    spy_ret: Optional[pd.Series] = None
    try:
        _spy = _yf.download("SPY", period="1y", interval="1d",
                            progress=False, auto_adjust=True)
        spy_ret = _spy["Close"].squeeze().pct_change().dropna()
        if isinstance(spy_ret, pd.DataFrame):
            spy_ret = spy_ret.iloc[:, 0]
    except Exception:
        pass

    rows = []
    for sym in symbols:
        try:
            raw = _yf.download(sym, period="1y", interval="1d",
                               progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 20:
                continue

            close = raw["Close"].squeeze()
            high = raw["High"].squeeze()
            low = raw["Low"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high,  pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low,   pd.DataFrame):
                low = low.iloc[:, 0]

            price = float(close.iloc[-1])
            sma50 = float(close.rolling(50).mean(
            ).iloc[-1]) if len(close) >= 50 else None
            sma200 = float(close.rolling(200).mean(
            ).iloc[-1]) if len(close) >= 200 else None

            # ── Trend Status (50/200 MA crossover — FT/professional standard) ──
            if sma50 is not None and sma200 is not None:
                if sma50 > sma200 and price > sma50:
                    trend_status, trend_clr = "Golden Cross ▲", "#00c851"
                elif sma50 < sma200 and price < sma50:
                    trend_status, trend_clr = "Death Cross ▼",  "#ff4444"
                elif price > sma50:
                    trend_status, trend_clr = "Above 50MA →",   "#7cb342"
                else:
                    trend_status, trend_clr = "Below 50MA →",   "#ff8800"
            elif sma50 is not None:
                if price > sma50:
                    trend_status, trend_clr = "Above 50MA →", "#7cb342"
                else:
                    trend_status, trend_clr = "Below 50MA →", "#ff8800"
            else:
                trend_status, trend_clr = "N/A", "#555"

            # ── RSI(14) ────────────────────────────────────────────────────────
            delta = close.diff().dropna()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-9)
            rsi = round(
                float(max(0.0, min(100.0, (100 - 100 / (1 + rs)).iloc[-1]))), 1)
            if rsi > 70:
                rsi_lbl, rsi_clr = "Overbought", "#ff4444"
            elif rsi < 30:
                rsi_lbl, rsi_clr = "Oversold",   "#00c851"
            else:
                rsi_lbl, rsi_clr = "Neutral",    "#9e9e9e"

            # ── ATR(14) — Volatility Alert if >20% above 30-day avg ───────────
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
            atr30 = float(tr.rolling(30).mean(
            ).iloc[-1]) if len(tr) >= 30 else atr14
            atr_pct = (atr14 / max(atr30, 1e-9) - 1.0)
            atr_alert = atr_pct >= 0.20

            # ── Beta vs SPY ────────────────────────────────────────────────────
            beta: Optional[float] = None
            if spy_ret is not None:
                sym_ret = close.pct_change().dropna()
                common = sym_ret.index.intersection(spy_ret.index)
                if len(common) >= 30:
                    s = sym_ret.loc[common]
                    m = spy_ret.loc[common]
                    var_m = float(m.var())
                    if var_m > 1e-12:
                        beta = round(float(s.cov(m)) / var_m, 2)

            # ── Daily Action (crossover + RSI) ─────────────────────────────────
            is_death = trend_status.startswith("Death")
            is_golden = trend_status.startswith("Golden")
            if rsi > 70 and is_death:
                daily_action, act_clr = "SELL", "#ff4444"
            elif rsi > 70:
                daily_action, act_clr = "TRIM", "#ff8800"
            elif is_golden and rsi < 65:
                daily_action, act_clr = "BUY",  "#00c851"
            elif is_death:
                daily_action, act_clr = "SELL", "#ff4444"
            elif rsi < 30:
                daily_action, act_clr = "BUY",  "#00c851"
            else:
                daily_action, act_clr = "HOLD", "#9e9e9e"

            rows.append({
                "symbol":       sym,
                "price":        price,
                "sma50":        round(sma50,  2) if sma50 else None,
                "sma200":       round(sma200, 2) if sma200 else None,
                "trend_status": trend_status,
                "trend_clr":    trend_clr,
                "rsi14":        rsi,
                "rsi_lbl":      rsi_lbl,
                "rsi_clr":      rsi_clr,
                "atr14":        round(atr14, 4),
                "atr30":        round(atr30, 4),
                "atr_alert":    atr_alert,
                "atr_pct":      round(atr_pct * 100, 1),
                "beta":         beta,
                "daily_action": daily_action,
                "act_clr":      act_clr,
            })
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ── Fundamental Quality Score (Buffett-style) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=86400)
def _get_quality_scores(symbols: Tuple) -> pd.DataFrame:
    """
    Buffett-style fundamental quality score — cached 24 hours.

    Criteria (all from yfinance.Ticker.info):
    - Debt-to-Equity < 1     (25 pts)
    - Positive Free Cash Flow (30 pts)
    - Net Margin >15%         (30 pts — scaled for 8–15%)
    - ROE > 15%               (15 pts)

    Value Insight label:
    ≥70 → Buffett Quality | ≥50 → Solid Business | ≥30 → Mixed | <30 → Avoid
    """
    try:
        import yfinance as _yf
    except ImportError:
        return pd.DataFrame()

    if not symbols:
        return pd.DataFrame()

    rows = []
    for sym in symbols:
        try:
            info = _yf.Ticker(sym).info or {}

            # yfinance returns %, 100 = ratio 1.0
            de = info.get("debtToEquity")
            fcf = info.get("freeCashflow")
            net_mg = info.get("profitMargins")     # fraction, e.g. 0.22 = 22%
            roe = info.get("returnOnEquity")    # fraction

            # Component scores 0–1
            de_score = (1.0 if de is not None and de < 100 else
                        0.5 if de is None else 0.0)
            fcf_score = (1.0 if fcf is not None and fcf > 0 else
                         0.5 if fcf is None else 0.0)
            mg_score = (1.0 if net_mg is not None and net_mg > 0.15 else
                        0.7 if net_mg is not None and net_mg > 0.08 else
                        0.3 if net_mg is not None and net_mg > 0 else
                        0.5 if net_mg is None else 0.0)
            roe_score = (1.0 if roe is not None and roe > 0.15 else
                         0.5 if roe is None else 0.3)

            quality = round(
                de_score * 25 +
                fcf_score * 30 +
                mg_score * 30 +
                roe_score * 15, 1
            )

            if quality >= 70:
                insight, ins_clr = "Buffett Quality", "#00c851"
            elif quality >= 50:
                insight, ins_clr = "Solid Business",  "#7cb342"
            elif quality >= 30:
                insight, ins_clr = "Mixed Signals",   "#ff8800"
            else:
                insight, ins_clr = "Avoid",            "#ff4444"

            rows.append({
                "symbol":        sym,
                "sector":        info.get("sector", "Unknown"),
                "quality_score": quality,
                "debt_equity":   round(de / 100, 2) if de is not None else None,
                "free_cashflow": fcf,
                "net_margin":    round(net_mg * 100, 1) if net_mg is not None else None,
                "roe":           round(roe * 100, 1) if roe is not None else None,
                "value_insight": insight,
                "insight_color": ins_clr,
            })
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ── Unified Conviction Engine ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=1800)
def _get_unified_conviction(
    symbols: Tuple,
    regime_lbl: str,
    log_dir_str: str,
) -> pd.DataFrame:
    """
    Single Unified Conviction score (0–1) per symbol.

    Weights
    ───────
    Fundamental  40%  — Buffett quality score normalised 0–1
    Technical    40%  — Price vs SMA200 (trend) + RSI discipline
    Intelligence 20%  — Intel composite from latest_scores.json

    Regime Multiplier (R)  ←  applied after blending
    ─────────────────────
    Bull / Euphoria  ×1.2   (risk-on)
    Bear / Panic     ×0.5   (defensive)
    Crash            ×0.0   (hard halt)

    Returns DataFrame with columns:
        symbol, conviction, fund_score, tech_score, intel_score,
        regime_mult, grade (A/B/C), grade_clr, price, atr14
    """
    if not symbols:
        return pd.DataFrame()

    qual_df = _get_quality_scores(symbols)
    tech_df = _get_technical_signals(symbols)

    # Intel scores from disk
    intel_map: Dict[str, float] = {}
    _sp = Path(log_dir_str) / "latest_scores.json"
    if _sp.exists():
        try:
            for r in json.loads(_sp.read_text(encoding="utf-8")):
                intel_map[r["symbol"]] = float(r.get("composite", 0.5))
        except Exception as _ie:
            st.warning(
                f"Intel scores could not be loaded — conviction will use neutral (0.50) for all symbols. Error: {_ie}")

    regime_mult = _REGIME_MULT.get(regime_lbl, 1.0)
    rows: List[Dict] = []

    for sym in symbols:
        # ── Fundamental (40%) ─────────────────────────────────────────────
        fund = 0.50
        if qual_df is not None and not qual_df.empty and "symbol" in qual_df.columns:
            qr = qual_df[qual_df["symbol"] == sym]
            if not qr.empty:
                fund = float(qr.iloc[0].get("quality_score", 50)) / 100.0

        # ── Technical (40%) ───────────────────────────────────────────────
        # Trend: price vs SMA200 (60% weight); RSI discipline (40% weight)
        tech = 0.50
        atr14_val: Optional[float] = None
        price_val: float = 0.0
        if tech_df is not None and not tech_df.empty and "symbol" in tech_df.columns:
            tr = tech_df[tech_df["symbol"] == sym]
            if not tr.empty:
                r = tr.iloc[0]
                price_val = float(r.get("price", 0))
                sma50_v = r.get("sma50")
                sma200_v = r.get("sma200")
                rsi_v = float(r.get("rsi14", 50))
                atr14_val = float(r.get("atr14", 0)) or None

                # Trend component (golden / death cross logic)
                if sma200_v is not None and sma50_v is not None:
                    if price_val > sma200_v and sma50_v > sma200_v:
                        trend_c = 0.80   # full golden cross
                    elif price_val > sma200_v:
                        trend_c = 0.65   # above 200, mixed MAs
                    elif price_val > sma50_v:
                        trend_c = 0.42   # between MAs
                    else:
                        trend_c = 0.18   # death cross
                elif sma200_v is not None:
                    trend_c = 0.65 if price_val > sma200_v else 0.25
                elif sma50_v is not None:
                    trend_c = 0.55 if price_val > sma50_v else 0.35
                else:
                    trend_c = 0.50

                # RSI discipline — RSI < 60 preferred; overbought penalised
                if rsi_v < 30:
                    rsi_c = 0.90    # oversold — value entry
                elif rsi_v < 45:
                    rsi_c = 0.75
                elif rsi_v < 60:
                    rsi_c = 0.60
                elif rsi_v < 70:
                    rsi_c = 0.38
                else:
                    rsi_c = 0.18    # overbought — avoid

                tech = round(0.60 * trend_c + 0.40 * rsi_c, 4)

        # ── Intelligence (20%) ────────────────────────────────────────────
        intel = intel_map.get(sym, 0.50)

        # ── Blend + Regime Multiplier ─────────────────────────────────────
        raw = 0.40 * fund + 0.40 * tech + 0.20 * intel
        conv = round(min(1.0, max(0.0, raw * regime_mult)), 4)

        if conv >= 0.70:
            grade, grade_clr = "A", "#00c851"
        elif conv >= 0.42:   # aligned with direction="neutral" band lower boundary
            grade, grade_clr = "B", "#ffbb33"
        else:
            grade, grade_clr = "C", "#ff4444"

        rows.append({
            "symbol":      sym,
            "conviction":  conv,
            "fund_score":  round(fund, 4),
            "tech_score":  round(tech, 4),
            "intel_score": round(intel, 4),
            "regime_mult": regime_mult,
            "grade":       grade,
            "grade_clr":   grade_clr,
            "price":       price_val,
            "atr14":       atr14_val,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ── Correlation Guard ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=1800)
def _compute_geo_risk_score() -> Dict[str, object]:
    """
    Composite geopolitical risk score [0, 1] derived from three independent
    market proxies — all via yfinance, no external API key needed.

    Components (equal 1/3 weight each):
      • VIX elevation   — (VIX − 15) / 25, clamped [0, 1].
                          VIX > 15 = elevated fear; > 25 = panic territory.
      • Gold momentum   — 20-day log-return of GC=F scaled [0, 1].
                          Gold rising = flight-to-safety demand.
      • Oil premium     — 20-day log-return of CL=F scaled [0, 1].
                          Oil rising during USD weakness = genuine war/supply risk.

    Returns dict with keys:
      score     [0, 1], label (str), vix (float), vix_comp [0,1],
      gold_20d (%), gold_comp [0,1], oil_20d (%), oil_comp [0,1]
    """
    import math as _m
    try:
        import yfinance as _yf
        _raw = _yf.download(
            ["^VIX", "GC=F", "CL=F"],
            period="60d", interval="1d",
            progress=False, auto_adjust=True,
        )

        def _last(ticker: str, col: str = "Close") -> Optional[float]:
            try:
                if isinstance(_raw.columns, pd.MultiIndex):
                    s = _raw[col][ticker].dropna()
                else:
                    s = _raw[col].dropna()
                return float(s.iloc[-1]) if not s.empty else None
            except Exception:
                return None

        def _ret20(ticker: str) -> Optional[float]:
            try:
                if isinstance(_raw.columns, pd.MultiIndex):
                    s = _raw["Close"][ticker].dropna()
                else:
                    s = _raw["Close"].dropna()
                if len(s) < 21:
                    return None
                return float(s.iloc[-1] / s.iloc[-21] - 1)
            except Exception:
                return None

        vix_val = _last("^VIX") or 20.0
        gold_20d = _ret20("GC=F")
        oil_20d = _ret20("CL=F")

        # Component scores — each clamped to [0, 1]
        vix_comp = max(0.0, min(1.0, (vix_val - 15.0) / 25.0))
        # Gold: 10% rally → score 1.0; flat → 0.5; -10% → 0.0
        gold_comp = max(0.0, min(1.0, 0.5 + (gold_20d or 0.0) * 5.0))
        # Oil: 10% rally → score 1.0; flat → 0.5; -10% → 0.0
        oil_comp = max(0.0, min(1.0, 0.5 + (oil_20d or 0.0) * 5.0))

        score = round((vix_comp + gold_comp + oil_comp) / 3.0, 3)

        if score >= 0.65:
            label = "ELEVATED WAR RISK"
            color = "#ff4444"
        elif score >= 0.45:
            label = "MODERATE GEOPOLITICAL TENSION"
            color = "#ff8800"
        else:
            label = "LOW RISK"
            color = "#00c851"

        return {
            "score":     score,
            "label":     label,
            "color":     color,
            "vix":       vix_val,
            "vix_comp":  round(vix_comp, 3),
            "gold_20d":  round((gold_20d or 0.0) * 100, 2),
            "gold_comp": round(gold_comp, 3),
            "oil_20d":   round((oil_20d or 0.0) * 100, 2),
            "oil_comp":  round(oil_comp, 3),
        }
    except Exception:
        return {
            "score": 0.5, "label": "DATA UNAVAILABLE", "color": "#9e9e9e",
            "vix": 20.0, "vix_comp": 0.33,
            "gold_20d": 0.0, "gold_comp": 0.5,
            "oil_20d": 0.0, "oil_comp": 0.5,
        }


# AI/Tech mega-cap tickers for concentration check
_AI_TICKERS: frozenset = frozenset({
    "NVDA", "MSFT", "AAPL", "GOOGL", "GOOG", "META",
    "AMZN", "TSLA", "AMD", "AVGO", "CRM", "ORCL",
})


def _ai_concentration_check(positions: List[Dict]) -> Dict:
    """
    Compute the fraction of portfolio market value held in AI/tech mega-caps.

    Returns dict with keys:
      pct (float 0-1), held (list of symbols), ai_mv (float), label (str), color (str)
    """
    ai_mv = sum(abs(float(p.get("market_value", 0))) for p in positions
                if p.get("symbol") in _AI_TICKERS)
    total_mv = sum(abs(float(p.get("market_value", 0))) for p in positions)
    pct = ai_mv / total_mv if total_mv > 0 else 0.0
    held = [p["symbol"] for p in positions if p.get("symbol") in _AI_TICKERS]
    if pct > 0.40:
        label = "AI CONCENTRATION RISK"
        color = "#ff4444"
    elif pct > 0.25:
        label = "ELEVATED AI EXPOSURE"
        color = "#ff8800"
    else:
        label = "AI EXPOSURE WITHIN LIMITS"
        color = "#00c851"
    return {"pct": pct, "held": held, "ai_mv": ai_mv, "label": label, "color": color}


@st.cache_data(ttl=3600)
def _get_correlation_matrix(symbols: Tuple) -> Optional[pd.DataFrame]:
    """
    Pearson correlation of daily returns (1-year lookback) for all given symbols.
    Returns None if fewer than 2 symbols or yfinance unavailable.
    Handles any portfolio size — caller limits display for readability.
    """
    if len(symbols) < 2:
        return None
    try:
        import yfinance as _yf
    except ImportError:
        return None
    try:
        raw = _yf.download(
            list(symbols), period="1y", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        # Handle both single-ticker (Series) and multi-ticker (DataFrame) return shapes
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            # Multi-ticker: extract Close for each symbol
            close_frames = {}
            for sym in symbols:
                if sym in raw.columns.get_level_values(0):
                    _s = raw[sym]["Close"].dropna()
                    if not _s.empty:
                        close_frames[sym] = _s
            if len(close_frames) < 2:
                return None
            close = pd.DataFrame(close_frames)
        else:
            if "Close" not in raw.columns:
                return None
            close = raw[["Close"]].rename(columns={"Close": symbols[0]})
            if close.shape[1] < 2:
                return None
        rets = close.pct_change().dropna()
        corr = rets.corr().round(2)
        common = [s for s in symbols if s in corr.columns]
        return corr.loc[common, common] if len(common) >= 2 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ── Macro strip data (VIX · SPY · 10Y) — 5-min cache ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _get_macro_header_data() -> Dict[str, Any]:
    """Fetch VIX, SPY 1-day %, and 10Y yield for the fixed header strip."""
    out = {"vix": None, "spy_chg": None, "yield10": None}
    try:
        import yfinance as _yf
        raw = _yf.download(
            ["^VIX", "SPY", "^TNX"],
            period="2d", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        if isinstance(raw.columns, pd.MultiIndex):
            def _last2(sym: str):
                try:
                    s = raw[sym]["Close"].dropna()
                    return float(s.iloc[-1]), float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1])
                except Exception:
                    return None, None
            v_now, _ = _last2("^VIX")
            spy_now, spy_prev = _last2("SPY")
            tnx_now, _ = _last2("^TNX")
            if v_now is not None:
                out["vix"] = round(v_now, 2)
            if spy_now is not None and spy_prev is not None and spy_prev > 0:
                out["spy_chg"] = round((spy_now / spy_prev - 1) * 100, 2)
            if tnx_now is not None:
                out["yield10"] = round(tnx_now, 3)
    except Exception:
        pass
    return out


_macro_hdr = _get_macro_header_data()

# ══════════════════════════════════════════════════════════════════════════════
# ── Fixed Alpha Terminal Header ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

_hdr_ts, _hdr_status_cls, _hdr_status_lbl = _get_last_updated()

# Derive live regime for the header pill (reads from cache — no extra I/O)
_hdr_regime_state = _get_regime_state()
_hdr_regime_lbl = _hdr_regime_state.get("label", "Unknown")
_hdr_regime_clr = _REGIME_COLORS.get(_hdr_regime_lbl, "#9e9e9e")
_hdr_regime_bg = _hdr_regime_clr + "18"
_hdr_regime_bdr = _hdr_regime_clr + "40"

# Status pill dot colour matches pill class
_hdr_dot_clr = {
    "ah-status-live":  "#00c851",
    "ah-status-stale": "#ffbb33",
    "ah-status-off":   "#ff4444",
}.get(_hdr_status_cls, "#555")

# Blinking dot only when truly live
_blink_cls = "blink" if _hdr_status_cls == "ah-status-live" else ""

_vix = _macro_hdr.get("vix")
_spy_chg = _macro_hdr.get("spy_chg")
_yield10 = _macro_hdr.get("yield10")

# VIX: warn-colour when > 20, red when > 30
_vix_cls = "mstrip-warn" if (_vix and _vix >
                             20) else ("mstrip-down" if (_vix and _vix > 30) else "")
_vix_str = f"{_vix:.1f}" if _vix is not None else "—"

_spy_str = (("+" if _spy_chg >= 0 else "") +
            f"{_spy_chg:.2f}%") if _spy_chg is not None else "—"
_spy_cls = "mstrip-up" if (_spy_chg and _spy_chg >
                           0) else ("mstrip-down" if (_spy_chg and _spy_chg < 0) else "")

_yield_str = f"{_yield10:.2f}%" if _yield10 is not None else "—"

# Build header as a compact string — NO HTML comments, NO indented lines.
# Indented lines inside st.markdown are treated as code blocks by the MD parser.
_hdr_html = (
    "<div id='alpha-header'>"

    # Left cell — logo (overlaps sidebar area)
    "<div id='alpha-header-logo'>"
    "<span class='ah-icon'>⚡</span>"
    "<div>"
    "<div class='ah-title'>Alpha&nbsp;Terminal</div>"
    "<div class='ah-version'>regime_trader&nbsp;v1.0</div>"
    "</div>"
    "</div>"

    # Right cell — macro strip + regime pill + divider + timestamp + status pill
    "<div id='alpha-header-right'>"

    # Macro strip cells
    "<div class='mstrip-cell'>"
    "<span class='mstrip-label'>VIX</span>"
    "<span class='mstrip-val " + _vix_cls + "'>" + _vix_str + "</span>"
    "</div>"
    "<div class='mstrip-cell'>"
    "<span class='mstrip-label'>SPY 1D</span>"
    "<span class='mstrip-val " + _spy_cls + "'>" + _spy_str + "</span>"
    "</div>"
    "<div class='mstrip-cell'>"
    "<span class='mstrip-label'>10Y</span>"
    "<span class='mstrip-val'>" + _yield_str + "</span>"
    "</div>"

    "<span style='width:1px;height:24px;background:#1e1e1e;display:inline-block;margin:0 4px;'></span>"

    "<span class='ah-regime-pill' style='"
    "background:" + _hdr_regime_bg + ";"
    "color:" + _hdr_regime_clr + ";"
    "border:1px solid " + _hdr_regime_bdr + ";'>"
    + _hdr_regime_lbl +
    "</span>"

    "<span style='width:1px;height:24px;background:#1e1e1e;display:inline-block;'></span>"

    "<div class='ah-ts-block'>"
    "<span class='ah-ts-label'>Last&nbsp;Updated</span>"
    "<span class='ah-ts-value'>" + _hdr_ts + "</span>"
    "</div>"

    "<span class='ah-status-pill " + _hdr_status_cls + "'>"
    "<span class='ah-status-dot " + _blink_cls +
    "' style='background:" + _hdr_dot_clr + ";'></span>"
    + _hdr_status_lbl +
    "</span>"

    "</div>"  # end right cell
    "</div>"  # end alpha-header
)
st.markdown(_hdr_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── Mobile dashboard (triggered by ?mobile=1 in the URL) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def _render_mobile_dashboard() -> None:
    """Phone-optimised single-column view for the Expo WebView app."""
    _regime = _get_regime_state()
    _portf = _get_portfolio_state()
    _rlbl = _regime.get("label", "Unknown")
    _rconf = _regime.get("confidence", 0.0)
    _rc = regime_color(_rlbl)
    _pos = _portf.get("positions", [])
    _equity = _portf.get("equity", 0.0)
    _cash = _portf.get("cash", 0.0)
    _dpnl = _portf.get("daily_pnl", 0.0)

    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:12px 0 4px">'
        '<span style="font-size:1.1rem;font-weight:900;color:#00c851;letter-spacing:6px">ALPHA</span>'
        '<span style="font-size:0.65rem;font-weight:700;color:#AAAAAA;letter-spacing:5px;display:block">TERMINAL</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Regime badge ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="text-align:center;padding:10px;margin:8px 0;'
        f'background:{_rc}1a;border:1px solid {_rc}44;border-radius:6px">'
        f'<div style="font-size:1.3rem;font-weight:900;color:{_rc};letter-spacing:4px">'
        f'{_rlbl.upper()}</div>'
        f'<div style="font-size:0.7rem;color:#AAAAAA;margin-top:4px">REGIME · CONF {_rconf:.0%}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Portfolio KPIs ────────────────────────────────────────────────────────
    _total_mv = sum(float(p.get("market_value", 0)) for p in _pos)
    _total_unrl = sum(float(p.get("unrealized_pnl", 0)) for p in _pos)
    _dpnl_clr = "#00c851" if _dpnl >= 0 else "#ff4444"

    _c1, _c2 = st.columns(2)
    _c1.metric("Equity", f"${_equity:,.0f}")
    _c2.metric("Daily P&L", f"${_dpnl:+,.0f}")

    # ── Positions list ────────────────────────────────────────────────────────
    if _pos:
        st.markdown(
            '<div style="font-size:0.7rem;color:#AAAAAA;letter-spacing:2px;margin:12px 0 6px">POSITIONS</div>',
            unsafe_allow_html=True,
        )
        for _p in _pos[:12]:
            _sym = _p.get("symbol", "?")
            _pl = float(_p.get("unrealized_pnl", _p.get("unrealized_pl", 0)))
            _plpc = float(_p.get("unrealized_plpc", 0))
            _mv = float(_p.get("market_value", 0))
            _clr = "#00c851" if _pl >= 0 else "#ff4444"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 10px;margin-bottom:4px;background:#0d0d0d;'
                f'border-left:3px solid {_clr}55;border-radius:3px">'
                f'<span style="font-weight:700;color:#ccc;font-size:0.85rem">{_sym}</span>'
                f'<span style="color:{_clr};font-size:0.85rem">'
                f'{_plpc:+.1%}&nbsp;<span style="color:#AAAAAA;font-size:0.72rem">${_mv:,.0f}</span>'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="text-align:center;color:#AAAAAA;font-size:0.75rem;padding:20px">No positions loaded</div>',
            unsafe_allow_html=True,
        )

    # ── Sector rotation hint ──────────────────────────────────────────────────
    _SECTOR_HINTS_MOB = {
        "Crash":    "DEFENSIVE: Cash + Short-duration Treasuries. Exit all risk assets.",
        "Panic":    "FLIGHT TO QUALITY: GLD, TLT, XLP, XLU. Avoid cyclicals.",
        "Bear":     "DEFENSIVE CYCLICALS: XLE, XLV. Reduce tech.",
        "Neutral":  "BALANCED: Mix XLF, XLI with XLV, XLP.",
        "Bull":     "GROWTH: Broad equities. Overweight XLK + XLI.",
        "Euphoria": "ROTATE OUT: Trim high-beta. Build GLD, Defense hedge.",
        "Mania":    "RISK REDUCTION: Reduce beta. Add GLD + RTX/LMT. Watch vol.",
    }
    st.info(
        f"**{_rlbl} Rotation:** {_SECTOR_HINTS_MOB.get(_rlbl, 'Monitor closely.')}")

    # ── Refresh ───────────────────────────────────────────────────────────────
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    if st.button("↻  Refresh", use_container_width=True, key="mob_refresh"):
        st.rerun()


# Detect mobile mode — set by Expo WebView via ?mobile=1 in the URL
_is_mobile = st.query_params.get("mobile", "0") == "1"
if _is_mobile:
    _render_mobile_dashboard()
    st.stop()

# ── Global session state defaults ─────────────────────────────────────────────
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = None

# ══════════════════════════════════════════════════════════════════════════════
# ── Tabs ───────────────────────────────────────════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

tab_monitor, tab_alpha, tab_trades, tab_regime, tab_sync, tab_macro, tab_valuation = st.tabs([
    "📊 Decision Matrix",
    "🔍 Alpha Hunter",
    "📋 Trade Log",
    "📈 Regime History",
    "🔄 Portfolio Sync",
    "🌍 Macro Center",
    "💹 Valuation Suite",
])

# Create inner sub-tabs for the combined Alpha Hunter tab
with tab_alpha:
    tab_intel, tab_discovery = st.tabs(
        ["🧠 Market Intel", "🔭 Discovery Scanner"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DECISION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Load and compute everything before any rendering
    # ══════════════════════════════════════════════════════════════════════════

    regime = _get_regime_state()
    portf = _get_portfolio_state()
    risk = _get_risk_state()
    regime_lbl = regime["label"]
    rc = regime_color(regime_lbl)
    equity = portf.get("equity", 0.0)
    daily_pnl = portf.get("daily_pnl", 0.0)
    cash = portf.get("cash", 0.0)
    positions = portf.get("positions", [])
    conf = regime["confidence"]
    alpaca_ok = portf.get("source") == "alpaca"

    daily_pnl_pct = daily_pnl / max(equity, 1)
    alloc_frac = 1.0 - cash / max(equity, 1)
    total_mv = sum(float(p.get("market_value", 0)) for p in positions)
    total_unreal = sum(float(p.get("unrealized_pnl", 0)) for p in positions)

    # Intel scores
    _scores_path_m = _LOG_DIR / "latest_scores.json"
    _scores_df_m: Optional[pd.DataFrame] = None
    if _scores_path_m.exists():
        try:
            _scores_df_m = pd.DataFrame(json.loads(
                _scores_path_m.read_text(encoding="utf-8")))
        except Exception as _e:
            _dm_logger.exception(
                "Failed to load latest_scores.json -- using empty scores_df: %s", _e)
            _scores_df_m = pd.DataFrame()

    risk_total, risk_breakdown = _portfolio_risk_score(
        positions, regime_lbl, _scores_df_m)
    if risk_total < 30:
        risk_color, risk_label = "#00c851", "Low"
    elif risk_total < 55:
        risk_color, risk_label = "#ffbb33", "Moderate"
    elif risk_total < 75:
        risk_color, risk_label = "#ff8800", "Elevated"
    else:
        risk_color, risk_label = "#ff4444", "High"

    # Technical signals + portfolio beta
    _held_syms_m: List[str] = [p["symbol"]
                               for p in positions] if positions else []
    _tech_df_m: Optional[pd.DataFrame] = None
    _portf_beta: Optional[float] = None
    if _held_syms_m:
        _tech_df_m = _get_technical_signals(tuple(_held_syms_m))
        if _tech_df_m is not None and not _tech_df_m.empty and "beta" in _tech_df_m.columns:
            _total_mv_m = sum(float(p.get("market_value", 0))
                              for p in positions)
            _beta_parts: List[float] = []
            for _p in positions:
                _br = _tech_df_m[_tech_df_m["symbol"] == _p["symbol"]]
                if not _br.empty:
                    _b = _br.iloc[0].get("beta")
                    _mv = float(_p.get("market_value", 0))
                    if _b is not None and _total_mv_m > 0:
                        _beta_parts.append(float(_b) * _mv / _total_mv_m)
            _portf_beta = round(sum(_beta_parts), 2) if _beta_parts else None

    # Unified Conviction Engine
    _conviction_df = _get_unified_conviction(
        tuple(_held_syms_m), regime_lbl, str(_LOG_DIR)
    ) if _held_syms_m else pd.DataFrame()

    # Crash conviction override — force all scores to 0/C so Brief is consistent
    if regime_lbl == "Crash" and not _conviction_df.empty:
        _conviction_df = _conviction_df.copy()
        _conviction_df["conviction"] = 0.0
        _conviction_df["grade"] = "C"
        _conviction_df["grade_clr"] = "#ff4444"

    # ATR alert flag
    _any_atr_alert = (
        _tech_df_m is not None and not _tech_df_m.empty
        and "atr_alert" in _tech_df_m.columns and _tech_df_m["atr_alert"].any()
    )

    # ── Build action rows (used in Brief + Positions table) ───────────────────
    _ACTION_URGENCY = {"SELL": 0, "TRIM": 1,
                       "BUY MORE": 2, "ADD": 3, "HOLD": 4}
    action_rows: List[Dict] = []
    if positions:
        _pos_df = pd.DataFrame(positions)
        _rename = {
            "unrealized_pnl": "unreal_pnl", "unrealized_pnl_pct": "unreal_pct",
            "avg_entry_price": "avg_cost",   "current_price": "price",
            "market_value": "mkt_value",     "unrealized_pl": "unreal_pnl",
            "unrealized_plpc": "unreal_pct",
        }
        _pos_df.rename(columns={k: v for k, v in _rename.items(
        ) if k in _pos_df.columns}, inplace=True)
        for _c in ["avg_cost", "price", "mkt_value", "unreal_pnl", "unreal_pct", "qty"]:
            if _c in _pos_df.columns:
                _pos_df[_c] = pd.to_numeric(
                    _pos_df[_c], errors="coerce").fillna(0.0)

        for _, _row in _pos_df.iterrows():
            _sym = _row.get("symbol", "")
            _upct = float(_row.get("unreal_pct", 0.0))
            _intel = 0.50
            if _scores_df_m is not None and not _scores_df_m.empty:
                _mm = _scores_df_m[_scores_df_m["symbol"] == _sym]
                if not _mm.empty:
                    _intel = float(_mm.iloc[0].get("composite", 0.50))

            _action, _act_clr, _reason = _action_rating(
                _intel, _upct, regime_lbl)

            _tr: Dict = {}
            if _tech_df_m is not None and not _tech_df_m.empty:
                _trm = _tech_df_m[_tech_df_m["symbol"] == _sym]
                if not _trm.empty:
                    _tr = _trm.iloc[0].to_dict()

            _cv: Dict = {}
            if not _conviction_df.empty:
                _cvm = _conviction_df[_conviction_df["symbol"] == _sym]
                if not _cvm.empty:
                    _cv = _cvm.iloc[0].to_dict()

            _atr14_v = _tr.get("atr14")
            _price_v = float(_row.get("price", 0))
            _mv_v = float(_row.get("mkt_value", 0))
            _atr_stop = round(_price_v - 3.0 * float(_atr14_v),
                              2) if _atr14_v and _price_v > 0 else None
            _risk_usd = round(_mv_v * (float(_atr14_v) / max(_price_v, 0.01)),
                              2) if _atr14_v and _price_v > 0 else None

            action_rows.append({
                "sym":          _sym,
                "action":       _action,
                "color":        _act_clr,
                "intel":        _intel,
                "upct":         _upct,
                "qty":          float(_row.get("qty", 0)),
                "cost":         float(_row.get("avg_cost", 0)),
                "price":        _price_v,
                "mv":           _mv_v,
                "pnl":          float(_row.get("unreal_pnl", 0)),
                "reason":       _reason,
                "_urgency":     _ACTION_URGENCY.get(_action, 99),
                "trend_status": _tr.get("trend_status", ""),
                "trend_clr":    _tr.get("trend_clr",    "#555"),
                "rsi14":        _tr.get("rsi14"),
                "rsi_lbl":      _tr.get("rsi_lbl",      ""),
                "rsi_clr":      _tr.get("rsi_clr",      "#555"),
                "daily_action": _tr.get("daily_action", ""),
                "act_clr":      _tr.get("act_clr",      "#555"),
                "atr_alert":    _tr.get("atr_alert",    False),
                "atr_pct":      _tr.get("atr_pct",      0),
                "atr14":        _atr14_v,
                "atr_stop":     _atr_stop,
                "risk_usd":     _risk_usd,
                "conviction":   float(_cv.get("conviction", 0.5)),
                "grade":        _cv.get("grade",     "B"),
                "grade_clr":    _cv.get("grade_clr", "#ffbb33"),
            })

        action_rows.sort(key=lambda r: (
            r["_urgency"], -r.get("mv", 0), r.get("upct", 0)))

    # ── Sector audit (>25% in one GICS sector = concentration warning) ────────
    _sector_warnings: List[str] = []
    if _held_syms_m:
        _qual_held = _get_quality_scores(tuple(_held_syms_m))
        if _qual_held is not None and not _qual_held.empty and "sector" in _qual_held.columns:
            _sec_mv: Dict[str, float] = {}
            for _p in positions:
                _ps = _p["symbol"]
                _pmv = float(_p.get("market_value", 0))
                _qr = _qual_held[_qual_held["symbol"] == _ps]
                _sec = str(_qr.iloc[0].get("sector", "Unknown")
                           ) if not _qr.empty else "Unknown"
                _sec_mv[_sec] = _sec_mv.get(_sec, 0.0) + _pmv
            _total_sec = sum(_sec_mv.values())
            if _total_sec > 0:
                for _sn, _sv in _sec_mv.items():
                    if _sn != "Unknown" and (_sv / _total_sec) > 0.25:
                        _sector_warnings.append(
                            f"**{_sn}** ({_sv / _total_sec:.0%} of MV > 25% cap)")

    # ── Senior Trader's Brief — build action items ────────────────────────────
    _brief_items: List[Tuple[str, str, str, str]] = []

    if regime_lbl == "Crash":
        _brief_items.append(("HALT", "#ff4444", "REGIME: CRASH DETECTED",
                             "Hard halt on ALL new long positions. Unified Conviction = 0 across all symbols. Consider full liquidation."))
    elif regime_lbl == "Panic":
        _brief_items.append(("REDUCE", "#ff4444", "REGIME: PANIC MODE",
                             "Reduce all position sizes immediately. No new long entries permitted. Capital preservation priority."))
    elif regime_lbl == "Bear":
        _brief_items.append(("DEFENSIVE", "#ff8800", "REGIME: BEAR",
                             "No new long entries. Hold only Grade-A conviction names with hard stops below ATR."))

    for _ar in action_rows:
        _rsi_b = _ar.get("rsi14")
        _atr_b = _ar.get("atr_alert", False)
        _ts_b = _ar.get("trend_status", "")
        _apct_b = _ar.get("atr_pct", 0)
        _gr_b = _ar.get("grade", "B")
        _cv_b = _ar.get("conviction", 0.5)

        if _rsi_b and _rsi_b > 70 and regime_lbl in ("Bull", "Euphoria", "Mania"):
            _brief_items.append(("TRIM", "#ffbb33", f"TRIM {_ar['sym']}",
                                 f"Overbought (RSI {_rsi_b:.0f}) in {regime_lbl} regime. Take partial profits on strength."))

        if _ts_b.startswith("Death") and regime_lbl in ("Bear", "Panic", "Crash"):
            _brief_items.append(("LIQUIDATE", "#ff4444", f"LIQUIDATE {_ar['sym']}",
                                 f"Price below SMA200 (Death Cross confirmed) in {regime_lbl} regime. Investment thesis invalidated."))

        if _atr_b:
            _brief_items.append(("ALERT", "#ff8800", f"VOLATILITY ALERT: {_ar['sym']}",
                                 f"ATR is {_apct_b:.0f}% above 30-day average. Widen stops or reduce position size immediately."))

        if _gr_b == "C" and _ar["mv"] > 0:
            _brief_items.append(("EXIT", "#ff4444", f"EXIT {_ar['sym']}",
                                 f"Unified Conviction = {_cv_b:.0%} (Grade C, regime-adjusted). Risk/reward below threshold."))

    for _sw in _sector_warnings:
        _brief_items.append(("CONCENTRATION", "#ff8800", "SECTOR OVERWEIGHT",
                             f"{_sw} — breaches 25% sector cap. Rebalance to reduce correlated exposure."))

    # Deduplicate brief items on (badge, title) — same symbol can't fire twice
    _seen_brief: set = set()
    _brief_items = [
        _bi for _bi in _brief_items
        if (_bi[0], _bi[2]) not in _seen_brief
        and not _seen_brief.add((_bi[0], _bi[2]))
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Render
    # ══════════════════════════════════════════════════════════════════════════

    _alpaca_err = st.session_state.pop("_alpaca_error", None)
    if _alpaca_err:
        st.warning(f"Alpaca: {_alpaca_err}")
    cb_level = risk.get("cb_level", "NONE")
    if cb_level and cb_level != "NONE":
        st.error(f"⚡ Circuit Breaker: **{cb_level}**")

    # ── Regime-change alert ────────────────────────────────────────────────────
    _prev_lbl = st.session_state.get("_last_known_regime", regime_lbl)
    if _prev_lbl != regime_lbl:
        _chg_clr = regime_color(regime_lbl)
        _chg_icon = "🔴" if regime_lbl in ("Crash", "Panic") else (
                    "🟠" if regime_lbl == "Bear" else (
                        "🟡" if regime_lbl in ("Mania", "Euphoria") else "🟢"))
        # Toast fires once per session-state transition
        st.toast(
            f"{_chg_icon} Regime changed: **{_prev_lbl} → {regime_lbl}**  "
            f"(conf {conf:.0%})",
            icon="⚠️",
        )
        # Persistent banner stays visible until next page refresh clears it
        st.markdown(
            f'<div style="padding:10px 16px;margin-bottom:10px;border-radius:6px;'
            f'background:{_chg_clr}18;border:1px solid {_chg_clr}55;'
            f'font-size:0.85rem;font-weight:700;color:{_chg_clr}">'
            f'{_chg_icon}&nbsp; REGIME CHANGE DETECTED &nbsp;·&nbsp; '
            f'{_prev_lbl} <span style="color:#888">→</span> {regime_lbl} &nbsp;·&nbsp; '
            f'Confidence {conf:.0%}'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.session_state["_last_known_regime"] = regime_lbl

    # ── Section index (quick reference) ───────────────────────────────────────
    _DM_SECTIONS = [
        "01 · Market", "02 · Brief", "03 · Regime", "04 · Gauge",
        "05 · Positions", "06 · Correlation", "07 · Conviction", "08 · Status",
    ]
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:4px;padding:6px 0 10px;'>"
        + "".join(
            f"<span style='font-size:0.60rem;font-family:monospace;color:#AAAAAA;"
            f"background:#0a0a0a;border:1px solid #1a1a1a;border-radius:2px;"
            f"padding:2px 7px;white-space:nowrap'>{s}</span>"
            for s in _DM_SECTIONS
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── 1. GLOBAL RISK RIBBON ─────────────────────────────────────────────────
    _section_bar("01 · Market Status")
    _pnl_clr = "#00c851" if daily_pnl >= 0 else "#ff4444"
    _mode_lbl = "PAPER" if portf.get("paper_mode", True) else "LIVE"
    _mode_clr = "#ffbb33" if portf.get("paper_mode", True) else "#ff4444"
    _dd_frac = min(1.0, abs(daily_pnl_pct) / 0.03)
    _dd_clr = "#ff4444" if _dd_frac > 0.80 else "#ffbb33" if _dd_frac > 0.50 else "#00c851"
    _beta_clr = ("#ff4444" if _portf_beta is not None and _portf_beta > 1.5
                 else "#ffbb33" if _portf_beta is not None and _portf_beta > 1.2
                 else "#00c851")
    _beta_str = f"β {_portf_beta:.2f}" if _portf_beta is not None else "β —"
    _beta_sub = ("high sensitivity" if _portf_beta and _portf_beta > 1.3
                 else "defensive" if _portf_beta and _portf_beta < 0.8 else "vs S&P 500")

    st.markdown(
        f'<div class="risk-ribbon">'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">HMM Regime</div>'
        f'<div class="ribbon-value" style="color:{rc}">{regime_lbl}</div>'
        f'<div class="ribbon-sub">conf {conf:.0%} · {"⚠ uncertain" if regime["is_uncertain"] else "stable"}</div>'
        f'</div>'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">Portfolio Beta</div>'
        f'<div class="ribbon-value" style="color:{_beta_clr}">{_beta_str}</div>'
        f'<div class="ribbon-sub">{_beta_sub}</div>'
        f'</div>'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">Daily Drawdown</div>'
        f'<div class="ribbon-value" style="color:{_dd_clr}">{daily_pnl_pct:+.2%}</div>'
        f'<div class="ribbon-sub">limit −3.00% · used {_dd_frac:.0%}</div>'
        f'</div>'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">Invested Value</div>'
        f'<div class="ribbon-value">{fmt_usd(total_mv)}</div>'
        f'<div class="ribbon-sub">of {fmt_usd(equity)} equity · {alloc_frac:.0%} deployed</div>'
        f'</div>'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">Daily P&L</div>'
        f'<div class="ribbon-value" style="color:{_pnl_clr}">{fmt_usd(daily_pnl)}</div>'
        f'<div class="ribbon-sub">unreal {fmt_usd(total_unreal)} · {len(positions)} pos</div>'
        f'</div>'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">Portfolio Risk</div>'
        f'<div class="ribbon-value" style="color:{risk_color}">{risk_total:.0f}'
        f'<span style="font-size:0.82rem;font-weight:500"> {risk_label}</span></div>'
        f'<div class="ribbon-sub">regime · conc · intel</div>'
        f'</div>'

        f'<div class="ribbon-block">'
        f'<div class="ribbon-label">Mode</div>'
        f'<div class="ribbon-value" style="color:{_mode_clr};font-size:0.95rem">{_mode_lbl}</div>'
        f'<div class="ribbon-sub" style="color:{"#00c851" if alpaca_ok else "#ffbb33"}">'
        f'{"● live api" if alpaca_ok else "● log data"}</div>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )

    _section_bar("02 · Senior Trader's Brief",
                 f"{len(_brief_items)} action(s)")
    # ── 2. SENIOR TRADER'S BRIEF ──────────────────────────────────────────────
    _brief_title = (f"⚡ Senior Trader's Brief — {len(_brief_items)} Action(s) Required"
                    if _brief_items else "Senior Trader's Brief — No Urgent Actions")
    with st.expander(_brief_title, expanded=bool(_brief_items)):
        if _brief_items:
            _BRIEF_STYLES = {
                "HALT":          ("#ff4444", "#ff000014"),
                "REDUCE":        ("#ff4444", "#ff000014"),
                "LIQUIDATE":     ("#ff4444", "#ff000012"),
                "EXIT":          ("#ff4444", "#ff000010"),
                "TRIM":          ("#ffbb33", "#ffbb3310"),
                "ALERT":         ("#ff8800", "#ff880010"),
                "DEFENSIVE":     ("#ff8800", "#ff88000d"),
                "CONCENTRATION": ("#ff8800", "#ff88000d"),
            }
            for _badge, _bclr, _btitle, _bdesc in _brief_items:
                _bc, _bbg = _BRIEF_STYLES.get(_badge, ("#9e9e9e", "#9e9e9e0d"))
                st.markdown(
                    f'<div class="brief-item" style="background:{_bbg};border-color:{_bc}55">'
                    f'<span class="brief-badge" style="background:{_bc}20;color:{_bc}">{_badge}</span>'
                    f'<div><div style="font-size:0.80rem;font-weight:700;color:#d8d8d8">{_btitle}</div>'
                    f'<div style="font-size:0.70rem;color:#AAAAAA;margin-top:1px">{_bdesc}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#AAAAAA;padding:6px 0">'
                '✓ No urgent actions. Portfolio conviction and risk are within parameters.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Regime-aware sector rotation hint ─────────────────────────────────
        _SECTOR_ROTATION_HINTS: Dict[str, Tuple[str, str]] = {
            "Crash":    ("DEFENSIVE",  "Move to CASH + Short-Duration Treasuries (SHY/BIL). Exit all risk assets immediately."),
            "Panic":    ("FLIGHT TO QUALITY", "Rotate → GLD, TLT, XLP, XLU. Avoid cyclicals and high-beta tech."),
            "Bear":     ("DEFENSIVE CYCLICALS", "Favour XLE (Energy), XLV (Healthcare). Reduce tech (XLK). Add GLD hedge."),
            "Neutral":  ("BALANCED",   "Mix cyclicals (XLF, XLI) with defensives (XLV, XLP). Keep 10–15% cash reserve."),
            "Bull":     ("GROWTH",     "Broad equities. Overweight XLK (Tech) + XLI (Industrials). Reduce bonds."),
            "Euphoria": ("ROTATE OUT", "TRIM high-beta tech. Build Energy (XLE), Gold (GLD), Defense (RTX/LMT) hedge."),
            "Mania":    ("RISK REDUCTION", "Reduce equity beta. ADD GLD + RTX/LMT. Watch VIX for blow-off reversal."),
        }
        _sr_badge, _sr_hint = _SECTOR_ROTATION_HINTS.get(
            regime_lbl, ("NEUTRAL",
                         "Regime unknown — maintain balanced allocation.")
        )
        _sr_color = ("#ff4444" if regime_lbl in ("Crash", "Panic") else
                     "#ff8800" if regime_lbl in ("Bear", "Euphoria", "Mania") else
                     "#00c851" if regime_lbl == "Bull" else "#9e9e9e")
        st.markdown(
            f'<div style="margin-top:6px;padding:7px 10px;border-radius:4px;'
            f'background:{_sr_color}0d;border-left:3px solid {_sr_color}66;">'
            f'<span style="font-size:0.60rem;font-weight:900;font-family:monospace;'
            f'color:{_sr_color};letter-spacing:0.10em;text-transform:uppercase">'
            f'SECTOR ROTATION · {regime_lbl.upper()} REGIME &nbsp;→&nbsp; {_sr_badge}</span>'
            f'<div style="font-size:0.70rem;color:#AAAAAA;margin-top:2px">{_sr_hint}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Refresh row ───────────────────────────────────────────────────────────
    _rb1, _rb2, _ = st.columns([1, 1, 4])
    with _rb1:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with _rb2:
        st.markdown(
            f'<div style="padding:5px 0;font-size:0.72rem;font-family:monospace">'
            f'<span class="{"dot-ok" if alpaca_ok else "dot-warn"}">●</span> '
            f'{"live" if alpaca_ok else "log"}'
            f'&nbsp;&nbsp;<span style="color:#AAAAAA">{datetime.now().strftime("%H:%M:%S")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    _s3_hdr_l, _s3_hdr_r = st.columns([3, 1])
    with _s3_hdr_l:
        _section_bar("03 · Regime Analysis", f"conf {conf:.0%}")
    with _s3_hdr_r:
        if st.button("Recalculate Regime", use_container_width=True, key="recalc_03"):
            with st.spinner("Running HMM on SPY …"):
                _live = _run_live_regime()
            if "_error" in _live:
                _hmm_err = _live["_error"]
                if "No module named" in _hmm_err:
                    pass  # Regime unavailable on cloud, using fallback
                else:
                    st.error(f"HMM failed: {_hmm_err}")
            else:
                regime = _live
                regime_lbl = regime["label"]
                rc = regime_color(regime_lbl)
                conf = regime["confidence"]
                _get_unified_conviction.clear()
                st.rerun()
    # ── 3. REGIME PROBS + RISK BREAKDOWN ─────────────────────────────────────
    rp_l, rp_r = st.columns([1, 1])

    with rp_l:
        probs: Dict[str, float] = regime.get("probs", {})
        if probs:
            for lbl_p, val_p in sorted(probs.items(), key=lambda x: -x[1]):
                c_p = regime_color(lbl_p)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                    f'<span style="width:58px;font-size:0.78rem;font-weight:700;color:{c_p}">{lbl_p}</span>'
                    f'<div style="flex:1;background:#141414;border-radius:3px;height:7px">'
                    f'<div style="width:{val_p*100:.1f}%;background:{c_p};height:7px;border-radius:3px"></div></div>'
                    f'<span style="font-size:0.78rem;font-weight:700;color:{c_p};width:38px;text-align:right;font-family:monospace">{val_p:.0%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No regime probability data yet.")

        # Explain what the probabilities mean (and their limits)
        _n_st = regime.get("n_states")
        _stk = regime.get("stability_bars", 0)
        _src = "HMM · live" if "_live_calc_ts" in regime else "GMM · log"
        st.markdown(
            f'<div style="margin-top:8px;font-size:0.60rem;color:#AAAAAA;'
            f'font-family:monospace;line-height:1.6">'
            + (f'{_n_st}-state model &nbsp;·&nbsp; ' if _n_st else '')
            + f'streak {_stk}d &nbsp;·&nbsp; source: {_src}<br>'
            f'Probs floored at 2% — HMM forward posteriors, not calibrated '
            f'probabilities. Streak &gt; 5 days = higher reliability.</div>',
            unsafe_allow_html=True,
        )

    with rp_r:
        for comp_name, comp_val in risk_breakdown.items():
            comp_frac = comp_val / 40.0
            comp_clr = "#ff4444" if comp_frac > 0.7 else "#ffbb33" if comp_frac > 0.4 else "#00c851"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                f'<span style="width:100px;font-size:0.76rem;color:#888">{comp_name}</span>'
                f'<div style="flex:1;background:#141414;border-radius:3px;height:7px">'
                f'<div style="width:{min(100, comp_frac*100):.0f}%;background:{comp_clr};height:7px;border-radius:3px"></div></div>'
                f'<span style="font-size:0.78rem;font-weight:700;color:{comp_clr};width:28px;text-align:right;font-family:monospace">{comp_val:.0f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        _regime_advice = {
            "Bull": "Stay invested — regime supports longs",
            "Euphoria": "Consider trimming top performers",
            "Mania": "Reduce risk — market overheated",
            "Neutral": "Hold current allocations",
            "Bear": "Reduce exposure — defensive mode",
            "Panic": "Cash or hedges — high risk",
            "Crash": "Capital preservation priority",
            "Unknown": "Maintain disciplined stops",
        }
        st.markdown(
            f'<div style="margin-top:6px;padding:5px 9px;border-radius:4px;'
            f'background:{rc}0d;border-left:3px solid {rc}30;font-size:0.76rem;color:#999">'
            f'{_regime_advice.get(regime_lbl, "Monitor closely")}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    _section_bar("04 · Regime Gauge & Circuit Breakers")
    # ── 4. REGIME GAUGE + CIRCUIT BREAKER BOARD ──────────────────────────────

    # Live recalculate button — runs HMM on SPY on demand
    _rc_btn_col, _rc_ts_col, _ = st.columns([1, 2, 3])
    with _rc_btn_col:
        if st.button("Recalculate Regime", use_container_width=True, key="recalc_04"):
            with st.spinner("Running HMM on SPY …"):
                _live = _run_live_regime()
            if "_error" in _live:
                _hmm_err = _live["_error"]
                if "No module named" in _hmm_err:
                    pass  # Regime unavailable on cloud, using fallback
                else:
                    st.error(f"HMM failed: {_hmm_err}")
            else:
                regime = _live
                regime_lbl = regime["label"]
                rc = regime_color(regime_lbl)
                conf = regime["confidence"]
                _get_unified_conviction.clear()
                st.rerun()
    with _rc_ts_col:
        _lr = st.session_state.get("live_regime_state", {})
        if "_live_calc_ts" in _lr:
            _lr_ts = datetime.fromisoformat(
                _lr["_live_calc_ts"]).strftime("%H:%M:%S UTC")
            st.markdown(
                f'<div style="padding:5px 0;font-size:0.72rem;font-family:monospace;color:#AAAAAA">'
                f'● live calc @ {_lr_ts}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding:5px 0;font-size:0.72rem;font-family:monospace;color:#AAAAAA">'
                '● reading from log — hit Recalculate for live result</div>',
                unsafe_allow_html=True,
            )

    gauge_col, cb_col = st.columns([1, 2])

    with gauge_col:
        _REGIME_SCALE = {"Crash": 0, "Panic": 1, "Bear": 2,
                         "Neutral": 3, "Bull": 4, "Euphoria": 5, "Mania": 6}
        _gauge_val = _REGIME_SCALE.get(regime_lbl, 3)
        _fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=_gauge_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{regime_lbl}</b>",
                   "font": {"size": 16, "color": rc}},
            number={"suffix": "", "font": {"color": "rgba(0,0,0,0)"}},
            gauge={
                "axis": {
                    "range": [0, 6],
                    "tickvals": [0, 1, 2, 3, 4, 5, 6],
                    "ticktext": ["Crash", "Panic", "Bear", "Neutral", "Bull", "Euph", "Mania"],
                    "tickfont": {"size": 8, "color": "#AAAAAA"},
                    "tickcolor": "#333",
                },
                "bar": {"color": rc, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 1.5],  "color": "#330000"},
                    {"range": [1.5, 2.5], "color": "#1a0800"},
                    {"range": [2.5, 3.5], "color": "#0d0d0d"},
                    {"range": [3.5, 4.5], "color": "#001a00"},
                    {"range": [4.5, 6.0], "color": "#002200"},
                ],
                "threshold": {"line": {"color": rc, "width": 3}, "thickness": 0.80, "value": _gauge_val},
            },
        ))
        _fig_gauge.update_layout(
            height=200, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
        )
        st.plotly_chart(_fig_gauge, use_container_width=True)
        _conf_clr = "#00c851" if conf >= 0.65 else "#ffbb33" if conf >= 0.45 else "#ff4444"
        _n_states = regime.get("n_states", "?")
        _streak = regime.get("stability_bars", 0)
        _flicker = regime.get("flicker_count",  0)
        st.markdown(
            f'<div style="font-size:0.66rem;color:#AAAAAA;margin:-8px 0 3px 0">'
            f'HMM Confidence &nbsp;·&nbsp; '
            f'<span style="color:#AAAAAA">{_n_states}-state model</span></div>'
            f'<div style="background:#141414;border-radius:3px;height:5px">'
            f'<div style="width:{conf*100:.0f}%;background:{_conf_clr};height:5px;border-radius:3px"></div></div>'
            f'<div style="font-size:0.68rem;font-weight:700;color:{_conf_clr};margin-top:2px;font-family:monospace">{conf:.0%}</div>'
            f'<div style="font-size:0.60rem;color:#AAAAAA;margin-top:4px;font-family:monospace">'
            f'streak {_streak}d &nbsp;·&nbsp; flicker {_flicker}/20</div>',
            unsafe_allow_html=True,
        )

    with cb_col:
        _cb_items: List[Tuple[str, bool, str]] = [
            ("Longs Halted",             regime_lbl in ("Crash", "Panic"),
             f"{regime_lbl} regime — all new long positions halted"),
            ("Mania Caution  (0.7× sz)", regime_lbl in ("Mania", "Euphoria"),
             f"{regime_lbl} regime — blow-off top protection: long size ×0.7"),
            ("Low HMM Confidence",        conf < 0.45 or _flicker > 4,
             f"Two triggers — (1) forward posterior {conf:.0%} < 45%, or "
             f"(2) regime flicker {_flicker}/20 > 4 transitions. "
             f"Either alone halves position sizes."),
            ("Drawdown Brake",            daily_pnl_pct < -0.07,
             f"Daily P&L {daily_pnl_pct:.1%} — size halved below −7%"),
            ("Volatility Sizing Active",  _any_atr_alert,
             "One or more holdings: ATR >20% above monthly baseline"),
            ("Sector Concentration",      bool(_sector_warnings),
             _sector_warnings[0] if _sector_warnings else "Engine caps 3 longs per GICS sector"),
        ]
        for _cb_label, _cb_active, _cb_desc in _cb_items:
            st.markdown(
                _alert_row_html(_cb_label, _cb_active,
                                _cb_desc, warn_color="#ff4444"),
                unsafe_allow_html=True,
            )

    with st.expander("Trigger Trace -- raw Minsky values", expanded=False):
        try:
            from backend.utils.triggers import compute_minsky_conditions, minsky_ui_line as _minsky_ui
            _g = st.session_state.get("garch_result") or {}
            _garch_persist = float(_g.get("persistence", 0.0))
            _cape_pct_val = float(st.session_state.get("cape_percentile", 0.0))
            _yield_bps_val = float(
                st.session_state.get("yield_spread_bps", 50.0))
            _trace = compute_minsky_conditions(
                _garch_persist, _cape_pct_val, _yield_bps_val)
            st.code(_minsky_ui(_trace))
            st.json(_trace)
        except Exception as _te:
            _dm_logger.warning("Minsky trace expander error: %s", _te)
            st.caption(
                "Minsky trace unavailable -- GARCH/CAPE/yield inputs not in session state.")

    st.divider()

    _section_bar("05 · Positions — Execution Table", f"{len(positions)} open")
    # ── 5. EXECUTION TABLE ────────────────────────────────────────────────────

    if _tech_df_m is not None and not _tech_df_m.empty and "atr_alert" in _tech_df_m.columns:
        _vol_syms = _tech_df_m[_tech_df_m["atr_alert"]
                               == True]["symbol"].tolist()
        if _vol_syms:
            _atr_det = []
            for _vs in _vol_syms:
                _vr = _tech_df_m[_tech_df_m["symbol"] == _vs]
                if not _vr.empty:
                    _atr_det.append(
                        f"**{_vs}** (+{_vr.iloc[0]['atr_pct']:.0f}%)")
            st.warning(
                "⚡ Volatility Alert — ATR >20% above baseline: " + ", ".join(_atr_det))

    if action_rows:
        _ACTION_EMOJI = {
            "SELL": "⚠️ ", "TRIM": "⚖️ ",
            "BUY MORE": "🚀 ", "ADD": "➕ ", "HOLD": "✅ ",
        }
        _pos_table = pd.DataFrame([{
            "Symbol":         ar["sym"] + (" ⚡" if ar.get("atr_alert") else ""),
            "Quality":        f'{ar["grade"]} · {ar["conviction"]:.0%}',
            "Execution":      _ACTION_EMOJI.get(ar["action"], "") + ar["action"],
            "Momentum (RSI)": float(ar.get("rsi14") or 50.0),
            "Trend":          ar.get("trend_status", "—"),
            "Risk $":         ar.get("risk_usd"),
            "ATR Stop":       ar.get("atr_stop"),
            "Entry":          float(ar.get("cost", 0.0)),
            "Current":        float(ar.get("price", 0.0)),
            "P&L $":          float(ar.get("pnl", 0.0)),
            "P&L %":          float(ar.get("upct", 0.0)) * 100.0,
            "Mkt Value":      float(ar.get("mv", 0.0)),
        } for ar in action_rows])

        st.dataframe(
            _pos_table,
            column_config={
                "Symbol":         st.column_config.TextColumn("Symbol", width="small"),
                "Quality":        st.column_config.TextColumn("Quality (Buffett)", width="small"),
                "Execution":      st.column_config.TextColumn("Execution", width="small"),
                "Momentum (RSI)": st.column_config.ProgressColumn(
                    "Momentum (RSI)", min_value=0, max_value=100, format="%.0f",
                ),
                "Trend":          st.column_config.TextColumn("Trend"),
                "Risk $":         st.column_config.NumberColumn("Risk $", format="$%.0f"),
                "ATR Stop":       st.column_config.NumberColumn("ATR Stop", format="$%.2f"),
                "Entry":          st.column_config.NumberColumn("Entry", format="$%.2f"),
                "Current":        st.column_config.NumberColumn("Current", format="$%.2f"),
                "P&L $":          st.column_config.NumberColumn("P&L $", format="$%.2f"),
                "P&L %":          st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                "Mkt Value":      st.column_config.NumberColumn("Mkt Value", format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True,
        )

        action_counts: Dict[str, int] = {}
        for ar in action_rows:
            action_counts[ar["action"]] = action_counts.get(
                ar["action"], 0) + 1
        _clr_map = {"SELL": "#ff4444", "TRIM": "#ffbb33",
                    "BUY MORE": "#00c851", "ADD": "#7cb342", "HOLD": "#9e9e9e"}
        summary_html = " &nbsp;·&nbsp; ".join(
            f'<span style="color:{_clr_map.get(k, "#555")};font-weight:700;font-family:monospace">{v}× {k}</span>'
            for k, v in sorted(action_counts.items(), key=lambda x: _ACTION_URGENCY.get(x[0], 99))
        )
        st.markdown(f'<div style="font-size:0.70rem;color:#AAAAAA;margin:5px 0 0 10px">{summary_html}</div>',
                    unsafe_allow_html=True)
    else:
        st.info("No open positions — connect Alpaca API or start the trading engine.")

    # ── 6. CORRELATION GUARD ─────────────────────────────────────────────────
    if len(_held_syms_m) >= 2:
        _section_bar("06 · Correlation Guard", f"{len(_held_syms_m)} symbols")
        # Use all held symbols — cap heatmap display at 15 for readability
        _corr_syms = tuple(_held_syms_m)
        _corr_df = _get_correlation_matrix(_corr_syms)

        if _corr_df is not None and not _corr_df.empty:
            _cls = _corr_df.columns.tolist()

            # ── Build pair risk table (all symbols, all pairs) ────────────────
            _pair_rows: List[Dict] = []
            for _i in range(len(_cls)):
                for _j in range(_i + 1, len(_cls)):
                    _cv_val = float(_corr_df.iloc[_i, _j])
                    if _cv_val > 0.50:   # only show meaningful correlations
                        if _cv_val > 0.85:
                            _risk_tier, _tier_clr = "CRITICAL", "#ff4444"
                        elif _cv_val > 0.70:
                            _risk_tier, _tier_clr = "HIGH", "#ff8800"
                        else:
                            _risk_tier, _tier_clr = "MODERATE", "#ffbb33"
                        _pair_rows.append({
                            "A": _cls[_i], "B": _cls[_j],
                            "corr": _cv_val, "tier": _risk_tier, "clr": _tier_clr,
                        })
            _pair_rows.sort(key=lambda x: -x["corr"])

            # ── Portfolio diversification score (avg off-diagonal abs corr) ──
            import numpy as _np_cg
            _corr_vals = _corr_df.values
            _mask = ~_np_cg.eye(len(_cls), dtype=bool)
            _avg_corr = float(_np_cg.abs(
                _corr_vals[_mask]).mean()) if _mask.any() else 0.0
            # 1.0 = perfectly diversified
            _div_score = round(1.0 - _avg_corr, 2)
            _div_clr = "#00c851" if _div_score >= 0.60 else "#ffbb33" if _div_score >= 0.45 else "#ff4444"
            _n_critical = sum(1 for r in _pair_rows if r["tier"] == "CRITICAL")
            _n_high = sum(1 for r in _pair_rows if r["tier"] == "HIGH")

            # ── Layout: metrics | pair table | heatmap ────────────────────────
            _cg_kpi, _cg_pairs, _cg_heat = st.columns([1, 2, 3])

            with _cg_kpi:
                st.markdown(
                    f'<div style="padding:10px 0">'
                    f'<div style="font-size:0.65rem;color:#AAAAAA;font-family:monospace;'
                    f'text-transform:uppercase;letter-spacing:0.1em">Diversification</div>'
                    f'<div style="font-size:2rem;font-weight:700;color:{_div_clr};'
                    f'font-family:monospace;line-height:1.1">{_div_score:.2f}</div>'
                    f'<div style="font-size:0.62rem;color:#AAAAAA;margin-top:2px">1.0 = uncorrelated</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="margin-top:8px;font-size:0.70rem;font-family:monospace">'
                    f'<span style="color:#ff4444">{_n_critical} Critical</span>'
                    f'  <span style="color:#ff8800">{_n_high} High</span>'
                    f'  <span style="color:#AAAAAA">{len(_pair_rows)-_n_critical-_n_high} Moderate</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if _n_critical > 0 or _n_high > 0:
                    st.markdown(
                        '<div style="margin-top:10px;font-size:0.64rem;color:#AAAAAA;'
                        'border-left:2px solid #ff880050;padding-left:8px;line-height:1.5">'
                        'High-corr pairs move together in a crash. Consider reducing '
                        'one position from each flagged pair.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="margin-top:10px;font-size:0.64rem;color:#00c85180;'
                        'border-left:2px solid #00c85130;padding-left:8px">'
                        'Portfolio is well diversified — no high-correlation pairs.</div>',
                        unsafe_allow_html=True,
                    )

            with _cg_pairs:
                st.caption(
                    f"Correlated pairs (>{len(_pair_rows)} total, sorted by risk)")
                if _pair_rows:
                    for _pr in _pair_rows[:12]:   # show top 12 pairs
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:6px;'
                            f'margin:2px 0;font-size:0.72rem;font-family:monospace">'
                            f'<span style="color:{_pr["clr"]};font-weight:700;width:58px">'
                            f'{_pr["tier"]}</span>'
                            f'<span style="color:#888">{_pr["A"]}</span>'
                            f'<span style="color:#AAAAAA">↔</span>'
                            f'<span style="color:#888">{_pr["B"]}</span>'
                            f'<span style="color:{_pr["clr"]};margin-left:auto;font-weight:700">'
                            f'{_pr["corr"]:.2f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if len(_pair_rows) > 12:
                        st.caption(
                            f"… and {len(_pair_rows)-12} more moderate pairs")
                else:
                    st.markdown(
                        '<div style="font-size:0.72rem;color:#00c851;padding:8px 0">'
                        'All pairs < 0.50 — excellent diversification.</div>',
                        unsafe_allow_html=True,
                    )

            with _cg_heat:
                # Show all symbols in heatmap; suppress text labels if >10 for legibility
                _show_text = len(_cls) <= 12
                _font_sz = max(7, 11 - max(0, len(_cls) - 8))
                _ht_px = max(280, min(480, len(_cls) * 26))
                _z_vals = _corr_df.values.tolist()
                _txt_z = [[f"{v:.2f}" for v in row] for row in _corr_df.values]
                _fig_corr = go.Figure(go.Heatmap(
                    z=_z_vals, x=_cls, y=_cls,
                    text=_txt_z if _show_text else None,
                    texttemplate="%{text}" if _show_text else None,
                    textfont={"size": _font_sz, "color": "#ccc",
                              "family": "JetBrains Mono, Consolas, monospace"},
                    colorscale=[
                        [0.00, "#0d2137"],   # negative: blue
                        [0.50, "#0a0a0a"],   # zero: black
                        [0.75, "#2e1000"],   # 0.50 corr: dark orange
                        [0.85, "#7a1a00"],   # 0.70 corr: deep red-orange
                        [1.00, "#ff4444"],   # 1.00: red
                    ],
                    zmin=-1, zmax=1,
                    showscale=True,
                    hovertemplate="<b>%{y} ↔ %{x}</b><br>ρ = %{z:.2f}<extra></extra>",
                    colorbar=dict(tickformat=".1f", tickfont=dict(size=9, color="#AAAAAA"),
                                  thickness=8, len=0.85),
                ))
                _fig_corr.update_layout(
                    height=_ht_px, margin=dict(t=4, b=4, l=4, r=36),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    xaxis=dict(tickfont=dict(size=9, color="#AAAAAA")),
                    yaxis=dict(tickfont=dict(size=9, color="#AAAAAA")),
                )
                st.plotly_chart(_fig_corr, use_container_width=True)
                if not _show_text:
                    st.caption(
                        "Hover cells for exact values · red = highly correlated")

    # ── AI Concentration Check ────────────────────────────────────────────────
    if positions:
        _ai_chk = _ai_concentration_check(positions)
        _ai_pct = _ai_chk["pct"]
        _ai_clr = _ai_chk["color"]
        _ai_lbl = _ai_chk["label"]
        _ai_held = _ai_chk["held"]
        if _ai_pct > 0.0:
            if _ai_pct > 0.40:
                _ai_desc = (
                    f"{_ai_pct:.0%} of portfolio in AI mega-caps "
                    f"({', '.join(_ai_held)}). Consider trimming or hedging "
                    f"via SQQQ / put spreads on QQQ."
                )
            elif _ai_pct > 0.25:
                _ai_desc = (
                    f"{_ai_pct:.0%} in AI/tech mega-caps "
                    f"({', '.join(_ai_held)}). Monitor for AI sentiment reversal — "
                    f"single-factor risk elevated."
                )
            else:
                _ai_desc = (
                    f"{_ai_pct:.0%} in AI/tech mega-caps "
                    f"({', '.join(_ai_held) if _ai_held else 'none'}). "
                    f"Concentration within acceptable range."
                )
            _ai_active = _ai_pct > 0.25
            st.markdown(
                _alert_row_html(
                    f"AI Tech Concentration — {_ai_lbl}",
                    _ai_active,
                    _ai_desc,
                    warn_color=_ai_clr,
                ),
                unsafe_allow_html=True,
            )

    # ── 7. CONVICTION RADAR ───────────────────────────────────────────────────
    _section_bar("07 · Conviction Scoreboard")
    if _scores_df_m is not None and not _scores_df_m.empty and positions:
        # Filter to only held symbols with intel data
        _held_with_scores = [
            p["symbol"] for p in positions
            if p["symbol"] in _scores_df_m["symbol"].values
        ]

        if _held_with_scores:
            # Build per-symbol score rows
            _cv_rows: List[Dict] = []
            for _csym in _held_with_scores:
                _cr = _scores_df_m[_scores_df_m["symbol"] == _csym].iloc[0]
                _comp = float(_cr.get("composite", 0.5))
                _cv_rows.append({
                    "sym":   _csym,
                    "comp":  _comp,
                    "inst":  float(_cr.get("institutional_score", 0.5)),
                    "ins":   float(_cr.get("insider_score",       0.5)),
                    "sent":  float(_cr.get("sentiment_score",     0.5)),
                    "news":  float(_cr.get("news_score",          0.5)),
                    "macro": float(_cr.get("macro_score",         0.5)),
                    "dir":   str(_cr.get("direction",  "neutral")),
                    "lbl":   str(_cr.get("label",      "Neutral")),
                })
            _cv_rows.sort(key=lambda r: -r["comp"])

            # Portfolio average conviction
            _avg_conv = sum(r["comp"] for r in _cv_rows) / len(_cv_rows)
            _avg_clr = score_color(_avg_conv)
            _n_long = sum(1 for r in _cv_rows if r["dir"] == "long")
            _n_short = sum(1 for r in _cv_rows if r["dir"] == "short")
            _n_neut = len(_cv_rows) - _n_long - _n_short

            # KPI strip
            _cv_k1, _cv_k2, _cv_k3, _cv_k4 = st.columns(4)
            _cv_k1.markdown(
                f'<div style="font-size:0.64rem;color:#AAAAAA;font-family:monospace">Portfolio Intel</div>'
                f'<div style="font-size:1.6rem;font-weight:700;color:{_avg_clr};font-family:monospace">'
                f'{_avg_conv:.0%}</div>'
                f'<div style="font-size:0.60rem;color:#AAAAAA">avg conviction</div>',
                unsafe_allow_html=True,
            )
            _cv_k2.markdown(
                f'<div style="font-size:0.64rem;color:#AAAAAA;font-family:monospace">Longs</div>'
                f'<div style="font-size:1.6rem;font-weight:700;color:#00c851;font-family:monospace">'
                f'{_n_long}</div>'
                f'<div style="font-size:0.60rem;color:#AAAAAA">of {len(_cv_rows)} scored</div>',
                unsafe_allow_html=True,
            )
            _cv_k3.markdown(
                f'<div style="font-size:0.64rem;color:#AAAAAA;font-family:monospace">Neutral</div>'
                f'<div style="font-size:1.6rem;font-weight:700;color:#9e9e9e;font-family:monospace">'
                f'{_n_neut}</div>'
                f'<div style="font-size:0.60rem;color:#AAAAAA">hold</div>',
                unsafe_allow_html=True,
            )
            _cv_k4.markdown(
                f'<div style="font-size:0.64rem;color:#AAAAAA;font-family:monospace">Shorts</div>'
                f'<div style="font-size:1.6rem;font-weight:700;color:#ff4444;font-family:monospace">'
                f'{_n_short}</div>'
                f'<div style="font-size:0.60rem;color:#AAAAAA">reduce / exit</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='margin:6px 0'></div>",
                        unsafe_allow_html=True)

            # ── Conviction bar chart — all held symbols ───────────────────────
            _src_labels = ["Institutional", "Insider", "Sentiment", "News"]
            _src_keys = ["inst",          "ins",     "sent",      "news"]
            _src_clrs = ["#5c85d6",       "#00c851", "#ffbb33",   "#b39ddb"]

            _fig_bars = go.Figure()
            _sym_labels = [r["sym"] for r in _cv_rows]

            # Source bars (grouped)
            for _sl, _sk, _sc in zip(_src_labels, _src_keys, _src_clrs):
                _fig_bars.add_trace(go.Bar(
                    name=_sl,
                    x=_sym_labels,
                    y=[r[_sk] for r in _cv_rows],
                    marker_color=_sc,
                    opacity=0.80,
                    hovertemplate=f"<b>%{{x}}</b><br>{_sl}: %{{y:.0%}}<extra></extra>",
                ))

            # Composite score as scatter overlay
            _fig_bars.add_trace(go.Scatter(
                name="Composite",
                x=_sym_labels,
                y=[r["comp"] for r in _cv_rows],
                mode="markers+lines",
                marker=dict(size=9, color=[score_color(r["comp"]) for r in _cv_rows],
                            line=dict(width=1.5, color="#000")),
                line=dict(color="rgba(255,255,255,0.19)", width=1, dash="dot"),
                hovertemplate="<b>%{x}</b><br>Composite: %{y:.0%}<extra></extra>",
            ))

            # 0.58 long / 0.42 short threshold lines
            _fig_bars.add_hline(y=0.58, line=dict(color="rgba(0,200,81,0.31)", width=1, dash="dot"),
                                annotation_text="Long ▸", annotation_font_size=9,
                                annotation_font_color="rgba(0,200,81,0.44)")
            _fig_bars.add_hline(y=0.42, line=dict(color="rgba(255,68,68,0.31)", width=1, dash="dot"),
                                annotation_text="Short ▸", annotation_font_size=9,
                                annotation_font_color="rgba(255,68,68,0.44)")

            _fig_bars.update_layout(**_PLOTLY_LAYOUT)
            _fig_bars.update_layout(
                barmode="group",
                height=320,
                margin=dict(t=10, b=40, l=10, r=10),
                xaxis=dict(tickfont=dict(size=9, color="#AAAAAA"),
                           gridcolor="#1a1a1a"),
                yaxis=dict(tickformat=".0%", range=[0, 1.05],
                           gridcolor="#1a1a1a", tickfont=dict(size=9, color="#AAAAAA")),
            )
            st.plotly_chart(_fig_bars, use_container_width=True)

            # ── Compact score table ───────────────────────────────────────────
            with st.expander("Full score breakdown", expanded=False):
                _tbl_rows = []
                for _r in _cv_rows:
                    _dir_clr = "#00c851" if _r["dir"] == "long" else "#ff4444" if _r["dir"] == "short" else "#9e9e9e"
                    _tbl_rows.append({
                        "Symbol":   _r["sym"],
                        "Score":    f"{_r['comp']:.0%}",
                        "Label":    _r["lbl"],
                        "Dir":      _r["dir"].upper(),
                        "Inst":     f"{_r['inst']:.0%}",
                        "Insider":  f"{_r['ins']:.0%}",
                        "Sent":     f"{_r['sent']:.0%}",
                        "News":     f"{_r['news']:.0%}",
                    })
                st.dataframe(
                    pd.DataFrame(_tbl_rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Score":  st.column_config.TextColumn("Score",  width="small"),
                        "Label":  st.column_config.TextColumn("Label"),
                        "Dir":    st.column_config.TextColumn("Dir",    width="small"),
                    },
                )
        else:
            st.info("No intel scores for your held positions yet — click **Refresh Intelligence** in the Market Intel tab to score your portfolio.")
    else:
        st.info(
            "No intelligence scores available yet — click **Refresh Intelligence** in the Market Intel tab.")

    _section_bar("08 · System Status")
    # ── 8. SYSTEM FOOTER ─────────────────────────────────────────────────────
    df_main = _read_ndjson(_LOG_DIR / "main.log", tail=5)
    last_main = _latest_row(df_main) if not df_main.empty else {}
    data_ok = not df_main.empty
    dd_pct = abs(risk.get("daily_dd", 0.0))
    dd_lim = abs(risk.get("daily_dd_limit", 0.03))
    pk_pct = abs(risk.get("peak_dd", 0.0))
    pk_lim = abs(risk.get("peak_dd_limit", 0.10))

    sf1, sf2, sf3, sf4, sf5 = st.columns(5)
    sf1.markdown(
        f'<span class="{"dot-ok" if alpaca_ok else "dot-warn"}">●</span>'
        f'<span style="font-size:0.70rem;color:#AAAAAA;font-family:monospace"> API: {"live" if alpaca_ok else "log"}</span>',
        unsafe_allow_html=True,
    )
    sf2.markdown(
        f'<span class="{"dot-ok" if data_ok else "dot-err"}">●</span>'
        f'<span style="font-size:0.70rem;color:#AAAAAA;font-family:monospace"> Logs: {"ok" if data_ok else "missing"}</span>',
        unsafe_allow_html=True,
    )
    sf3.markdown(
        f'<span style="font-size:0.70rem;color:#AAAAAA;font-family:monospace">Daily DD: '
        f'<b style="color:{"#ff4444" if dd_pct/max(dd_lim, 1e-9) > 0.7 else "#484848"}">'
        f'{dd_pct:.2%}</b> / {dd_lim:.0%}</span>',
        unsafe_allow_html=True,
    )
    sf4.markdown(
        f'<span style="font-size:0.70rem;color:#AAAAAA;font-family:monospace">Peak DD: '
        f'<b style="color:{"#ff4444" if pk_pct/max(pk_lim, 1e-9) > 0.7 else "#484848"}">'
        f'{pk_pct:.2%}</b> / {pk_lim:.0%}</span>',
        unsafe_allow_html=True,
    )
    sf5.markdown(
        f'<span style="font-size:0.70rem;color:#AAAAAA;font-family:monospace">'
        f'Latency: {last_main.get("latency_ms", "—")} ms</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

# ── Inline fetchers (run inside Streamlit, no subprocess) ─────────────────────

@st.cache_data(ttl=300)
def _fetch_sentiment_live() -> Dict[str, float]:
    """
    Reddit/social sentiment scores — ApeWisdom public API.
    No API key required.  5 min Streamlit cache.
    Delegates to market_intel_macro.fetch_social_sentiment().
    """
    try:
        _fss = fetch_social_sentiment
        scores = _fss()
        if scores:
            st.session_state.pop("_sentiment_fetch_error", None)
        else:
            st.session_state["_sentiment_fetch_error"] = (
                "ApeWisdom returned no results."
            )
        return scores
    except Exception as _exc:
        st.session_state["_sentiment_fetch_error"] = str(_exc)
        return {}


# ── yfinance safety helpers ───────────────────────────────────────────────────

def _safe_yf_attr(ticker_obj, attr_name: str) -> Tuple[Any, Optional[str]]:
    """Return (value, error_code). error_code is None on success.

    Errors: "no_data" | "empty_df" | "exception:<msg>".
    Lets callers distinguish payload-present-but-empty from no-payload.
    """
    try:
        val = getattr(ticker_obj, attr_name, None)
        if val is None:
            return None, "no_data"
        if isinstance(val, pd.DataFrame) and val.empty:
            return None, "empty_df"
        return val, None
    except Exception as exc:
        return None, f"exception:{str(exc)[:120]}"


def _yf_retry(sym: str, attr_name: str, retries: int = 2) -> Tuple[Any, Optional[str]]:
    """Fetch a yfinance attribute with up to `retries` exponential-backoff retries."""
    import yfinance as _yf_r
    import time as _tr
    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        if attempt:
            _tr.sleep(0.1 * (2 ** (attempt - 1)))   # 0.1 s, 0.2 s
        try:
            val, err = _safe_yf_attr(_yf_r.Ticker(sym), attr_name)
            if err is None:
                return val, None
            last_err = err
        except Exception as exc:
            last_err = f"exception:{str(exc)[:120]}"
    return None, last_err


@st.cache_data(ttl=86400)
def _fetch_insider_live() -> Tuple[Dict[str, float], Set[str], Dict[str, str]]:
    """Insider trading scores via yfinance insider_transactions (SEC Form 4).

    Returns (scores, presence, errors).
    presence = tickers where the API returned any payload (even zero transactions).
    Score: 0.30 (all sells) → 0.90 (all buys); CEO/CFO buy → floored at 0.80.
    24 h Streamlit cache (SEC filings update once daily).
    """
    scores:   Dict[str, float] = {}
    presence: Set[str] = set()
    errors:   Dict[str, str] = {}

    try:
        import time as _t

        for sym in _SCAN_UNIVERSE:
            try:
                txns, err = _yf_retry(sym, "insider_transactions")
                if err is not None:
                    if len(errors) < 10:
                        errors[sym] = err
                    continue

                # API returned a DataFrame — mark as present regardless of content
                presence.add(sym)

                buy_count = sell_count = 0
                ceo_cfo_buy = 0.0
                for _, row in txns.iterrows():
                    text = str(row.get("Text", "") or row.get(
                        "Transaction", "")).upper()
                    val = 0.0
                    try:
                        val = abs(float(row.get("Value", 0) or 0))
                    except (TypeError, ValueError):
                        pass
                    pos = str(row.get("Position", "")).upper()

                    if "PURCHASE" in text or text.startswith("BUY"):
                        buy_count += 1
                        if any(k in pos for k in ("CEO", "CFO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL")):
                            ceo_cfo_buy += val
                    elif "SALE" in text or text.startswith("SELL"):
                        sell_count += 1

                total = buy_count + sell_count
                if total == 0:
                    # payload present, no transactions → neutral
                    scores[sym] = 0.50
                else:
                    buy_frac = buy_count / total
                    score = round(0.30 + 0.60 * buy_frac, 4)
                    if ceo_cfo_buy > 50_000:
                        score = round(max(score, 0.80), 4)
                    scores[sym] = score
                _t.sleep(0.10)
            except Exception as exc:
                if len(errors) < 10:
                    errors[sym] = f"exception:{str(exc)[:120]}"
                continue

    except Exception as _exc:
        st.session_state["_insider_fetch_error"] = str(_exc)
        return scores, presence, errors

    st.session_state.pop("_insider_fetch_error", None)
    return scores, presence, errors


@st.cache_data(ttl=86400)
def _fetch_institutional_live() -> Tuple[Dict[str, float], Set[str], Dict[str, str]]:
    """Institutional 13F conviction scores via yfinance.

    Returns (scores, presence, errors).
    presence = tickers where at least one payload layer (institutional_holders,
    major_holders, or FMP) returned real data. Neutral 0.50 is preserved in
    scores so the UI can show "present but neutral" rather than greyed-out.
    24 h Streamlit cache (13F filings update quarterly).
    """
    scores:   Dict[str, float] = {}
    presence: Set[str] = set()
    errors:   Dict[str, str] = {}

    try:
        import time as _t
        for sym in _SCAN_UNIVERSE:
            try:
                meta: Dict[str, Any] = {}
                s = fetch_institutional_data(sym, _meta=meta)
                # keep ALL scores — 0.50 means "holding steady"
                scores[sym] = s
                if meta.get("has_data"):
                    presence.add(sym)
                _t.sleep(0.10)
            except Exception as exc:
                if len(errors) < 10:
                    errors[sym] = f"exception:{str(exc)[:120]}"
                continue

        if any(v != 0.50 for v in scores.values()):
            st.session_state.pop("_institutional_fetch_error", None)
            st.session_state["_institutional_source"] = "Institutional 13F (yfinance)"
        else:
            st.session_state["_institutional_fetch_error"] = (
                "yfinance returned no institutional data."
            )
    except Exception as _exc:
        st.session_state["_institutional_fetch_error"] = str(_exc)

    return scores, presence, errors


@st.cache_data(ttl=3600)
def _fetch_news_sentiment_live() -> Dict[str, float]:
    """
    News sentiment scores from Yahoo Finance headlines + keyword dict.
    Scans _SCAN_UNIVERSE *and* current ApeWisdom tickers so that Reddit-
    trending stocks get a news score and can compete in the composite ranking.
    1 h Streamlit cache.
    """
    try:
        _hs = _headline_sentiment
        import yfinance as _yf
        import time as _t
        scores: Dict[str, float] = {}

        # Merge SCAN_UNIVERSE with live ApeWisdom tickers (deduplicated, order preserved)
        _ape_extra: List[str] = []
        try:
            _fss = fetch_social_sentiment
            _ape_extra = [s for s in _fss().keys(
            ) if s not in _SCAN_UNIVERSE][:30]
        except Exception:
            pass
        scan_tickers = list(_SCAN_UNIVERSE) + _ape_extra

        for sym in scan_tickers:
            try:
                news = _yf.Ticker(sym).news
                if not news:
                    continue
                hl = []
                for item in news[:10]:
                    content = item.get("content", {})
                    title = content.get("title", "") if isinstance(
                        content, dict) else item.get("title", "")
                    if title:
                        hl.append(_hs(title))
                if hl:
                    scores[sym] = round(sum(hl) / len(hl), 4)
                _t.sleep(0.05)
            except Exception:
                continue
        if scores:
            st.session_state.pop("_news_fetch_error", None)
        else:
            st.session_state["_news_fetch_error"] = "yfinance news returned no data."
        return scores
    except Exception as _exc:
        st.session_state["_news_fetch_error"] = str(_exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL FREE DATA SOURCES (2026-04-23)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def _fetch_fmp_insider() -> Tuple[Dict[str, float], Set[str], Dict[str, str]]:
    """FMP v4 insider-trading scores. Parallel fetch, 8 workers.

    Returns (scores, presence, errors).
    All scores are kept (including 0.50). presence = tickers where FMP returned
    a non-neutral signal (score != 0.50), indicating real data was found.
    """
    _fmp_fn = fetch_fmp_insider_score
    import time as _t
    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc

    scores:   Dict[str, float] = {}
    presence: Set[str] = set()
    errors:   Dict[str, str] = {}

    try:
        def _one(sym):
            s = _fmp_fn(sym)
            # ~40 req/s × 8 threads ≈ 320 req/min — within FMP free limit
            _t.sleep(0.15)
            return sym, s

        with ThreadPoolExecutor(max_workers=8) as pool:
            for fut in _asc({pool.submit(_one, sym): sym for sym in _SCAN_UNIVERSE}):
                try:
                    sym, score = fut.result()
                    scores[sym] = score
                    if score != 0.50:
                        presence.add(sym)
                except Exception as exc:
                    if len(errors) < 10:
                        errors[str(fut)] = f"exception:{str(exc)[:120]}"
    except Exception as _exc:
        st.session_state["_fmp_insider_fetch_error"] = str(_exc)

    return scores, presence, errors


@st.cache_data(ttl=3600)
def _fetch_fmp_institutional() -> Tuple[Dict[str, float], Set[str], Dict[str, str]]:
    """FMP v3 institutional-holder change scores. Parallel fetch, 8 workers.

    Returns (scores, presence, errors).
    All scores are kept. presence = tickers where FMP returned a non-neutral
    signal (score != 0.50), indicating real price-target or grades data was found.
    """
    _fmp_fn = fetch_fmp_institutional_score
    import time as _t
    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc

    scores:   Dict[str, float] = {}
    presence: Set[str] = set()
    errors:   Dict[str, str] = {}

    try:
        def _one(sym):
            s = _fmp_fn(sym)
            _t.sleep(0.15)
            return sym, s

        with ThreadPoolExecutor(max_workers=8) as pool:
            for fut in _asc({pool.submit(_one, sym): sym for sym in _SCAN_UNIVERSE}):
                try:
                    sym, score = fut.result()
                    scores[sym] = score
                    if score != 0.50:
                        presence.add(sym)
                except Exception as exc:
                    if len(errors) < 10:
                        errors[str(fut)] = f"exception:{str(exc)[:120]}"
    except Exception as _exc:
        st.session_state["_fmp_inst_fetch_error"] = str(_exc)

    return scores, presence, errors


@st.cache_data(ttl=1800)
def _fetch_stocktwits_sentiment() -> Dict[str, float]:
    """
    Stocktwits social sentiment. Parallel fetch, 4 workers.
    Includes tickers with active discussion even if no tagged sentiment.
    """
    _st_fn = fetch_stocktwits_sentiment
    import time as _t
    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc

    scores: Dict[str, float] = {}
    try:
        def _one(sym):
            result = _st_fn(sym)
            _t.sleep(0.4)   # Stocktwits: be polite
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            for fut in _asc({pool.submit(_one, sym): sym for sym in _SCAN_UNIVERSE}):
                scores.update(fut.result())
    except Exception as _exc:
        st.session_state["_stocktwits_fetch_error"] = str(_exc)
    return scores


@st.cache_data(ttl=3600)
def _fetch_finnhub_news() -> Dict[str, float]:
    """
    Finnhub news-sentiment scores. Sequential — Finnhub free tier = 60 req/min.
    Includes non-0.50 scores only (0.50 = no recent articles for that ticker).
    """
    _fnh_fn = fetch_finnhub_news_sentiment
    import time as _t

    scores: Dict[str, float] = {}
    try:
        for sym in _SCAN_UNIVERSE:
            score = _fnh_fn(sym)
            if score != 0.50:
                scores[sym] = score
            _t.sleep(1.1)   # 60 req/min free tier
    except Exception as _exc:
        st.session_state["_finnhub_news_error"] = str(_exc)
    return scores


@st.cache_data(ttl=86400)
def _fetch_finnhub_analyst() -> Dict[str, float]:
    """
    Finnhub analyst recommendation consensus for the scan universe.
    Endpoint: /api/v1/stock/recommendation  (strongBuy…strongSell).
    24 h cache — analyst ratings update weekly at most.
    Throttled to 1 call/s.
    """
    scores: Dict[str, float] = {}
    try:
        _fna = fetch_finnhub_recommendation_score
        import time as _t
        for sym in _SCAN_UNIVERSE:
            score = _fna(sym)
            if score != 0.50:
                scores[sym] = score
            _t.sleep(1.1)
    except Exception as _exc:
        st.session_state["_finnhub_analyst_error"] = str(_exc)
    return scores


def _geo_weighted(scores_dict: Dict[str, float], weights: Dict[str, float]) -> float:
    active = {k: v for k, v in scores_dict.items() if k in weights}
    if not active:
        return 0.50
    w_tot = sum(weights[k] for k in active)
    result = _math.exp(sum(weights[k] / w_tot * _math.log(max(1e-4, min(1.0, v)))
                           for k, v in active.items()))
    return round(result if _math.isfinite(result) else 0.50, 4)


_NEUTRAL_SCORE = 0.50

# Insider cluster 0.30 | Institutional 0.26 | News/Sentiment 0.38 | Macro 0.06
_EXTENDED_WEIGHTS = {
    "institutional":     0.18,
    "fmp_institutional": 0.08,
    "insider":           0.20,
    "fmp_insider":       0.10,
    "sentiment":         0.12,
    "stocktwits":        0.09,
    "news":              0.09,
    "finnhub_news":      0.08,
    "finnhub_analyst":   0.06,
    "macro":             0.06,
}


def _debug_fetch_one(sym: str) -> Dict[str, Any]:
    """Run all fetchers for a single ticker and write debug_<sym>.json to LOG_DIR.

    Use this to validate presence detection and score computation locally without
    triggering a full-universe fetch cycle.
    """
    import time as _td

    log_dir = _LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"symbol": sym,
                              "pillars": {}, "presence": {}, "errors": {}}

    # ── yfinance insider_transactions ─────────────────────────────────────────
    txns_val, txns_err = _yf_retry(sym, "insider_transactions")
    result["errors"]["yf_insider"] = txns_err
    result["presence"]["insider_yf"] = txns_err is None
    if txns_val is not None and hasattr(txns_val, "head"):
        result["pillars"]["yf_insider_raw"] = txns_val.head(
            3).to_dict("records")
    _td.sleep(0.3)

    # ── yfinance institutional_holders ────────────────────────────────────────
    ih_val, ih_err = _yf_retry(sym, "institutional_holders")
    result["errors"]["yf_institutional"] = ih_err
    result["presence"]["institutional_yf_holders"] = ih_err is None
    if ih_val is not None and hasattr(ih_val, "head"):
        result["pillars"]["yf_inst_raw"] = ih_val.head(3).to_dict("records")
    _td.sleep(0.3)

    # ── yfinance major_holders ────────────────────────────────────────────────
    mh_val, mh_err = _yf_retry(sym, "major_holders")
    result["presence"]["institutional_yf_major"] = mh_err is None
    if mh_val is not None and hasattr(mh_val, "to_dict"):
        result["pillars"]["yf_major_raw"] = mh_val.to_dict()

    # ── yfinance institutional score (3-layer) ────────────────────────────────
    try:
        meta: Dict[str, Any] = {}
        inst_score = fetch_institutional_data(sym, _meta=meta)
        result["pillars"]["institutional_score"] = inst_score
        result["presence"]["institutional_payload"] = meta.get(
            "has_data", False)
        result["pillars"]["institutional_source"] = meta.get("source", "none")
    except Exception as exc:
        result["errors"]["institutional_score"] = str(exc)
    _td.sleep(0.3)

    # ── yfinance news ─────────────────────────────────────────────────────────
    news_val, news_err = _yf_retry(sym, "news")
    result["errors"]["yf_news"] = news_err
    result["presence"]["news_yf"] = news_err is None and bool(news_val)
    if isinstance(news_val, list):
        result["pillars"]["yf_news_raw"] = news_val[:3]

    # ── FMP insider ───────────────────────────────────────────────────────────
    try:
        fmp_i = fetch_fmp_insider_score(sym)
        result["pillars"]["fmp_insider_score"] = fmp_i
        result["presence"]["insider_fmp"] = fmp_i != 0.50
    except Exception as exc:
        result["errors"]["fmp_insider"] = str(exc)

    # ── FMP institutional ─────────────────────────────────────────────────────
    try:
        fmp_inst = fetch_fmp_institutional_score(sym)
        result["pillars"]["fmp_institutional_score"] = fmp_inst
        result["presence"]["institutional_fmp"] = fmp_inst != 0.50
    except Exception as exc:
        result["errors"]["fmp_institutional"] = str(exc)

    # ── Canonical presence (normalized pillar names) ──────────────────────────
    result["canonical_presence"] = {
        "insider":       result["presence"].get("insider_yf", False)
        or result["presence"].get("insider_fmp", False),
        "institutional": result["presence"].get("institutional_payload", False)
        or result["presence"].get("institutional_fmp", False),
        "news":          result["presence"].get("news_yf", False),
    }

    out_path = log_dir / f"debug_{sym}.json"
    try:
        out_path.write_text(json.dumps(
            result, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return result


def _run_intel_fetch(macro_score: float, log_dir: Path) -> List[Dict]:
    """Fetch all intel sources, compute per-symbol composite scores, write JSON.

    active_sources uses canonical pillar names (insider / institutional / news /
    sentiment / finnhub_analyst / macro) so fmp_* and stocktwits are folded into
    their parent pillar for UI presence display.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch all sources ─────────────────────────────────────────────────────
    sentiment = _fetch_sentiment_live()
    insider,        insider_presence,  insider_errs = _fetch_insider_live()
    institutional,  inst_presence,     inst_errs = _fetch_institutional_live()
    news = _fetch_news_sentiment_live()
    fmp_insider,    fmp_ins_presence,  _ = _fetch_fmp_insider()
    fmp_institutional, fmp_inst_presence, _ = _fetch_fmp_institutional()
    stocktwits = _fetch_stocktwits_sentiment()
    finnhub_news = _fetch_finnhub_news()
    finnhub_analyst = _fetch_finnhub_analyst()

    # ── Source health log (count + presence) ──────────────────────────────────
    source_status = {
        "sentiment":       {"count": len(sentiment),          "present": len(sentiment)},
        "insider":         {"count": len(insider),            "present": len(insider_presence)},
        "institutional":   {"count": len(institutional),      "present": len(inst_presence)},
        "news":            {"count": len(news),               "present": len(news)},
        "fmp_insider":     {"count": len(fmp_insider),        "present": len(fmp_ins_presence)},
        "fmp_inst":        {"count": len(fmp_institutional),  "present": len(fmp_inst_presence)},
        "stocktwits":      {"count": len(stocktwits),         "present": len(stocktwits)},
        "finnhub_news":    {"count": len(finnhub_news),       "present": len(finnhub_news)},
        "finnhub_analyst": {"count": len(finnhub_analyst),    "present": len(finnhub_analyst)},
    }
    try:
        from utils.atomic_write import atomic_write_json
        atomic_write_json(log_dir / "intel_source_status.json", source_status)
    except Exception:
        pass

    # Save first-10 fetch errors for debugging
    for fname, errs in (("insider_errors.json", insider_errs), ("institutional_errors.json", inst_errs)):
        if errs:
            try:
                (log_dir / fname).write_text(json.dumps(errs, indent=2), encoding="utf-8")
            except Exception:
                pass

    # ── Build per-symbol rows ─────────────────────────────────────────────────
    universe = (
        set(sentiment) | set(insider) | set(institutional) | set(news)
        | set(fmp_insider) | set(fmp_institutional) | set(stocktwits)
        | set(finnhub_news) | set(finnhub_analyst)
    )
    rows = []
    for sym in sorted(universe):
        # Build sub-score dict for geo-weighted composite (individual source granularity)
        sub: Dict[str, float] = {"macro": macro_score}
        if sym in institutional:
            sub["institutional"] = institutional[sym]
        if sym in fmp_institutional:
            sub["fmp_institutional"] = fmp_institutional[sym]
        if sym in insider:
            sub["insider"] = insider[sym]
        if sym in fmp_insider:
            sub["fmp_insider"] = fmp_insider[sym]
        if sym in sentiment:
            sub["sentiment"] = sentiment[sym]
        if sym in stocktwits:
            sub["stocktwits"] = stocktwits[sym]
        if sym in news:
            sub["news"] = news[sym]
        if sym in finnhub_news:
            sub["finnhub_news"] = finnhub_news[sym]
        if sym in finnhub_analyst:
            sub["finnhub_analyst"] = finnhub_analyst[sym]

        comp = _geo_weighted(sub, _EXTENDED_WEIGHTS)

        # FMP is primary; yfinance is fallback. Composite sub-dict keeps granular
        # keys so _EXTENDED_WEIGHTS applies unchanged to the geo-weighted score.
        insider_score = fmp_insider.get(
            sym,       insider.get(sym,       _NEUTRAL_SCORE))
        inst_score = fmp_institutional.get(
            sym, institutional.get(sym, _NEUTRAL_SCORE))
        s = sentiment.get(sym,       _NEUTRAL_SCORE)
        stk = stocktwits.get(sym,      _NEUTRAL_SCORE)
        n = news.get(sym,            _NEUTRAL_SCORE)
        fh_n = finnhub_news.get(sym,    _NEUTRAL_SCORE)
        fh_a = finnhub_analyst.get(sym, _NEUTRAL_SCORE)

        direction = "long" if comp >= 0.58 else (
            "short" if comp <= 0.42 else "neutral")
        size_mult = (1.50 if comp >= 0.78 else 1.20 if comp >= 0.63 else
                     1.00 if comp >= 0.42 else 0.60 if comp >= 0.22 else 0.30)

        # Canonical presence: fmp_* and secondary sources fold into parent pillar.
        # presence_flags drives active_sources so the UI uses normalized pillar names.
        presence_flags: Dict[str, bool] = {
            "insider":         sym in insider_presence or sym in fmp_ins_presence,
            "institutional":   sym in inst_presence or sym in fmp_inst_presence,
            "news":            sym in news or sym in finnhub_news,
            "sentiment":       sym in sentiment or sym in stocktwits,
            "finnhub_analyst": sym in finnhub_analyst,
            "macro":           True,   # market-wide — always present
        }
        active_sources = [
            k for k, present in presence_flags.items() if present]

        _n_optional = len(presence_flags) - 1   # exclude macro
        confidence_score = round(
            sum(1 for k, v in presence_flags.items() if v and k != "macro")
            / max(_n_optional, 1), 2
        )

        rows.append({
            "symbol":                  sym,
            "composite":               comp,
            "institutional_score":     inst_score,
            "fmp_institutional_score": fmp_institutional.get(sym, _NEUTRAL_SCORE),
            "insider_score":           insider_score,
            "fmp_insider_score":       fmp_insider.get(sym, _NEUTRAL_SCORE),
            "sentiment_score":         s,
            "stocktwits_score":        stk,
            "news_score":              n,
            "finnhub_news_score":      fh_n,
            "finnhub_analyst_score":   fh_a,
            "macro_score":             macro_score,
            "flow_score":              _NEUTRAL_SCORE,   # schema compat
            "label":                   _label_for_score(comp),
            "direction":               direction,
            "size_multiplier":         size_mult,
            "active_sources":          active_sources,
            "presence_flags":          presence_flags,
            "confidence_score":        confidence_score,
        })
    rows.sort(key=lambda r: -r["composite"])

    out = log_dir / "latest_scores.json"
    try:
        with out.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    except OSError:
        pass
    st.session_state["_intel_rows_cache"] = rows
    return rows


# ── Tab rendering ─────────────────────────────────────────────────────────────
# tab_intel and tab_discovery are inner sub-tabs of tab_alpha (Alpha Hunter).
# Streamlit allows writing to a child tab's DeltaGenerator from outside the
# parent "with" block, so no re-indentation is needed here.

with tab_intel:
    scores_path = _LOG_DIR / "latest_scores.json"

    # ── Load scores immediately (auto, no button needed) ──────────────────────
    # Priority: in-memory cache from current session (set by _run_intel_fetch on
    # the same rerun that triggered st.rerun()).  This ensures that if the file
    # write fails (read-only / cloud filesystem) or the old file is stale, the
    # freshly-fetched rows are still displayed immediately after a refresh.
    score_rows: List[Dict] = st.session_state.get("_intel_rows_cache", [])
    _data_stale = False
    if not score_rows and scores_path.exists():
        try:
            _file_age_h = (datetime.now() - datetime.fromtimestamp(
                scores_path.stat().st_mtime)).total_seconds() / 3600
            if _file_age_h < 24:
                score_rows = json.loads(
                    scores_path.read_text(encoding="utf-8"))
            else:
                _data_stale = True
        except Exception:
            score_rows = []
    scores_df = pd.DataFrame(score_rows) if score_rows else pd.DataFrame()
    if not scores_df.empty:
        for col in ["composite", "institutional_score", "flow_score", "insider_score",
                    "sentiment_score", "news_score", "macro_score", "size_multiplier"]:
            if col in scores_df.columns:
                scores_df[col] = pd.to_numeric(
                    scores_df[col], errors="coerce").fillna(0.5)
        # Always sort descending by composite so head(N) reliably gives the top picks
        if "composite" in scores_df.columns:
            scores_df = scores_df.sort_values(
                "composite", ascending=False, ignore_index=True)

    def _intel_driver(row: Dict) -> str:
        """Map source score profile to a human-readable signal driver."""
        comp = float(row.get("composite",           0.5))
        inst = float(row.get("institutional_score", 0.5))
        i = float(row.get("insider_score",       0.5))
        s = float(row.get("sentiment_score",     0.5))
        n = float(row.get("news_score",          0.5))
        mac = float(row.get("macro_score",         0.5))

        # ── Bearish drivers ───────────────────────────────────────────────────
        if comp <= 0.42:
            if inst < 0.35 and i < 0.35:
                return "Institutions + Insiders Selling"
            if inst < 0.35:
                return "Institutional Distribution"
            if i < 0.35:
                return "Insider Selling Pressure"
            if s < 0.25:
                return "Negative Sentiment Spike"
            if n < 0.30:
                return "Negative News Flow"
            return "Multi-Source Bearish"

        # ── Bullish drivers ───────────────────────────────────────────────────
        if inst > 0.70 and i > 0.65:
            return "Institutions + Insiders Buying"
        if inst > 0.70 and s > 0.65:
            return "Institutional + Sentiment Alignment"
        if inst > 0.70:
            return "Institutional Accumulation"
        if i > 0.75:
            return "Unusual Insider Buying"
        if i > 0.65 and inst > 0.55:
            return "Smart Money Accumulation"
        if s > 0.80 and n > 0.65:
            return "Reddit + News Alignment"
        if s > 0.80 and i > 0.60:
            return "Retail + Insider Alignment"
        if s > 0.80:
            return "High Reddit Momentum"
        if n > 0.70 and s > 0.60:
            return "News + Sentiment Momentum"
        if n > 0.70:
            return "Positive News Flow"
        if mac > 0.70 and inst > 0.55:
            return "Macro Tailwind + Institutions"
        if inst > 0.60 and s > 0.60:
            return "Institutional + Sentiment Alignment"
        if i > 0.60:
            return "Insider Buying Activity"
        if s > 0.65:
            return "Social Momentum"
        return "Multi-Source Conviction"

    # ── Section index (quick reference) ───────────────────────────────────────
    _MI_SECTIONS = [
        "Top Buy Signals", "Multi-Timeframe Trends", "Aligned Trends",
        "Source Breakdown", "Radar · Compare", "Portfolio Scores", "Fundamental Filter",
    ]
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:4px;padding:6px 0 10px;'>"
        + "".join(
            f"<span style='font-size:0.60rem;font-family:monospace;color:#AAAAAA;"
            f"background:#0a0a0a;border:1px solid #1a1a1a;border-radius:2px;"
            f"padding:2px 7px;white-space:nowrap'>{s}</span>"
            for s in _MI_SECTIONS
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Top 5 BUY signals (first thing visible) ────────────────────────────────
    if not scores_df.empty:
        longs_df = scores_df[scores_df.get("direction", pd.Series(dtype=str)) == "long"].head(5) \
            if "direction" in scores_df.columns else scores_df.head(5)
        if not longs_df.empty:
            _section_bar("Top Buy Signals")
            _card_cols = st.columns(min(len(longs_df), 5))
            for _col, (_, _row) in zip(_card_cols, longs_df.iterrows()):
                _sc = float(_row["composite"])
                _clr = score_color(_sc)
                _sym = _row.get("symbol", "?")
                _lbl = _row.get("label", _label_for_score(_sc))
                _active = set(_row.get("active_sources") or [])
                _src_bars = ""
                for _src_lbl, _src_col, _src_key in [
                    ("Sent",  "sentiment_score",    "sentiment"),
                    ("Ins",   "insider_score",      "insider"),
                    ("Inst",  "institutional_score", "institutional"),
                    ("News",  "news_score",         "news"),
                    ("Macro", "macro_score",        None),
                ]:
                    _v = float(_row.get(_src_col, 0.5))
                    _has_real = (_src_key is None) or (_src_key in _active)
                    _bc = score_color(_v) if _has_real else "#333"
                    _lbl_clr = "#666" if _has_real else "#333"
                    _pct_clr = _bc
                    _bar_bg = "#222" if _has_real else "#111"
                    _src_bars += (
                        "<div style='display:flex;align-items:center;gap:4px;margin:2px 0'>"
                        "<span style='color:" + _lbl_clr +
                        ";font-size:0.60rem;width:28px'>" + _src_lbl + "</span>"
                        "<div style='flex:1;background:" + _bar_bg + ";border-radius:2px;height:4px'>"
                        "<div style='width:" +
                        f"{_v*100:.0f}" + "%;background:" + _bc +
                        ";height:4px;border-radius:2px'></div></div>"
                        "<span style='color:" + _pct_clr +
                        ";font-size:0.60rem;width:24px;text-align:right'>"
                        + (f"{_v:.0%}" if _has_real else "–") +
                        "</span>"
                        "</div>"
                    )
                _driver = _intel_driver(_row.to_dict())
                _is_selected = st.session_state.get("selected_ticker") == _sym
                _border_extra = f"box-shadow:0 0 0 2px {_clr}80;" if _is_selected else ""
                _col.markdown(
                    f'<div class="sig-card" style="border:1px solid {_clr}44;background:{_clr}0a;{_border_extra}">'
                    f'<div style="font-size:1.3rem;font-weight:900;color:{_clr};line-height:1">{_sym}</div>'
                    f'<div style="font-size:0.60rem;color:#AAAAAA;margin-bottom:2px">{_lbl}</div>'
                    f'<div style="font-size:0.58rem;font-weight:700;color:{_clr}cc;margin-bottom:5px;'
                    f'letter-spacing:0.05em;text-transform:uppercase">{_driver}</div>'
                    f'<div style="font-size:2.2rem;font-weight:900;color:#fff;line-height:1">{_sc:.0%}</div>'
                    f'<div style="font-size:0.62rem;color:#AAAAAA;margin-bottom:8px">'
                    f'×{float(_row.get("size_multiplier", 1)):.1f} size</div>'
                    f'{_src_bars}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with _col:
                    _btn_label = f"● {_sym}" if _is_selected else _sym
                    if st.button(_btn_label, key=f"sel_ticker_{_sym}", use_container_width=True,
                                 help=f"Inspect {_sym} pillar breakdown"):
                        st.session_state["selected_ticker"] = None if _is_selected else _sym
                        st.rerun()

    # ── Pillar Intelligence Panel for selected ticker ─────────────────────────
    _sel_ticker = st.session_state.get("selected_ticker")
    if _sel_ticker and not scores_df.empty:
        _sel_rows = scores_df[scores_df.get("symbol", pd.Series(dtype=str)) == _sel_ticker] \
            if "symbol" in scores_df.columns else pd.DataFrame()
        if not _sel_rows.empty:
            _sr = _sel_rows.iloc[0]
            _sc_val = float(_sr.get("composite", 0.5))
            _sc_clr = score_color(_sc_val)
            _active_src = set(_sr.get("active_sources") or [])

            _pillars = [
                ("SENT",  "sentiment_score",     "sentiment",     "#5c85d6"),
                ("INS",   "insider_score",        "insider",       "#d4a017"),
                ("INST",  "institutional_score",  "institutional", "#9c5fcb"),
                ("NEWS",  "news_score",           "news",          "#3ba7d6"),
                ("MACRO", "macro_score",          None,            "#5cb85c"),
            ]
            _rows_html = ""
            for _plbl, _pcol, _pkey, _default_clr in _pillars:
                _pv = float(_sr.get(_pcol, 0.5))
                _has = (_pkey is None) or (_pkey in _active_src)
                _pc = score_color(_pv) if _has else "#333"
                _pw = int(_pv * 100)
                _pct_txt = f"{_pv:.0%}" if _has else "—"
                _rows_html += (
                    f"<div class='pillar-row'>"
                    f"<span class='pillar-lbl'>{_plbl}</span>"
                    f"<div class='pillar-track'>"
                    f"<div class='pillar-fill' style='width:{_pw}%;background:{_pc};'></div>"
                    f"</div>"
                    f"<span class='pillar-pct' style='color:{_pc};'>{_pct_txt}</span>"
                    f"</div>"
                )

            st.markdown(
                f"<div class='pillar-panel'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                f"<span style='font-size:1.1rem;font-weight:900;color:{_sc_clr};"
                f"font-family:var(--font-mono)'>{_sel_ticker}</span>"
                f"<span style='font-size:0.58rem;color:#AAAAAA;font-family:var(--font-mono)'>"
                f"PILLAR INTELLIGENCE</span>"
                f"<span style='margin-left:auto;font-size:1.4rem;font-weight:900;"
                f"color:{_sc_clr};font-family:var(--font-mono)'>{_sc_val:.0%}</span>"
                f"<span style='font-size:0.55rem;color:{_sc_clr}80;"
                f"font-family:var(--font-mono)'>{_label_for_score(_sc_val).upper()}</span>"
                f"</div>"
                f"{_rows_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

    if not scores_df.empty:
        st.divider()

    # ── Fetch controls ─────────────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([2, 4])
    with hdr_l:
        fetch_btn = st.button("Refresh Intelligence", use_container_width=True,
                              help="Pulls ApeWisdom (sentiment) · yfinance 13F institutional · SEC insider filings · Yahoo News headlines")
    with hdr_r:
        if scores_path.exists():
            _mtime = datetime.fromtimestamp(scores_path.stat().st_mtime)
            _age_m = int((datetime.now() - _mtime).total_seconds() / 60)
            _age_s = f"{_age_m}m ago" if _age_m > 0 else "just now"
            st.caption(f"Last scan: **{_mtime.strftime('%H:%M:%S')}** ({_age_s}) · "
                       f"{len(score_rows)} symbols")
            # Source health indicators
            _src_status_path = _LOG_DIR / "intel_source_status.json"
            if _src_status_path.exists():
                try:
                    _src_st = json.loads(
                        _src_status_path.read_text(encoding="utf-8"))
                    # Map source name → session_state error key
                    _err_keys = {
                        "sentiment":     "_sentiment_fetch_error",
                        "insider":       "_insider_fetch_error",
                        "institutional": "_institutional_fetch_error",
                        "news":          "_news_fetch_error",
                    }
                    _src_html = ""
                    for _src_name, _src_count in _src_st.items():
                        # source_status values are now {"count": N, "present": N}
                        _cnt = _src_count.get("count",   0) if isinstance(
                            _src_count, dict) else _src_count
                        _present = _src_count.get("present", _cnt) if isinstance(
                            _src_count, dict) else _cnt
                        _ok = _cnt > 0
                        _sc = "#00c851" if _ok else "#ff4444"
                        _sl = f"{_cnt} tickers ({_present} present)" if _ok else "FAILED"
                        _src_html += (
                            "<span style='font-size:0.68rem;font-family:monospace;"
                            "margin-right:12px;'>"
                            "<span style='color:" + _sc + "'>●</span> "
                            "<span style='color:#888'>" + _src_name + ":</span> "
                            "<span style='color:" + _sc + ";font-weight:700'>" + _sl + "</span>"
                            "</span>"
                        )
                    st.markdown(_src_html, unsafe_allow_html=True)
                    # Show per-source error detail if captured
                    for _src_name, _ekey in _err_keys.items():
                        _emsg = st.session_state.get(_ekey)
                        if _emsg:
                            st.caption(f"⚠ {_src_name} error: {_emsg}")
                except Exception:
                    pass
        else:
            st.caption("No scan yet — click **Refresh Intelligence** to run.")

    # ── Auto-refresh when data > 24h or stale flag ────────────────────────────
    if _data_stale and not st.session_state.get("_auto_refresh_triggered"):
        st.session_state["_auto_refresh_triggered"] = True
        st.warning(
            "Intelligence data is older than 24h — refreshing automatically...")
        fetch_btn = True

    # ── Live fetch triggered ───────────────────────────────────────────────────
    if fetch_btn:
        # Derive macro score from latest VIX in regime log
        regime_now = _get_regime_state()
        _main_df = _read_ndjson(_LOG_DIR / "main.log", tail=5)
        vix_now = None
        if not _main_df.empty and "vix" in _main_df.columns:
            vix_now = _main_df["vix"].dropna(
            ).iloc[-1] if _main_df["vix"].notna().any() else None
        if vix_now is not None:
            try:
                _cms = compute_enhanced_macro_score
                _macro_detail = _cms(float(vix_now))
                macro_now = _macro_detail["composite"]
            except Exception:
                macro_now = round(
                    max(0.10, min(0.90, 1.0 - (float(vix_now) - 10) / 50)), 4)
        else:
            # Fallback: fetch live VIX from yfinance, then compute enhanced macro score
            try:
                import yfinance as _yf_mac
                _vix_live = _yf_mac.download("^VIX", period="2d", interval="1d",
                                             progress=False, auto_adjust=True)
                _vix_val = float(_vix_live["Close"].iloc[-1].squeeze())
                try:
                    _cms = compute_enhanced_macro_score
                    _macro_detail = _cms(_vix_val)
                    macro_now = _macro_detail["composite"]
                except Exception:
                    macro_now = round(
                        max(0.10, min(0.90, 1.0 - (_vix_val - 10) / 50)), 4)
            except Exception:
                macro_now = 0.60   # conservative neutral fallback

        # Clear all fetcher caches so stale results don't persist
        _fetch_sentiment_live.clear()
        _fetch_insider_live.clear()
        _fetch_institutional_live.clear()
        _fetch_news_sentiment_live.clear()
        # Reset the in-process institutional cache
        try:
            _ric = reset_institutional_cache
            _ric()
        except Exception:
            pass
        with st.spinner("Fetching ApeWisdom · OpenInsider · yfinance 13F · Yahoo News …"):
            fetched = _run_intel_fetch(macro_now, _LOG_DIR)
        st.cache_data.clear()
        st.success(f"Done — {len(fetched)} symbols scored. Page refreshing…")
        time.sleep(0.8)
        st.rerun()

    if not score_rows:
        st.info("No intelligence data yet — click **Refresh Intelligence** above.")

    if not scores_df.empty:
        st.divider()

        # ── KPI row ───────────────────────────────────────────────────────────────
        n_total = len(scores_df)
        n_long = int((scores_df["direction"] == "long").sum(
        )) if "direction" in scores_df.columns else 0
        n_short = int((scores_df["direction"] == "short").sum(
        )) if "direction" in scores_df.columns else 0
        avg_score = float(scores_df["composite"].mean(
        )) if "composite" in scores_df.columns else 0.5
        top_score = float(scores_df["composite"].max(
        )) if "composite" in scores_df.columns else 0.5

        ik1, ik2, ik3, ik4, ik5 = st.columns(5)
        ik1.metric("Symbols scored", n_total)
        ik2.metric("Long signals",   n_long)
        ik3.metric("Short signals",  n_short)
        ik4.metric("Avg conviction", f"{avg_score:.0%}")
        ik5.metric("Top score",      f"{top_score:.0%}")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME ANALYSIS + TOP 5 TRENDS
        # ══════════════════════════════════════════════════════════════════════

        _section_bar("Multi-Timeframe Trend Analysis")

        # Timeframe selector
        tf_tab_d, tf_tab_w, tf_tab_m, tf_tab_all = st.tabs(
            ["Daily", "Weekly", "Monthly", "All Timeframes"]
        )

        # Run multi-tf analysis on top 30 intel symbols (cached 1h)
        _top_syms_for_tf = scores_df["symbol"].head(30).tolist() \
            if "symbol" in scores_df.columns else []

        _run_tf = st.button("📊 Load Multi-Timeframe Analysis",
                            help="Downloads 2y of price data for top 30 symbols — takes ~20s",
                            key="run_tf_btn")
        _tf_cache_key = "tf_df_cache"
        if _run_tf:
            with st.spinner("Downloading price data for multi-timeframe analysis…"):
                _tf_df = _multi_tf_signals(_top_syms_for_tf)
            st.session_state[_tf_cache_key] = _tf_df
            st.rerun()
        _tf_df = st.session_state.get(_tf_cache_key, pd.DataFrame())

        def _trend_arrow(t: float) -> str:
            if t >= 0.75:
                return "▲"
            if t <= 0.25:
                return "▼"
            return "→"

        def _tf_badge(score: float, label: str, rsi: float) -> str:
            clr = score_color(score)
            arrow = _trend_arrow(score)
            return (
                f'<div style="text-align:center;padding:4px 0">'
                f'<div style="font-size:0.95rem;font-weight:700;color:{clr}">'
                f'{arrow} {score:.0%}</div>'
                f'<div style="font-size:0.65rem;color:#777">RSI {rsi:.0f}</div>'
                f'</div>'
            )

        def _render_tf_table(col_score: str, col_rsi: str, col_trend: str,
                             merged_df: pd.DataFrame, timeframe: str) -> None:
            if merged_df.empty or col_score not in merged_df.columns:
                st.info(
                    f"No {timeframe} data — click Fetch Live Intelligence first.")
                return
            top5 = merged_df.nlargest(5, col_score)
            st.markdown(
                f'<p style="font-size:0.72rem;color:#AAAAAA;margin-bottom:8px">'
                f'Top 5 {timeframe} trend picks (technical score + intel conviction)</p>',
                unsafe_allow_html=True,
            )
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                sym = row.get("symbol", "")
                ts = float(row.get(col_score, 0.5))
                trsi = float(row.get(col_rsi, 50))
                ttr = float(row.get(col_trend, 0.5))
                comp = float(row.get("composite", 0.5))
                clr = score_color(ts)
                arrow = _trend_arrow(ttr)

                medal = ["", "🥇", "🥈", "🥉", "4.", "5."][rank]

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;'
                    f'padding:9px 14px;margin:3px 0;border-radius:8px;'
                    f'background:{clr}10;border-left:3px solid {clr}">'

                    f'<span style="font-size:1rem;width:22px">{medal}</span>'
                    f'<span style="font-size:1.05rem;font-weight:800;color:#fff;width:60px">{sym}</span>'

                    f'<span style="font-size:1.5rem;font-weight:900;color:{clr};width:56px">'
                    f'{arrow}</span>'

                    f'<div style="flex:1">'
                    f'<div style="font-size:0.78rem;color:#ccc">'
                    f'{timeframe} score: <b style="color:{clr}">{ts:.0%}</b> '
                    f'| RSI {trsi:.0f} '
                    f'| Intel conviction: <b>{comp:.0%}</b></div>'
                    f'<div style="background:#222;border-radius:4px;height:5px;margin-top:4px">'
                    f'<div style="width:{ts*100:.0f}%;background:{clr};height:5px;'
                    f'border-radius:4px"></div></div>'
                    f'</div>'

                    f'<span style="font-size:0.72rem;color:#777;width:90px;text-align:right">'
                    f'{_label_for_score(comp)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Merge tf signals with intel scores for combined ranking
        if not _tf_df.empty and not scores_df.empty:
            _merged = _tf_df.merge(
                scores_df[["symbol", "composite", "label", "direction"]],
                on="symbol", how="left"
            )
            _merged["composite"] = _merged["composite"].fillna(0.5)
            # Combined rank: 60% tf_score + 40% intel composite
            _merged["combined_score"] = 0.60 * \
                _merged["tf_score"] + 0.40 * _merged["composite"]
        else:
            _merged = pd.DataFrame()

        with tf_tab_d:
            if not _merged.empty:
                _render_tf_table("daily_score", "daily_rsi",
                                 "daily_trend", _merged, "Daily")
            else:
                _dir_col = scores_df["direction"] if "direction" in scores_df.columns else pd.Series(
                    dtype=str)
                _daily_longs = scores_df[_dir_col == "long"].head(5) \
                    if not scores_df.empty else pd.DataFrame()
                st.info(
                    "Click **Load Multi-Timeframe Analysis** above — showing top Intel picks until ready.")
                if not _daily_longs.empty:
                    for _, r in _daily_longs.iterrows():
                        sc = float(r["composite"])
                        clr = score_color(sc)
                        st.markdown(
                            f'<div style="padding:8px 14px;margin:3px 0;border-radius:6px;'
                            f'background:{clr}10;border-left:3px solid {clr}">'
                            f'<b style="color:{clr}">{r["symbol"]}</b> — {sc:.0%} {r.get("label", "")}'
                            f'</div>', unsafe_allow_html=True)

        with tf_tab_w:
            if not _merged.empty:
                _render_tf_table("weekly_score", "weekly_rsi",
                                 "weekly_trend", _merged, "Weekly")
            else:
                st.info(
                    "Click **Load Multi-Timeframe Analysis** above to compute weekly signals.")

        with tf_tab_m:
            if not _merged.empty:
                _render_tf_table("monthly_score", "monthly_rsi",
                                 "monthly_trend", _merged, "Monthly")
            else:
                st.info(
                    "Click **Load Multi-Timeframe Analysis** above to compute monthly signals.")

        with tf_tab_all:
            _section_bar("Top 5 Trends · All Timeframes Aligned")
            st.caption(
                "These picks score well across Daily + Weekly + Monthly simultaneously. "
                "Multi-timeframe alignment is the strongest buy signal."
            )
            if not _merged.empty and "combined_score" in _merged.columns:
                _top5_all = _merged.nlargest(5, "combined_score")
                for rank, (_, row) in enumerate(_top5_all.iterrows(), 1):
                    sym = row.get("symbol", "")
                    ds = float(row.get("daily_score",   0.5))
                    ws = float(row.get("weekly_score",  0.5))
                    ms = float(row.get("monthly_score", 0.5))
                    cs = float(row.get("combined_score", 0.5))
                    comp = float(row.get("composite",     0.5))
                    clr = score_color(cs)
                    medal = ["", "🥇", "🥈", "🥉", "4.", "5."][rank]

                    # Three mini bars
                    def _mini_bar(v: float, label: str) -> str:
                        c = score_color(v)
                        return (
                            f'<div style="display:flex;align-items:center;gap:4px">'
                            f'<span style="font-size:0.62rem;color:#888;width:18px">{label}</span>'
                            f'<div style="width:60px;background:#333;border-radius:3px;height:5px">'
                            f'<div style="width:{v*100:.0f}%;background:{c};height:5px;border-radius:3px">'
                            f'</div></div>'
                            f'<span style="font-size:0.62rem;color:#aaa;width:28px">{v:.0%}</span>'
                            f'</div>'
                        )

                    bars_html = _mini_bar(
                        ds, "D") + _mini_bar(ws, "W") + _mini_bar(ms, "M")

                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:14px;'
                        f'padding:12px 16px;margin:5px 0;border-radius:10px;'
                        f'background:{clr}12;border:1px solid {clr}44">'

                        f'<span style="font-size:1.2rem;width:28px">{medal}</span>'
                        f'<span style="font-size:1.3rem;font-weight:800;color:#fff;width:66px">{sym}</span>'

                        f'<div style="width:110px">{bars_html}</div>'

                        f'<div style="flex:1;border-left:1px solid #333;padding-left:14px">'
                        f'<div style="font-size:0.82rem;color:#ccc">'
                        f'Combined score: <b style="color:{clr};font-size:1.05rem">{cs:.0%}</b> '
                        f'&nbsp;|&nbsp; Intel: <b>{comp:.0%}</b>'
                        f'</div>'
                        f'<div style="font-size:0.70rem;color:#777;margin-top:2px">'
                        f'{_label_for_score(comp)} — all timeframes aligned</div>'
                        f'</div>'

                        f'<span style="padding:5px 14px;border-radius:14px;'
                        f'background:{clr}33;color:{clr};font-size:0.80rem;font-weight:700">'
                        f'BUY SIGNAL</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "Fetch intelligence first to compute multi-timeframe alignment.")

        st.divider()

        # longs_df used by both the source chart and the radar default selection
        longs_df = scores_df[scores_df["direction"] == "long"].sort_values(
            "composite", ascending=False
        ).head(6) if "direction" in scores_df.columns else scores_df.head(6)

        # ── Layout: source breakdown chart  |  radar ─────────────────────────────
        chart_l, chart_r = st.columns([3, 2])

        with chart_l:
            _section_bar("Source Breakdown · Top 20")
            top20 = scores_df.head(20).copy()
            src_cols_map = {
                "Institutional": "institutional_score",
                "Insider":       "insider_score",
                "Sentiment":     "sentiment_score",
                "News":          "news_score",
                "Macro":         "macro_score",
            }
            src_colors = ["#5c85d6", "#e07b54",
                          "#6abf69", "#ffbb33", "#b39ddb"]

            fig_bar = go.Figure()
            for (label, col), color in zip(src_cols_map.items(), src_colors):
                if col in top20.columns:
                    fig_bar.add_trace(go.Bar(
                        name=label,
                        x=top20["symbol"],
                        y=top20[col],
                        marker_color=color,
                        opacity=0.85,
                    ))
            fig_bar.add_trace(go.Scatter(
                name="Composite",
                x=top20["symbol"],
                y=top20["composite"],
                mode="markers",
                marker=dict(symbol="diamond", size=9, color="#ffffff",
                            line=dict(color="#aaa", width=1)),
            ))
            fig_bar.update_layout(**_PLOTLY_LAYOUT)
            fig_bar.update_layout(
                barmode="group",
                height=340,
                margin=dict(t=10, b=30, l=10, r=10),
                yaxis=dict(range=[0, 1], tickformat=".0%",
                           gridcolor="#1a1a1a"),
                xaxis=dict(showgrid=False, tickangle=-35),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_r:
            _section_bar("Radar · Compare Symbols")
            all_syms = scores_df["symbol"].tolist(
            ) if "symbol" in scores_df.columns else []
            default_sel = [
                s for s in all_syms if s in longs_df["symbol"].values][:3] or all_syms[:3]
            selected_syms = st.multiselect("Symbols", all_syms, default=default_sel,
                                           label_visibility="collapsed")
            if selected_syms:
                cats = ["Institutional", "Insider",
                        "Sentiment", "News", "Macro"]
                src_keys = ["institutional_score", "insider_score",
                            "sentiment_score", "news_score", "macro_score"]
                fig_rad = go.Figure()
                for sym in selected_syms:
                    r = scores_df[scores_df["symbol"] == sym]
                    if r.empty:
                        continue
                    vals = [float(r.iloc[0].get(k, 0.5)) for k in src_keys]
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=cats + [cats[0]],
                        fill="toself", name=sym, opacity=0.65,
                    ))
                fig_rad.update_layout(**_PLOTLY_LAYOUT)
                fig_rad.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                                        gridcolor="#2a2a2a", linecolor="#2a2a2a"),
                        angularaxis=dict(gridcolor="#2a2a2a",
                                         linecolor="#2a2a2a"),
                    ),
                    showlegend=True,
                    height=300,
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_rad, use_container_width=True)

        st.divider()

        # ── Portfolio overlap ──────────────────────────────────────────────────────
        _section_bar("Your Portfolio Scores")
        portf_now = _get_alpaca_portfolio()
        held_syms = [p["symbol"]
                     for p in portf_now["positions"]] if portf_now else []

        if held_syms:
            overlap = scores_df[scores_df["symbol"].isin(held_syms)].copy()
            unscored = [
                s for s in held_syms if s not in scores_df["symbol"].values]

            if not overlap.empty:
                _overlap_cols = [c for c in ["symbol", "composite", "label", "direction",
                                             "institutional_score", "insider_score",
                                             "sentiment_score", "news_score", "macro_score"]
                                 if c in overlap.columns]
                disp_overlap = overlap[_overlap_cols].copy()
                disp_overlap = disp_overlap.sort_values(
                    "composite", ascending=False)
                for num_col in ["composite", "institutional_score", "insider_score",
                                "sentiment_score", "news_score", "macro_score"]:
                    if num_col in disp_overlap.columns:
                        disp_overlap[num_col] = disp_overlap[num_col].apply(
                            lambda x: round(x, 3))

                def _score_cell_css(v):
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        return ""
                    clr = score_color(f)
                    return f"color: {clr}; font-weight: 600"

                _grad_overlap = [c for c in ["composite", "institutional_score", "insider_score",
                                             "sentiment_score", "news_score", "macro_score"]
                                 if c in disp_overlap.columns]
                styled_overlap = disp_overlap.style.map(
                    _score_cell_css, subset=_grad_overlap)
                st.dataframe(styled_overlap,
                             use_container_width=True, hide_index=True)
            if unscored:
                st.caption(f"Not in scan universe: {', '.join(unscored)} "
                           f"— re-run fetch to include them.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # FUNDAMENTAL QUALITY (Buffett-style) — held + top long signals
        # ══════════════════════════════════════════════════════════════════════

        _section_bar("Fundamental Quality · Buffett Filter")
        st.caption(
            "Quality Score: Debt/Equity <1 (25pts) · Free Cash Flow >0 (30pts) · "
            "Net Margin >15% (30pts) · ROE >15% (15pts).  Cached 24 h."
        )

        _fund_targets = list(dict.fromkeys(
            held_syms +
            (longs_df["symbol"].tolist(
            ) if not longs_df.empty and "symbol" in longs_df.columns else [])
        ))[:15]  # cap at 15 to keep fetch fast

        _load_fund = st.button("Load Fundamental Analysis",
                               key="load_fund_btn",
                               help="Fetches balance-sheet metrics via yfinance — ~2s per symbol")
        _fund_cache_key = "fund_df_cache"
        if _load_fund:
            with st.spinner(f"Fetching fundamentals for {len(_fund_targets)} symbols…"):
                _fund_df = _get_quality_scores(tuple(_fund_targets))
            st.session_state[_fund_cache_key] = _fund_df
            st.rerun()
        _fund_df: Optional[pd.DataFrame] = st.session_state.get(
            _fund_cache_key)

        if _fund_df is not None and not _fund_df.empty:
            # Cards row for top 5 quality
            _top_qual = _fund_df.sort_values(
                "quality_score", ascending=False).head(5)
            _qcols = st.columns(min(len(_top_qual), 5))
            for _qc, (_, _qr) in zip(_qcols, _top_qual.iterrows()):
                _qs = float(_qr.get("quality_score", 0))
                _qclr = _qr.get("insight_color", "#9e9e9e")
                _qsym = _qr.get("symbol", "?")
                _qins = _qr.get("value_insight", "")
                _qde = _qr.get("debt_equity", None)
                _qmg = _qr.get("net_margin", None)
                _qroe = _qr.get("roe", None)
                _qfc = _qr.get("free_cashflow", None)
                _de_str = f"D/E {_qde:.2f}" if _qde is not None else "D/E —"
                _mg_str = f"Margin {_qmg:.1f}%" if _qmg is not None else "Margin —"
                _roe_str = f"ROE {_qroe:.1f}%" if _qroe is not None else "ROE —"
                _fcf_str = ("FCF ✓" if _qfc is not None and _qfc > 0
                            else "FCF ✗" if _qfc is not None else "FCF —")
                _fcf_clr = "#00c851" if _qfc is not None and _qfc > 0 else "#ff4444" if _qfc is not None else "#555"
                _qc.markdown(
                    f'<div class="sig-card" style="border:1px solid {_qclr}44;background:{_qclr}0a">'
                    f'<div style="font-size:1.2rem;font-weight:900;color:{_qclr}">{_qsym}</div>'
                    f'<div style="font-size:0.60rem;color:#AAAAAA;margin-bottom:4px">{_qins}</div>'
                    f'<div style="font-size:2.0rem;font-weight:900;color:#fff;line-height:1">{_qs:.0f}</div>'
                    f'<div style="font-size:0.60rem;color:#888;margin-bottom:6px">/ 100 pts</div>'
                    f'<div style="font-size:0.65rem;color:#aaa;line-height:1.6">'
                    f'{_de_str}<br>{_mg_str}<br>{_roe_str}<br>'
                    f'<span style="color:{_fcf_clr}">{_fcf_str}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Full table
            _fund_show = _fund_df[["symbol", "sector", "quality_score", "value_insight",
                                   "debt_equity", "net_margin", "roe"]].copy()
            _fund_show.columns = ["Symbol", "Sector", "Quality /100", "Insight",
                                  "D/E Ratio", "Net Margin %", "ROE %"]
            _fund_show = _fund_show.sort_values(
                "Quality /100", ascending=False)

            def _qual_css(v):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return ""
                clr = ("#00c851" if f >= 70 else "#7cb342" if f >= 50
                       else "#ff8800" if f >= 30 else "#ff4444")
                return f"color: {clr}; font-weight: 700"

            st.dataframe(
                _fund_show.style.map(_qual_css, subset=["Quality /100"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Click **Load Fundamental Analysis** to run the Buffett quality filter.")

        st.divider()

        # ── Full scores table (collapsible) ───────────────────────────────────────
        with st.expander(f"All {n_total} symbols — full table"):
            dir_filter = st.radio("Filter direction", ["All", "long", "neutral", "short"],
                                  horizontal=True, label_visibility="collapsed")
            tbl = scores_df if dir_filter == "All" else \
                scores_df[scores_df["direction"] == dir_filter]

            show_cols = [c for c in ["symbol", "composite", "label", "direction",
                                     "institutional_score", "insider_score",
                                     "sentiment_score", "news_score", "macro_score",
                                     "size_multiplier"]
                         if c in tbl.columns and c != "flow_score"]
            grad_cols = [c for c in ["composite", "institutional_score", "insider_score",
                                     "sentiment_score", "news_score", "macro_score"]
                         if c in show_cols]

            def _score_cell_css2(v):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return ""
                clr = score_color(f)
                return f"color: {clr}; font-weight: 600"

            styled_tbl = tbl[show_cols].style.map(
                _score_cell_css2, subset=grad_cols)
            st.dataframe(styled_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab_trades:

    df_trades = _read_ndjson(_LOG_DIR / "trades.log", tail=5000)

    if df_trades.empty:
        # ── Fallback: live Alpaca order history ───────────────────────────────
        _ak = os.getenv("ALPACA_API_KEY", "")
        _sk = os.getenv("ALPACA_SECRET_KEY", "")
        if not _ak or not _sk or _ak == "PKXXXXXXXXXXXXXXXXXXXXXXXX":
            st.info(
                "No trade log found and no Alpaca API credentials configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in `.env` to load live order history."
            )
        else:
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                _tc = TradingClient(_ak, _sk, paper=os.getenv(
                    "ENV", "paper").lower() != "live")
                _olist = _tc.get_orders(GetOrdersRequest(
                    status=QueryOrderStatus.ALL, limit=200))
                if _olist:
                    _orows = []
                    for o in _olist:
                        _price = float(o.filled_avg_price or 0)
                        _fqty = float(o.filled_qty or 0)
                        _orows.append({
                            "date":   str(o.filled_at or o.submitted_at)[:10],
                            "time":   str(o.filled_at or o.submitted_at)[11:19],
                            "symbol": o.symbol,
                            "side":   str(o.side).split(".")[-1].upper(),
                            "qty":    float(o.qty or 0),
                            "filled": _fqty,
                            "price":  _price,
                            "value":  _price * _fqty,
                            "status": str(o.status).split(".")[-1].lower(),
                        })
                    _odf = pd.DataFrame(_orows)
                    _status = _odf["status"].str.lower()
                    _n_filled = int((_status == "filled").sum())
                    _n_pending = int(_status.isin(
                        ["new", "accepted", "pending_new", "held"]).sum())
                    _n_cancel = int(_status.isin(
                        ["canceled", "cancelled", "expired"]).sum())
                    _total_val = float(
                        _odf.loc[_status == "filled", "value"].sum())

                    # Hero strip
                    st.markdown(
                        f'<div class="kpi-strip">'
                        f'<div class="kpi-cell"><div class="kpi-label">Orders</div>'
                        f'<div class="kpi-value">{len(_odf)}</div></div>'
                        f'<div class="kpi-cell"><div class="kpi-label">Filled</div>'
                        f'<div class="kpi-value" style="color:#00c851">{_n_filled}</div></div>'
                        f'<div class="kpi-cell"><div class="kpi-label">Pending</div>'
                        f'<div class="kpi-value" style="color:#ffbb33">{_n_pending}</div></div>'
                        f'<div class="kpi-cell"><div class="kpi-label">Cancelled</div>'
                        f'<div class="kpi-value" style="color:#AAAAAA">{_n_cancel}</div></div>'
                        f'<div class="kpi-cell"><div class="kpi-label">Total Deployed</div>'
                        f'<div class="kpi-value">${_total_val:,.0f}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Column header
                    st.markdown(
                        '<div style="display:flex;gap:10px;padding:2px 10px;font-size:0.62rem;'
                        'color:#AAAAAA;font-weight:700;letter-spacing:.08em;text-transform:uppercase">'
                        '<span style="width:80px">Date</span>'
                        '<span style="width:56px">Symbol</span>'
                        '<span style="width:50px">Side</span>'
                        '<span style="width:60px">Qty</span>'
                        '<span style="width:80px;text-align:right">Price</span>'
                        '<span style="width:90px;text-align:right">Value</span>'
                        '<span style="width:70px;text-align:right">Status</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    _STATUS_CLR = {"filled": "#00c851", "new": "#ffbb33", "accepted": "#ffbb33",
                                   "canceled": "#555", "expired": "#555", "pending_new": "#ffbb33"}
                    _SIDE_CLR = {"BUY": "#00c851", "SELL": "#ff4444"}

                    for _, _or in _odf.sort_values("date", ascending=False).iterrows():
                        _sc = _STATUS_CLR.get(_or["status"], "#888")
                        _sdc = _SIDE_CLR.get(_or["side"], "#aaa")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'padding:5px 10px;margin:2px 0;border-radius:5px;background:#0d0d0d">'
                            f'<span style="font-size:0.75rem;color:#AAAAAA;width:80px">{_or["date"]}</span>'
                            f'<span style="font-size:0.88rem;font-weight:800;color:#fff;width:56px">{_or["symbol"]}</span>'
                            f'<span style="font-size:0.72rem;font-weight:700;color:{_sdc};width:50px">{_or["side"]}</span>'
                            f'<span style="font-size:0.75rem;color:#aaa;width:60px">{_or["filled"]:,.3g}/{_or["qty"]:,.3g}</span>'
                            f'<span style="font-size:0.80rem;color:#ccc;width:80px;text-align:right">${_or["price"]:,.2f}</span>'
                            f'<span style="font-size:0.80rem;font-weight:600;color:#ddd;width:90px;text-align:right">${_or["value"]:,.0f}</span>'
                            f'<span style="font-size:0.68rem;font-weight:700;color:{_sc};width:70px;text-align:right">{_or["status"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No orders found on this account.")
            except Exception as _e:
                st.error(f"Could not load Alpaca orders: {_e}")
    else:
        # ── Filters ────────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([2, 2, 3])

        symbols_all = sorted(df_trades["symbol"].dropna().unique().tolist()) \
            if "symbol" in df_trades.columns else []
        actions_all = sorted(df_trades["action"].dropna().unique().tolist()) \
            if "action" in df_trades.columns else []

        with fc1:
            selected_syms = st.multiselect("Symbol", symbols_all, default=[])
        with fc2:
            selected_acts = st.multiselect("Action", actions_all, default=[])
        with fc3:
            if "timestamp" in df_trades.columns and df_trades["timestamp"].notna().any():
                ts_min = df_trades["timestamp"].min().date()
                ts_max = df_trades["timestamp"].max().date()
                date_range = st.date_input(
                    "Date range",
                    value=(ts_min, ts_max),
                    min_value=ts_min,
                    max_value=ts_max,
                )
            else:
                date_range = None

        mask = pd.Series(True, index=df_trades.index)
        if selected_syms and "symbol" in df_trades.columns:
            mask &= df_trades["symbol"].isin(selected_syms)
        if selected_acts and "action" in df_trades.columns:
            mask &= df_trades["action"].isin(selected_acts)
        if date_range and len(date_range) == 2 and "timestamp" in df_trades.columns:
            lo = pd.Timestamp(date_range[0], tz="UTC")
            hi = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
            mask &= (df_trades["timestamp"] >= lo) & (
                df_trades["timestamp"] < hi)

        filtered = df_trades[mask].copy()

        # ── Summary KPIs ───────────────────────────────────────────────────────
        total_trades = len(filtered)
        closed = filtered[filtered["action"] == "CLOSE"] \
            if "action" in filtered.columns else pd.DataFrame()
        total_pnl = closed["pnl"].sum() if "pnl" in closed.columns else 0.0
        win_rate = (closed["pnl"] > 0).mean() if (
            "pnl" in closed.columns and not closed.empty) else 0.0

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Events", total_trades)
        sm2.metric("Closed Trades", len(closed))
        sm3.metric("Total P&L", fmt_usd(total_pnl),
                   delta_color=pnl_delta_color(total_pnl))
        sm4.metric("Win Rate", fmt_pct(win_rate, plus=False))

        # ── Cumulative P&L chart ───────────────────────────────────────────────
        if "pnl" in closed.columns and not closed.empty and "timestamp" in closed.columns:
            pnl_ts = closed[["timestamp", "pnl", "symbol"]].dropna(subset=[
                                                                   "pnl"])
            pnl_ts = pnl_ts.sort_values("timestamp")
            pnl_ts["cum_pnl"] = pnl_ts["pnl"].cumsum()

            fig_pnl = px.area(
                pnl_ts, x="timestamp", y="cum_pnl",
                labels={"cum_pnl": "Cumulative P&L ($)", "timestamp": ""},
                title="Cumulative P&L",
                color_discrete_sequence=["#00c851"],
            )
            fig_pnl.update_layout(
                height=250,
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_pnl.update_xaxes(showgrid=False)
            fig_pnl.update_yaxes(gridcolor="#222")
            st.plotly_chart(fig_pnl, use_container_width=True)

        # ── Trade table ────────────────────────────────────────────────────────
        show_trade_cols = [c for c in [
            "timestamp", "symbol", "action", "qty", "price", "pnl",
            "stop", "reason", "order_id", "regime",
        ] if c in filtered.columns]

        display_trades = filtered[show_trade_cols].copy().sort_values(
            "timestamp", ascending=False
        )
        if "timestamp" in display_trades.columns:
            display_trades["timestamp"] = display_trades["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        if "pnl" in display_trades.columns:
            display_trades["pnl"] = display_trades["pnl"].apply(
                lambda x: f"${float(x):+,.2f}" if pd.notna(x) else ""
            )
        if "price" in display_trades.columns:
            display_trades["price"] = display_trades["price"].apply(
                lambda x: f"${float(x):,.4f}" if pd.notna(x) else ""
            )
        if "stop" in display_trades.columns:
            display_trades["stop"] = display_trades["stop"].apply(
                lambda x: f"${float(x):,.2f}" if pd.notna(x) else ""
            )

        st.dataframe(display_trades, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REGIME HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_regime:
    st.markdown("### Regime History")

    # ── Backfill controls ──────────────────────────────────────────────────────
    _bf_col1, _bf_col2, _bf_col3, _ = st.columns([1, 1, 2, 2])
    with _bf_col1:
        _bf_years = st.selectbox("History", [3, 5, 7, 10], index=1,
                                 label_visibility="collapsed")
    with _bf_col2:
        _do_backfill = st.button("Run Full Backfill", use_container_width=True,
                                 help="Fit HMM on SPY + VIX + Yield Curve + DXY and classify every trading day")
    with _bf_col3:
        _bf_ts = st.session_state.get("regime_backfill_ts")
        if _bf_ts:
            _bf_ts_str = datetime.fromisoformat(
                _bf_ts).strftime("%Y-%m-%d %H:%M UTC")
            _bf_yrs = st.session_state.get("regime_backfill_years", "?")
            st.markdown(
                f'<div style="padding:6px 0;font-size:0.72rem;font-family:monospace;color:#AAAAAA">'
                f'● backfill computed {_bf_ts_str} · {_bf_yrs}y · macro features ON</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding:6px 0;font-size:0.72rem;font-family:monospace;color:#AAAAAA">'
                '● no backfill yet — hit Run Full Backfill for complete history</div>',
                unsafe_allow_html=True,
            )

    if _do_backfill:
        with st.spinner(f"Fitting HMM on {_bf_years}y SPY + macro data …"):
            _bf_result = _run_regime_backfill(years=_bf_years)
        if _bf_result is None:
            st.error(
                f"Backfill failed: {st.session_state.get('regime_backfill_error', 'unknown error')}")
        else:
            st.success(
                f"Backfill complete — {len(_bf_result):,} bars classified")
            st.rerun()

    # Priority: backfill > regime.log
    _bf_df = st.session_state.get("regime_backfill_df")
    if _bf_df is not None and not _bf_df.empty:
        df_reg = _bf_df.copy()
        # Ensure timestamp column is datetime
        if "timestamp" in df_reg.columns and not pd.api.types.is_datetime64_any_dtype(df_reg["timestamp"]):
            df_reg["timestamp"] = pd.to_datetime(df_reg["timestamp"])
        _data_source_label = "backfill (HMM · SPY + VIX + Yield Curve + DXY)"
    else:
        df_reg = _read_ndjson(_LOG_DIR / "regime.log", tail=5000)
        _data_source_label = "regime.log (live engine)"

    if df_reg.empty:
        st.info(
            "No regime data yet — hit **Run Full Backfill** above to generate history from SPY data.")
    else:
        st.caption(f"Data source: {_data_source_label} · {len(df_reg):,} bars")

        # ── Transition events ──────────────────────────────────────────────────
        # Prefer the explicit `transition` boolean column; fall back to comparing
        # old_regime vs new_regime (handles legacy records that predate the column).
        if "transition" in df_reg.columns:
            _tr_mask = df_reg["transition"].apply(
                lambda x: bool(x) if x is not None and x == x else False
            )
            transitions = df_reg[_tr_mask].copy()
            # If `transition` column exists but all False (old logs written before the fix),
            # fall back to string comparison so the tab still shows historical data.
            if transitions.empty and all(c in df_reg.columns for c in ["old_regime", "new_regime"]):
                transitions = df_reg[df_reg["old_regime"]
                                     != df_reg["new_regime"]].copy()
        elif all(c in df_reg.columns for c in ["old_regime", "new_regime"]):
            transitions = df_reg[df_reg["old_regime"]
                                 != df_reg["new_regime"]].copy()
        else:
            transitions = pd.DataFrame()

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("Total Records", len(df_reg))
        rm2.metric("Regime Changes", len(transitions))

        if "confidence" in df_reg.columns:
            avg_conf = df_reg["confidence"].mean()
            rm3.metric("Avg Confidence", fmt_pct(avg_conf, plus=False))

        st.divider()

        # ── Regime probability timeline ────────────────────────────────────────
        prob_cols = [c for c in df_reg.columns if c in
                     ["Bull", "Bear", "Neutral", "Euphoria", "Panic", "Crash", "Mania"]]

        if prob_cols and "timestamp" in df_reg.columns:
            _section_bar("Probability Timeline")
            prob_df = df_reg[["timestamp"] + prob_cols].copy()
            # Fill missing regime columns with 0 (e.g. Bear absent when pure Neutral/Bull run)
            for _pc in prob_cols:
                prob_df[_pc] = pd.to_numeric(
                    prob_df[_pc], errors="coerce").fillna(0.0)
            prob_df = prob_df.dropna(
                subset=["timestamp"]).sort_values("timestamp")

            fig_probs = go.Figure()
            for col in prob_cols:
                fig_probs.add_trace(go.Scatter(
                    x=prob_df["timestamp"], y=prob_df[col],
                    name=col, mode="lines",
                    line=dict(color=regime_color(col), width=2),
                    stackgroup="one",
                    groupnorm="fraction",
                ))
            fig_probs.update_layout(**_PLOTLY_LAYOUT)
            fig_probs.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(tickformat=".0%", gridcolor="#1a1a1a"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_probs, use_container_width=True)

        # ── Macro overlay (VIX / Yield Curve / DXY) ───────────────────────────
        _macro_ov = st.session_state.get("regime_backfill_macro")
        if _macro_ov is not None and not _macro_ov.empty and "timestamp" in df_reg.columns:
            _section_bar("Macro Context Overlay")
            _m_ts = pd.to_datetime(df_reg["timestamp"])
            _m_start, _m_end = _m_ts.min(), _m_ts.max()
            _mo = _macro_ov[(_macro_ov.index >= _m_start) &
                            (_macro_ov.index <= _m_end)].copy()
            if not _mo.empty:
                _mc1, _mc2, _mc3 = st.columns(3)
                with _mc1:
                    if "vix" in _mo.columns:
                        _fig_vix = go.Figure(go.Scatter(
                            x=_mo.index, y=_mo["vix"], mode="lines",
                            line=dict(color="#ff8800", width=1.5), name="VIX",
                        ))
                        _fig_vix.add_hline(y=20, line_dash="dash", line_color="#AAAAAA",
                                           annotation_text="20 (avg)")
                        _fig_vix.add_hline(y=30, line_dash="dash", line_color="#ff4444",
                                           annotation_text="30 (fear)")
                        _fig_vix.update_layout(
                            title=dict(text="VIX (Fear)", font=dict(size=11)),
                            height=180, margin=dict(t=30, b=5, l=5, r=5),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(gridcolor="#1a1a1a"), xaxis=dict(showgrid=False),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_vix, use_container_width=True)
                with _mc2:
                    if "tnx" in _mo.columns and "irx" in _mo.columns:
                        _yc = _mo["tnx"] - _mo["irx"]
                        _yc_clr = "#ff4444" if float(
                            _yc.iloc[-1]) < 0 else "#00c851"
                        _fig_yc = go.Figure(go.Scatter(
                            x=_yc.index, y=_yc, mode="lines",
                            line=dict(color=_yc_clr, width=1.5), name="10Y−3M",
                        ))
                        _fig_yc.add_hline(y=0, line_dash="dash", line_color="#ff4444",
                                          annotation_text="Inversion")
                        _fig_yc.update_layout(
                            title=dict(text="Yield Curve (10Y−3M %)",
                                       font=dict(size=11)),
                            height=180, margin=dict(t=30, b=5, l=5, r=5),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(gridcolor="#1a1a1a"), xaxis=dict(showgrid=False),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_yc, use_container_width=True)
                with _mc3:
                    if "dxy" in _mo.columns:
                        _fig_dxy = go.Figure(go.Scatter(
                            x=_mo.index, y=_mo["dxy"], mode="lines",
                            line=dict(color="#9e9e9e", width=1.5), name="DXY",
                        ))
                        _fig_dxy.add_hline(y=100, line_dash="dash", line_color="#AAAAAA",
                                           annotation_text="Par (100)")
                        _fig_dxy.update_layout(
                            title=dict(text="DXY (USD Index)",
                                       font=dict(size=11)),
                            height=180, margin=dict(t=30, b=5, l=5, r=5),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(gridcolor="#1a1a1a"), xaxis=dict(showgrid=False),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_dxy, use_container_width=True)

        # ── Confidence over time ───────────────────────────────────────────────
        if "confidence" in df_reg.columns and "timestamp" in df_reg.columns:
            _section_bar("HMM Confidence")
            conf_df = df_reg[["timestamp", "confidence",
                              "new_regime" if "new_regime" in df_reg.columns else "regime"
                              ]].dropna().copy()
            conf_df = conf_df.sort_values("timestamp")
            regime_col_name = "new_regime" if "new_regime" in conf_df.columns else "regime"

            fig_conf = px.scatter(
                conf_df,
                x="timestamp", y="confidence",
                color=regime_col_name,
                color_discrete_map=_REGIME_COLORS,
                opacity=0.7,
                labels={"confidence": "Confidence", "timestamp": ""},
            )
            fig_conf.add_hline(y=0.45, line_dash="dash", line_color="#ff8800",
                               annotation_text="Low confidence threshold")
            fig_conf.update_layout(
                height=250,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#222"),
                xaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # ── Transition table ───────────────────────────────────────────────────
        if not transitions.empty:
            _section_bar("Regime Transitions")
            trans_cols = [c for c in [
                "timestamp", "old_regime", "new_regime", "confidence",
                "stability_bars", "flicker_count",
            ] if c in transitions.columns]
            trans_display = transitions[trans_cols].sort_values(
                "timestamp", ascending=False
            ).copy()
            if "timestamp" in trans_display.columns:
                trans_display["timestamp"] = trans_display["timestamp"].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            if "confidence" in trans_display.columns:
                trans_display["confidence"] = trans_display["confidence"].apply(
                    lambda x: f"{float(x):.1%}" if pd.notna(x) else ""
                )
            st.dataframe(trans_display, use_container_width=True,
                         hide_index=True)

        # ── Regime distribution bar ────────────────────────────────────────────
        regime_label_col = "new_regime" if "new_regime" in df_reg.columns else (
            "regime" if "regime" in df_reg.columns else None
        )
        if regime_label_col:
            _section_bar("Time in Each Regime")
            _vc = (
                df_reg[regime_label_col]
                .dropna()
                .astype(str)
                .replace("nan", pd.NA)
                .dropna()
                .value_counts()
                .reset_index()
            )
            # pandas ≥2.0 names the count column "count"; older versions use position
            _vc.columns = ["Regime", "Bars"]
            # only known regimes
            _vc = _vc[_vc["Regime"].isin(_REGIME_COLORS)]

            if not _vc.empty:
                fig_bar = px.bar(
                    _vc, x="Regime", y="Bars",
                    color="Regime",
                    color_discrete_map=_REGIME_COLORS,
                    labels={"Bars": "Trading Days"},
                )
                fig_bar.update_layout(
                    height=220,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    yaxis=dict(gridcolor="#222"),
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.caption("No known regime labels found in dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PORTFOLIO SYNC
# ══════════════════════════════════════════════════════════════════════════════

with tab_sync:
    st.markdown("### Portfolio Sync")
    st.caption(
        "Upload a brokerage statement CSV and compare it against your current "
        "Alpaca positions.  Use **--execute** to submit market orders for any shortfall."
    )

    uploaded = st.file_uploader(
        "Upload brokerage statement CSV",
        type=["csv"],
        help="Expects columns: Ticker, Type, Quantity, Currency — as exported by IBKR/Degiro/etc.",
    )

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(raw_df):,} rows.")

            with st.expander("Preview raw CSV"):
                st.dataframe(raw_df.head(30), use_container_width=True)

            # ── Compute net holdings (replicate portfolio_sync.py logic) ──────
            required_cols = {"Ticker", "Type", "Quantity", "Currency"}
            if not required_cols.issubset(raw_df.columns):
                st.error(
                    f"Missing columns: {required_cols - set(raw_df.columns)}. "
                    "Expected: Ticker, Type, Quantity, Currency."
                )
            else:
                usd = raw_df[raw_df["Currency"].str.upper() == "USD"].copy()
                type_upper = usd["Type"].str.upper().str.strip()

                buy_mask = type_upper.str.startswith("BUY")
                sell_mask = type_upper.str.startswith("SELL")
                merger_stock_mask = type_upper == "MERGER - STOCK"
                trade_rows = usd[buy_mask | sell_mask |
                                 merger_stock_mask].copy()

                trade_rows["signed_qty"] = 0.0
                trade_rows.loc[trade_rows.index[buy_mask[trade_rows.index]],
                               "signed_qty"] = trade_rows.loc[
                    trade_rows.index[buy_mask[trade_rows.index]], "Quantity"]
                trade_rows.loc[trade_rows.index[sell_mask[trade_rows.index]],
                               "signed_qty"] = -trade_rows.loc[
                    trade_rows.index[sell_mask[trade_rows.index]], "Quantity"]
                trade_rows.loc[trade_rows.index[merger_stock_mask[trade_rows.index]],
                               "signed_qty"] = trade_rows.loc[
                    trade_rows.index[merger_stock_mask[trade_rows.index]], "Quantity"]

                net: Dict[str, float] = (
                    trade_rows.dropna(subset=["signed_qty"])
                    .groupby("Ticker")["signed_qty"]
                    .sum()
                    .to_dict()
                )
                net_holdings = {t: q for t, q in net.items() if q > 0.001}

                _section_bar("Net Holdings from CSV")
                net_df = pd.DataFrame(
                    [{"Symbol": k, "Net Qty (CSV)": round(v, 4)}
                     for k, v in sorted(net_holdings.items())]
                )
                st.dataframe(net_df, use_container_width=True, hide_index=True)

                # ── Compare vs Alpaca ──────────────────────────────────────────
                _section_bar("Diff vs Alpaca")

                alpaca_data = _get_alpaca_portfolio()
                if alpaca_data:
                    alpaca_pos = {
                        p["symbol"]: float(p["qty"])
                        for p in alpaca_data["positions"]
                    }
                    rows = []
                    all_syms = set(net_holdings) | set(alpaca_pos)
                    for sym in sorted(all_syms):
                        csv_qty = net_holdings.get(sym, 0.0)
                        alpaca_qty = alpaca_pos.get(sym, 0.0)
                        delta = csv_qty - alpaca_qty
                        status = "✅ OK" if abs(delta) < 0.001 else (
                            "🟢 BUY" if delta > 0 else "🔴 SELL")
                        rows.append({
                            "Symbol":    sym,
                            "CSV Qty":   round(csv_qty, 4),
                            "Alpaca Qty": round(alpaca_qty, 4),
                            "Delta":     round(delta, 4),
                            "Action":    status,
                        })
                    diff_df = pd.DataFrame(rows)
                    st.dataframe(
                        diff_df, use_container_width=True, hide_index=True)

                    buys_needed = [r for r in rows if r["Delta"] > 0.001]

                    if buys_needed:
                        n_buys = len(buys_needed)
                        st.warning(
                            f"{n_buys} symbol(s) need buying. "
                            "Use the button below to execute via portfolio_sync.py."
                        )

                        ex_col, _ = st.columns([2, 5])
                        with ex_col:
                            if st.button(
                                "⚡ Execute sync (paper)", type="primary",
                                use_container_width=True,
                            ):
                                import tempfile
                                with tempfile.NamedTemporaryFile(
                                    suffix=".csv", delete=False, mode="wb"
                                ) as tmp:
                                    uploaded.seek(0)
                                    tmp.write(uploaded.read())
                                    tmp_path = tmp.name

                                with st.spinner("Running portfolio_sync.py --execute …"):
                                    res = subprocess.run(
                                        [sys.executable,
                                         str(_HERE / "portfolio_sync.py"),
                                         tmp_path, "--execute"],
                                        capture_output=True, text=True,
                                        timeout=60, cwd=str(_HERE),
                                    )
                                Path(tmp_path).unlink(missing_ok=True)

                                if res.returncode == 0:
                                    st.success("Sync executed.")
                                    st.code(
                                        res.stdout[-3000:], language="text")
                                else:
                                    st.error("Sync failed.")
                                    st.code(
                                        res.stderr[-3000:], language="text")
                    else:
                        st.success(
                            "Portfolio is already in sync with the CSV. Nothing to buy.")
                else:
                    st.info(
                        "Alpaca API not connected. "
                        "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in `.env` "
                        "to enable the live diff and execute buttons."
                    )

                    # Still let the user run portfolio_sync.py directly
                    sync_col, _ = st.columns([2, 5])
                    with sync_col:
                        if st.button(
                            "▶ Run dry-run sync (no API key)", use_container_width=True
                        ):
                            import tempfile
                            with tempfile.NamedTemporaryFile(
                                suffix=".csv", delete=False, mode="wb"
                            ) as tmp:
                                uploaded.seek(0)
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name

                            with st.spinner("Running portfolio_sync.py dry-run …"):
                                res = subprocess.run(
                                    [sys.executable,
                                     str(_HERE / "portfolio_sync.py"), tmp_path],
                                    capture_output=True, text=True,
                                    timeout=60, cwd=str(_HERE),
                                )
                            Path(tmp_path).unlink(missing_ok=True)
                            st.code(res.stdout + "\n" +
                                    res.stderr, language="text")

        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MACRO COMMAND CENTER
# ══════════════════════════════════════════════════════════════════════════════

with tab_macro:
    # ── Lazy import of the macro engine ──────────────────────────────────────
    _MACRO_AVAILABLE = True

    if _MACRO_AVAILABLE:
        # ── Cached data-fetch functions ───────────────────────────────────────

        @st.cache_data(ttl=3600)
        def _load_all_commodity_prices() -> Dict[str, Any]:
            """Fetch all 9 commodity price dicts keyed by futures ticker."""
            out: Dict[str, Any] = {}
            for comm in COMMODITY_UNIVERSE:
                data = fetch_commodity_prices(comm)
                out[comm["ticker"]] = data
            return out

        @st.cache_data(ttl=3600)
        def _load_macro_indicators() -> Dict[str, Any]:
            """Fetch TNX, DXY, VIX."""
            out: Dict[str, Any] = {}
            for ind in MACRO_INDICATORS:
                out[ind["ticker"]] = fetch_macro_indicator(ind["ticker"])
            return out

        # ── Refresh button ────────────────────────────────────────────────────
        _macro_hdr_col, _macro_btn_col = st.columns([7, 1])
        with _macro_hdr_col:
            st.markdown(
                "<h4 style='margin:0;color:#00c851;font-family:monospace;'>"
                "GLOBAL MACRO & COMMODITIES — INSTITUTIONAL FRAMEWORK</h4>",
                unsafe_allow_html=True,
            )
        with _macro_btn_col:
            if st.button("Refresh", use_container_width=True,
                         help="Clear cache and re-fetch all macro data"):
                st.cache_data.clear()
                st.rerun()

        st.markdown("<hr style='border-color:#1a1a1a;margin:4px 0 10px 0;'>",
                    unsafe_allow_html=True)

        # ── Load data ─────────────────────────────────────────────────────────
        with st.spinner("Fetching macro data…"):
            _prices = _load_all_commodity_prices()
            _indicators = _load_macro_indicators()

        # Build conviction scores (no caching — fast, pure computation)
        # placeholder; no live sentiment here
        _sentiment_map: Dict[str, float] = {}
        _convictions: Dict[str, Dict] = {}
        for _comm in COMMODITY_UNIVERSE:
            _tk = _comm["ticker"]
            if _prices.get(_tk):
                _convictions[_tk] = calc_macro_conviction(
                    _prices[_tk], _sentiment_map)

        # ── SECTION 00 — GEOPOLITICAL RISK GAUGE ─────────────────────────────
        _section_bar("00 · Geopolitical Risk",
                     "War · Inflation · Safe-Haven Flows")
        _geo = _compute_geo_risk_score()
        _geo_score = _geo["score"]
        _geo_color = _geo["color"]
        _geo_label = _geo["label"]
        _geo_c1, _geo_c2, _geo_c3, _geo_c4 = st.columns([1, 1, 1, 2])
        with _geo_c1:
            _vc_clr = "#ff4444" if _geo["vix_comp"] >= 0.65 else "#ff8800" if _geo["vix_comp"] >= 0.40 else "#00c851"
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;border-radius:6px;padding:10px 12px;">'
                f'<div style="font-size:0.55rem;color:#AAAAAA;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px">VIX Level</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{_vc_clr};font-family:monospace">{_geo["vix"]:.1f}</div>'
                f'<div style="height:3px;background:#222;border-radius:2px;margin-top:5px">'
                f'<div style="width:{_geo["vix_comp"]*100:.0f}%;height:3px;background:{_vc_clr};border-radius:2px"></div></div>'
                f'<div style="font-size:0.58rem;color:#AAAAAA;margin-top:3px">'
                f'{"⚠ Elevated" if _geo["vix_comp"] >= 0.40 else "Normal"} · {_geo["vix_comp"]*100:.0f}% risk contribution</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with _geo_c2:
            _gc_clr = "#ff4444" if _geo["gold_comp"] >= 0.65 else "#ff8800" if _geo["gold_comp"] >= 0.50 else "#00c851"
            _gc_sign = "+" if _geo["gold_20d"] >= 0 else ""
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;border-radius:6px;padding:10px 12px;">'
                f'<div style="font-size:0.55rem;color:#AAAAAA;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px">Gold 20d</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{_gc_clr};font-family:monospace">{_gc_sign}{_geo["gold_20d"]:.1f}%</div>'
                f'<div style="height:3px;background:#222;border-radius:2px;margin-top:5px">'
                f'<div style="width:{_geo["gold_comp"]*100:.0f}%;height:3px;background:{_gc_clr};border-radius:2px"></div></div>'
                f'<div style="font-size:0.58rem;color:#AAAAAA;margin-top:3px">'
                f'{"⚠ Flight-to-safety" if _geo["gold_comp"] >= 0.55 else "Neutral"} · {_geo["gold_comp"]*100:.0f}% risk contribution</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with _geo_c3:
            _oc_clr = "#ff4444" if _geo["oil_comp"] >= 0.65 else "#ff8800" if _geo["oil_comp"] >= 0.50 else "#00c851"
            _oc_sign = "+" if _geo["oil_20d"] >= 0 else ""
            st.markdown(
                f'<div style="background:#0a0a0a;border:1px solid #1e1e1e;border-radius:6px;padding:10px 12px;">'
                f'<div style="font-size:0.55rem;color:#AAAAAA;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px">Oil 20d</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{_oc_clr};font-family:monospace">{_oc_sign}{_geo["oil_20d"]:.1f}%</div>'
                f'<div style="height:3px;background:#222;border-radius:2px;margin-top:5px">'
                f'<div style="width:{_geo["oil_comp"]*100:.0f}%;height:3px;background:{_oc_clr};border-radius:2px"></div></div>'
                f'<div style="font-size:0.58rem;color:#AAAAAA;margin-top:3px">'
                f'{"⚠ Supply risk" if _geo["oil_comp"] >= 0.55 else "Neutral"} · {_geo["oil_comp"]*100:.0f}% risk contribution</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with _geo_c4:
            _rotation_hint = ""
            if _geo_score >= 0.65:
                _rotation_hint = "CONSIDER ROTATING → RTX · LMT · GLD · XLE as geopolitical hedge"
            elif _geo_score >= 0.45:
                _rotation_hint = "Monitor RTX/LMT for entry. Ensure gold allocation > 5% of portfolio."
            else:
                _rotation_hint = "No defensive rotation required at current risk level."
            st.markdown(
                f'<div style="background:{_geo_color}0d;border:1px solid {_geo_color}44;border-radius:6px;padding:10px 14px;height:100%;">'
                f'<div style="font-size:0.58rem;color:{_geo_color};font-weight:900;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:6px">'
                f'Composite Score: {_geo_score:.0%}</div>'
                f'<div style="font-size:1.35rem;font-weight:900;color:{_geo_color};font-family:monospace;margin-bottom:4px">{_geo_label}</div>'
                f'<div style="height:4px;background:#222;border-radius:2px;margin-bottom:6px">'
                f'<div style="width:{_geo_score*100:.0f}%;height:4px;background:{_geo_color};border-radius:2px"></div></div>'
                f'<div style="font-size:0.65rem;color:#888;line-height:1.4">{_rotation_hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── MACRO SHOCK ALERTS ────────────────────────────────────────────────
        _shocks = check_macro_shocks(_prices)
        if _shocks:
            _section_bar("⚡ MACRO SHOCK ALERTS", str(len(_shocks)))
            for _shock in _shocks:
                if _shock["level"] == "error":
                    st.error(f"{_shock['icon']}  {_shock['message']}")
                else:
                    st.warning(f"{_shock['icon']}  {_shock['message']}")

        # ── SECTION 01 — MACRO GAUGES ROW ─────────────────────────────────────
        _section_bar("01 · Macro Gauges",
                     "TNX · DXY · VIX · Cu/Au · Yield Curve")

        _g1, _g2, _g3, _g4, _g5 = st.columns(5)

        def _delta_str(val: float, fmt: str = "+.2f") -> str:
            """Format a delta value for st.metric."""
            return f"{val:{fmt}}"

        # Cu/Au ratio
        _gold_d = _prices.get("GC=F") or {}
        _copper_d = _prices.get("HG=F") or {}
        _cu_price = _copper_d.get("price", 0.0)
        _au_price = _gold_d.get("price", 1.0)
        _cu_au = round(_cu_price / _au_price * 1000,
                       3) if _au_price > 0 else 0.0
        _cu_au_5d_delta = round(
            (_copper_d.get("ret_5d", 0.0) - _gold_d.get("ret_5d", 0.0)) * 100, 2
        )
        _cu_au_signal = (
            "Risk-On" if _cu_au_5d_delta > 0.5
            else "Defensive" if _cu_au_5d_delta < -0.5
            else "Neutral"
        )
        with _g1:
            st.metric(
                label="Cu/Au Ratio (×1000)",
                value=f"{_cu_au:.3f}",
                delta=f"{_cu_au_5d_delta:+.2f}pp (5d)",
                help="Copper/Gold × 1000. Rising = risk-on / economic expansion. Falling = defensive / recession signal.",
            )
            _cu_au_color = "#00c851" if _cu_au_5d_delta > 0 else "#ff4444"
            st.markdown(
                f"<p style='font-size:0.65rem;color:{_cu_au_color};margin:0;'>"
                f"Dr. Copper Signal: <b>{_cu_au_signal}</b></p>",
                unsafe_allow_html=True,
            )

        # TNX
        _tnx_d = _indicators.get("^TNX") or {}
        _tnx_v = _tnx_d.get("price", 0.0)
        _tnx_1d = _tnx_d.get("ret_1d", 0.0)
        _tnx_regime = "Restrictive" if _tnx_v > 4.5 else "Neutral" if _tnx_v > 3.5 else "Accommodative"
        with _g2:
            st.metric(
                label="US 10Y Yield",
                value=f"{_tnx_v:.2f}%",
                delta=f"{_tnx_1d*100:+.1f}bp (1d)",
                help="US 10-year Treasury yield. >4.5% = restrictive for equities/commodities.",
            )
            _tnx_color = "#ff4444" if _tnx_v > 4.5 else "#ff8800" if _tnx_v > 3.5 else "#00c851"
            st.markdown(
                f"<p style='font-size:0.65rem;color:{_tnx_color};margin:0;'>"
                f"Rate Regime: <b>{_tnx_regime}</b></p>",
                unsafe_allow_html=True,
            )

        # DXY
        _dxy_d = _indicators.get("DX-Y.NYB") or {}
        _dxy_v = _dxy_d.get("price", 0.0)
        _dxy_5d = _dxy_d.get("ret_5d", 0.0)
        _dxy_regime = "USD Strong" if _dxy_5d > 0.01 else "USD Weak" if _dxy_5d < - \
            0.01 else "USD Stable"
        with _g3:
            st.metric(
                label="Dollar Index (DXY)",
                value=f"{_dxy_v:.1f}",
                delta=f"{_dxy_5d*100:+.2f}% (5d)",
                help="Strong USD = headwind for commodities. Weak USD = commodity tailwind.",
            )
            _dxy_color = "#ff8800" if _dxy_5d > 0.01 else "#00c851" if _dxy_5d < -0.01 else "#9e9e9e"
            st.markdown(
                f"<p style='font-size:0.65rem;color:{_dxy_color};margin:0;'>"
                f"<b>{_dxy_regime}</b></p>",
                unsafe_allow_html=True,
            )

        # VIX
        _vix_d = _indicators.get("^VIX") or {}
        _vix_v = _vix_d.get("price", 0.0)
        _vix_1d = _vix_d.get("ret_1d", 0.0)
        _vix_regime = "Extreme Fear" if _vix_v > 30 else "Elevated" if _vix_v > 20 else "Calm"
        with _g4:
            st.metric(
                label="VIX",
                value=f"{_vix_v:.1f}",
                delta=f"{_vix_1d*100:+.1f}% (1d)",
                help="Equity volatility index. >30 = extreme fear / tail risk environment.",
            )
            _vix_color = "#ff4444" if _vix_v > 30 else "#ff8800" if _vix_v > 20 else "#00c851"
            st.markdown(
                f"<p style='font-size:0.65rem;color:{_vix_color};margin:0;'>"
                f"Vol Regime: <b>{_vix_regime}</b></p>",
                unsafe_allow_html=True,
            )

        # Yield Curve (10Y − 3M)
        _irx_d = _indicators.get("^IRX") or {}
        _irx_v = _irx_d.get("price", 0.0)
        _yc_spread = round(_tnx_v - _irx_v, 2)
        if _yc_spread < -0.5:
            _yc_label = "INVERTED — RECESSION WATCH"
            _yc_color = "#ff4444"
        elif _yc_spread < 0.0:
            _yc_label = "FLATTENING"
            _yc_color = "#ff8800"
        elif _yc_spread < 0.5:
            _yc_label = "FLAT"
            _yc_color = "#ffbb33"
        else:
            _yc_label = "NORMAL CURVE"
            _yc_color = "#00c851"
        with _g5:
            st.metric(
                label="Yield Curve (10Y−3M)",
                value=f"{_yc_spread:+.2f}%",
                delta=f"{_irx_v:.2f}% 3M",
                help=(
                    "10Y minus 3M Treasury yield spread. "
                    "Inversion (< 0%) precedes recession by avg 12–18 months. "
                    "Deep inversion < −0.5% = high alert."
                ),
            )
            st.markdown(
                f"<p style='font-size:0.65rem;color:{_yc_color};margin:0;font-weight:700'>"
                f"{_yc_label}</p>"
                f"<p style='font-size:0.58rem;color:#AAAAAA;margin:1px 0 0 0'>"
                f"Inversion precedes recession 12–18 months avg</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── SECTION 02 — COMMODITY HEATMAP GRID ───────────────────────────────
        _section_bar("02 · Commodity Universe",
                     "Energy · Metals · Agriculture")

        # Color helpers
        def _ret_color(r: float) -> str:
            if r >= 0.03:
                return "#00c851"
            if r >= 0.01:
                return "#7cb342"
            if r >= -0.01:
                return "#9e9e9e"
            if r >= -0.03:
                return "#ff8800"
            return "#ff4444"

        def _score_color(s: float) -> str:
            if s >= 0.72:
                return "#00c851"
            if s >= 0.58:
                return "#7cb342"
            if s >= 0.42:
                return "#9e9e9e"
            if s >= 0.28:
                return "#ff8800"
            return "#ff4444"

        for _sector_name in ["Energy", "Metals", "Agriculture"]:
            _sector_comms = [
                c for c in COMMODITY_UNIVERSE if c["sector"] == _sector_name]
            st.markdown(
                f"<p style='font-size:0.62rem;font-weight:900;letter-spacing:0.12em;"
                f"text-transform:uppercase;color:#AAAAAA;margin:8px 0 4px 0;'>"
                f"{_sector_name}</p>",
                unsafe_allow_html=True,
            )
            _cols = st.columns(len(_sector_comms))
            for _ci, _comm in enumerate(_sector_comms):
                _tk = _comm["ticker"]
                _pd = _prices.get(_tk) or {}
                _cv = _convictions.get(_tk, {})
                with _cols[_ci]:
                    _price_val = _pd.get("price", 0.0)
                    _r1 = _pd.get("ret_1d",  0.0)
                    _r5 = _pd.get("ret_5d",  0.0)
                    _r20 = _pd.get("ret_20d", 0.0)
                    _rsi = _pd.get("rsi14",   50.0)
                    _pct = _pd.get("pct_52",  0.5)
                    _comp = _cv.get("composite", 0.5)
                    _cv_lbl = _cv.get("conviction_label", "—")
                    _cv_clr = _cv.get("conviction_clr", "#9e9e9e")
                    _source = _pd.get("source", "—")

                    st.markdown(
                        f"""<div style="background:#0d0d0d;border:1px solid #1e1e1e;
                            border-radius:6px;padding:10px 12px;margin-bottom:6px;">
                          <div style="font-size:0.6rem;color:#AAAAAA;font-family:monospace;
                               margin-bottom:2px;">{_tk} · {_source}</div>
                          <div style="font-size:0.85rem;font-weight:700;color:#e0e0e0;
                               font-family:monospace;">{_comm['name']}</div>
                          <div style="font-size:1.1rem;font-weight:900;color:#e0e0e0;
                               font-family:monospace;margin:4px 0;">
                            {"N/A" if _price_val == 0 else f"{_price_val:,.4g}"} <span style="font-size:0.6rem;color:#AAAAAA;">{_comm['unit']}</span>
                          </div>
                          <div style="display:flex;gap:6px;font-size:0.62rem;font-family:monospace;margin-bottom:6px;">
                            <span style="color:{_ret_color(_r1)}">{_r1:+.1%} 1d</span>
                            <span style="color:{_ret_color(_r5)}">{_r5:+.1%} 5d</span>
                            <span style="color:{_ret_color(_r20)}">{_r20:+.1%} 20d</span>
                          </div>
                          <div style="font-size:0.6rem;color:#AAAAAA;margin-bottom:4px;font-family:monospace;">
                            RSI {_rsi:.0f} · 52w%ile {_pct:.0%}
                          </div>
                          <div style="font-size:0.68rem;font-weight:700;color:{_cv_clr};
                               font-family:monospace;padding:2px 0;border-top:1px solid #1e1e1e;
                               margin-top:4px;">
                            {_cv_lbl} &nbsp;<span style="font-size:0.6rem;color:#AAAAAA;">{_comp:.2f}</span>
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── SECTION 03 — TOP CONVICTION STOCK PICKS ───────────────────────────
        _section_bar("03 · Top Conviction Trades",
                     "Triple-Filter: Macro + Fundamental + HMM Regime")

        @st.cache_data(ttl=3600)
        def _load_sector_picks(
            category: str,
            prices_key: str,
            regime_lbl: str,
            institutional_tuple: tuple,
        ) -> Dict:
            """Cached wrapper — keyed by category + prices hash + regime + institutional data."""
            import inspect as _insp
            _institutional_scores = dict(institutional_tuple)
            _sig = _insp.signature(get_top_sector_picks).parameters
            _kwargs: Dict = {"regime_lbl": regime_lbl}
            if "institutional_scores" in _sig:
                _kwargs["institutional_scores"] = _institutional_scores
            return get_top_sector_picks(category, _prices, _convictions, **_kwargs)

        # Build a hashable key from conviction composites so cache invalidates on refresh
        _prices_hash = str(sorted(
            (k, round(v.get("composite", 0), 2))
            for k, v in _convictions.items()
        ))

        # Fetch regime label and institutional scores for the triple-filter
        _regime_for_macro = _get_regime_state().get("label", "Neutral")
        _institutional_for_macro, _, _ = _fetch_institutional_live()
        _institutional_tuple = tuple(sorted(_institutional_for_macro.items()))

        _sector_icons = {"Energy": "⚡", "Metals": "🥇", "Agriculture": "🌾"}
        _sector_comm_label = {
            "Energy":      "Crude Oil & Brent",
            "Metals":      "Gold & Copper",
            "Agriculture": "Wheat & Corn",
        }

        # Regime badge color
        _regime_colors = {
            "Mania": "#ff6b6b", "Euphoria": "#00c851", "Bull": "#4fc3f7",
            "Neutral": "#9e9e9e", "Bear": "#ff9800", "Panic": "#ff4444", "Crash": "#d32f2f",
        }
        _regime_badge_clr = _regime_colors.get(_regime_for_macro, "#9e9e9e")

        for _cat in ["Energy", "Metals", "Agriculture"]:
            _picks_data = _load_sector_picks(
                _cat, _prices_hash, _regime_for_macro, _institutional_tuple)
            _m_score = _picks_data["macro_score"]
            _m_ok = _picks_data["macro_ok"]
            _picks = _picks_data["picks"]
            _icon = _sector_icons.get(_cat, "")
            _regime_mult_v = _picks_data.get("regime_mult", 1.0)

            # Sector sub-header with macro + regime badges
            _m_clr = "#00c851" if _m_ok else "#ff4444"
            _macro_badge = "✓ MACRO PASS" if _m_ok else "✗ MACRO BLOCK"
            _macro_score_s = f"({_m_score:.0%})"
            _regime_mult_s = f"×{_regime_mult_v:.2f}"
            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "margin:14px 0 8px 0;padding:8px 12px;"
                "background:#0a0a0a;border:1px solid #1e1e1e;border-radius:6px;'>"
                "<span style='font-size:1.2rem'>" + _icon + "</span>"
                "<span style='font-size:0.75rem;font-weight:900;letter-spacing:0.12em;"
                "text-transform:uppercase;color:#ccc;font-family:monospace'>" + _cat + "</span>"
                "<span style='font-size:0.65rem;color:#AAAAAA;font-family:monospace'>"
                "— " + _sector_comm_label.get(_cat, "") +
                " macro filter</span>"
                "<span style='margin-left:auto;display:flex;align-items:center;gap:8px;'>"
                "<span style='font-size:0.68rem;font-weight:700;color:" +
                _m_clr + ";font-family:monospace'>"
                + _macro_badge + " " + _macro_score_s +
                "</span>"
                "<span style='font-size:0.65rem;font-weight:700;color:" + _regime_badge_clr + ";"
                "font-family:monospace;padding:1px 6px;border:1px solid " + _regime_badge_clr + "33;"
                "border-radius:3px;background:" + _regime_badge_clr + "11;'>"
                + _regime_for_macro + " " + _regime_mult_s +
                "</span>"
                "</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            if not _picks:
                st.caption("No data — click Refresh to load stock data.")
                continue

            _pcols = st.columns(3)
            for _pi, _pk in enumerate(_picks[:3]):
                with _pcols[_pi]:
                    _fs = _pk["final_score"]
                    _bclr = _pk["badge_clr"]
                    _r1d = _pk["ret_1d"]
                    _r1d_clr = "#00c851" if _r1d >= 0 else "#ff4444"
                    _de = _pk.get("de_ratio")
                    _mg = _pk.get("net_margin")
                    _sma_clr = "#00c851" if _pk.get(
                        "above_sma200") else "#ff4444"
                    _sma_lbl = "Above SMA200" if _pk.get(
                        "above_sma200") else "Below SMA200"
                    _fb_tag = " ★ fallback" if _pk.get("is_fallback") else ""

                    # Institutional score for this ticker
                    _inst_v = _pk.get("institutional_score")
                    _inst_clr = "#00c851" if (_inst_v or 0) >= 0.60 else "#9e9e9e" if (
                        _inst_v or 0) >= 0.40 else "#ff9800"

                    # Pre-compute optional HTML snippets to avoid nested f-strings
                    _de_html = (
                        '<span style="color:#AAAAAA"> · </span>'
                        '<span style="color:#888">D/E ' + f"{_de:.1f}x</span>"
                    ) if _de is not None else ""
                    _mg_html = (
                        '<span style="color:#AAAAAA"> · </span>'
                        '<span style="color:#888">Margin ' +
                        f"{_mg:.0f}%</span>"
                    ) if _mg is not None else ""
                    _cong_html = (
                        '<span style="color:#AAAAAA"> · </span>'
                        '<span style="color:' + _inst_clr +
                        '">Inst ' + f"{_inst_v:.0%}</span>"
                    ) if _inst_v is not None else ""
                    _badge_bg = _bclr + "22"
                    _badge_bdr = _bclr + "44"
                    _card_bdr = _bclr + "33"
                    _price_str = f"${_pk['price']:,.2f}"
                    _r1d_str = f"{_r1d:+.2%}"
                    _score_str = f"{_fs:.0%}"

                    st.markdown(
                        "<div style='background:#0d0d0d;border:1px solid " + _card_bdr + ";"
                        "border-top:3px solid " + _bclr + ";border-radius:6px;"
                        "padding:14px 14px 10px 14px;margin-bottom:6px;'>"

                        "<div style='display:flex;justify-content:space-between;"
                        "align-items:center;margin-bottom:8px;'>"
                        "<span style='font-size:0.58rem;font-weight:900;"
                        "letter-spacing:0.14em;padding:3px 8px;border-radius:3px;"
                        "background:" + _badge_bg + ";color:" + _bclr + ";"
                        "font-family:monospace;border:1px solid " + _badge_bdr + ";'>"
                        + _pk["badge"] +
                        "</span>"
                        "<span style='font-size:0.72rem;font-weight:900;"
                        "color:" + _bclr + ";font-family:monospace;'>" + _score_str + "</span>"
                        "</div>"

                        "<div style='font-size:1.4rem;font-weight:900;color:#ffffff;"
                        "font-family:monospace;line-height:1;'>" +
                        _pk["ticker"] + "</div>"
                        "<div style='font-size:0.65rem;color:#AAAAAA;margin-bottom:8px;'>"
                        + _pk["name"] + _fb_tag +
                        "</div>"

                        "<div style='font-size:1.1rem;font-weight:700;color:#e0e0e0;"
                        "font-family:monospace;'>" + _price_str +
                        "<span style='font-size:0.72rem;font-weight:600;"
                        "color:" + _r1d_clr + ";margin-left:6px;'>" + _r1d_str + "</span>"
                        "</div>"

                        "<div style='display:flex;align-items:center;gap:4px;margin:6px 0;"
                        "font-size:0.62rem;font-family:monospace;flex-wrap:wrap;'>"
                        "<span style='color:" + _sma_clr + ";'>" + _sma_lbl + "</span>"
                        + _de_html + _mg_html + _cong_html +
                        "</div>"

                        "<div style='font-size:0.65rem;color:#777;line-height:1.4;"
                        "margin-top:6px;padding-top:6px;border-top:1px solid #1a1a1a;"
                        "font-style:italic;'>"
                        + _pk["reason"] +
                        "</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # 30-day chart in expander
                    _cl30 = _pk.get("close_30d")
                    if _cl30 is not None and len(_cl30) > 1:
                        with st.expander("30-day chart", expanded=False):
                            _fig_s = go.Figure()
                            _fig_s.add_trace(go.Scatter(
                                x=_cl30.index, y=_cl30.values,
                                mode="lines", name=_pk["ticker"],
                                line=dict(color=_bclr, width=2),
                                fill="tozeroy",
                                fillcolor=_hex_to_rgba(_bclr, 0.10),
                            ))
                            _fig_s.update_layout(
                                height=140,
                                margin=dict(t=4, b=4, l=4, r=4),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                showlegend=False,
                                xaxis=dict(showgrid=False,
                                           showticklabels=False),
                                yaxis=dict(showgrid=True, gridcolor="#111",
                                           tickfont=dict(size=8, color="#AAAAAA")),
                            )
                            st.plotly_chart(_fig_s, use_container_width=True,
                                            config={"displayModeBar": False})

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── SECTION 04 — CONVICTION MATRIX TABLE ──────────────────────────────
        _section_bar("04 · Conviction Matrix", "4-Pillar Scoring")

        _matrix_rows = []
        for _comm in COMMODITY_UNIVERSE:
            _tk = _comm["ticker"]
            _pd2 = _prices.get(_tk) or {}
            _cv2 = _convictions.get(_tk, {})
            if not _pd2:
                continue
            _matrix_rows.append({
                "Commodity":    _comm["name"],
                "Sector":       _comm["sector"],
                "Price":        _pd2.get("price", 0.0),
                "1d %":         _pd2.get("ret_1d",  0.0) * 100,
                "5d %":         _pd2.get("ret_5d",  0.0) * 100,
                "TS Score":     _cv2.get("ts_score",   0.5),
                "COT Proxy":    _cv2.get("cot_score",  0.5),
                "Sentiment":    _cv2.get("sent_score", 0.5),
                "Trend":        _cv2.get("tr_score",   0.5),
                "Conviction":   _cv2.get("composite",  0.5),
                "Signal":       _cv2.get("conviction_label", "—"),
                "TS":           _cv2.get("ts_label",   "—"),
                "COT":          _cv2.get("cot_label",  "—"),
            })

        if _matrix_rows:
            _df_matrix = pd.DataFrame(_matrix_rows)
            st.dataframe(
                _df_matrix,
                use_container_width=True,
                height=370,
                column_config={
                    "Commodity":  st.column_config.TextColumn("Commodity", width="medium"),
                    "Sector":     st.column_config.TextColumn("Sector",    width="small"),
                    "Price":      st.column_config.NumberColumn("Price", format="%.4g"),
                    "1d %":       st.column_config.NumberColumn("1d %", format="%.2f%%"),
                    "5d %":       st.column_config.NumberColumn("5d %", format="%.2f%%"),
                    "TS Score":   st.column_config.ProgressColumn("Term Structure", min_value=0, max_value=1, format="%.2f"),
                    "COT Proxy":  st.column_config.ProgressColumn("COT Proxy",      min_value=0, max_value=1, format="%.2f"),
                    "Sentiment":  st.column_config.ProgressColumn("Sentiment",      min_value=0, max_value=1, format="%.2f"),
                    "Trend":      st.column_config.ProgressColumn("Trend",          min_value=0, max_value=1, format="%.2f"),
                    "Conviction": st.column_config.ProgressColumn("Conviction",     min_value=0, max_value=1, format="%.2f"),
                    "Signal":     st.column_config.TextColumn("Signal", width="small"),
                    "TS":         st.column_config.TextColumn("Term Structure Detail"),
                    "COT":        st.column_config.TextColumn("COT Signal"),
                },
                hide_index=True,
            )
        else:
            st.info(
                "No commodity data available. Click Refresh to fetch live prices.")

        # ── SECTION 04 — EXPERT SYNTHESIS ─────────────────────────────────────
        _section_bar("05 · Senior Macro Trader's View")

        _synthesis = generate_macro_synthesis(
            _prices, _convictions, _indicators)
        for _para in _synthesis:
            _comm_tag = _para.split("[")[0].strip()
            _tag_color = (
                "#00c851" if any(x in _para for x in ["RISK-ON", "Buy", "BACKWARDATION"])
                else "#ff4444" if any(x in _para for x in ["DEFENSIVE", "Recession", "Avoid", "CONTANGO"])
                else "#ff8800" if any(x in _para for x in ["Watch", "WARNING"])
                else "#9e9e9e"
            )
            st.markdown(
                f"""<div style="background:#080808;border-left:3px solid {_tag_color};
                    border-radius:0 4px 4px 0;padding:10px 14px;margin-bottom:8px;">
                  <p style="font-size:0.78rem;color:#d0d0d0;font-family:monospace;
                     line-height:1.55;margin:0;">{_para}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── SECTION 05 — PRICE CHARTS (expanders per commodity) ───────────────
        _section_bar("06 · Price & MA Charts", "SMA20 · SMA50 · SMA200")

        for _sector_name in ["Energy", "Metals", "Agriculture"]:
            _sector_comms2 = [
                c for c in COMMODITY_UNIVERSE if c["sector"] == _sector_name]
            with st.expander(f"{_sector_name} — Price & Moving Averages", expanded=False):
                _chart_cols = st.columns(len(_sector_comms2))
                for _ci2, _comm2 in enumerate(_sector_comms2):
                    _tk2 = _comm2["ticker"]
                    _pd3 = _prices.get(_tk2) or {}
                    _close_series = _pd3.get("_close")
                    with _chart_cols[_ci2]:
                        st.markdown(
                            f"<p style='font-size:0.7rem;font-weight:700;color:#888;"
                            f"margin:0 0 4px 0;font-family:monospace;'>"
                            f"{_comm2['name']}</p>",
                            unsafe_allow_html=True,
                        )
                        if _close_series is not None and len(_close_series) >= 20:
                            _fig_c = go.Figure()
                            _x_vals = list(range(len(_close_series)))

                            # Trim to last 252 bars for clarity
                            _trim = min(252, len(_close_series))
                            _cs = _close_series.iloc[-_trim:]
                            _xv = list(range(_trim))

                            _fig_c.add_trace(go.Scatter(
                                x=_cs.index, y=_cs.values,
                                mode="lines", name="Price",
                                line=dict(color="#e0e0e0", width=1.5),
                            ))
                            # SMAs
                            for _w, _col, _nm in [(20, "#4fc3f7", "SMA20"),
                                                  (50, "#ffb74d", "SMA50"),
                                                  (200, "#ef5350", "SMA200")]:
                                if len(_cs) >= _w:
                                    _sma_line = _cs.rolling(_w).mean()
                                    _fig_c.add_trace(go.Scatter(
                                        x=_sma_line.index, y=_sma_line.values,
                                        mode="lines", name=_nm,
                                        line=dict(
                                            color=_col, width=1, dash="dot"),
                                    ))
                            _fig_c.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=0, b=0),
                                paper_bgcolor="#080808",
                                plot_bgcolor="#080808",
                                font=dict(color="#888", size=9),
                                legend=dict(
                                    orientation="h", y=1.08, x=0,
                                    font=dict(size=8), bgcolor="rgba(0,0,0,0)",
                                ),
                                xaxis=dict(showgrid=False,
                                           showticklabels=False),
                                yaxis=dict(showgrid=True, gridcolor="#111",
                                           tickfont=dict(size=8)),
                            )
                            st.plotly_chart(_fig_c, use_container_width=True,
                                            config={"displayModeBar": False})
                        else:
                            st.caption("No data")

        # ── MACRO PLAYBOOK ─────────────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        with st.expander("📖 Macro Playbook — War · Inflation · AI Disruption", expanded=False):
            st.markdown(
                "<p style='font-size:0.72rem;color:#888;font-family:monospace;"
                "margin:0 0 10px 0'>"
                "Reference guide for navigating difficult macro regimes. "
                "Cross-check against live Geopolitical Risk Score (Section 00) "
                "and Yield Curve (Section 01) before acting."
                "</p>",
                unsafe_allow_html=True,
            )
            _playbook_rows = [
                ("War Escalation",          "#ff4444",
                 "Geo Risk ≥ 65% + Gold > 52w high",
                 "Long RTX · LMT · NOC · GLD · XLE · CCJ (uranium). Short XLY (consumer discretionary)."),
                ("Sticky Inflation",        "#ff8800",
                 "Oil rising + Real yield falling + CPI > 4%",
                 "Long XLE · XOM · CVX · VALE · FCX · NEM. Reduce XLK (tech). Add TIPS or commodity ETF."),
                ("Yield Curve Inversion",   "#ff8800",
                 "10Y − 3M spread < −0.5% for 3+ months",
                 "Extend bond duration (TLT). Favour XLU (utilities) + XLP (staples). Reduce cyclicals. Cash 20%."),
                ("AI Bubble Risk",          "#ffbb33",
                 "AI concentration > 40% + Semis P/E > 35x",
                 "Trim NVDA · MSFT · META. Buy defensive put spreads on QQQ. Rotate to XLV · XLF."),
                ("AI Infrastructure Bull",  "#00c851",
                 "Semis outperforming SPY + data centre capex rising",
                 "Overweight NVDA · AMD · AVGO · AMAT · MSFT (Azure). Add CDNS · SNPS (EDA tools)."),
                ("Recession Incoming",      "#ff4444",
                 "Curve inverted > 6 months + Leading indicators declining",
                 "Cash 30–40%. Long TLT · SHY · GLD. Short XLY · XLI. Keep defensive dividend payers."),
                ("Stagflation",             "#ff4444",
                 "Inflation rising + GDP slowing + Curve flat",
                 "Hardest regime: overweight commodities (GLD · OIL · AGG). Short bonds. Minimal equities."),
            ]
            for _pb_scenario, _pb_color, _pb_signal, _pb_action in _playbook_rows:
                st.markdown(
                    f'<div style="display:flex;gap:12px;padding:7px 10px;margin:3px 0;'
                    f'background:{_pb_color}08;border-left:3px solid {_pb_color}55;border-radius:0 4px 4px 0;">'
                    f'<div style="min-width:170px">'
                    f'<div style="font-size:0.68rem;font-weight:900;color:{_pb_color};'
                    f'text-transform:uppercase;letter-spacing:0.08em;font-family:monospace">{_pb_scenario}</div>'
                    f'<div style="font-size:0.60rem;color:#AAAAAA;margin-top:1px">Signal: {_pb_signal}</div>'
                    f'</div>'
                    f'<div style="font-size:0.68rem;color:#888;line-height:1.45">{_pb_action}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<p style='font-size:0.60rem;color:#AAAAAA;font-family:monospace;"
                "margin:8px 0 0 0'>"
                "⚠ This playbook is a reference framework, not financial advice. "
                "Always validate signals with your own risk parameters and position sizing rules. "
                "Past regime correlations are not guaranteed to repeat."
                "</p>",
                unsafe_allow_html=True,
            )

        # ── SECTION 06 — GLOBAL ECONOMIC MOMENTUM (FRED) ─────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _section_bar("06 · Global Economic Momentum",
                     "GDP · PMI · Claims · Consumer Confidence — FRED/OECD")

        @st.cache_data(ttl=3600, show_spinner=False)
        def _load_global_macro() -> Dict[str, Any]:
            try:
                import sys as _sys
                _core_p = str(_HERE / "core")
                if _core_p not in _sys.path:
                    _sys.path.insert(0, _core_p)
                from macro_global import GlobalMacroEngine as _GME  # type: ignore
                return _GME().fetch_all()
            except Exception as _exc:
                return {"_error": str(_exc)}

        with st.spinner("Fetching global macro from FRED…"):
            _gm_data = _load_global_macro()

        if "_error" in _gm_data:
            st.warning(
                f"FRED data unavailable: {_gm_data['_error']}. "
                "Ensure `pandas-datareader` is installed (`pip install pandas-datareader`)."
            )
        else:
            _gm_zone_keys = ["US", "EU", "Asia"]
            _gm_zone_labels = ["US  United States",
                               "EU  Europe", "Asia  Japan"]

            # ── Z-score heatmap (RdYlGn, divergent) ───────────────────────────
            _gm_hm = macro_heatmap_fig(_gm_data)
            st.plotly_chart(_gm_hm, use_container_width=True,
                            config={"displayModeBar": False})

            _gm_zcols = st.columns(3)

            for _zi, (_gzk, _gzl) in enumerate(zip(_gm_zone_keys, _gm_zone_labels)):
                _zdat = _gm_data.get(_gzk, {})
                _zzc = [v.get("z_composite", 0.0) for v in _zdat.values()
                        if v.get("latest") is not None]
                _zs = float(
                    1.0 / (1.0 + np.exp(-np.mean(_zzc) * 0.7))) if _zzc else 0.5
                _zclr = "#00c851" if _zs >= 0.58 else "#ff4444" if _zs <= 0.42 else "#ff8800"
                _zlbl = "Expanding" if _zs >= 0.58 else "Contracting" if _zs <= 0.42 else "Neutral"

                with _gm_zcols[_zi]:
                    st.markdown(
                        f'<div style="background:#0a0a0a;border:1px solid {_zclr}33;'
                        f'border-radius:8px;padding:10px 14px;margin-bottom:8px">'
                        f'<div style="font-size:0.80rem;font-weight:900;color:{_zclr};'
                        f'font-family:monospace">{_gzl}</div>'
                        f'<div style="font-size:0.62rem;color:#AAAAAA;margin-top:2px">'
                        f'Composite: <span style="color:{_zclr};font-weight:700">{_zlbl}</span>'
                        f' · Score {_zs:.0%}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    for _ind_nm, _ind_d in _zdat.items():
                        _lv = _ind_d.get("latest")
                        _tr = _ind_d.get("trend", "neutral")
                        _ic = "#00c851" if _tr == "expanding" else "#ff4444" if _tr == "contracting" else "#666"
                        _arr = "+" if _tr == "expanding" else "-" if _tr == "contracting" else "~"
                        _lvf = f"{_lv:.1f}" if _lv is not None else "n/a"
                        st.markdown(
                            f'<div style="background:#111;border-left:3px solid {_ic};'
                            f'border-radius:0 4px 4px 0;padding:5px 10px;margin:2px 0">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center">'
                            f'<span style="font-size:0.63rem;color:#888;font-family:monospace">{_ind_nm}</span>'
                            f'<span style="font-size:0.72rem;font-weight:700;color:{_ic};'
                            f'font-family:monospace">{_arr} {_lvf}</span>'
                            f'</div>'
                            f'<div style="font-size:0.57rem;color:#AAAAAA;margin-top:1px">'
                            f'Z(3M)={_ind_d.get("z3", 0.0):+.2f}  '
                            f'Z(6M)={_ind_d.get("z6", 0.0):+.2f}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA HUNTER — SUB-TAB 2: DISCOVERY SCANNER
# ══════════════════════════════════════════════════════════════════════════════

with tab_discovery:

    # ── Header ────────────────────────────────────────────────────────────────
    _disc_hdr, _disc_btn_col = st.columns([6, 2])
    with _disc_hdr:
        st.markdown(
            "<h4 style='margin:0;color:#00c851;font-family:monospace;'>"
            "DISCOVERY SCANNER — MID-CAP &amp; SMALL-CAP ALPHA</h4>"
            "<p style='font-size:0.65rem;color:#AAAAAA;margin:2px 0 0 0;font-family:monospace'>"
            "FMP screener → liquidity filter → full 5-pillar engine · cache 6 h"
            "</p>",
            unsafe_allow_html=True,
        )
    with _disc_btn_col:
        _disc_force = st.button(
            "🔄 Force Refresh",
            use_container_width=True,
            help="Clear 6-hour cache and re-run full scan (takes ~30 s)",
            key="disc_force_refresh",
        )

    st.markdown("<hr style='border-color:#1a1a1a;margin:6px 0 12px 0;'>",
                unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────

    @st.cache_data(ttl=21600, show_spinner=False)   # 6 h — mirrors scanner TTL
    def _load_discovery(force: bool = False) -> Dict[str, Any]:
        try:
            if force:
                return force_refresh_sync(limit=5)
            return get_top_alpha_picks_sync(limit=5)
        except Exception as _exc:
            return {"mid_cap": [], "small_cap": [], "cached": False,
                    "computed_at": "", "cache_expires_at": "", "_error": str(_exc)}

    if _disc_force:
        st.cache_data.clear()
        # Also clear the intelligence engine's diskcache so ins/inst re-fetch from live APIs
        try:
            import diskcache as _dc
            import os as _os
            _intel_cache_dir = _os.path.join(_os.path.dirname(
                __file__), "intelligence", ".intel_cache")
            if _os.path.exists(_intel_cache_dir):
                _dc.Cache(_intel_cache_dir).clear()
        except Exception:
            pass

    with st.spinner("Running discovery scan — scoring 40 candidates across 5 pillars…"):
        _disc_data = _load_discovery(force=_disc_force)

    _disc_err = _disc_data.get("_error")
    if _disc_err:
        st.error(f"Discovery scan failed: {_disc_err}")

    # ── Cache status pill ─────────────────────────────────────────────────────
    _disc_cached = _disc_data.get("cached", False)
    _disc_computed = _disc_data.get("computed_at", "")
    _disc_expires = _disc_data.get("cache_expires_at", "")
    _disc_pill_clr = "#1a3a1a" if _disc_cached else "#1a1a3a"
    _disc_pill_txt = "#00c851" if _disc_cached else "#5c85d6"
    _disc_pill_lbl = "CACHED" if _disc_cached else "LIVE"
    if _disc_computed:
        try:
            _disc_age = datetime.fromisoformat(_disc_computed)
            _disc_age_str = _disc_age.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            _disc_age_str = _disc_computed[:19]
    else:
        _disc_age_str = "—"

    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:14px">'
        f'<span style="background:{_disc_pill_clr};color:{_disc_pill_txt};'
        f'font-size:0.58rem;font-weight:900;letter-spacing:0.12em;padding:3px 9px;'
        f'border-radius:3px;font-family:monospace">{_disc_pill_lbl}</span>'
        f'<span style="font-size:0.60rem;color:#AAAAAA;font-family:monospace">'
        f'computed {_disc_age_str} · expires {_disc_expires[:19] if _disc_expires else "—"} UTC'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Helper: render a single alpha-pick card ───────────────────────────────

    def _disc_card(pick: Dict[str, Any], card_key: str = "") -> None:
        sym = pick.get("symbol", "?")
        price = pick.get("price", 0.0)
        conv = pick.get("conviction", 0.5)
        pct = pick.get("conviction_pct", 50.0)
        lbl = pick.get("label", "Neutral")
        pfort = pick.get("point_fort", "—")
        direct = pick.get("direction", "neutral")
        conf = pick.get("confidence", 0.5)
        ps = pick.get("pillar_scores", {})
        vol_spike = pick.get("volume_spike")
        price_chg = pick.get("price_change_pct")

        clr = score_color(conv)
        dir_clr = "#00c851" if direct == "long" else "#ff4444" if direct == "short" else "#9e9e9e"
        bar_w = int(conv * 100)

        price_str = f"${price:,.2f}" if price > 0 else "N/A"
        conf_pct = int(conf * 100)

        # Volume spike badge
        _vol_html = ""
        if vol_spike is not None:
            _vs = float(vol_spike)
            if _vs >= 3.0:
                _vcls, _vtxt = "vol-spike", f"VOL ×{_vs:.1f}"
            elif _vs >= 1.5:
                _vcls, _vtxt = "vol-high", f"VOL ×{_vs:.1f}"
            else:
                _vcls, _vtxt = "vol-norm", f"×{_vs:.1f}"
            _vol_html = f"<span class='vol-badge {_vcls}'>{_vtxt}</span>"

        # Price change badge
        _pchg_html = ""
        if price_chg is not None:
            _pc_val = float(price_chg)
            _pc_sign = "+" if _pc_val >= 0 else ""
            _pc_clr = "#00c851" if _pc_val >= 0 else "#ff4444"
            _pchg_html = (
                f"<span style='font-size:0.50rem;font-weight:800;color:{_pc_clr};"
                f"font-family:monospace;margin-left:4px'>{_pc_sign}{_pc_val:.1f}%</span>"
            )

        # Pillar mini-bars
        _pillar_html = ""
        for _pk, _plbl in [("inst", "INST"), ("ins", "INS"), ("sent", "SENT"), ("news", "NEWS"), ("macro", "MAC")]:
            _pv = ps.get(_pk, 0.5)
            _pc = score_color(_pv)
            _pw = int(_pv * 100)
            _pillar_html += (
                f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:1px">'
                f'<div style="font-size:0.45rem;color:#AAAAAA;width:26px;font-family:monospace'
                f';text-align:right">{_plbl}</div>'
                f'<div style="flex:1;height:2px;background:#111;border-radius:1px">'
                f'<div style="width:{_pw}%;height:2px;background:{_pc};border-radius:1px"></div>'
                f'</div>'
                f'<div style="font-size:0.45rem;color:#AAAAAA;width:22px;font-family:monospace'
                f';text-align:right">{_pv:.0%}</div>'
                f'</div>'
            )

        _is_sel = st.session_state.get("selected_ticker") == sym
        _border_extra = f"box-shadow:0 0 0 2px {clr}80;" if _is_sel else ""

        st.markdown(
            f'<div style="background:#0a0a0a;border:1px solid #1a1a1a;{_border_extra}'
            f'border-left:3px solid {clr};border-radius:6px;padding:12px 13px;height:100%">'

            # ── Header row ──
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
            f'margin-bottom:6px">'
            f'<div>'
            f'<div style="font-size:1.15rem;font-weight:900;color:#fff;'
            f'font-family:monospace;letter-spacing:0.04em;line-height:1">{sym}</div>'
            f'<div style="margin-top:3px;display:flex;align-items:center;gap:2px">'
            f'{_vol_html}{_pchg_html}'
            f'</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:0.80rem;font-weight:700;color:#bbb;'
            f'font-family:monospace">{price_str}</div>'
            f'<div style="font-size:0.52rem;color:{dir_clr};font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.08em">{direct}</div>'
            f'</div>'
            f'</div>'

            # ── Conviction bar ──
            f'<div style="height:4px;background:#1a1a1a;border-radius:2px;margin-bottom:5px">'
            f'<div style="width:{bar_w}%;height:4px;background:{clr};border-radius:2px"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
            f'<div style="font-size:0.60rem;color:{clr};font-weight:700;'
            f'font-family:monospace">{lbl.upper()}</div>'
            f'<div style="font-size:0.60rem;color:#AAAAAA;font-family:monospace">'
            f'{pct:.0f}% · {conf_pct}% conf</div>'
            f'</div>'

            # ── Point Fort ──
            f'<div style="background:#111;border-radius:4px;padding:5px 8px;'
            f'margin-bottom:8px">'
            f'<div style="font-size:0.50rem;color:#AAAAAA;text-transform:uppercase;'
            f'letter-spacing:0.10em;margin-bottom:2px">Point Fort</div>'
            f'<div style="font-size:0.62rem;color:#00c851;font-weight:700;'
            f'font-family:monospace">{pfort}</div>'
            f'</div>'

            # ── Pillar mini-bars ──
            f'{_pillar_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Ticker selector button below the card
        _btn_lbl = f"● {sym}" if _is_sel else sym
        if st.button(_btn_lbl, key=f"disc_sel_{card_key}_{sym}",
                     use_container_width=True, help=f"Inspect {sym} in Market Intel"):
            st.session_state["selected_ticker"] = None if _is_sel else sym
            st.rerun()

    # ── Section: Top 5 Mid-Cap Alpha ──────────────────────────────────────────
    _section_bar("01 · Top 5 Mid-Cap Alpha", "$2B – $10B Market Cap")

    _mid_picks = _disc_data.get("mid_cap", [])
    if _mid_picks:
        _mid_cols = st.columns(min(len(_mid_picks), 5))
        for _i, _pick in enumerate(_mid_picks[:5]):
            with _mid_cols[_i]:
                _disc_card(_pick, card_key="mid")
    else:
        st.info(
            "No mid-cap picks available yet. "
            "Click **Force Refresh** to run the discovery scan.",
            icon="ℹ️",
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Section: Top 5 Small-Cap Gems ─────────────────────────────────────────
    _section_bar("02 · Top 5 Small-Cap Gems", "$300M – $2B Market Cap")

    _sm_picks = _disc_data.get("small_cap", [])
    if _sm_picks:
        _sm_cols = st.columns(min(len(_sm_picks), 5))
        for _i, _pick in enumerate(_sm_picks[:5]):
            with _sm_cols[_i]:
                _disc_card(_pick, card_key="sm")
    else:
        st.info(
            "No small-cap picks available yet. "
            "Click **Force Refresh** to run the discovery scan.",
            icon="ℹ️",
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Legend / methodology note ─────────────────────────────────────────────
    with st.expander("📖 How Discovery Scoring Works", expanded=False):
        st.markdown(
            """
<div style='font-size:0.68rem;color:#AAAAAA;font-family:monospace;line-height:1.7'>

**Universe** — FMP v3 screener filters ~5 000 US equities by market-cap band.
Liquidity gate: dollar volume > $1 M / day removes illiquid names.

**Pre-selection** — Top 20 candidates per category ranked by dollar-volume × |β − 1|.
High-beta, liquid names get priority; low-volatility names are de-prioritised.

**Scoring** — Full 5-pillar intelligence engine runs on each candidate:
| Pillar | Source | Base Weight |
|--------|--------|-------------|
| INST | FMP institutional holders | 20 % |
| INS | FMP v4 insider trading | 20 % |
| SENT | StockTwits social sentiment | 20 % |
| NEWS | Finnhub + VADER NLP | 20 % |
| MAC | Finnhub analyst consensus | 20 % |

**Dynamic weighting** adjusts these at runtime (CEO buy > $1M → INS boosted to 40 %;
volume > 2σ → SENT boosted; missing data → redistributed to active pillars).

**Point Fort** is derived from the dominant pillar signal — the reason the score stands out.

**Cache** — Results are stored for 6 hours in `logs/discovery_cache.json`.
Use **Force Refresh** to bypass the cache and re-run the full scan.

</div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — VALUATION SUITE
# DCF-inspired ETF fair value · Regime-aware Monte Carlo · Scenario analysis
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def _fetch_valuation_market_data() -> Dict[str, Any]:
    """
    Fetch live market data for the Valuation Suite via yfinance.
    Cached for 1 hour.  Returns a dict with prices, yields, and fundamentals.
    Falls back to reasonable defaults when yfinance is unavailable.
    """
    data: Dict[str, Any] = {
        "yield_10y":  0.043,
        "real_yield": 0.019,
        "prices":     {},
        "extra":      {},
        "source":     "demo",
    }
    try:
        import yfinance as yf

        # ── 10-year Treasury yield (^TNX) ─────────────────────────────────────
        try:
            tnx = yf.Ticker("^TNX").history(period="5d")
            if not tnx.empty:
                data["yield_10y"] = float(tnx["Close"].iloc[-1]) / 100.0
        except Exception:
            pass

        # ── TIPS 10Y real yield proxy — use TIP ETF 30-day move as proxy ──────
        try:
            tip = yf.Ticker("^DFII10").history(period="5d")  # FRED: 10Y TIPS
            if not tip.empty:
                data["real_yield"] = float(tip["Close"].iloc[-1]) / 100.0
        except Exception:
            data["real_yield"] = max(data["yield_10y"] - 0.025, -0.005)

        # ── ETF prices + fundamentals ──────────────────────────────────────────
        tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SHY", "VNQ", "DBC"]
        for sym in tickers:
            try:
                t = yf.Ticker(sym)
                info = t.info
                price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("navPrice")
                    or 0.0
                )
                data["prices"][sym] = float(price)

                # Forward P/E for equity ETFs
                for key in ("forwardPE", "trailingPE"):
                    val = info.get(key)
                    if val and val > 0:
                        data["extra"][f"{sym}_forward_pe"] = float(val)
                        break

                # Dividend yield for VNQ
                div = info.get("dividendYield") or info.get("yield")
                if div and div > 0:
                    data["extra"][f"{sym}_dividend_yield"] = float(div)

                # 52-week range for DBC momentum
                high = info.get("fiftyTwoWeekHigh")
                low = info.get("fiftyTwoWeekLow")
                if high:
                    data["extra"][f"{sym}_52w_high"] = float(high)
                if low:
                    data["extra"][f"{sym}_52w_low"] = float(low)

            except Exception:
                pass

        data["source"] = "live" if data["prices"] else "demo"

    except ImportError:
        pass

    # Inject reasonable demo prices for any missing tickers
    _demo_prices = {
        "SPY": 540.0, "QQQ": 465.0, "IWM": 205.0, "GLD": 218.0,
        "TLT": 94.0,  "SHY": 84.5,  "VNQ": 90.0,  "DBC": 22.5,
    }
    for sym, px in _demo_prices.items():
        if data["prices"].get(sym, 0.0) <= 0.0:
            data["prices"][sym] = px

    return data


with tab_valuation:

    # ── Load regime + portfolio context ───────────────────────────────────────
    _val_regime = _get_regime_state()
    _val_portf = _get_portfolio_state()
    _val_lbl = _val_regime.get("label", "Neutral")
    _val_probs = _val_regime.get("regime_probs", {})
    _val_equity = float(_val_portf.get("equity", 100_000.0))
    _val_conf = float(_val_regime.get("confidence", 0.5))
    _val_rc = regime_color(_val_lbl)

    # ── Import valuation engines (lazy) ───────────────────────────────────────
    _val_engines_ok = False
    try:
        import importlib.util as _ilu
        import sys as _sys
        _val_eng_path = _HERE / "valuation" / "engine.py"
        _val_spec = _ilu.spec_from_file_location(
            "valuation.engine", _val_eng_path)
        _val_mod = _ilu.module_from_spec(_val_spec)
        _val_spec.loader.exec_module(_val_mod)
        _RegimeMC = _val_mod.RegimeMonteCarloEngine
        _ScenarioEng = _val_mod.ScenarioEngine
        _FairValueEng = _val_mod.ETFFairValueEngine
        _val_engines_ok = True
    except Exception as _e:
        st.error(f"⚠️ Valuation engines failed to load: {_e}")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:16px;padding:12px 0 8px 0;
                    border-bottom:1px solid #1a1a1a;margin-bottom:16px'>
          <span style='font-size:1.4rem;font-weight:700;color:#e0e0e0'>
            💹 Valuation Suite
          </span>
          <span style='font-size:0.72rem;color:{_val_rc};font-family:monospace;
                        background:#111;padding:3px 10px;border-radius:4px;
                        border:1px solid {_val_rc}33'>
            Regime: {_val_lbl} · Conf {_val_conf:.0%}
          </span>
          <span style='font-size:0.72rem;color:#AAAAAA;font-family:monospace'>
            Portfolio ${_val_equity:,.0f}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _val_engines_ok:
        st.info("Valuation engines could not be loaded. Check valuation/engine.py.")
        st.stop()

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    _v_mc_tab, _v_sc_tab, _v_fv_tab, _v_gbm_tab, _v_ml_tab = st.tabs([
        "🎲 Monte Carlo Forecast",
        "📐 Scenario Analysis",
        "📊 Fair Value Dashboard",
        "📉 GBM Risk Simulator",
        "🤖 ML-DCF Valuation",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB A — MONTE CARLO FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    with _v_mc_tab:
        st.markdown(
            "<p style='font-size:0.72rem;color:#AAAAAA;margin-bottom:12px'>"
            "Regime-aware Markov chain Monte Carlo — portfolio paths driven by "
            "HMM transition dynamics and regime-conditional ETF return distributions.</p>",
            unsafe_allow_html=True,
        )

        # Controls
        _mc_col1, _mc_col2, _mc_col3, _mc_col4 = st.columns([2, 2, 2, 2])
        with _mc_col1:
            _mc_horizon_label = st.selectbox(
                "Horizon",
                ["1 Month (21d)", "3 Months (63d)", "6 Months (126d)",
                 "1 Year (252d)", "2 Years (504d)"],
                index=3,
                key="mc_horizon",
            )
            _mc_horizon_map = {
                "1 Month (21d)": 21, "3 Months (63d)": 63,
                "6 Months (126d)": 126, "1 Year (252d)": 252, "2 Years (504d)": 504,
            }
            _mc_horizon = _mc_horizon_map[_mc_horizon_label]

        with _mc_col2:
            _mc_n_sims = st.select_slider(
                "Simulations",
                options=[1_000, 3_000, 5_000, 10_000],
                value=5_000,
                key="mc_n_sims",
            )

        with _mc_col3:
            _mc_init_val = st.number_input(
                "Portfolio Value ($)",
                min_value=1_000,
                max_value=10_000_000,
                value=int(_val_equity) if _val_equity > 0 else 100_000,
                step=1_000,
                key="mc_init_val",
            )

        with _mc_col4:
            _mc_use_probs = st.toggle(
                "Start from HMM posterior",
                value=True,
                key="mc_use_probs",
                help="If ON, regime start is sampled from current HMM posterior probabilities.",
            )

        _mc_run = st.button("▶ Run Monte Carlo",
                            key="mc_run_btn", type="primary")

        if _mc_run or "mc_result" not in st.session_state:
            with st.spinner(f"Running {_mc_n_sims:,} paths × {_mc_horizon} days …"):
                _mc_eng = _RegimeMC(random_state=42)
                _mc_res = _mc_eng.run(
                    current_regime=_val_lbl,
                    regime_probs=_val_probs if _mc_use_probs else None,
                    initial_value=float(_mc_init_val),
                    n_simulations=_mc_n_sims,
                    horizon_days=_mc_horizon,
                )
                st.session_state["mc_result"] = _mc_res
        else:
            _mc_res = st.session_state["mc_result"]

        # ── Key metrics row ───────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _m1, _m2, _m3, _m4, _m5, _m6, _m7 = st.columns(7)
        _mc_pct_ret = (_mc_res.median_terminal / _mc_res.initial_value - 1)
        _mc_exp_ret = (_mc_res.expected_terminal / _mc_res.initial_value - 1)

        def _mc_kpi(col, label, val, sub="", color="#e0e0e0"):
            col.markdown(
                f"""<div style='background:#111;border:1px solid #1a1a1a;border-radius:6px;
                                padding:10px 8px;text-align:center'>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>{label}</div>
                      <div style='font-size:1.0rem;font-weight:700;color:{color};
                                  font-family:monospace'>{val}</div>
                      <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>{sub}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        _mc_kpi(_m1, "Median Return",
                f"{_mc_pct_ret:+.1%}",
                f"CAGR {_mc_res.median_cagr:+.1%}",
                "#00c851" if _mc_pct_ret >= 0 else "#ff4444")
        _mc_kpi(_m2, "Expected Return",
                f"{_mc_exp_ret:+.1%}",
                f"${_mc_res.expected_terminal:,.0f}",
                "#00c851" if _mc_exp_ret >= 0 else "#ff4444")
        _mc_kpi(_m3, "VaR 95%",
                f"{(_mc_res.var_95 / _mc_res.initial_value - 1):+.1%}",
                f"${_mc_res.var_95:,.0f}",
                "#ff8800")
        _mc_kpi(_m4, "CVaR 95%",
                f"{(_mc_res.cvar_95 / _mc_res.initial_value - 1):+.1%}",
                "Exp. Shortfall",
                "#ff4444")
        _mc_kpi(_m5, "P(Profit)",
                f"{_mc_res.prob_profit:.0%}",
                "return > 0%",
                "#00c851" if _mc_res.prob_profit > 0.5 else "#ff8800")
        _mc_kpi(_m6, "P(+10%)",
                f"{_mc_res.prob_10pct:.0%}",
                f"P(+20%) {_mc_res.prob_20pct:.0%}",
                "#9e9e9e")
        _mc_kpi(_m7, "E[Max DD]",
                f"{_mc_res.expected_max_drawdown:.1%}",
                f"Sharpe {_mc_res.median_sharpe:.2f}",
                "#ff8800")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Fan chart ─────────────────────────────────────────────────────────
        _t_axis = list(range(_mc_horizon + 1))
        _pcts = _mc_res.percentiles

        _fan = go.Figure()

        # Outer band P5–P95
        _fan.add_trace(go.Scatter(
            x=_t_axis + _t_axis[::-1],
            y=list(_pcts[5]) + list(_pcts[95])[::-1],
            fill="toself", fillcolor="rgba(0,200,81,0.05)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P5–P95", showlegend=True, hoverinfo="skip",
        ))

        # Inner band P25–P75
        _fan.add_trace(go.Scatter(
            x=_t_axis + _t_axis[::-1],
            y=list(_pcts[25]) + list(_pcts[75])[::-1],
            fill="toself", fillcolor="rgba(0,200,81,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P25–P75", showlegend=True, hoverinfo="skip",
        ))

        # P10 / P90 lines (thin)
        for _p, _c, _dash in [(10, "rgba(255,136,0,0.40)", "dot"),
                              (90, "rgba(0,200,81,0.40)",  "dot")]:
            _fan.add_trace(go.Scatter(
                x=_t_axis, y=list(_pcts[_p]),
                line=dict(color=_c, width=1, dash=_dash),
                name=f"P{_p}", showlegend=True,
            ))

        # Median
        _fan.add_trace(go.Scatter(
            x=_t_axis, y=list(_pcts[50]),
            line=dict(color="#00c851", width=2.5),
            name="Median (P50)", showlegend=True,
        ))

        # Initial value reference
        _fan.add_hline(
            y=_mc_init_val, line_width=1, line_dash="dash",
            line_color="rgba(255,255,255,0.25)",
            annotation_text=f"Initial ${_mc_init_val:,.0f}",
            annotation_font_size=10,
        )

        _fan.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
            height=360,
            margin=dict(l=8, r=8, t=28, b=8),
            title=dict(
                text=f"Portfolio Fan Chart — {_mc_n_sims:,} Simulations · {_mc_horizon}d Horizon",
                font=dict(size=12, color="#9e9e9e"),
            ),
            xaxis=dict(
                title="Trading Days", color="#AAAAAA",
                gridcolor="#111", zeroline=False,
            ),
            yaxis=dict(
                title="Portfolio Value ($)", color="#AAAAAA",
                gridcolor="#111", zeroline=False,
                tickformat="$,.0f",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0,
                font=dict(size=10, color="#9e9e9e"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        apply_pro_theme(_fan)
        st.plotly_chart(_fan, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Regime visit distribution ──────────────────────────────────────────
        _RC = getattr(_val_mod, "REGIME_COLORS", {})
        _rv_labels = [
            k for k, v in _mc_res.regime_visit_fractions.items() if v > 0.001]
        _rv_values = [_mc_res.regime_visit_fractions[k] for k in _rv_labels]
        _rv_colors = [_RC.get(k, "#9e9e9e") for k in _rv_labels]

        _rv_col1, _rv_col2 = st.columns([1, 2])
        with _rv_col1:
            _rv_fig = go.Figure(go.Pie(
                labels=_rv_labels,
                values=_rv_values,
                marker_colors=_rv_colors,
                textinfo="label+percent",
                hole=0.45,
                textfont=dict(size=10, color="#ccc"),
            ))
            _rv_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                height=240, margin=dict(l=0, r=0, t=28, b=0),
                title=dict(text="Expected Regime Distribution",
                           font=dict(size=11, color="#9e9e9e")),
                showlegend=False,
            )
            apply_pro_theme(_rv_fig)
            st.plotly_chart(_rv_fig, use_container_width=True,
                            config={"displayModeBar": False})

        with _rv_col2:
            # Return distribution histogram
            _hist_fig = go.Figure()
            _mc_terminal_rets = (_mc_res.terminal_values /
                                 _mc_res.initial_value - 1) * 100
            _hist_fig.add_trace(go.Histogram(
                x=_mc_terminal_rets,
                nbinsx=60,
                marker_color="#00c851",
                opacity=0.7,
                name="Terminal Return %",
            ))
            _hist_fig.add_vline(
                x=float(np.median(_mc_terminal_rets)),
                line_width=2, line_color="#00c851",
                annotation_text=f"Median {float(np.median(_mc_terminal_rets)):+.1f}%",
                annotation_font=dict(size=10, color="#00c851"),
            )
            _hist_fig.add_vline(
                x=float((_mc_res.var_95 / _mc_res.initial_value - 1) * 100),
                line_width=1.5, line_color="#ff4444", line_dash="dash",
                annotation_text="VaR 95%",
                annotation_font=dict(size=10, color="#ff4444"),
            )
            _hist_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                height=240, margin=dict(l=8, r=8, t=28, b=8),
                title=dict(text="Terminal Return Distribution",
                           font=dict(size=11, color="#9e9e9e")),
                xaxis=dict(title="Return (%)",
                           color="#AAAAAA", gridcolor="#111"),
                yaxis=dict(title="Frequency",
                           color="#AAAAAA", gridcolor="#111"),
                showlegend=False,
            )
            apply_pro_theme(_hist_fig)
            st.plotly_chart(_hist_fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB B — SCENARIO ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with _v_sc_tab:
        st.markdown(
            "<p style='font-size:0.72rem;color:#AAAAAA;margin-bottom:12px'>"
            "Named forward scenarios derived from current HMM posteriors and Markov "
            "transition dynamics.  Expected portfolio value computed at 1 / 3 / 6 / 12 month horizons.</p>",
            unsafe_allow_html=True,
        )

        _sc_init_val = st.number_input(
            "Portfolio Value ($)",
            min_value=1_000, max_value=10_000_000,
            value=int(_val_equity) if _val_equity > 0 else 100_000,
            step=1_000, key="sc_init_val",
        )

        _sc_eng = _ScenarioEng()
        _sc_results = _sc_eng.run(
            current_regime=_val_lbl,
            regime_probs=_val_probs,
            portfolio_value=float(_sc_init_val),
        )
        _sc_ev = _sc_eng.expected_value(_sc_results, float(_sc_init_val))

        # ── Expected Value summary ────────────────────────────────────────────
        st.markdown(
            """<div style='background:#0d1a0d;border:1px solid #1a2a1a;border-radius:8px;
                           padding:14px 18px;margin-bottom:16px'>
               <div style='font-size:0.68rem;color:#9e9e9e;font-family:monospace;
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
                 Probability-Weighted Expected Value
               </div>""",
            unsafe_allow_html=True,
        )
        _ev_c1, _ev_c2, _ev_c3, _ev_c4 = st.columns(4)
        for _col, _lbl, _val, _ret in [
            (_ev_c1, "1 Month",  _sc_ev["ev_1m"],  _sc_ev["ret_1m"]),
            (_ev_c2, "3 Months", _sc_ev["ev_3m"],  _sc_ev["ret_3m"]),
            (_ev_c3, "6 Months", _sc_ev["ev_6m"],  _sc_ev["ret_6m"]),
            (_ev_c4, "12 Months", _sc_ev["ev_12m"], _sc_ev["ret_12m"]),
        ]:
            _ret_color = "#00c851" if _ret >= 0 else "#ff4444"
            _col.markdown(
                f"""<div style='text-align:center'>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>{_lbl}</div>
                      <div style='font-size:1.05rem;font-weight:700;color:#e0e0e0;
                                  font-family:monospace'>${_val:,.0f}</div>
                      <div style='font-size:0.75rem;color:{_ret_color};font-family:monospace'>
                        {_ret:+.1%}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Scenario cards ────────────────────────────────────────────────────
        _RISK_COLORS = {
            "Low": "#00c851", "Moderate": "#9e9e9e",
            "Elevated": "#ffbb33", "High": "#ff8800", "Extreme": "#ff4444",
        }
        _ACTION_COLORS = {
            "Hold": "#9e9e9e", "Increase": "#00c851",
            "Reduce": "#ff8800", "Exit": "#ff4444",
            "Monitor / Trim": "#ffbb33", "Hold / Trim": "#ffbb33",
            "Prepare Re-entry": "#00e676",
        }

        for _sc in _sc_results:
            _prob_bar = int(_sc.probability * 100)
            _r_color = _RISK_COLORS.get(_sc.risk_level, "#9e9e9e")
            _a_color = _ACTION_COLORS.get(_sc.action, "#9e9e9e")
            _ret12_color = "#00c851" if _sc.expected_return_12m >= 0 else "#ff4444"
            _dd_color = "#ff8800" if _sc.expected_max_drawdown < -0.05 else "#9e9e9e"

            st.markdown(
                f"""
                <div style='background:#111;border:1px solid #1a1a1a;border-left:3px solid {_sc.color};
                             border-radius:6px;padding:14px 16px;margin-bottom:10px'>
                  <div style='display:flex;align-items:center;justify-content:space-between;
                              margin-bottom:6px'>
                    <div>
                      <span style='font-size:0.85rem;font-weight:700;color:#e0e0e0'>
                        {_sc.name}
                      </span>
                      <span style='font-size:0.68rem;color:{_r_color};font-family:monospace;
                                    margin-left:10px;background:{_r_color}18;
                                    padding:2px 8px;border-radius:3px'>
                        Risk: {_sc.risk_level}
                      </span>
                    </div>
                    <span style='font-size:0.72rem;color:{_a_color};font-family:monospace;
                                  background:{_a_color}18;padding:2px 10px;border-radius:3px'>
                      {_sc.action}
                    </span>
                  </div>
                  <div style='font-size:0.67rem;color:#AAAAAA;margin-bottom:8px'>{_sc.description}</div>

                  <!-- Probability bar -->
                  <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
                    <div style='font-size:0.62rem;color:#AAAAAA;font-family:monospace;width:60px'>
                      P = {_sc.probability:.0%}
                    </div>
                    <div style='flex:1;background:#1a1a1a;border-radius:3px;height:5px'>
                      <div style='width:{_prob_bar}%;background:{_sc.color};
                                   height:5px;border-radius:3px'></div>
                    </div>
                  </div>

                  <!-- Horizon metrics -->
                  <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px'>
                    <div style='text-align:center'>
                      <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>1 Month</div>
                      <div style='font-size:0.78rem;color:{"#00c851" if _sc.expected_return_1m >= 0 else "#ff4444"};
                                   font-family:monospace'>{_sc.expected_return_1m:+.1%}</div>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>
                        ${_sc.portfolio_value_1m:,.0f}</div>
                    </div>
                    <div style='text-align:center'>
                      <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>3 Months</div>
                      <div style='font-size:0.78rem;color:{"#00c851" if _sc.expected_return_3m >= 0 else "#ff4444"};
                                   font-family:monospace'>{_sc.expected_return_3m:+.1%}</div>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>
                        ${_sc.portfolio_value_3m:,.0f}</div>
                    </div>
                    <div style='text-align:center'>
                      <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>6 Months</div>
                      <div style='font-size:0.78rem;color:{"#00c851" if _sc.expected_return_6m >= 0 else "#ff4444"};
                                   font-family:monospace'>{_sc.expected_return_6m:+.1%}</div>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>
                        ${_sc.portfolio_value_6m:,.0f}</div>
                    </div>
                    <div style='text-align:center'>
                      <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>12 Months</div>
                      <div style='font-size:0.78rem;color:{_ret12_color};
                                   font-family:monospace'>{_sc.expected_return_12m:+.1%}</div>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>
                        ${_sc.portfolio_value_12m:,.0f}</div>
                    </div>
                    <div style='text-align:center'>
                      <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>Max DD / Sharpe</div>
                      <div style='font-size:0.78rem;color:{_dd_color};
                                   font-family:monospace'>{_sc.expected_max_drawdown:.1%}</div>
                      <div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace'>
                        Sharpe {_sc.sharpe_estimate:.2f}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Scenario probability chart ─────────────────────────────────────────
        with st.expander("📊 Scenario Comparison Chart", expanded=False):
            _sc_names = [s.name for s in _sc_results]
            _sc_r12m = [s.expected_return_12m * 100 for s in _sc_results]
            _sc_probs = [s.probability * 100 for s in _sc_results]
            _sc_clrs = [s.color for s in _sc_results]

            _sc_fig = go.Figure()
            _sc_fig.add_trace(go.Bar(
                x=_sc_names, y=_sc_r12m,
                marker_color=_sc_clrs,
                name="12M Expected Return",
                text=[f"{r:+.1f}%" for r in _sc_r12m],
                textposition="outside",
                textfont=dict(size=10),
            ))
            _sc_fig.add_trace(go.Scatter(
                x=_sc_names, y=_sc_probs,
                mode="markers+text",
                marker=dict(size=10, color="#ffbb33", symbol="diamond"),
                name="Probability %",
                text=[f"{p:.0f}%" for p in _sc_probs],
                textposition="top center",
                textfont=dict(size=9, color="#ffbb33"),
                yaxis="y2",
            ))
            _sc_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a",
                height=320, margin=dict(l=8, r=8, t=28, b=8),
                yaxis=dict(title="12M Expected Return (%)",
                           color="#AAAAAA", gridcolor="#111"),
                yaxis2=dict(title="Probability (%)", overlaying="y", side="right",
                            color="#ffbb33", range=[0, 100]),
                legend=dict(font=dict(size=10, color="#9e9e9e"),
                            bgcolor="rgba(0,0,0,0)"),
                title=dict(text="Scenario Return vs Probability",
                           font=dict(size=11, color="#9e9e9e")),
            )
            apply_pro_theme(_sc_fig)
            st.plotly_chart(_sc_fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB C — FAIR VALUE DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with _v_fv_tab:
        st.markdown(
            "<p style='font-size:0.72rem;color:#AAAAAA;margin-bottom:12px'>"
            "Yield-model fair value for the full ETF universe. "
            "Data via yfinance (cached 1h). Fed Model, Duration Gap, Real Rate, "
            "REIT Spread, and 52-week Momentum models.</p>",
            unsafe_allow_html=True,
        )

        _fv_col_r, _fv_col_b = st.columns([3, 1])
        with _fv_col_b:
            _fv_refresh = st.button("⟳ Refresh Market Data", key="fv_refresh")
        if _fv_refresh:
            st.cache_data.clear()

        with st.spinner("Fetching market data …"):
            _mkt = _fetch_valuation_market_data()

        _fv_source_badge = (
            "<span style='color:#00c851'>● Live</span>"
            if _mkt["source"] == "live"
            else "<span style='color:#ffbb33'>● Demo</span>"
        )
        st.markdown(
            f"<div style='font-size:0.60rem;color:#AAAAAA;font-family:monospace;"
            f"margin-bottom:8px'>Data source: {_fv_source_badge} "
            f"· 10Y yield {_mkt['yield_10y']:.2%} "
            f"· Real yield {_mkt['real_yield']:.2%}</div>",
            unsafe_allow_html=True,
        )

        _fv_eng = _FairValueEng(risk_free_rate=_mkt["yield_10y"])
        _fv_results = _fv_eng.estimate_all(
            prices=_mkt["prices"],
            yield_10y=_mkt["yield_10y"],
            real_yield=_mkt["real_yield"],
            extra_data=_mkt["extra"],
        )

        if not _fv_results:
            st.info("No fair value data available.  Check yfinance connectivity.")
        else:
            # ── Valuation table ───────────────────────────────────────────────
            _fv_rows = []
            for sym, fv in _fv_results.items():
                _upside_sign = "▲" if fv.upside_pct > 0 else (
                    "▼" if fv.upside_pct < 0 else "—")
                _fv_rows.append({
                    "Ticker":          sym,
                    "Price":           f"${fv.current_price:.2f}",
                    "Fair Value":      f"${fv.fair_value:.2f}",
                    "Upside":          f"{_upside_sign} {abs(fv.upside_pct):.1f}%",
                    "Signal":          fv.signal,
                    "Confidence":      fv.confidence,
                    "Model":           fv.model,
                    "Key Metric":      fv.key_metric,
                    "_signal_color":   fv.signal_color,
                    "_upside_raw":     fv.upside_pct,
                })

            # Render table as styled HTML cards (2-column grid)
            _fv_items = list(_fv_results.items())
            for _i in range(0, len(_fv_items), 2):
                _cols = st.columns(2)
                for _j, _col in enumerate(_cols):
                    if _i + _j >= len(_fv_items):
                        break
                    _sym, _fv = _fv_items[_i + _j]
                    _upside_c = "#00c851" if _fv.upside_pct > 0 else (
                        "#ff4444" if _fv.upside_pct < 0 else "#9e9e9e")
                    _conf_c = {"High": "#00c851", "Medium": "#ffbb33", "Low": "#ff8800"}.get(
                        _fv.confidence, "#9e9e9e")

                    # Gauge bar: upside clamped to ±30%
                    _bar_pct = max(
                        min((_fv.upside_pct + 30) / 60 * 100, 100), 0)
                    _bar_fill = _fv.signal_color

                    _col.markdown(
                        f"""
                        <div style='background:#111;border:1px solid #1a1a1a;
                                     border-radius:8px;padding:14px 16px;height:100%'>
                          <div style='display:flex;justify-content:space-between;
                                      align-items:center;margin-bottom:4px'>
                            <span style='font-size:1.0rem;font-weight:700;
                                          color:#e0e0e0;font-family:monospace'>{_sym}</span>
                            <span style='font-size:0.72rem;color:{_fv.signal_color};
                                          font-family:monospace;background:{_fv.signal_color}18;
                                          padding:2px 8px;border-radius:3px'>{_fv.signal}</span>
                          </div>
                          <div style='display:flex;gap:20px;margin-bottom:6px'>
                            <div>
                              <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>Current</div>
                              <div style='font-size:0.88rem;color:#ccc;font-family:monospace'>
                                ${_fv.current_price:.2f}</div>
                            </div>
                            <div>
                              <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>Fair Value</div>
                              <div style='font-size:0.88rem;color:{_upside_c};font-family:monospace'>
                                ${_fv.fair_value:.2f}</div>
                            </div>
                            <div>
                              <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>Upside</div>
                              <div style='font-size:0.88rem;font-weight:700;
                                           color:{_upside_c};font-family:monospace'>
                                {_fv.upside_pct:+.1f}%</div>
                            </div>
                            <div>
                              <div style='font-size:0.58rem;color:#AAAAAA;font-family:monospace'>Confidence</div>
                              <div style='font-size:0.72rem;color:{_conf_c};font-family:monospace'>
                                {_fv.confidence}</div>
                            </div>
                          </div>

                          <!-- Upside gauge bar -->
                          <div style='background:#1a1a1a;border-radius:3px;height:4px;
                                       margin-bottom:8px;position:relative'>
                            <div style='position:absolute;left:50%;top:0;width:1px;
                                         height:4px;background:#333'></div>
                            <div style='position:absolute;
                                         left:{"50%" if _fv.upside_pct == 0 else
                                               (f"calc(50% + {min(_fv.upside_pct/60*50, 50):.0f}%)"
                                                if _fv.upside_pct > 0 else
                                                f"calc(50% - {min(abs(_fv.upside_pct)/60*50, 50):.0f}%)")};
                                         {"width:" + str(min(abs(_fv.upside_pct)/60*50, 50)).__add__("%;") if _fv.upside_pct != 0 else "width:0;"}
                                         height:4px;background:{_bar_fill};
                                         border-radius:3px;
                                         {"right:auto" if _fv.upside_pct > 0 else "right:calc(50% - " + str(min(abs(_fv.upside_pct)/60*50, 50)).__add__("%);")}'>
                            </div>
                          </div>

                          <div style='font-size:0.62rem;color:#AAAAAA;font-family:monospace;
                                       margin-bottom:4px'>{_fv.key_metric}</div>
                          <div style='font-size:0.60rem;color:#AAAAAA;font-style:italic'>
                            {_fv.regime_note}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("<div style='height:8px'></div>",
                            unsafe_allow_html=True)

            # ── Portfolio-weighted valuation score ────────────────────────────
            with st.expander("📐 Portfolio-Weighted Fair Value Score", expanded=False):
                _portf_positions = _val_portf.get("positions", [])
                _total_mv = sum(float(p.get("market_value", 0))
                                for p in _portf_positions)

                if _portf_positions and _total_mv > 0:
                    _wt_upside = 0.0
                    _wt_rows = []
                    for _p in _portf_positions:
                        _sym = _p.get("symbol", "")
                        _mv = float(_p.get("market_value", 0))
                        _w = _mv / _total_mv
                        _fv_est = _fv_results.get(_sym)
                        if _fv_est:
                            _wt_upside += _w * _fv_est.upside_pct
                            _wt_rows.append({
                                "Symbol": _sym,
                                "Weight": f"{_w:.1%}",
                                "Upside": f"{_fv_est.upside_pct:+.1f}%",
                                "Signal": _fv_est.signal,
                                "Contribution": f"{_w * _fv_est.upside_pct:+.2f}%",
                            })

                    if _wt_rows:
                        _score_color = "#00c851" if _wt_upside > 2 else (
                                       "#ff4444" if _wt_upside < -2 else "#9e9e9e")
                        st.markdown(
                            f"""<div style='background:#0d1a0d;border:1px solid #1a2a1a;
                                             border-radius:6px;padding:12px 16px;
                                             margin-bottom:12px;text-align:center'>
                                  <div style='font-size:0.62rem;color:#AAAAAA;font-family:monospace'>
                                    Weighted Portfolio Fair Value Gap</div>
                                  <div style='font-size:1.8rem;font-weight:700;
                                               color:{_score_color};font-family:monospace'>
                                    {_wt_upside:+.1f}%</div>
                                  <div style='font-size:0.60rem;color:#AAAAAA'>
                                    {"Portfolio appears undervalued relative to model fair value"
                                     if _wt_upside > 2 else
                                     "Portfolio appears overvalued relative to model fair value"
                                     if _wt_upside < -2 else
                                     "Portfolio near model fair value"}</div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            pd.DataFrame(_wt_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.info(
                        "No open positions found.  The portfolio-weighted score "
                        "will appear once positions are loaded from Alpaca."
                    )

            # ── Methodology note ──────────────────────────────────────────────
            with st.expander("📖 Valuation Methodology", expanded=False):
                st.markdown(
                    """
<div style='font-size:0.67rem;color:#AAAAAA;font-family:monospace;line-height:1.8'>

| ETF | Model | Key Driver |
|-----|-------|------------|
| **SPY** | Fed Model | Earnings Yield (1/FwdPE) vs. 10Y Treasury |
| **QQQ** | Adjusted Fed Model | Nasdaq growth premium of −1.5% on hurdle yield |
| **IWM** | Fed Model + Risk Premium | Small-cap +1.0% required yield above 10Y |
| **TLT** | Duration Gap | Modified duration × (10Y − 3.5% neutral) |
| **SHY** | Par-Anchored | 2yr duration, always near par |
| **GLD** | Real Rate Sensitivity | −8%/100bps TIPS real yield move |
| **VNQ** | REIT Yield Spread | Dividend yield vs. 10Y + 150bps fair spread |
| **DBC** | 52-Week Momentum | Price position within 52-week high/low range |

**Important caveats**
- All models are simplified regime-aware heuristics, not formal DCF models.
- ETF fair values are inherently harder to pin than single-stock DCF because they
  reflect macro factor exposures rather than company cash flows.
- Use as a directional input alongside the HMM regime and conviction scores —
  not as a standalone buy/sell signal.
- Yield data cached hourly via yfinance; P/E ratios may lag 1–2 days.

</div>
                    """,
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB D — GBM RISK SIMULATOR
    # ══════════════════════════════════════════════════════════════════════════
    with _v_gbm_tab:
        st.markdown(
            "<p style='font-size:0.72rem;color:#AAAAAA;margin-bottom:12px'>"
            "Geometric Brownian Motion — 10 000 paths · VaR 95%/99% · "
            "CVaR (Expected Shortfall) at 30 / 90 / 252-day horizons.</p>",
            unsafe_allow_html=True,
        )

        @st.cache_data(ttl=1800, show_spinner=False)
        def _run_gbm_sim(weights_json: str, port_value: float) -> Dict[str, Any]:
            import json as _json
            import sys as _sys
            _core_p = str(_HERE / "core")
            if _core_p not in _sys.path:
                _sys.path.insert(0, _core_p)
            from monte_carlo import RiskSimulator  # type: ignore
            _weights = _json.loads(weights_json)
            _sim = RiskSimulator(_weights, port_value)
            _res = _sim.run([30, 90, 252])
            _max_h = _res.paths.shape[1]
            _percs = np.percentile(_res.paths, [10, 25, 50, 75, 90], axis=0)
            return {
                "horizons": {
                    int(k): {sk: float(sv) for sk, sv in v.items()}
                    for k, v in _res.horizons.items()
                },
                "fan_x":      list(range(1, _max_h + 1)),
                "fan_p10":    _percs[0].tolist(),
                "fan_p25":    _percs[1].tolist(),
                "fan_median": _percs[2].tolist(),
                "fan_p75":    _percs[3].tolist(),
                "fan_p90":    _percs[4].tolist(),
            }

        # Build weights from portfolio positions (list of dicts)
        _gbm_pos_list = _val_portf.get("positions", [])
        _gbm_equity = max(_val_equity, 1.0)
        if isinstance(_gbm_pos_list, list) and _gbm_pos_list:
            _gbm_weights = {
                p["symbol"]: float(p.get("market_value", 0.0)) / _gbm_equity
                for p in _gbm_pos_list
                if float(p.get("market_value", 0.0)) > 0
            }
        else:
            _gbm_weights = {"SPY": 0.50, "QQQ": 0.30, "GLD": 0.20}
            st.info(
                "No live positions found — showing demo simulation "
                "(SPY 50% / QQQ 30% / GLD 20%). "
                "Load Alpaca credentials to simulate your actual portfolio."
            )

        _gbm_c1, _gbm_c2 = st.columns([4, 1])
        with _gbm_c2:
            _gbm_pv = st.number_input(
                "Portfolio Value ($)",
                min_value=1_000,
                max_value=50_000_000,
                value=int(_gbm_equity),
                step=1_000,
                key="gbm_port_val",
            )
        with _gbm_c1:
            _gbm_wt_str = "  ·  ".join(
                f"{s} {w:.0%}" for s, w in list(_gbm_weights.items())[:8]
            )
            st.markdown(
                f"<p style='font-size:0.63rem;color:#AAAAAA;margin-top:26px;"
                f"font-family:monospace'>"
                f"{len(_gbm_weights)} positions: {_gbm_wt_str}"
                f"{'  ...' if len(_gbm_weights) > 8 else ''}</p>",
                unsafe_allow_html=True,
            )

        _gbm_run = st.button("Run GBM Simulation",
                             key="gbm_run", type="primary")
        if _gbm_run or "gbm_result" not in st.session_state:
            with st.spinner("Running 10 000 GBM paths..."):
                import json as _json_gbm
                _gbm_res = _run_gbm_sim(
                    _json_gbm.dumps(_gbm_weights),
                    float(_gbm_pv),
                )
                st.session_state["gbm_result"] = _gbm_res
        else:
            _gbm_res = st.session_state["gbm_result"]

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _gbm_hs = [30, 90, 252]
        _gbm_hls = ["30 Days", "90 Days", "1 Year"]
        _gbm_kc = st.columns(len(_gbm_hs) * 4)

        for _hi, (h, hl) in enumerate(zip(_gbm_hs, _gbm_hls)):
            _hd = _gbm_res["horizons"].get(h, {})
            _er = _hd.get("expected_return", 0.0)
            _v95 = _hd.get("var95", 0.0)
            _v99 = _hd.get("var99", 0.0)
            _cv = _hd.get("cvar95", 0.0)
            _rc = "#00c851" if _er >= 0 else "#ff4444"
            _b = _hi * 4

            def _gbm_kpi(col, lbl, val, color="#e0e0e0"):
                col.markdown(
                    f'<div style="background:#111;border:1px solid #1a1a1a;border-radius:6px;'
                    f'padding:7px 6px;text-align:center;margin:2px">'
                    f'<div style="font-size:0.57rem;color:#AAAAAA;font-family:monospace">{lbl}</div>'
                    f'<div style="font-size:0.85rem;font-weight:700;color:{color};'
                    f'font-family:monospace">{val}</div></div>',
                    unsafe_allow_html=True,
                )

            _gbm_kpi(_gbm_kc[_b],   f"{hl} E[R]", f"{_er:+.1%}", _rc)
            _gbm_kpi(_gbm_kc[_b+1], "VaR 95%",    f"-{_v95:.1%}", "#ff8800")
            _gbm_kpi(_gbm_kc[_b+2], "VaR 99%",    f"-{_v99:.1%}", "#ff4444")
            _gbm_kpi(_gbm_kc[_b+3], "CVaR 95%",   f"-{_cv:.1%}",  "#cc0000")

        # ── Fan chart ─────────────────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        _fan_x = _gbm_res["fan_x"]
        _fan_pv = float(_gbm_pv)
        _fig_fan = go.Figure()

        _fig_fan.add_trace(go.Scatter(
            x=_fan_x + _fan_x[::-1],
            y=[_fan_pv + v for v in _gbm_res["fan_p90"]] +
              [_fan_pv + v for v in _gbm_res["fan_p10"]][::-1],
            fill="toself", fillcolor="rgba(0,200,81,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="P10-P90",
        ))
        _fig_fan.add_trace(go.Scatter(
            x=_fan_x + _fan_x[::-1],
            y=[_fan_pv + v for v in _gbm_res["fan_p75"]] +
              [_fan_pv + v for v in _gbm_res["fan_p25"]][::-1],
            fill="toself", fillcolor="rgba(0,200,81,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="P25-P75",
        ))
        _fig_fan.add_trace(go.Scatter(
            x=_fan_x,
            y=[_fan_pv + v for v in _gbm_res["fan_median"]],
            line=dict(color="#00c851", width=2),
            name="Median",
        ))
        _fig_fan.add_hline(y=_fan_pv, line=dict(
            color="#AAAAAA", width=1, dash="dot"))
        apply_pro_theme(
            _fig_fan,
            title="GBM Fan Chart — Portfolio P&L Projection (USD)",
            height=300,
            x_title="Trading Days",
            y_title="Portfolio Value ($)",
        )
        _fig_fan.update_yaxes(tickformat="$,.0f")
        st.plotly_chart(_fig_fan, use_container_width=True,
                        config={"displayModeBar": False})

        with st.expander("GBM Methodology", expanded=False):
            st.markdown(
                """<div style='font-size:0.67rem;color:#AAAAAA;font-family:monospace;line-height:1.8'>

**GBM** — Each asset follows S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
where Z ~ N(0,1). Daily mu and sigma estimated from last 252 trading days (yfinance).
Portfolio return = weighted average of per-asset returns across 10 000 paths.

**VaR 95%** -- 5th percentile of the terminal return distribution.
**VaR 99%** -- 1st percentile.
**CVaR 95%** -- Expected value of returns below VaR 95% (Expected Shortfall).

**Caveats** -- GBM assumes constant volatility and log-normal returns.
Real returns exhibit fat tails and volatility clustering. Use alongside the
regime-aware Monte Carlo (Tab A) for a fuller risk picture.
</div>""",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB E — ML-DCF VALUATION
    # ══════════════════════════════════════════════════════════════════════════
    with _v_ml_tab:
        st.markdown(
            "<p style='font-size:0.72rem;color:#AAAAAA;margin-bottom:12px'>"
            "Ridge regression on FRED macro features predicts revenue growth used in a "
            "5-year DCF. Compare Classic DCF (base case) vs ML-Adjusted DCF.</p>",
            unsafe_allow_html=True,
        )

        @st.cache_data(ttl=1800, show_spinner=False)
        def _run_ml_dcf(ticker: str, wacc: float, tg: float) -> Dict[str, Any]:
            import sys as _sys
            _core_p = str(_HERE / "core")
            if _core_p not in _sys.path:
                _sys.path.insert(0, _core_p)
            from valuation_ml import MLDCFEngine  # type: ignore
            try:
                _r = MLDCFEngine(wacc=wacc, terminal_growth=tg).value(ticker)
                return {
                    "ticker":         _r.ticker,
                    "price":          _r.current_price,
                    "classic_fv":     _r.classic_fv,
                    "ml_fv":          _r.ml_fv,
                    "classic_upside": _r.classic_upside,
                    "ml_upside":      _r.ml_upside,
                    "base_growth":    _r.base_growth,
                    "ml_growth":      _r.ml_growth,
                    "model":          _r.model,
                    "confidence":     _r.macro_confidence,
                    "error":          None,
                }
            except Exception as _exc:
                return {"ticker": ticker, "error": str(_exc)}

        # Controls
        _ml_c1, _ml_c2, _ml_c3, _ml_c4 = st.columns([3, 1, 1, 1])
        with _ml_c1:
            _pos_list = _val_portf.get("positions", [])
            _pos_syms = [p["symbol"]
                         for p in _pos_list if isinstance(p, dict)][:6]
            _ml_default = ", ".join(
                _pos_syms) if _pos_syms else "AAPL, MSFT, NVDA, AMZN"
            _ml_tickers_raw = st.text_input(
                "Tickers (comma-separated)",
                value=_ml_default,
                key="ml_tickers",
            )
        with _ml_c2:
            _ml_wacc = st.number_input(
                "WACC (%)", min_value=4.0, max_value=20.0,
                value=9.0, step=0.5, key="ml_wacc",
            ) / 100
        with _ml_c3:
            _ml_tg = st.number_input(
                "Terminal Growth (%)", min_value=0.5, max_value=6.0,
                value=2.5, step=0.5, key="ml_tg",
            ) / 100
        with _ml_c4:
            _ml_run = st.button("Value", key="ml_run", type="primary")

        _ml_tickers = [t.strip().upper()
                       for t in _ml_tickers_raw.split(",") if t.strip()]

        if _ml_run or "ml_results" not in st.session_state:
            with st.spinner(f"Running ML-DCF for {', '.join(_ml_tickers)}..."):
                _ml_results = [_run_ml_dcf(t, _ml_wacc, _ml_tg)
                               for t in _ml_tickers]
                st.session_state["ml_results"] = _ml_results
        else:
            _ml_results = st.session_state["ml_results"]

        # Summary table
        _ml_rows = []
        for _r in _ml_results:
            if _r.get("error"):
                _ml_rows.append({
                    "Ticker":      _r["ticker"],
                    "Price":       "--",
                    "Classic FV":  "--",
                    "ML FV":       "--",
                    "Classic Gap": "error",
                    "ML Gap":      str(_r["error"])[:50],
                    "Base Growth": "--",
                    "ML Growth":   "--",
                    "Confidence":  "--",
                })
            else:
                _cu, _mu = _r["classic_upside"], _r["ml_upside"]
                _ml_rows.append({
                    "Ticker":      _r["ticker"],
                    "Price":       f"${_r['price']:.2f}",
                    "Classic FV":  f"${_r['classic_fv']:.2f}",
                    "ML FV":       f"${_r['ml_fv']:.2f}",
                    "Classic Gap": f"{'+'if _cu >= 0 else ''}{_cu:.1f}%",
                    "ML Gap":      f"{'+'if _mu >= 0 else ''}{_mu:.1f}%",
                    "Base Growth": f"{_r['base_growth']:.1%}",
                    "ML Growth":   f"{_r['ml_growth']:.1%}",
                    "Confidence":  f"{_r['confidence']:.0%}",
                })

        if _ml_rows:
            st.dataframe(
                pd.DataFrame(_ml_rows),
                use_container_width=True,
                hide_index=True,
            )

        # ── DCF Waterfall — Classic FV → Macro Adjustment → ML-DCF ───────────
        _wf_valid = [r for r in _ml_results
                     if not r.get("error") and r.get("classic_fv", 0) > 0]
        if _wf_valid:
            _wf_fig = dcf_waterfall_fig(_wf_valid[:4])
            st.plotly_chart(_wf_fig, use_container_width=True,
                            config={"displayModeBar": False})

        # Per-ticker cards (first 4)
        _ml_valid = [r for r in _ml_results if not r.get("error")]
        if _ml_valid:
            st.markdown("<div style='height:8px'></div>",
                        unsafe_allow_html=True)
            _ml_card_cols = st.columns(min(len(_ml_valid), 4))
            for _ci, _r in enumerate(_ml_valid[:4]):
                _cu = _r["classic_upside"]
                _mu = _r["ml_upside"]
                _cc = "#00c851" if _cu >= 0 else "#ff4444"
                _mc = "#00c851" if _mu >= 0 else "#ff4444"
                with _ml_card_cols[_ci]:
                    st.markdown(
                        f'<div style="background:#0a0a0a;border:1px solid #1a1a1a;'
                        f'border-radius:8px;padding:12px;margin-top:8px">'
                        f'<div style="font-size:0.82rem;font-weight:900;color:#e0e0e0;'
                        f'font-family:monospace">{_r["ticker"]}</div>'
                        f'<div style="font-size:0.60rem;color:#AAAAAA;margin-bottom:6px">'
                        f'Price: ${_r["price"]:.2f}</div>'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                        f'<span style="font-size:0.63rem;color:#AAAAAA">Classic DCF</span>'
                        f'<span style="font-size:0.75rem;font-weight:700;color:{_cc};'
                        f'font-family:monospace">{"+"if _cu >= 0 else ""}{_cu:.1f}%</span>'
                        f'</div>'
                        f'<div style="display:flex;justify-content:space-between">'
                        f'<span style="font-size:0.63rem;color:#AAAAAA">ML-DCF</span>'
                        f'<span style="font-size:0.75rem;font-weight:700;color:{_mc};'
                        f'font-family:monospace">{"+"if _mu >= 0 else ""}{_mu:.1f}%</span>'
                        f'</div>'
                        f'<div style="font-size:0.58rem;color:#AAAAAA;margin-top:5px;font-family:monospace">'
                        f'Growth: {_r["base_growth"]:.1%} -> {_r["ml_growth"]:.1%} (ML)</div>'
                        f'<div style="font-size:0.55rem;color:#AAAAAA;margin-top:1px">'
                        f'Macro conf: {_r["confidence"]:.0%} · {_r["model"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with st.expander("ML-DCF Methodology", expanded=False):
            st.markdown(
                """<div style='font-size:0.67rem;color:#AAAAAA;font-family:monospace;line-height:1.8'>

**Classic DCF** -- 5-year EPS-based DCF using yfinance revenueGrowth as the base growth
rate, discounted at WACC. Terminal value: Gordon Growth Model.

**ML-DCF** -- Ridge regression trained on a synthetic macro-to-growth dataset anchored
on FRED indicators (US Leading Index, Consumer Confidence, OECD Business PMI).
Current macro readings adjust the predicted revenue growth rate up or down vs the base.

**Confidence** -- fraction of FRED series fetched (0% = no macro adjustment applied).

**Caveats** -- EPS-based DCF is a simplification. Accuracy depends on yfinance data
quality. Use as a directional premium/discount signal, not a precise price target.
</div>""",
                unsafe_allow_html=True,
            )
