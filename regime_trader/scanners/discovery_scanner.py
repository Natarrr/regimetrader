"""regime_trader/discovery_scanner.py
Smart Money Discovery Scanner — v3.0

Akerlof (2001 Nobel) — Information asymmetry: insiders buying with their own
money is a credible signal precisely because it is costly to fake.

Pipeline:
  1. fmp_screener()                  → liquid momentum candidates
  2. fmp_insider_buys()              → CEO/CFO/Director open-market purchases
  3. fmp_institutional_accumulation()→ per-symbol net institutional change
  4. select_candidates()             → insider stocks guaranteed entry
  5. _smart_money_prescore()         → 0.45·insider + 0.35·inst + 0.20·momentum
  6. run_scan()                      → final ranking, smart-money first

Design:
  • Functional: every public function is pure or documents its side-effects.
  • Async-first with sync wrappers for Streamlit.
  • Any single API failure degrades gracefully — never crashes the loop.
  • FMP free-tier aware: institutional calls batched to ≤ 50 symbols.

CLI:
  python -m regime_trader.scanners.discovery_scanner --limit 5
  python -m regime_trader.scanners.discovery_scanner --force-refresh --limit 5
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from regime_trader.utils.io import load_json_safe, save_json_atomic
from regime_trader.services.fmp_client import FMPClient as _FMPClient

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_FMP_BASE    = "https://financialmodelingprep.com/api"
_FMP_STABLE  = "https://financialmodelingprep.com/stable"
_TIMEOUT = 15
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 0.8

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_KEY_ROLES = {
    "CEO", "CFO", "COO", "CTO", "DIRECTOR", "PRESIDENT",
    "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING",
    "CHAIRMAN", "FOUNDER",
}

_MIN_KEY_INSIDER_USD = 25_000.0
_MIN_NORMALIZED_SIGNAL = 0.00005
_SCREENER_MIN_MCAP = 200_000_000
_SCREENER_MIN_VOLUME = 200_000

_W_INSIDER = 0.45
_W_INST = 0.35
_W_MOMENTUM = 0.20

_DISC_CACHE_FILE = Path(__file__).parent.parent / "logs" / "discovery_cache.json"
_DISC_CACHE_TTL = 6 * 3600

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=12, thread_name_prefix="scanner"
)

_FALLBACK_PICKS: List[str] = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",
    "AMZN", "TSLA", "JPM", "V", "UNH",
]

# Curated watchlist used by the yfinance-based screener.
# ~130 liquid US large/mid caps spanning all 11 GICS sectors.
_YF_WATCHLIST: List[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "AMD", "INTC", "QCOM",
    "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "TXN", "NOW", "AMAT", "LRCX",
    # Healthcare
    "JNJ", "LLY", "ABBV", "MRK", "PFE", "UNH", "ABT", "MDT", "BMY", "TMO",
    "DHR", "SYK", "ISRG", "VRTX", "REGN",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA",
    "PGR", "CB", "ICE", "CME",
    # Energy
    "XOM", "CVX", "COP", "SLB", "PSX", "VLO", "MPC", "HAL", "OXY",
    # Industrials
    "CAT", "DE", "HON", "GE", "BA", "LMT", "RTX", "NOC", "UPS", "FDX",
    "ETN", "PH", "GD", "WM",
    # Consumer Discretionary
    "TSLA", "NKE", "MCD", "SBUX", "HD", "LOW", "TGT", "TJX", "BKNG", "CMG",
    "ABNB", "GM", "F",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KHC", "GIS", "STZ",
    # Materials
    "LIN", "APD", "SHW", "NEM", "FCX", "ALB", "DD", "IFF",
    # Communication Services
    "NFLX", "DIS", "VZ", "CMCSA", "T", "CHTR", "SNAP", "ROKU",
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "O", "SPG", "EQR",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC",
]


# ── Result schema ──────────────────────────────────────────────────────────────

class ScanResult(TypedDict):
    symbol: str
    smart_money_score: float
    insider_score: float
    institutional_score: float
    momentum_score: float
    insider_value_usd: float
    insider_value_pct_mcap: float
    key_insider_roles: List[str]
    institutional_net_shares: float
    institutional_pct_change: float
    volume_spike: float
    price_change_pct: float
    market_cap: float
    source_flags: List[str]


# ── HTTP session with retries ──────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Build a requests.Session with exponential-backoff retry logic.

    Granger (2003 Nobel) — reliable data collection precedes causal inference.
    """
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(_HEADERS)
    return session


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = _make_session()
    return _session


def disc_get_json(
    url: str,
    params: Optional[Dict] = None,
    timeout: int = _TIMEOUT,
) -> Any:
    """GET *url* and return parsed JSON; return None on any error.

    Akerlof (2001 Nobel) — reliable data retrieval with hard timeout.

    Args:
        url:     Full URL to request.
        params:  Query parameters dict.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON (list or dict) or None on failure.
    """
    try:
        resp = _get_session().get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        log.warning("HTTP %s from %s", resp.status_code, url.split("?")[0])
        return None
    except Exception as exc:
        log.warning("Request failed %s: %s", url.split("?")[0], exc)
        return None


def _fmp_key() -> str:
    return os.getenv("FMP_API_KEY", "")


# ── Step 1: FMP screener ───────────────────────────────────────────────────────

def fmp_screener(limit: int = 200) -> List[Dict]:
    """Fama (2013 Nobel) — screen liquid US equities via yfinance momentum/volume.

    FMP v3/v4 screener endpoints were retired Aug 2025. This implementation
    downloads 30-day OHLCV for the curated _YF_WATCHLIST and ranks by volume
    spike + 5-day price momentum, preserving the same return schema.

    Args:
        limit: Max candidates to return.

    Returns:
        List of dicts: {sym, market_cap, price, volume, volume_spike,
                        price_change_pct, avg_volume, sector}
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        log.warning("[SCREENER] yfinance not installed — screener skipped")
        return []

    syms = _YF_WATCHLIST
    try:
        raw = yf.download(
            syms, period="30d", interval="1d",
            progress=False, auto_adjust=True, threads=True,
        )
    except Exception as exc:
        log.warning("[SCREENER] yfinance batch download failed: %s", exc)
        return []

    results = []
    for sym in syms:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(sym, level=1, axis=1).dropna(how="all")
            else:
                df = raw.dropna(how="all")

            if df.empty or len(df) < 6:
                continue

            close = df["Close"].squeeze().astype(float)
            volume = df["Volume"].squeeze().astype(float)

            price = float(close.iloc[-1])
            today_vol = float(volume.iloc[-1])
            avg_vol = float(volume.iloc[:-1].mean()) if len(volume) > 1 else today_vol
            volume_spike = round(today_vol / max(avg_vol, 1.0), 3)
            price_change_pct = round(
                (price / float(close.iloc[-6]) - 1) * 100, 4
            ) if len(close) >= 6 else 0.0

            if price < 1.0 or today_vol * price < _SCREENER_MIN_VOLUME:
                continue

            results.append({
                "sym": sym,
                "market_cap": 0.0,
                "price": round(price, 4),
                "volume": round(today_vol, 0),
                "avg_volume": round(avg_vol, 0),
                "volume_spike": volume_spike,
                "price_change_pct": price_change_pct,
                "sector": "",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["volume_spike"] + abs(x["price_change_pct"]) / 10, reverse=True)
    log.info("[SCREENER] %d candidates from yfinance watchlist", len(results))
    return results[:limit]


# ── Step 2: FMP insider buys ───────────────────────────────────────────────────

def fmp_insider_buys(limit: int = 500) -> List[Dict]:
    """Insider open-market purchases via FMP stable/insider-trading/search.

    Insider conviction signal (Cohen, Malloy & Pomorski 2012): opportunistic
    CEO/CFO purchases carry measurable forward alpha (~7% annualised).

    Uses FMPClient.get_insider_purchases() (stable/ route, PASS in Phase-0
    smoke-test) for total acquisition USD and recency, then batch-quotes for
    market caps. Replaced yfinance insider_transactions which scraped from
    a fragile SEC HTML endpoint with no rate control.

    Args:
        limit: Max symbols to return (sorted by normalized_pct_mcap desc).

    Returns:
        List of dicts (same schema as before):
        {sym, key_value_usd, normalized_pct_mcap, market_cap, roles,
         tx_count, most_recent_date}
    """
    client = _FMPClient()
    if not client._api_key:
        log.warning("[INSIDER] FMP_API_KEY not set — insider discovery skipped")
        return []

    def _fetch_insider(sym: str) -> Optional[Dict]:
        try:
            total_usd, days_since = client.get_insider_purchases(sym, lookback_days=180)
            if total_usd < _MIN_KEY_INSIDER_USD:
                return None
            return {
                "sym": sym,
                "key_value_usd": round(total_usd, 0),
                "normalized_pct_mcap": 0.0,
                "market_cap": 0.0,
                "roles": [],           # FMP purchase summary doesn't split by role
                "tx_count": 1,         # conservative: at least one qualifying purchase
                "most_recent_date": f"-{days_since}d" if days_since else "",
            }
        except Exception:
            return None

    raw_results: List[Dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_fetch_insider, sym): sym for sym in _YF_WATCHLIST}
        for fut in concurrent.futures.as_completed(futures, timeout=90):
            r = fut.result()
            if r is not None:
                raw_results.append(r)

    if not raw_results:
        log.warning("[INSIDER] No key-insider purchases found in watchlist")
        return []

    # Batch market caps in one call instead of N serial profile calls
    batch_caps = client.get_batch_quotes([r["sym"] for r in raw_results])
    results = []
    for r in raw_results:
        mcap = float(batch_caps.get(r["sym"], {}).get("marketCap", 0) or 0)
        key_val = r["key_value_usd"]
        norm_pct = round((key_val / mcap * 100) if mcap > 1e6 else 0.0, 6)
        if norm_pct < _MIN_NORMALIZED_SIGNAL * 100:
            continue
        r["market_cap"] = mcap
        r["normalized_pct_mcap"] = norm_pct
        results.append(r)

    results.sort(key=lambda x: x["normalized_pct_mcap"], reverse=True)
    log.info("[INSIDER] %d symbols with key-insider buys (FMP stable/)", len(results))
    return results[:limit]


# ── Profile batch ──────────────────────────────────────────────────────────────

def fmp_profile_batch(symbols: List[str]) -> Dict[str, float]:
    """Fetch market caps via FMP stable/batch-quote (one call for all symbols).

    Uses FMPClient.get_batch_quotes() which confirmed PASS in Phase-0 smoke-test.
    Replaces N serial stable/profile calls with a single bulk call.

    Returns:
        {sym: market_cap_float}  — 0.0 for missing symbols (safe for division).
    """
    if not symbols:
        return {}
    client = _FMPClient()
    if not client._api_key:
        return {}
    batch = client.get_batch_quotes(symbols)
    return {
        sym: float(row.get("marketCap", 0) or 0)
        for sym, row in batch.items()
    }


def fmp_profile(sym: str) -> Optional[Dict]:
    """Fetch a single ticker's FMP profile via stable/profile.

    Returns:
        Profile dict or None if unavailable.
    """
    key = _fmp_key()
    if not key:
        return None
    data = disc_get_json(
        f"{_FMP_STABLE}/profile",
        params={"symbol": sym, "apikey": key},
    )
    if isinstance(data, list) and data:
        return data[0]
    return None


# ── Step 3: Institutional accumulation ────────────────────────────────────────

def fmp_institutional_accumulation(
    candidate_symbols: List[str],
    limit: int = 50,
) -> List[Dict]:
    """Institutional accumulation via yfinance 13F data.

    FMP stable/institutional-ownership/symbol-positions-summary returned HTTP 400
    in the Phase-0 smoke-test (2026-05-30) — not available on current plan.
    yfinance institutional_holders (quarterly 13F-sourced) is the documented
    fallback until FMP restores the 13F endpoint. Keep return schema unchanged.

    Args:
        candidate_symbols: Pool of tickers to check.
        limit:             Max symbols to query.

    Returns:
        List of dicts sorted by accumulation_score descending:
        {sym, net_shares_change, pct_change_avg, major_fund_count,
         holder_count, accumulation_score}
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("[INST] yfinance not installed — institutional data skipped")
        return []

    symbols = list(dict.fromkeys(s.upper() for s in candidate_symbols))[:limit]
    _MAJOR = {"VANGUARD", "BLACKROCK", "STATE STREET", "FIDELITY",
              "JP MORGAN", "RENAISSANCE", "CITADEL", "MILLENNIUM"}

    def _fetch_one(sym: str) -> Optional[Dict]:
        try:
            ih = yf.Ticker(sym).institutional_holders
            if ih is None or ih.empty:
                return None

            pct_changes = ih["pctChange"].dropna().tolist() if "pctChange" in ih.columns else []
            if not pct_changes:
                return None

            total_shares = float(ih["Shares"].sum()) if "Shares" in ih.columns else 0.0
            avg_pct = sum(pct_changes) / len(pct_changes)
            net_change = avg_pct * total_shares / 100.0

            major_count = 0
            if "Holder" in ih.columns and "pctChange" in ih.columns:
                for _, row in ih.iterrows():
                    name = str(row.get("Holder", "")).upper()
                    chg = float(row.get("pctChange", 0) or 0)
                    if any(m in name for m in _MAJOR) and chg > 0:
                        major_count += 1

            raw_acc = 0.50 + min(0.40, max(-0.40, avg_pct * 8))
            major_boost = min(0.15, major_count * 0.04)
            acc_score = min(1.0, max(0.0, raw_acc + major_boost))

            return {
                "sym": sym,
                "net_shares_change": round(net_change, 0),
                "pct_change_avg": round(avg_pct, 6),
                "major_fund_count": major_count,
                "holder_count": len(ih),
                "accumulation_score": round(acc_score, 4),
            }
        except Exception:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures, timeout=90):
            r = fut.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda x: x["accumulation_score"], reverse=True)
    log.info("[INST] Institutional data for %d/%d symbols", len(results), len(symbols))
    return results


# ── Step 4: Liquidity filter ───────────────────────────────────────────────────

def liquidity_filter(
    tickers: List[Dict],
    min_dollar_vol: float = 1_000_000,
) -> List[Dict]:
    """Remove illiquid tickers below a dollar-volume threshold.

    Args:
        tickers:         List of screener dicts with 'price' and 'volume'.
        min_dollar_vol:  Minimum price × volume (default $1M).

    Returns:
        Filtered list.
    """
    return [
        t for t in tickers
        if float(t.get("price", 0) or 0) > 0.50
        and float(t.get("price", 0) or 0) * float(t.get("volume", 0) or 0) >= min_dollar_vol
    ]


# ── Step 5: Momentum enrichment ────────────────────────────────────────────────

def enrich_with_momentum(
    candidates: List[Dict],
    max_workers: int = 8,
) -> List[Dict]:
    """Add volume_spike and price_change_pct via yfinance (threadpool).

    Fama (2013 Nobel) — price momentum as efficient-market proxy signal.

    Each symbol is fetched independently; failures default to (spike=1.0, chg=0.0)
    so the list is never shorter than the input.

    Args:
        candidates:  List of dicts with a 'sym' key.
        max_workers: Thread-pool bound (default 8, respects yfinance rate limit).

    Returns:
        Same list with 'volume_spike' and 'price_change_pct' added in-place.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("[MOMENTUM] yfinance not installed — momentum enrichment skipped")
        return candidates

    import pandas as pd

    syms = [str(r.get("sym", "")) for r in candidates if r.get("sym")]
    if not syms:
        return candidates

    def _fetch(sym: str) -> Tuple[str, float, float]:
        try:
            df = yf.download(sym, period="25d", interval="1d",
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 6:
                return sym, 1.0, 0.0
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            close = df["Close"].squeeze().astype(float)
            volume = df["Volume"].squeeze().astype(float)
            avg_vol = float(volume.iloc[:-1].mean()) if len(volume) > 1 else 1.0
            spike = round(float(volume.iloc[-1]) / max(avg_vol, 1), 2)
            price_chg = round(
                (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100, 2
            ) if len(close) >= 6 else 0.0
            return sym, spike, price_chg
        except Exception:
            return sym, 1.0, 0.0

    enrichment: Dict[str, Tuple[float, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, s): s for s in syms}
        for fut in concurrent.futures.as_completed(futures):
            sym, spike, pchg = fut.result()
            enrichment[sym] = (spike, pchg)

    for r in candidates:
        sym = str(r.get("sym", ""))
        r["volume_spike"], r["price_change_pct"] = enrichment.get(sym, (1.0, 0.0))

    return candidates


# ── Step 6: Candidate selection ────────────────────────────────────────────────

def select_candidates(
    screener_results: List[Dict],
    insider_buys: List[Dict],
    n: int = 50,
) -> Tuple[List[str], Dict[str, str]]:
    """Guarantee insider stocks enter the candidate pool; screener fills slack.

    Akerlof (2001 Nobel) — the intersection trap: insider stocks were silently
    dropped unless they happened to appear in screener results. Fixed: top
    insider stocks are guaranteed seats regardless of screener coverage.

    Edge cases:
        - Zero market cap symbols: included via insider path, score normalised to 0.
        - Missing profile: skipped with a warning.
        - Empty screener: candidates drawn exclusively from insider_buys.
        - Empty insider_buys: candidates drawn exclusively from screener.

    Args:
        screener_results: From fmp_screener(), pre-fetched.
        insider_buys:     From fmp_insider_buys(), sorted by norm signal.
        n:                Total candidate pool size.

    Returns:
        (selected_symbols, source_map)
        source_map: {sym -> "insider" | "screener" | "both"}
    """
    guaranteed = [e["sym"] for e in insider_buys if e.get("sym")]
    screener_syms = [e["sym"] for e in screener_results if e.get("sym")]
    screener_set = set(screener_syms)

    selected: List[str] = []
    source_map: Dict[str, str] = {}

    for sym in guaranteed:
        if sym not in source_map:
            selected.append(sym)
            source_map[sym] = "insider"

    for sym in selected:
        if sym in screener_set:
            source_map[sym] = "both"

    remaining = max(0, n - len(selected))
    for sym in screener_syms:
        if remaining == 0:
            break
        if sym not in source_map:
            selected.append(sym)
            source_map[sym] = "screener"
            remaining -= 1

    log.info(
        "[CANDIDATES] %d total | %d insider | %d screener-only | %d both",
        len(selected),
        sum(1 for v in source_map.values() if v == "insider"),
        sum(1 for v in source_map.values() if v == "screener"),
        sum(1 for v in source_map.values() if v == "both"),
    )
    return selected, source_map


# ── Step 7: Smart Money Pre-Score ──────────────────────────────────────────────

def _smart_money_prescore(
    sym: str,
    insider_map: Dict[str, Dict],
    inst_map: Dict[str, Dict],
    screener_map: Dict[str, Dict],
) -> Tuple[float, float, float, float]:
    """Granger (2003 Nobel) — causal precedence of smart-money over momentum.

    Weights: insider 45%, institutional 35%, momentum 20%.
    Insider stocks with flat price action outrank pure momentum plays.

    Returns:
        (composite, insider_component, inst_component, momentum_component)
        All values in [0, 1].
    """
    ins_data = insider_map.get(sym)
    if ins_data:
        norm_pct = ins_data.get("normalized_pct_mcap", 0.0)
        ins_raw = min(1.0, norm_pct / 0.5)
        ins_score = 0.30 + 0.70 * ins_raw
    else:
        ins_score = 0.0

    inst_data = inst_map.get(sym)
    if inst_data:
        acc = inst_data.get("accumulation_score", 0.5)
        inst_score = max(0.0, (acc - 0.50) * 2.0)
    else:
        inst_score = 0.0

    scr_data = screener_map.get(sym)
    if scr_data:
        vs = scr_data.get("volume_spike", 1.0)
        chg = scr_data.get("price_change_pct", 0.0)
        vs_sc = min(1.0, max(0.0, (vs - 1.0) / 4.0))
        chg_sc = min(1.0, max(0.0, 0.5 + chg / 10.0))
        mom_score = 0.5 * vs_sc + 0.5 * chg_sc
    else:
        mom_score = 0.0

    composite = _W_INSIDER * ins_score + _W_INST * inst_score + _W_MOMENTUM * mom_score
    return (
        round(composite, 4),
        round(ins_score, 4),
        round(inst_score, 4),
        round(mom_score, 4),
    )


# ── Async wrappers ─────────────────────────────────────────────────────────────

async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ── Main scan entry point ──────────────────────────────────────────────────────

async def run_scan_async(n: int = 20) -> List[ScanResult]:
    """Akerlof + Tirole — parallel smart money discovery.

    Screener, insider buys, and institutional accumulation run concurrently;
    results are merged and ranked by smart-money composite score.

    Args:
        n: Number of top results to return.

    Returns:
        List[ScanResult] sorted by smart_money_score descending.
    """
    log.info("[SCAN] Starting smart-money discovery scan (n=%d)", n)
    t0 = time.monotonic()

    screener_raw, insider_raw = await asyncio.gather(
        _run_in_executor(fmp_screener, 200),
        _run_in_executor(fmp_insider_buys, 500),
        return_exceptions=True,
    )

    if isinstance(screener_raw, Exception):
        log.warning("[SCAN] Screener failed: %s", screener_raw)
        screener_raw = []
    if isinstance(insider_raw, Exception):
        log.warning("[SCAN] Insider fetch failed: %s", insider_raw)
        insider_raw = []

    candidates, source_map = select_candidates(
        screener_results=screener_raw,
        insider_buys=insider_raw,
        n=100,
    )

    inst_raw = await _run_in_executor(
        fmp_institutional_accumulation, candidates, 50
    )
    if isinstance(inst_raw, Exception):
        log.warning("[SCAN] Institutional fetch failed: %s", inst_raw)
        inst_raw = []

    screener_map: Dict[str, Dict] = {e["sym"]: e for e in screener_raw}
    insider_map: Dict[str, Dict] = {e["sym"]: e for e in insider_raw}
    inst_map: Dict[str, Dict] = {e["sym"]: e for e in inst_raw}

    results: List[ScanResult] = []
    for sym in candidates:
        composite, ins_sc, inst_sc, mom_sc = _smart_money_prescore(
            sym, insider_map, inst_map, screener_map
        )
        if composite == 0.0:
            continue

        ins_data = insider_map.get(sym, {})
        inst_data = inst_map.get(sym, {})
        scr_data = screener_map.get(sym, {})

        source_flags = [source_map.get(sym, "screener")]
        if sym in insider_map:
            source_flags.append("insider_confirmed")
        if sym in inst_map and inst_data.get("accumulation_score", 0.5) > 0.55:
            source_flags.append("institutional_accumulating")

        results.append(ScanResult(
            symbol=sym,
            smart_money_score=composite,
            insider_score=ins_sc,
            institutional_score=inst_sc,
            momentum_score=mom_sc,
            insider_value_usd=ins_data.get("key_value_usd", 0.0),
            insider_value_pct_mcap=ins_data.get("normalized_pct_mcap", 0.0),
            key_insider_roles=ins_data.get("roles", []),
            institutional_net_shares=inst_data.get("net_shares_change", 0.0),
            institutional_pct_change=inst_data.get("pct_change_avg", 0.0),
            volume_spike=scr_data.get("volume_spike", 0.0),
            price_change_pct=scr_data.get("price_change_pct", 0.0),
            market_cap=(
                ins_data.get("market_cap") or scr_data.get("market_cap", 0.0)
            ),
            source_flags=list(set(source_flags)),
        ))

    results.sort(key=lambda r: r["smart_money_score"], reverse=True)
    top = results[:n]

    elapsed = time.monotonic() - t0
    log.info(
        "[SCAN] Done %.1fs | %d scored | %d returned | insider-led=%d screener-only=%d",
        elapsed, len(results), len(top),
        sum(1 for r in top if "insider_confirmed" in r["source_flags"]),
        sum(1 for r in top if r["source_flags"] == ["screener"]),
    )
    return top


def run_scan(n: int = 20) -> List[ScanResult]:
    """Blocking wrapper for Streamlit / sync callers.

    Granger (2003 Nobel) — synchronous entry for causal signal discovery.
    """
    try:
        return asyncio.run(run_scan_async(n))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(run_scan_async(n))
        finally:
            loop.close()


# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_disc_cache() -> Optional[Dict[str, Any]]:
    """Return cached discovery payload if still within TTL, else None."""
    raw = load_json_safe(_DISC_CACHE_FILE)
    if raw is None:
        return None
    if time.time() > float(raw.get("_expires_at", 0)):
        return None
    out = {k: v for k, v in raw.items() if not k.startswith("_")}
    out["cached"] = True
    return out


def save_disc_cache(payload: Dict[str, Any]) -> None:
    """Atomically persist *payload* to the discovery cache file."""
    save_json_atomic(_DISC_CACHE_FILE, payload)


def _enrich_with_quiver(result_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Quiver deprecated — returns empty quiver payload for UI backward compatibility."""
    for r in result_dicts:
        r.setdefault("quiver", {})
    return result_dicts


def _build_payload(results: List[ScanResult]) -> Dict[str, Any]:
    now = time.time()
    expires_at = now + _DISC_CACHE_TTL
    result_dicts = _enrich_with_quiver([dict(r) for r in results])
    return {
        "results": result_dicts,
        "cached": False,
        "computed_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "cache_expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "_expires_at": expires_at,
    }


# ── Public sync API ────────────────────────────────────────────────────────────

def get_top_alpha_picks_sync(limit: int = 5) -> Dict[str, Any]:
    """Return discovery payload, using cache when fresh.

    Akerlof (2001 Nobel) — synchronous entry for Streamlit callers.

    Args:
        limit: Number of top picks to return.

    Returns:
        JSON-serialisable dict with 'results', 'cached', 'computed_at'.
    """
    cached = load_disc_cache()
    if cached is not None:
        log.info("[DISC] Serving cached discovery results")
        return cached

    results = run_scan(n=limit)
    if not results:
        log.warning("[DISC] Scan returned 0 results — using fallback tickers")
        results = [
            ScanResult(
                symbol=sym,
                smart_money_score=0.0,
                insider_score=0.0,
                institutional_score=0.0,
                momentum_score=0.0,
                insider_value_usd=0.0,
                insider_value_pct_mcap=0.0,
                key_insider_roles=[],
                institutional_net_shares=0.0,
                institutional_pct_change=0.0,
                volume_spike=0.0,
                price_change_pct=0.0,
                market_cap=0.0,
                source_flags=["fallback"],
            )
            for sym in _FALLBACK_PICKS[:limit]
        ]

    payload = _build_payload(results)
    save_disc_cache(payload)
    return payload


def force_refresh_sync(limit: int = 5) -> Dict[str, Any]:
    """Invalidate cache and run a fresh scan.

    Args:
        limit: Number of top picks to return.

    Returns:
        Fresh JSON-serialisable discovery payload.
    """
    try:
        _DISC_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return get_top_alpha_picks_sync(limit=limit)


def explain_result(result: ScanResult) -> str:
    """Return a human-readable summary for a single scan result."""
    lines = [
        f"{result['symbol']}  smart_money={result['smart_money_score']:.3f}",
        f"  Insider  ({_W_INSIDER:.0%}): score={result['insider_score']:.3f}"
        f" | ${result['insider_value_usd']:,.0f}"
        f" = {result['insider_value_pct_mcap']:.4f}% of mktcap"
        f" | roles: {', '.join(result['key_insider_roles']) or 'none'}",
        f"  Inst     ({_W_INST:.0%}): score={result['institutional_score']:.3f}"
        f" | net_shares={result['institutional_net_shares']:+,.0f}"
        f" avg_chg={result['institutional_pct_change']:+.3%}",
        f"  Momentum ({_W_MOMENTUM:.0%}): score={result['momentum_score']:.3f}"
        f" | vol_spike={result['volume_spike']:.2f}x"
        f" price_chg={result['price_change_pct']:+.2f}%",
        f"  Sources: {', '.join(result['source_flags'])}",
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Money Discovery Scanner — regime_trader"
    )
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of top picks to return (default: 5)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass cache and run a fresh scan")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Debug-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    if args.force_refresh:
        payload = force_refresh_sync(limit=args.limit)
    else:
        payload = get_top_alpha_picks_sync(limit=args.limit)

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    _main()
