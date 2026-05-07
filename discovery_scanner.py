"""discovery_scanner.py
Smart Money Discovery Scanner — v2.0

Akerlof (2001 Nobel) — Information asymmetry: insiders buying with their own money
is a credible signal precisely because it is costly to fake.

Data pipeline (all async, failures isolated per symbol):
  1. _fmp_screener()                 → 200 momentum/volume candidates
  2. _fmp_insider_buys()             → recent CEO/CFO/Director open-market purchases,
                                       normalised by market-cap (relative conviction)
  3. _fmp_institutional_accumulation()→ per-symbol institutional net-change (13F proxy)
  4. _select_candidates()            → insider stocks always in; screener fills slack
  5. _smart_money_prescore()         → 0.45·insider + 0.35·inst + 0.20·momentum
  6. run_scan()                      → final ranking, smart-money first

Design rules
────────────
• Functional: every public function is pure or clearly documented as having side-effects.
• Async-first with sync wrapper for Streamlit.
• Any single API failure degrades gracefully — never crashes the loop.
• FMP free-tier aware: institutional calls are batched to ≤ 50 symbols.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import requests

try:
    from log_manager.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)


# ── Environment / constants ────────────────────────────────────────────────────

def _fmp_key() -> str:
    return os.getenv("FMP_API_KEY", "")


_FMP_BASE   = "https://financialmodelingprep.com/api"
_TIMEOUT    = 15
_HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Key insider roles whose purchases carry maximum conviction (Akerlof 2001 — costly signal)
_KEY_ROLES = {
    "CEO", "CFO", "COO", "CTO", "DIRECTOR", "PRESIDENT",
    "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING",
    "CHAIRMAN", "FOUNDER",
}

# Minimum values to filter noise
_MIN_KEY_INSIDER_USD   = 25_000.0    # ignore sub-$25k buys (noise)
_MIN_NORMALIZED_SIGNAL = 0.00005     # 0.005% of market cap minimum
_INSTITUTIONAL_LOOKBACK_DAYS = 90    # 13F window

# Screener defaults
_SCREENER_MIN_MCAP   = 200_000_000   # $200M min market cap
_SCREENER_MIN_VOLUME = 200_000       # daily volume floor

# Score weights for Smart Money Pre-Score
_W_INSIDER = 0.45
_W_INST    = 0.35
_W_MOMENTUM = 0.20

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=12, thread_name_prefix="scanner")


# ── Result schema ──────────────────────────────────────────────────────────────

class ScanResult(TypedDict):
    symbol:                  str
    smart_money_score:       float    # composite [0, 1]
    insider_score:           float    # normalised insider component [0, 1]
    institutional_score:     float    # institutional accumulation component [0, 1]
    momentum_score:          float    # volume/price momentum component [0, 1]
    insider_value_usd:       float    # total key-insider buy value ($)
    insider_value_pct_mcap:  float    # insider buys / market cap (%)
    key_insider_roles:       List[str]
    institutional_net_shares: float   # net shares change across top holders
    institutional_pct_change: float   # avg % change in holdings
    volume_spike:            float    # ratio vs 20-day avg
    price_change_pct:        float    # 1-day price change
    market_cap:              float
    source_flags:            List[str]  # which data sources contributed


# ── Shared HTTP helper ─────────────────────────────────────────────────────────

def _get_json(url: str, params: Optional[Dict] = None, timeout: int = _TIMEOUT) -> Any:
    """Akerlof (2001 Nobel) — fetch with hard timeout; returns None on any error."""
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        log.warning("[SCANNER] HTTP {} from {}", resp.status_code, url.split("?")[0])
        return None
    except Exception as exc:
        log.warning("[SCANNER] Request failed for {}: {}", url.split("?")[0], exc)
        return None


# ── Step 1 foundation: FMP screener ───────────────────────────────────────────

def _fmp_screener_sync(limit: int = 200) -> List[Dict]:
    """Fama (2013 Nobel) — Screen for liquid, actively-traded equities.

    Fetches the FMP stock screener filtered to mid/large-cap names with
    meaningful daily volume. Returns a list of dicts with momentum signals
    that feed the candidate pool, NOT the final ranking.

    Args:
        limit: max stocks to fetch (FMP screener returns up to 250).

    Returns:
        [{sym, market_cap, volume, price_change_pct, volume_spike, avg_volume}]
    """
    key = _fmp_key()
    if not key:
        log.warning("[SCREENER] FMP_API_KEY not set — screener skipped")
        return []

    data = _get_json(
        f"{_FMP_BASE}/v3/stock-screener",
        params={
            "marketCapMoreThan": _SCREENER_MIN_MCAP,
            "volumeMoreThan":    _SCREENER_MIN_VOLUME,
            "isActivelyTraded":  "true",
            "country":           "US",
            "limit":             limit,
            "apikey":            key,
        },
    )
    if not isinstance(data, list):
        log.warning("[SCREENER] Unexpected response format: {}", type(data))
        return []

    results = []
    for item in data:
        try:
            sym         = str(item.get("symbol", "")).upper().strip()
            price       = float(item.get("price", 0) or 0)
            volume      = float(item.get("volume", 0) or 0)
            avg_vol     = float(item.get("avgVolume", volume) or volume)
            mcap        = float(item.get("marketCap", 0) or 0)
            chg_pct     = float(item.get("changesPercentage", 0) or 0)
            sector      = str(item.get("sector", "") or "")
            exchange    = str(item.get("exchangeShortName", "") or "")

            if not sym or mcap < _SCREENER_MIN_MCAP or "." in sym:
                continue
            # Skip ETFs / foreign listings
            if exchange not in ("NYSE", "NASDAQ", "AMEX", ""):
                continue

            volume_spike = volume / avg_vol if avg_vol > 0 else 1.0

            results.append({
                "sym":              sym,
                "market_cap":       mcap,
                "price":            price,
                "volume":           volume,
                "avg_volume":       avg_vol,
                "volume_spike":     round(volume_spike, 3),
                "price_change_pct": round(chg_pct, 4),
                "sector":           sector,
            })
        except Exception:
            continue

    log.info("[SCREENER] {} candidates fetched", len(results))
    return results


# ── Step 4: FMP insider buys — normalized, key-role filtered ──────────────────

def _fmp_insider_buys_sync(limit: int = 500) -> List[Dict]:
    """Akerlof (2001 Nobel) — Insider purchases as credible costly signals.

    Fetches recent open-market purchases across ALL symbols from FMP.
    Filters to key executive roles (CEO, CFO, Director) only — these insiders
    have the most material non-public information and the most reputational risk
    from making a bad bet.

    Normalisation: raw_value / market_cap — a $1M buy in a $200M company
    (0.5%) outranks a $1M buy in a $20B company (0.005%) by 100×.

    Args:
        limit: max recent transactions to pull from FMP.

    Returns:
        [{sym, total_value_usd, key_value_usd, normalized_pct_mcap,
          market_cap, roles, transaction_count, most_recent_date}]
        Sorted by normalized_pct_mcap descending.
    """
    key = _fmp_key()
    if not key:
        log.warning("[INSIDER] FMP_API_KEY not set — insider discovery skipped")
        return []

    # Bulk recent P-PURCHASE across all symbols
    data = _get_json(
        f"{_FMP_BASE}/v4/insider-trading",
        params={
            "transactionType": "P-PURCHASE",
            "limit":           limit,
            "apikey":          key,
        },
    )
    if not isinstance(data, list):
        log.warning("[INSIDER] Unexpected response: {}", type(data))
        return []

    cutoff = datetime.now() - timedelta(days=90)

    # Group by symbol — accumulate values per symbol
    sym_data: Dict[str, Dict] = {}
    for tx in data:
        try:
            sym   = str(tx.get("symbol", "")).upper().strip()
            if not sym or "." in sym:
                continue

            owner = str(tx.get("typeOfOwner", "") or "").upper()
            is_key = any(role in owner for role in _KEY_ROLES)
            if not is_key:
                continue  # only count key insiders

            shares = abs(float(tx.get("securitiesTransacted", 0) or 0))
            price  = abs(float(tx.get("price", 0) or 0))
            value  = shares * price
            if value < _MIN_KEY_INSIDER_USD:
                continue

            # Respect cutoff
            date_str = str(tx.get("transactionDate", tx.get("date", "")) or "")
            try:
                tx_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                if tx_date < cutoff:
                    continue
            except Exception:
                pass

            if sym not in sym_data:
                sym_data[sym] = {
                    "sym":          sym,
                    "key_value_usd": 0.0,
                    "roles":        set(),
                    "tx_count":     0,
                    "most_recent":  date_str,
                }

            entry = sym_data[sym]
            entry["key_value_usd"] += value
            entry["tx_count"]      += 1
            entry["roles"].add(owner)
            if date_str > entry["most_recent"]:
                entry["most_recent"] = date_str

        except Exception:
            continue

    if not sym_data:
        log.warning("[INSIDER] No key-insider purchases found in last 90 days")
        return []

    # Fetch market caps for normalisation (batched profile endpoint)
    symbols_list = list(sym_data.keys())
    market_caps  = _fetch_market_caps(symbols_list)

    results = []
    for sym, entry in sym_data.items():
        mcap       = market_caps.get(sym, 0.0)
        key_val    = entry["key_value_usd"]
        norm_pct   = (key_val / mcap * 100) if mcap > 1e6 else 0.0

        if norm_pct < _MIN_NORMALIZED_SIGNAL * 100:
            continue  # signal too small relative to market cap

        results.append({
            "sym":                 sym,
            "key_value_usd":       round(key_val, 0),
            "normalized_pct_mcap": round(norm_pct, 6),
            "market_cap":          round(mcap, 0),
            "roles":               sorted(entry["roles"]),
            "tx_count":            entry["tx_count"],
            "most_recent_date":    entry["most_recent"],
        })

    # Sort by normalized_pct_mcap — relative conviction, not raw dollar size
    results.sort(key=lambda x: x["normalized_pct_mcap"], reverse=True)
    log.info("[INSIDER] {} symbols with key-insider buys (normalised)", len(results))
    return results


def _fetch_market_caps(symbols: List[str]) -> Dict[str, float]:
    """Granger (2003 Nobel) — fetch market caps in batches for normalisation.

    FMP profile endpoint supports comma-separated symbols.
    Falls back to 0 (not None) so callers can divide safely.
    """
    key = _fmp_key()
    if not key or not symbols:
        return {}

    caps: Dict[str, float] = {}
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        syms_str = ",".join(batch)
        data = _get_json(
            f"{_FMP_BASE}/v3/profile/{syms_str}",
            params={"apikey": key},
        )
        if not isinstance(data, list):
            continue
        for item in data:
            try:
                sym  = str(item.get("symbol", "")).upper()
                mcap = float(item.get("mktCap", 0) or 0)
                caps[sym] = mcap
            except Exception:
                continue

    return caps


# ── Step 2: Institutional accumulation discovery ───────────────────────────────

def _fmp_institutional_accumulation_sync(
    candidate_symbols: List[str],
    limit: int = 50,
) -> List[Dict]:
    """Tirole (2014 Nobel) — institutional intermediaries aggregate information.

    For each candidate symbol, fetches FMP's institutional-holder data and
    computes a net accumulation metric: how much are the top holders increasing
    vs decreasing their positions?

    A net positive change across multiple large funds is a second-order
    confirmation signal that complements insider purchases.

    Args:
        candidate_symbols: pool of tickers to check (capped at `limit`).
        limit: max symbols to hit (FMP free-tier has daily request budgets).

    Returns:
        [{sym, net_shares_change, pct_change_avg, major_fund_count,
          holder_count, accumulation_score}]
        Sorted by accumulation_score descending.
    """
    key = _fmp_key()
    if not key:
        log.warning("[INST] FMP_API_KEY not set — institutional discovery skipped")
        return []

    # Cap to preserve FMP free-tier quota
    symbols = list(dict.fromkeys(s.upper() for s in candidate_symbols))[:limit]

    def _fetch_one(sym: str) -> Optional[Dict]:
        data = _get_json(
            f"{_FMP_BASE}/v3/institutional-holder/{sym}",
            params={"apikey": key},
        )
        if not isinstance(data, list) or not data:
            return None

        _MAJOR = {"BLACKROCK", "VANGUARD", "STATE STREET", "FIDELITY",
                  "JP MORGAN", "RENAISSANCE", "CITADEL", "MILLENNIUM"}

        net_change   = 0.0
        total_shares = 0.0
        pct_changes  = []
        major_count  = 0

        for holder in data[:20]:  # top 20 holders
            try:
                change = float(holder.get("change", 0) or 0)
                shares = float(holder.get("shares", 0) or 0)
                name   = str(holder.get("holderName", "") or "").upper()
                net_change   += change
                total_shares += shares

                if shares > 0 and change != 0:
                    pct_change = change / shares
                    pct_changes.append(pct_change)

                if any(mf in name for mf in _MAJOR) and change > 0:
                    major_count += 1
            except Exception:
                continue

        if total_shares == 0:
            return None

        avg_pct = sum(pct_changes) / len(pct_changes) if pct_changes else 0.0

        # Accumulation score: sigmoid-ish mapping of net_change / total_shares
        net_ratio     = net_change / total_shares
        raw_acc       = 0.50 + min(0.40, max(-0.40, net_ratio * 8))
        major_boost   = min(0.15, major_count * 0.04)
        acc_score     = min(1.0, max(0.0, raw_acc + major_boost))

        return {
            "sym":              sym,
            "net_shares_change": round(net_change, 0),
            "pct_change_avg":    round(avg_pct, 6),
            "major_fund_count":  major_count,
            "holder_count":      len(data),
            "accumulation_score": round(acc_score, 4),
        }

    results = []
    for sym in symbols:
        try:
            result = _fetch_one(sym)
            if result is not None:
                results.append(result)
        except Exception as exc:
            log.warning("[INST] Failed for {}: {}", sym, exc)
            continue

    results.sort(key=lambda x: x["accumulation_score"], reverse=True)
    log.info("[INST] Institutional data fetched for {}/{} symbols", len(results), len(symbols))
    return results


# ── Step 1: Fixed candidate selection — no intersection trap ──────────────────

def _select_candidates(
    screener_results:  List[Dict],
    insider_buys:      List[Dict],
    n:                 int = 50,
) -> Tuple[List[str], Dict[str, str]]:
    """Akerlof (2001 Nobel) — insider stocks force entry; screener fills remaining slots.

    Previously: insider stocks were silently dropped unless they happened to
    appear in the screener results (the "intersection trap").

    Now: the top insider-buy stocks (by normalized % of market cap) are
    guaranteed seats in the candidate pool. The screener fills the remaining
    n - len(guaranteed) slots with stocks not already selected.

    Args:
        screener_results: from _fmp_screener_sync()
        insider_buys:     from _fmp_insider_buys_sync(), pre-sorted by norm signal
        n:                total candidate pool size

    Returns:
        (selected_symbols, source_map)
        source_map: {sym -> "insider" | "screener" | "both"}
    """
    # Insider buys are already sorted by normalized_pct_mcap (strongest signal first)
    guaranteed  = [entry["sym"] for entry in insider_buys]
    screener_syms = [entry["sym"] for entry in screener_results]

    # Guaranteed slots: all insider buys (no cap — they earned their place)
    selected:    List[str]       = []
    source_map:  Dict[str, str]  = {}

    for sym in guaranteed:
        selected.append(sym)
        source_map[sym] = "insider"

    # Mark overlaps
    screener_set = set(screener_syms)
    for sym in list(selected):
        if sym in screener_set:
            source_map[sym] = "both"

    # Fill remaining slots from screener (stocks not already selected)
    remaining_slots = max(0, n - len(selected))
    for sym in screener_syms:
        if remaining_slots == 0:
            break
        if sym not in source_map:
            selected.append(sym)
            source_map[sym] = "screener"
            remaining_slots -= 1

    log.info(
        "[CANDIDATES] {} total | {} insider | {} screener-only | {} both",
        len(selected),
        sum(1 for v in source_map.values() if v == "insider"),
        sum(1 for v in source_map.values() if v == "screener"),
        sum(1 for v in source_map.values() if v == "both"),
    )
    return selected, source_map


# ── Step 3: Smart Money Pre-Score ─────────────────────────────────────────────

def _smart_money_prescore(
    sym:          str,
    insider_map:  Dict[str, Dict],
    inst_map:     Dict[str, Dict],
    screener_map: Dict[str, Dict],
) -> Tuple[float, float, float, float]:
    """Granger (2003 Nobel) — causal precedence of smart-money signals over price momentum.

    Computes a weighted composite of three independent signals:
      - Insider conviction (45%): normalised vs market cap, key-role filtered
      - Institutional accumulation (35%): net holdings change across top funds
      - Momentum (20%): volume spike + price change

    Stocks with massive insider or institutional signals rank above momentum
    plays even when their short-term price action is flat — this is the key
    fix over the original momentum-only sort.

    Args:
        sym:          ticker symbol
        insider_map:  {sym -> insider data dict}
        inst_map:     {sym -> institutional data dict}
        screener_map: {sym -> screener data dict}

    Returns:
        (composite_score, insider_component, inst_component, momentum_component)
        All values in [0, 1].
    """
    # ── Insider component ──────────────────────────────────────────────────────
    ins_data = insider_map.get(sym)
    if ins_data:
        norm_pct = ins_data.get("normalized_pct_mcap", 0.0)
        # 0.5% of market cap → full score (diminishing returns above that)
        ins_raw   = min(1.0, norm_pct / 0.5)
        ins_score = 0.30 + 0.70 * ins_raw   # floor at 0.30 for any insider activity
    else:
        ins_score = 0.0  # no insider signal → zero contribution (not neutral 0.5)

    # ── Institutional component ────────────────────────────────────────────────
    inst_data = inst_map.get(sym)
    if inst_data:
        acc = inst_data.get("accumulation_score", 0.5)
        # Only reward positive accumulation (> 0.5)
        inst_score = max(0.0, (acc - 0.50) * 2.0)  # [0.50, 1.0] → [0.0, 1.0]
    else:
        inst_score = 0.0

    # ── Momentum component ─────────────────────────────────────────────────────
    scr_data = screener_map.get(sym)
    if scr_data:
        vs    = scr_data.get("volume_spike", 1.0)
        chg   = scr_data.get("price_change_pct", 0.0)
        # Volume spike: 2× avg → score 0.5; 5× → score 1.0
        vs_sc = min(1.0, max(0.0, (vs - 1.0) / 4.0))
        # Price change: +5% → score 1.0; -5% → 0.0; 0% → 0.5
        chg_sc = min(1.0, max(0.0, 0.5 + chg / 10.0))
        mom_score = 0.5 * vs_sc + 0.5 * chg_sc
    else:
        mom_score = 0.0

    composite = (
        _W_INSIDER   * ins_score
        + _W_INST    * inst_score
        + _W_MOMENTUM * mom_score
    )

    return (
        round(composite,  4),
        round(ins_score,  4),
        round(inst_score, 4),
        round(mom_score,  4),
    )


# ── Async wrappers ─────────────────────────────────────────────────────────────

async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def _async_screener(limit: int = 200) -> List[Dict]:
    return await _run_in_executor(_fmp_screener_sync, limit)


async def _async_insider_buys(limit: int = 500) -> List[Dict]:
    return await _run_in_executor(_fmp_insider_buys_sync, limit)


async def _async_institutional(candidate_symbols: List[str], limit: int = 50) -> List[Dict]:
    return await _run_in_executor(_fmp_institutional_accumulation_sync, candidate_symbols, limit)


# ── Main entry point ───────────────────────────────────────────────────────────

async def run_scan_async(n: int = 20) -> List[ScanResult]:
    """Akerlof (2001 Nobel) + Tirole (2014 Nobel) — parallel smart money discovery.

    Runs screener, insider buys and institutional accumulation in parallel,
    then merges and ranks by smart-money pre-score so that a CEO buying
    $5M of their own stock at $10/share is never dropped in favour of a
    random stock that popped 3% on volume.

    Args:
        n: number of top results to return.

    Returns:
        List[ScanResult] sorted by smart_money_score descending.
    """
    log.info("[SCAN] Starting smart-money discovery scan (n={})", n)
    t0 = time.monotonic()

    # Phase 1: parallel data fetch (screener + insider bulk — no per-symbol calls yet)
    screener_raw, insider_raw = await asyncio.gather(
        _async_screener(200),
        _async_insider_buys(500),
        return_exceptions=True,
    )

    if isinstance(screener_raw, Exception):
        log.warning("[SCAN] Screener failed: {}", screener_raw)
        screener_raw = []
    if isinstance(insider_raw, Exception):
        log.warning("[SCAN] Insider fetch failed: {}", insider_raw)
        insider_raw = []

    # Phase 2: select candidates (insider stocks guaranteed entry)
    candidates, source_map = _select_candidates(
        screener_results=screener_raw,
        insider_buys=insider_raw,
        n=100,  # wider pool before final cut
    )

    # Phase 3: fetch institutional data for all candidates in parallel (capped at 50)
    inst_raw = await _async_institutional(candidates, limit=50)
    if isinstance(inst_raw, Exception):
        log.warning("[SCAN] Institutional fetch failed: {}", inst_raw)
        inst_raw = []

    # Phase 4: build lookup maps
    screener_map: Dict[str, Dict] = {e["sym"]: e for e in screener_raw}
    insider_map:  Dict[str, Dict] = {e["sym"]: e for e in insider_raw}
    inst_map:     Dict[str, Dict] = {e["sym"]: e for e in inst_raw}

    # Phase 5: score and rank
    results: List[ScanResult] = []
    for sym in candidates:
        composite, ins_sc, inst_sc, mom_sc = _smart_money_prescore(
            sym, insider_map, inst_map, screener_map
        )

        # Skip stocks with zero signal across all three components
        if composite == 0.0:
            continue

        ins_data  = insider_map.get(sym, {})
        inst_data = inst_map.get(sym, {})
        scr_data  = screener_map.get(sym, {})

        source_flags = [source_map.get(sym, "screener")]
        if sym in insider_map:
            source_flags.append("insider_confirmed")
        if sym in inst_map and inst_data.get("accumulation_score", 0.5) > 0.55:
            source_flags.append("institutional_accumulating")

        results.append(ScanResult(
            symbol                  = sym,
            smart_money_score       = composite,
            insider_score           = ins_sc,
            institutional_score     = inst_sc,
            momentum_score          = mom_sc,
            insider_value_usd       = ins_data.get("key_value_usd", 0.0),
            insider_value_pct_mcap  = ins_data.get("normalized_pct_mcap", 0.0),
            key_insider_roles       = ins_data.get("roles", []),
            institutional_net_shares= inst_data.get("net_shares_change", 0.0),
            institutional_pct_change= inst_data.get("pct_change_avg", 0.0),
            volume_spike            = scr_data.get("volume_spike", 0.0),
            price_change_pct        = scr_data.get("price_change_pct", 0.0),
            market_cap              = (
                ins_data.get("market_cap")
                or scr_data.get("market_cap", 0.0)
            ),
            source_flags            = list(set(source_flags)),
        ))

    # Sort by composite smart-money score — momentum can no longer override
    results.sort(key=lambda r: r["smart_money_score"], reverse=True)
    top = results[:n]

    elapsed = time.monotonic() - t0
    log.info(
        "[SCAN] Done in {:.1f}s | {} total scored | {} returned | "
        "insider-led={} screener-only={}",
        elapsed, len(results), len(top),
        sum(1 for r in top if "insider_confirmed" in r["source_flags"]),
        sum(1 for r in top if r["source_flags"] == ["screener"]),
    )
    return top


def run_scan(n: int = 20) -> List[ScanResult]:
    """Blocking wrapper for Streamlit / sync callers.

    Granger (2003 Nobel) — synchronous entry point for causal signal discovery.
    """
    try:
        return asyncio.run(run_scan_async(n))
    except RuntimeError:
        # Already inside a running event loop (Streamlit)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(run_scan_async(n))
        finally:
            loop.close()


# ── Diagnostic helper ──────────────────────────────────────────────────────────

def explain_result(result: ScanResult) -> str:
    """Return a one-paragraph human-readable explanation for a scan result."""
    lines = [
        f"{result['symbol']}  smart_money={result['smart_money_score']:.3f}",
        f"  Insider  ({_W_INSIDER:.0%} weight): score={result['insider_score']:.3f} "
        f"| ${result['insider_value_usd']:,.0f} = {result['insider_value_pct_mcap']:.4f}% of mktcap"
        f" | roles: {', '.join(result['key_insider_roles']) or 'none'}",
        f"  Inst     ({_W_INST:.0%} weight): score={result['institutional_score']:.3f} "
        f"| net_shares={result['institutional_net_shares']:+,.0f} "
        f"avg_chg={result['institutional_pct_change']:+.3%}",
        f"  Momentum ({_W_MOMENTUM:.0%} weight): score={result['momentum_score']:.3f} "
        f"| vol_spike={result['volume_spike']:.2f}x "
        f"price_chg={result['price_change_pct']:+.2f}%",
        f"  Sources: {', '.join(result['source_flags'])}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import json as _json
    results = run_scan(n=10)
    for r in results:
        print(explain_result(r))
        print()
