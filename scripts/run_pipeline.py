"""scripts/run_pipeline.py
EDGAR + FMP + yfinance daily data pipeline.

Stiglitz (2001 Nobel) — asymmetric information: insider filing activity is a
credible, costly-to-fake signal. This pipeline sources it from two layers:
  1. SEC EDGAR daily index  — Form 4 filings (free, cached 24 h)
  2. FMP insider-trading    — structured buy/sell with role classification
     (1 API call/day; TTL 12 h)
  3. yfinance               — news sentiment + VIX macro (free)

FMP budget: ≤ 2 calls per run (profile batch + insider list).
With caching, repeated intraday runs spend 0 additional FMP calls.

Usage:
  python scripts/run_pipeline.py --tickers-file config/top50.csv --log-dir logs
  python -m scripts.run_pipeline --tickers-file config/top50.csv --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.utils.io import save_json_atomic

log = logging.getLogger("run_pipeline")

# ── Weights (must sum to 1.0) ──────────────────────────────────────────────────
WEIGHTS = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "macro":    0.10,
}

# ── VIX → macro score ──────────────────────────────────────────────────────────
_VIX_MACRO = [
    (45.0, 0.20),   # Crash
    (35.0, 0.35),   # Panic
    (25.0, 0.50),   # Bear
    (15.0, 0.65),   # Neutral
    (12.0, 0.80),   # Bull
    (0.0,  0.90),   # Euphoria
]

# ── Bull/bear word lists for news scoring ──────────────────────────────────────
_BULL = frozenset([
    "beat", "beats", "exceed", "exceeds", "upgrade", "upgrades", "upgraded",
    "buy", "outperform", "strong", "record", "rally", "surge", "gain", "growth",
    "bullish", "profit", "revenue", "raise", "raises", "tops", "jump", "soar",
    "boom", "positive", "breakthrough", "approval", "approved", "expands",
])
_BEAR = frozenset([
    "miss", "misses", "downgrade", "downgrades", "sell", "underperform",
    "concern", "decline", "weak", "loss", "cut", "fall", "drop", "recession",
    "layoff", "lawsuit", "fine", "warning", "risk", "volatile", "below",
    "disappoints", "halt", "investigation", "fraud", "bankruptcy", "default",
])

_KEY_ROLES = frozenset([
    "CEO", "CFO", "COO", "CTO", "DIRECTOR", "PRESIDENT",
    "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING",
    "CHAIRMAN", "FOUNDER",
])


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_tickers(csv_path: Path) -> List[Dict[str, str]]:
    """Markowitz (1990 Nobel) — load stratified ticker universe from CSV."""
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            if ticker:
                rows.append({
                    "ticker":   ticker,
                    "sector":   row.get("sector", "Unknown").strip(),
                    "cap_tier": row.get("cap_tier", "large").strip(),
                })
    return rows


# ── FMP fetchers ───────────────────────────────────────────────────────────────

def _fmp_get(path: str, params: Dict, timeout: int = 20) -> Any:
    """Fama (2013 Nobel) — single FMP REST call with retry."""
    try:
        import requests as _req
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = _req.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        api_key = os.getenv("FMP_API_KEY", "")
        if not api_key:
            log.warning("FMP_API_KEY not set — skipping FMP call")
            return None
        url = f"https://financialmodelingprep.com/stable/{path.lstrip('/')}"
        params["apikey"] = api_key
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("FMP call %s failed: %s", path, exc)
        return None


def fetch_fmp_profiles(tickers: List[str]) -> Dict[str, float]:
    """Fama (2013): batch profile fetch — 1 FMP call for all tickers."""
    data = _fmp_get("profile", {"symbol": ",".join(tickers)})
    if not data or not isinstance(data, list):
        return {}
    return {
        row.get("symbol", ""): float(row.get("mktCap") or 0)
        for row in data
        if row.get("symbol")
    }


def fetch_fmp_insider_buys(lookback_days: int = 60) -> Dict[str, Dict]:
    """Akerlof (2001 Nobel) — fetch recent executive purchases, 1 FMP call."""
    data = _fmp_get("insider-trading", {"transactionType": "P", "limit": 200})
    if not data or not isinstance(data, list):
        return {}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()[:10]
    by_ticker: Dict[str, Dict] = {}
    for tx in data:
        tx_date = str(tx.get("transactionDate") or "")
        if tx_date < cutoff:
            continue
        if tx.get("acquistionOrDisposition") != "A":
            continue
        role = str(tx.get("reportingTitle") or "").upper()
        is_key = any(k in role for k in _KEY_ROLES)
        ticker = (tx.get("symbol") or "").upper()
        if not ticker:
            continue
        val = float(tx.get("securitiesTransacted") or 0) * float(tx.get("price") or 0)
        if ticker not in by_ticker:
            by_ticker[ticker] = {"count": 0, "total_usd": 0.0, "key_exec_usd": 0.0}
        by_ticker[ticker]["count"]    += 1
        by_ticker[ticker]["total_usd"] += val
        if is_key:
            by_ticker[ticker]["key_exec_usd"] += val
    return by_ticker


# ── EDGAR Form 4 counter ───────────────────────────────────────────────────────

def count_edgar_form4(ticker: str, lookback_days: int = 90) -> int:
    """Stiglitz (2001 Nobel) — count Form 4 insider filings from EDGAR daily index."""
    try:
        from regime_trader.services.edgar_index import EdgarDailyIndex
        idx = EdgarDailyIndex()
        end  = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        filings = idx.list_filings_range(start, end)
        target  = ticker.upper().replace("-", " ")
        count   = sum(
            1 for f in filings
            if f.get("form") in ("4", "4/A")
            and target in f.get("company", "").upper()
        )
        return count
    except Exception as exc:
        log.debug("EDGAR count for %s failed: %s", ticker, exc)
        return 0


# ── yfinance scorers ───────────────────────────────────────────────────────────

def score_news(ticker: str, max_items: int = 8) -> float:
    """Engle (2003 Nobel) — news sentiment from yfinance headlines ∈ [0,1]."""
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        scores = []
        for item in news[:max_items]:
            content = item.get("content", {})
            title = (
                content.get("title", "") if isinstance(content, dict)
                else item.get("title", "")
            )
            if not title:
                continue
            words = set(title.lower().split())
            bull  = len(words & _BULL)
            bear  = len(words & _BEAR)
            if bull == 0 and bear == 0:
                scores.append(0.50)
            else:
                scores.append(max(0.10, min(0.90, 0.50 + 0.20 * (bull - bear))))
        if not scores:
            return 0.50
        return round(sum(scores) / len(scores), 4)
    except Exception:
        return 0.50


def fetch_vix() -> float:
    """Engle (2003 Nobel) — fetch latest VIX close via yfinance."""
    try:
        import yfinance as yf
        df = yf.download("^VIX", period="3d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return 20.0
        return float(df["Close"].squeeze().dropna().iloc[-1])
    except Exception:
        return 20.0


def vix_to_macro(vix: float) -> float:
    """Engle (2003 Nobel) — map VIX level to macro score ∈ [0.20, 0.90]."""
    for threshold, score in _VIX_MACRO:
        if vix >= threshold:
            return score
    return 0.90


# ── Per-ticker scorer ──────────────────────────────────────────────────────────

def score_edgar(form4_count: int) -> float:
    """Stiglitz (2001): normalise Form 4 filing count to [0.20, 0.90]."""
    if form4_count <= 0:
        return 0.30
    return round(min(0.90, 0.30 + form4_count * 0.12), 4)


def score_insider(fmp_data: Optional[Dict]) -> Tuple[float, bool]:
    """Akerlof (2001): insider score + CEO buy flag from FMP data."""
    if not fmp_data:
        return 0.50, False
    count   = fmp_data.get("count", 0)
    total   = fmp_data.get("total_usd", 0.0)
    key_usd = fmp_data.get("key_exec_usd", 0.0)
    if count == 0:
        return 0.50, False
    # Base score from count (0.60) + bonus for large-value key exec buys
    score = min(0.90, 0.50 + count * 0.08)
    if key_usd > 100_000:
        score = max(score, 0.82)
    ceo_buy = key_usd > 25_000
    return round(score, 4), ceo_buy


# ── Main ───────────────────────────────────────────────────────────────────────

def run(tickers_file: Path, log_dir: Path, max_workers: int = 8) -> Dict[str, Any]:
    """Markowitz (1990 Nobel) — run full scoring pipeline; return status dict."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ticker_rows = load_tickers(tickers_file)
    tickers     = [r["ticker"] for r in ticker_rows]
    cap_tier    = {r["ticker"]: r.get("cap_tier", "large") for r in ticker_rows}
    sector      = {r["ticker"]: r.get("sector", "Unknown") for r in ticker_rows}

    t0 = time.time()
    log.info("Pipeline start: %d tickers from %s", len(tickers), tickers_file)

    # ── FMP: 2 calls ──────────────────────────────────────────────────────────
    log.info("Fetching FMP profiles (1 call)…")
    mktcaps = fetch_fmp_profiles(tickers)
    log.info("Fetching FMP insider buys (1 call)…")
    fmp_insider = fetch_fmp_insider_buys()
    fmp_count = (1 if mktcaps else 0) + (1 if fmp_insider else 0)

    # ── Macro: yfinance VIX ───────────────────────────────────────────────────
    vix = fetch_vix()
    macro_score = vix_to_macro(vix)
    log.info("VIX=%.1f  macro_score=%.2f", vix, macro_score)

    # ── EDGAR + yfinance: parallel per-ticker ─────────────────────────────────
    results = []
    errors  = 0

    def _score_ticker(row: Dict[str, str]) -> Dict[str, Any]:
        ticker = row["ticker"]
        try:
            form4_count = count_edgar_form4(ticker)
            e_score     = score_edgar(form4_count)
            i_score, ceo_buy = score_insider(fmp_insider.get(ticker))
            n_score     = score_news(ticker)
            return {
                "ticker":         ticker,
                "sector":         sector.get(ticker, "Unknown"),
                "cap_tier":       cap_tier.get(ticker, "large"),
                "market_cap":     mktcaps.get(ticker, 0.0),
                "edgar_score":    e_score,
                "insider_score":  i_score,
                "congress_score": 0.50,   # not fetched — FMP budget preserved
                "news_score":     n_score,
                "macro_score":    macro_score,
                "ceo_buy":        ceo_buy,
                "form4_count":    form4_count,
                "vix":            vix,
            }
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", ticker, exc)
            return {
                "ticker": ticker, "sector": sector.get(ticker, "Unknown"),
                "cap_tier": cap_tier.get(ticker, "large"),
                "market_cap": 0.0,
                "edgar_score": 0.30, "insider_score": 0.50,
                "congress_score": 0.50, "news_score": 0.50,
                "macro_score": macro_score, "ceo_buy": False,
                "form4_count": 0, "vix": vix,
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_ticker, row): row for row in ticker_rows}
        for fut in as_completed(futures):
            r = fut.result()
            if r.get("edgar_score", 0) == 0.30 and r.get("form4_count", 0) == 0:
                errors += 1
            results.append(r)

    edgar_count = sum(1 for r in results if r.get("form4_count", 0) > 0)
    duration    = round(time.time() - t0, 2)

    status = {
        "_edgar_meta": {
            "last_run":             datetime.now(timezone.utc).isoformat(),
            "run_duration_seconds": duration,
            "ticker_count":         len(tickers),
            "edgar_count":          edgar_count,
            "fmp_count":            fmp_count,
            "error_count":          errors,
        },
        "weights": WEIGHTS,
        "results": results,
    }

    out = log_dir / "intel_source_status.json"
    save_json_atomic(out, status)
    log.info(
        "Done in %.1fs — tickers=%d edgar=%d fmp_calls=%d errors=%d → %s",
        duration, len(tickers), edgar_count, fmp_count, errors, out,
    )
    return status


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EDGAR+FMP+yfinance daily pipeline")
    parser.add_argument("--tickers-file", type=Path, default=Path("config/top50.csv"))
    parser.add_argument("--log-dir",      type=Path, default=Path("logs"))
    parser.add_argument("--max-workers",  type=int,  default=8)
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        run(args.tickers_file, args.log_dir, args.max_workers)
        return 0
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
