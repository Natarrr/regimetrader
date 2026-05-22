"""scripts/run_pipeline.py
EDGAR + FMP + yfinance daily data pipeline.

Stiglitz (2001 Nobel) — asymmetric information: insider filing activity is a
credible, costly-to-fake signal. This pipeline sources it from three layers:
  1. Quiver Quantitative     — pre-parsed Form 4 (primary, QUIVER_API_KEY, 6h TTL)
  2. Finnhub                 — open-market purchases (fallback, FINNHUB_API_KEY)
  3. SEC EDGAR direct        — Form 4 count + CEO buy flag (always, free)

FMP budget: ≤ 80 calls per run (per-ticker profile, first 80 tickers only).
Tickers 81+ fall back to yfinance for market cap to stay within 250/day limit.

Usage:
  python scripts/run_pipeline.py --tickers-file config/universe.csv --log-dir logs
  python -m scripts.run_pipeline --tickers-file config/universe.csv --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.utils.io import save_json_atomic  # noqa: E402
from regime_trader.services.quiver_client import QuiverClient as _QuiverClient  # noqa: E402
from backend.market_intel.validator import validate_raw  # noqa: E402

log = logging.getLogger("run_pipeline")

# ── Weights (must sum to 1.0) ──────────────────────────────────────────────────
WEIGHTS = {
    "edgar":    0.28,
    "insider":  0.23,
    "congress": 0.22,
    "news":     0.15,
    "momentum": 0.12,
}

# ── Congress feed cache path (module-level so tests can monkeypatch it) ────────
CONGRESS_CACHE_PATH = ROOT / ".cache" / "congress_cache.json"

# ── SEC ticker→CIK map (fetched once, disk-cached 24 h) ───────────────────────
_CIK_CACHE_PATH  = ROOT / ".cache" / "sec_cik_map.json"
_CIK_TTL_SECONDS = 24 * 3600
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_cik_map: Dict[str, str] = {}   # TICKER → zero-padded 10-digit CIK
_cik_map_loaded  = False

# ── SEC rate-limited HTTP ─────────────────────────────────────────────────────
# data.sec.gov allows up to 10 req/sec; we stay at ~8 with a shared lock so
# all worker threads collectively respect the limit (not per-thread).
_SEC_RATE_LOCK: threading.Lock = threading.Lock()
_SEC_RATE_LAST: float = 0.0
_SEC_MIN_DELAY: float = 0.12   # 1/8 s ≈ 8 req/s globally

_CONGRESS_TTL_HOURS = 24
_HOUSE_URL   = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL  = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_INVALID_TICKERS = frozenset({"N/A", "--", "", "NONE", "NO TICKER"})

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
    """Fama (2013 Nobel): fetch market caps — FMP single-ticker then yfinance fallback.

    FMP /stable/profile batch (comma-separated) is broken on the free tier (returns []).
    We fetch one ticker at a time, then use yfinance for any misses to avoid
    consuming the 250 calls/day FMP budget on repeated pipeline runs.

    Strategy: FMP for up to 80 tickers (half of 160), yfinance for the rest.
    This caps FMP usage at ~240 calls/day across 3 daily runs while keeping
    market cap data complete.
    """
    result: Dict[str, float] = {}
    api_key = os.getenv("FMP_API_KEY", "")

    fmp_cap = 80   # max tickers to fetch via FMP per run (conserves daily budget)

    if api_key:
        try:
            import requests as _req
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            s = _req.Session()
            retry_cfg = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503])
            s.mount("https://", HTTPAdapter(max_retries=retry_cfg))

            for i, ticker in enumerate(tickers[:fmp_cap]):
                try:
                    url = (
                        f"https://financialmodelingprep.com/stable/profile"
                        f"?symbol={ticker}&apikey={api_key}"
                    )
                    r = s.get(url, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    if data and isinstance(data, list):
                        row = data[0]
                        cap = float(row.get("marketCap") or row.get("mktCap") or 0)
                        result[ticker] = cap
                except Exception as exc:
                    log.debug("FMP profile failed for %s: %s", ticker, exc)
                if i < fmp_cap - 1:
                    time.sleep(0.15)   # ~6.5 req/s — safe for free tier

            fmp_hits = sum(1 for v in result.values() if v > 0)
            log.info("FMP profiles: %d/%d tickers with market cap", fmp_hits, min(len(tickers), fmp_cap))
        except Exception as exc:
            log.warning("FMP profile fetch failed: %s", exc)
    else:
        log.warning("FMP_API_KEY not set — using yfinance for all market caps")

    # yfinance fallback for tickers not covered by FMP (or when FMP key absent)
    missing = [t for t in tickers if t not in result or result[t] == 0]
    if missing:
        try:
            import yfinance as yf
            log.info("yfinance market cap fallback for %d tickers…", len(missing))
            for ticker in missing:
                try:
                    info = yf.Ticker(ticker).info
                    cap = float(info.get("marketCap") or 0)
                    result[ticker] = cap
                except Exception:
                    result[ticker] = 0.0
        except Exception as exc:
            log.warning("yfinance market cap fallback failed: %s", exc)

    nonzero = sum(1 for v in result.values() if v > 0)
    log.info("Market cap complete: %d/%d tickers", nonzero, len(tickers))
    return result




def _parse_congress_transactions(
    transactions: List[Dict],
    cutoff: str,
    by_ticker: Dict[str, Dict],
    date_key: str = "transaction_date",
    type_key: str = "type",
    ticker_key: str = "ticker",
) -> None:
    """Shared parser for Stock Watcher (S3) and Quiver Quantitative transaction lists."""
    for tx in transactions:
        date_str = str(tx.get(date_key) or tx.get("date") or "")
        if date_str[:10] < cutoff:
            continue
        ticker = str(tx.get(ticker_key) or "").upper().strip()
        if not ticker or ticker in _INVALID_TICKERS:
            continue
        tx_type = str(tx.get(type_key) or "").lower()
        is_purchase = "purchase" in tx_type
        is_sale = "sale" in tx_type
        if not (is_purchase or is_sale):
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = {"purchases": 0, "sales": 0, "total": 0}
        if is_purchase:
            by_ticker[ticker]["purchases"] += 1
        else:
            by_ticker[ticker]["sales"] += 1
        by_ticker[ticker]["total"] += 1


def _fetch_quiver_congress(cutoff: str) -> Optional[Dict[str, Dict]]:
    """Delegate to QuiverClient.congress_by_ticker() — avoids duplicating HTTP/cache logic.

    Returns populated by_ticker dict (with recency_days) on success,
    None if key is absent or call fails.
    """
    try:
        client = _QuiverClient()
        if not client._api_key:
            return None
        result = client.congress_by_ticker(lookback_days=180)
        if result:
            n_tickers = len(result)
            n_tx = sum(v["total"] for v in result.values())
            log.info(
                "Quiver congress: %d transactions across %d tickers (via QuiverClient)",
                n_tx, n_tickers,
            )
        return result or None
    except Exception as exc:
        log.warning("Quiver congress delegation failed: %s", exc)
        return None


def fetch_congress_buys(lookback_days: int = 90) -> Dict[str, Dict]:
    """Stiglitz (2001 Nobel) — fetch congressional trading data.

    Primary:  House/Senate Stock Watcher public S3 feeds (no API key).
    Fallback: Quiver Quantitative /beta/live/congresstrading (QUIVER_API_KEY).

    Filters to the lookback window and counts purchase vs sale transactions
    per ticker.  Results are cached for 24 h.

    $\\text{net\\_score} = \\frac{(purchases - sales)}{total + 1}$

    Returns:
        Dict keyed by ticker → {"purchases": int, "sales": int, "total": int}
        Returns {} when all sources fail (score_congress will return 0.0).
    """
    import requests as _req

    # ── Check 24-hour cache ───────────────────────────────────────────────────
    # Truthy check on by_ticker: an empty {} from a failed S3 run must not be
    # served as a valid cache hit — it would silence the Quiver fallback for
    # up to 24h, leaving congress_score=0.0 for all tickers.
    if CONGRESS_CACHE_PATH.exists():
        try:
            cached = json.loads(CONGRESS_CACHE_PATH.read_text(encoding="utf-8"))
            age_h = (time.time() - float(cached.get("_ts", 0))) / 3600
            by_ticker_cached = cached.get("by_ticker", {})
            if age_h < _CONGRESS_TTL_HOURS and by_ticker_cached:
                return by_ticker_cached
        except Exception:
            pass

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    by_ticker: Dict[str, Dict] = {}
    s3_ok = False

    # ── Primary: Stock Watcher S3 feeds ──────────────────────────────────────
    for url, label in [(_HOUSE_URL, "house"), (_SENATE_URL, "senate")]:
        try:
            resp = _req.get(url, timeout=30)
            if resp.status_code == 403:
                log.warning(
                    "Congress feed %s returned 403 — S3 bucket restricted; "
                    "will use Quiver Quantitative fallback (QUIVER_API_KEY).",
                    label,
                )
                continue
            resp.raise_for_status()
            _parse_congress_transactions(resp.json(), cutoff, by_ticker)
            s3_ok = True
            log.info("Congress feed %s: OK — %d tickers", label, len(by_ticker))
        except Exception as exc:
            log.warning("Congress feed %s failed: %s", label, exc)

    # ── Fallback: Quiver Quantitative (when S3 yields nothing) ───────────────
    if not s3_ok or not by_ticker:
        log.info("S3 congress feeds unavailable — trying Quiver Quantitative fallback…")
        quiver_data = _fetch_quiver_congress(cutoff)
        if quiver_data:
            by_ticker = quiver_data
        elif not os.getenv("QUIVER_API_KEY"):
            log.warning(
                "No QUIVER_API_KEY set and S3 feeds are down — "
                "congress factor will be 0.0 (penalised) for all tickers."
            )

    # ── Persist cache ─────────────────────────────────────────────────────────
    try:
        save_json_atomic(CONGRESS_CACHE_PATH, {"_ts": time.time(), "by_ticker": by_ticker})
    except Exception as exc:
        log.debug("Congress cache write failed: %s", exc)

    return by_ticker


def score_congress(data: Optional[Dict]) -> float:
    """Stiglitz (2001 Nobel) — congressional net buy signal $\\in [0, 1]$.

    $score = \\frac{(purchases - sales) / (total + 1) + 1}{2}$

    A ticker with only purchases scores >0.5; only sales scores <0.5.
    Equal purchases/sales (truly neutral) scores 0.5.
    Missing data (None / empty dict) scores 0.0 so the cross-sectional
    normaliser sees a dead feed and penalises rather than grants neutral credit.

    Recency weighting: trades within 30 days get full credit; older trades
    decay linearly to 0.70× signal strength at 180 days (dampens towards 0.5,
    not towards 0, so the direction is preserved but urgency is lower).
    """
    if not data:
        # Dead API or ticker not traded by congress → 0.0, not 0.5.
        # When ALL tickers return 0.0, _cross_sectional_normalize treats it as
        # a fully-failed feed (all-zero branch) rather than the "all identical
        # non-zero → neutral 0.5" branch, which would silently waste 20% weight.
        return 0.0
    purchases = int(data.get("purchases", 0))
    sales     = int(data.get("sales", 0))
    total     = purchases + sales   # compute from actual values, not stored field
    if total == 0:
        return 0.50   # data present but no net activity → genuinely neutral
    raw = (purchases - sales) / (total + 1)   # $\in (-1, 1)$
    base_score = round((raw + 1) / 2, 4)       # $\to (0, 1)$

    # Recency multiplier: full credit ≤30 days, decay to 0.70× at 180 days.
    # Dampens *towards neutral (0.5)*, not towards zero — direction is preserved.
    recency_days = data.get("recency_days")
    if recency_days is not None and recency_days > 30:
        decay = max(0.70, 1.0 - 0.30 * min(recency_days - 30, 150) / 150)
        base_score = 0.5 + (base_score - 0.5) * decay

    return round(base_score, 4)


def _fetch_spy_return() -> float:
    """Fetch SPY 3-month return once before the thread pool starts.

    Called once in run() and passed as a closure to _score_ticker() so all
    160 worker threads share the same SPY baseline instead of each downloading
    it separately (which causes yfinance cache collisions under concurrency).
    Returns 0.0 on any failure so momentum still scores relative to flat market.
    """
    try:
        import yfinance as yf
        spy_df = yf.download("SPY", period="3mo", interval="1d",
                              progress=False, auto_adjust=True)
        if spy_df is None or spy_df.empty or len(spy_df) < 2:
            return 0.0
        spy_close = spy_df["Close"].squeeze().dropna()
        return float((spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0])
    except Exception as exc:
        log.warning("SPY baseline fetch failed: %s — momentum will use 0.0 baseline", exc)
        return 0.0


def fetch_price_data(ticker: str, spy_return: float = 0.0) -> Dict[str, float]:
    """Thaler (2017 Nobel) — 20-day SPY-relative return + volume spike.

    `spy_return` should be pre-fetched via _fetch_spy_return() before the thread
    pool starts so all workers share a consistent SPY baseline.
    Volume spike = 5-day avg volume / full-window avg volume.

    Returns {"return_20d": float, "spy_return_20d": float, "volume_spike": float}.
    Returns {"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0} on any error.
    """
    _default = {"return_20d": 0.0, "spy_return_20d": spy_return, "volume_spike": 1.0}
    try:
        import yfinance as yf

        df = yf.download(ticker, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 5:
            return _default

        close = df["Close"].squeeze().dropna()
        if len(close) < 2:
            return _default
        ret = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])

        if "Volume" in df.columns:
            vol = df["Volume"].squeeze().dropna()
            if len(vol) >= 10:
                recent_avg = float(vol.iloc[-5:].mean())
                full_avg   = float(vol.mean())
                volume_spike = round(recent_avg / full_avg, 4) if full_avg > 0 else 1.0
            else:
                volume_spike = 1.0
        else:
            volume_spike = 1.0

        return {
            "return_20d":     round(ret, 6),
            "spy_return_20d": round(spy_return, 6),
            "volume_spike":   volume_spike,
        }
    except Exception as exc:
        log.debug("fetch_price_data %s failed: %s", ticker, exc)
        return _default


def score_momentum(
    ticker_return_20d: float,
    spy_return_20d: float = 0.0,
    volume_spike: float = 1.0,
) -> float:
    """Thaler (2017 Nobel) — SPY-relative momentum + volume confirmation in [0, 1].

    relative_return = ticker_return_20d - spy_return_20d, clipped to +-30%.
    return_score maps (-0.30, +0.30) linearly to (0, 1).
    vol_score maps volume_spike (ratio of recent 5d avg to 90d avg) to (0, 1):
      1.0x (flat) -> 0.0, 5.0x spike -> 1.0.

    Combined: 0.65 x return_score + 0.35 x vol_score
    """
    r = max(-0.30, min(0.30, ticker_return_20d - spy_return_20d))
    return_score = round((r + 0.30) / 0.60, 4)
    vol_score    = round(min(1.0, max(0.0, (volume_spike - 1.0) / 4.0)), 4)
    return round(0.65 * return_score + 0.35 * vol_score, 4)


# ── SEC helpers ────────────────────────────────────────────────────────────────

def _sec_get(url: str, timeout: int = 20):
    """Rate-limited GET to any SEC endpoint (data.sec.gov or www.sec.gov/Archives).

    Uses a module-level shared lock so all worker threads together respect the
    SEC's 10 req/sec guidance.  Lock is released BEFORE the HTTP call so
    threads can execute requests in parallel within the rate budget.

    Raises requests.HTTPError on non-200.
    """
    global _SEC_RATE_LAST
    import requests as _req
    # `or` handles both absent key AND empty-string (e.g. when secret is unset
    # and workflow sets EDGAR_USER_AGENT='').  An empty User-Agent causes 403.
    ua = os.getenv("EDGAR_USER_AGENT") or "regime-trader-research n.tardy@hotmail.fr"
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    # Hold lock only for the rate-limit sleep + timestamp update, NOT for the
    # HTTP call — allows genuine parallelism across worker threads.
    with _SEC_RATE_LOCK:
        elapsed = time.time() - _SEC_RATE_LAST
        if elapsed < _SEC_MIN_DELAY:
            time.sleep(_SEC_MIN_DELAY - elapsed)
        _SEC_RATE_LAST = time.time()

    resp = _req.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


# ── EDGAR Form 4 counter ───────────────────────────────────────────────────────

def _load_cik_map() -> Dict[str, str]:
    """Fetch SEC ticker→CIK map (disk-cached 24 h). Raises on network failure."""
    global _cik_map, _cik_map_loaded
    if _cik_map_loaded:
        return _cik_map

    if _CIK_CACHE_PATH.exists():
        try:
            cached = json.loads(_CIK_CACHE_PATH.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("_ts", 0)) < _CIK_TTL_SECONDS:
                _cik_map = cached["tickers"]
                _cik_map_loaded = True
                log.info("CIK map loaded from cache: %d tickers", len(_cik_map))
                return _cik_map
        except Exception:
            pass

    resp = _sec_get(_SEC_TICKERS_URL, timeout=30)
    data = resp.json()
    ticker_map: Dict[str, str] = {
        str(e["ticker"]).upper(): str(e["cik_str"]).zfill(10)
        for e in data.values()
        if e.get("ticker") and e.get("cik_str")
    }
    _CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(_CIK_CACHE_PATH, {"_ts": time.time(), "tickers": ticker_map})
    _cik_map = ticker_map
    _cik_map_loaded = True
    log.info("CIK map fetched from SEC: %d tickers", len(_cik_map))
    return _cik_map



# ── yfinance scorers ───────────────────────────────────────────────────────────

def _score_news_yfinance(ticker: str) -> float:
    """yfinance headline word-count fallback. Returns 0.0 (not 0.5) on any failure."""
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        scores = []
        for item in news[:8]:
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
            return 0.0
        return round(sum(scores) / len(scores), 4)
    except Exception:
        return 0.0


def score_news_finnhub(ticker: str, api_key: str) -> float:
    """Engle (2003 Nobel) — Finnhub pre-computed sentiment score in [0, 1].

    Finnhub /news-sentiment returns:
      buzz.weeklyAverage       — normalized buzz volume (0-1)
      sentiment.bullishPercent — fraction of bullish articles (0-1)

    Score = 0.60 x bullishPercent + 0.40 x min(1.0, weeklyAverage / 0.5)

    Falls back to _score_news_yfinance() on any API failure.
    Returns 0.0 (not 0.5) if both sources fail — dead feed is penalised.
    """
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={api_key}"
    try:
        import requests as _req
        resp = _req.get(url, timeout=10)
        resp.raise_for_status()
        d        = resp.json()
        bullish  = float(d.get("sentiment", {}).get("bullishPercent", 0.5))
        buzz     = float(d.get("buzz", {}).get("weeklyAverage", 0.0))
        buzz_norm = min(1.0, buzz / 0.5)
        return round(0.60 * bullish + 0.40 * buzz_norm, 4)
    except Exception:
        try:
            return _score_news_yfinance(ticker)
        except Exception:
            return 0.0


def fetch_finnhub_insider_purchases(
    ticker: str,
    api_key: str,
    lookback_days: int = 180,
) -> Tuple[float, int]:
    """Stiglitz (2001 Nobel) — fetch open-market insider purchases via Finnhub.

    Calls /stock/insider-transactions and filters to:
      - transactionCode == "P"  (open-market purchase, not award/grant/exercise)
      - isDerivative == False   (actual shares, not options)
      - transactionDate >= cutoff (within lookback_days)

    Returns (total_purchases_usd, days_since_most_recent).
    Returns (0.0, 0) on any failure or when no qualifying purchases exist.
    """
    if not api_key:
        return 0.0, 0
    from datetime import date as _date
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    try:
        import requests as _req
        url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={api_key}"
        resp = _req.get(url, timeout=10)
        resp.raise_for_status()
        transactions = resp.json().get("data", []) or []

        purchase_dates: List[str] = []
        total_usd = 0.0
        for tx in transactions:
            if tx.get("transactionCode") != "P":
                continue
            if tx.get("isDerivative", False):
                continue
            tx_date = str(tx.get("transactionDate") or "")[:10]
            if tx_date < cutoff:
                continue
            shares = float(tx.get("share") or 0)
            price  = float(tx.get("transactionPrice") or 0)
            if shares > 0 and price > 0:
                total_usd += shares * price
                purchase_dates.append(tx_date)

        if not purchase_dates:
            return 0.0, 0

        most_recent = max(purchase_dates)
        days_ago = (datetime.now(timezone.utc).date() - _date.fromisoformat(most_recent)).days
        return round(total_usd, 2), max(0, days_ago)

    except Exception as exc:
        log.debug("Finnhub insider fetch failed for %s: %s", ticker, exc)
        return 0.0, 0


def fetch_all_finnhub_insider(
    tickers: List[str],
    api_key: str,
    lookback_days: int = 180,
    calls_per_minute: int = 55,
) -> Dict[str, Tuple[float, int]]:
    """Pre-fetch Finnhub insider-transactions for all tickers before the thread pool.

    Finnhub free tier: 60 calls/min.  This function serialises the calls with a
    small sleep between each to stay safely under the rate limit (default 55/min
    = ~1.09s per call).  A 429 response triggers an exponential backoff retry.

    Returns a dict mapping ticker → (total_purchases_usd, days_since_most_recent).
    Missing or failed tickers get (0.0, 0).
    """
    if not api_key or not tickers:
        return {}

    import requests as _req
    from datetime import date as _date

    min_interval = 60.0 / calls_per_minute   # seconds between calls
    results: Dict[str, Tuple[float, int]] = {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    log.info(
        "Finnhub insider pre-fetch: %d tickers at %.1f s/call (rate: %d/min)",
        len(tickers), min_interval, calls_per_minute,
    )

    for i, ticker in enumerate(tickers):
        backoff = min_interval
        for attempt in range(4):
            try:
                url = (
                    f"https://finnhub.io/api/v1/stock/insider-transactions"
                    f"?symbol={ticker}&token={api_key}"
                )
                resp = _req.get(url, timeout=10)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", backoff * (2 ** attempt)))
                    log.warning(
                        "Finnhub 429 for %s (attempt %d/4) — sleeping %.1fs", ticker, attempt + 1, wait
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                transactions = resp.json().get("data", []) or []

                purchase_dates: List[str] = []
                total_usd = 0.0
                for tx in transactions:
                    if tx.get("transactionCode") != "P":
                        continue
                    if tx.get("isDerivative", False):
                        continue
                    tx_date = str(tx.get("transactionDate") or "")[:10]
                    if tx_date < cutoff:
                        continue
                    shares = float(tx.get("share") or 0)
                    price  = float(tx.get("transactionPrice") or 0)
                    if shares > 0 and price > 0:
                        total_usd += shares * price
                        purchase_dates.append(tx_date)

                if purchase_dates:
                    most_recent = max(purchase_dates)
                    days_ago = (datetime.now(timezone.utc).date() - _date.fromisoformat(most_recent)).days
                    results[ticker] = (round(total_usd, 2), max(0, days_ago))
                else:
                    results[ticker] = (0.0, 0)
                break   # success — exit retry loop

            except Exception as exc:
                log.debug("Finnhub insider pre-fetch error for %s (attempt %d): %s", ticker, attempt + 1, exc)
                if attempt < 3:
                    time.sleep(backoff * (2 ** attempt))
                else:
                    results[ticker] = (0.0, 0)

        # Rate-limit pacing between tickers (after success or final failure)
        if i < len(tickers) - 1:
            time.sleep(min_interval)

    nonzero = sum(1 for v in results.values() if v[0] > 0)
    log.info(
        "Finnhub insider pre-fetch complete: %d/%d tickers with purchases",
        nonzero, len(tickers),
    )
    return results


def _parse_quiver_trades(
    trades: List[Dict],
    cutoff: str,
) -> Tuple[float, int]:
    """Parse a Quiver insider-trades response for one ticker.

    Returns (total_purchases_usd, days_since_most_recent).
    Pure function — no I/O, safe to call from threads.
    """
    from datetime import date as _date

    purchase_dates: List[str] = []
    total_usd = 0.0

    for tx in trades:
        tx_code = str(tx.get("TransactionCode") or "").strip().upper()
        ad_code  = str(tx.get("AcquiredDisposedCode") or "").strip().upper()
        if tx_code != "P" or ad_code != "A":
            continue

        is_officer  = bool(tx.get("isOfficer") or tx.get("IsOfficer"))
        is_director = bool(tx.get("isDirector") or tx.get("IsDirector"))
        title = str(tx.get("Title") or tx.get("OfficerTitle") or "").upper()
        if not (is_officer or is_director or any(role in title for role in _KEY_ROLES)):
            continue

        tx_date = str(tx.get("Date") or tx.get("TransactionDate") or "")[:10]
        if not tx_date or tx_date < cutoff:
            continue

        total_val = tx.get("TotalValue")
        if total_val is not None:
            usd = abs(float(total_val or 0))
        else:
            shares = float(tx.get("Shares") or 0)
            price  = float(tx.get("PricePerShare") or 0)
            usd    = shares * price

        if usd > 0:
            total_usd += usd
            purchase_dates.append(tx_date)

    if not purchase_dates:
        return 0.0, 0

    most_recent = max(purchase_dates)
    days_ago = (
        datetime.now(timezone.utc).date() - _date.fromisoformat(most_recent)
    ).days
    return round(total_usd, 2), max(0, days_ago)


def fetch_quiver_insider_all(
    tickers: List[str],
    lookback_days: int = 180,
    max_workers: int = 5,
) -> Dict[str, Tuple[float, int]]:
    """Fetch insider purchase data for all tickers via Quiver Quantitative.

    Stiglitz (2001 Nobel) — Quiver pre-parses SEC Form 4 filings into structured
    JSON, eliminating brittle XML parsing. Uses ThreadPoolExecutor(max_workers=5)
    to parallelize HTTP calls while respecting Quiver's rate limits.

    QuiverClient has a 6h file-based TTL: the first daily run makes real HTTP
    calls; the two subsequent runs within the same 6h window read from disk and
    are effectively free (no network, no concurrency overhead).

    max_workers=5 chosen deliberately:
      - 160 tickers ÷ 5 workers ≈ 32 batches
      - At ~200–400ms per Quiver call: total ~6–13s (vs. ~48s serial)
      - 5 concurrent connections stays well within typical REST API limits

    Returns {ticker: (total_purchases_usd, days_since_most_recent)}.
    Tickers with no qualifying purchases get (0.0, 0).
    Returns {} if QUIVER_API_KEY is not set.
    """
    client = _QuiverClient()
    if not client._api_key:
        log.info("QUIVER_API_KEY not set — skipping Quiver insider pre-fetch")
        return {}

    if not tickers:
        return {}

    # Probe with the first ticker before spinning up the thread pool.
    # A 403 means the endpoint is not included in the current plan — log once
    # and return {} immediately without making 159 more pointless HTTP calls.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    try:
        _ = client.get_insider_trades(tickers[0])   # sets _insider_plan_restricted on 403
    except Exception as exc:
        log.debug("Quiver insider probe failed for %s: %s", tickers[0], type(exc).__name__)
    if client._insider_plan_restricted:
        log.info(
            "Quiver insider not available on current plan — "
            "insider scores will use Finnhub/EDGAR fallback."
        )
        return {}

    def _fetch_one(ticker: str) -> Tuple[str, Tuple[float, int]]:
        t0 = time.monotonic()
        try:
            trades = client.get_insider_trades(ticker) or []
            result = _parse_quiver_trades(trades, cutoff)
        except Exception as exc:
            log.debug("Quiver insider fetch failed for %s: %s", ticker, type(exc).__name__)
            result = (0.0, 0)
        elapsed = time.monotonic() - t0
        log.debug("Quiver insider %s: $%.0f elapsed=%.2fs", ticker, result[0], elapsed)
        return ticker, result

    results: Dict[str, Tuple[float, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, value in pool.map(_fetch_one, tickers):
            results[ticker] = value

    nonzero = sum(1 for v in results.values() if v[0] > 0)
    log.info(
        "Quiver insider pre-fetch complete: %d/%d tickers with purchases",
        nonzero, len(tickers),
    )
    return results


# ── Per-ticker scorer ──────────────────────────────────────────────────────────

def score_edgar(form4_count: int) -> float:
    """Stiglitz (2001): normalise Form 4 filing count to [0, 0.90] using log-scale.

    0 filings -> 0.0 (penalised, not neutral).  Consistent with score_insider_value
    and score_congress: a dead/absent signal is 0.0 so the cross-sectional
    normaliser triggers the all-zero branch (treats it as a dead feed) rather than
    the all-same-non-zero branch (which silently returns 0.50 neutral for everyone).

    Log-scale prevents the old linear formula from saturating at 0.90 for any company
    with >= 5 filings (which is virtually every large-cap), making the factor useless
    cross-sectionally.  log1p(n)/log1p(200) maps [1, 200+] to roughly [0.10, 0.90]
    with good spread across the typical 5-160 filing range.
    """
    if form4_count <= 0:
        return 0.0
    return round(min(0.90, math.log1p(form4_count) / math.log1p(200)), 4)


def score_insider_value(
    key_purchases_usd: float,
    market_cap: float,
    days_since_most_recent: int = 0,
) -> float:
    """Stiglitz (2001 Nobel) — dollar conviction score for insider purchases in [0, 1].

    Maps total open-market purchase value as % of market cap to a score:
      0%      -> 0.0   (no purchases = dead signal, penalised not neutral)
      0.01%   -> ~0.30 (floor — small but credible)
      0.10%   -> ~0.65 (mid — meaningful conviction)
      1.00%+  -> ~0.90 (ceiling — exceptional conviction)

    Uses log-scale so small buys still count while large buys don't explode.
    Recency decay: purchases older than 30 days decay toward 0.50 neutral
    (direction preserved but urgency reduced — same decay as score_congress).
    """
    if key_purchases_usd <= 0 or market_cap <= 0:
        return 0.0

    pct = key_purchases_usd / market_cap
    raw = min(1.0, math.log1p(pct * 10000) / math.log1p(100))
    base_score = round(0.30 + 0.60 * raw, 4)

    if days_since_most_recent > 30:
        decay = max(0.70, 1.0 - 0.30 * min(days_since_most_recent - 30, 150) / 150)
        base_score = round(0.5 + (base_score - 0.5) * decay, 4)

    return base_score



# ── EDGAR submissions API (replaces CGI browse + yfinance insider) ─────────────

def _parse_form4_xml(cik: str, accession: str, primary_doc: str) -> List[Dict]:
    """Fetch and parse a Form 4 XML filing from EDGAR Archives.

    Returns list of dicts: {code, value, title}.
      code:  P=open-market purchase, S=sale, A=award, F=tax withholding, etc.
      value: shares × price (USD, approximate)
      title: officer/director title of the reporting owner

    Returns [] on any failure (network, non-XML .htm filing, parse error).
    HTTP call goes through the shared _sec_get() rate limiter.

    Note: the SEC submissions API returns `primaryDocument` as the XSLT-styled
    path (e.g. "xslF345X06/form4.xml"), which serves rendered HTML, not raw XML.
    We strip any leading subdirectory prefix so we always fetch the machine-readable
    XML at the accession root (e.g. "form4.xml").
    """
    from xml.etree import ElementTree as ET
    acc_nodash = accession.replace("-", "")
    cik_int = str(int(cik))   # strip leading zeros for the Archives path segment

    # Strip XSLT subdirectory prefix — the real XML lives at the accession root.
    # "xslF345X06/form4.xml" -> "form4.xml", "form4.xml" -> "form4.xml"
    doc_filename = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodash}/{doc_filename}"
    )
    try:
        resp = _sec_get(url)
    except Exception as exc:
        log.debug("Form 4 fetch failed %s: %s", url, exc)
        return []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []   # .htm or plain-text filing — not XML-parseable

    # Strip namespace prefixes so findall() works regardless of xmlns= declaration
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    # Reporting owner's title (first occurrence)
    officer_title = ""
    title_el = root.find(".//officerTitle")
    if title_el is not None and title_el.text:
        officer_title = title_el.text.strip()
    if not officer_title:
        is_dir_el = root.find(".//isDirector")
        if is_dir_el is not None and is_dir_el.text == "1":
            officer_title = "Director"

    transactions: List[Dict] = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code_el = tx.find(".//transactionCode")
        if code_el is None or not code_el.text:
            continue
        code = code_el.text.strip()

        shares_el = tx.find(".//transactionShares/value")
        price_el  = tx.find(".//transactionPricePerShare/value")
        try:
            shares = float(shares_el.text) if shares_el is not None else 0.0
        except (TypeError, ValueError):
            shares = 0.0
        try:
            price = float(price_el.text) if price_el is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0

        transactions.append({"code": code, "value": shares * price, "title": officer_title})

    return transactions


def fetch_edgar_data(ticker: str, lookback_days: int = 180) -> Tuple[int, float, bool, int]:
    """Fetch Form 4 count and insider score from the SEC submissions API.

    Stiglitz (2001 Nobel) / Akerlof (2001 Nobel) — replaces both the legacy
    CGI browse endpoint (EdgarService.list_filings) and the yfinance insider
    path, both of which fail silently in GitHub Actions CI environments.

    Uses data.sec.gov/submissions/CIK{cik}.json (official programmatic API)
    for EDGAR count, then optionally fetches up to 3 Form 4 XML documents for
    insider buy/sell classification.

    Returns:
        (form4_count, total_purchases_usd, ceo_buy_flag, days_since_most_recent)

    Raises on network failure so the caller can set _edgar_ok=False.
    """
    cik_map = _load_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        log.debug("No SEC CIK for %s", ticker)
        return 0, 0.0, False, 0

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = _sec_get(url)   # raises on failure → caller sets _edgar_ok=False
    data = resp.json()

    recent     = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    prim_docs  = recent.get("primaryDocument", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    form4_filings = [
        {
            "date":      dates[i],
            "accession": accessions[i] if i < len(accessions) else "",
            "doc":       prim_docs[i]  if i < len(prim_docs)  else "",
        }
        for i in range(len(forms))
        if forms[i] == "4" and i < len(dates) and dates[i] >= cutoff
    ]
    form4_count = len(form4_filings)
    log.debug("EDGAR %s: CIK=%s form4=%d (last %dd)", ticker, cik, form4_count, lookback_days)

    # Parse up to 3 most-recent Form 4 XMLs to determine open-market purchases
    key_purchases: List[float] = []   # USD values of purchases by key officers
    for filing in form4_filings[:3]:
        if not filing["accession"] or not filing["doc"]:
            continue
        txs = _parse_form4_xml(cik, filing["accession"], filing["doc"])
        for tx in txs:
            if tx["code"] != "P":
                continue
            if any(role in tx["title"].upper() for role in _KEY_ROLES):
                key_purchases.append(tx["value"])

    total_purchases_usd = 0.0
    ceo_buy = False
    days_since_most_recent = 0
    if key_purchases:
        total_purchases_usd = sum(key_purchases)
        ceo_buy = total_purchases_usd > 25_000
        log.debug(
            "INSIDER %s: %d key purchases $%.0f ceo_buy=%s",
            ticker, len(key_purchases), total_purchases_usd, ceo_buy,
        )
        if form4_filings:
            most_recent_date_str = form4_filings[0]["date"]
            try:
                from datetime import date as _date
                delta = (
                    datetime.now(timezone.utc).date()
                    - _date.fromisoformat(most_recent_date_str)
                ).days
                days_since_most_recent = max(0, delta)
            except Exception:
                days_since_most_recent = 0

    return form4_count, total_purchases_usd, ceo_buy, days_since_most_recent


# ── Multi-market helpers ───────────────────────────────────────────────────────

def _load_registry_tickers() -> Dict[str, List[str]]:
    """Load EU/Asia ticker lists from config/ticker_registry.json."""
    reg_path = ROOT / "config" / "ticker_registry.json"
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        return {
            "EUROPE": [e["ticker"] for e in data.get("europe", [])],
            "ASIA":   [e["ticker"] for e in data.get("asia", [])],
        }
    except Exception as exc:
        log.warning("ticker_registry load failed: %s — EU/Asia skipped", exc)
        return {"EUROPE": [], "ASIA": []}


def _registry_meta() -> Dict[str, Dict[str, Any]]:
    """Return {ticker: {sector, cap_tier}} from ticker_registry.json."""
    reg_path = ROOT / "config" / "ticker_registry.json"
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        meta: Dict[str, Dict[str, Any]] = {}
        for e in data.get("europe", []) + data.get("asia", []):
            meta[e["ticker"]] = {"sector": e["sector"], "cap_tier": e["cap_tier"]}
        return meta
    except Exception:
        return {}


def _badge_from_score(score: float) -> str:
    if score >= 0.65:
        return "HIGH BUY"
    if score >= 0.45:
        return "TACTICAL BUY"
    return "WATCHLIST"


def _score_ticker_eu(entry: Any) -> Optional[Dict[str, Any]]:
    """Score a European ticker from FMPFetcher raw_factors."""
    try:
        rf = entry.raw_factors
        momentum = float(rf.get("momentum", 0))
        eps = float(rf.get("eps", 0))
        score = round((momentum * 0.5 + min(eps / 100.0, 1.0) * 0.5) *
                      entry.source_reliability, 4)
        return {
            "ticker": entry.ticker,
            "final_score": score,
            "badge": _badge_from_score(score),
            "factors": {"momentum": momentum, "eps_proxy": eps},
            "sector": entry.sector,
            "cap_tier": entry.cap_tier,
            "source_reliability": entry.source_reliability,
            "market": "EUROPE",
        }
    except Exception as exc:
        log.debug("_score_ticker_eu skip %s: %s", entry.ticker, exc)
        return None


def _score_ticker_asia(entry: Any) -> Optional[Dict[str, Any]]:
    """Score an Asian ticker from AsianMarketFetcher raw_factors."""
    try:
        rf = entry.raw_factors
        momentum = float(rf.get("momentum", 0))
        eps = float(rf.get("eps", 0))
        score = round((momentum * 0.5 + min(eps / 1000.0, 1.0) * 0.5) *
                      entry.source_reliability, 4)
        return {
            "ticker": entry.ticker,
            "final_score": score,
            "badge": _badge_from_score(score),
            "factors": {"momentum": momentum, "eps_proxy": eps},
            "sector": entry.sector,
            "cap_tier": entry.cap_tier,
            "source_reliability": entry.source_reliability,
            "market": "ASIA",
        }
    except Exception as exc:
        log.debug("_score_ticker_asia skip %s: %s", entry.ticker, exc)
        return None


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

    # ── EDGAR connectivity preflight ──────────────────────────────────────────
    # Log the effective User-Agent and test data.sec.gov with AAPL (CIK 320193).
    # This diagnostic appears in verbose CI output and helps trace 403/timeout.
    _ua = os.getenv("EDGAR_USER_AGENT") or "regime-trader-research n.tardy@hotmail.fr"
    log.info("EDGAR User-Agent: %.40s%s", _ua, "…" if len(_ua) > 40 else "")
    try:
        _test = _sec_get("https://data.sec.gov/submissions/CIK0000320193.json", timeout=15)
        _d = _test.json().get("filings", {}).get("recent", {})
        _f4 = sum(1 for f in _d.get("form", []) if f == "4")
        log.info("EDGAR preflight OK — AAPL submissions: %d form4 filings", _f4)
    except Exception as _exc:
        log.warning("EDGAR preflight FAILED — data.sec.gov unreachable: %s", _exc)

    # ── FMP: per-ticker profile (up to 80/run to stay within 250/day budget) ────
    log.info("Fetching FMP profiles (per-ticker, up to 80)…")
    fmp_cap = 80
    mktcaps = fetch_fmp_profiles(tickers)
    fmp_count = sum(1 for t in tickers[:fmp_cap] if mktcaps.get(t, 0) > 0)

    # ── Congress feed ─────────────────────────────────────────────────────────
    log.info("Fetching congress trading data…")
    congress_data = fetch_congress_buys()

    # ── SPY baseline — fetched once so all threads share same benchmark ───────
    log.info("Fetching SPY 3-month return (shared baseline for momentum)…")
    spy_return_baseline = _fetch_spy_return()
    log.info("SPY 3-month return: %.4f (%.1f%%)", spy_return_baseline, spy_return_baseline * 100)

    # ── Quiver insider — primary source (pre-parsed Form 4, cached 6h) ───────────
    # Quiver returns structured JSON per ticker — no XML parsing, no per-ticker
    # rate limit.  QuiverClient has a 6h file-based TTL so repeated runs within
    # the same window skip HTTP and read from disk.
    log.info("Pre-fetching Quiver insider transactions for %d tickers…", len(tickers))
    quiver_insider_cache: Dict[str, Tuple[float, int]] = fetch_quiver_insider_all(tickers)

    # ── Finnhub insider — fallback when Quiver key absent ────────────────────────
    # Running inside the thread pool (160 concurrent calls) saturates the free
    # tier immediately; all 429 errors are silently swallowed by the per-ticker
    # try/except, causing insider=0.0 for every ticker.  Only pre-fetched when
    # Quiver produced no results (key absent or all zeros).
    finnhub_key_global = os.getenv("FINNHUB_API_KEY", "")
    finnhub_insider_cache: Dict[str, Tuple[float, int]] = {}
    quiver_has_data = any(v[0] > 0 for v in quiver_insider_cache.values())
    if not quiver_has_data and finnhub_key_global:
        log.info(
            "Quiver insider returned no data — falling back to Finnhub for %d tickers…",
            len(tickers),
        )
        finnhub_insider_cache = fetch_all_finnhub_insider(tickers, finnhub_key_global)
    elif not quiver_has_data:
        log.info("QUIVER_API_KEY and FINNHUB_API_KEY both absent — insider scoring uses EDGAR XML only")

    # ── EDGAR + yfinance: parallel per-ticker ─────────────────────────────────
    results = []
    errors  = 0

    def _score_ticker(row: Dict[str, str]) -> Dict[str, Any]:
        ticker = row["ticker"]
        edgar_ok    = False
        form4_count = 0
        total_purchases_usd = 0.0
        ceo_buy     = False
        days_since_most_recent = 0
        try:
            form4_count, total_purchases_usd, ceo_buy, days_since_most_recent = fetch_edgar_data(ticker)
            edgar_ok = True
        except Exception as exc:
            log.warning("EDGAR unreachable for %s: %s", ticker, exc)

        mktcap = mktcaps.get(ticker, 0.0)
        congress_raw = congress_data.get(ticker)

        try:
            finnhub_key = finnhub_key_global  # use pre-fetched key (avoids per-thread env lookup)

            # Insider purchases: Quiver (primary) → Finnhub (fallback) → EDGAR XML.
            # Quiver and Finnhub caches were both built serially before the thread pool.
            if ticker in quiver_insider_cache:
                quiver_usd, quiver_days = quiver_insider_cache[ticker]
                if quiver_usd > 0:
                    total_purchases_usd    = quiver_usd
                    days_since_most_recent = quiver_days
                    ceo_buy = total_purchases_usd > 25_000
            elif finnhub_key and ticker in finnhub_insider_cache:
                insider_usd_finnhub, days_finnhub = finnhub_insider_cache[ticker]
                if insider_usd_finnhub > 0:
                    total_purchases_usd    = insider_usd_finnhub
                    days_since_most_recent = days_finnhub
                    ceo_buy = total_purchases_usd > 25_000

            e_score = score_edgar(form4_count)
            i_score = score_insider_value(total_purchases_usd, mktcap, days_since_most_recent)
            c_score = score_congress(congress_raw)
            n_score = (
                score_news_finnhub(ticker, finnhub_key)
                if finnhub_key
                else _score_news_yfinance(ticker)
            )
            price_data = fetch_price_data(ticker, spy_return=spy_return_baseline)
            m_score = score_momentum(
                price_data["return_20d"],
                price_data["spy_return_20d"],
                price_data["volume_spike"],
            )

            # Determine which insider source actually provided data
            _quiver_usd = quiver_insider_cache.get(ticker, (0.0, 0))[0]
            _finnhub_usd = finnhub_insider_cache.get(ticker, (0.0, 0))[0]
            if _quiver_usd > 0:
                insider_source = "quiver"
            elif _finnhub_usd > 0:
                insider_source = "finnhub"
            elif total_purchases_usd > 0:
                insider_source = "edgar"
            else:
                insider_source = "none"

            quiver_evidence = {
                "congress": {
                    "purchases":       int(congress_raw.get("purchases", 0)) if congress_raw else 0,
                    "sales":           int(congress_raw.get("sales", 0)) if congress_raw else 0,
                    "net":             int(
                        congress_raw.get(
                            "net",
                            congress_raw.get("purchases", 0) - congress_raw.get("sales", 0),
                        )
                    ) if congress_raw else 0,
                    "recency_days":    congress_raw.get("recency_days") if congress_raw else None,
                    "representatives": congress_raw.get("representatives", []) if congress_raw else [],
                },
                "source": "quiver" if (congress_raw and congress_raw.get("recency_days") is not None) else "s3",
                "insider_source": insider_source,
            }

            if finnhub_key:
                news_source = "finnhub" if n_score > 0.0 else "none"
            else:
                news_source = "yfinance" if n_score > 0.0 else "none"

            return {
                "ticker":          ticker,
                "sector":          sector.get(ticker, "Unknown"),
                "cap_tier":        cap_tier.get(ticker, "large"),
                "market_cap":      mktcap,
                "edgar_score":     e_score,
                "insider_score":   i_score,
                "congress_score":  c_score,
                "news_score":      n_score,
                "momentum_score":  m_score,
                "ceo_buy":         ceo_buy,
                "form4_count":     form4_count,
                "quiver_evidence": quiver_evidence,
                "news_source":             news_source,
                "insider_usd":             float(total_purchases_usd),
                "momentum_spy_relative":   float(price_data["return_20d"] - price_data["spy_return_20d"]),
                "volume_spike":            float(price_data["volume_spike"]),
                "_edgar_ok":       edgar_ok,
                "_scoring_error":  False,
            }
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", ticker, exc)
            return {
                "ticker":          ticker,
                "sector":          sector.get(ticker, "Unknown"),
                "cap_tier":        cap_tier.get(ticker, "large"),
                "market_cap":      mktcap,
                "edgar_score":     0.0,
                "insider_score":   0.0,
                "congress_score":  0.0,
                "news_score":      0.0,
                "momentum_score":  0.0,
                "ceo_buy":         ceo_buy,
                "form4_count":     form4_count,
                "quiver_evidence": {},
                "news_source":             "none",
                "insider_usd":             float(total_purchases_usd),
                "momentum_spy_relative":   0.0,
                "volume_spike":            1.0,
                "_edgar_ok":       edgar_ok,
                "_scoring_error":  True,
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_ticker, row): row for row in ticker_rows}
        for fut in as_completed(futures):
            r = fut.result()
            if r.get("_scoring_error", False):
                errors += 1
            results.append(r)

    # ── EU / Asia scoring ─────────────────────────────────────────────────────
    registry_tickers = _load_registry_tickers()
    _meta = _registry_meta()
    if any(registry_tickers.values()):
        from regime_trader.fetchers import Orchestrator  # noqa: PLC0415
        from regime_trader.fetchers.fmp_fetcher import FMPFetcher  # noqa: PLC0415
        from regime_trader.fetchers.asian_fetcher import AsianMarketFetcher  # noqa: PLC0415

        fmp_key = os.environ.get("FMP_API_KEY", "")
        eu_asia_fetchers = []
        if fmp_key and registry_tickers.get("EUROPE"):
            eu_asia_fetchers.append(FMPFetcher(api_key=fmp_key))
        if registry_tickers.get("ASIA"):
            eu_asia_fetchers.append(AsianMarketFetcher())

        if eu_asia_fetchers:
            orch = Orchestrator(eu_asia_fetchers)
            raw_entries = orch.run(registry_tickers)
            for e in raw_entries:
                m = _meta.get(e.ticker, {})
                e.sector = m.get("sector", "Unknown")
                e.cap_tier = m.get("cap_tier", "large")

            scorer_map = {"EUROPE": _score_ticker_eu, "ASIA": _score_ticker_asia}
            with ThreadPoolExecutor(max_workers=4) as eu_pool:
                eu_futures = {
                    eu_pool.submit(scorer_map[e.market.value], e): e.ticker
                    for e in raw_entries
                    if e.market.value in scorer_map
                }
                for fut in as_completed(eu_futures):
                    scored = fut.result()
                    if scored:
                        results.append(scored)

    # edgar_count = tickers where EDGAR was reachable (even if 0 filings returned).
    edgar_count   = sum(1 for r in results if r.get("_edgar_ok", False))
    congress_count = len(congress_data)
    duration      = round(time.time() - t0, 2)

    # ── Stage 1 gate: stamp computed_at + run validate_raw ────────────────────
    # Stamp a row-level timestamp on every result so validate_dates() has a
    # per-row anchor.  Rows already carrying computed_at are left unchanged.
    pipeline_run_ts = datetime.now(timezone.utc).isoformat()
    for row in results:
        if "computed_at" not in row:
            row["computed_at"] = pipeline_run_ts

    # Build source_meta from the live run timestamps so validate_dates() can
    # check whether Quiver/Finnhub/EDGAR feeds are stale at source level.
    source_meta: Dict[str, Dict[str, Any]] = {
        "quiver":   {"last_updated": pipeline_run_ts},
        "finnhub":  {"last_updated": pipeline_run_ts},
        "edgar":    {"last_updated": pipeline_run_ts},
        "none":     {"last_updated": pipeline_run_ts},
    }

    try:
        clean_rows, quarantined_rows, val_issues = validate_raw(results, source_meta)
        quarantine_count = len(quarantined_rows)
        if quarantine_count:
            log.warning(
                "Stage 1 gate: %d/%d tickers quarantined — %s",
                quarantine_count,
                len(results),
                ", ".join({i.code for i in val_issues if i.code != "STALE_DATA"}),
            )
        else:
            log.info("Stage 1 gate: all %d tickers passed validation", len(results))
    except Exception as exc:
        # PipelineIntegrityError or unexpected — log but do not swallow; re-raise
        log.error("Stage 1 gate FAILED: %s", exc)
        raise

    status = {
        "_edgar_meta": {
            "last_run":             pipeline_run_ts,
            "run_duration_seconds": duration,
            "ticker_count":         len(tickers),
            "edgar_count":          edgar_count,
            "fmp_count":            fmp_count,
            "congress_count":       congress_count,
            "error_count":          errors,
            "quarantine_count":     quarantine_count,
        },
        "source_meta":   source_meta,
        "weights":       WEIGHTS,
        "results":       results,   # all rows (clean + quarantined with _validation_failed flag)
        "computed_at":   pipeline_run_ts,
    }

    out = log_dir / "intel_source_status.json"
    save_json_atomic(out, status)
    log.info(
        "Done in %.1fs — tickers=%d edgar=%d fmp_calls=%d congress=%d errors=%d → %s",
        duration, len(tickers), edgar_count, fmp_count, congress_count, errors, out,
    )
    return status


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EDGAR+FMP+yfinance daily pipeline")
    parser.add_argument("--tickers-file", type=Path, default=Path("config/universe.csv"))
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
