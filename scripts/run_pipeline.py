"""scripts/run_pipeline.py
EDGAR + FMP + yfinance daily data pipeline.

Stiglitz (2001 Nobel) — asymmetric information: insider filing activity is a
credible, costly-to-fake signal. This pipeline sources it from two layers:
  1. SEC EDGAR daily index  — Form 4 filings (free, cached 24 h)
  2. FMP insider-trading    — structured buy/sell with role classification
     (1 API call/day; TTL 12 h)
  3. yfinance               — news sentiment + 20-day price momentum (free)

FMP budget: ≤ 2 calls per run (profile batch + insider list).
With caching, repeated intraday runs spend 0 additional FMP calls.

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

from regime_trader.utils.io import save_json_atomic
from regime_trader.services.quiver_client import QuiverClient as _QuiverClient

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
    """Fama (2013 Nobel): batch profile fetch — ≤2 FMP calls for 160 tickers (chunks of 100).

    FMP /stable/profile accepts comma-separated symbols. Chunking at 100 keeps
    each request well within URL length limits and allows partial success.
    """
    result: Dict[str, float] = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i : i + 100]
        data  = _fmp_get("profile", {"symbol": ",".join(chunk)})
        if data and isinstance(data, list):
            for row in data:
                sym = row.get("symbol", "")
                if sym:
                    result[sym] = float(row.get("mktCap") or 0)
        else:
            log.warning("FMP profile chunk %d–%d returned no data", i, i + len(chunk) - 1)
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
    if CONGRESS_CACHE_PATH.exists():
        try:
            cached = json.loads(CONGRESS_CACHE_PATH.read_text(encoding="utf-8"))
            age_h = (time.time() - float(cached.get("_ts", 0))) / 3600
            if age_h < _CONGRESS_TTL_HOURS:
                return cached.get("by_ticker", {})
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


def fetch_price_data(ticker: str) -> Dict[str, float]:
    """Thaler (2017 Nobel) — 20-day SPY-relative return + volume spike.

    Fetches ticker and SPY for fair comparison.
    Volume spike = 5-day avg volume / full-window avg volume.

    Returns {"return_20d": float, "spy_return_20d": float, "volume_spike": float}.
    Returns {"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0} on any error.
    """
    _default = {"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0}
    try:
        import yfinance as yf
        import numpy as np

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

        spy_df = yf.download("SPY", period="3mo", interval="1d",
                              progress=False, auto_adjust=True)
        if spy_df is None or spy_df.empty or len(spy_df) < 2:
            spy_ret = 0.0
        else:
            spy_close = spy_df["Close"].squeeze().dropna()
            spy_ret = float((spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0])

        return {
            "return_20d":     round(ret, 6),
            "spy_return_20d": round(spy_ret, 6),
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
    import requests as _req
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={api_key}"
    try:
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


# ── Per-ticker scorer ──────────────────────────────────────────────────────────

def score_edgar(form4_count: int) -> float:
    """Stiglitz (2001): normalise Form 4 filing count to [0.20, 0.90]."""
    if form4_count <= 0:
        return 0.30
    return round(min(0.90, 0.30 + form4_count * 0.12), 4)


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
    """
    from xml.etree import ElementTree as ET
    acc_nodash = accession.replace("-", "")
    cik_int = str(int(cik))   # strip leading zeros for the Archives path segment
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodash}/{primary_doc}"
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

    # ── FMP: profile (chunked) — insider now uses yfinance per-ticker ────────
    log.info("Fetching FMP profiles (chunked at 100)…")
    mktcaps = fetch_fmp_profiles(tickers)
    n_profile_chunks = math.ceil(len(tickers) / 100) if tickers else 0
    fmp_count = n_profile_chunks if mktcaps else 0

    # ── Congress feed ─────────────────────────────────────────────────────────
    log.info("Fetching congress trading data…")
    congress_data = fetch_congress_buys()

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
            finnhub_key = os.getenv("FINNHUB_API_KEY", "")
            e_score = score_edgar(form4_count)
            i_score = score_insider_value(total_purchases_usd, mktcap, days_since_most_recent)
            c_score = score_congress(congress_raw)
            n_score = (
                score_news_finnhub(ticker, finnhub_key)
                if finnhub_key
                else _score_news_yfinance(ticker)
            )
            price_data = fetch_price_data(ticker)
            m_score = score_momentum(
                price_data["return_20d"],
                price_data["spy_return_20d"],
                price_data["volume_spike"],
            )

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
            }

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
                "edgar_score":     0.30,
                "insider_score":   0.0,
                "congress_score":  0.0,
                "news_score":      0.0,
                "momentum_score":  0.0,
                "ceo_buy":         ceo_buy,
                "form4_count":     form4_count,
                "quiver_evidence": {},
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

    # edgar_count = tickers where EDGAR was reachable (even if 0 filings returned).
    edgar_count   = sum(1 for r in results if r.get("_edgar_ok", False))
    congress_count = len(congress_data)
    duration      = round(time.time() - t0, 2)

    status = {
        "_edgar_meta": {
            "last_run":             datetime.now(timezone.utc).isoformat(),
            "run_duration_seconds": duration,
            "ticker_count":         len(tickers),
            "edgar_count":          edgar_count,
            "fmp_count":            fmp_count,
            "congress_count":       congress_count,
            "error_count":          errors,
        },
        "weights": WEIGHTS,
        "results": results,
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
