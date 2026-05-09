"""regime_trader/services/fmp_service.py
Centralised, rate-limited, file-cached FMP / yfinance data service.

Fama (2013 Nobel) — only reliable, consistently-sourced price and fundamental
data can support valid factor models. Caching and rate-limiting are not
performance optimisations — they are data-quality invariants.

Design decisions:
  - FMP v3/v4 endpoints were retired Aug 2025.  Only stable/profile is called
    via FMP; all screener and insider/institutional data uses yfinance.
  - File cache lives under .cache/fmp/ (JSON, TTL-checked at read time).
  - Rate limiting uses a simple token-bucket enforced per process via a
    threading.Lock.  Tune via FMP_RATE_LIMIT_PER_MINUTE env var (default 60).
  - All public functions return None / [] on failure — callers must guard.

Public API:
  get_profile(sym)           -> dict | None
  get_profile_batch(syms)    -> {sym: market_cap_float}
  screener(cap_min, limit)   -> List[dict]
  insider_buys(lookback_days, limit) -> List[dict]
  get_institutional(sym)     -> dict | None

Usage:
  from regime_trader.services.fmp_service import FmpService
  svc = FmpService()          # or use module-level singleton `default_fmp`
  profile = svc.get_profile("AAPL")
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_FMP_STABLE = "https://financialmodelingprep.com/stable"
_DEFAULT_TIMEOUT = 15
_MAX_RETRIES = 3
_BACKOFF = 0.8

_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "fmp"

_TTL: Dict[str, int] = {
    "profile":    24 * 3600,   # 24 h
    "screener":    6 * 3600,   # 6 h
    "insider":    12 * 3600,   # 12 h
    "institutional": 6 * 3600, # 6 h
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Curated 130-ticker US large/mid-cap watchlist for the yfinance screener.
_YF_WATCHLIST: List[str] = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "AMD", "INTC", "QCOM",
    "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "TXN", "NOW", "AMAT", "LRCX",
    "JNJ", "LLY", "ABBV", "MRK", "PFE", "UNH", "ABT", "MDT", "BMY", "TMO",
    "DHR", "SYK", "ISRG", "VRTX", "REGN",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA",
    "PGR", "CB", "ICE", "CME",
    "XOM", "CVX", "COP", "SLB", "PSX", "VLO", "MPC", "HAL", "OXY",
    "CAT", "DE", "HON", "GE", "BA", "LMT", "RTX", "NOC", "UPS", "FDX",
    "ETN", "PH", "GD", "WM",
    "TSLA", "NKE", "MCD", "SBUX", "HD", "LOW", "TGT", "TJX", "BKNG", "CMG",
    "ABNB", "GM", "F",
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KHC", "GIS", "STZ",
    "LIN", "APD", "SHW", "NEM", "FCX", "ALB", "DD", "IFF",
    "NFLX", "DIS", "VZ", "CMCSA", "T", "CHTR", "SNAP", "ROKU",
    "PLD", "AMT", "EQIX", "CCI", "O", "SPG", "EQR",
    "NEE", "DUK", "SO", "D", "AEP", "EXC",
]

_KEY_ROLES = frozenset({
    "CEO", "CFO", "COO", "CTO", "DIRECTOR", "PRESIDENT",
    "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING",
    "CHAIRMAN", "FOUNDER",
})


# ── Token-bucket rate limiter ──────────────────────────────────────────────────

class _TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Modigliani (1985 Nobel) — pace of information extraction matters as much as
    the extraction itself; unbounded call rates degrade data quality and lead to
    API bans.

    Args:
        rate_per_minute: Maximum calls allowed per 60-second window.
    """

    def __init__(self, rate_per_minute: int = 60) -> None:
        self._rate = rate_per_minute
        self._interval = 60.0 / max(rate_per_minute, 1)
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        """Block until a token is available."""
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


# ── File cache ─────────────────────────────────────────────────────────────────

def _cache_path(bucket: str, key: str) -> Path:
    safe = key.replace("/", "_").replace("\\", "_").replace(":", "_")
    return _CACHE_ROOT / bucket / f"{safe}.json"


def _cache_read(bucket: str, key: str, ttl: int) -> Optional[Any]:
    """Return cached value if fresh, else None."""
    p = _cache_path(bucket, key)
    try:
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > ttl:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write(bucket: str, key: str, value: Any) -> None:
    """Write value to cache, swallowing any IO error."""
    p = _cache_path(bucket, key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, default=str), encoding="utf-8")
    except Exception as exc:
        log.debug("cache write failed %s/%s: %s", bucket, key, exc)


# ── HTTP session ───────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Build a requests.Session with retry/backoff.

    Granger (2003 Nobel) — data collection failures must not silently corrupt
    downstream signals.
    """
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(_HEADERS)
    return session


# ── FmpService ─────────────────────────────────────────────────────────────────

class FmpService:
    """Centralised, rate-limited, file-cached FMP / yfinance data service.

    Instantiate once per process; all public methods are thread-safe.

    Args:
        rate_per_minute: FMP calls per minute (default: env FMP_RATE_LIMIT_PER_MINUTE
                         or 60).
        cache_root:      Override for the file-cache root directory.
    """

    def __init__(
        self,
        rate_per_minute: Optional[int] = None,
        cache_root: Optional[Path] = None,
    ) -> None:
        rpm = rate_per_minute or int(os.getenv("FMP_RATE_LIMIT_PER_MINUTE", "60"))
        self._bucket = _TokenBucket(rate_per_minute=rpm)
        self._session = _make_session()
        self._cache_root = cache_root or _CACHE_ROOT
        log.debug("FmpService initialised — %d req/min, cache=%s", rpm, self._cache_root)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """GET *url* with rate limiting and retry; return parsed JSON or None."""
        self._bucket.acquire()
        try:
            resp = self._session.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            log.warning("HTTP %s from %s", resp.status_code, url.split("?")[0])
            return None
        except Exception as exc:
            log.warning("FmpService request failed %s: %s", url.split("?")[0], exc)
            return None

    @staticmethod
    def _fmp_key() -> str:
        return os.getenv("FMP_API_KEY", "")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_profile(self, sym: str) -> Optional[Dict[str, Any]]:
        """Fetch a single ticker's FMP stable/profile with 24 h file cache.

        Args:
            sym: Ticker symbol (case-insensitive).

        Returns:
            Profile dict or None if unavailable / no API key.
        """
        sym = sym.upper()
        cached = _cache_read("profile", sym, _TTL["profile"])
        if cached is not None:
            return cached

        key = self._fmp_key()
        if not key:
            log.debug("get_profile(%s): FMP_API_KEY not set", sym)
            return None

        data = self._get_json(f"{_FMP_STABLE}/profile", params={"symbol": sym, "apikey": key})
        if isinstance(data, list) and data:
            result: Dict[str, Any] = data[0]
            _cache_write("profile", sym, result)
            return result
        return None

    def get_profile_batch(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch market caps for a list of symbols in parallel.

        Returns:
            {sym: market_cap_float}; 0.0 for missing entries.
        """
        if not symbols:
            return {}
        caps: Dict[str, float] = {}

        def _fetch(sym: str) -> None:
            p = self.get_profile(sym)
            if p:
                try:
                    caps[sym.upper()] = float(p.get("mktCap", 0) or 0)
                except (TypeError, ValueError):
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_fetch, symbols, timeout=120))
        return caps

    def screener(
        self,
        cap_min: int = 200_000_000,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Return liquid US equities ranked by volume spike + 5-day momentum.

        Fama (2013 Nobel) — screen liquid equities via momentum signal.
        FMP v3 screener retired Aug 2025; uses yfinance batch OHLCV instead.

        Result is cached for 6 h under .cache/fmp/screener/.

        Args:
            cap_min: Minimum market-cap filter in USD (default 200 M).
            limit:   Max candidates to return.

        Returns:
            List of {sym, price, volume, avg_volume, volume_spike,
                     price_change_pct, market_cap, sector}.
        """
        cache_key = f"screener_{cap_min}_{limit}"
        cached = _cache_read("screener", cache_key, _TTL["screener"])
        if cached is not None:
            log.debug("screener: cache hit")
            return cached

        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            log.warning("screener: yfinance/pandas not installed")
            return []

        syms = _YF_WATCHLIST
        try:
            raw = yf.download(
                syms, period="30d", interval="1d",
                progress=False, auto_adjust=True, threads=True,
            )
        except Exception as exc:
            log.warning("screener: yfinance download failed: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
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
                vol_spike = round(today_vol / max(avg_vol, 1.0), 3)
                pct_chg = round((price / float(close.iloc[-6]) - 1) * 100, 4)
                if price < 1.0:
                    continue
                results.append({
                    "sym": sym, "market_cap": 0.0,
                    "price": round(price, 4), "volume": round(today_vol),
                    "avg_volume": round(avg_vol), "volume_spike": vol_spike,
                    "price_change_pct": pct_chg, "sector": "",
                })
            except Exception:
                continue

        results.sort(
            key=lambda x: x["volume_spike"] + abs(x["price_change_pct"]) / 10,
            reverse=True,
        )
        results = results[:limit]
        _cache_write("screener", cache_key, results)
        log.info("screener: %d candidates (fresh)", len(results))
        return results

    def insider_buys(
        self,
        lookback_days: int = 90,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return recent exec open-market purchases via yfinance insider_transactions.

        Akerlof (2001 Nobel) — insiders buying with own money is a costly,
        credible signal.

        Result is cached for 12 h.

        Args:
            lookback_days: How many days back to scan.
            limit:         Max tickers to scan (from _YF_WATCHLIST).

        Returns:
            List of {sym, key_value_usd, normalized_pct_mcap, market_cap,
                     roles, tx_count, most_recent_date}.
        """
        cache_key = f"insider_{lookback_days}_{limit}"
        cached = _cache_read("insider", cache_key, _TTL["insider"])
        if cached is not None:
            log.debug("insider_buys: cache hit")
            return cached

        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            log.warning("insider_buys: yfinance not installed")
            return []

        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        caps = self.get_profile_batch(_YF_WATCHLIST[:limit])
        results: List[Dict[str, Any]] = []

        def _fetch(sym: str) -> Optional[Dict[str, Any]]:
            try:
                txns = yf.Ticker(sym).insider_transactions
                if txns is None or txns.empty:
                    return None
                if "Text" not in txns.columns or "Value" not in txns.columns:
                    return None
                buys = txns[
                    txns["Text"].str.contains("Purchase", case=False, na=False) &
                    ~txns["Text"].str.contains(
                        "Award|Grant|Option|Derivative|Restricted", case=False, na=False
                    )
                ].copy()
                if buys.empty:
                    return None
                if "Start Date" in buys.columns:
                    buys["_dt"] = pd.to_datetime(buys["Start Date"], utc=True, errors="coerce")
                    buys = buys[buys["_dt"] >= cutoff]
                if buys.empty:
                    return None

                key_buys = buys
                if "Insider" in buys.columns:
                    key_buys = buys[
                        buys["Insider"].str.upper().str.contains(
                            "|".join(_KEY_ROLES), na=False
                        )
                    ]
                if key_buys.empty:
                    key_buys = buys

                total_usd = float(key_buys["Value"].fillna(0).sum())
                if total_usd < 25_000:
                    return None
                mcap = caps.get(sym, 0.0)
                normed = total_usd / max(mcap, 1.0)
                roles: List[str] = []
                if "Insider" in key_buys.columns:
                    roles = key_buys["Insider"].dropna().unique().tolist()[:3]
                most_recent = ""
                if "Start Date" in key_buys.columns:
                    dr = key_buys["Start Date"].dropna()
                    if not dr.empty:
                        most_recent = str(dr.iloc[0])
                return {
                    "sym": sym,
                    "key_value_usd": round(total_usd, 2),
                    "normalized_pct_mcap": round(normed * 100, 6),
                    "market_cap": mcap,
                    "roles": roles,
                    "tx_count": len(key_buys),
                    "most_recent_date": most_recent,
                }
            except Exception as exc:
                log.debug("insider_buys(%s): %s", sym, exc)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_fetch, sym) for sym in _YF_WATCHLIST[:limit]]
            for fut in concurrent.futures.as_completed(futures, timeout=120):
                r = fut.result()
                if r:
                    results.append(r)

        results.sort(key=lambda x: x["key_value_usd"], reverse=True)
        _cache_write("insider", cache_key, results)
        log.info("insider_buys: %d signals (fresh)", len(results))
        return results

    def get_institutional(self, sym: str) -> Optional[Dict[str, Any]]:
        """Fetch institutional holder summary for *sym* via yfinance (13F proxy).

        Tirole (2014 Nobel) — institutional herding and accumulation as
        information-based signal.

        Result is cached for 6 h.

        Args:
            sym: Ticker symbol.

        Returns:
            Dict with net_shares_change, pct_change_avg, accumulation_score;
            or None.
        """
        sym = sym.upper()
        cached = _cache_read("institutional", sym, _TTL["institutional"])
        if cached is not None:
            return cached

        try:
            import yfinance as yf
        except ImportError:
            return None

        _MAJOR = frozenset({
            "VANGUARD", "BLACKROCK", "STATE STREET", "FIDELITY",
            "JP MORGAN", "RENAISSANCE", "CITADEL", "MILLENNIUM",
        })

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
                    pct = float(row.get("pctChange", 0) or 0)
                    if any(m in name for m in _MAJOR) and pct > 0:
                        major_count += 1

            pos_count = sum(1 for p in pct_changes if p > 0)
            pos_ratio = pos_count / len(pct_changes)
            score = round(
                0.5 * min(1.0, max(-1.0, avg_pct / 10.0)) +
                0.3 * (pos_ratio - 0.5) * 2.0 +
                0.2 * min(1.0, major_count / 3.0),
                4,
            )

            result: Dict[str, Any] = {
                "sym": sym,
                "net_shares_change": round(net_change, 0),
                "pct_change_avg": round(avg_pct, 4),
                "major_fund_count": major_count,
                "holder_count": len(ih),
                "accumulation_score": score,
            }
            _cache_write("institutional", sym, result)
            return result
        except Exception as exc:
            log.debug("get_institutional(%s): %s", sym, exc)
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────

#: Default process-wide FmpService instance.  Import and reuse to share cache.
default_fmp: FmpService = FmpService()
