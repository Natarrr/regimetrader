"""regime_trader/services/fmp_client.py
FMP Ultimate unified client — replaces QuiverClient + Finnhub.

Endpoints:
  /api/v4/senate-trading?symbol=TICKER    — senate congressional trades
  /api/v4/house-trades?symbol=TICKER      — house congressional trades
  /api/v4/insider-trading?symbol=TICKER   — SEC Form 4 (limit=500)
  /api/v3/stock_news?tickers=TICKER       — news articles with sentiment
  /api/v3/quote/TICKER                    — price, marketCap, volume

Auth: FMP_API_KEY env var (Ultimate plan).
Cache: file-based under .cache/fmp/<bucket>/<ticker>.json with per-bucket TTL.
Rate: FMP_MAX_RPS env var (default 20; Ultimate cap is 50 req/s).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com"
_TIMEOUT = 15
_DEFAULT_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "fmp"
_DEFAULT_MAX_RPS = 20.0

_TTL: Dict[str, int] = {
    "congress": 12 * 3600,
    "insider":  12 * 3600,
    "news":      2 * 3600,
    "quote":         5 * 60,   # 5 minutes — fresh close for 21h UTC run
}


def _make_session(api_key: str) -> requests.Session:
    session = requests.Session()
    retry_kwargs: Dict[str, Any] = {
        "total": 3,
        "backoff_factor": 1.0,
        "status_forcelist": {429, 500, 502, 503, 504},
        "raise_on_status": False,
    }
    # urllib3 >= 1.26 renamed method_whitelist -> allowed_methods
    try:
        retry = Retry(**retry_kwargs, allowed_methods={"GET"})
    except TypeError:
        retry = Retry(**retry_kwargs, method_whitelist={"GET"})  # type: ignore[call-arg]
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FMPClient:
    """Unified FMP Ultimate client for all 5 scoring factors.

    Args:
        api_key:    FMP API key. Defaults to FMP_API_KEY env var.
        cache_root: Directory for file-based TTL cache. Defaults to .cache/fmp/.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_root: Optional[Path] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("FMP_API_KEY", "")
        self._cache_root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        self._session = _make_session(self._api_key) if self._api_key else None
        max_rps = float(os.getenv("FMP_MAX_RPS", _DEFAULT_MAX_RPS))
        self._min_delay = 1.0 / max(max_rps, 0.001)
        self._last_call = 0.0

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_path(self, bucket: str, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace("\\", "_")
        return self._cache_root / bucket / f"{safe}.json"

    def _cache_read(self, bucket: str, key: str, bypass_cache: bool = False) -> Optional[Any]:
        if bypass_cache:
            return None
        p = self._cache_path(bucket, key)
        ttl = _TTL.get(bucket, 6 * 3600)
        try:
            if not p.exists():
                return None
            if time.time() - p.stat().st_mtime > ttl:
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_write(self, bucket: str, key: str, value: Any) -> None:
        p = self._cache_path(bucket, key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value, default=str), encoding="utf-8")
        except Exception as exc:
            log.debug("fmp cache write failed %s/%s: %s", bucket, key, exc)

    # ── Rate-limited HTTP ──────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Any]:
        if not self._api_key or self._session is None:
            return None
        # Enforce rate limit
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_call = time.monotonic()

        url = f"{_FMP_BASE}{path}"
        p = dict(params or {})
        p["apikey"] = self._api_key
        try:
            resp = self._session.get(url, params=p, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("FMPClient GET %s failed: %s", path, exc)
            return None

    # ── Congress ───────────────────────────────────────────────────────────────

    def get_congress_trades(self, ticker: str, lookback_days: int = 180) -> Dict:
        """Aggregate congressional trades into {purchases, sales, total, recency_days}.

        Queries /api/v4/senate-trading (senate + house members both appear here on
        FMP Ultimate). Uses disclosureDate (not transactionDate) for recency_days —
        alpha decay starts when information is public, not when the trade occurred.
        Returns {} when no data or API key absent.
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("congress", ticker)
        if cached is not None:
            return cached

        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
        purchases = sales = total = 0
        recency_days = 9999
        now_date = datetime.now(timezone.utc).date()

        data = self._get("/api/v4/senate-trading", {"symbol": ticker}) or []
        for rec in data:
            disclosure = rec.get("disclosureDate", "")
            if not disclosure or disclosure < cutoff:
                continue
            tx_type = (rec.get("type") or "").lower()
            is_purchase = "purchase" in tx_type
            is_sale = "sale" in tx_type or "sold" in tx_type
            if is_purchase:
                purchases += 1
            elif is_sale:
                sales += 1
            else:
                continue
            total += 1
            try:
                d = datetime.fromisoformat(disclosure).date()
                recency_days = min(recency_days, (now_date - d).days)
            except Exception:
                pass

        if total == 0:
            self._cache_write("congress", ticker, {})
            return {}

        result = {
            "purchases": purchases,
            "sales": sales,
            "total": total,
            "recency_days": recency_days if recency_days < 9999 else None,
        }
        self._cache_write("congress", ticker, result)
        return result

    # ── Insider ────────────────────────────────────────────────────────────────

    def get_insider_purchases(
        self, ticker: str, lookback_days: int = 180
    ) -> Tuple[float, int]:
        """Return (total_purchases_usd, days_since_most_recent) from FMP Form 4.

        Filters to acquistionOrDisposition == 'A' (Acquisition) only.
        Uses limit=500 to cover mega-caps with frequent option grants.
        Null/empty securitiesTransacted or price are treated as 0 (no ValueError).
        Returns (0.0, 0) on empty response or API error.
        """
        if not self._api_key:
            return 0.0, 0
        cached = self._cache_read("insider", ticker)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
        data = self._get("/api/v4/insider-trading", {"symbol": ticker, "limit": 500}) or []

        total_usd = 0.0
        most_recent_days = 0
        now_date = datetime.now(timezone.utc).date()

        for r in data:
            if r.get("acquistionOrDisposition") != "A":
                continue
            shares = float(r.get("securitiesTransacted") or 0)
            price = float(r.get("price") or 0)
            if shares <= 0 or price <= 0:
                continue
            tx_date = r.get("transactionDate", "")
            if tx_date and tx_date < cutoff:
                continue
            total_usd += shares * price
            try:
                d = datetime.fromisoformat(tx_date).date()
                most_recent_days = max(most_recent_days, (now_date - d).days)
            except Exception:
                pass

        result = (round(total_usd, 2), most_recent_days)
        self._cache_write("insider", ticker, list(result))
        return result

    # ── News ───────────────────────────────────────────────────────────────────

    def get_news_raw_articles(self, ticker: str) -> List[Dict]:
        """Return raw FMP news articles for a ticker (cached 2h).

        Callers (score_news_fmp in run_pipeline.py) handle scoring and fallback.
        Returns [] on error or empty response.
        """
        if not self._api_key:
            return []
        cached = self._cache_read("news", ticker)
        if cached is not None:
            return cached
        data = self._get("/api/v3/stock_news", {"tickers": ticker, "limit": 50}) or []
        result: List[Dict] = data if isinstance(data, list) else []
        self._cache_write("news", ticker, result)
        return result

    # ── Quote ──────────────────────────────────────────────────────────────────

    def get_quote(self, ticker: str, bypass_cache: bool = False) -> Dict:
        """Return FMP quote dict: {price, marketCap, volume, avgVolume, eps, ...}.

        bypass_cache=True forces a live call regardless of the 5-min TTL.
        Accepts international suffixes natively (SAP.DE, 7203.T) on Ultimate.
        Returns {} on error or empty response.
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("quote", ticker, bypass_cache=bypass_cache)
        if cached is not None:
            return cached
        data = self._get(f"/api/v3/quote/{ticker}") or []
        result: Dict = data[0] if isinstance(data, list) and data else {}
        if result:
            self._cache_write("quote", ticker, result)
        return result
