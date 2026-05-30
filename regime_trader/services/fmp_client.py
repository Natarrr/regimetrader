"""regime_trader/services/fmp_client.py
FMP Ultimate unified client — migrated to stable/ routes (2026-05).

WHY THIS REWRITE
----------------
FMP retired /api/v3/ and /api/v4/ routes in Aug 2025 and consolidated
everything under https://financialmodelingprep.com/stable/.
The previous client silently 404'd on insider/news/quote, which zeroed
insider_conviction_score / news_sentiment_score across the universe.

CONGRESS ROUTES (senate-trading, house-trading):
Both returned HTTP 404 in the Phase-0 smoke-test (2026-05-30). These routes
are not available on the current plan. Congress scoring uses the S3 Stock
Watcher feeds as primary (run_pipeline.fetch_congress_buys). The FMP
get_congress_trades() method returns {} so existing callers keep working
without hitting a dead route.

This client:
  1. Uses stable/ routes exclusively (auth: ?apikey=).
  2. Raises FMPEndpointError on 401/403/404 instead of swallowing — a dead
     endpoint is a pipeline integrity event, NOT a sparse-data event.
  3. Exposes the full Ultimate surface used by the 7-factor model PLUS new
     premium endpoints (ratings, key-metrics, COT, transcripts, bulk quote).
  4. Tracks per-endpoint failure counts so monitoring can distinguish
     "ticker has no insider trades" (valid 0) from "insider route is down".

Auth: FMP_API_KEY env var (Ultimate plan).
Cache: file-based under .cache/fmp/<bucket>/<ticker>.json with per-bucket TTL.
Rate: FMP_MAX_RPS env var (Ultimate cap is 50 req/s). Default 30.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com"
_STABLE = f"{_FMP_BASE}/stable"
_TIMEOUT = 15
_DEFAULT_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "fmp"
_DEFAULT_MAX_RPS = 30.0

# Per-bucket cache TTL (seconds).
_TTL: Dict[str, int] = {
    "congress":     12 * 3600,   # stub — nothing fetched, but preserved for compatibility
    "insider":      12 * 3600,
    "news":          2 * 3600,
    "quote":             5 * 60,
    "ratings":       6 * 3600,
    "key_metrics":  24 * 3600,
    "ratios":       24 * 3600,
    "f13":          24 * 3600,
    "transcript":   24 * 3600,
    "profile":      24 * 3600,
}


class FMPEndpointError(RuntimeError):
    """Raised on 401/403/404 — a structural endpoint failure, not empty data.

    Distinguishing this from an empty-but-valid 200 response is the whole point:
    a dead route must trip the pipeline circuit-breaker, while a ticker that
    genuinely has no insider trades must score 0.0 without alarm.
    """

    def __init__(self, path: str, status: int) -> None:
        self.path = path
        self.status = status
        super().__init__(f"FMP endpoint {path} returned {status}")


def _make_session() -> requests.Session:
    session = requests.Session()
    retry_kwargs: Dict[str, Any] = {
        "total": 3,
        "backoff_factor": 1.0,
        # 403/404 are NOT retried — structural failures, retrying wastes quota.
        "status_forcelist": {429, 500, 502, 503, 504},
        "raise_on_status": False,
    }
    try:
        retry = Retry(**retry_kwargs, allowed_methods={"GET"})
    except TypeError:
        retry = Retry(**retry_kwargs, method_whitelist={"GET"})  # type: ignore[call-arg]
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FMPClient:
    """Unified FMP Ultimate client — stable/ routes, full premium surface.

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
        self._session = _make_session() if self._api_key else None
        max_rps = float(os.getenv("FMP_MAX_RPS", _DEFAULT_MAX_RPS))
        self._min_delay = 1.0 / max(max_rps, 0.001)
        self._last_call = 0.0
        # Observability: count structural failures per endpoint.
        self.endpoint_failures: Dict[str, int] = defaultdict(int)
        self.endpoint_calls: Dict[str, int] = defaultdict(int)

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

    def _get(self, path: str, params: Optional[Dict] = None,
             bucket: str = "") -> Optional[Any]:
        """GET stable/<path>. Returns parsed JSON, or None on transient/empty.

        Raises FMPEndpointError on 401/403/404 (structural) so callers can
        decide whether to trip the circuit-breaker.
        """
        if not self._api_key or self._session is None:
            return None

        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_call = time.monotonic()

        url = f"{_STABLE}/{path.lstrip('/')}"
        p = dict(params or {})
        p["apikey"] = self._api_key
        self.endpoint_calls[path] += 1
        try:
            resp = self._session.get(url, params=p, timeout=_TIMEOUT)
        except Exception as exc:
            log.warning("FMP GET %s network error: %s", path, exc)
            return None

        if resp.status_code in (401, 403, 404):
            self.endpoint_failures[path] += 1
            log.error(
                "FMP STRUCTURAL FAILURE %s -> HTTP %s. "
                "Endpoint may be deprecated or not in plan. "
                "This zeroes a factor; investigate before trusting scores.",
                path, resp.status_code,
            )
            raise FMPEndpointError(path, resp.status_code)

        if resp.status_code != 200:
            log.warning("FMP GET %s -> HTTP %s", path, resp.status_code)
            return None
        try:
            return resp.json()
        except Exception as exc:
            log.warning("FMP GET %s JSON decode failed: %s", path, exc)
            return None

    # ── Congress ───────────────────────────────────────────────────────────────

    def get_congress_trades(self, ticker: str, lookback_days: int = 180) -> Dict:
        """Congress trades stub — FMP stable/ senate/house routes returned HTTP 404
        in the Phase-0 smoke-test (2026-05-30). Congress scoring uses the S3
        Stock Watcher feeds (run_pipeline.fetch_congress_buys) as primary.

        Returns {} so existing callers keep working without hitting a dead route.
        If FMP restores these routes in a future plan, replace this stub with
        the get_congress_trades implementation from the provided fmp_client.py.
        """
        if not self._api_key:
            return {}
        log.debug(
            "get_congress_trades(%s): FMP congress routes are 404 on this plan — "
            "returning {} (S3 Stock Watcher is the active source).", ticker
        )
        return {}

    # ── Insider (stable/insider-trading/search) ────────────────────────────────

    def get_insider_purchases(
        self, ticker: str, lookback_days: int = 180
    ) -> Tuple[float, int]:
        """Return (total_acquisition_usd, days_since_most_recent) from Form 4.

        Filters acquisitionOrDisposition == 'A' only. Empty 200 -> (0.0, 0).
        """
        if not self._api_key:
            return 0.0, 0
        cached = self._cache_read("insider", ticker)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
        data = self._get("insider-trading/search",
                         {"symbol": ticker, "page": 0, "limit": 500},
                         bucket="insider") or []

        total_usd = 0.0
        most_recent_days = 0
        now_date = datetime.now(timezone.utc).date()

        for r in data:
            # Handle historical FMP typo: acquistionOrDisposition (missing 'i')
            aod = (r.get("acquisitionOrDisposition")
                   or r.get("acquistionOrDisposition") or "")
            if aod != "A":
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
                d = datetime.fromisoformat(tx_date[:10]).date()
                most_recent_days = max(most_recent_days, (now_date - d).days)
            except Exception:
                pass

        result = (round(total_usd, 2), most_recent_days)
        self._cache_write("insider", ticker, list(result))
        return result

    def get_insider_transactions(self, ticker: str, lookback_days: int = 90) -> Dict[str, List[Dict]]:
        """Return {'P': [...], 'S': [...]} for the breadth signal.

        score_insider_breadth needs P vs S by distinct insider_id.
        Uses the same stable/insider-trading/search route as get_insider_purchases.
        """
        if not self._api_key:
            return {"P": [], "S": []}
        data = self._get("insider-trading/search",
                         {"symbol": ticker, "page": 0, "limit": 500},
                         bucket="insider") or []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
        out: Dict[str, List[Dict]] = {"P": [], "S": []}
        for r in data:
            tx_date = str(r.get("transactionDate", ""))[:10]
            if tx_date and tx_date < cutoff:
                continue
            aod = (r.get("acquisitionOrDisposition")
                   or r.get("acquistionOrDisposition") or "")
            entry = {
                "insider_id": r.get("reportingCik") or r.get("reportingName"),
                "title": r.get("typeOfOwner", ""),
                "date": tx_date,
            }
            if aod == "A":
                out["P"].append(entry)
            elif aod == "D":
                out["S"].append(entry)
        return out

    # ── News (stable/news/stock) ───────────────────────────────────────────────

    def get_news_raw_articles(self, ticker: str) -> List[Dict]:
        """Return raw news articles for sentiment+buzz scoring (cached 2h)."""
        if not self._api_key:
            return []
        cached = self._cache_read("news", ticker)
        if cached is not None:
            return cached
        data = self._get("news/stock", {"symbols": ticker, "limit": 50},
                         bucket="news") or []
        result: List[Dict] = data if isinstance(data, list) else []
        self._cache_write("news", ticker, result)
        return result

    # ── Quote (stable/quote) ───────────────────────────────────────────────────

    def get_quote(self, ticker: str, bypass_cache: bool = False) -> Dict:
        """Return quote dict {price, marketCap, volume, avgVolume, eps, ...}.

        International suffixes (SAP.DE, 7203.T) confirmed live on Ultimate
        per Phase-0 smoke-test (2026-05-30).
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("quote", ticker, bypass_cache=bypass_cache)
        if cached is not None:
            return cached
        data = self._get("quote", {"symbol": ticker}, bucket="quote") or []
        result: Dict = data[0] if isinstance(data, list) and data else {}
        if result:
            self._cache_write("quote", ticker, result)
        return result

    # ── New Ultimate-tier endpoints ────────────────────────────────────────────

    def get_analyst_ratings(self, ticker: str) -> Dict:
        """Analyst consensus (stable/grades-consensus). PASS in smoke-test."""
        if not self._api_key:
            return {}
        cached = self._cache_read("ratings", ticker)
        if cached is not None:
            return cached
        data = self._get("grades-consensus", {"symbol": ticker}, bucket="ratings") or []
        result = data[0] if isinstance(data, list) and data else (data or {})
        self._cache_write("ratings", ticker, result)
        return result

    def get_key_metrics_ttm(self, ticker: str) -> Dict:
        """TTM key metrics (stable/key-metrics-ttm). PASS in smoke-test.

        Replaces yfinance .info scraping for quality/cannibal filters.
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("key_metrics", ticker)
        if cached is not None:
            return cached
        data = self._get("key-metrics-ttm", {"symbol": ticker}, bucket="key_metrics") or []
        result = data[0] if isinstance(data, list) and data else {}
        self._cache_write("key_metrics", ticker, result)
        return result

    def get_ratios_ttm(self, ticker: str) -> Dict:
        """TTM financial ratios (stable/ratios-ttm). PASS in smoke-test."""
        if not self._api_key:
            return {}
        cached = self._cache_read("ratios", ticker)
        if cached is not None:
            return cached
        data = self._get("ratios-ttm", {"symbol": ticker}, bucket="ratios") or []
        result = data[0] if isinstance(data, list) and data else {}
        self._cache_write("ratios", ticker, result)
        return result

    def get_price_target_consensus(self, ticker: str) -> Dict:
        """Price target consensus (stable/price-target-consensus). PASS in smoke-test."""
        if not self._api_key:
            return {}
        data = self._get("price-target-consensus", {"symbol": ticker},
                         bucket="ratings") or []
        return data[0] if isinstance(data, list) and data else {}

    def get_batch_quotes(self, tickers: List[str]) -> Dict[str, Dict]:
        """Batch quote (stable/batch-quote). PASS in smoke-test.

        One call for up to 100 tickers instead of N serial calls.
        Uses Ultimate bulk capability.
        """
        if not self._api_key or not tickers:
            return {}
        out: Dict[str, Dict] = {}
        CHUNK = 100
        for i in range(0, len(tickers), CHUNK):
            chunk = tickers[i:i + CHUNK]
            data = self._get("batch-quote", {"symbols": ",".join(chunk)},
                             bucket="quote") or []
            for row in (data if isinstance(data, list) else []):
                sym = row.get("symbol")
                if sym:
                    out[sym] = row
        return out

    def get_cot_report(self) -> List[Dict]:
        """Full Commitment of Traders report (stable/commitment-of-traders-report).

        PASS in Phase-0 smoke-test (536 rows). Returns one row per commodity
        contract with commPositionsLongAll / commPositionsShortAll (commercial
        hedger positions) and noncomm* (speculator positions).
        No symbol filter — returns the full universe; caller filters by name/symbol.
        Cached 12h (COT is weekly, published Fridays).
        """
        if not self._api_key:
            return []
        cached = self._cache_read("key_metrics", "_cot_full")
        if cached is not None:
            return cached
        data = self._get("commitment-of-traders-report", {}, bucket="key_metrics") or []
        result = data if isinstance(data, list) else []
        self._cache_write("key_metrics", "_cot_full", result)
        return result

    def get_cash_flow_statements(self, ticker: str, limit: int = 4) -> List[Dict]:
        """Quarterly cash flow statements (stable/cash-flow-statement). PASS in smoke-test.

        Used by satellite_factors cannibal filter (buyback yield).
        """
        if not self._api_key:
            return []
        data = self._get("cash-flow-statement",
                         {"symbol": ticker, "period": "quarter", "limit": limit},
                         bucket="key_metrics") or []
        return data if isinstance(data, list) else []

    # ── Health report ──────────────────────────────────────────────────────────

    def health_report(self) -> Dict[str, Any]:
        """Return per-endpoint call/failure counts for monitoring.

        A non-zero failure count on insider/news means the factor is being zeroed
        by a dead route — the circuit-breaker should NOT be lowered to compensate;
        the route should be fixed.
        """
        return {
            "calls": dict(self.endpoint_calls),
            "failures": dict(self.endpoint_failures),
            "has_structural_failure": any(self.endpoint_failures.values()),
        }
