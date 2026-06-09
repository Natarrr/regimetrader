"""regime_trader/services/fmp_client.py
FMP Ultimate unified client — migrated to stable/ routes (2026-05).

WHY THIS REWRITE
----------------
FMP retired /api/v3/ and /api/v4/ routes in Aug 2025 and consolidated
everything under https://financialmodelingprep.com/stable/.
The previous client silently 404'd on insider/news/quote, which zeroed
insider_conviction_score / news_sentiment_score across the universe.

PHASE-1 ENDPOINT VERIFICATION (2026-06-09, FMP Ultimate, scripts/fmp_endpoint_probe.py)
----------------------------------------------------------------------------------------
All active scoring endpoints confirmed HTTP 200:
  historical-price-eod/full          quote                  batch-quote
  grades-consensus                   ratios-ttm             enterprise-values
  cash-flow-statement                earning-call-transcript-latest
  commitment-of-traders-report       news/stock             institutional-ownership/symbol-positions-summary
  insider-trading/search             analyst-estimates      price-target-consensus
  upgrades-downgrades-consensus-bulk ratios-ttm-bulk        key-metrics-ttm-bulk

Confirmed HTTP 404 (quarantined):
  upgrades-downgrades    senate-trading    house-trading

PEAD NOTE: "earnings-surprises" is HTTP 404 on stable/ and is no longer called.
get_earnings_surprise() computes the surprise from stable/ "earnings"
(epsActual vs epsEstimated) instead — same return contract.

IMPORTANT — insider path distinction:
  "insider-trading"         → HTTP 404 (bare path, NOT used by this client)
  "insider-trading/search"  → HTTP 200 LIVE (the actual scoring path)
  Do not conflate these when reading error logs or editing probes.

CONGRESS ROUTES (senate-trading, house-trading):
Both returned HTTP 404 in Phase-0 (2026-05-30) and Phase-1 (2026-06-09). Congress
scoring uses S3 Stock Watcher feeds as primary (run_pipeline.fetch_congress_buys).
The get_congress_trades() method returns {} so callers keep working without the route.

This client:
  1. Uses stable/ routes exclusively (auth: ?apikey=).
  2. Raises FMPEndpointError on 401/403/404 instead of swallowing — a dead
     endpoint is a pipeline integrity event, NOT a sparse-data event.
  3. Exposes the full Ultimate surface used by the 9-factor model PLUS
     premium endpoints (ratings, key-metrics, COT, transcripts, bulk quote).
  4. Tracks per-endpoint failure counts so monitoring can distinguish
     "ticker has no insider trades" (valid 0) from "insider route is down".

Auth: FMP_API_KEY env var (Ultimate plan).
Cache: file-based under .cache/fmp/<bucket>/<ticker>.json with per-bucket TTL.
Rate: FMP_MAX_RPS env var (Ultimate cap is 50 req/s). Default 50.
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

import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from regime_trader.utils.io import save_json_atomic

log = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com"
_STABLE = f"{_FMP_BASE}/stable"
_TIMEOUT = 15
_DEFAULT_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "fmp"
_DEFAULT_MAX_RPS = 50.0

# Endpoints confirmed HTTP 404 on stable/ routes.
# _get() skips these entirely — no HTTP call, no failure count, no health-report alarm.
# Verified against FMP Ultimate via scripts/fmp_endpoint_probe.py.
#
# NOTE: "insider-trading" (bare) also 404s, but is NOT quarantined here because
# it is never called by this client. The actual scoring path is "insider-trading/search"
# which returns HTTP 200 (confirmed 2026-06-09). Do not conflate the two.
_DEAD_ENDPOINTS: frozenset[str] = frozenset({
    "upgrades-downgrades",  # HTTP 404 — renamed to grades-consensus on stable/ (confirmed 2026-06-09)
})

# Per-bucket cache TTL (seconds).
_TTL: Dict[str, int] = {
    "congress":     12 * 3600,   # stub — nothing fetched, but preserved for compatibility
    "insider":      12 * 3600,
    "news":          8 * 3600,   # pipeline runs 3× daily (8h interval) — 2h caused cache miss every run
    "quote":             5 * 60,
    "ratings":       6 * 3600,
    "key_metrics":  24 * 3600,
    "ratios":       24 * 3600,
    "f13":          24 * 3600,
    "transcript":   24 * 3600,
    "profile":      24 * 3600,
}


def fmp_prices_to_arrays(
    rows: List[Dict],
) -> "tuple[list[float], list[float], list[str]]":
    """Convert FMP historical-price-eod/full rows → (close_prices, volumes, dates).

    FMP returns newest-first; this reverses to oldest-first (matches yfinance df order).
    All three lists are aligned by index.

    Returns empty lists if rows is empty.
    """
    if not rows:
        return [], [], []
    ordered = list(reversed(rows))   # oldest-first
    closes = [float(r.get("close",  0) or 0) for r in ordered]
    volumes = [float(r.get("volume", 0) or 0) for r in ordered]
    dates = [str(r.get("date", "")) for r in ordered]
    return closes, volumes, dates


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
        # type: ignore[call-arg]
        retry = Retry(**retry_kwargs, method_whitelist={"GET"})
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FMPClient:
    """Unified FMP Ultimate client — stable/ routes, full premium surface.

    Args:
        api_key:    FMP API key. Defaults to FMP_API_KEY env var.
        cache_root: Directory for file-based TTL cache. Defaults to .cache/fmp/.
    """

    # Class-level rate-limiter — shared across all instances in the same process.
    # Prevents 8 scoring threads from collectively exceeding FMP_MAX_RPS.
    # Cross-runner 429s (US + INTL jobs on separate GitHub Actions VMs) are
    # handled by exponential backoff inside _get(); a process-level lock cannot
    # coordinate across VMs.
    _rate_lock: threading.Lock = threading.Lock()
    _rate_last_call: float = 0.0

    # Class-level flag: congress route probe fires once per process lifetime.
    # Prevents per-ticker spam; can be reset in tests via FMPClient._fmp_congress_probe_done = False.
    _fmp_congress_probe_done: bool = False

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_root: Optional[Path] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv(
            "FMP_API_KEY", "")
        self._cache_root = Path(
            cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        self._session = _make_session() if self._api_key else None
        max_rps = float(os.getenv("FMP_MAX_RPS", _DEFAULT_MAX_RPS))
        self._min_delay = 1.0 / max(max_rps, 0.001)
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
            save_json_atomic(p, value)   # atomic tmp → rename; safe under 8 concurrent threads
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

        if path.lstrip("/") in _DEAD_ENDPOINTS:
            log.debug("_get: skipping quarantined endpoint %r", path)
            return None

        # Intra-process rate gate — class-level lock shared across all threads.
        with FMPClient._rate_lock:
            elapsed = time.monotonic() - FMPClient._rate_last_call
            if elapsed < self._min_delay:
                time.sleep(self._min_delay - elapsed)
            FMPClient._rate_last_call = time.monotonic()
        # HTTP call is intentionally outside the lock to preserve parallelism.

        url = f"{_STABLE}/{path.lstrip('/')}"
        p = dict(params or {})
        p["apikey"] = self._api_key
        self.endpoint_calls[path] += 1

        _MAX_429_RETRIES = 4
        resp = None
        for _attempt in range(_MAX_429_RETRIES):
            try:
                resp = self._session.get(url, params=p, timeout=_TIMEOUT)
            except Exception as exc:
                log.warning("FMP GET %s network error: %s", path, exc)
                return None

            if resp.status_code != 429:
                break   # success or structural error — exit retry loop

            # 429: read Retry-After from response then back off exponentially.
            try:
                _ra = float(resp.json().get("retry_after", 0) or 0)
            except Exception:
                _ra = 0.0
            if not _ra:
                _ra = float(resp.headers.get("Retry-After", 0) or 0)
            if not _ra:
                _ra = float(2 ** _attempt)   # 1s, 2s, 4s, 8s
            log.warning(
                "FMP 429 on %s (attempt %d/%d) — backing off %.1fs",
                path, _attempt + 1, _MAX_429_RETRIES, _ra,
            )
            time.sleep(_ra)
        else:
            log.error("FMP GET %s: exhausted %d retries on 429", path, _MAX_429_RETRIES)
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

    # ── Historical prices (replaces all yfinance.download calls) ──────────────

    def get_historical_prices(self, ticker: str, limit: int = 280) -> List[Dict]:
        """Daily OHLCV from stable/historical-price-eod/full.

        Returns list of dicts sorted newest-first:
            [{symbol, date, open, high, low, close, volume, change, changePercent, vwap}, ...]

        Works for US, EU (SAP.DE), Asia (7203.T), indices (^VIX, ^TNX),
        ETFs (SPY, GLD), and futures (CL=F, GC=F) — confirmed in Phase-0 tests.

        Args:
            limit: Number of trading days to return. 280 ≈ 13 months (12-1m window).
        """
        if not self._api_key:
            return []
        cached = self._cache_read("quote", f"hist_{ticker}_{limit}")
        if cached is not None:
            return cached
        data = self._get("historical-price-eod/full",
                         {"symbol": ticker, "limit": limit},
                         bucket="quote") or []
        result = data if isinstance(data, list) else []
        if result:
            self._cache_write("quote", f"hist_{ticker}_{limit}", result)
        return result

    # ── Congress ───────────────────────────────────────────────────────────────

    def get_congress_trades(self, ticker: str, lookback_days: int = 180) -> Dict:
        """Congressional trading data from public S3 Stock Watcher feeds.

        FMP stable/ senate-trading and house-trading routes return HTTP 404 —
        FMP has not migrated these endpoints from the deprecated v4 paths to
        stable/ as of 2026-05-30. Contact FMP support to request migration.

        Fallback: fetches directly from the free public S3 feeds maintained
        by House/Senate Stock Watcher (no API key required, same source that
        run_pipeline.fetch_congress_buys uses as primary).

        Returns dict matching the 7-factor pipeline contract:
            {purchases, sales, total, net, recency_days, representatives}
        Returns {} when no trades found in the lookback window.
        """
        from datetime import timedelta as _td
        import requests as _req

        cutoff = (datetime.now(timezone.utc) -
                  _td(days=lookback_days)).date().isoformat()
        purchases = sales = total = 0
        recency_days = 9999
        reps: set[str] = set()
        now_date = datetime.now(timezone.utc).date()

        _HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
        _SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"

        cached = self._cache_read("congress", ticker)
        if cached is not None:
            return cached

        # Probe FMP congress routes once per process lifetime.
        # These return HTTP 404 (not in current plan per Phase-0 smoke test).
        # We track the failure in endpoint_failures so fmp_health.json is
        # accurate — the S3 fallback below is the actual data source.
        # Class-level flag avoids per-ticker API calls (probe only needed once).
        if self._api_key and not FMPClient._fmp_congress_probe_done:
            FMPClient._fmp_congress_probe_done = True
            try:
                self._get(
                    "senate-trading",
                    {"symbol": ticker, "page": 0, "limit": 1},
                    bucket="congress",
                )
                log.info("FMP senate-trading is LIVE — route may have been migrated")
            except FMPEndpointError as exc:
                if exc.status == 404:
                    log.debug(
                        "FMP senate-trading: HTTP 404 (not in plan). "
                        "S3 Stock Watcher fallback active. "
                        "Failure recorded in health_report()."
                    )
            except Exception as exc:
                log.debug("FMP congress probe failed (non-4xx): %s", exc)

        for url, name_key in [(_SENATE_URL, "senator"), (_HOUSE_URL, "representative")]:
            try:
                resp = _req.get(url, timeout=30)
                if resp.status_code == 403:
                    log.warning(
                        "S3 congress feed %s returned 403 — bucket restricted", name_key)
                    continue
                resp.raise_for_status()
                for rec in resp.json():
                    ticker_field = str(rec.get("ticker", "")
                                       or "").upper().strip()
                    if ticker_field != ticker.upper():
                        continue
                    disclosure = (rec.get("disclosure_date")
                                  or rec.get("transaction_date") or "")
                    if not disclosure or disclosure[:10] < cutoff:
                        continue
                    tx_type = (rec.get("type") or rec.get(
                        "transaction_type") or "").lower()
                    if "purchase" in tx_type or "buy" in tx_type:
                        purchases += 1
                    elif "sale" in tx_type or "sold" in tx_type or "sell" in tx_type:
                        sales += 1
                    else:
                        continue
                    total += 1
                    rep = str(rec.get(name_key) or rec.get(
                        "name") or "").strip()
                    if rep:
                        reps.add(rep)
                    try:
                        from datetime import date as _date
                        d = _date.fromisoformat(disclosure[:10])
                        recency_days = min(recency_days, (now_date - d).days)
                    except Exception:
                        pass
            except Exception as exc:
                log.debug("S3 congress feed %s failed: %s", name_key, exc)

        if total == 0:
            self._cache_write("congress", ticker, {})
            return {}

        result = {
            "purchases":      purchases,
            "sales":          sales,
            "total":          total,
            "net":            purchases - sales,
            "recency_days":   recency_days if recency_days < 9999 else None,
            "representatives": sorted(reps),
        }
        self._cache_write("congress", ticker, result)
        return result

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

        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=lookback_days)).date().isoformat()
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])
        data: list = []
        for sym in symbols_to_try:
            data = self._get("insider-trading/search",
                             {"symbol": sym, "page": 0, "limit": 500},
                             bucket="insider") or []
            if data:
                break

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
        Falls back to base symbol for EU/Asia tickers.
        """
        if not self._api_key:
            return {"P": [], "S": []}
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])
        data: List[Dict] = []
        for sym in symbols_to_try:
            data = self._get("insider-trading/search",
                             {"symbol": sym, "page": 0, "limit": 500},
                             bucket="insider") or []
            if data:
                break
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=lookback_days)).date().isoformat()
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
        """Return raw news articles for sentiment+buzz scoring (cached 2h).

        Falls back to base symbol (strips exchange suffix) when dotted ticker
        returns no results — FMP news API may not index EU/Asia suffixed tickers.
        """
        if not self._api_key:
            return []
        cached = self._cache_read("news", ticker)
        if cached is not None:
            return cached
        data = self._get("news/stock", {"symbols": ticker, "limit": 50},
                         bucket="news") or []
        result: List[Dict] = data if isinstance(data, list) else []
        if not result and "." in ticker:
            base = ticker.split(".")[0]
            cached_base = self._cache_read("news", base)
            if cached_base is not None:
                return cached_base
            data2 = self._get("news/stock", {"symbols": base, "limit": 50},
                              bucket="news") or []
            result = data2 if isinstance(data2, list) else []
            if result:
                self._cache_write("news", base, result)
        if result:
            self._cache_write("news", ticker, result)
        return result

    def get_earnings_surprise(self, ticker: str) -> Tuple[Optional[float], int]:
        """Return (surprise_pct, days_since) for the most recent quarter.

        Post-Earnings Announcement Drift (PEAD): Bernard & Thomas (1989, JAE)
        showed that standardized unexpected earnings (SUE) predict returns for
        60–90 days post-announcement — the most robust anomaly in event studies.

        surprise_pct = (epsActual - epsEstimated) / abs(epsEstimated)
        days_since   = calendar days from the announcement date to today

        Source: stable/ "earnings" (the legacy "earnings-surprises" route is
        HTTP 404 on stable/). The earnings calendar mixes future scheduled
        quarters (epsActual=None) with past reports, newest-first; the most
        recent PAST report with both actual and estimated EPS is used.

        Returns (None, 0) gracefully on any error, empty response, or zero estimate
        (avoids division-by-zero on pre-revenue companies).

        Uses the "news" TTL bucket — earnings surprise data changes at most
        once per quarter so the news TTL is conservative and keeps the cache coherent.
        """
        if not self._api_key:
            return None, 0

        cache_key = f"eps_surprise_{ticker}"
        cached = self._cache_read("news", cache_key)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        try:
            # limit=8: covers up to ~4 future scheduled quarters plus >=4 past reports.
            data = self._get(
                "earnings",
                {"symbol": ticker, "limit": 8},
                bucket="news",
            ) or []
            if not isinstance(data, list) or not data:
                self._cache_write("news", cache_key, [None, 0])
                return None, 0

            from datetime import date as _date
            today = datetime.now(timezone.utc).date()

            for row in data:
                date_str = str(row.get("date") or "")[:10]
                try:
                    announced = _date.fromisoformat(date_str)
                except ValueError:
                    continue
                if announced > today:
                    continue  # scheduled future quarter — no actuals yet
                actual = row.get("epsActual")
                estimate = row.get("epsEstimated")
                if actual is None or estimate is None:
                    continue
                actual = float(actual)
                estimate = float(estimate)

                # Guard: zero or near-zero estimate → undefined surprise % (pre-revenue)
                if abs(estimate) < 1e-6:
                    break

                surprise_pct = (actual - estimate) / abs(estimate)
                days_since = max(0, (today - announced).days)
                result = (round(surprise_pct, 6), days_since)
                self._cache_write("news", cache_key, list(result))
                return result

            self._cache_write("news", cache_key, [None, 0])
            return None, 0

        except FMPEndpointError:
            # Structural failure already logged by _get(); propagate to health_report
            return None, 0
        except Exception as exc:
            log.debug("get_earnings_surprise %s failed: %s", ticker, exc)
            return None, 0

    def get_analyst_estimate_revision(self, ticker: str) -> Tuple[Optional[float], int]:
        """Return (revision_pct, n_analysts) measuring EPS estimate revision momentum.

        Analyst estimate revision momentum is a core quant factor used by AQR,
        Two Sigma, and most systematic equity funds. The intuition is that analysts
        revising EPS estimates upward signal an improving fundamental view that is
        not yet fully reflected in price — orthogonal to price momentum (Jegadeesh-
        Titman 1993) which captures past returns. Academic grounding:

          Chan, Jegadeesh & Lakonishok (1996, JF): "Momentum Strategies" —
          estimate revisions predict future abnormal returns independently of
          past price performance.

        revision_pct = (estimates[0].estimatedEpsAvg - estimates[2].estimatedEpsAvg)
                       / abs(estimates[2].estimatedEpsAvg)

        estimates[0] = most recent quarter, estimates[2] = ~3 quarters ago.
        FMP returns newest-first so index 0 is the freshest estimate.

        n_analysts is taken from estimates[0].numberAnalystEstimatedEps and used
        by the scorer as a coverage weight (thin coverage → low confidence).

        Returns (None, 0) when:
          - No API key
          - Fewer than 3 estimates available (can't compute a revision)
          - Base estimate is zero or near-zero (division guard)
          - Any network / parse error

        Cache bucket: "ratings" (6h TTL) — analyst estimates change slowly, at
        most once per quarter, so 6h is conservative relative to the signal horizon.
        """
        if not self._api_key:
            return None, 0

        cache_key = f"eps_revision_{ticker}"
        cached = self._cache_read("ratings", cache_key)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        _null = [None, 0]
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])

        for sym in symbols_to_try:
            try:
                data = self._get(
                    "analyst-estimates",
                    {"symbol": sym, "period": "quarter", "limit": 4},
                    bucket="ratings",
                ) or []
                if not isinstance(data, list) or len(data) < 3:
                    continue  # try next symbol

                recent = data[0]
                base_est = data[2]

                recent_eps = recent.get("estimatedEpsAvg")
                base_eps = base_est.get("estimatedEpsAvg")

                if recent_eps is None or base_eps is None:
                    continue

                recent_eps = float(recent_eps)
                base_eps = float(base_eps)

                if abs(base_eps) < 1e-6:
                    continue

                revision_pct = (recent_eps - base_eps) / abs(base_eps)
                n_analysts = int(recent.get("numberAnalystEstimatedEps") or 0)
                result = [round(revision_pct, 6), n_analysts]
                self._cache_write("ratings", cache_key, result)
                return tuple(result)  # type: ignore[return-value]

            except FMPEndpointError:
                self._cache_write("ratings", cache_key, _null)
                return None, 0
            except Exception as exc:
                log.debug("get_analyst_estimate_revision %s (%s) failed: %s", ticker, sym, exc)
                continue

        self._cache_write("ratings", cache_key, _null)
        return None, 0

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
        data = self._get("grades-consensus",
                         {"symbol": ticker}, bucket="ratings") or []
        result = data[0] if isinstance(data, list) and data else (data or {})
        self._cache_write("ratings", ticker, result)
        return result

    def get_recent_upgrades_downgrades(self, ticker: str, lookback_days: int = 7) -> Dict:
        """Fetch recent upgrades/downgrades within lookback_days.

        Returns a dict with keys: action, from_grade, to_grade, analyst_firm,
        days_ago, score_delta. Returns {'action': 'none'} on error or no records.
        """
        if not self._api_key:
            return {"action": "none"}
        cache_key = f"upgrades_{ticker}_{lookback_days}d"
        cached = self._cache_read("ratings", cache_key)
        if cached is not None:
            return cached

        try:
            data = self._get("upgrades-downgrades", {"symbol": ticker, "page": 0}, bucket="ratings") or []
        except FMPEndpointError:
            return {"action": "none"}
        except Exception:
            return {"action": "none"}

        if not isinstance(data, list) or not data:
            return {"action": "none"}

        # Grade score map
        from datetime import date as _dt_date
        _GRADE_SCORE = {
            "strongbuy": 1.0, "buy": 0.75, "outperform": 0.70, "overweight": 0.70,
            "hold": 0.50, "neutral": 0.50, "underperform": 0.25, "sell": 0.10,
            "underweight": 0.10, "strongsell": 0.0,
        }

        best_record = None
        best_days = None
        best_action = None

        for rec in data:
            # date field may be 'publishedDate' or 'date'
            raw_date = str(rec.get("publishedDate") or rec.get("date") or "")[:10]
            try:
                d = _dt_date.fromisoformat(raw_date)
            except Exception:
                continue
            days_ago = (datetime.now(timezone.utc).date() - d).days
            if days_ago > lookback_days:
                continue
            action_raw = str(rec.get("action") or "").lower()
            if "upgrade" in action_raw:
                action = "upgrade"
            elif "downgrade" in action_raw:
                action = "downgrade"
            elif "initiat" in action_raw or "cover" in action_raw:
                action = "initiate"
            else:
                continue

            if best_record is None or days_ago < best_days:
                best_record = rec
                best_days = days_ago
                best_action = action

        if not best_record:
            return {"action": "none"}

        from_grade = best_record.get("fromGrade") or best_record.get("from") or None
        to_grade = best_record.get("toGrade") or best_record.get("to") or None
        firm = best_record.get("analystFirm") or best_record.get("firm") or None

        from_score = _GRADE_SCORE.get(str(from_grade).lower(), None) if from_grade else None
        to_score = _GRADE_SCORE.get(str(to_grade).lower(), None) if to_grade else None
        score_delta = None
        if to_score is not None and from_score is not None:
            score_delta = to_score - from_score

        result = {
            "action": best_action or "none",
            "from_grade": from_grade,
            "to_grade": to_grade,
            "analyst_firm": firm,
            "days_ago": int(best_days) if best_days is not None else None,
            "score_delta": float(score_delta) if score_delta is not None else None,
        }

        try:
            self._cache_write("ratings", cache_key, result)
        except Exception:
            pass
        return result

    def get_ratios_ttm(self, ticker: str) -> Dict:
        """TTM financial ratios (stable/ratios-ttm). Falls back to base symbol for EU/Asia."""
        if not self._api_key:
            return {}
        cached = self._cache_read("ratios", ticker)
        if cached is not None:
            return cached
        data = self._get(
            "ratios-ttm", {"symbol": ticker}, bucket="ratios") or []
        result = data[0] if isinstance(data, list) and data else {}
        if not result and "." in ticker:
            base = ticker.split(".")[0]
            cached_base = self._cache_read("ratios", base)
            if cached_base is not None:
                return cached_base
            data2 = self._get("ratios-ttm", {"symbol": base}, bucket="ratios") or []
            result = data2[0] if isinstance(data2, list) and data2 else {}
            if result:
                self._cache_write("ratios", base, result)
        if result:
            self._cache_write("ratios", ticker, result)
        return result

    def get_enterprise_value(self, ticker: str) -> Optional[float]:
        """Most recent enterprise value in USD from stable/enterprise-values.

        Falls back to base symbol for dotted EU/Asia tickers (e.g., ASML.AS → ASML).
        Returns None when FMP has no coverage (not 0.0 — absence is distinct from zero EV).

        Reference: Damodaran (2006) — FCF Yield denominator.
        """
        if not self._api_key:
            return None
        cached = self._cache_read("ev", ticker)
        if cached is not None:
            return float(cached) if cached else None

        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])

        for sym in symbols_to_try:
            data = self._get("enterprise-values", {"symbol": sym, "limit": 1},
                             bucket="key_metrics") or []
            if data and isinstance(data, list):
                ev = data[0].get("enterpriseValue")
                if ev is not None:
                    ev_float = float(ev) or None
                    self._cache_write("ev", ticker, ev_float)
                    return ev_float
        self._cache_write("ev", ticker, None)
        return None

    def get_quality_score(self, ticker: str) -> tuple[float, int]:
        """Piotroski F-score quality gate from cached ratios-ttm data.

        Calls get_ratios_ttm(ticker) — already cached in "ratios" bucket (24h TTL).
        Zero additional API calls.

        Returns (score, raw_count) where score is in [0, 1] and raw_count is the
        integer F-score (0–8). The raw count is used by _piotroski_gate_multiplier
        to apply the suppress/discount gate independently of the normalised score.

        Dead signal is (0.0, 0) — NOT Optional. This differs from get_upside_to_target
        (which returns None for missing analyst coverage) because quality data is
        universally available for any listed company. A missing ratios response means
        a broken endpoint, not "no quality data for this ticker."

        Returns (0.0, 0) on exception or when get_ratios_ttm() returns empty dict.

        References: Piotroski (2000) JAR; Novy-Marx (2013) JFE.
        """
        if not self._api_key:
            return 0.0, 0
        try:
            from regime_trader.scoring.momentum_signals import score_quality_piotroski  # noqa: PLC0415
            ratios = self.get_ratios_ttm(ticker)
            score, raw_count = score_quality_piotroski(ratios)
            return score, raw_count
        except Exception as exc:
            log.debug("get_quality_score %s failed: %s", ticker, exc)
            return 0.0, 0

    def get_institutional_ownership(self, ticker: str) -> Dict:
        """13F institutional holdings summary.

        Uses stable/institutional-ownership/symbol-positions-summary with
        year + quarter params (required — returns HTTP 400 without them).
        Fetches the most recently completed quarter automatically.

        Returns aggregate fields: investorsHolding, investorsHoldingChange,
        increasedPositions, reducedPositions, newPositions, closedPositions,
        numberOf13FsharesChange, ownershipPercent, ownershipPercentChange.
        Returns {} if no data or key absent.
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("f13", ticker)
        if cached is not None:
            return cached

        # Determine most recently completed quarter (13F lags ~45 days)
        now = datetime.now(timezone.utc)
        # Back off 45 days to ensure the quarter has been filed
        as_of = now.date() - timedelta(days=45)
        year = as_of.year
        quarter = (as_of.month - 1) // 3 + 1

        data = self._get(
            "institutional-ownership/symbol-positions-summary",
            {"symbol": ticker, "year": year,
                "quarter": quarter, "page": 0, "limit": 1},
            bucket="f13",
        ) or []
        result = data[0] if isinstance(data, list) and data else {}
        if result:
            self._cache_write("f13", ticker, result)
        return result

    def get_price_target_consensus(self, ticker: str) -> Dict:
        """Price target consensus (stable/price-target-consensus).

        Falls back to base symbol for EU/Asia tickers (e.g. ASML.AS → ASML).
        """
        if not self._api_key:
            return {}
        data = self._get("price-target-consensus", {"symbol": ticker},
                         bucket="ratings") or []
        result = data[0] if isinstance(data, list) and data else {}
        if not result and "." in ticker:
            base = ticker.split(".")[0]
            data2 = self._get("price-target-consensus", {"symbol": base},
                              bucket="ratings") or []
            result = data2[0] if isinstance(data2, list) and data2 else {}
        return result

    def get_upside_to_target(self, ticker: str, max_age_days: int = 90) -> Optional[float]:
        """Analyst consensus price target upside score in [0, 1], or None.

        Computes score_price_target_upside(targetConsensus, currentPrice)
        using two already-cached FMP calls:
          - get_price_target_consensus() → stable/price-target-consensus (ratings bucket, 6h TTL)
          - get_quote()                  → stable/quote (quote bucket, 5min TTL)

        Staleness check: if the consensus target is older than max_age_days (default 90),
        returns None rather than a misleading score based on a stale target.

        Returns None when:
          - No API key
          - targetConsensus or price is missing, zero, or non-numeric
          - Target is older than max_age_days (stale — treated as dead signal)
          - Either delegated call raises an exception

        None → caller converts to 0.0 via `or 0.0` → dead signal penalized
        in cross-sectional normalization. Distinct from 0.50 (at-target, valid data).
        """
        if not self._api_key:
            return None
        try:
            from regime_trader.scoring.momentum_signals import score_price_target_upside  # noqa: PLC0415
            from datetime import date as _date  # noqa: PLC0415
            target_data = self.get_price_target_consensus(ticker)
            quote_data = self.get_quote(ticker)
            target = target_data.get("targetConsensus")
            price = quote_data.get("price")
            if not target or not price:
                return None

            target_date_str = (
                target_data.get("targetConsensusDate")
                or target_data.get("lastUpdated")
                or ""
            )
            if target_date_str:
                try:
                    target_date = _date.fromisoformat(str(target_date_str)[:10])
                    age_days = (datetime.now(timezone.utc).date() - target_date).days
                    if age_days > max_age_days:
                        log.debug(
                            "get_upside_to_target %s: target is %dd old (> %dd threshold) — "
                            "returning None (stale, treated as dead signal)",
                            ticker, age_days, max_age_days,
                        )
                        return None
                except Exception:
                    pass  # unparseable date — proceed without staleness filter

            return score_price_target_upside(float(target), float(price))
        except Exception as exc:
            log.debug("get_upside_to_target %s failed: %s", ticker, exc)
            return None

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
        data = self._get("commitment-of-traders-report",
                         {}, bucket="key_metrics") or []
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

    def get_earnings_transcript(self, ticker: str, max_chars: int = 3000) -> Optional[str]:
        """Executive remarks from the most recent earnings call.

        Fetches stable/earning-call-transcript-latest (limit=1).
        Returns content[:max_chars] on success; None on any failure.

        max_chars (default 3000) is intentionally larger than build_prompt's
        transcript_max_chars (default 2000) — the delta sits in memory and is
        discarded. This avoids a second network call if the prompt budget changes.

        Cache bucket: "transcript" (24h TTL — transcripts don't change after
        publication). Soft-fail: FMPEndpointError and network exceptions return
        None; the transcript is additive enrichment, not a scored factor.
        """
        if not self._api_key:
            return None
        cached = self._cache_read("transcript", ticker)
        if cached is not None:
            return cached
        try:
            data = self._get(
                "earning-call-transcript-latest",
                {"symbol": ticker, "limit": 1},
                bucket="transcript",
            ) or []
            if not isinstance(data, list) or not data:
                return None
            content = data[0].get("content")
            if not content:
                # FMP returned a record but without transcript text yet.
                # Cache empty sentinel so we don't re-fetch within the 24h TTL.
                self._cache_write("transcript", ticker, "")
                return None
            result = content[:max_chars]
            self._cache_write("transcript", ticker, result)
            return result
        except FMPEndpointError:
            return None
        except Exception as exc:
            log.debug("get_earnings_transcript %s failed: %s", ticker, exc)
            return None

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
            "quarantined_endpoints": sorted(_DEAD_ENDPOINTS),
        }
