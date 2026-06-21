"""src/services/fmp_client.py
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
  upgrades-downgrades-consensus-bulk ratios-ttm-bulk

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
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.io import save_json_atomic

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
    "revenue-estimates",    # HTTP 404 on stable/ for all tickers (confirmed 2026-06-16); revenue_revision factor degrades to no_coverage
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
    "screener":     12 * 3600,   # universe composition moves slowly intra-day
    "dcf":          24 * 3600,   # candidate factor — model output, daily refresh
    "sector_pe":    24 * 3600,   # candidate factor — sector snapshot, daily refresh
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
    # backoff_jitter (urllib3 ≥ 2.0) decorrelates retry timing across the
    # US + INTL runners so a shared 429 wave does not resync into a thundering
    # herd. Degrade gracefully on older urllib3 (no jitter / method_whitelist).
    try:
        retry = Retry(**retry_kwargs, allowed_methods={"GET"}, backoff_jitter=1.0)
    except TypeError:
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

class FMPCore:
    """Shared HTTP / cache / rate-limit / circuit-breaker / telemetry core.

    Behaviour-preserving extraction (comparison plan, Track C): all endpoint
    methods live in per-category mixins composed onto FMPClient; this base
    holds the single shared rate limiter, file TTL cache, circuit breaker and
    telemetry. Class-level rate state is shared across all instances exactly as
    before — the only concrete subclass is FMPClient, so `type(self)` resolves
    to it everywhere the old code wrote `FMPClient.`.
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

    # Dynamic circuit-breaker threshold: consecutive HTTP 404s on one endpoint
    # before it is quarantined for the rest of the run (see _get / _record_404).
    _BREAKER_THRESHOLD: int = 3


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
        # Dynamic circuit breaker — instance-level state shared across the
        # scoring thread pool (one client per run). An endpoint that 404s
        # repeatedly is dead for the whole run (route pulled from the plan),
        # not sparse for one ticker; we stop hammering it after a short streak.
        self._breaker_lock = threading.Lock()
        self._runtime_dead: set[str] = set()
        self._consecutive_404: Dict[str, int] = defaultdict(int)
        # Telemetry (WS4): per-endpoint latency + per-bucket cache hit/miss.
        # Pure counters — no trading math (the client stays an API wrapper).
        self.endpoint_latency_ms_sum: Dict[str, float] = defaultdict(float)
        self.endpoint_latency_ms_max: Dict[str, float] = defaultdict(float)
        self.cache_hits: Dict[str, int] = defaultdict(int)
        self.cache_misses: Dict[str, int] = defaultdict(int)

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
                self.cache_misses[bucket] += 1
                return None
            if time.time() - p.stat().st_mtime > ttl:
                self.cache_misses[bucket] += 1   # expired — refetch
                return None
            self.cache_hits[bucket] += 1
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            self.cache_misses[bucket] += 1
            return None

    def _cache_write(self, bucket: str, key: str, value: Any) -> None:
        p = self._cache_path(bucket, key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            save_json_atomic(p, value)   # atomic tmp → rename; safe under 8 concurrent threads
        except Exception as exc:
            log.debug("fmp cache write failed %s/%s: %s", bucket, key, exc)

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

        # Dynamic circuit breaker: this endpoint already tripped (3+ consecutive
        # 404s) earlier this run. Fail structurally without another round-trip —
        # callers' existing `except FMPEndpointError` branches fire identically
        # to a live 404, minus the wasted call and rate-gate wait.
        if path in self._runtime_dead:
            raise FMPEndpointError(path, 404)

        # Intra-process rate gate — class-level lock shared across all threads.
        with type(self)._rate_lock:
            elapsed = time.monotonic() - type(self)._rate_last_call
            if elapsed < self._min_delay:
                time.sleep(self._min_delay - elapsed)
            type(self)._rate_last_call = time.monotonic()
        # HTTP call is intentionally outside the lock to preserve parallelism.

        url = f"{_STABLE}/{path.lstrip('/')}"
        p = dict(params or {})
        p["apikey"] = self._api_key
        self.endpoint_calls[path] += 1

        _MAX_429_RETRIES = 4
        resp = None
        _last_dt_ms = 0.0
        for _attempt in range(_MAX_429_RETRIES):
            _t0 = time.monotonic()
            try:
                resp = self._session.get(url, params=p, timeout=_TIMEOUT)
            except Exception as exc:
                log.warning("FMP GET %s network error: %s", path, exc)
                return None
            _last_dt_ms = (time.monotonic() - _t0) * 1000.0

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
                # Equal jitter (AWS "Exponential Backoff And Jitter"): half a
                # fixed exponential term + half random, so concurrent threads and
                # the separate US/INTL runners de-sync instead of retrying in
                # lockstep. Server-supplied Retry-After (above) still wins.
                _base = float(2 ** _attempt)        # 1s, 2s, 4s, 8s
                _ra = _base / 2.0 + random.uniform(0.0, _base / 2.0)
            log.warning(
                "FMP 429 on %s (attempt %d/%d) — backing off %.1fs",
                path, _attempt + 1, _MAX_429_RETRIES, _ra,
            )
            time.sleep(_ra)
        else:
            log.error("FMP GET %s: exhausted %d retries on 429", path, _MAX_429_RETRIES)
            return None

        # Latency telemetry: wall-clock of the resolving GET (excludes rate-gate
        # wait and 429 backoff). One sample per logical call → avg = sum / calls.
        self.endpoint_latency_ms_sum[path] += _last_dt_ms
        if _last_dt_ms > self.endpoint_latency_ms_max[path]:
            self.endpoint_latency_ms_max[path] = _last_dt_ms

        if resp.status_code in (401, 403, 404):
            self.endpoint_failures[path] += 1
            # 404 is endpoint-dead (breaker target); 401/403 is global auth and
            # is left to the preflight probe (sys.exit) — never breaker-managed.
            if resp.status_code == 404:
                self._record_404(path)
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
        # 200 — endpoint is alive; clear any prior 404 streak (recovered blip).
        if self._consecutive_404.get(path):
            with self._breaker_lock:
                self._consecutive_404.pop(path, None)
        try:
            return resp.json()
        except Exception as exc:
            log.warning("FMP GET %s JSON decode failed: %s", path, exc)
            return None

    def _record_404(self, path: str) -> None:
        """Count a structural 404; quarantine the endpoint at the threshold.

        Logs the trip exactly once. The endpoint stays quarantined for the rest
        of the run — a route pulled from the plan does not come back mid-run.
        """
        with self._breaker_lock:
            self._consecutive_404[path] += 1
            count = self._consecutive_404[path]
            tripped = (
                count >= self._BREAKER_THRESHOLD and path not in self._runtime_dead
            )
            if tripped:
                self._runtime_dead.add(path)
        if tripped:
            log.error(
                "FMP CIRCUIT BREAKER TRIPPED for %s after %d consecutive HTTP 404s "
                "— short-circuiting further calls this run. Factors sourced from "
                "this endpoint are unavailable until the next run.",
                path, count,
            )

    def reset_circuit_breaker(self) -> None:
        """Clear dynamic-breaker state (new-run reset / test hook)."""
        with self._breaker_lock:
            self._runtime_dead.clear()
            self._consecutive_404.clear()

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
            # Endpoints the dynamic circuit breaker killed this run (3+ consecutive
            # 404s) — distinct from the static plan-level quarantine above.
            "runtime_quarantined": sorted(self._runtime_dead),
        }

    def telemetry_snapshot(self) -> Dict[str, Any]:
        """Per-endpoint call/failure/latency + global cache stats for monitoring.

        Pure counters — no trading math (keeps the client an API wrapper). Latency
        is wall-clock around the resolving HTTP GET only (excludes the rate-gate
        wait and 429 backoff). Cache hit/miss is bucket-grained (one bucket backs
        several endpoints), so it is reported as a single global pair under
        ``totals`` rather than per endpoint.

        Shape::

            {
              "endpoints": {path: {calls, failures, latency_ms_avg, latency_ms_max}},
              "totals":    {calls, failures, cache_hits, cache_misses,
                            runtime_quarantined},
            }
        """
        endpoints: Dict[str, Dict[str, Any]] = {}
        for path, calls in self.endpoint_calls.items():
            n = int(calls)
            lat_sum = float(self.endpoint_latency_ms_sum.get(path, 0.0))
            endpoints[path] = {
                "calls":          n,
                "failures":       int(self.endpoint_failures.get(path, 0)),
                "latency_ms_avg": round(lat_sum / n, 2) if n else 0.0,
                "latency_ms_max": round(float(self.endpoint_latency_ms_max.get(path, 0.0)), 2),
            }
        return {
            "endpoints": endpoints,
            "totals": {
                "calls":        sum(int(c) for c in self.endpoint_calls.values()),
                "failures":     sum(int(f) for f in self.endpoint_failures.values()),
                "cache_hits":   sum(int(h) for h in self.cache_hits.values()),
                "cache_misses": sum(int(m) for m in self.cache_misses.values()),
                "runtime_quarantined": sorted(self._runtime_dead),
            },
        }
