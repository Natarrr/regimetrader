"""regime_trader/services/fmp_client.py
FMP API client with strict 200-calls/day budget enforcement.

Fama (2013 Nobel) — reliable, consistently-sourced data is a prerequisite for
valid factor models. The daily quota is a hard constraint: exceeding it causes
HTTP 429s that cascade to data gaps worse than conservative caching.

Design:
  - Daily budget: 200 calls (configurable via FMP_DAILY_BUDGET env).
  - Budget persisted atomically in .cache/fmp/daily_quota.json (resets at UTC midnight).
  - reserve_calls(n) / commit_calls(n) / release_calls(n): two-phase budget.
  - Batch endpoint: get_profiles(tickers) fetches multiple tickers in one call.
  - Fallback: yfinance for price/time-series when quota exhausted.
  - TTLs: profile 24h, screener 6h, insider 12h.

Public API:
    FmpClient.get_profile(ticker)          → dict | None
    FmpClient.get_profiles(tickers)        → Dict[str, dict]
    FmpClient.get_screener(limit)          → List[dict]
    FmpClient.get_insider_buys(limit)      → List[dict]
    FmpClient.budget_remaining()           → int

Usage:
    from regime_trader.services.fmp_client import FmpClient
    client = FmpClient()
    profiles = client.get_profiles(["AAPL", "MSFT", "NVDA"])
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_FMP_STABLE      = "https://financialmodelingprep.com/stable"
_CACHE_ROOT      = Path(__file__).parent.parent.parent / ".cache" / "fmp2"
_QUOTA_PATH      = _CACHE_ROOT / "daily_quota.json"
_DEFAULT_BUDGET  = int(os.getenv("FMP_DAILY_BUDGET", "200"))
_MAX_RETRIES     = 2
_BACKOFF         = 0.8
_TIMEOUT         = 15

_TTL: Dict[str, int] = {
    "profile":    24 * 3600,
    "screener":    6 * 3600,
    "insider":    12 * 3600,
}

# FMP batch profile endpoint supports up to 50 tickers per call
_BATCH_SIZE = 50

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}


# ── Budget state ───────────────────────────────────────────────────────────────

class _BudgetManager:
    """Thread-safe daily budget tracker with atomic persistence.

    Fama (2013): consistent data access requires disciplined quota management —
    exhausting 200 FMP calls before market close leaves the pipeline blind.

    The budget resets at UTC midnight.  Reserved-but-uncommitted calls are
    tracked so parallel callers don't race to exhaust the budget.
    """

    def __init__(self, daily_limit: int, quota_path: Path) -> None:
        self._limit      = daily_limit
        self._path       = quota_path
        self._lock       = threading.Lock()
        self._state      = self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def reserve_calls(self, n: int) -> bool:
        """Reserve n calls (reduce available budget).  Returns False if insufficient."""
        with self._lock:
            self._maybe_reset()
            avail = self._limit - self._state["used"] - self._state["reserved"]
            if n > avail:
                log.warning(
                    "fmp_client: budget insufficient (need %d, available %d)", n, avail
                )
                return False
            self._state["reserved"] += n
            self._save()
            return True

    def commit_calls(self, n: int) -> None:
        """Confirm that n reserved calls were actually made."""
        with self._lock:
            self._state["used"]     += n
            self._state["reserved"] -= min(n, self._state["reserved"])
            self._save()

    def release_calls(self, n: int) -> None:
        """Release n previously reserved calls that were NOT made (e.g., cache hit)."""
        with self._lock:
            self._state["reserved"] -= min(n, self._state["reserved"])
            self._save()

    def remaining(self) -> int:
        """Unencumbered calls left for today."""
        with self._lock:
            self._maybe_reset()
            return max(0, self._limit - self._state["used"] - self._state["reserved"])

    # ── Internal ───────────────────────────────────────────────────────────────

    def _maybe_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._state.get("date") != today:
            self._state = {"date": today, "used": 0, "reserved": 0}
            self._save()

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {"date": "", "used": 0, "reserved": 0}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            # Reset if stale
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if data.get("date") != today:
                return {"date": today, "used": 0, "reserved": 0}
            return data
        except Exception:
            return {"date": "", "used": 0, "reserved": 0}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, indent=2).encode("utf-8")
        fd, tmp = tempfile.mkstemp(
            prefix=".quota.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(category: str, key: str) -> Path:
    safe_key = key.replace("/", "_").replace("\\", "_")[:64]
    return _CACHE_ROOT / category / f"{safe_key}.json"


def _cache_read(category: str, key: str) -> Optional[Any]:
    p = _cache_path(category, key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - data.get("_ts", 0) > _TTL.get(category, 3600):
            return None
        return data.get("payload")
    except Exception:
        return None


def _cache_write(category: str, key: str, payload: Any) -> None:
    p = _cache_path(category, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = json.dumps({"_ts": time.time(), "payload": payload},
                     indent=2, ensure_ascii=False).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(obj)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── HTTP session ───────────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(_HEADERS)
    return s


# ── yfinance fallback ──────────────────────────────────────────────────────────

def _yfinance_profile(ticker: str) -> Optional[Dict[str, Any]]:
    """Fama (2013): when the primary data source is unavailable, a secondary
    source with known limitations is better than no data."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if not info:
            return None
        return {
            "symbol":      ticker,
            "marketCap":   info.get("marketCap"),
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            "source":      "yfinance_fallback",
        }
    except Exception as exc:
        log.debug("yfinance fallback failed for %s: %s", ticker, exc)
        return None


# ── Main client ────────────────────────────────────────────────────────────────

class FmpClient:
    """Fama (2013 Nobel) — FMP client with strict 200-calls/day budget.

    All network calls go through the budget manager.  When the daily budget is
    exhausted, methods fall back to yfinance or cached data silently.

    Args:
        api_key:       FMP API key (default from env FMP_API_KEY).
        daily_budget:  Max API calls per day (default 200).
        cache_root:    Override cache directory (useful in tests).
        quota_path:    Override quota file path (useful in tests).
        session:       Override requests.Session (for mocking in tests).
    """

    def __init__(
        self,
        api_key:      str = "",
        daily_budget: int = _DEFAULT_BUDGET,
        cache_root:   Optional[Path] = None,
        quota_path:   Optional[Path] = None,
        session:      Optional[requests.Session] = None,
    ) -> None:
        self._key        = api_key or os.getenv("FMP_API_KEY", "")
        self._cache_root = Path(cache_root) if cache_root else _CACHE_ROOT
        self._budget     = _BudgetManager(
            daily_limit=daily_budget,
            quota_path=Path(quota_path) if quota_path else _QUOTA_PATH,
        )
        self._session    = session or _build_session()

    # ── Public API ─────────────────────────────────────────────────────────────

    def budget_remaining(self) -> int:
        """Return the number of unencumbered API calls left today."""
        return self._budget.remaining()

    def get_profile(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch a single company profile.

        Fama (2013): single-ticker profile = 1 FMP call. Cache hit = 0 calls.

        Returns:
            Profile dict or None on failure.
        """
        cached = _cache_read("profile", ticker)
        if cached is not None:
            return cached

        if not self._budget.reserve_calls(1):
            log.info("fmp_client quota exhausted, using yfinance for %s", ticker)
            return _yfinance_profile(ticker)

        data = self._get(f"/profile/{ticker}")
        self._budget.commit_calls(1)

        if not data:
            return _yfinance_profile(ticker)

        result = data[0] if isinstance(data, list) else data
        _cache_write("profile", ticker, result)
        return result

    def get_profiles(self, tickers: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Fetch profiles for multiple tickers, batching to minimise API calls.

        Fama (2013): batch endpoint = ceil(N/50) calls instead of N calls.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Dict mapping ticker → profile dict (or None).
        """
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        uncached = []

        for t in tickers:
            hit = _cache_read("profile", t)
            if hit is not None:
                results[t] = hit
            else:
                uncached.append(t)

        # Batch uncached tickers into groups of _BATCH_SIZE
        for i in range(0, len(uncached), _BATCH_SIZE):
            batch = uncached[i : i + _BATCH_SIZE]
            n_calls = 1

            if not self._budget.reserve_calls(n_calls):
                # Fall back to yfinance for entire batch
                for t in batch:
                    results[t] = _yfinance_profile(t)
                continue

            syms  = ",".join(batch)
            data  = self._get(f"/profile/{syms}")
            self._budget.commit_calls(n_calls)

            if not data:
                for t in batch:
                    results[t] = _yfinance_profile(t)
                continue

            by_sym = {item["symbol"]: item for item in data if "symbol" in item}
            for t in batch:
                p = by_sym.get(t)
                results[t] = p
                if p:
                    _cache_write("profile", t, p)

        # Fill any remaining None with yfinance
        for t in tickers:
            if results.get(t) is None:
                results[t] = _yfinance_profile(t)

        return results

    def get_screener(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch a screener result (1 FMP call, cached 6h).

        Returns:
            List of screener row dicts, or [] on failure/quota exhaustion.
        """
        cache_key = f"screener_{limit}"
        cached = _cache_read("screener", cache_key)
        if cached is not None:
            return cached

        if not self._budget.reserve_calls(1):
            return []

        data = self._get(
            "/stock-screener",
            params={"marketCapMoreThan": 50_000_000, "limit": limit},
        )
        self._budget.commit_calls(1)

        result = data if isinstance(data, list) else []
        if result:
            _cache_write("screener", cache_key, result)
        return result

    def get_insider_buys(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent insider purchase transactions (1 FMP call, cached 12h).

        Returns:
            List of insider transaction dicts.
        """
        cache_key = f"insider_buys_{limit}"
        cached = _cache_read("insider", cache_key)
        if cached is not None:
            return cached

        if not self._budget.reserve_calls(1):
            return []

        data = self._get(
            "/insider-trading",
            params={"transactionType": "P-Purchase", "limit": limit},
        )
        self._budget.commit_calls(1)

        result = data if isinstance(data, list) else []
        if result:
            _cache_write("insider", cache_key, result)
        return result

    # ── Internal HTTP ──────────────────────────────────────────────────────────

    def _get(
        self,
        path:   str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a single authenticated GET request to the FMP stable API."""
        if not self._key:
            log.debug("fmp_client: no API key — skipping %s", path)
            return None

        url = f"{_FMP_STABLE}{path}"
        p   = {"apikey": self._key, **(params or {})}
        try:
            resp = self._session.get(url, params=p, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("fmp_client: %s failed — %s", path, exc)
            return None
