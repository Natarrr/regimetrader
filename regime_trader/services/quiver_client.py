"""regime_trader/services/quiver_client.py
Quiver Quantitative API client.

Stiglitz (2001 Nobel) — congressional trading exploits non-public information.
QuiverQuant surfaces this asymmetric signal via structured legislative data.

Endpoints used:
  /beta/live/congresstrading   — house/senate trades (live feed)
  /beta/live/insidertrading/{ticker} — SEC Form 4 structured insider trades
  /beta/live/historical/13f/{ticker} — institutional 13F history
  /beta/live/lobbying/{ticker} — corporate lobbying spend
  /beta/live/govcontracts/{ticker}   — government contract awards

Auth: Bearer token via QUIVER_API_KEY env var (Hobbyist plan).
Cache: file-based under .cache/quiver/, TTL-checked at read time.
CI isolation: calls are blocked by conftest when CI=1.

Public API:
  QuiverClient(api_key, cache_root)
  .get_politician_trades(lookback_days) -> List[dict]
  .get_insider_trades(ticker)           -> List[dict]
  .get_13f_summary(ticker)             -> List[dict]
  .get_lobbying(ticker)                -> List[dict]
  .get_gov_contracts(ticker)           -> List[dict]
  .congress_by_ticker(lookback_days)   -> dict[ticker, aggregated]

Usage:
  from regime_trader.services.quiver_client import QuiverClient
  client = QuiverClient()
  trades = client.get_politician_trades(lookback_days=90)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_BASE_URL = "https://api.quiverquant.com"
_TIMEOUT = 30
_MAX_RETRIES = 3
_BACKOFF = 1.0

_DEFAULT_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "quiver"

_TTL: Dict[str, int] = {
    "congresstrading": 6 * 3600,    # 6 h — updates ~daily
    "insidertrading":  6 * 3600,
    "13f":            24 * 3600,    # 24 h — quarterly filings
    "lobbying":       24 * 3600,
    "govcontracts":   24 * 3600,
}

# HTTP status codes that mean "plan doesn't include this endpoint" — don't retry,
# don't spam per-ticker warnings; log once and short-circuit all remaining calls.
_PLAN_RESTRICTED_STATUSES = frozenset({403})

_INVALID_TICKERS = frozenset({"N/A", "--", "", "NONE", "NO TICKER"})


def _make_session(api_key: str) -> requests.Session:
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
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


class QuiverClient:
    """Thin wrapper around the Quiver Quantitative REST API.

    Args:
        api_key:    Bearer token. Defaults to QUIVER_API_KEY env var.
        cache_root: Directory for file-based TTL cache. Defaults to .cache/quiver/.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_root: Optional[Path] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("QUIVER_API_KEY", "")
        self._cache_root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        self._session: Optional[requests.Session] = (
            _make_session(self._api_key) if self._api_key else None
        )
        # Set to True after the first 403 on an insider endpoint so all
        # subsequent per-ticker calls short-circuit without network I/O.
        # _insider_lock guards the flag across ThreadPoolExecutor workers.
        self._insider_plan_restricted: bool = False
        self._insider_lock: threading.Lock = threading.Lock()

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_path(self, bucket: str, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace("\\", "_")
        return self._cache_root / bucket / f"{safe}.json"

    def _cache_read(self, bucket: str, key: str) -> Optional[Any]:
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
            log.debug("quiver cache write failed %s/%s: %s", bucket, key, exc)

    # ── HTTP helper ────────────────────────────────────────────────────────────

    def _get(self, path: str) -> Optional[Any]:
        if not self._api_key or self._session is None:
            return None
        url = f"{_BASE_URL}{path}"
        try:
            resp = self._session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("quiver GET %s failed: %s", path, exc)
            return None

    # ── Public endpoints ───────────────────────────────────────────────────────

    def get_politician_trades(self, lookback_days: int = 180) -> List[dict]:
        """Return recent congressional trading records (all tickers).

        Response fields: Representative, BioGuideID, ReportDate, TransactionDate,
        Ticker, Transaction, Range, House, Amount, Party, TickerType,
        ExcessReturn, PriceChange.

        Cache note: an empty list [] is NOT cached — a previous failed run
        (S3 403, network error) may have written [] to disk.  Treating [] as
        a valid cache hit would silence the live feed indefinitely until the
        6-hour TTL expires.  We only cache non-empty results.
        """
        cached = self._cache_read("congresstrading", "all")
        if cached:   # truthy check: ignores None AND stale empty-list cache
            return cached
        if not self._api_key:
            return []
        data = self._get("/beta/live/congresstrading")
        result: List[dict] = data if isinstance(data, list) else []
        if result:
            self._cache_write("congresstrading", "all", result)
        return result

    def get_insider_trades(self, ticker: str) -> List[dict]:
        """Return SEC Form 4 insider trades for a ticker from Quiver.

        Response fields: Name, Title, Date, Ticker,
        AcquisitionOrDisposition (A/D), Shares, PricePerShare,
        TotalValue, FilingURL.

        Returns [] immediately (without a network call) if a previous call
        already received a 403, meaning the endpoint is not included in the
        current subscription plan.
        """
        # Fast path: another thread already confirmed this endpoint is plan-restricted.
        if self._insider_plan_restricted:
            return []
        cached = self._cache_read("insidertrading", ticker)
        if cached is not None:
            return cached
        if not self._api_key or self._session is None:
            return []
        url = f"{_BASE_URL}/beta/live/insiders"
        try:
            resp = self._session.get(url, params={"ticker": ticker}, timeout=_TIMEOUT)
            if resp.status_code in _PLAN_RESTRICTED_STATUSES:
                with self._insider_lock:
                    if not self._insider_plan_restricted:
                        self._insider_plan_restricted = True
                        log.warning(
                            "Quiver insider endpoint returned %d — not included in "
                            "current subscription plan. "
                            "Falling back to Finnhub/EDGAR for all tickers.",
                            resp.status_code,
                        )
                return []
            resp.raise_for_status()
            result: List[dict] = resp.json() if isinstance(resp.json(), list) else []
        except Exception as exc:
            log.warning("quiver GET /beta/live/insiders?ticker=%s failed: %s", ticker, exc)
            return []
        self._cache_write("insidertrading", ticker, result)
        return result

    def get_13f_summary(self, ticker: str) -> List[dict]:
        """Return institutional 13F history for a ticker.

        Response fields: Date, Shares, Value, Pct, PctChange.
        """
        cached = self._cache_read("13f", ticker)
        if cached is not None:
            return cached
        if not self._api_key:
            return []
        data = self._get(f"/beta/live/historical/13f/{ticker}")
        result: List[dict] = data if isinstance(data, list) else []
        self._cache_write("13f", ticker, result)
        return result

    def get_lobbying(self, ticker: str) -> List[dict]:
        """Return lobbying spend records for a ticker.

        Response fields: Ticker, Amount, Client, Issue, Date.
        """
        cached = self._cache_read("lobbying", ticker)
        if cached is not None:
            return cached
        if not self._api_key:
            return []
        data = self._get(f"/beta/live/lobbying/{ticker}")
        result: List[dict] = data if isinstance(data, list) else []
        self._cache_write("lobbying", ticker, result)
        return result

    def get_gov_contracts(self, ticker: str) -> List[dict]:
        """Return government contract awards for a ticker.

        Response fields: Ticker, Amount, Agency, Description, Date.
        """
        cached = self._cache_read("govcontracts", ticker)
        if cached is not None:
            return cached
        if not self._api_key:
            return []
        data = self._get(f"/beta/live/govcontracts/{ticker}")
        result: List[dict] = data if isinstance(data, list) else []
        self._cache_write("govcontracts", ticker, result)
        return result

    def congress_by_ticker(self, lookback_days: int = 180) -> Dict[str, Dict]:
        """Aggregate politician trades into per-ticker buy/sell/net counts.

        Returns:
            {TICKER: {purchases, sales, total, net, representatives, recency_days}}
        """
        records = self.get_politician_trades(lookback_days=lookback_days)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        by_ticker: Dict[str, Dict] = {}
        for rec in records:
            ticker = rec.get("Ticker", "").strip().upper()
            if not ticker or ticker in _INVALID_TICKERS:
                continue
            if rec.get("TickerType", "ST") != "ST":
                continue
            tx_date = rec.get("TransactionDate", "")
            if tx_date < cutoff:
                continue
            tx_type = rec.get("Transaction", "").lower()
            rep = rec.get("Representative", "Unknown")

            if ticker not in by_ticker:
                by_ticker[ticker] = {
                    "purchases": 0, "sales": 0, "total": 0, "net": 0,
                    "representatives": [], "recency_days": 9999,
                }
            entry = by_ticker[ticker]

            if tx_type == "purchase":
                entry["purchases"] += 1
                entry["net"] += 1
            elif tx_type in ("sale", "sale_full", "sale_partial"):
                entry["sales"] += 1
                entry["net"] -= 1
            entry["total"] += 1

            if rep not in entry["representatives"]:
                entry["representatives"].append(rep)

            try:
                days_ago = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(tx_date).replace(tzinfo=timezone.utc)
                ).days
                entry["recency_days"] = min(entry["recency_days"], days_ago)
            except Exception:
                pass

        return by_ticker


# Module-level singleton — use when no custom cache root needed
default_quiver = QuiverClient()
