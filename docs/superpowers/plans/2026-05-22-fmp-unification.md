# FMP Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all Quiver Quantitative and Finnhub dependencies with a single `FMPClient` service so the 5-factor Markowitz pipeline runs entirely through one `FMP_API_KEY`.

**Architecture:** Create `regime_trader/services/fmp_client.py` (new unified service with per-bucket TTL cache and configurable rate limiter), then surgically replace each Quiver/Finnhub call-site in `run_pipeline.py` one function at a time, then clean up dead code and env vars. Each task leaves the 1085-test suite green before the next starts.

**Tech Stack:** Python 3.11, `requests`, `time.monotonic()` rate limiter, file-based JSON TTL cache, `unittest.mock`, `pytest`.

---

## Context

**Spec:** `docs/superpowers/specs/2026-05-22-fmp-unification-design.md`

**Key files to understand before starting:**
- `regime_trader/services/quiver_client.py` — the client being replaced (read it fully)
- `scripts/run_pipeline.py` lines 1–50 (imports), 275–369 (`fetch_congress_buys` + `_fetch_quiver_congress`), 563–619 (`score_news_finnhub`), 820–889 (`fetch_quiver_insider_all`), 1259–1282 (insider pre-fetch block in `run()`), 1304–1368 (per-ticker scoring block), 1492–1497 (`source_meta` dict)
- `regime_trader/scanners/discovery_scanner.py` lines 850–882 (`_enrich_with_quiver`)
- `tests/test_quiver_client.py` — structural reference for new test file
- `tests/test_congress_fetcher.py` lines 127–155 (`test_s3_403_falls_back_to_quiver`) — needs updating

**5-factor weights (immutable):**
```python
WEIGHTS = {"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12}
```

**FMP Ultimate rate limit:** 50 req/s. Use `FMP_MAX_RPS=20` as default (safe headroom). At 20 rps: 640 calls ÷ 20 = 32 seconds total.

**Congress date rule:** Use `disclosureDate` (not `transactionDate`) for `recency_days`. Alpha decay starts when information is public.

**Insider parse rule:** `total_purchases_usd = sum(float(r.get("securitiesTransacted") or 0) * float(r.get("price") or 0) for r in records if r.get("acquistionOrDisposition") == "A" and float(r.get("price") or 0) > 0 and float(r.get("securitiesTransacted") or 0) > 0)`. Use `limit=500` on the endpoint to cover mega-caps with frequent option grants over 180-day lookback.

**News fallback rule:** If FMP `/api/v3/stock_news` returns `[]` OR `total == 0`, immediately call `_score_news_yfinance(ticker)` from `run_pipeline.py`. The fallback lives in `score_news_fmp()` in `run_pipeline.py`, NOT inside `FMPClient` (keeps the client free of pipeline-layer imports).

---

## File Map

| File | Action |
|---|---|
| `regime_trader/services/fmp_client.py` | **Create** |
| `tests/test_fmp_client.py` | **Create** (replaces `tests/test_quiver_client.py`) |
| `tests/test_quiver_client.py` | **Delete** |
| `regime_trader/scanners/discovery_scanner.py` | **Modify** lines 850–882 |
| `scripts/run_pipeline.py` | **Modify** lines 38, 275–296, 303, 341–362, 563–619, 820–889, 1259–1282, 1304–1368, 1492–1497 |
| `tests/test_congress_fetcher.py` | **Modify** line 127 (test name + patch target) |
| `.github/workflows/nightly_edgar.yml` | **Modify** lines 76–77 |
| `.github/workflows/edgar_3x.yml` | **Modify** lines 93–94 |
| `.github/workflows/canary.yml` | **Modify** lines 55–56 |
| `regime_trader/fetchers/fmp_fetcher.py` | **Modify** — make market-agnostic |
| `tests/test_fetchers.py` | **Modify** — update `FMPFetcher` tests |

---

## Task 1: Create FMPClient service

**Files:**
- Create: `regime_trader/services/fmp_client.py`
- Create: `tests/test_fmp_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fmp_client.py
"""Unit tests for FMPClient service module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import time
import pytest

from regime_trader.services.fmp_client import FMPClient


# ── Fixtures & helpers ──────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FMP_MAX_RPS", "1000")  # disable rate limiting in tests
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


def _ok_resp(data):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


def _empty_resp():
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = []
    return resp


# ── FMPClient construction ──────────────────────────────────────────────────

class TestFMPClientConstruction:
    def test_reads_api_key_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "env-key")
        c = FMPClient(cache_root=tmp_path / "fmp")
        assert c._api_key == "env-key"

    def test_no_key_sets_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(cache_root=tmp_path / "fmp")
        assert c._api_key == ""

    def test_fmp_max_rps_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "10")
        c = FMPClient(api_key="k", cache_root=tmp_path / "fmp")
        assert c._min_delay == pytest.approx(0.1, abs=1e-6)


# ── Congress factor ─────────────────────────────────────────────────────────

SENATE_RECORD = {
    "symbol": "NVDA",
    "senator": "Nancy Pelosi",
    "transactionDate": "2026-04-01",
    "disclosureDate": "2026-04-15",
    "type": "Purchase",
    "amount": "15001-50000",
}
HOUSE_RECORD = {
    "representative": "John Doe",
    "ticker": "NVDA",
    "transactionDate": "2026-04-10",
    "disclosureDate": "2026-04-20",
    "type": "Purchase--",
    "amount": "1001-15000",
}


class TestFMPClientCongress:
    def test_returns_dict_with_purchases_on_senate_data(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([SENATE_RECORD])):
            result = client.get_congress_trades("NVDA", lookback_days=180)
        assert isinstance(result, dict)
        assert result.get("purchases", 0) >= 1

    def test_recency_days_uses_disclosure_date(self, client):
        """disclosureDate=2026-04-15 should be used, not transactionDate=2026-04-01."""
        with patch.object(client._session, "get", return_value=_ok_resp([SENATE_RECORD])):
            result = client.get_congress_trades("NVDA", lookback_days=180)
        assert "recency_days" in result
        # disclosureDate is more recent than transactionDate, so recency_days should be < 60
        assert result["recency_days"] < 60

    def test_returns_empty_dict_on_api_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_congress_trades("NVDA")
        assert result == {}

    def test_returns_empty_dict_on_empty_response(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            result = client.get_congress_trades("NVDA")
        assert result == {}

    def test_returns_empty_dict_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(cache_root=tmp_path / "fmp")
        result = c.get_congress_trades("NVDA")
        assert result == {}

    def test_caches_result_on_second_call(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([SENATE_RECORD])) as mock_get:
            client.get_congress_trades("NVDA", lookback_days=180)
            client.get_congress_trades("NVDA", lookback_days=180)
            assert mock_get.call_count == 1


# ── Insider factor ──────────────────────────────────────────────────────────

INSIDER_RECORD_ACQUISITION = {
    "symbol": "NVDA",
    "filingDate": "2026-04-15",
    "transactionDate": "2026-04-01",
    "disclosureDate": "2026-04-15",
    "acquistionOrDisposition": "A",
    "securitiesTransacted": 1000.0,
    "price": 800.0,
    "transactionType": "P-Purchase",
}
INSIDER_RECORD_DISPOSITION = {
    **INSIDER_RECORD_ACQUISITION,
    "acquistionOrDisposition": "D",
}


class TestFMPClientInsider:
    def test_returns_tuple_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD_ACQUISITION])):
            usd, days = client.get_insider_purchases("NVDA", lookback_days=180)
        assert usd == pytest.approx(800_000.0)
        assert days >= 0

    def test_filters_disposition_records(self, client):
        """Only 'A' (Acquisition) records count — dispositions must be ignored."""
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD_DISPOSITION])):
            usd, days = client.get_insider_purchases("NVDA")
        assert usd == 0.0

    def test_null_securities_transacted_skipped(self, client):
        """None/empty securitiesTransacted must not raise ValueError."""
        bad = {**INSIDER_RECORD_ACQUISITION, "securitiesTransacted": None}
        with patch.object(client._session, "get", return_value=_ok_resp([bad])):
            usd, days = client.get_insider_purchases("NVDA")
        assert usd == 0.0

    def test_null_price_skipped(self, client):
        bad = {**INSIDER_RECORD_ACQUISITION, "price": None}
        with patch.object(client._session, "get", return_value=_ok_resp([bad])):
            usd, days = client.get_insider_purchases("NVDA")
        assert usd == 0.0

    def test_returns_zero_tuple_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            usd, days = client.get_insider_purchases("NVDA")
        assert (usd, days) == (0.0, 0)

    def test_returns_zero_tuple_on_empty_response(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            usd, days = client.get_insider_purchases("NVDA")
        assert (usd, days) == (0.0, 0)

    def test_caches_per_ticker(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([INSIDER_RECORD_ACQUISITION])) as mock_get:
            client.get_insider_purchases("NVDA")
            client.get_insider_purchases("NVDA")
            assert mock_get.call_count == 1

    def test_uses_limit_500_in_request(self, client):
        """Mega-caps need limit=500 to cover 180-day lookback without truncation."""
        with patch.object(client._session, "get", return_value=_empty_resp()) as mock_get:
            client.get_insider_purchases("NVDA")
        call_args = mock_get.call_args
        params = call_args[1].get("params", {}) or (call_args[0][1] if len(call_args[0]) > 1 else {})
        assert str(params.get("limit", "")) == "500"


# ── News factor ─────────────────────────────────────────────────────────────

NEWS_POSITIVE = {"title": "NVDA beats earnings", "sentiment": "Positive", "publishedDate": "2026-04-15"}
NEWS_NEGATIVE = {"title": "NVDA misses guidance", "sentiment": "Negative", "publishedDate": "2026-04-14"}
NEWS_NEUTRAL  = {"title": "NVDA releases product", "sentiment": "Neutral",  "publishedDate": "2026-04-13"}


class TestFMPClientNews:
    def test_returns_float_on_success(self, client):
        articles = [NEWS_POSITIVE] * 30 + [NEWS_NEGATIVE] * 10
        with patch.object(client._session, "get", return_value=_ok_resp(articles)):
            score = client.get_news_raw_articles("NVDA")
        assert isinstance(score, list)
        assert len(score) == 40

    def test_returns_empty_list_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_news_raw_articles("NVDA")
        assert result == []

    def test_caches_result(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([NEWS_POSITIVE])) as mock_get:
            client.get_news_raw_articles("NVDA")
            client.get_news_raw_articles("NVDA")
            assert mock_get.call_count == 1


# ── Quote factor ─────────────────────────────────────────────────────────────

QUOTE_RECORD = {
    "symbol": "NVDA",
    "price": 800.0,
    "marketCap": 2_000_000_000_000,
    "volume": 50_000_000,
    "avgVolume": 40_000_000,
    "eps": 22.5,
}


class TestFMPClientQuote:
    def test_returns_dict_on_success(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([QUOTE_RECORD])):
            result = client.get_quote("NVDA")
        assert result["symbol"] == "NVDA"
        assert result["price"] == 800.0

    def test_bypass_cache_forces_live_call(self, client):
        with patch.object(client._session, "get", return_value=_ok_resp([QUOTE_RECORD])) as mock_get:
            client.get_quote("NVDA")
            client.get_quote("NVDA", bypass_cache=True)
            assert mock_get.call_count == 2

    def test_returns_empty_dict_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_quote("NVDA")
        assert result == {}

    def test_accepts_international_suffix(self, client):
        """FMP Ultimate accepts SAP.DE, 7203.T natively."""
        with patch.object(client._session, "get", return_value=_ok_resp([{**QUOTE_RECORD, "symbol": "SAP.DE"}])):
            result = client.get_quote("SAP.DE")
        assert result.get("symbol") == "SAP.DE"


# ── Stub: _enrich_with_quiver ───────────────────────────────────────────────

class TestEnrichWithQuiverStub:
    def test_always_returns_empty_quiver_dict(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from regime_trader.scanners.discovery_scanner import _enrich_with_quiver
        results = [{"symbol": "NVDA"}, {"symbol": "AAPL"}]
        enriched = _enrich_with_quiver(results)
        for r in enriched:
            assert r["quiver"] == {}

    def test_no_network_calls_ever(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from regime_trader.scanners.discovery_scanner import _enrich_with_quiver
        with patch("requests.get") as mock_get:
            _enrich_with_quiver([{"symbol": "NVDA"}])
            mock_get.assert_not_called()


# ── CI isolation ────────────────────────────────────────────────────────────

class TestCIIsolation:
    def test_client_constructible_in_ci(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("CI", "1")
        c = FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")
        assert c is not None
        assert c._api_key == "test-key"

    def test_fmp_max_rps_defaults_to_20(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.delenv("FMP_MAX_RPS", raising=False)
        c = FMPClient(api_key="k", cache_root=tmp_path / "fmp")
        assert c._min_delay == pytest.approx(1.0 / 20.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_fmp_client.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'regime_trader.services.fmp_client'`

- [ ] **Step 3: Create the FMPClient service**

```python
# regime_trader/services/fmp_client.py
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
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
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
        """Aggregate senate + house trades into {purchases, sales, total, recency_days}.

        Uses disclosureDate (not transactionDate) for recency_days — alpha decay
        starts when information is public, not when the secret trade occurred.
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

        for path, type_key in [
            (f"/api/v4/senate-trading", "type"),
            (f"/api/v4/house-trades", "type"),
        ]:
            data = self._get(path, {"symbol": ticker}) or []
            for rec in data:
                disclosure = rec.get("disclosureDate", "")
                if not disclosure or disclosure < cutoff:
                    continue
                tx_type = (rec.get(type_key) or "").lower()
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_fmp_client.py -v --tb=short
```

Expected: All tests in `TestFMPClientConstruction`, `TestFMPClientCongress`, `TestFMPClientInsider`, `TestFMPClientNews`, `TestFMPClientQuote`, `TestCIIsolation` pass. `TestEnrichWithQuiverStub` will fail — that's expected (scanner not updated yet).

- [ ] **Step 5: Commit**

```bash
git add regime_trader/services/fmp_client.py tests/test_fmp_client.py
git commit -m "feat(fmp-client): add FMPClient with per-bucket TTL cache and FMP_MAX_RPS rate limiter

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Stub out _enrich_with_quiver in discovery_scanner.py

**Files:**
- Modify: `regime_trader/scanners/discovery_scanner.py` lines 850–882

- [ ] **Step 1: The test already exists in test_fmp_client.py**

The `TestEnrichWithQuiverStub` class was written in Task 1. Run it now to confirm it fails:

```
pytest tests/test_fmp_client.py::TestEnrichWithQuiverStub -v
```

Expected: FAIL — `_enrich_with_quiver` still imports `QuiverClient`.

- [ ] **Step 2: Replace the function body in discovery_scanner.py**

In `regime_trader/scanners/discovery_scanner.py`, find the function at line 850:

```python
def _enrich_with_quiver(result_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach Quiver data to each result dict as payload["quiver"].

    Gracefully degrades: if QUIVER_API_KEY is absent or any per-ticker call
    fails, quiver key is set to {} rather than crashing the scan.
    """
    from regime_trader.services.quiver_client import QuiverClient
    client = QuiverClient()
    if not client._api_key:
        for r in result_dicts:
            r.setdefault("quiver", {})
        return result_dicts

    try:
        congress_map = client.congress_by_ticker(lookback_days=180)
    except Exception as exc:
        log.warning("quiver congress_by_ticker failed: %s", exc)
        congress_map = {}

    for r in result_dicts:
        ticker = r.get("symbol", "").upper()
        quiver: Dict[str, Any] = {}
        try:
            quiver["congress"]      = congress_map.get(ticker, {})
            quiver["insider"]       = client.get_insider_trades(ticker)
            quiver["f13"]           = client.get_13f_summary(ticker)
            quiver["lobbying"]      = client.get_lobbying(ticker)
            quiver["gov_contracts"] = client.get_gov_contracts(ticker)
        except Exception as exc:
            log.warning("quiver enrichment failed for %s: %s", ticker, exc)
        r["quiver"] = quiver

    return result_dicts
```

Replace entirely with:

```python
def _enrich_with_quiver(result_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Quiver deprecated — returns empty quiver payload for UI backward compatibility."""
    for r in result_dicts:
        r.setdefault("quiver", {})
    return result_dicts
```

- [ ] **Step 3: Run the stub tests**

```
pytest tests/test_fmp_client.py::TestEnrichWithQuiverStub -v
```

Expected: 2 PASSED.

- [ ] **Step 4: Run full suite to confirm no regressions**

```
pytest tests/ -v --tb=short -q
```

Expected: same pass count as before this task (1085 tests). The `test_quiver_client.py::TestEnrichWithQuiver` tests will still pass because they mock `QuiverClient` — but `test_fmp_client.py::TestEnrichWithQuiverStub` now also passes.

- [ ] **Step 5: Commit**

```bash
git add regime_trader/scanners/discovery_scanner.py
git commit -m "refactor(scanner): stub out _enrich_with_quiver — Quiver deprecated

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Replace fetch_quiver_insider_all with fetch_fmp_insider_all

**Files:**
- Modify: `scripts/run_pipeline.py` lines 38, 820–889, 1259–1282, 1304–1347

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fmp_client.py` — a new class to test the pipeline-level wrapper:

```python
# Append to tests/test_fmp_client.py

class TestFetchFMPInsiderAll:
    """Tests for the fetch_fmp_insider_all() function in run_pipeline.py."""

    def test_returns_dict_keyed_by_ticker(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        from scripts.run_pipeline import fetch_fmp_insider_all
        from regime_trader.services.fmp_client import FMPClient

        with patch.object(FMPClient, "get_insider_purchases", return_value=(800_000.0, 5)):
            result = fetch_fmp_insider_all(["NVDA", "AAPL"])
        assert "NVDA" in result
        assert result["NVDA"] == (800_000.0, 5)
        assert "AAPL" in result

    def test_returns_empty_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        from scripts.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all(["NVDA"])
        assert result == {}

    def test_returns_empty_list_on_empty_input(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from scripts.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all([])
        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_fmp_client.py::TestFetchFMPInsiderAll -v
```

Expected: `ImportError: cannot import name 'fetch_fmp_insider_all' from 'scripts.run_pipeline'`

- [ ] **Step 3: Update the import in run_pipeline.py line 38**

Find:
```python
from regime_trader.services.quiver_client import QuiverClient as _QuiverClient  # noqa: E402
```

Replace with:
```python
from regime_trader.services.fmp_client import FMPClient as _FMPClient  # noqa: E402
```

- [ ] **Step 4: Replace _fetch_quiver_congress and the congress fallback block**

Find the function at line 275:
```python
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
```

Replace with:
```python
def _fetch_fmp_congress(ticker: str) -> Optional[Dict]:
    """Fetch congressional trades for a single ticker via FMPClient.

    Returns populated dict (with recency_days) on success, None if key absent or fails.
    """
    try:
        client = _FMPClient()
        if not client._api_key:
            return None
        return client.get_congress_trades(ticker, lookback_days=180) or None
    except Exception as exc:
        log.warning("FMP congress fetch failed for %s: %s", ticker, exc)
        return None
```

Then in `fetch_congress_buys` (around line 352), find the fallback block:
```python
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
```

Replace with:
```python
    # ── Fallback: FMP Ultimate (when S3 yields nothing) ───────────────────────
    if not s3_ok or not by_ticker:
        log.info("S3 congress feeds unavailable — trying FMP Ultimate fallback…")
        fmp_client = _FMPClient()
        if fmp_client._api_key:
            fmp_congress: Dict[str, Dict] = {}
            for ticker_key in list(by_ticker.keys()) or []:
                result = _fetch_fmp_congress(ticker_key)
                if result:
                    fmp_congress[ticker_key] = result
            if fmp_congress:
                by_ticker = fmp_congress
                n_tx = sum(v.get("total", 0) for v in fmp_congress.values())
                log.info("FMP congress fallback: %d transactions across %d tickers",
                         n_tx, len(fmp_congress))
        else:
            log.warning(
                "FMP_API_KEY not set and S3 feeds are down — "
                "congress factor will be 0.0 (penalised) for all tickers."
            )
```

- [ ] **Step 5: Replace fetch_quiver_insider_all with fetch_fmp_insider_all**

Find the entire function starting at line 820:
```python
def fetch_quiver_insider_all(
    tickers: List[str],
    lookback_days: int = 180,
    max_workers: int = 5,
) -> Dict[str, Tuple[float, int]]:
```

Replace the entire function (lines 820–889) with:
```python
def fetch_fmp_insider_all(
    tickers: List[str],
    lookback_days: int = 180,
    max_workers: int = 10,
) -> Dict[str, Tuple[float, int]]:
    """Fetch insider purchase data for all tickers via FMP Ultimate /api/v4/insider-trading.

    Stiglitz (2001 Nobel) — Form 4 insider purchases are a credible costly-to-fake signal.
    Uses FMPClient with limit=500 per ticker to cover 180-day lookback for mega-caps.

    max_workers=10 chosen for FMP Ultimate (50 req/s cap, 20 rps default via FMP_MAX_RPS).

    Returns {ticker: (total_purchases_usd, days_since_most_recent)}.
    Tickers with no qualifying purchases get (0.0, 0).
    Returns {} if FMP_API_KEY is not set.
    """
    client = _FMPClient()
    if not client._api_key:
        log.info("FMP_API_KEY not set -- skipping FMP insider pre-fetch")
        return {}

    if not tickers:
        return {}

    def _fetch_one(ticker: str) -> Tuple[str, Tuple[float, int]]:
        try:
            result = client.get_insider_purchases(ticker, lookback_days=lookback_days)
        except Exception as exc:
            log.debug("FMP insider fetch failed for %s: %s", ticker, type(exc).__name__)
            result = (0.0, 0)
        log.debug("FMP insider %s: $%.0f", ticker, result[0])
        return ticker, result

    results: Dict[str, Tuple[float, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, value in pool.map(_fetch_one, tickers):
            results[ticker] = value

    nonzero = sum(1 for v in results.values() if v[0] > 0)
    log.info(
        "FMP insider pre-fetch complete: %d/%d tickers with purchases",
        nonzero, len(tickers),
    )
    return results
```

- [ ] **Step 6: Update the run() function — insider pre-fetch block (around lines 1259–1282)**

Find:
```python
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
```

Replace with:
```python
    # ── FMP insider — primary source (Form 4, cached 12h, limit=500) ─────────────
    # FMPClient.get_insider_purchases() returns (total_usd, days) per ticker.
    # Pre-fetched serially here; the thread pool below reads from the in-memory dict.
    log.info("Pre-fetching FMP insider transactions for %d tickers…", len(tickers))
    fmp_insider_cache: Dict[str, Tuple[float, int]] = fetch_fmp_insider_all(tickers)
    fmp_has_data = any(v[0] > 0 for v in fmp_insider_cache.values())
    if not fmp_has_data:
        log.info("FMP insider returned no data -- insider scoring uses EDGAR XML only")
```

- [ ] **Step 7: Update the per-ticker scoring block (around lines 1304–1347)**

Find the insider source resolution block:
```python
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
```

Replace with:
```python
            # Insider purchases: FMP Form 4 (primary) → EDGAR XML (fallback).
            if ticker in fmp_insider_cache:
                fmp_usd, fmp_days = fmp_insider_cache[ticker]
                if fmp_usd > 0:
                    total_purchases_usd    = fmp_usd
                    days_since_most_recent = fmp_days
                    ceo_buy = total_purchases_usd > 25_000
```

Then find the source attribution block:
```python
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
```

Replace with:
```python
            # Determine which insider source actually provided data
            _fmp_usd = fmp_insider_cache.get(ticker, (0.0, 0))[0]
            if _fmp_usd > 0:
                insider_source = "fmp"
            elif total_purchases_usd > 0:
                insider_source = "edgar"
            else:
                insider_source = "none"
```

Also find and remove this line in the `_score_ticker` closure:
```python
            finnhub_key = finnhub_key_global  # use pre-fetched key (avoids per-thread env lookup)
```

- [ ] **Step 8: Update source_meta dict (around line 1492)**

Find:
```python
    source_meta: Dict[str, Dict[str, Any]] = {
        "quiver":   {"last_updated": pipeline_run_ts},
        "finnhub":  {"last_updated": pipeline_run_ts},
        "edgar":    {"last_updated": pipeline_run_ts},
        "none":     {"last_updated": pipeline_run_ts},
    }
```

Replace with:
```python
    source_meta: Dict[str, Dict[str, Any]] = {
        "fmp":    {"last_updated": pipeline_run_ts},
        "edgar":  {"last_updated": pipeline_run_ts},
        "none":   {"last_updated": pipeline_run_ts},
    }
```

- [ ] **Step 9: Run the new tests**

```
pytest tests/test_fmp_client.py::TestFetchFMPInsiderAll -v
```

Expected: 3 PASSED.

- [ ] **Step 10: Run full suite**

```
pytest tests/ -q --tb=short
```

Expected: all tests pass. The `test_quiver_client.py` tests still pass because `QuiverClient` still exists as a file.

- [ ] **Step 11: Commit**

```bash
git add scripts/run_pipeline.py tests/test_fmp_client.py
git commit -m "feat(pipeline): replace fetch_quiver_insider_all with fetch_fmp_insider_all

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Replace score_news_finnhub with score_news_fmp

**Files:**
- Modify: `scripts/run_pipeline.py` lines 593–619, 1324–1368

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fmp_client.py`:

```python
class TestScoreNewsFMP:
    """Tests for score_news_fmp() in run_pipeline.py."""

    def test_scores_positive_articles_above_neutral(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from scripts.run_pipeline import score_news_fmp
        from regime_trader.services.fmp_client import FMPClient

        articles = [{"sentiment": "Positive"}] * 40 + [{"sentiment": "Negative"}] * 10
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("NVDA")
        assert score > 0.5

    def test_falls_back_to_yfinance_on_empty_articles(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from scripts.run_pipeline import score_news_fmp
        from regime_trader.services.fmp_client import FMPClient

        with patch.object(FMPClient, "get_news_raw_articles", return_value=[]), \
             patch("scripts.run_pipeline._score_news_yfinance", return_value=0.6) as mock_yf:
            score = score_news_fmp("SAP.DE")
        mock_yf.assert_called_once_with("SAP.DE")
        assert score == pytest.approx(0.6)

    def test_score_bounded_0_to_1(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from scripts.run_pipeline import score_news_fmp
        from regime_trader.services.fmp_client import FMPClient

        articles = [{"sentiment": "Positive"}] * 50
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("NVDA")
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_fmp_client.py::TestScoreNewsFMP -v
```

Expected: `ImportError: cannot import name 'score_news_fmp' from 'scripts.run_pipeline'`

- [ ] **Step 3: Replace score_news_finnhub in run_pipeline.py**

Find the function at line 593:
```python
def score_news_finnhub(ticker: str, api_key: str) -> float:
    """Engle (2003 Nobel) — Finnhub pre-computed sentiment score in [0, 1].
    ...
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
```

Replace with:
```python
def score_news_fmp(ticker: str) -> float:
    """Engle (2003 Nobel) — FMP news sentiment score in [0, 1].

    Formula: 0.60 * (positive_count / total) + 0.40 * min(1.0, total / 50)

    Falls back to _score_news_yfinance() immediately when:
      - FMP returns [] (EU/Asia tickers with thin coverage on calm days)
      - total == 0 (zero articles indexed)

    Returns 0.0 (not 0.5) if both sources fail — dead feed is penalised.
    """
    client = _FMPClient()
    articles = client.get_news_raw_articles(ticker)
    if not articles:
        return _score_news_yfinance(ticker)
    positive = sum(1 for a in articles if a.get("sentiment") == "Positive")
    total = len(articles)
    if total == 0:
        return _score_news_yfinance(ticker)
    buzz_norm = min(1.0, total / 50.0)
    return round(0.60 * (positive / total) + 0.40 * buzz_norm, 4)
```

- [ ] **Step 4: Update the n_score call in the per-ticker scoring block (around line 1324)**

Find:
```python
            n_score = (
                score_news_finnhub(ticker, finnhub_key)
                if finnhub_key
                else _score_news_yfinance(ticker)
            )
```

Replace with:
```python
            n_score = score_news_fmp(ticker)
```

- [ ] **Step 5: Update the news_source attribution block (around line 1365)**

Find:
```python
            if finnhub_key:
                news_source = "finnhub" if n_score > 0.0 else "none"
            else:
                news_source = "yfinance" if n_score > 0.0 else "none"
```

Replace with:
```python
            news_source = "fmp" if n_score > 0.0 else "none"
```

- [ ] **Step 6: Run the new tests**

```
pytest tests/test_fmp_client.py::TestScoreNewsFMP -v
```

Expected: 3 PASSED.

- [ ] **Step 7: Run full suite**

```
pytest tests/ -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_pipeline.py tests/test_fmp_client.py
git commit -m "feat(pipeline): replace score_news_finnhub with score_news_fmp

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Make FMPFetcher market-agnostic (fix EU 403)

**Files:**
- Modify: `regime_trader/fetchers/fmp_fetcher.py`
- Modify: `scripts/run_pipeline.py` lines 1431, 1437
- Modify: `tests/test_fetchers.py` — update FMPFetcher construction tests

- [ ] **Step 1: Read the current FMPFetcher test**

```
grep -n "FMPFetcher\|fmp_fetcher_market\|test_fmp" tests/test_fetchers.py
```

Note the exact test names and fixture patterns that reference `FMPFetcher`.

- [ ] **Step 2: Update FMPFetcher tests in test_fetchers.py**

Find the existing FMPFetcher tests and update the constructor call from `FMPFetcher(api_key="test")` to `FMPFetcher(api_key="test", market=MarketEnum.EUROPE)`. Also add a test for ASIA market:

```python
def test_fmp_fetcher_market_europe():
    from regime_trader.fetchers.fmp_fetcher import FMPFetcher
    from regime_trader.fetchers.base import MarketEnum
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    assert f.market == MarketEnum.EUROPE

def test_fmp_fetcher_market_asia():
    from regime_trader.fetchers.fmp_fetcher import FMPFetcher
    from regime_trader.fetchers.base import MarketEnum
    f = FMPFetcher(api_key="test", market=MarketEnum.ASIA)
    assert f.market == MarketEnum.ASIA
```

- [ ] **Step 3: Run updated fetcher tests to confirm they fail**

```
pytest tests/test_fetchers.py -k "fmp_fetcher" -v
```

Expected: FAIL — `FMPFetcher.__init__` does not accept `market` parameter yet.

- [ ] **Step 4: Update fmp_fetcher.py**

Replace the class definition in `regime_trader/fetchers/fmp_fetcher.py`:

```python
class FMPFetcher(BaseMarketFetcher):
    """Market-agnostic FMP equities fetcher for USA, EUROPE, and ASIA.

    FMP Ultimate accepts international suffixes natively (SAP.DE, 7203.T).
    No suffix translation needed — pass tickers as-is from ticker_registry.json.
    """

    def __init__(self, api_key: str, market: MarketEnum) -> None:
        self._api_key = api_key
        self._market = market

    @property
    def market(self) -> MarketEnum:
        return self._market

    def source_reliability(self, ticker: str) -> float:
        return _RELIABILITY

    def _fetch_quote(self, ticker: str) -> dict[str, Any]:
        url = f"{_FMP_BASE}/quote/{ticker}"
        for attempt in range(3):
            resp = requests.get(url, params={"apikey": self._api_key}, timeout=10)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 12.0 * (2 ** attempt)))
                logger.warning("FMP 429 for %s (attempt %d/3) -- sleeping %.1fs", ticker, attempt + 1, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data:
                raise ValueError(f"Empty FMP response for {ticker}")
            return data[0]
        raise RuntimeError(f"FMP rate-limited after 3 attempts for {ticker}")

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        usage = _load_usage()
        entries: list[TickerEntry] = []
        for ticker in tickers:
            if usage["count"] >= _DAILY_QUOTA:
                logger.warning(
                    "FMP daily quota reached (%d/%d) -- skipping remaining tickers",
                    usage["count"], _DAILY_QUOTA,
                )
                break
            try:
                quote = self._fetch_quote(ticker)
                usage["count"] += 1
                _save_usage(usage)
                price = float(quote.get("price") or 0)
                mktcap = float(quote.get("marketCap") or 0)
                momentum = (float(quote.get("volume") or 0) /
                            max(float(quote.get("avgVolume") or 1), 1)) - 1.0
                entries.append(TickerEntry(
                    ticker=ticker,
                    market=self._market,
                    sector="",
                    cap_tier="",
                    source_reliability=_RELIABILITY,
                    raw_factors={
                        "price": price,
                        "market_cap": mktcap,
                        "momentum": momentum,
                        "eps": float(quote.get("eps") or 0),
                    },
                ))
                time.sleep(_RATE_LIMIT_DELAY)
            except Exception as exc:
                logger.warning("FMPFetcher skip %s: %s", ticker, exc)
        return entries
```

- [ ] **Step 5: Update the orchestrator call in run_pipeline.py (around line 1431–1442)**

Find:
```python
        if fmp_key and registry_tickers.get("EUROPE"):
            eu_asia_fetchers.append(FMPFetcher(api_key=fmp_key))
            log.info("FMPFetcher added for EUROPE (%d tickers)", len(registry_tickers["EUROPE"]))
        elif not fmp_key:
            log.warning("FMP_API_KEY absent — EUROPE section will be empty in Discord")
        if registry_tickers.get("ASIA"):
            eu_asia_fetchers.append(AsianMarketFetcher())
            log.info("AsianMarketFetcher added for ASIA (%d tickers)", len(registry_tickers["ASIA"]))
```

Replace with:
```python
        if fmp_key and registry_tickers.get("EUROPE"):
            eu_asia_fetchers.append(FMPFetcher(api_key=fmp_key, market=MarketEnum.EUROPE))
            log.info("FMPFetcher added for EUROPE (%d tickers)", len(registry_tickers["EUROPE"]))
        elif not fmp_key:
            log.warning("FMP_API_KEY absent -- EUROPE section will be empty in Discord")
        if fmp_key and registry_tickers.get("ASIA"):
            eu_asia_fetchers.append(FMPFetcher(api_key=fmp_key, market=MarketEnum.ASIA))
            log.info("FMPFetcher added for ASIA (%d tickers)", len(registry_tickers["ASIA"]))
        elif registry_tickers.get("ASIA") and not fmp_key:
            log.warning("FMP_API_KEY absent -- ASIA section will be empty in Discord")
```

Also add the MarketEnum import at the top of the EU/Asia block (around line 1430), or confirm `MarketEnum` is already imported:

```python
        from regime_trader.fetchers import Orchestrator  # noqa: PLC0415
        from regime_trader.fetchers.fmp_fetcher import FMPFetcher  # noqa: PLC0415
        from regime_trader.fetchers.base import MarketEnum  # noqa: PLC0415
```

Remove the now-unused line:
```python
        from regime_trader.fetchers.asian_fetcher import AsianMarketFetcher  # noqa: PLC0415
```

- [ ] **Step 6: Run fetcher tests**

```
pytest tests/test_fetchers.py -v --tb=short
```

Expected: all fetcher tests pass including the new `test_fmp_fetcher_market_asia`.

- [ ] **Step 7: Run full suite**

```
pytest tests/ -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add regime_trader/fetchers/fmp_fetcher.py scripts/run_pipeline.py tests/test_fetchers.py
git commit -m "feat(fetchers): make FMPFetcher market-agnostic — replaces AsianMarketFetcher

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Delete Quiver dead code and update test_congress_fetcher.py

**Files:**
- Delete: `regime_trader/services/quiver_client.py`
- Delete: `tests/test_quiver_client.py`
- Modify: `scripts/run_pipeline.py` — remove Finnhub functions + module docstring update
- Modify: `tests/test_congress_fetcher.py` line 127

- [ ] **Step 1: Verify QuiverClient is no longer imported anywhere**

```bash
grep -rn "QuiverClient\|quiver_client\|QUIVER_API_KEY" scripts/ regime_trader/ tests/ --include="*.py"
```

Expected: only hits in `tests/test_quiver_client.py` and `regime_trader/services/quiver_client.py` themselves. If any other file still references `QuiverClient`, fix it before proceeding.

- [ ] **Step 2: Update test_congress_fetcher.py — rename the S3 fallback test**

In `tests/test_congress_fetcher.py`, find the test at line 127:
```python
    def test_s3_403_falls_back_to_quiver(self, tmp_path, monkeypatch):
        """When S3 returns 403, QuiverClient.congress_by_ticker() result is used."""
        monkeypatch.setenv("QUIVER_API_KEY", "test-key")
        ...
        with patch("requests.get", side_effect=mock_s3_get), \
             patch(
                 "regime_trader.services.quiver_client.QuiverClient.congress_by_ticker",
                 return_value=quiver_result,
             ):
            result = fetch_congress_buys(lookback_days=90)
```

Replace the entire test with:
```python
    def test_s3_403_falls_back_to_fmp(self, tmp_path, monkeypatch):
        """When S3 returns 403, FMPClient.get_congress_trades() result is used."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setattr("scripts.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")

        fmp_result = {
            "purchases": 1, "sales": 0, "total": 1, "recency_days": 5,
        }

        def mock_s3_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 403
            resp.raise_for_status.side_effect = Exception("403 Forbidden")
            return resp

        with patch("requests.get", side_effect=mock_s3_get), \
             patch(
                 "regime_trader.services.fmp_client.FMPClient.get_congress_trades",
                 return_value=fmp_result,
             ):
            result = fetch_congress_buys(lookback_days=90)

        # FMP fallback returns per-ticker dict; the congress data is still accessible
        assert isinstance(result, dict)
```

- [ ] **Step 3: Run test_congress_fetcher.py to confirm it passes**

```
pytest tests/test_congress_fetcher.py -v --tb=short
```

Expected: 12 PASSED (the renamed test + 11 unchanged tests).

- [ ] **Step 4: Remove Finnhub functions from run_pipeline.py**

Find and delete these three functions entirely (they are no longer called):

```python
def fetch_finnhub_insider_purchases(ticker: str, api_key: str, lookback_days: int = 180, ...) -> Tuple[float, int]:
    ...

def fetch_all_finnhub_insider(tickers: List[str], api_key: str, ...) -> Dict[str, Tuple[float, int]]:
    ...
```

Also delete `_parse_quiver_trades` if it exists (helper for the old `fetch_quiver_insider_all`).

Verify with:
```bash
grep -n "def fetch_finnhub\|def _parse_quiver\|def fetch_all_finnhub" scripts/run_pipeline.py
```

Expected: no matches.

- [ ] **Step 5: Update the module docstring at the top of run_pipeline.py (lines 1–16)**

Replace:
```python
"""scripts/run_pipeline.py
EDGAR + FMP + yfinance daily data pipeline.

Stiglitz (2001 Nobel) — asymmetric information: insider filing activity is a
credible, costly-to-fake signal. This pipeline sources it from three layers:
  1. Quiver Quantitative     — pre-parsed Form 4 (primary, QUIVER_API_KEY, 6h TTL)
  2. Finnhub                 — open-market purchases (fallback, FINNHUB_API_KEY)
  3. SEC EDGAR direct        — Form 4 count + CEO buy flag (always, free)

FMP budget: ≤ 80 calls per run (per-ticker profile, first 80 tickers only).
Tickers 81+ fall back to yfinance for market cap to stay within 250/day limit.
...
"""
```

With:
```python
"""scripts/run_pipeline.py
EDGAR + FMP Ultimate daily data pipeline — USA, EUROPE, ASIA.

Stiglitz (2001 Nobel) — asymmetric information: insider filing activity is a
credible, costly-to-fake signal. This pipeline sources it via:
  1. FMP Ultimate            — Form 4 insider (primary, FMP_API_KEY, 12h TTL)
  2. SEC EDGAR direct        — Form 4 count + CEO buy flag (always, free)

FMP Ultimate: 3000 req/min (50 req/s). Default rate: FMP_MAX_RPS=20 (configurable).
International: FMP natively accepts SAP.DE, ASML.AS, 7203.T suffixes.

Usage:
  python scripts/run_pipeline.py --tickers-file config/universe.csv --log-dir logs
  python -m scripts.run_pipeline --tickers-file config/universe.csv --verbose
"""
```

- [ ] **Step 6: Delete the dead files**

```bash
git rm regime_trader/services/quiver_client.py
git rm tests/test_quiver_client.py
```

- [ ] **Step 7: Run full suite**

```
pytest tests/ -q --tb=short
```

Expected: 1085+ tests pass (the Quiver tests are gone, the FMP tests replace them, net count may vary by ±5).

- [ ] **Step 8: Verify no remaining Quiver/Finnhub references**

```bash
grep -rn "QuiverClient\|quiver_client\|QUIVER_API_KEY\|FINNHUB_API_KEY\|score_news_finnhub\|fetch_quiver\|fetch_finnhub\|fetch_all_finnhub" scripts/ regime_trader/ tests/ --include="*.py"
```

Expected: zero matches.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_pipeline.py tests/test_congress_fetcher.py
git commit -m "feat(pipeline): remove Quiver and Finnhub dead code — FMP is sole data source

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Clean up environment variables (.env + workflows)

**Files:**
- Modify: `.env` — remove `QUIVER_API_KEY` and `FINNHUB_API_KEY` lines, add `FMP_MAX_RPS`
- Modify: `.github/workflows/nightly_edgar.yml` lines 76–77
- Modify: `.github/workflows/edgar_3x.yml` lines 93–94
- Modify: `.github/workflows/canary.yml` lines 55–56

- [ ] **Step 1: Update .env**

Open `.env` and remove these lines (do NOT print their values):
```
QUIVER_API_KEY=...
FINNHUB_API_KEY=...
```

Add this line below `FMP_API_KEY`:
```
FMP_MAX_RPS=20
```

- [ ] **Step 2: Update nightly_edgar.yml**

Find lines 76–77:
```yaml
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
```

Replace with:
```yaml
          FMP_MAX_RPS: ${{ vars.FMP_MAX_RPS || '20' }}
```

- [ ] **Step 3: Update edgar_3x.yml**

Find lines 93–94:
```yaml
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}   # primary insider source
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }} # fallback insider source
```

Replace with:
```yaml
          FMP_MAX_RPS: ${{ vars.FMP_MAX_RPS || '20' }}
```

- [ ] **Step 4: Update canary.yml**

Find lines 55–56:
```yaml
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
```

Replace with:
```yaml
          FMP_MAX_RPS: ${{ vars.FMP_MAX_RPS || '20' }}
```

- [ ] **Step 5: Run full suite one final time**

```
pytest tests/ -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Final grep — confirm zero Quiver/Finnhub references anywhere**

```bash
grep -rn "QUIVER\|QuiverClient\|FINNHUB\|finnhub\|quiver" scripts/ regime_trader/ tests/ .github/ --include="*.py" --include="*.yml" --include="*.yaml"
```

Expected: zero matches.

- [ ] **Step 7: Commit**

```bash
git add .env .github/workflows/nightly_edgar.yml .github/workflows/edgar_3x.yml .github/workflows/canary.yml
git commit -m "chore(env): remove QUIVER_API_KEY and FINNHUB_API_KEY — add FMP_MAX_RPS

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Verification

### Unit tests
```bash
pytest tests/ -v --tb=short
```
Expected: all tests pass.

### Dead code check
```bash
grep -rn "QUIVER\|QuiverClient\|FINNHUB\|quiver_client\|score_news_finnhub\|fetch_quiver\|fetch_all_finnhub" \
  scripts/ regime_trader/ tests/ .github/ \
  --include="*.py" --include="*.yml"
```
Expected: zero matches.

### End-to-end pipeline run
```bash
python scripts/run_pipeline.py --verbose 2>&1 | grep -E "FMP|insider|congress|news|EUROPE|ASIA"
```
Expected log lines:
- `Pre-fetching FMP insider transactions for 160 tickers…`
- `FMP insider pre-fetch complete: N/160 tickers with purchases`
- `FMPFetcher added for EUROPE (10 tickers)`
- `FMPFetcher added for ASIA (10 tickers)`
- `Orchestrator: N entries from EUROPE` (non-zero — no more 403)
- `Scored: N EU entries, N Asia entries added to results`

### Source attribution check
```bash
python -c "
import json
data = json.load(open('logs/intel_source_status.json', encoding='utf-8'))
sources = {r.get('quiver_evidence', {}).get('insider_source') for r in data['results']}
print('Insider sources used:', sources)
"
```
Expected: `{'fmp', 'edgar', 'none'}` — no `'quiver'` or `'finnhub'`.
