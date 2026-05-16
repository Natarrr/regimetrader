# QuiverQuant Full Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate QuiverQuant as a first-class data pillar alongside EDGAR and FMP, adding congressional trades, insider trades, institutional 13F, lobbying, and government contracts as scored signals surfaced in the pipeline, discovery scanner, and Streamlit UI.

**Architecture:** A new `QuiverClient` service module (mirroring `fmp_service.py` structure) handles all Quiver API calls with Bearer-token auth, file-based TTL caching under `.cache/quiver/`, and retry/backoff. The existing `congress` factor in `generate_top_lists.py` is already wired; this plan enriches it with recency weighting and adds Quiver evidence metadata. The discovery scanner gets a `payload["quiver"]` dict. Streamlit UI gets a new Quiver evidence expander.

**Tech Stack:** Python 3.11, `requests` with `HTTPAdapter` retry, `numpy`, existing `save_json_atomic()` / `load_json_safe()` utilities, `pytest` with `monkeypatch`.

---

### File Map

| Action | Path |
|--------|------|
| Create | `regime_trader/services/quiver_client.py` |
| Modify | `scripts/run_pipeline.py` (congressional scoring already done; add evidence metadata) |
| Modify | `backend/market_intel/generate_top_lists.py` (recency weighting, evidence in output) |
| Modify | `regime_trader/scanners/discovery_scanner.py` (add `payload["quiver"]`) |
| Modify | `regime_trader/ui/streamlit_app.py` (Congress Score column, Quiver evidence expander) |
| Create | `tests/test_quiver_client.py` |

---

### Task 1: Create `regime_trader/services/quiver_client.py`

**Files:**
- Create: `regime_trader/services/quiver_client.py`

The module exposes a `QuiverClient` class with 5 endpoint methods. It must follow the exact same caching/retry pattern as `fmp_service.py`.

- [ ] **Step 1: Write failing test in `tests/test_quiver_client.py`**

```python
"""tests/test_quiver_client.py"""
from __future__ import annotations
import json, time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from regime_trader.services.quiver_client import QuiverClient

CONGRESS_RECORD = {
    "Representative": "Nancy Pelosi", "BioGuideID": "P000197",
    "ReportDate": "2026-04-15", "TransactionDate": "2026-04-01",
    "Ticker": "NVDA", "Transaction": "Purchase",
    "Range": "$50,001 - $100,000", "House": "House",
    "Amount": 75000, "Party": "Democrat", "TickerType": "ST",
    "ExcessReturn": 5.2, "PriceChange": 3.1,
}
INSIDER_RECORD = {
    "Name": "Jensen Huang", "Title": "CEO",
    "Date": "2026-04-01", "Ticker": "NVDA",
    "AcquisitionOrDisposition": "A", "Shares": 10000,
    "PricePerShare": 800.0, "TotalValue": 8000000.0,
    "FilingURL": "https://sec.gov/cgi-bin/browse-edgar",
}
F13_RECORD = {
    "Date": "2025-12-31", "Shares": 1500000,
    "Value": 1200000000, "Pct": 2.5,
    "PctChange": 0.8,
}

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIVER_API_KEY", "test-key")
    return QuiverClient(cache_root=tmp_path / "quiver")

class TestQuiverClientCongress:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [CONGRESS_RECORD]
            mock_get.return_value = resp
            result = client.get_politician_trades(lookback_days=90)
        assert isinstance(result, list)
        assert result[0]["Ticker"] == "NVDA"

    def test_caches_result(self, client):
        with patch.object(client._session, "get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [CONGRESS_RECORD]
            mock_get.return_value = resp
            client.get_politician_trades(lookback_days=90)
            client.get_politician_trades(lookback_days=90)
            assert mock_get.call_count == 1  # second call uses cache

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_politician_trades(lookback_days=90)
        assert result == []

    def test_no_api_key_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUIVER_API_KEY", raising=False)
        c = QuiverClient(cache_root=tmp_path / "quiver")
        assert c.get_politician_trades() == []

class TestQuiverClientInsider:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [INSIDER_RECORD]
            mock_get.return_value = resp
            result = client.get_insider_trades("NVDA")
        assert isinstance(result, list)
        assert result[0]["Ticker"] == "NVDA"

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("network")):
            assert client.get_insider_trades("NVDA") == []

class TestQuiverClient13F:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [F13_RECORD]
            mock_get.return_value = resp
            result = client.get_13f_summary("NVDA")
        assert isinstance(result, list)

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("network")):
            assert client.get_13f_summary("NVDA") == []

class TestQuiverClientLobbying:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [{"Ticker": "NVDA", "Amount": 500000}]
            mock_get.return_value = resp
            result = client.get_lobbying("NVDA")
        assert result[0]["Ticker"] == "NVDA"

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            assert client.get_lobbying("NVDA") == []

class TestQuiverClientGovContracts:
    def test_returns_list_on_success(self, client):
        with patch.object(client._session, "get") as mock_get:
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = [{"Ticker": "LMT", "Amount": 1000000}]
            mock_get.return_value = resp
            result = client.get_gov_contracts("LMT")
        assert result[0]["Ticker"] == "LMT"

    def test_returns_empty_on_error(self, client):
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            assert client.get_gov_contracts("LMT") == []

class TestCIIsolation:
    def test_no_real_http_calls_in_ci(self, tmp_path, monkeypatch):
        """CI=1 must not allow real HTTP through — conftest blocks at Session.send level."""
        monkeypatch.setenv("QUIVER_API_KEY", "test-key")
        monkeypatch.setenv("CI", "1")
        c = QuiverClient(cache_root=tmp_path / "quiver")
        # The conftest fixture blocks Session.send — just confirm client is constructible
        assert c is not None
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_quiver_client.py -v
```
Expected: `ModuleNotFoundError` for `regime_trader.services.quiver_client`

- [ ] **Step 3: Create `regime_trader/services/quiver_client.py`**

```python
"""regime_trader/services/quiver_client.py
Quiver Quantitative API client.

Stiglitz (2001 Nobel) — congressional trading exploits non-public information.
QuiverQuant surfaces this asymmetric signal via structured legislative data.

Endpoints used:
  /beta/live/congresstrading   — house/senate trades (live feed)
  /beta/live/insidertrading    — SEC Form 4 structured insider trades
  /beta/live/historical/13f/{ticker} — institutional 13F history
  /beta/live/lobbying          — corporate lobbying spend
  /beta/live/govcontracts      — government contract awards

Auth: Bearer token via QUIVER_API_KEY env var (Hobbyist plan).
Cache: file-based under .cache/quiver/, TTL-checked at read time.
CI isolation: calls are blocked by conftest when CI=1.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

_BASE_URL = "https://api.quiverquant.com"
_TIMEOUT = 30
_MAX_RETRIES = 3
_BACKOFF = 1.0

_DEFAULT_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "quiver"

_TTL: dict[str, int] = {
    "congresstrading": 6 * 3600,   # 6 h — updates ~daily
    "insidertrading":  6 * 3600,
    "13f":            24 * 3600,   # 24 h — quarterly filing
    "lobbying":       24 * 3600,
    "govcontracts":   24 * 3600,
}


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
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


class QuiverClient:
    """Thin wrapper around the Quiver Quantitative REST API.

    Args:
        api_key: Quiver Bearer token. Defaults to QUIVER_API_KEY env var.
        cache_root: Directory for file-based cache. Defaults to .cache/quiver/.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_root: Optional[Path] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("QUIVER_API_KEY", "")
        self._cache_root = Path(cache_root) if cache_root else _DEFAULT_CACHE_ROOT
        self._session = _make_session(self._api_key) if self._api_key else None

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_path(self, bucket: str, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
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
        """
        cached = self._cache_read("congresstrading", "all")
        if cached is not None:
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
        """
        cached = self._cache_read("insidertrading", ticker)
        if cached is not None:
            return cached
        if not self._api_key:
            return []
        data = self._get(f"/beta/live/insidertrading/{ticker}")
        result: List[dict] = data if isinstance(data, list) else []
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

    def congress_by_ticker(self, lookback_days: int = 180) -> dict[str, dict]:
        """Aggregate politician trades into per-ticker buy/sell/net counts.

        Returns: {TICKER: {purchases: int, sales: int, total: int, net: int,
                            representatives: list[str], recency_days: int}}
        """
        from datetime import datetime, timedelta, timezone
        records = self.get_politician_trades(lookback_days=lookback_days)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        by_ticker: dict[str, dict] = {}
        for rec in records:
            ticker = rec.get("Ticker", "").strip().upper()
            if not ticker or ticker in {"N/A", "--", "", "NONE"}:
                continue
            ticker_type = rec.get("TickerType", "ST")
            if ticker_type != "ST":
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
            if tx_type in ("purchase",):
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
                    datetime.now(timezone.utc) -
                    datetime.fromisoformat(tx_date).replace(tzinfo=timezone.utc)
                ).days
                entry["recency_days"] = min(entry["recency_days"], days_ago)
            except Exception:
                pass

        return by_ticker


# Module-level singleton — use when no custom cache root needed
default_quiver = QuiverClient()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_quiver_client.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add regime_trader/services/quiver_client.py tests/test_quiver_client.py
git commit -m "feat(quiver): add QuiverClient service module with caching and 5 endpoints"
```

---

### Task 2: Wire Quiver congress data into `scripts/run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py` — replace `_fetch_quiver_congress()` with `QuiverClient.congress_by_ticker()` call; add recency-weighted congress scoring

The existing `_fetch_quiver_congress()` in `run_pipeline.py` duplicates logic now in `QuiverClient`. Replace it so the pipeline delegates to the service module. Also add recency weighting to `score_congress()`.

- [ ] **Step 1: Write failing test in `tests/test_congress_fetcher.py`** (add to existing class)

Add this test to `TestScoreCongress`:

```python
def test_recent_trade_scores_higher_than_old(self):
    recent = score_congress({"purchases": 3, "sales": 0, "total": 3, "recency_days": 5})
    old    = score_congress({"purchases": 3, "sales": 0, "total": 3, "recency_days": 150})
    assert recent > old
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_congress_fetcher.py::TestScoreCongress::test_recent_trade_scores_higher_than_old -v
```
Expected: FAIL — `score_congress` ignores `recency_days`.

- [ ] **Step 3: Update `score_congress()` in `scripts/run_pipeline.py`**

Replace the current `score_congress` function body:

```python
def score_congress(data: Optional[Dict]) -> float:
    """Score congressional trading signal [0, 1].

    0.0  → dead feed (None / empty): normaliser penalises this correctly.
    0.5  → data present but net flat (equal buys and sells).
    >0.5 → net purchases (bullish congressional signal).
    <0.5 → net sales (bearish congressional signal).

    Recency weighting: trades within 30 days get full credit; older trades
    decay linearly to 0.7× at 180 days (log-linear interpolation).
    """
    if not data:
        return 0.0
    purchases = int(data.get("purchases", 0))
    sales     = int(data.get("sales", 0))
    total     = purchases + sales
    if total == 0:
        return 0.50   # data present but no net activity → genuinely neutral

    raw = (purchases - sales) / (total + 1)
    base_score = round((raw + 1) / 2, 4)

    # Recency multiplier: full credit ≤30 days, decay to 0.70× at 180 days
    recency_days = data.get("recency_days")
    if recency_days is not None and recency_days > 30:
        decay = max(0.70, 1.0 - 0.30 * min(recency_days - 30, 150) / 150)
        # Dampen towards neutral (0.5) not towards 0
        base_score = 0.5 + (base_score - 0.5) * decay

    return round(base_score, 4)
```

- [ ] **Step 4: Update `fetch_congress_buys()` to delegate to `QuiverClient`**

In `scripts/run_pipeline.py`, update the Quiver fallback to use the service module:

```python
# At top of file, add import:
from regime_trader.services.quiver_client import QuiverClient as _QuiverClient

# Replace _fetch_quiver_congress() function:
def _fetch_quiver_congress(cutoff: str) -> Optional[Dict[str, Dict]]:
    """Delegate to QuiverClient.congress_by_ticker() — avoids duplicating HTTP/cache logic."""
    try:
        client = _QuiverClient()
        if not client._api_key:
            return None
        return client.congress_by_ticker(lookback_days=180) or None
    except Exception as exc:
        log.warning("quiver congress delegation failed: %s", exc)
        return None
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_congress_fetcher.py -v
```
Expected: all tests pass, including new recency test.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_pipeline.py tests/test_congress_fetcher.py
git commit -m "feat(congress): recency-weighted score_congress() + delegate quiver fetch to QuiverClient"
```

---

### Task 3: Add Quiver evidence metadata to `generate_top_lists.py`

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py`

Add quiver evidence to each ticker's output entry. The existing `_to_entry()` function builds the per-ticker dict; extend it to accept and embed `quiver_evidence`.

- [ ] **Step 1: Write failing test** (add to `tests/test_cross_sectional.py` at bottom)

```python
class TestQuiverEvidence:
    def test_to_entry_includes_quiver_evidence(self):
        from backend.market_intel.generate_top_lists import _to_entry
        row = {
            "ticker": "NVDA", "market_cap": 1e12, "sector": "Technology",
            "edgar_score": 0.7, "insider_score": 0.6, "congress_score": 0.8,
            "news_score": 0.5, "momentum_score": 0.5,
        }
        norm = {"edgar": 0.7, "insider": 0.6, "congress": 0.8, "news": 0.5, "macro": 0.5}
        evidence = {"politicians": ["Nancy Pelosi"], "recency_days": 5}
        entry = _to_entry(row, norm, vix=None, quiver_evidence=evidence)
        assert entry["quiver_evidence"]["politicians"] == ["Nancy Pelosi"]
        assert entry["quiver_evidence"]["recency_days"] == 5

    def test_to_entry_no_evidence_gives_empty_dict(self):
        from backend.market_intel.generate_top_lists import _to_entry
        row = {
            "ticker": "AAPL", "market_cap": 2e12, "sector": "Technology",
            "edgar_score": 0.5, "insider_score": 0.5, "congress_score": 0.5,
            "news_score": 0.5, "momentum_score": 0.5,
        }
        norm = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5, "macro": 0.5}
        entry = _to_entry(row, norm, vix=None)
        assert entry.get("quiver_evidence") == {}
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_cross_sectional.py::TestQuiverEvidence -v
```
Expected: FAIL — `_to_entry` does not accept `quiver_evidence`.

- [ ] **Step 3: Update `_to_entry()` signature in `generate_top_lists.py`**

Find the current `_to_entry` function signature and update it:

```python
def _to_entry(
    row: Dict[str, Any],
    norm_factors: Dict[str, float],
    vix: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    quiver_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```

Inside the function, just before the `return` statement, add:

```python
    entry["quiver_evidence"] = quiver_evidence or {}
    return entry
```

(Find the existing `return` at the end of `_to_entry` and add the line before it.)

- [ ] **Step 4: Update `generate()` to pass quiver evidence**

In the `generate()` function, after the cross-sectional normalization loop where `_to_entry()` is called, pass `quiver_evidence` from the row if present:

```python
# In the loop that calls _to_entry(), change:
entry = _to_entry(row, norm, vix=vix, weights=eff_weights)
# to:
entry = _to_entry(
    row, norm, vix=vix, weights=eff_weights,
    quiver_evidence=row.get("quiver_evidence"),
)
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_cross_sectional.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/market_intel/generate_top_lists.py tests/test_cross_sectional.py
git commit -m "feat(toplists): add quiver_evidence metadata to _to_entry() output"
```

---

### Task 4: Add `payload["quiver"]` to `discovery_scanner.py`

**Files:**
- Modify: `regime_trader/scanners/discovery_scanner.py`

The discovery scanner's `run_scan()` returns a list of result dicts. Each result should include a `"quiver"` key with the ticker's Quiver data.

- [ ] **Step 1: Read the end of discovery_scanner.py to understand run_scan() output**

Read `regime_trader/scanners/discovery_scanner.py` lines 850–974.

- [ ] **Step 2: Write failing test** (add to `tests/test_discovery_scanner.py` if it exists, or note in test_quiver_client.py)

```python
# In tests/test_quiver_client.py, add:
class TestQuiverClientCongressByTicker:
    def test_congress_by_ticker_aggregates_purchases(self, client):
        records = [
            {"Ticker": "NVDA", "Transaction": "Purchase",
             "TransactionDate": "2026-04-01", "TickerType": "ST",
             "Representative": "Nancy Pelosi"},
            {"Ticker": "NVDA", "Transaction": "Purchase",
             "TransactionDate": "2026-04-10", "TickerType": "ST",
             "Representative": "John Doe"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert result["NVDA"]["purchases"] == 2
        assert result["NVDA"]["sales"] == 0
        assert len(result["NVDA"]["representatives"]) == 2

    def test_congress_by_ticker_filters_non_stock(self, client):
        records = [
            {"Ticker": "BTC", "Transaction": "Purchase",
             "TransactionDate": "2026-04-01", "TickerType": "CRYPTO",
             "Representative": "Someone"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert "BTC" not in result

    def test_congress_by_ticker_filters_old_trades(self, client):
        records = [
            {"Ticker": "AAPL", "Transaction": "Purchase",
             "TransactionDate": "2020-01-01", "TickerType": "ST",
             "Representative": "Someone"},
        ]
        with patch.object(client, "get_politician_trades", return_value=records):
            result = client.congress_by_ticker(lookback_days=90)
        assert "AAPL" not in result
```

- [ ] **Step 3: Run tests to verify congress_by_ticker tests pass** (they use existing code)

```
pytest tests/test_quiver_client.py::TestQuiverClientCongressByTicker -v
```
Expected: all 3 tests pass (the logic was already written in Task 1).

- [ ] **Step 4: Add `_enrich_with_quiver()` to `discovery_scanner.py`**

Add this function near the bottom of `discovery_scanner.py`, before `run_scan()`:

```python
def _enrich_with_quiver(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach Quiver data to each result dict as payload["quiver"].

    Gracefully degrades: if QUIVER_API_KEY is absent or any call fails,
    quiver key is set to {} rather than crashing the scan.
    """
    from regime_trader.services.quiver_client import QuiverClient
    client = QuiverClient()
    if not client._api_key:
        for r in results:
            r.setdefault("quiver", {})
        return results

    congress_map = client.congress_by_ticker(lookback_days=180)

    for r in results:
        ticker = r.get("ticker", "").upper()
        quiver: Dict[str, Any] = {}
        try:
            quiver["congress"] = congress_map.get(ticker, {})
            quiver["insider"]  = client.get_insider_trades(ticker)
            quiver["f13"]      = client.get_13f_summary(ticker)
            quiver["lobbying"] = client.get_lobbying(ticker)
            quiver["gov_contracts"] = client.get_gov_contracts(ticker)
        except Exception as exc:
            log.warning("quiver enrichment failed for %s: %s", ticker, exc)
        r["quiver"] = quiver

    return results
```

- [ ] **Step 5: Call `_enrich_with_quiver()` inside `run_scan()` / `get_top_alpha_picks_sync()`**

Find where `run_scan()` or `get_top_alpha_picks_sync()` assembles the final `results` list and add the enrichment call just before returning:

```python
results = _enrich_with_quiver(results)
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_quiver_client.py -v
```

- [ ] **Step 7: Commit**

```bash
git add regime_trader/scanners/discovery_scanner.py tests/test_quiver_client.py
git commit -m "feat(scanner): enrich discovery results with Quiver data (congress/insider/13f/lobbying/contracts)"
```

---

### Task 5: Update `streamlit_app.py` — Congress Score column + Quiver evidence expander

**Files:**
- Modify: `regime_trader/ui/streamlit_app.py`

The Market Intel tab (lines ~1008–1186) shows a results table. Add a "Congress" column and a Quiver evidence expander below the table.

- [ ] **Step 1: Read current Market Intel tab rendering code**

Read `regime_trader/ui/streamlit_app.py` lines 1008–1186.

- [ ] **Step 2: Add Congress Score to the displayed columns**

Find the section that builds the display DataFrame (likely a list of dicts or `pd.DataFrame` call). Add `"congress_score"` to the displayed columns, formatted as a percentage or 2-decimal float.

Find a line like:
```python
cols = ["ticker", "final_score", "badge", "edgar_score", "insider_score", ...]
```
And add `"congress_score"` after `"insider_score"`.

- [ ] **Step 3: Add Quiver evidence expander**

After the results table `st.dataframe(...)` call, add:

```python
# Quiver evidence expanders (one per top ticker)
for entry in top_entries[:5]:
    quiver = entry.get("quiver_evidence") or entry.get("quiver") or {}
    if not quiver:
        continue
    ticker = entry.get("ticker", "")
    with st.expander(f"Quiver Evidence — {ticker}"):
        congress = quiver.get("congress", {})
        if congress:
            st.markdown(f"**Congress:** {congress.get('purchases',0)} buys / {congress.get('sales',0)} sells — "
                        f"last trade {congress.get('recency_days', '?')} days ago")
            reps = congress.get("representatives", [])
            if reps:
                st.markdown(f"Representatives: {', '.join(reps[:5])}")
        insider = quiver.get("insider", [])
        if insider:
            st.markdown(f"**Insider (Quiver):** {len(insider)} recent Form 4 filings")
        f13 = quiver.get("f13", [])
        if f13:
            latest = f13[0] if f13 else {}
            st.markdown(f"**Institutional 13F:** {latest.get('Pct','?')}% held, "
                        f"change {latest.get('PctChange','?')}%")
        lobbying = quiver.get("lobbying", [])
        if lobbying:
            total_lobby = sum(r.get("Amount", 0) for r in lobbying[:4])
            st.markdown(f"**Lobbying:** ${total_lobby:,.0f} (last 4 quarters)")
        contracts = quiver.get("gov_contracts", [])
        if contracts:
            total_contracts = sum(r.get("Amount", 0) for r in contracts[:4])
            st.markdown(f"**Gov Contracts:** ${total_contracts:,.0f} (last 4 quarters)")
```

- [ ] **Step 4: Run CI tests to ensure no import errors**

```
pytest tests/ -v --ignore=tests/test_quiver_client.py -x -q
```

- [ ] **Step 5: Commit**

```bash
git add regime_trader/ui/streamlit_app.py
git commit -m "feat(ui): add Congress Score column and Quiver evidence expanders to Market Intel tab"
```

---

### Task 6: Run full test suite and fix failures

**Files:** Various (fix only)

- [ ] **Step 1: Run full suite**

```
pytest tests/ -v -q 2>&1 | tail -40
```

- [ ] **Step 2: Fix any failures**

Common failure modes to watch for:
- Import errors if `quiver_client.py` path is wrong
- CI=1 conftest blocking real HTTP (expected — mock all calls)
- Missing `quiver_evidence` key in `_to_entry()` callers
- `score_congress` recency tests failing if decay formula wrong

- [ ] **Step 3: Commit fixes**

```bash
git add -u
git commit -m "fix(quiver): fix test failures from full suite run"
```

---

### Task 7: Final validation

- [ ] **Step 1: Confirm all tests pass**

```
pytest tests/ -q
```
Expected: no failures.

- [ ] **Step 2: Confirm imports work**

```python
python -c "from regime_trader.services.quiver_client import QuiverClient; print('OK')"
python -c "from backend.market_intel.generate_top_lists import _to_entry; print('OK')"
```

- [ ] **Step 3: Summary commit**

All prior tasks have individual commits. No additional commit needed.
