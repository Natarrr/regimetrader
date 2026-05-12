# Discord Market Checkup — Signal Quality & Universe Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two dead scoring factors (hardcoded congress=0.50, uniform VIX macro) with real congressional trading data and 20-day price momentum; expand the ticker universe from 50 to 165 tickers balanced across 11 GICS sectors; apply cross-sectional normalization so the Top 5 Buys reflect genuine peer-relative conviction.

**Architecture:** Five tasks touching four files. Tasks 1–3 produce a better `intel_source_status.json`; Task 4 adds cross-sectional normalization in `generate_top_lists.py`; Task 5 updates the Discord label and workflow default. Each task is independently commitable. All functions follow the existing pattern: Nobel laureate docstring, lazy yfinance/requests imports, exception → log + safe default.

**Tech Stack:** Python 3.11 · yfinance (price data) · requests (House/Senate Stock Watcher S3 feeds) · `regime_trader/scoring/normalize.py` (already exists, provides `normalize_score()`) · FMP REST API (profile batch, insider list) · EDGAR daily index (Form 4)

---

## File Map

| File | Role |
|---|---|
| `config/universe.csv` | **Create** — 165 tickers, 11 GICS sectors replacing `config/top50.csv` |
| `scripts/run_pipeline.py` | **Modify** — add `fetch_congress_buys()`, `score_congress()`, `fetch_price_data()`, `score_momentum()`; extend EDGAR lookback 90→180d; chunk FMP profile batch at 100; rename `macro`→`momentum` in WEIGHTS; remove unused VIX helpers |
| `scripts/generate_top_lists.py` | **Modify** — add `_cross_sectional_normalize()`, update `WEIGHTS` key and `FACTOR_FIELDS` mapping, refactor `generate()` to use normalized factors |
| `scripts/send_toplists_discord.py` | **Modify** — update `_FACTOR_EMOJI` and `_format_factor_line` for `momentum` key |
| `.github/workflows/edgar_3x.yml` | **Modify** — update default `tickers_file` to `config/universe.csv` |
| `tests/test_congress_fetcher.py` | **Create** — tests for `fetch_congress_buys()` and `score_congress()` |
| `tests/test_pipeline_momentum.py` | **Create** — tests for `fetch_price_data()` and `score_momentum()` |
| `tests/test_cross_sectional.py` | **Create** — tests for `_cross_sectional_normalize()` in generate_top_lists |

---

## Task 1: Universe CSV (165 tickers, 11 GICS sectors)

**Files:**
- Create: `config/universe.csv`
- Modify: `.github/workflows/edgar_3x.yml` (default tickers_file)

- [ ] **Step 1: Create `config/universe.csv`**

```
ticker,sector,cap_tier
META,Communication Services,large
GOOGL,Communication Services,large
NFLX,Communication Services,large
DIS,Communication Services,large
CHTR,Communication Services,large
CMCSA,Communication Services,large
T,Communication Services,large
VZ,Communication Services,large
TMUS,Communication Services,large
EA,Communication Services,large
WBD,Communication Services,large
FOXA,Communication Services,large
IPG,Communication Services,large
OMC,Communication Services,large
TTD,Communication Services,large
AMZN,Consumer Discretionary,large
TSLA,Consumer Discretionary,large
HD,Consumer Discretionary,large
NKE,Consumer Discretionary,large
MCD,Consumer Discretionary,large
TGT,Consumer Discretionary,large
LOW,Consumer Discretionary,large
SBUX,Consumer Discretionary,large
GM,Consumer Discretionary,large
BKNG,Consumer Discretionary,large
ORLY,Consumer Discretionary,large
MAR,Consumer Discretionary,large
RCL,Consumer Discretionary,large
ULTA,Consumer Discretionary,large
ROST,Consumer Discretionary,large
WMT,Consumer Staples,large
PG,Consumer Staples,large
KO,Consumer Staples,large
PEP,Consumer Staples,large
COST,Consumer Staples,large
PM,Consumer Staples,large
MO,Consumer Staples,large
CL,Consumer Staples,large
GIS,Consumer Staples,large
ADM,Consumer Staples,large
HSY,Consumer Staples,large
STZ,Consumer Staples,large
CHD,Consumer Staples,large
HRL,Consumer Staples,large
SYY,Consumer Staples,large
XOM,Energy,large
CVX,Energy,large
COP,Energy,large
SLB,Energy,large
EOG,Energy,large
MPC,Energy,large
PSX,Energy,large
VLO,Energy,large
OXY,Energy,large
HAL,Energy,large
DVN,Energy,large
BKR,Energy,large
FANG,Energy,large
HES,Energy,large
MRO,Energy,large
JPM,Financials,large
BAC,Financials,large
WFC,Financials,large
GS,Financials,large
MS,Financials,large
BLK,Financials,large
SPGI,Financials,large
CB,Financials,large
AXP,Financials,large
PNC,Financials,large
USB,Financials,large
TRV,Financials,large
MET,Financials,large
AFL,Financials,large
PRU,Financials,large
JNJ,Healthcare,large
LLY,Healthcare,large
ABBV,Healthcare,large
UNH,Healthcare,large
MRK,Healthcare,large
PFE,Healthcare,large
ABT,Healthcare,large
DHR,Healthcare,large
AMGN,Healthcare,large
TMO,Healthcare,large
ISRG,Healthcare,large
CVS,Healthcare,large
MDT,Healthcare,large
SYK,Healthcare,large
BMY,Healthcare,large
CAT,Industrials,large
HON,Industrials,large
GE,Industrials,large
BA,Industrials,large
UPS,Industrials,large
RTX,Industrials,large
LMT,Industrials,large
DE,Industrials,large
MMM,Industrials,large
EMR,Industrials,large
ETN,Industrials,large
FDX,Industrials,large
WM,Industrials,large
CARR,Industrials,large
PCAR,Industrials,large
AAPL,Information Technology,large
MSFT,Information Technology,large
NVDA,Information Technology,large
ORCL,Information Technology,large
CRM,Information Technology,large
NOW,Information Technology,large
ADBE,Information Technology,large
AMD,Information Technology,large
QCOM,Information Technology,large
IBM,Information Technology,large
INTC,Information Technology,large
TXN,Information Technology,large
AMAT,Information Technology,large
MU,Information Technology,large
ACN,Information Technology,large
LIN,Materials,large
APD,Materials,large
ECL,Materials,large
NEM,Materials,large
FCX,Materials,large
NUE,Materials,large
ALB,Materials,large
CF,Materials,large
MOS,Materials,large
PPG,Materials,large
RPM,Materials,large
IP,Materials,large
PKG,Materials,large
DOW,Materials,large
IFF,Materials,large
AMT,Real Estate,large
PLD,Real Estate,large
CCI,Real Estate,large
EQIX,Real Estate,large
PSA,Real Estate,large
O,Real Estate,large
DLR,Real Estate,large
SPG,Real Estate,large
VICI,Real Estate,large
AVB,Real Estate,large
EQR,Real Estate,large
WY,Real Estate,large
NEE,Utilities,large
DUK,Utilities,large
SO,Utilities,large
D,Utilities,large
AEP,Utilities,large
EXC,Utilities,large
XEL,Utilities,large
PCG,Utilities,large
WEC,Utilities,large
ETR,Utilities,large
PPL,Utilities,large
CMS,Utilities,large
ES,Utilities,large
```

- [ ] **Step 2: Verify 165 rows**

Run:
```bash
python -c "import csv; rows=list(csv.DictReader(open('config/universe.csv'))); print(len(rows), 'tickers'); [print(s, sum(1 for r in rows if r['sector']==s)) for s in sorted(set(r['sector'] for r in rows))]"
```

Expected output:
```
165 tickers
Communication Services 15
Consumer Discretionary 15
Consumer Staples 15
Energy 15
Financials 15
Healthcare 15
Industrials 15
Information Technology 15
Materials 15
Real Estate 12
Utilities 13
```

- [ ] **Step 3: Update `edgar_3x.yml` default tickers_file**

In `.github/workflows/edgar_3x.yml`, change the two occurrences of `config/top50.csv` to `config/universe.csv`:

```yaml
# In workflow_dispatch inputs section:
      tickers_file:
        description: "Tickers CSV (relative to repo root)"
        required: false
        default: "config/universe.csv"   # was config/top50.csv

# In the EDGAR fetch step:
          TICKERS_FILE="${{ github.event.inputs.tickers_file || 'config/universe.csv' }}"
```

- [ ] **Step 4: Commit**

```bash
git add config/universe.csv .github/workflows/edgar_3x.yml
git commit -m "feat(universe): expand ticker universe to 165 tickers across 11 GICS sectors"
```

---

## Task 2: Congressional Trading Factor

**Files:**
- Modify: `scripts/run_pipeline.py` (add `fetch_congress_buys`, `score_congress`)
- Create: `tests/test_congress_fetcher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_congress_fetcher.py`:

```python
"""tests/test_congress_fetcher.py
Unit tests for congressional trading data fetching and scoring.

Stiglitz (2001 Nobel) — asymmetric information: congressional trading
is a credible signal of non-public knowledge. Tests use mocked S3 feeds;
no live network calls in CI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the two functions under test
from scripts.run_pipeline import fetch_congress_buys, score_congress


class TestScoreCongress:
    def test_no_data_returns_neutral(self):
        assert score_congress(None) == pytest.approx(0.5, abs=1e-4)

    def test_empty_dict_returns_neutral(self):
        assert score_congress({}) == pytest.approx(0.5, abs=1e-4)

    def test_all_purchases_above_neutral(self):
        score = score_congress({"purchases": 4, "sales": 0, "total": 4})
        assert score > 0.5

    def test_all_sales_below_neutral(self):
        score = score_congress({"purchases": 0, "sales": 4, "total": 4})
        assert score < 0.5

    def test_equal_purchases_and_sales_near_neutral(self):
        score = score_congress({"purchases": 3, "sales": 3, "total": 6})
        assert score == pytest.approx(0.5, abs=0.05)

    def test_output_bounded_0_to_1(self):
        for p, s in [(10, 0), (0, 10), (5, 5), (1, 0)]:
            score = score_congress({"purchases": p, "sales": s, "total": p + s})
            assert 0.0 <= score <= 1.0

    def test_zero_total_returns_neutral(self):
        score = score_congress({"purchases": 0, "sales": 0, "total": 0})
        assert score == pytest.approx(0.5, abs=1e-4)


class TestFetchCongressBuys:
    def _make_mock_get(self, house_data, senate_data):
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "house" in url:
                resp.json.return_value = house_data
            else:
                resp.json.return_value = senate_data
            return resp
        return side_effect

    def test_counts_house_purchases(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            {"transaction_date": "2026-04-01", "ticker": "AAPL", "type": "purchase"},
            {"transaction_date": "2026-04-10", "ticker": "AAPL", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)

        assert "AAPL" in result
        assert result["AAPL"]["purchases"] == 2
        assert result["AAPL"]["sales"] == 0

    def test_counts_senate_sales(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        senate = [
            {"transaction_date": "2026-04-05", "ticker": "MSFT", "type": "sale"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get([], senate)):
            result = fetch_congress_buys(lookback_days=90)

        assert "MSFT" in result
        assert result["MSFT"]["sales"] == 1

    def test_ignores_transactions_outside_lookback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            # 200 days ago — outside 90-day window
            {"transaction_date": "2025-10-01", "ticker": "NVDA", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)

        assert "NVDA" not in result

    def test_skips_invalid_tickers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            {"transaction_date": "2026-04-01", "ticker": "N/A", "type": "purchase"},
            {"transaction_date": "2026-04-01", "ticker": "", "type": "purchase"},
            {"transaction_date": "2026-04-01", "ticker": "--", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])):
            result = fetch_congress_buys(lookback_days=90)

        assert result == {}

    def test_network_failure_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        with patch("requests.get", side_effect=Exception("timeout")):
            result = fetch_congress_buys(lookback_days=90)

        assert result == {}

    def test_cache_is_used_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_pipeline.CONGRESS_CACHE_PATH",
            tmp_path / "congress_cache.json",
        )
        house = [
            {"transaction_date": "2026-04-01", "ticker": "JPM", "type": "purchase"},
        ]
        with patch("requests.get", side_effect=self._make_mock_get(house, [])) as mock_get:
            fetch_congress_buys(lookback_days=90)
            call_count_first = mock_get.call_count
            fetch_congress_buys(lookback_days=90)
            # Second call should not hit network
            assert mock_get.call_count == call_count_first
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_congress_fetcher.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — `fetch_congress_buys` and `score_congress` don't exist yet.

- [ ] **Step 3: Add `CONGRESS_CACHE_PATH`, `fetch_congress_buys()`, and `score_congress()` to `scripts/run_pipeline.py`**

After the existing imports at the top of `scripts/run_pipeline.py`, add:

```python
import math
```

Replace the existing `WEIGHTS` block (lines 41–47) and the `_VIX_MACRO` block (lines 50–57) with:

```python
# ── Weights (must sum to 1.0) ──────────────────────────────────────────────────
WEIGHTS = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "momentum": 0.10,   # renamed from macro — now 20-day price return
}

# ── Congress feed cache path (module-level so tests can monkeypatch it) ────────
CONGRESS_CACHE_PATH = ROOT / ".cache" / "congress_cache.json"

_CONGRESS_TTL_HOURS = 24
_HOUSE_URL   = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL  = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_INVALID_TICKERS = frozenset({"N/A", "--", "", "NONE"})
```

Then add the two new functions after `fetch_fmp_insider_buys()` (around line 163):

```python
# ── Congressional trading fetcher ──────────────────────────────────────────────

def fetch_congress_buys(lookback_days: int = 90) -> Dict[str, Dict]:
    """Stiglitz (2001 Nobel) — fetch congressional trading from House/Senate Stock Watcher.

    Downloads both chamber feeds from public S3 (no API key required), filters
    to the lookback window, and counts purchase vs sale transactions per ticker.
    Results are cached for 24 h to avoid redundant downloads within a trading day.

    Returns:
        Dict keyed by ticker → {"purchases": int, "sales": int, "total": int}
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

    for url, label in [(_HOUSE_URL, "house"), (_SENATE_URL, "senate")]:
        try:
            resp = _req.get(url, timeout=30)
            resp.raise_for_status()
            transactions = resp.json()
            for tx in transactions:
                date_str = str(tx.get("transaction_date") or tx.get("date") or "")
                if date_str[:10] < cutoff:
                    continue
                ticker = str(tx.get("ticker") or "").upper().strip()
                if not ticker or ticker in _INVALID_TICKERS:
                    continue
                tx_type = str(tx.get("type") or "").lower()
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
        except Exception as exc:
            log.warning("Congress feed %s failed: %s", label, exc)

    # ── Persist cache ─────────────────────────────────────────────────────────
    try:
        CONGRESS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_json_atomic(CONGRESS_CACHE_PATH, {"_ts": time.time(), "by_ticker": by_ticker})
    except Exception as exc:
        log.debug("Congress cache write failed: %s", exc)

    return by_ticker


def score_congress(data: Optional[Dict]) -> float:
    """Stiglitz (2001 Nobel) — congressional net buy signal ∈ [0, 1].

    $score = \\frac{(purchases - sales) / (total + 1) + 1}{2}$

    A ticker with only purchases scores >0.5; only sales scores <0.5;
    no congressional activity or equal purchases/sales scores 0.5.
    """
    if not data:
        return 0.50
    total = int(data.get("total", 0))
    if total == 0:
        return 0.50
    purchases = int(data.get("purchases", 0))
    sales     = int(data.get("sales", 0))
    raw = (purchases - sales) / (total + 1)   # ∈ (-1, 1)
    return round((raw + 1) / 2, 4)             # → (0, 1)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_congress_fetcher.py -v
```

Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_congress_fetcher.py
git commit -m "feat(congress): add House/Senate Stock Watcher congressional trading factor"
```

---

## Task 3: Momentum Factor + EDGAR Lookback + FMP Chunking + Wire Into Pipeline

**Files:**
- Modify: `scripts/run_pipeline.py`
- Create: `tests/test_pipeline_momentum.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_momentum.py`:

```python
"""tests/test_pipeline_momentum.py
Unit tests for momentum factor and pipeline wiring changes.

Thaler (2017 Nobel) — behavioral momentum: prices continue trending
in the direction of institutional conviction. Tests use injected DataFrames;
no live yfinance calls in CI.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_pipeline import (
    fetch_price_data,
    score_momentum,
    fetch_fmp_profiles,
)


class TestScoreMomentum:
    def test_positive_return_above_neutral(self):
        assert score_momentum(0.10) > 0.5

    def test_negative_return_below_neutral(self):
        assert score_momentum(-0.10) < 0.5

    def test_zero_return_near_neutral(self):
        assert score_momentum(0.0) == pytest.approx(0.5, abs=0.02)

    def test_large_positive_capped(self):
        """Returns > 30% are clamped — score should equal score at +30%."""
        assert score_momentum(0.99) == pytest.approx(score_momentum(0.30), abs=1e-6)

    def test_large_negative_capped(self):
        assert score_momentum(-0.99) == pytest.approx(score_momentum(-0.30), abs=1e-6)

    def test_output_bounded_0_to_1(self):
        for r in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            assert 0.0 <= score_momentum(r) <= 1.0


class TestFetchPriceData:
    def _fake_df(self, start_price: float, end_price: float, n: int = 21) -> pd.DataFrame:
        """Create a fake yfinance Close DataFrame."""
        prices = np.linspace(start_price, end_price, n)
        idx    = pd.date_range("2026-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": prices}, index=idx)

    def test_positive_momentum_detected(self):
        fake = self._fake_df(100.0, 110.0)   # +10% over the period
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("AAPL")
        assert result["return_20d"] > 0.0

    def test_negative_momentum_detected(self):
        fake = self._fake_df(100.0, 90.0)    # -10%
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("AAPL")
        assert result["return_20d"] < 0.0

    def test_flat_market_returns_near_zero(self):
        fake = self._fake_df(100.0, 100.0)
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("AAPL")
        assert abs(result["return_20d"]) < 1e-6

    def test_empty_dataframe_returns_zero(self):
        fake = pd.DataFrame()
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("INVALID")
        assert result == {"return_20d": 0.0}

    def test_exception_returns_zero(self):
        with patch("yfinance.download", side_effect=Exception("network error")):
            result = fetch_price_data("AAPL")
        assert result == {"return_20d": 0.0}

    def test_return_key_present(self):
        fake = self._fake_df(100.0, 105.0)
        with patch("yfinance.download", return_value=fake):
            result = fetch_price_data("MSFT")
        assert "return_20d" in result


class TestFetchFmpProfilesChunking:
    def test_165_tickers_makes_two_calls(self):
        """165 tickers chunked at 100 → 2 FMP GET calls."""
        tickers = [f"T{i:03d}" for i in range(165)]
        payload = [{"symbol": t, "mktCap": 1e9} for t in tickers[:100]]

        call_count = 0
        def fake_fmp_get(path, params, timeout=20):
            nonlocal call_count
            call_count += 1
            symbols = params.get("symbol", "").split(",")
            return [{"symbol": s, "mktCap": 1e9} for s in symbols]

        with patch("scripts.run_pipeline._fmp_get", side_effect=fake_fmp_get):
            result = fetch_fmp_profiles(tickers)

        assert call_count == 2
        assert len(result) == 165

    def test_50_tickers_makes_one_call(self):
        tickers = [f"T{i:02d}" for i in range(50)]
        call_count = 0
        def fake_fmp_get(path, params, timeout=20):
            nonlocal call_count
            call_count += 1
            return [{"symbol": s, "mktCap": 1e9} for s in params["symbol"].split(",")]
        with patch("scripts.run_pipeline._fmp_get", side_effect=fake_fmp_get):
            fetch_fmp_profiles(tickers)
        assert call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_momentum.py -v 2>&1 | head -30
```

Expected: `ImportError` for `fetch_price_data` and `score_momentum`; chunking test fails because current `fetch_fmp_profiles` sends all tickers in one call.

- [ ] **Step 3: Implement changes in `scripts/run_pipeline.py`**

**3a — Remove the now-unused VIX helpers** (delete `_VIX_MACRO`, `fetch_vix()`, `vix_to_macro()`). These are replaced by `fetch_price_data()` + `score_momentum()`.

**3b — Update `fetch_fmp_profiles()` to chunk at 100 tickers:**

Replace the existing `fetch_fmp_profiles` function with:

```python
def fetch_fmp_profiles(tickers: List[str]) -> Dict[str, float]:
    """Fama (2013 Nobel): batch profile fetch — ≤2 FMP calls for 165 tickers (chunks of 100).

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
    return result
```

**3c — Add `fetch_price_data()` and `score_momentum()`** after `fetch_vix()` is deleted:

```python
# ── yfinance price fetcher ─────────────────────────────────────────────────────

def fetch_price_data(ticker: str) -> Dict[str, float]:
    """Thaler (2017 Nobel) — 20-day price return for behavioral momentum.

    $r_{20d} = (P_t - P_{t-20}) / P_{t-20}$

    Returns {"return_20d": float} where float ∈ ℝ (unbounded before normalization).
    Returns {"return_20d": 0.0} on any error — treated as neutral after
    cross-sectional normalization.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period="1mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 5:
            return {"return_20d": 0.0}
        close = df["Close"].squeeze().dropna()
        if len(close) < 2:
            return {"return_20d": 0.0}
        ret = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
        return {"return_20d": ret}
    except Exception:
        return {"return_20d": 0.0}


def score_momentum(return_20d: float) -> float:
    """Thaler (2017 Nobel) — map 20-day return to pre-normalization score ∈ [0.10, 0.90].

    Clips return at ±30% before mapping:
    $score = 0.10 + \\frac{clip(r, -0.30, 0.30) + 0.30}{0.60} \\times 0.80$

    This bounded value is subsequently normalized cross-sectionally across the
    universe in generate_top_lists.py; absolute value matters less than ordering.
    """
    r = max(-0.30, min(0.30, return_20d))
    return round(0.10 + (r + 0.30) / 0.60 * 0.80, 4)
```

**3d — Update `count_edgar_form4()` lookback** from 90 to 180 days:

```python
def count_edgar_form4(ticker: str, lookback_days: int = 180) -> int:
```

**3e — Update `run()` to fetch congress data and remove VIX:**

In `run()`, replace:
```python
    # ── Macro: yfinance VIX ───────────────────────────────────────────────────
    vix = fetch_vix()
    macro_score = vix_to_macro(vix)
    log.info("VIX=%.1f  macro_score=%.2f", vix, macro_score)
```

With:
```python
    # ── Congressional trading: House + Senate Stock Watcher ───────────────────
    log.info("Fetching congressional trading data…")
    congress_data = fetch_congress_buys()
    congress_count = len(congress_data)
    log.info("Congressional data: %d tickers with activity", congress_count)
```

**3f — Update `_score_ticker()` to use new factors:**

Replace the existing `_score_ticker` inner function:

```python
    def _score_ticker(row: Dict[str, str]) -> Dict[str, Any]:
        ticker = row["ticker"]
        try:
            form4_count  = count_edgar_form4(ticker)
            e_score      = score_edgar(form4_count)
            i_score, ceo_buy = score_insider(fmp_insider.get(ticker))
            c_score      = score_congress(congress_data.get(ticker))
            n_score      = score_news(ticker)
            price_data   = fetch_price_data(ticker)
            m_score      = score_momentum(price_data["return_20d"])
            return {
                "ticker":          ticker,
                "sector":          sector.get(ticker, "Unknown"),
                "cap_tier":        cap_tier.get(ticker, "large"),
                "market_cap":      mktcaps.get(ticker, 0.0),
                "edgar_score":     e_score,
                "insider_score":   i_score,
                "congress_score":  c_score,
                "news_score":      n_score,
                "momentum_score":  m_score,
                "ceo_buy":         ceo_buy,
                "form4_count":     form4_count,
            }
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", ticker, exc)
            return {
                "ticker":         ticker,
                "sector":         sector.get(ticker, "Unknown"),
                "cap_tier":       cap_tier.get(ticker, "large"),
                "market_cap":     0.0,
                "edgar_score":    0.30,
                "insider_score":  0.50,
                "congress_score": 0.50,
                "news_score":     0.50,
                "momentum_score": 0.50,
                "ceo_buy":        False,
                "form4_count":    0,
            }
```

**3g — Update `run()` status block** to add `congress_count` and fix `fmp_count`:

```python
    n_profile_chunks = math.ceil(len(tickers) / 100) if tickers else 0
    fmp_count = (n_profile_chunks if mktcaps else 0) + (1 if fmp_insider else 0)

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
```

Also update the `log.info` at the end of `run()`:
```python
    log.info(
        "Done in %.1fs — tickers=%d edgar=%d fmp_calls=%d congress=%d errors=%d → %s",
        duration, len(tickers), edgar_count, fmp_count, congress_count, errors, out,
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_pipeline_momentum.py -v
```

Expected: all 12 tests pass.

- [ ] **Step 5: Also run the congress tests to confirm no regressions**

```bash
pytest tests/test_congress_fetcher.py tests/test_pipeline_momentum.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_pipeline.py tests/test_pipeline_momentum.py
git commit -m "feat(pipeline): add momentum factor, extend EDGAR lookback to 180d, chunk FMP at 100"
```

---

## Task 4: Cross-Sectional Normalization in `generate_top_lists.py`

**Files:**
- Modify: `scripts/generate_top_lists.py`
- Create: `tests/test_cross_sectional.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cross_sectional.py`:

```python
"""tests/test_cross_sectional.py
Unit tests for cross-sectional factor normalization in generate_top_lists.

Markowitz (1990 Nobel) — portfolio construction requires comparable, bounded
signals. Validates that normalization produces peer-relative scores rather
than absolute thresholds, and that uniform factors don't crash or mislead.
"""
from __future__ import annotations

import numpy as np
import pytest

# Import the function under test — it will not exist yet
from scripts.generate_top_lists import _cross_sectional_normalize, FACTOR_FIELDS


def _make_results(n: int, overrides: dict | None = None) -> list:
    """Build n neutral result rows, optionally overriding specific fields."""
    base = {
        "edgar_score":    0.50,
        "insider_score":  0.50,
        "congress_score": 0.50,
        "news_score":     0.50,
        "momentum_score": 0.50,
    }
    rows = [{**base} for _ in range(n)]
    if overrides:
        for key, values in overrides.items():
            for i, v in enumerate(values):
                rows[i][key] = v
    return rows


class TestCrossSectionalNormalize:
    def test_higher_raw_score_gives_higher_normalized_score(self):
        results = _make_results(2, {"edgar_score": [0.30, 0.90]})
        normed  = _cross_sectional_normalize(results)
        assert normed[0]["edgar"] < normed[1]["edgar"]

    def test_normalized_scores_bounded_0_to_1(self):
        results = _make_results(10, {
            "edgar_score": np.random.default_rng(42).uniform(0.3, 0.9, 10).tolist()
        })
        normed = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_all_identical_scores_return_half(self):
        """When all tickers have the same raw score, normalized output is 0.5."""
        results = _make_results(5)   # all 0.50 by default
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert v == pytest.approx(0.5, abs=1e-4)

    def test_all_five_factors_present_in_output(self):
        results = _make_results(3)
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            assert set(row.keys()) == {"edgar", "insider", "congress", "news", "momentum"}

    def test_output_length_matches_input(self):
        results = _make_results(7)
        normed  = _cross_sectional_normalize(results)
        assert len(normed) == 7

    def test_single_ticker_returns_neutral(self):
        """One ticker — no peer comparison possible — returns 0.5 for all factors."""
        results = _make_results(1, {"edgar_score": [0.90]})
        normed  = _cross_sectional_normalize(results)
        assert normed[0]["edgar"] == pytest.approx(0.5, abs=1e-4)

    def test_congress_factor_key_renamed(self):
        """FACTOR_FIELDS maps congress_score → 'congress' (not 'macro')."""
        assert FACTOR_FIELDS.get("congress") == "congress_score"
        assert "macro" not in FACTOR_FIELDS

    def test_momentum_factor_present(self):
        assert FACTOR_FIELDS.get("momentum") == "momentum_score"

    def test_2008_crash_outlier_does_not_collapse_scores(self):
        """2020 COVID analog: one ticker with extreme momentum → others not collapsed to 0."""
        scores = [0.50] * 49 + [9999.0]   # 1 extreme outlier
        results = _make_results(50, {"momentum_score": scores})
        normed  = _cross_sectional_normalize(results)
        # The 49 normal tickers should not all map to near-zero
        normal_scores = [normed[i]["momentum"] for i in range(49)]
        assert max(normal_scores) > 0.30   # not all collapsed
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cross_sectional.py -v 2>&1 | head -20
```

Expected: `ImportError` — `_cross_sectional_normalize` and `FACTOR_FIELDS` don't exist yet.

- [ ] **Step 3: Implement changes in `scripts/generate_top_lists.py`**

**3a — Add numpy import and normalize_score import** at the top of the file after existing imports:

```python
import numpy as np

from regime_trader.scoring.normalize import normalize_score, fallback_reweight
```

**3b — Replace the `WEIGHTS` dict** (rename `macro` → `momentum`):

```python
WEIGHTS: Dict[str, float] = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "momentum": 0.10,   # renamed from macro
}
```

**3c — Add `FACTOR_FIELDS` mapping** right after `WEIGHTS`:

```python
# Maps factor key → field name in intel_source_status.json results
FACTOR_FIELDS: Dict[str, str] = {
    "edgar":    "edgar_score",
    "insider":  "insider_score",
    "congress": "congress_score",
    "news":     "news_score",
    "momentum": "momentum_score",
}
```

**3d — Add `_cross_sectional_normalize()` function** after the `_badge()` function:

```python
def _cross_sectional_normalize(results: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Markowitz (1990 Nobel) — normalize each factor cross-sectionally to [0, 1].

    For each of the five factors, winsorizes at the 5th/95th percentile and
    min-max scales across the full universe. A ticker with no peers (n=1) or
    all identical scores receives 0.5 (neutral) for that factor.

    $x_{norm,i} = \\frac{winsorize(x_i) - \\min}{\\max - \\min}$

    Args:
        results: List of raw result dicts from intel_source_status.json.

    Returns:
        One dict per result with keys matching FACTOR_FIELDS (edgar, insider,
        congress, news, momentum), values normalized to [0, 1].
    """
    n = len(results)
    if n == 0:
        return []

    normed_factors: Dict[str, np.ndarray] = {}
    for factor, field in FACTOR_FIELDS.items():
        raw = np.array([float(r.get(field, 0.0)) for r in results])
        if n == 1 or float(np.nanmax(raw)) == float(np.nanmin(raw)):
            # No variance — return neutral 0.5 for all entries
            normed_factors[factor] = np.full(n, 0.5)
        else:
            normed_factors[factor] = normalize_score(raw, lo_pct=5, hi_pct=95) / 100.0

    return [
        {f: round(float(normed_factors[f][i]), 4) for f in normed_factors}
        for i in range(n)
    ]
```

**3e — Refactor `_final_score()` and `_to_entry()`** to use pre-normalized factors:

Remove `_final_score()` entirely. Replace `_to_entry()` with a version that accepts normalized factors:

```python
def _to_entry(row: Dict[str, Any], norm_factors: Dict[str, float]) -> Dict[str, Any]:
    """Markowitz (1990 Nobel) — build a ranked-list entry from a normalized factor dict."""
    score = round(
        WEIGHTS["edgar"]    * norm_factors["edgar"] +
        WEIGHTS["insider"]  * norm_factors["insider"] +
        WEIGHTS["congress"] * norm_factors["congress"] +
        WEIGHTS["news"]     * norm_factors["news"] +
        WEIGHTS["momentum"] * norm_factors["momentum"],
        4,
    )
    return {
        "ticker":      row.get("ticker", "?"),
        "sector":      row.get("sector", "Unknown"),
        "cap_tier":    row.get("cap_tier", "large"),
        "market_cap":  float(row.get("market_cap", 0.0)),
        "final_score": score,
        "badge":       _badge(score),
        "ceo_buy":     bool(row.get("ceo_buy", False)),
        "form4_count": int(row.get("form4_count", 0)),
        "factors":     norm_factors,
    }
```

**3f — Update `generate()`** to call `_cross_sectional_normalize()`:

Replace the two lines:
```python
    entries = [_to_entry(row) for row in results]
    _assign_cap_tiers(entries)
```

With:
```python
    norm_factor_list = _cross_sectional_normalize(results)
    entries = [_to_entry(row, nf) for row, nf in zip(results, norm_factor_list)]
    _assign_cap_tiers(entries)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_cross_sectional.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass (the `test_normalize.py` tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_top_lists.py tests/test_cross_sectional.py
git commit -m "feat(scoring): add cross-sectional normalization and real momentum/congress factors"
```

---

## Task 5: Update Discord Factor Emoji + Run End-to-End Smoke Test

**Files:**
- Modify: `scripts/send_toplists_discord.py`

- [ ] **Step 1: Update `_FACTOR_EMOJI` and `_format_factor_line`**

In `scripts/send_toplists_discord.py`, replace the `_FACTOR_EMOJI` dict:

```python
_FACTOR_EMOJI = {
    "edgar":    "📋",
    "insider":  "🏦",
    "congress": "🏛️",
    "news":     "📰",
    "momentum": "📈",   # was "macro": "🌍"
}
```

And update the `_format_factor_line` function — change the tuple of factor keys:

```python
def _format_factor_line(factors: Dict[str, float]) -> str:
    """One compact line: 📋0.72 🏦0.90 🏛️0.50 📰0.65 📈0.58"""
    parts = []
    for key in ("edgar", "insider", "congress", "news", "momentum"):   # was "macro"
        v = factors.get(key, 0.50)
        parts.append(f"{_FACTOR_EMOJI[key]}`{v:.2f}`")
    return "  ".join(parts)
```

- [ ] **Step 2: Create a minimal smoke `top_lists.json`**

```bash
python -c "
import json, datetime, pathlib

top_lists = {
    'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
    'source_run_id': 'smoke-test',
    'ticker_count': 3,
    'weights': {'edgar': 0.30, 'insider': 0.25, 'congress': 0.20, 'news': 0.15, 'momentum': 0.10},
    'top_buys': [
        {
            'ticker': 'AAPL', 'final_score': 0.7200, 'badge': 'TACTICAL BUY',
            'ceo_buy': True,
            'factors': {'edgar': 0.72, 'insider': 0.85, 'congress': 0.60, 'news': 0.65, 'momentum': 0.70}
        },
        {
            'ticker': 'JPM', 'final_score': 0.6800, 'badge': 'TACTICAL BUY',
            'ceo_buy': False,
            'factors': {'edgar': 0.55, 'insider': 0.70, 'congress': 0.75, 'news': 0.60, 'momentum': 0.65}
        },
    ],
    'mid_caps': [],
    'small_caps': [],
}
pathlib.Path('logs').mkdir(exist_ok=True)
pathlib.Path('logs/top_lists_smoke.json').write_text(json.dumps(top_lists, indent=2))
print('Written logs/top_lists_smoke.json')
"
```

- [ ] **Step 3: Verify Discord payload renders correctly**

```bash
python scripts/send_toplists_discord.py --input logs/top_lists_smoke.json --dry-run
```

Expected output: a JSON payload containing `📈` (momentum emoji) in the factor line — confirm `momentum` key is used, not `macro`. Output should show embed with `📋`, `🏦`, `🏛️`, `📰`, `📈` emojis.

- [ ] **Step 4: Clean up smoke file and commit**

```bash
python -c "import pathlib; pathlib.Path('logs/top_lists_smoke.json').unlink(missing_ok=True)"
git add scripts/send_toplists_discord.py
git commit -m "feat(discord): update factor emoji — macro 🌍 replaced by momentum 📈"
```

---

## Final Validation

- [ ] **Run full test suite**

```bash
pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass including the five new test files.

- [ ] **Verify universe CSV line count**

```bash
python -c "
import csv
rows = list(csv.DictReader(open('config/universe.csv')))
print(f'{len(rows)} tickers across {len(set(r[\"sector\"] for r in rows))} sectors')
"
```

Expected: `165 tickers across 11 sectors`

- [ ] **Verify WEIGHTS consistency across scripts**

```bash
python -c "
from scripts.run_pipeline import WEIGHTS as W1
from scripts.generate_top_lists import WEIGHTS as W2
assert W1 == W2, f'WEIGHTS mismatch: {W1} vs {W2}'
assert abs(sum(W1.values()) - 1.0) < 1e-9
assert 'momentum' in W1
assert 'macro' not in W1
print('WEIGHTS consistent:', W1)
"
```

Expected: prints `WEIGHTS consistent: {'edgar': 0.30, 'insider': 0.25, 'congress': 0.20, 'news': 0.15, 'momentum': 0.10}`
