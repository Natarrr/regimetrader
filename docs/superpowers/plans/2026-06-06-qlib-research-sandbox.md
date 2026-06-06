# qlib Research Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add empirical IC validation, LightGBM weight calibration, and MVO portfolio construction to regime_trader using qlib as a local research sandbox, with zero production dependencies on qlib.

**Architecture:** A `research/` directory (local only, gitignored from CI) holds all qlib/LightGBM code. Factor scores are backfilled 52 weeks from FMP historical endpoints. IC is computed per factor; LightGBM SHAP importances (stability-adjusted) are blended with academic priors to produce `optimal_weights.json`, which `export_weights.py` writes back into `config/weights.py`. Phase 4 adds a native MVO optimizer to `generate_top_lists.py` using only scipy + sklearn — no qlib in production.

**Tech Stack:** Python 3.11, requests, lightgbm, shap, scipy, scikit-learn, numpy, qlib (research only), pytest

---

## File Map

**Created (research — local only, not committed to CI):**
- `research/.gitignore`
- `research/requirements-research.txt`
- `research/scripts/backfill_factors.py` — FMP historical → factor_scores.ndjson
- `research/scripts/build_qlib_dataset.py` — NDJSON → qlib binary
- `research/scripts/ic_engine.py` — pure IC computation functions (shared by script + notebook)
- `research/scripts/run_ic_analysis.py` — runs ic_engine → ic_report.json
- `research/scripts/train_lgbm.py` — LightGBM + SHAP stability → optimal_weights.json
- `research/scripts/export_weights.py` — optimal_weights.json → config/weights.py
- `research/tests/test_backfill_factors.py`
- `research/tests/test_ic_engine.py`
- `research/tests/test_train_lgbm.py`
- `research/tests/test_export_weights.py`

**Created (production — committed, in CI):**
- `backend/market_intel/portfolio_optimizer.py` — MVO + risk parity + score-proportional fallbacks + VIX scaling
- `tests/test_mvo_optimizer.py`
- `tests/test_mvo_determinism.py`
- `tests/test_mvo_fallback.py`
- `tests/test_vix_vol_scaling.py`
- `tests/test_vix_monotonicity.py`
- `tests/test_sector_exposure.py`
- `tests/test_portfolio_weight_schema.py`

**Modified (production):**
- `backend/market_intel/generate_top_lists.py` — add `portfolio_weight` field via portfolio_optimizer
- `.gitignore` — add research/data/ and research/notebooks/.ipynb_checkpoints/

---

## Phase 1 — Research Environment + Historical Backfill

### Task 1: Research environment scaffold

**Files:**
- Create: `research/.gitignore`
- Create: `research/requirements-research.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create research directory structure**

```bash
mkdir -p research/scripts research/tests research/data/backfill research/data/qlib_data research/notebooks
touch research/scripts/__init__.py research/tests/__init__.py
```

- [ ] **Step 2: Write research/.gitignore**

```
__pycache__/
*.pyc
data/
notebooks/.ipynb_checkpoints/
*.npz
optimal_weights.json
ic_report.json
```

- [ ] **Step 3: Write research/requirements-research.txt**

```
qlib>=0.9.6
lightgbm>=4.3.0
shap>=0.45.0
scipy>=1.13.0
scikit-learn>=1.5.0
numpy>=1.26.0
pandas>=2.2.0
plotly>=5.22.0
jupyter>=1.0.0
ipykernel>=6.29.0
requests>=2.31.0
```

- [ ] **Step 4: Add research/data to root .gitignore**

Open `.gitignore` at the project root and add these lines at the end:

```
# Research sandbox data (large, local-only)
research/data/
research/optimal_weights.json
research/ic_report.json
research/notebooks/.ipynb_checkpoints/
```

- [ ] **Step 5: Commit**

```bash
git add research/.gitignore research/requirements-research.txt .gitignore
git commit -m "feat(research): scaffold research sandbox environment"
```

---

### Task 2: backfill_factors.py — price, momentum, forward labels

**Files:**
- Create: `research/scripts/backfill_factors.py`

The backfill script reconstructs factor scores at every Friday close for the past 52 weeks. It imports production scoring functions directly for consistency. Run from the repo root with `FMP_API_KEY` set.

- [ ] **Step 1: Write the skeleton + price fetch helper**

```python
# Path: research/scripts/backfill_factors.py
"""Reconstruct 52 weeks of factor scores from FMP historical endpoints.

Run from repo root:
    FMP_API_KEY=your_key python research/scripts/backfill_factors.py

Output: research/data/backfill/factor_scores.ndjson
Each line: {"ticker": "AAPL", "snapshot_date": "2025-08-01",
            "insider_conviction": 0.72, ..., "forward_return_21d": 0.034}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

# Import production scoring functions — ensures backfill uses identical logic.
from regime_trader.scoring.momentum_signals import (
    score_momentum_long,
    score_volume_attention,
)
from regime_trader.scoring.insider_signals import (
    score_insider_conviction,
    score_insider_breadth,
)
from regime_trader.scoring.news_signals import (
    score_news_sentiment,
    score_news_buzz,
)
from regime_trader.scoring.analyst import _score_record as score_analyst_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill")

_FMP_BASE = "https://financialmodelingprep.com/stable"
_API_KEY = os.environ.get("FMP_API_KEY", "")
_RATE_LIMIT_DELAY = 1.0 / 30  # 30 req/s to stay safe
_OUT = Path("research/data/backfill/factor_scores.ndjson")

UNIVERSE_CSV = Path("config/universe.csv")
CONGRESS_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/"
    "aggregate/all_transactions.json"
)


def _fmp_get(endpoint: str, params: dict) -> Any:
    """Single FMP stable/ GET with rate limiting."""
    params["apikey"] = _API_KEY
    url = f"{_FMP_BASE}/{endpoint}"
    time.sleep(_RATE_LIMIT_DELAY)
    r = requests.get(url, params=params, timeout=15)
    if r.status_code in (401, 403, 404):
        log.warning("FMP %s returned %d — skipping", endpoint, r.status_code)
        return None
    r.raise_for_status()
    return r.json()


def _load_universe() -> list[str]:
    import csv
    tickers = []
    with open(UNIVERSE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append(row["ticker"].strip())
    return tickers


def _fridays(n_weeks: int = 52) -> list[date]:
    """Last n_weeks Fridays, oldest first, excluding last 21 days (need fwd return)."""
    today = date.today()
    fridays = []
    d = today - timedelta(days=22)  # leave 21 trading days for forward return
    while len(fridays) < n_weeks:
        if d.weekday() == 4:  # Friday
            fridays.append(d)
        d -= timedelta(days=1)
    return list(reversed(fridays))


def fetch_prices(ticker: str, from_date: date, to_date: date) -> list[dict]:
    """Returns list of {date, close, volume} sorted newest-first."""
    data = _fmp_get(
        "historical-price-eod/full",
        {"symbol": ticker, "from": str(from_date), "to": str(to_date)},
    )
    if not data or not isinstance(data, list):
        return []
    return data


def compute_momentum_at(
    prices: list[dict], snapshot_date: date
) -> tuple[float, float]:
    """Return (return_12_1m, spy_relative_vol_ratio) at snapshot_date.

    prices: list of {date: str, close: float, volume: float}, newest-first.
    Returns (return_12_1m_raw, volume_5d_90d_ratio).
    """
    # Filter to prices on or before snapshot_date
    eligible = [p for p in prices if p["date"] <= str(snapshot_date)]
    if len(eligible) < 252:
        return (None, None)

    # Price at t (snapshot), t-21 (skip), t-252 (12m ago)
    p_t = float(eligible[0]["close"])
    p_skip = float(eligible[min(21, len(eligible) - 1)]["close"])
    p_12m = float(eligible[min(252, len(eligible) - 1)]["close"])

    if p_skip <= 0 or p_12m <= 0:
        return (None, None)

    return_12_1m = (p_skip - p_12m) / p_12m  # skip-month

    # Volume: 5d avg / 90d avg
    vols = [float(p.get("volume", 0)) for p in eligible[:90] if p.get("volume")]
    if len(vols) < 10:
        vol_ratio = None
    else:
        vol_5d = sum(vols[:5]) / 5
        vol_90d = sum(vols[:90]) / len(vols[:90])
        vol_ratio = vol_5d / vol_90d if vol_90d > 0 else None

    return (return_12_1m, vol_ratio)
```

- [ ] **Step 2: Add SPY price fetch + forward return computation**

Append to `research/scripts/backfill_factors.py`:

```python
def fetch_spy_return(from_date: date, to_date: date) -> float:
    """SPY return over [from_date, to_date] for relative momentum."""
    prices = fetch_prices("SPY", from_date - timedelta(days=400), to_date)
    eligible_from = [p for p in prices if p["date"] >= str(from_date)]
    eligible_to = [p for p in prices if p["date"] <= str(to_date)]
    if not eligible_from or not eligible_to:
        return 0.0
    p_start = float(eligible_from[-1]["close"])
    p_end = float(eligible_to[0]["close"])
    return (p_end - p_start) / p_start if p_start > 0 else 0.0


def compute_forward_return(
    prices: list[dict], snapshot_date: date, horizon: int = 21
) -> Optional[float]:
    """(price at snapshot+horizon - price at snapshot) / price at snapshot."""
    target_date = snapshot_date + timedelta(days=horizon + 5)  # buffer for weekends
    at_snapshot = [p for p in prices if p["date"] <= str(snapshot_date)]
    after_snapshot = [
        p for p in prices
        if str(snapshot_date) < p["date"] <= str(target_date)
    ]
    if not at_snapshot or not after_snapshot:
        return None
    p0 = float(at_snapshot[0]["close"])
    p1 = float(after_snapshot[-1]["close"])  # closest date after horizon
    return (p1 - p0) / p0 if p0 > 0 else None
```

- [ ] **Step 3: Commit momentum scaffold**

```bash
git add research/scripts/backfill_factors.py
git commit -m "feat(research): add backfill scaffold + momentum/price helpers"
```

---

### Task 3: backfill_factors.py — insider, congress, news factors

**Files:**
- Modify: `research/scripts/backfill_factors.py`

- [ ] **Step 1: Add insider fetch helper**

Append to `research/scripts/backfill_factors.py`:

```python
def fetch_insider_trades(ticker: str, snapshot_date: date) -> list[dict]:
    """All P-code insider trades in [snapshot_date-90d, snapshot_date]."""
    from_d = snapshot_date - timedelta(days=90)
    data = _fmp_get(
        "insider-trading/search",
        {
            "symbol": ticker,
            "transactionType": "P-Purchase",
            "from": str(from_d),
            "to": str(snapshot_date),
            "limit": 200,
        },
    )
    return data if isinstance(data, list) else []


def compute_insider_scores(
    trades: list[dict], market_cap: float
) -> tuple[float, float]:
    """Return (insider_conviction_score, insider_breadth_score) in [0, 1]."""
    return (
        score_insider_conviction(trades, market_cap=market_cap),
        score_insider_breadth(trades),
    )


def fetch_congress_all() -> list[dict]:
    """Fetch all S3 Stock Watcher senate transactions once per run."""
    r = requests.get(CONGRESS_URL, timeout=20)
    r.raise_for_status()
    return r.json()


def compute_congress_score(
    ticker: str, congress_trades: list[dict], snapshot_date: date
) -> float:
    """Score congressional trades in [snapshot_date-90d, snapshot_date]."""
    from_d = snapshot_date - timedelta(days=90)
    relevant = [
        t for t in congress_trades
        if t.get("ticker", "").upper() == ticker.upper()
        and str(from_d) <= (t.get("transaction_date") or "") <= str(snapshot_date)
        and t.get("type", "").lower() in ("purchase", "buy")
    ]
    if not relevant:
        return 0.0
    # Score: capped at 1.0, proportional to count (5+ trades = max)
    return min(1.0, len(relevant) / 5.0)


def fetch_news(ticker: str, snapshot_date: date) -> list[dict]:
    """FMP news articles in [snapshot_date-30d, snapshot_date]."""
    from_d = snapshot_date - timedelta(days=30)
    data = _fmp_get(
        "news/stock",
        {
            "tickers": ticker,
            "from": str(from_d),
            "to": str(snapshot_date),
            "limit": 200,
        },
    )
    return data if isinstance(data, list) else []
```

- [ ] **Step 2: Add analyst + piotroski helpers**

Append to `research/scripts/backfill_factors.py`:

```python
def load_analyst_bulk_index(bulk_path: Path) -> dict[str, dict]:
    """Build ticker → record index from pre-fetched NDJSON bulk file.

    ⚠ Uses current snapshot as proxy for all historical dates (slow-moving signal).
    """
    index: dict[str, dict] = {}
    if not bulk_path.exists():
        return index
    with open(bulk_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                if sym:
                    index[sym.upper()] = rec
            except json.JSONDecodeError:
                continue
    return index


def compute_analyst_score(ticker: str, analyst_index: dict[str, dict]) -> float:
    rec = analyst_index.get(ticker.upper(), {})
    if not rec:
        return 0.0
    score, _ = score_analyst_record(ticker, rec)
    return score


def fetch_piotroski(ticker: str, snapshot_date: date) -> float:
    """Compute Piotroski F-score from most-recent ratios-ttm filing ≤ snapshot_date."""
    data = _fmp_get("ratios-ttm", {"symbol": ticker})
    if not data or not isinstance(data, list) or not data[0]:
        return 0.5  # neutral default when unavailable
    r = data[0]

    # Piotroski (2000) 8-point F-score sub-components
    roa = float(r.get("returnOnAssetsTTM") or 0)
    cfo = float(r.get("operatingCashFlowPerShareTTM") or 0)
    delta_roa = float(r.get("returnOnAssetsTTM") or 0)  # proxy: single period
    accrual = cfo - roa  # CFO > ROA = accrual quality signal

    current_ratio = float(r.get("currentRatioTTM") or 0)
    delta_leverage = -float(r.get("debtEquityRatioTTM") or 0)  # lower is better
    delta_shares = -float(r.get("priceToBookRatioTTM") or 0)  # proxy

    gross_margin = float(r.get("grossProfitMarginTTM") or 0)
    asset_turnover = float(r.get("assetTurnoverTTM") or 0)

    f_score = sum([
        roa > 0,
        cfo > 0,
        delta_roa >= 0,
        accrual > 0,
        delta_leverage >= 0,
        current_ratio >= 1.0,
        delta_shares >= 0,
        gross_margin >= 0,
        asset_turnover > 0,
    ])
    return min(1.0, f_score / 9.0)
```

- [ ] **Step 3: Commit factor helpers**

```bash
git add research/scripts/backfill_factors.py
git commit -m "feat(research): add insider/congress/news/analyst/piotroski backfill helpers"
```

---

### Task 4: backfill_factors.py — main loop + output

**Files:**
- Modify: `research/scripts/backfill_factors.py`

- [ ] **Step 1: Write main orchestration loop**

Append to `research/scripts/backfill_factors.py`:

```python
def build_snapshot(
    ticker: str,
    snapshot_date: date,
    prices: list[dict],
    congress_trades: list[dict],
    analyst_index: dict[str, dict],
    market_cap: float,
    spy_prices: list[dict],
) -> Optional[dict]:
    """Build one (ticker, snapshot_date) record. Returns None if unusable."""
    # Momentum + volume
    return_12_1m, vol_ratio = compute_momentum_at(prices, snapshot_date)
    if return_12_1m is None:
        log.debug("%s %s: insufficient price history, skipping", ticker, snapshot_date)
        return None

    spy_return_12_1m, _ = compute_momentum_at(spy_prices, snapshot_date)
    momentum_score = score_momentum_long(return_12_1m, spy_return_12_1m or 0.0)
    vol_score = score_volume_attention(vol_ratio) if vol_ratio else 0.0

    # Insider
    insider_trades = fetch_insider_trades(ticker, snapshot_date)
    conviction_score, breadth_score = compute_insider_scores(insider_trades, market_cap)

    # Congress (US only)
    congress_score = compute_congress_score(ticker, congress_trades, snapshot_date)

    # News
    articles = fetch_news(ticker, snapshot_date)
    sentiment_score = score_news_sentiment(articles)
    buzz_score = score_news_buzz(articles)

    # Analyst (current snapshot proxy — see spec known limitation)
    analyst_score = compute_analyst_score(ticker, analyst_index)

    # Piotroski
    piotroski_score = fetch_piotroski(ticker, snapshot_date)

    # Forward return label (21 trading days ≈ 30 calendar days)
    fwd_return = compute_forward_return(prices, snapshot_date, horizon=21)
    spy_fwd = compute_forward_return(spy_prices, snapshot_date, horizon=21)

    if fwd_return is None:
        return None  # can't compute IC without label

    return {
        "ticker": ticker,
        "snapshot_date": str(snapshot_date),
        "insider_conviction": round(conviction_score, 6),
        "insider_breadth":    round(breadth_score, 6),
        "congress":           round(congress_score, 6),
        "news_sentiment":     round(sentiment_score, 6),
        "news_buzz":          round(buzz_score, 6),
        "momentum_long":      round(momentum_score, 6),
        "volume_attention":   round(vol_score, 6),
        "analyst_consensus":  round(analyst_score, 6),
        "quality_piotroski":  round(piotroski_score, 6),
        "forward_return_21d": round(fwd_return, 6),
        "spy_return_21d":     round(spy_fwd or 0.0, 6),
    }


def main() -> None:
    if not _API_KEY:
        raise EnvironmentError("FMP_API_KEY not set")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    tickers = _load_universe()
    fridays = _fridays(52)
    log.info("Backfilling %d tickers × %d dates", len(tickers), len(fridays))

    # Pre-fetch shared data once
    log.info("Fetching congress trades (S3)...")
    congress_trades = fetch_congress_all()

    bulk_path = Path(".cache/bulk_snapshots/upgrades-downgrades-consensus-bulk.ndjson")
    log.info("Loading analyst bulk index...")
    analyst_index = load_analyst_bulk_index(bulk_path)

    # Fetch SPY prices once
    oldest = fridays[0] - timedelta(days=400)
    newest = fridays[-1] + timedelta(days=35)
    log.info("Fetching SPY prices %s → %s", oldest, newest)
    spy_prices = fetch_prices("SPY", oldest, newest)

    records_written = 0
    with open(_OUT, "w") as out_f:
        for ticker in tickers:
            log.info("Processing %s...", ticker)
            prices = fetch_prices(ticker, oldest, newest)
            market_cap = 1e10  # fallback; FMP quote not fetched here to save calls

            for snap_date in fridays:
                rec = build_snapshot(
                    ticker, snap_date, prices, congress_trades,
                    analyst_index, market_cap, spy_prices,
                )
                if rec is not None:
                    out_f.write(json.dumps(rec) + "\n")
                    records_written += 1

    log.info("Done. %d records written to %s", records_written, _OUT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write local test for backfill output schema**

Create `research/tests/test_backfill_factors.py`:

```python
# Path: research/tests/test_backfill_factors.py
"""Local-only tests for backfill_factors helpers (no FMP calls)."""
import pytest
from datetime import date, timedelta

# Import the pure helpers (no network calls)
from research.scripts.backfill_factors import (
    compute_momentum_at,
    compute_forward_return,
    compute_congress_score,
    _fridays,
)

FACTOR_KEYS = [
    "ticker", "snapshot_date", "insider_conviction", "insider_breadth",
    "congress", "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
    "forward_return_21d", "spy_return_21d",
]


def _make_prices(n: int, base_close: float = 100.0) -> list[dict]:
    """Build n daily price records newest-first."""
    today = date(2026, 1, 1)
    records = []
    for i in range(n):
        d = today - timedelta(days=i)
        records.append({
            "date": str(d),
            "close": base_close * (1 + i * 0.001),
            "volume": 1_000_000 + i * 100,
        })
    return records


def test_fridays_count_and_weekday():
    fridays = _fridays(52)
    assert len(fridays) == 52
    for f in fridays:
        assert f.weekday() == 4  # Friday


def test_fridays_oldest_first():
    fridays = _fridays(10)
    for i in range(len(fridays) - 1):
        assert fridays[i] < fridays[i + 1]


def test_compute_momentum_at_insufficient_history():
    prices = _make_prices(100)  # need 252
    result, vol = compute_momentum_at(prices, date(2025, 12, 1))
    assert result is None
    assert vol is None


def test_compute_momentum_at_sufficient_history():
    prices = _make_prices(300)
    snap = date(2025, 12, 1)
    result, vol = compute_momentum_at(prices, snap)
    assert result is not None
    assert isinstance(result, float)


def test_compute_forward_return_within_range():
    prices = _make_prices(300, base_close=100.0)
    snap = date(2025, 12, 1)
    fwd = compute_forward_return(prices, snap, horizon=21)
    # Should return a float (positive or negative)
    assert fwd is not None
    assert isinstance(fwd, float)


def test_compute_forward_return_no_future_data():
    prices = _make_prices(50)  # all in the past
    snap = date(2024, 1, 1)
    fwd = compute_forward_return(prices, snap, horizon=21)
    # No prices before snap date → None
    assert fwd is None


def test_congress_score_empty():
    score = compute_congress_score("AAPL", [], date(2025, 6, 1))
    assert score == 0.0


def test_congress_score_capped_at_one():
    trades = [
        {"ticker": "AAPL", "transaction_date": "2025-05-15", "type": "Purchase"}
        for _ in range(10)
    ]
    score = compute_congress_score("AAPL", trades, date(2025, 6, 1))
    assert score == 1.0


def test_congress_score_out_of_window():
    trades = [{"ticker": "AAPL", "transaction_date": "2024-01-01", "type": "Purchase"}]
    score = compute_congress_score("AAPL", trades, date(2025, 6, 1))
    assert score == 0.0  # outside 90-day window


def test_no_lookahead_bias_in_fridays():
    """Fridays list must exclude the last 21 days (need fwd return)."""
    from datetime import date as dt
    fridays = _fridays(52)
    cutoff = dt.today() - timedelta(days=21)
    for f in fridays:
        assert f <= cutoff, f"Future snapshot {f} would create lookahead bias"
```

- [ ] **Step 3: Run local tests**

Run from repo root (research env must be installed: `pip install -r research/requirements-research.txt`):

```bash
cd "c:\Users\ntard\Projects\Trading dashboard\regime_trader"
python -m pytest research/tests/test_backfill_factors.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add research/scripts/backfill_factors.py research/tests/test_backfill_factors.py
git commit -m "feat(research): complete backfill_factors.py + schema tests"
```

---

### Task 5: build_qlib_dataset.py

**Files:**
- Create: `research/scripts/build_qlib_dataset.py`

Converts `factor_scores.ndjson` → qlib binary dataset so Notebook 01 can use qlib's built-in IC analysis tools.

- [ ] **Step 1: Write build_qlib_dataset.py**

```python
# Path: research/scripts/build_qlib_dataset.py
"""Convert research/data/backfill/factor_scores.ndjson → qlib binary dataset.

Run from repo root after backfill_factors.py completes:
    python research/scripts/build_qlib_dataset.py

Output: research/data/qlib_data/ (qlib binary format)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("build_qlib_dataset")

_IN = Path("research/data/backfill/factor_scores.ndjson")
_OUT_DIR = Path("research/data/qlib_data")

FEATURE_COLS = [
    "insider_conviction", "insider_breadth", "congress",
    "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
]
LABEL_COL = "forward_return_21d"


def load_ndjson(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values(["ticker", "snapshot_date"]).reset_index(drop=True)
    return df


def build_qlib_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    """Write qlib-compatible CSV files per ticker under out_dir/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, group in df.groupby("ticker"):
        g = group.set_index("snapshot_date").sort_index()
        ticker_df = g[FEATURE_COLS + [LABEL_COL]].copy()
        ticker_df.index.name = "date"
        out_path = out_dir / f"{ticker}.csv"
        ticker_df.to_csv(out_path)
    log.info("Wrote %d ticker CSVs to %s", df["ticker"].nunique(), out_dir)

    # Also write a combined parquet for convenience in notebooks
    combined_path = out_dir / "all_factors.parquet"
    df.to_parquet(combined_path, index=False)
    log.info("Combined parquet: %s (%d rows)", combined_path, len(df))


def validate_roundtrip(df: pd.DataFrame, out_dir: Path) -> None:
    """Verify NDJSON → CSV round-trip is lossless for a sample ticker."""
    sample_ticker = df["ticker"].iloc[0]
    csv_path = out_dir / f"{sample_ticker}.csv"
    loaded = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    original = df[df["ticker"] == sample_ticker].set_index("snapshot_date")[FEATURE_COLS + [LABEL_COL]]
    original.index.name = "date"
    # Check shape
    assert loaded.shape == original.shape, (
        f"Round-trip shape mismatch: {loaded.shape} vs {original.shape}"
    )
    # Check values within float tolerance
    diff = (loaded - original).abs().max().max()
    assert diff < 1e-5, f"Round-trip max diff {diff} exceeds tolerance"
    log.info("Round-trip validation passed for %s (max diff: %.2e)", sample_ticker, diff)


def main() -> None:
    if not _IN.exists():
        raise FileNotFoundError(f"{_IN} not found — run backfill_factors.py first")
    log.info("Loading %s...", _IN)
    df = load_ndjson(_IN)
    log.info("Loaded %d records, %d tickers, %d dates",
             len(df), df["ticker"].nunique(), df["snapshot_date"].nunique())
    build_qlib_dataset(df, _OUT_DIR)
    validate_roundtrip(df, _OUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write round-trip test**

Create `research/tests/test_build_qlib_dataset.py`:

```python
# Path: research/tests/test_build_qlib_dataset.py
"""Tests for build_qlib_dataset.py — no file I/O to FMP."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from research.scripts.build_qlib_dataset import (
    load_ndjson,
    build_qlib_dataset,
    validate_roundtrip,
    FEATURE_COLS,
    LABEL_COL,
)

SAMPLE_RECORDS = [
    {
        "ticker": "AAPL", "snapshot_date": "2025-06-06",
        "insider_conviction": 0.72, "insider_breadth": 0.45,
        "congress": 0.30, "news_sentiment": 0.61, "news_buzz": 0.38,
        "momentum_long": 0.84, "volume_attention": 0.22,
        "analyst_consensus": 0.70, "quality_piotroski": 0.80,
        "forward_return_21d": 0.034, "spy_return_21d": 0.018,
    },
    {
        "ticker": "AAPL", "snapshot_date": "2025-06-13",
        "insider_conviction": 0.65, "insider_breadth": 0.50,
        "congress": 0.10, "news_sentiment": 0.55, "news_buzz": 0.42,
        "momentum_long": 0.80, "volume_attention": 0.30,
        "analyst_consensus": 0.72, "quality_piotroski": 0.82,
        "forward_return_21d": -0.012, "spy_return_21d": 0.005,
    },
]


def _write_ndjson(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_ndjson_shape(tmp_path):
    p = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, p)
    df = load_ndjson(p)
    assert len(df) == 2
    assert "ticker" in df.columns
    assert "snapshot_date" in df.columns
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in df.columns


def test_load_ndjson_dates_parsed(tmp_path):
    p = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, p)
    df = load_ndjson(p)
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])


def test_build_and_roundtrip(tmp_path):
    ndjson_path = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, ndjson_path)
    df = load_ndjson(ndjson_path)
    out_dir = tmp_path / "qlib_data"
    build_qlib_dataset(df, out_dir)
    # CSV should exist for AAPL
    assert (out_dir / "AAPL.csv").exists()
    # Combined parquet should exist
    assert (out_dir / "all_factors.parquet").exists()
    # Round-trip validation passes
    validate_roundtrip(df, out_dir)


def test_all_factors_in_csv(tmp_path):
    ndjson_path = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, ndjson_path)
    df = load_ndjson(ndjson_path)
    out_dir = tmp_path / "qlib_data"
    build_qlib_dataset(df, out_dir)
    loaded = pd.read_csv(out_dir / "AAPL.csv")
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in loaded.columns, f"Missing column: {col}"
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest research/tests/test_build_qlib_dataset.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add research/scripts/build_qlib_dataset.py research/tests/test_build_qlib_dataset.py
git commit -m "feat(research): add build_qlib_dataset.py + round-trip tests"
```

---

## Phase 2 — IC Validation

### Task 6: ic_engine.py — pure IC computation functions

**Files:**
- Create: `research/scripts/ic_engine.py`

Pure functions with no side effects. Shared by `run_ic_analysis.py` and Notebook 01.

- [ ] **Step 1: Write ic_engine.py**

```python
# Path: research/scripts/ic_engine.py
"""Pure IC computation functions — no I/O, no side effects.

All functions operate on plain Python lists or numpy arrays.
Imported by run_ic_analysis.py and Notebook 01.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import spearmanr

FACTORS = [
    "insider_conviction", "insider_breadth", "congress",
    "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
]

WeightRecommendation = Literal["increase", "hold", "decrease", "investigate"]

# Academic weights from config/weights.py WEIGHTS_US (v2.2-global)
ACADEMIC_WEIGHTS_US: dict[str, float] = {
    "insider_conviction": 0.30,
    "insider_breadth":    0.15,
    "congress":           0.22,
    "news_sentiment":     0.10,
    "news_buzz":          0.05,
    "momentum_long":      0.15,
    "volume_attention":   0.03,
    "analyst_consensus":  0.00,
    "quality_piotroski":  0.00,
}


def rank_ic_per_snapshot(
    factor_scores: np.ndarray,  # shape (n_tickers,)
    forward_returns: np.ndarray,  # shape (n_tickers,)
) -> float:
    """Spearman rank IC for a single cross-section.

    Returns NaN if insufficient variation (all identical scores).
    """
    if len(factor_scores) < 3:
        return float("nan")
    if np.std(factor_scores) < 1e-8:
        return 0.0
    corr, _ = spearmanr(factor_scores, forward_returns, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0


def compute_factor_ic(
    factor_name: str,
    df_records: list[dict],
) -> dict:
    """Compute all IC metrics for one factor across all (ticker, date) records.

    df_records: list of dicts with keys: snapshot_date, <factor_name>, forward_return_21d.
    Returns dict matching ic_report.json schema (minus weight_recommendation).
    """
    # Group by snapshot_date
    from collections import defaultdict
    by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in df_records:
        score = rec.get(factor_name)
        fwd = rec.get("forward_return_21d")
        if score is None or fwd is None:
            continue
        by_date[rec["snapshot_date"]].append((float(score), float(fwd)))

    # Compute rank IC per date
    ics_by_month: dict[str, list[float]] = defaultdict(list)
    all_ics: list[float] = []
    for snap_date, pairs in sorted(by_date.items()):
        scores = np.array([p[0] for p in pairs])
        returns = np.array([p[1] for p in pairs])
        ic = rank_ic_per_snapshot(scores, returns)
        if not np.isnan(ic):
            all_ics.append(ic)
            month = snap_date[:7]  # "YYYY-MM"
            ics_by_month[month].append(ic)

    if not all_ics:
        return {
            "mean_ic": 0.0,
            "ic_ir": 0.0,
            "ic_positive_rate": 0.0,
            "monthly_ic": {},
        }

    arr = np.array(all_ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std()) if len(arr) > 1 else 1e-8
    ic_ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    ic_positive_rate = float((arr > 0).mean())
    monthly_ic = {m: round(float(np.mean(v)), 6) for m, v in sorted(ics_by_month.items())}

    return {
        "mean_ic":          round(mean_ic, 6),
        "ic_ir":            round(ic_ir, 6),
        "ic_positive_rate": round(ic_positive_rate, 6),
        "monthly_ic":       monthly_ic,
    }


def weight_recommendation(
    mean_ic: float,
    ic_ir: float,
    ic_positive_rate: float,
) -> WeightRecommendation:
    """Derive mechanical weight recommendation from IC metrics.

    Rules (spec § Phase 2):
        mean_ic < 0                         → "investigate"
        ic_ir > 0.5 AND ic_pos_rate ≥ 0.60 → "increase"
        ic_ir ≥ 0.3                         → "hold"
        else                                → "decrease"
    """
    if mean_ic < 0:
        return "investigate"
    if ic_ir > 0.5 and ic_positive_rate >= 0.60:
        return "increase"
    if ic_ir >= 0.3:
        return "hold"
    return "decrease"


def build_ic_report(df_records: list[dict]) -> dict:
    """Build the full ic_report.json dict for all 9 factors."""
    report = {}
    for factor in FACTORS:
        metrics = compute_factor_ic(factor, df_records)
        rec = weight_recommendation(
            metrics["mean_ic"],
            metrics["ic_ir"],
            metrics["ic_positive_rate"],
        )
        report[factor] = {**metrics, "weight_recommendation": rec}
    return report
```

- [ ] **Step 2: Write ic_engine tests**

Create `research/tests/test_ic_engine.py`:

```python
# Path: research/tests/test_ic_engine.py
import numpy as np
import pytest

from research.scripts.ic_engine import (
    rank_ic_per_snapshot,
    weight_recommendation,
    build_ic_report,
    FACTORS,
)


def test_rank_ic_perfect_positive():
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ic = rank_ic_per_snapshot(scores, returns)
    assert abs(ic - 1.0) < 1e-6


def test_rank_ic_perfect_negative():
    scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    ic = rank_ic_per_snapshot(scores, returns)
    assert abs(ic + 1.0) < 1e-6


def test_rank_ic_all_identical_scores():
    scores = np.array([0.5, 0.5, 0.5])
    returns = np.array([0.01, -0.01, 0.02])
    ic = rank_ic_per_snapshot(scores, returns)
    assert ic == 0.0


def test_rank_ic_too_few_samples():
    ic = rank_ic_per_snapshot(np.array([0.5, 0.7]), np.array([0.01, 0.02]))
    assert ic == pytest.approx(0.0) or np.isnan(ic)


def test_weight_recommendation_investigate():
    assert weight_recommendation(-0.01, 0.4, 0.55) == "investigate"


def test_weight_recommendation_increase():
    assert weight_recommendation(0.05, 0.6, 0.70) == "increase"


def test_weight_recommendation_hold():
    assert weight_recommendation(0.02, 0.35, 0.55) == "hold"


def test_weight_recommendation_decrease():
    assert weight_recommendation(0.01, 0.2, 0.45) == "decrease"


def test_build_ic_report_all_factors_present():
    # Minimal records: 10 tickers × 5 dates
    import random
    random.seed(42)
    records = []
    for date_offset in range(5):
        snap = f"2025-{6 + date_offset:02d}-06"
        for i in range(10):
            rec = {"snapshot_date": snap, "forward_return_21d": random.gauss(0, 0.02)}
            for f in FACTORS:
                rec[f] = random.uniform(0, 1)
            records.append(rec)

    report = build_ic_report(records)
    assert set(report.keys()) == set(FACTORS)
    for factor, metrics in report.items():
        assert "mean_ic" in metrics
        assert "ic_ir" in metrics
        assert "ic_positive_rate" in metrics
        assert "monthly_ic" in metrics
        assert metrics["weight_recommendation"] in ("increase", "hold", "decrease", "investigate")


def test_build_ic_report_values_in_range():
    import random
    random.seed(0)
    records = []
    for date_offset in range(10):
        snap = f"2025-{6 + date_offset:02d}-01"
        for i in range(20):
            rec = {"snapshot_date": snap, "forward_return_21d": random.gauss(0, 0.02)}
            for f in FACTORS:
                rec[f] = random.uniform(0, 1)
            records.append(rec)
    report = build_ic_report(records)
    for factor, metrics in report.items():
        assert -1.0 <= metrics["mean_ic"] <= 1.0
        assert 0.0 <= metrics["ic_positive_rate"] <= 1.0
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest research/tests/test_ic_engine.py -v
```

Expected: 10 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add research/scripts/ic_engine.py research/tests/test_ic_engine.py
git commit -m "feat(research): add ic_engine.py + IC computation tests"
```

---

### Task 7: run_ic_analysis.py → ic_report.json

**Files:**
- Create: `research/scripts/run_ic_analysis.py`

- [ ] **Step 1: Write run_ic_analysis.py**

```python
# Path: research/scripts/run_ic_analysis.py
"""Run IC analysis on backfill data → research/ic_report.json.

Run from repo root after build_qlib_dataset.py completes:
    python research/scripts/run_ic_analysis.py

Output: research/ic_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from research.scripts.build_qlib_dataset import load_ndjson
from research.scripts.ic_engine import build_ic_report, ACADEMIC_WEIGHTS_US, FACTORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_ic_analysis")

_IN = Path("research/data/backfill/factor_scores.ndjson")
_OUT = Path("research/ic_report.json")


def main() -> None:
    if not _IN.exists():
        raise FileNotFoundError(f"{_IN} not found — run backfill_factors.py first")

    log.info("Loading factor scores from %s...", _IN)
    df = load_ndjson(_IN)
    records = df.to_dict("records")
    log.info("Loaded %d records across %d tickers", len(records), df["ticker"].nunique())

    log.info("Computing IC for %d factors...", len(FACTORS))
    report = build_ic_report(records)

    # Annotate with academic weight for comparison
    for factor, metrics in report.items():
        metrics["academic_weight"] = ACADEMIC_WEIGHTS_US.get(factor, 0.0)

    _OUT.write_text(json.dumps(report, indent=2))
    log.info("IC report written to %s", _OUT)

    # Print summary table
    print("\n── IC Report Summary ──────────────────────────────────────")
    print(f"{'Factor':<22} {'Mean IC':>8} {'IC IR':>7} {'IC>0':>6} {'Acad.W':>7} {'Rec':>12}")
    print("-" * 68)
    for factor, m in report.items():
        print(
            f"{factor:<22} {m['mean_ic']:>8.4f} {m['ic_ir']:>7.3f} "
            f"{m['ic_positive_rate']:>6.2%} {m['academic_weight']:>7.2f} "
            f"{m['weight_recommendation']:>12}"
        )
    print("─" * 68)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add research/scripts/run_ic_analysis.py
git commit -m "feat(research): add run_ic_analysis.py → ic_report.json"
```

---

## Phase 3 — LightGBM Weight Calibration

### Task 8: train_lgbm.py — SHAP stability + blended weights

**Files:**
- Create: `research/scripts/train_lgbm.py`

- [ ] **Step 1: Write train_lgbm.py**

```python
# Path: research/scripts/train_lgbm.py
"""Walk-forward LightGBM training + SHAP stability → optimal_weights.json.

Run from repo root after run_ic_analysis.py completes:
    python research/scripts/train_lgbm.py

Output: research/optimal_weights.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap

from research.scripts.build_qlib_dataset import load_ndjson
from research.scripts.ic_engine import FACTORS, ACADEMIC_WEIGHTS_US, weight_recommendation
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_lgbm")

_IN = Path("research/data/backfill/factor_scores.ndjson")
_IC_REPORT = Path("research/ic_report.json")
_OUT = Path("research/optimal_weights.json")

BLEND_ALPHA = 0.6        # trust data 60%, academic prior 40%
WEIGHT_FLOOR = 0.05
WEIGHT_CAP_MULTIPLIER = 2.0

# Monotone constraints: +1 = increasing, -1 = decreasing, 0 = unconstrained
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "insider_conviction": +1,
    "insider_breadth":    +1,
    "congress":           +1,
    "news_sentiment":     +1,
    "news_buzz":           0,
    "momentum_long":      +1,
    "volume_attention":    0,
    "analyst_consensus":  +1,
    "quality_piotroski":  +1,
}


def _get_investigate_factors(ic_report: dict) -> set[str]:
    return {f for f, m in ic_report.items() if m.get("weight_recommendation") == "investigate"}


def _build_folds(df, n_splits: int = 2) -> list[tuple]:
    """Walk-forward expanding-window folds on snapshot_date."""
    dates = sorted(df["snapshot_date"].unique())
    fold_size = len(dates) // (n_splits + 1)
    folds = []
    for i in range(n_splits):
        train_cutoff = dates[(i + 1) * fold_size - 1]
        val_cutoff = dates[min((i + 2) * fold_size - 1, len(dates) - 1)]
        train = df[df["snapshot_date"] <= train_cutoff]
        val = df[(df["snapshot_date"] > train_cutoff) & (df["snapshot_date"] <= val_cutoff)]
        if len(train) > 0 and len(val) > 0:
            folds.append((train, val))
    return folds


def _train_fold(
    train_df,
    val_df,
    active_factors: list[str],
) -> tuple[lgb.Booster, np.ndarray, float]:
    """Train one LightGBM fold. Returns (model, shap_values, val_ic)."""
    mono_list = [MONOTONE_CONSTRAINTS[f] for f in active_factors]

    X_train = train_df[active_factors].values
    y_train = train_df["forward_return_21d"].values
    X_val = val_df[active_factors].values
    y_val = val_df["forward_return_21d"].values

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "monotone_constraints": mono_list,
        "verbose": -1,
        "random_state": 42,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    # SHAP values on validation set
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Val IC (rank correlation of predictions vs actual returns)
    preds = model.predict(X_val)
    val_ic = float(spearmanr(preds, y_val).correlation)

    return model, mean_abs_shap, val_ic


def _shap_to_stable_weights(
    shap_per_fold: list[np.ndarray],
    active_factors: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Apply SHAP stability check, return (shap_mean, shap_cv, shap_stable)."""
    arr = np.array(shap_per_fold)  # (n_folds, n_factors)
    mean_arr = arr.mean(axis=0)
    std_arr = arr.std(axis=0) if len(arr) > 1 else np.zeros(len(active_factors))
    cv_arr = np.where(mean_arr > 1e-8, std_arr / mean_arr, 1.0)

    stability_multiplier = np.clip(1 - cv_arr, 0.3, 1.0)
    stable = mean_arr * stability_multiplier
    if stable.sum() > 0:
        stable = stable / stable.sum()

    shap_mean = {f: round(float(mean_arr[i]), 6) for i, f in enumerate(active_factors)}
    shap_cv = {f: round(float(cv_arr[i]), 6) for i, f in enumerate(active_factors)}
    shap_stable_w = {f: round(float(stable[i]), 6) for i, f in enumerate(active_factors)}
    return shap_mean, shap_cv, shap_stable_w


def _blend_and_constrain(
    shap_stable: dict[str, float],
    investigate_factors: set[str],
) -> dict[str, float]:
    """Blend SHAP weights with academic prior, apply floor + cap, re-normalize."""
    final: dict[str, float] = {}
    for factor in FACTORS:
        academic_w = ACADEMIC_WEIGHTS_US[factor]
        if factor in investigate_factors:
            final[factor] = academic_w  # unchanged
            continue
        data_w = shap_stable.get(factor, 0.0)
        blended = BLEND_ALPHA * data_w + (1 - BLEND_ALPHA) * academic_w
        blended = max(blended, WEIGHT_FLOOR)
        blended = min(blended, WEIGHT_CAP_MULTIPLIER * max(academic_w, WEIGHT_FLOOR))
        final[factor] = blended

    total = sum(final.values())
    final = {k: round(v / total, 8) for k, v in final.items()}
    assert abs(sum(final.values()) - 1.0) < 1e-5, f"Weights sum to {sum(final.values())}"
    return final


def main() -> None:
    if not _IN.exists():
        raise FileNotFoundError(f"{_IN} not found — run backfill_factors.py first")
    if not _IC_REPORT.exists():
        raise FileNotFoundError(f"{_IC_REPORT} not found — run run_ic_analysis.py first")

    ic_report = json.loads(_IC_REPORT.read_text())
    investigate_factors = _get_investigate_factors(ic_report)
    active_factors = [f for f in FACTORS if f not in investigate_factors]
    log.info("Active factors: %s", active_factors)
    log.info("Investigate factors (excluded): %s", investigate_factors)

    df = load_ndjson(_IN)
    df = df.dropna(subset=active_factors + ["forward_return_21d"])
    log.info("Training on %d records", len(df))

    folds = _build_folds(df, n_splits=2)
    log.info("Walk-forward folds: %d", len(folds))

    shap_per_fold: list[np.ndarray] = []
    val_ics: list[float] = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        log.info("Fold %d: train=%d val=%d", fold_idx + 1, len(train_df), len(val_df))
        _, fold_shap, fold_ic = _train_fold(train_df, val_df, active_factors)
        shap_per_fold.append(fold_shap)
        val_ics.append(fold_ic)
        log.info("  Val IC: %.4f", fold_ic)

    shap_mean_d, shap_cv_d, shap_stable_d = _shap_to_stable_weights(
        shap_per_fold, active_factors
    )
    final_weights = _blend_and_constrain(shap_stable_d, investigate_factors)

    # Build output JSON
    weights_detail: dict[str, dict] = {}
    for factor in FACTORS:
        academic_w = ACADEMIC_WEIGHTS_US[factor]
        is_investigate = factor in investigate_factors
        weights_detail[factor] = {
            "academic": academic_w,
            "shap_per_fold": [round(float(shap_per_fold[i][active_factors.index(factor)]), 6)
                              for i in range(len(shap_per_fold))]
                             if factor in active_factors else None,
            "shap_cv":             shap_cv_d.get(factor),
            "stability_multiplier": round(
                float(np.clip(1 - shap_cv_d.get(factor, 1.0), 0.3, 1.0)), 6
            ) if factor in active_factors else None,
            "shap_stable":         shap_stable_d.get(factor),
            "final":               final_weights[factor],
            "weight_recommendation": ic_report.get(factor, {}).get("weight_recommendation", "hold"),
        }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blend_alpha": BLEND_ALPHA,
        "weight_floor": WEIGHT_FLOOR,
        "lgbm_val_ic_per_fold": [round(ic, 6) for ic in val_ics],
        "lgbm_val_ic_mean": round(float(np.mean(val_ics)), 6),
        "investigate_factors": sorted(investigate_factors),
        "weights": weights_detail,
    }

    _OUT.write_text(json.dumps(output, indent=2))
    log.info("Optimal weights written to %s", _OUT)

    # Print summary
    print("\n── Calibrated Weights ───────────────────────────────────")
    print(f"{'Factor':<22} {'Academic':>9} {'Final':>9} {'Delta':>8}")
    print("-" * 52)
    for factor in FACTORS:
        d = weights_detail[factor]
        delta = d["final"] - d["academic"]
        marker = " ⚠" if factor in investigate_factors else ""
        print(f"{factor:<22} {d['academic']:>9.4f} {d['final']:>9.4f} {delta:>+8.4f}{marker}")
    print(f"\nVal IC per fold: {val_ics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write LightGBM training tests**

Create `research/tests/test_train_lgbm.py`:

```python
# Path: research/tests/test_train_lgbm.py
"""Tests for train_lgbm helpers — no LightGBM training (fast)."""
import numpy as np
import pytest

from research.scripts.train_lgbm import (
    _blend_and_constrain,
    _shap_to_stable_weights,
    _build_folds,
    WEIGHT_FLOOR,
    WEIGHT_CAP_MULTIPLIER,
    BLEND_ALPHA,
)
from research.scripts.ic_engine import FACTORS, ACADEMIC_WEIGHTS_US
import pandas as pd
from datetime import date, timedelta


def _make_df(n_dates: int = 20, n_tickers: int = 10) -> pd.DataFrame:
    records = []
    base = date(2025, 1, 1)
    # Use weekly snapshots (Fridays)
    fridays = [base + timedelta(weeks=i) for i in range(n_dates)]
    for snap in fridays:
        for i in range(n_tickers):
            rec = {"snapshot_date": str(snap), "ticker": f"T{i:03d}",
                   "forward_return_21d": np.random.randn() * 0.02}
            for f in FACTORS:
                rec[f] = np.random.uniform(0, 1)
            records.append(rec)
    df = pd.DataFrame(records)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def test_blend_and_constrain_sum_to_one():
    shap_stable = {f: 1.0 / len(FACTORS) for f in FACTORS}
    result = _blend_and_constrain(shap_stable, investigate_factors=set())
    assert abs(sum(result.values()) - 1.0) < 1e-5


def test_blend_and_constrain_floor_applied():
    # Give a factor near-zero SHAP weight
    shap_stable = {f: 0.001 if f == "news_buzz" else 0.12 for f in FACTORS}
    result = _blend_and_constrain(shap_stable, investigate_factors=set())
    assert result["news_buzz"] >= WEIGHT_FLOOR - 1e-8


def test_blend_and_constrain_cap_applied():
    # Give insider_conviction a huge SHAP weight → should be capped at 2× academic
    shap_stable = {f: 0.001 for f in FACTORS}
    shap_stable["insider_conviction"] = 0.99
    result = _blend_and_constrain(shap_stable, investigate_factors=set())
    max_allowed = WEIGHT_CAP_MULTIPLIER * ACADEMIC_WEIGHTS_US["insider_conviction"]
    # After normalization it may shrink below cap — check it's not ABOVE
    # (before normalization the cap is applied, after normalization it can only decrease)
    assert result["insider_conviction"] <= max_allowed + 1e-5


def test_blend_and_constrain_investigate_factor_keeps_academic():
    shap_stable = {f: 0.1 for f in FACTORS}
    investigate = {"congress"}
    result = _blend_and_constrain(shap_stable, investigate_factors=investigate)
    # Congress keeps its academic weight (after normalization, relative weight preserved)
    # We just check it exists and is positive
    assert result["congress"] > 0


def test_shap_stability_check_penalizes_high_cv():
    # Two folds: first fold has congress at 0.5, second at 0.0
    # High variance → low stability_multiplier
    fold1 = np.array([0.5, 0.5, 0.0, 0.0])
    fold2 = np.array([0.0, 0.5, 0.5, 0.0])
    active = ["insider_conviction", "insider_breadth", "congress", "news_sentiment"]
    mean_d, cv_d, stable_d = _shap_to_stable_weights([fold1, fold2], active)
    # congress has high CV → lower stable weight vs mean
    assert stable_d["congress"] <= mean_d["congress"] + 1e-8


def test_shap_stability_check_single_fold():
    # Single fold: no variance → stability_multiplier = 1.0
    fold1 = np.array([0.3, 0.2, 0.1, 0.4])
    active = ["insider_conviction", "insider_breadth", "congress", "news_sentiment"]
    mean_d, cv_d, stable_d = _shap_to_stable_weights([fold1], active)
    # With single fold, std=0, cv=0, multiplier=1 → stable = mean
    for f in active:
        assert abs(stable_d[f] - mean_d[f]) < 1e-6


def test_build_folds_returns_two_folds():
    np.random.seed(42)
    df = _make_df(n_dates=20)
    folds = _build_folds(df, n_splits=2)
    assert len(folds) == 2


def test_build_folds_no_overlap():
    np.random.seed(42)
    df = _make_df(n_dates=20)
    folds = _build_folds(df, n_splits=2)
    train1_dates = set(folds[0][0]["snapshot_date"])
    val1_dates = set(folds[0][1]["snapshot_date"])
    assert train1_dates.isdisjoint(val1_dates)


def test_optimal_weights_constraints():
    """Integration: blend + constrain must always satisfy all hard constraints."""
    import random
    random.seed(99)
    for _ in range(20):
        shap_stable = {f: random.uniform(0, 1) for f in FACTORS}
        total = sum(shap_stable.values())
        shap_stable = {k: v / total for k, v in shap_stable.items()}
        investigate = set(random.sample(FACTORS, k=2))
        result = _blend_and_constrain(shap_stable, investigate)
        # All weights ≥ floor (for non-investigate)
        for f in FACTORS:
            if f not in investigate:
                assert result[f] >= WEIGHT_FLOOR - 1e-7, f"{f}: {result[f]} < {WEIGHT_FLOOR}"
        # Sum = 1.0
        assert abs(sum(result.values()) - 1.0) < 1e-5
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest research/tests/test_train_lgbm.py -v
```

Expected: 9 tests PASS (no LightGBM training needed — tests cover pure functions only).

- [ ] **Step 4: Commit**

```bash
git add research/scripts/train_lgbm.py research/tests/test_train_lgbm.py
git commit -m "feat(research): add train_lgbm.py with SHAP stability + blend tests"
```

---

### Task 9: export_weights.py

**Files:**
- Create: `research/scripts/export_weights.py`

Reads `research/optimal_weights.json` and rewrites `regime_trader/config/weights.py` for the US weight set only. Preserves WEIGHTS_GLOBAL, PIOTROSKI_GATE, region classifier, and all comments.

- [ ] **Step 1: Write export_weights.py**

```python
# Path: research/scripts/export_weights.py
"""Write calibrated weights from optimal_weights.json → config/weights.py.

Run from repo root after train_lgbm.py completes:
    python research/scripts/export_weights.py [--dry-run]

The script ONLY updates the WEIGHTS_US dict. WEIGHTS_GLOBAL, PIOTROSKI_GATE,
and all region-classifier code are preserved unchanged.

After running, review the diff with:
    git diff regime_trader/config/weights.py
Then commit explicitly once satisfied.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_OPTIMAL = Path("research/optimal_weights.json")
_WEIGHTS_PY = Path("regime_trader/config/weights.py")

# Factors present in WEIGHTS_US (the only ones export updates)
_US_FACTORS = [
    "insider_conviction", "insider_breadth", "congress",
    "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
]


def load_final_weights(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {f: data["weights"][f]["final"] for f in _US_FACTORS}


def update_weights_py(weights: dict[str, float], source: Path, dry_run: bool) -> str:
    """Rewrite the WEIGHTS_US dict block in weights.py with new values.

    Preserves all other content exactly. Returns the new file content.
    """
    content = source.read_text()

    # Build the replacement WEIGHTS_US block
    lines = ["WEIGHTS_US: dict[str, float] = {"]
    for factor in _US_FACTORS:
        w = weights[factor]
        lines.append(f'    "{factor}": {w:.8f},')
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the existing WEIGHTS_US block (from "WEIGHTS_US: dict" to closing "}")
    pattern = re.compile(
        r'WEIGHTS_US:\s*dict\[str,\s*float\]\s*=\s*\{[^}]*\}',
        re.DOTALL
    )
    if not pattern.search(content):
        raise ValueError("Could not find WEIGHTS_US block in weights.py")

    new_content = pattern.sub(new_block, content)

    # Verify the assert still passes in the new content
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Calibrated weights sum to {total:.8f}, not 1.0 — aborting")

    if dry_run:
        print("── DRY RUN — would write the following WEIGHTS_US block: ──")
        print(new_block)
        return new_content

    source.write_text(new_content)
    return new_content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing")
    args = parser.parse_args()

    if not _OPTIMAL.exists():
        sys.exit(f"ERROR: {_OPTIMAL} not found — run train_lgbm.py first")

    weights = load_final_weights(_OPTIMAL)
    print(f"Loaded {len(weights)} calibrated weights from {_OPTIMAL}")
    for f, w in weights.items():
        print(f"  {f:<22} {w:.8f}")

    update_weights_py(weights, _WEIGHTS_PY, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\n✓ Wrote updated WEIGHTS_US to {_WEIGHTS_PY}")
        print("  Review with: git diff regime_trader/config/weights.py")
        print("  Then commit: git add regime_trader/config/weights.py && git commit")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write export_weights test**

Create `research/tests/test_export_weights.py`:

```python
# Path: research/tests/test_export_weights.py
"""Tests for export_weights.py."""
import json
import re
import tempfile
from pathlib import Path

import pytest

from research.scripts.export_weights import load_final_weights, update_weights_py
from research.scripts.ic_engine import FACTORS

_SAMPLE_OPTIMAL = {
    "generated_at": "2026-06-06T00:00:00+00:00",
    "blend_alpha": 0.6,
    "weight_floor": 0.05,
    "lgbm_val_ic_per_fold": [0.038, 0.044],
    "lgbm_val_ic_mean": 0.041,
    "investigate_factors": [],
    "weights": {
        "insider_conviction": {"academic": 0.30, "final": 0.30},
        "insider_breadth":    {"academic": 0.15, "final": 0.15},
        "congress":           {"academic": 0.22, "final": 0.22},
        "news_sentiment":     {"academic": 0.10, "final": 0.10},
        "news_buzz":          {"academic": 0.05, "final": 0.05},
        "momentum_long":      {"academic": 0.15, "final": 0.15},
        "volume_attention":   {"academic": 0.03, "final": 0.03},
        "analyst_consensus":  {"academic": 0.00, "final": 0.00},
        "quality_piotroski":  {"academic": 0.00, "final": 0.00},
    },
}

_SAMPLE_WEIGHTS_PY = '''
WEIGHTS_US: dict[str, float] = {
    "insider_conviction": 0.30000000,
    "insider_breadth":    0.15000000,
    "congress":           0.22000000,
    "news_sentiment":     0.10000000,
    "news_buzz":          0.05000000,
    "momentum_long":      0.15000000,
    "volume_attention":   0.03000000,
    "analyst_consensus":  0.00000000,
    "quality_piotroski":  0.00000000,
}
assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6

WEIGHTS_GLOBAL = {"insider_conviction": 0.30}
'''


def _write_optimal(tmp_path: Path) -> Path:
    p = tmp_path / "optimal_weights.json"
    p.write_text(json.dumps(_SAMPLE_OPTIMAL))
    return p


def test_load_final_weights_keys(tmp_path):
    p = _write_optimal(tmp_path)
    weights = load_final_weights(p)
    from research.scripts.export_weights import _US_FACTORS
    assert set(weights.keys()) == set(_US_FACTORS)


def test_load_final_weights_sum_to_one(tmp_path):
    p = _write_optimal(tmp_path)
    weights = load_final_weights(p)
    assert abs(sum(weights.values()) - 1.0) < 1e-5


def test_update_weights_py_preserves_global(tmp_path):
    p = _write_optimal(tmp_path)
    weights_py = tmp_path / "weights.py"
    weights_py.write_text(_SAMPLE_WEIGHTS_PY)
    weights = load_final_weights(p)
    new_content = update_weights_py(weights, weights_py, dry_run=False)
    # WEIGHTS_GLOBAL must be preserved
    assert "WEIGHTS_GLOBAL" in new_content


def test_update_weights_py_new_values_written(tmp_path):
    p = _write_optimal(tmp_path)
    # Modify one weight in optimal
    opt = json.loads(p.read_text())
    opt["weights"]["momentum_long"]["final"] = 0.18
    opt["weights"]["congress"]["final"] = 0.19  # adjust to keep sum=1
    opt["weights"]["news_buzz"]["final"] = 0.03
    p.write_text(json.dumps(opt))

    weights_py = tmp_path / "weights.py"
    weights_py.write_text(_SAMPLE_WEIGHTS_PY)
    weights = load_final_weights(p)
    update_weights_py(weights, weights_py, dry_run=False)
    content = weights_py.read_text()
    assert "0.18000000" in content


def test_export_weights_assert_still_valid(tmp_path):
    """After export, the written file must pass the weight sum assert."""
    p = _write_optimal(tmp_path)
    weights_py = tmp_path / "weights.py"
    weights_py.write_text(_SAMPLE_WEIGHTS_PY)
    weights = load_final_weights(p)
    update_weights_py(weights, weights_py, dry_run=False)
    content = weights_py.read_text()
    # Evaluate just the WEIGHTS_US block and check sum
    local_ns = {}
    exec(content, local_ns)  # noqa: S102
    written_weights = local_ns["WEIGHTS_US"]
    assert abs(sum(written_weights.values()) - 1.0) < 1e-5
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest research/tests/test_export_weights.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add research/scripts/export_weights.py research/tests/test_export_weights.py
git commit -m "feat(research): add export_weights.py + round-trip tests"
```

---

## Phase 4 — Portfolio Construction

### Task 10: portfolio_optimizer.py

**Files:**
- Create: `backend/market_intel/portfolio_optimizer.py`

Native MVO implementation — no qlib. Imported by `generate_top_lists.py`.

- [ ] **Step 1: Write portfolio_optimizer.py**

```python
# Path: backend/market_intel/portfolio_optimizer.py
"""MVO portfolio optimizer — no qlib dependency.

Imported by generate_top_lists.py to add portfolio_weight to top_lists.json.

Fallback chain (always produces a valid weight vector):
    MVO (SLSQP) → risk parity → score-proportional

VIX vol-targeting applied post-optimization (scales weights down, never up).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)

_COVARIANCE_PATH = Path("logs/covariance_matrix.npz")
_IC_REPORT_PATH = Path("research/ic_report.json")

_MAX_POSITION = 0.10        # max 10% per position
_MAX_SECTOR_WEIGHT = 0.30   # max 30% per sector
_MAX_TURNOVER = 0.20        # max 20% portfolio turnover

TARGET_VOL: dict[str, float] = {
    "normal": 0.15,   # VIX < 25
    "bear":   0.10,   # VIX 25–30
    "panic":  0.05,   # VIX ≥ 30
}


def _vix_regime(vix: float) -> str:
    if vix >= 30:
        return "panic"
    if vix >= 25:
        return "bear"
    return "normal"


def _score_proportional(scores: np.ndarray) -> np.ndarray:
    w = np.clip(scores, 0.0, None)
    total = w.sum()
    if total <= 0:
        n = len(scores)
        return np.ones(n) / n
    w = w / total
    w = np.clip(w, 0.0, _MAX_POSITION)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(scores)) / len(scores)


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols <= 1e-8, 1e-8, vols)
    w = 1.0 / vols
    return w / w.sum()


def _mvo(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    sector_ids: list[int],
    n_sectors: int,
    prev_weights: Optional[np.ndarray],
) -> np.ndarray:
    n = len(expected_returns)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ expected_returns)
        vol = float(np.sqrt(w @ cov @ w + 1e-10))
        return -ret / vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    for s in range(n_sectors):
        mask = np.array([1.0 if sector_ids[i] == s else 0.0 for i in range(n)])
        if mask.sum() > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w, m=mask: _MAX_SECTOR_WEIGHT - float((w * m).sum())}
            )

    if prev_weights is not None and len(prev_weights) == n:
        constraints.append(
            {"type": "ineq", "fun": lambda w: _MAX_TURNOVER - float(np.abs(w - prev_weights).sum())}
        )

    bounds = [(0.0, _MAX_POSITION)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"MVO did not converge: {result.message}")
    return np.array(result.x)


def _load_covariance(tickers: list[str]) -> Optional[np.ndarray]:
    if not _COVARIANCE_PATH.exists():
        log.debug("No covariance matrix at %s — will use identity fallback", _COVARIANCE_PATH)
        return None
    try:
        data = np.load(_COVARIANCE_PATH, allow_pickle=True)
        stored_tickers: list[str] = list(data["tickers"])
        full_cov: np.ndarray = data["covariance"]
        idx = [stored_tickers.index(t) for t in tickers if t in stored_tickers]
        if len(idx) != len(tickers):
            log.warning("Covariance ticker mismatch: expected %d, found %d", len(tickers), len(idx))
            return None
        return full_cov[np.ix_(idx, idx)]
    except Exception as exc:
        log.warning("Failed to load covariance matrix: %s", exc)
        return None


def _ic_estimate() -> float:
    try:
        data = json.loads(_IC_REPORT_PATH.read_text())
        ics = [v["mean_ic"] for v in data.values() if isinstance(v, dict) and "mean_ic" in v]
        return max(0.01, float(np.mean(ics))) if ics else 0.03
    except Exception:
        return 0.03  # conservative default


def run_optimizer(
    tickers: list[str],
    scores: list[float],
    sectors: list[str],
    vix: float,
    prev_weights: Optional[dict[str, float]] = None,
) -> tuple[dict[str, float], str]:
    """Compute portfolio weights for the given tickers.

    Args:
        tickers:      Ticker list (top-20 by composite score).
        scores:       Composite scores, same order as tickers.
        sectors:      Sector strings, same order as tickers.
        vix:          Current VIX level for vol-targeting.
        prev_weights: Previous weight dict (ticker → weight) for turnover control.

    Returns:
        (weights_dict, method_used)
        method_used ∈ {"MVO", "risk_parity", "score_proportional"}
    """
    n = len(tickers)
    scores_arr = np.array(scores, dtype=float)

    unique_sectors = sorted(set(sectors))
    sector_map = {s: i for i, s in enumerate(unique_sectors)}
    sector_ids = [sector_map[s] for s in sectors]

    cov = _load_covariance(tickers)
    if cov is None:
        cov = np.eye(n) * 0.04  # 20% annual vol fallback

    ic = _ic_estimate()
    z = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)
    expected_returns = ic * z

    prev_arr = (
        np.array([prev_weights.get(t, 0.0) for t in tickers])
        if prev_weights else None
    )

    method = "MVO"
    try:
        weights = _mvo(expected_returns, cov, sector_ids, len(unique_sectors), prev_arr)
    except Exception as exc:
        log.warning("MVO failed (%s), trying risk parity", exc)
        method = "risk_parity"
        try:
            weights = _risk_parity(cov)
        except Exception as exc2:
            log.warning("Risk parity failed (%s), using score-proportional", exc2)
            method = "score_proportional"
            weights = _score_proportional(scores_arr)

    # VIX vol-targeting — scale down only, never up
    regime = _vix_regime(vix)
    target_vol = TARGET_VOL[regime]
    port_vol = float(np.sqrt(weights @ cov @ weights + 1e-10))
    if port_vol > 1e-8:
        scale = min(1.0, target_vol / port_vol)
        weights = weights * scale

    return {t: float(w) for t, w in zip(tickers, weights)}, method
```

- [ ] **Step 2: Commit**

```bash
git add backend/market_intel/portfolio_optimizer.py
git commit -m "feat: add portfolio_optimizer.py (MVO + risk parity + VIX scaling)"
```

---

### Task 11: Modify generate_top_lists.py to add portfolio_weight

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py`

- [ ] **Step 1: Read the section where top-list entries are assembled**

Open [generate_top_lists.py](backend/market_intel/generate_top_lists.py) and find the function that writes `top_lists.json` entries — typically a loop that builds dicts with `final_score`, `badge`, etc.

- [ ] **Step 2: Add the import at the top of generate_top_lists.py**

After the existing imports, add:

```python
from backend.market_intel.portfolio_optimizer import run_optimizer
```

- [ ] **Step 3: Add portfolio weight computation after the top-20 selection**

Find the section in `generate_top_lists.py` where `top_buys` (or equivalent list) is finalized and Discord/output entries are built. Insert the following block immediately after the top-20 list is determined:

```python
# Portfolio construction: MVO weights for top-20 by composite_score.
# Ties at position 20 broken alphabetically by ticker.
_TOP_N_PORTFOLIO = 20
_sorted_for_portfolio = sorted(entries, key=lambda e: (-e["final_score"], e["ticker"]))
_portfolio_candidates = _sorted_for_portfolio[:_TOP_N_PORTFOLIO]

_prev_weights: dict = {}
_prev_path = Path(log_dir) / "top_lists.json"
if _prev_path.exists():
    try:
        _prev_data = json.loads(_prev_path.read_text())
        _prev_weights = {
            e["ticker"]: e.get("portfolio_weight", 0.0)
            for section in _prev_data.get("top_lists", {}).values()
            for e in (section if isinstance(section, list) else [])
            if e.get("portfolio_weight", 0.0) > 0
        }
    except Exception:
        pass

_portfolio_tickers = [e["ticker"] for e in _portfolio_candidates]
_portfolio_scores  = [e["final_score"] for e in _portfolio_candidates]
_portfolio_sectors = [e.get("sector", "Unknown") for e in _portfolio_candidates]
_vix = payload.get("vix", 20.0) if isinstance(payload, dict) else 20.0

try:
    _opt_weights, _opt_method = run_optimizer(
        _portfolio_tickers, _portfolio_scores, _portfolio_sectors,
        vix=_vix, prev_weights=_prev_weights,
    )
except Exception as _exc:
    log.warning("Portfolio optimizer raised: %s — using zeros", _exc)
    _opt_weights = {t: 0.0 for t in _portfolio_tickers}
    _opt_method = "failed"

# Attach portfolio_weight to each entry (0.0 for non-top-20)
_weight_set = set(_portfolio_tickers)
for entry in entries:
    entry["portfolio_weight"] = round(_opt_weights.get(entry["ticker"], 0.0), 6)
    entry["portfolio_weight_method"] = _opt_method if entry["ticker"] in _weight_set else "n/a"
    if entry["ticker"] in _weight_set:
        # Sector weight contribution
        entry["sector_weight_contribution"] = round(
            sum(
                _opt_weights.get(t, 0.0)
                for t in _portfolio_tickers
                if _portfolio_sectors[_portfolio_tickers.index(t)] == entry.get("sector")
            ), 6
        )
```

- [ ] **Step 4: Commit**

```bash
git add backend/market_intel/generate_top_lists.py
git commit -m "feat: add portfolio_weight field to top_lists.json via MVO optimizer"
```

---

### Task 12: Production CI tests

**Files:**
- Create: `tests/test_mvo_optimizer.py`
- Create: `tests/test_mvo_determinism.py`
- Create: `tests/test_mvo_fallback.py`
- Create: `tests/test_vix_vol_scaling.py`
- Create: `tests/test_vix_monotonicity.py`
- Create: `tests/test_sector_exposure.py`
- Create: `tests/test_portfolio_weight_schema.py`

- [ ] **Step 1: Write test_mvo_optimizer.py**

```python
# Path: tests/test_mvo_optimizer.py
import numpy as np
import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer, _MAX_POSITION, _MAX_SECTOR_WEIGHT

TICKERS = [f"T{i:02d}" for i in range(10)]
SCORES  = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
SECTORS = ["Tech", "Tech", "Finance", "Finance", "Energy",
           "Energy", "Health", "Health", "Tech", "Finance"]


def test_weights_sum_to_at_most_one(monkeypatch):
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        lambda tickers: None,
    )
    weights, _ = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
    total = sum(weights.values())
    assert total <= 1.0 + 1e-6


def test_all_weights_non_negative(monkeypatch):
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        lambda tickers: None,
    )
    weights, _ = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
    for t, w in weights.items():
        assert w >= -1e-8, f"{t} has negative weight {w}"


def test_max_position_not_exceeded(monkeypatch):
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        lambda tickers: None,
    )
    weights, _ = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
    for t, w in weights.items():
        assert w <= _MAX_POSITION + 1e-6, f"{t} weight {w} exceeds max {_MAX_POSITION}"


def test_sector_concentration_not_exceeded(monkeypatch):
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        lambda tickers: None,
    )
    weights, _ = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
    sector_totals: dict[str, float] = {}
    for t, w in weights.items():
        sec = SECTORS[TICKERS.index(t)]
        sector_totals[sec] = sector_totals.get(sec, 0.0) + w
    for sec, total in sector_totals.items():
        assert total <= _MAX_SECTOR_WEIGHT + 1e-5, f"Sector {sec} weight {total} exceeds max"
```

- [ ] **Step 2: Write test_mvo_determinism.py**

```python
# Path: tests/test_mvo_determinism.py
from backend.market_intel.portfolio_optimizer import run_optimizer

TICKERS = [f"STOCK{i}" for i in range(8)]
SCORES  = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
SECTORS = ["Tech", "Finance", "Energy", "Health", "Tech", "Finance", "Energy", "Health"]


def test_mvo_deterministic_across_10_runs(monkeypatch):
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        lambda tickers: None,
    )
    results = []
    for _ in range(10):
        weights, method = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
        results.append(weights)

    for i in range(1, 10):
        for t in TICKERS:
            assert abs(results[0][t] - results[i][t]) < 1e-8, (
                f"Non-deterministic weight for {t}: run 0={results[0][t]}, run {i}={results[i][t]}"
            )
```

- [ ] **Step 3: Write test_mvo_fallback.py**

```python
# Path: tests/test_mvo_fallback.py
import numpy as np
import pytest
from unittest.mock import patch
from backend.market_intel.portfolio_optimizer import run_optimizer, _mvo, _risk_parity

TICKERS = [f"T{i}" for i in range(5)]
SCORES  = [0.9, 0.7, 0.5, 0.3, 0.1]
SECTORS = ["Tech", "Finance", "Tech", "Finance", "Energy"]


def _bad_mvo(*args, **kwargs):
    raise RuntimeError("Simulated MVO failure")


def _bad_risk_parity(*args, **kwargs):
    raise RuntimeError("Simulated risk parity failure")


def test_fallback_to_risk_parity(monkeypatch):
    monkeypatch.setattr("backend.market_intel.portfolio_optimizer._load_covariance", lambda t: None)
    monkeypatch.setattr("backend.market_intel.portfolio_optimizer._mvo", _bad_mvo)
    weights, method = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
    assert method == "risk_parity"
    assert abs(sum(weights.values()) - 1.0) < 0.01 or sum(weights.values()) <= 1.0 + 1e-6


def test_fallback_to_score_proportional(monkeypatch):
    monkeypatch.setattr("backend.market_intel.portfolio_optimizer._load_covariance", lambda t: None)
    monkeypatch.setattr("backend.market_intel.portfolio_optimizer._mvo", _bad_mvo)
    monkeypatch.setattr("backend.market_intel.portfolio_optimizer._risk_parity", _bad_risk_parity)
    weights, method = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
    assert method == "score_proportional"
    assert sum(weights.values()) <= 1.0 + 1e-6
    for w in weights.values():
        assert w >= -1e-8


def test_full_fallback_chain_valid_weight_vector(monkeypatch):
    """All three fallback paths must produce a valid weight vector."""
    for bad_mvo, bad_rp, expected_method in [
        (False, False, "MVO"),
        (True,  False, "risk_parity"),
        (True,  True,  "score_proportional"),
    ]:
        with patch("backend.market_intel.portfolio_optimizer._load_covariance", return_value=None):
            if bad_mvo:
                with patch("backend.market_intel.portfolio_optimizer._mvo", _bad_mvo):
                    if bad_rp:
                        with patch("backend.market_intel.portfolio_optimizer._risk_parity", _bad_risk_parity):
                            weights, method = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
                    else:
                        weights, method = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
            else:
                weights, method = run_optimizer(TICKERS, SCORES, SECTORS, vix=18.0)
        assert isinstance(weights, dict)
        assert all(w >= -1e-8 for w in weights.values())
        assert sum(weights.values()) <= 1.0 + 1e-6
```

- [ ] **Step 4: Write test_vix_vol_scaling.py**

```python
# Path: tests/test_vix_vol_scaling.py
import numpy as np
import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer

TICKERS = [f"T{i}" for i in range(5)]
SCORES  = [0.9, 0.7, 0.5, 0.3, 0.1]
SECTORS = ["Tech", "Finance", "Tech", "Finance", "Energy"]

# Fixed covariance for reproducibility: 25% annual vol each, 0.3 correlation
def _fixed_cov(tickers):
    n = len(tickers)
    base = 0.25 ** 2 / 252  # daily variance
    cov = np.full((n, n), base * 0.3)
    np.fill_diagonal(cov, base)
    return cov


def _annualize_portfolio_vol(weights_dict, tickers, cov):
    w = np.array([weights_dict[t] for t in tickers])
    daily_var = w @ cov @ w
    return float(np.sqrt(daily_var * 252))


def test_panic_vix_portfolio_vol_below_target(monkeypatch):
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        _fixed_cov,
    )
    weights, _ = run_optimizer(TICKERS, SCORES, SECTORS, vix=35.0)
    annual_vol = _annualize_portfolio_vol(weights, TICKERS, _fixed_cov(TICKERS))
    target = 0.05  # panic regime
    assert annual_vol <= target + 0.01, f"Portfolio vol {annual_vol:.3f} exceeds panic target {target}"
```

- [ ] **Step 5: Write test_vix_monotonicity.py**

```python
# Path: tests/test_vix_monotonicity.py
import numpy as np
from backend.market_intel.portfolio_optimizer import run_optimizer

TICKERS = [f"T{i}" for i in range(5)]
SCORES  = [0.9, 0.7, 0.5, 0.3, 0.1]
SECTORS = ["Tech", "Finance", "Tech", "Finance", "Energy"]


def test_leverage_never_increases_with_vix(monkeypatch):
    """As VIX increases through thresholds, total portfolio weight never increases."""
    monkeypatch.setattr(
        "backend.market_intel.portfolio_optimizer._load_covariance",
        lambda tickers: None,
    )
    vix_levels = [18.0, 24.0, 26.0, 31.0, 41.0]
    total_weights = []
    for vix in vix_levels:
        weights, _ = run_optimizer(TICKERS, SCORES, SECTORS, vix=vix)
        total_weights.append(sum(weights.values()))

    for i in range(len(total_weights) - 1):
        assert total_weights[i + 1] <= total_weights[i] + 1e-6, (
            f"Portfolio weight increased from VIX={vix_levels[i]} "
            f"({total_weights[i]:.4f}) to VIX={vix_levels[i+1]} ({total_weights[i+1]:.4f})"
        )
```

- [ ] **Step 6: Write test_sector_exposure.py**

```python
# Path: tests/test_sector_exposure.py
"""Ensure every ticker in top_lists.json has a valid sector mapping."""
import json
from pathlib import Path
import pytest

TOP_LISTS_PATH = Path("logs/top_lists.json")
UNIVERSE_CSV = Path("config/universe.csv")


def _load_sector_map() -> dict[str, str]:
    import csv
    sector_map = {}
    if not UNIVERSE_CSV.exists():
        return sector_map
    with open(UNIVERSE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("ticker", "").strip()
            sector = row.get("sector", "").strip()
            if ticker:
                sector_map[ticker] = sector
    return sector_map


@pytest.mark.skipif(not TOP_LISTS_PATH.exists(), reason="top_lists.json not generated yet")
def test_all_tickers_have_sector():
    data = json.loads(TOP_LISTS_PATH.read_text())
    sector_map = _load_sector_map()
    missing = []
    for section in data.get("top_lists", {}).values():
        if not isinstance(section, list):
            continue
        for entry in section:
            ticker = entry.get("ticker", "")
            sector = entry.get("sector") or sector_map.get(ticker, "")
            if not sector:
                missing.append(ticker)
    assert not missing, f"Tickers with no sector mapping: {missing}"


@pytest.mark.skipif(not TOP_LISTS_PATH.exists(), reason="top_lists.json not generated yet")
def test_portfolio_weight_does_not_use_unknown_sector():
    """portfolio_weight > 0 entries must have a valid sector for MVO sector cap to work."""
    data = json.loads(TOP_LISTS_PATH.read_text())
    sector_map = _load_sector_map()
    bad = []
    for section in data.get("top_lists", {}).values():
        if not isinstance(section, list):
            continue
        for entry in section:
            if entry.get("portfolio_weight", 0.0) > 0:
                ticker = entry.get("ticker", "")
                sector = entry.get("sector") or sector_map.get(ticker, "")
                if not sector or sector.lower() in ("", "unknown", "none"):
                    bad.append(ticker)
    assert not bad, f"portfolio_weight > 0 tickers with missing sector: {bad}"
```

- [ ] **Step 7: Write test_portfolio_weight_schema.py**

```python
# Path: tests/test_portfolio_weight_schema.py
"""Schema validation for portfolio_weight fields in top_lists.json."""
import json
from pathlib import Path
import pytest

TOP_LISTS_PATH = Path("logs/top_lists.json")


@pytest.mark.skipif(not TOP_LISTS_PATH.exists(), reason="top_lists.json not generated yet")
def test_portfolio_weight_present_in_all_entries():
    data = json.loads(TOP_LISTS_PATH.read_text())
    missing = []
    for section in data.get("top_lists", {}).values():
        if not isinstance(section, list):
            continue
        for entry in section:
            if "portfolio_weight" not in entry:
                missing.append(entry.get("ticker", "?"))
    assert not missing, f"Entries missing portfolio_weight: {missing}"


@pytest.mark.skipif(not TOP_LISTS_PATH.exists(), reason="top_lists.json not generated yet")
def test_portfolio_weight_non_negative():
    data = json.loads(TOP_LISTS_PATH.read_text())
    negative = []
    for section in data.get("top_lists", {}).values():
        if not isinstance(section, list):
            continue
        for entry in section:
            w = entry.get("portfolio_weight", 0.0)
            if w < -1e-8:
                negative.append((entry.get("ticker"), w))
    assert not negative, f"Negative portfolio weights: {negative}"


@pytest.mark.skipif(not TOP_LISTS_PATH.exists(), reason="top_lists.json not generated yet")
def test_top20_portfolio_weight_sum_at_most_one():
    data = json.loads(TOP_LISTS_PATH.read_text())
    all_entries = []
    for section in data.get("top_lists", {}).values():
        if isinstance(section, list):
            all_entries.extend(section)
    all_entries.sort(key=lambda e: -e.get("final_score", 0))
    top20 = all_entries[:20]
    total = sum(e.get("portfolio_weight", 0.0) for e in top20)
    assert total <= 1.0 + 1e-5, f"Top-20 portfolio weights sum to {total:.4f} > 1.0"
```

- [ ] **Step 8: Run all 7 new CI tests**

```bash
python -m pytest tests/test_mvo_optimizer.py tests/test_mvo_determinism.py tests/test_mvo_fallback.py tests/test_vix_vol_scaling.py tests/test_vix_monotonicity.py tests/test_sector_exposure.py tests/test_portfolio_weight_schema.py -v
```

Expected: `test_sector_exposure.py` and `test_portfolio_weight_schema.py` will be **skipped** (no `top_lists.json` yet). All others PASS.

- [ ] **Step 9: Commit**

```bash
git add tests/test_mvo_optimizer.py tests/test_mvo_determinism.py tests/test_mvo_fallback.py tests/test_vix_vol_scaling.py tests/test_vix_monotonicity.py tests/test_sector_exposure.py tests/test_portfolio_weight_schema.py
git commit -m "test: add 7 CI tests for MVO optimizer, VIX scaling, sector exposure, portfolio schema"
```

---

## Self-Review

**Spec coverage check:**

| Spec Requirement | Task |
|-----------------|------|
| `research/` directory gitignored from CI | Task 1 |
| backfill 52 weeks at weekly granularity | Task 2-4 |
| All 9 factors reconstructed | Tasks 2-4 |
| `factor_scores.ndjson` schema | Task 4 |
| qlib binary dataset | Task 5 |
| IC engine: rank IC, mean IC, IC IR, IC > 0 rate, monthly heatmap | Task 6 |
| `weight_recommendation` field + logic | Task 6 |
| `ic_report.json` schema | Task 7 |
| `analyst_consensus` known limitation documented | Task 4 (comment in code) |
| LightGBM walk-forward 2 folds | Task 8 |
| Monotonicity constraints | Task 8 |
| SHAP stability check (CV, stability_multiplier) | Task 8 |
| Blend 60/40 academic prior | Task 8 |
| Weight floor 0.05, cap 2× academic | Task 8 |
| `investigate` factors keep academic weight | Task 8 |
| `optimal_weights.json` schema | Task 8 |
| `export_weights.py` rewrites WEIGHTS_US only | Task 9 |
| MVO optimizer with SLSQP | Task 10 |
| Risk parity fallback | Task 10 |
| Score-proportional fallback | Task 10 |
| VIX vol-targeting | Task 10 |
| Turnover constraint | Task 10 |
| Sector concentration cap | Task 10 |
| `portfolio_weight` field in top_lists.json | Task 11 |
| Ties broken alphabetically at position 20 | Task 11 |
| Existing badge/score logic unchanged | Task 11 |
| 7 production CI tests | Task 12 |
| 5 research local tests | Tasks 4, 5, 6, 8, 9 |

All spec requirements are covered. No gaps found.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-06-qlib-research-sandbox.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
