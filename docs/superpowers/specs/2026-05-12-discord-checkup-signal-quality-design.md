# Discord Market Checkup — Signal Quality & Universe Expansion Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two dead scoring factors (congress hardcoded 0.50, macro uniform VIX) with real congressional trading data and price momentum; expand the ticker universe from 50 to 165 tickers balanced across all 11 GICS sectors; apply cross-sectional normalization so the Top 5 Buys reflect genuine peer-relative conviction.

**Architecture:** Three existing scripts each keep their single responsibility. Changes are confined to `run_pipeline.py` (fetch two new data sources, extend EDGAR lookback), `generate_top_lists.py` (add cross-sectional normalization), and `send_toplists_discord.py` (update one emoji label). A new `config/universe.csv` replaces `config/top50.csv`.

**Tech Stack:** Python 3.11 · yfinance (price data, news) · EDGAR daily index (Form 4) · FMP REST API (insider buys, profile batch) · House/Senate Stock Watcher public S3 feeds (congressional trades, no API key) · `regime_trader/scoring/normalize.py` (winsorize + cross-sectional normalization, already exists)

---

## 1. Factor Table

| Key | Weight | Source | Lookback | Change from current |
|---|---|---|---|---|
| `edgar` | 0.30 | EDGAR daily index — Form 4 filing count | **180 days** (was 90) | Longer lookback for thin-filing tickers |
| `insider` | 0.25 | FMP `/stable/insider-trading` global fetch | 60 days | Unchanged |
| `congress` | 0.20 | House Stock Watcher + Senate Stock Watcher S3 | 90 days | **Was hardcoded 0.50** — now real data |
| `news` | 0.15 | yfinance `.news` headline keyword scoring | Live | Unchanged |
| `momentum` | 0.10 | yfinance 20-day price return | 20 days | **Was uniform VIX** — now ticker-specific |

All five factors are cross-sectionally normalized across the full 165-ticker universe before the weighted sum is computed. This means every factor score reflects peer-relative standing (0 = lowest in universe, 1 = highest), not an arbitrary absolute threshold.

---

## 2. New Data Sources

### 2a. Congressional Trading — House + Senate Stock Watcher

Both feeds are public S3 JSON arrays maintained by [housestockwatcher.com](https://housestockwatcher.com) and [senatestockwatcher.com](https://senatestockwatcher.com). No API key required.

**Endpoints:**
```
House:  https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json
Senate: https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json
```

**Key fields used:** `ticker`, `transaction_date`, `type` (`purchase` / `sale`), `amount` (range string e.g. `"$15,001 - $50,000"`).

**Scoring logic per ticker:**
```
purchases = count of purchase transactions in last 90 days
sales     = count of sale transactions in last 90 days
raw_score = (purchases - sales) / (purchases + sales + 1)   ∈ (-1, 1)
```
Shifted to [0, 1] before cross-sectional normalization: `score = (raw_score + 1) / 2`.

**Caching:** Both feeds are fetched once per run and written to `.cache/congress_cache.json` with a 24-hour TTL. Subsequent intra-day calls read from cache.

**Implementation:** New function `fetch_congress_buys(lookback_days=90) -> Dict[str, Dict]` in `scripts/run_pipeline.py`. Returns the same structure as `fetch_fmp_insider_buys()` — keyed by ticker, usable in `_score_ticker()`.

### 2b. Momentum — yfinance 20-day price return

```python
def fetch_price_data(ticker: str) -> Dict[str, float]:
    df = yf.download(ticker, period="1mo", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 5:
        return {"return_20d": 0.0}
    close = df["Close"].squeeze().dropna()
    ret = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
    return {"return_20d": ret}
```

Called per-ticker inside the existing `ThreadPoolExecutor`. Raw return is stored in `intel_source_status.json`; cross-sectional normalization in `generate_top_lists.py` converts it to [0, 1].

---

## 3. Cross-Sectional Normalization

`generate_top_lists.py` currently applies absolute thresholds (e.g. `edgar_score = 0.30 + count × 0.12`). After this change it normalizes each factor across the full universe before weighting.

**Algorithm (in `generate_top_lists.py`):**
```python
from regime_trader.scoring.normalize import normalize_score
import numpy as np

# Field names in intel_source_status.json["results"] entries
FACTOR_FIELDS = {
    "edgar":    "edgar_score",
    "insider":  "insider_score",
    "congress": "congress_score",
    "news":     "news_score",
    "momentum": "momentum_score",   # renamed from macro_score
}

# Collect raw vectors across all 165 result rows
raw = {
    f: np.array([float(r.get(field, 0.0)) for r in results])
    for f, field in FACTOR_FIELDS.items()
}

# Normalize each to [0, 1] with 5th/95th percentile winsorization
normed = {f: normalize_score(raw[f], lo_pct=5, hi_pct=95) / 100.0 for f in raw}

# Build entries using normalized scores
for i, row in enumerate(results):
    entry["factors"] = {f: round(float(normed[f][i]), 4) for f in normed}
```

`run_pipeline.py` writes pre-normalization scores using the existing field names (`edgar_score`, `insider_score`, `congress_score`, `news_score`, `momentum_score`). No format change to `intel_source_status.json` is required — `generate_top_lists.py` reads these fields directly as the raw cross-sectional inputs.

---

## 4. Universe — 165 Tickers, 11 GICS Sectors

File: `config/universe.csv` (replaces `config/top50.csv`). Columns: `ticker, sector, cap_tier`.

| Sector | Count | Tickers |
|---|---|---|
| Communication Services | 15 | META GOOGL NFLX DIS CHTR CMCSA T VZ TMUS EA WBD FOXA IPG OMC TTD |
| Consumer Discretionary | 15 | AMZN TSLA HD NKE MCD TGT LOW SBUX GM BKNG ORLY MAR RCL ULTA ROST |
| Consumer Staples | 15 | WMT PG KO PEP COST PM MO CL GIS ADM HSY STZ CHD HRL SYY |
| Energy | 15 | XOM CVX COP SLB EOG MPC PSX VLO OXY HAL DVN BKR FANG HES MRO |
| Financials | 15 | JPM BAC WFC GS MS BLK SPGI CB AXP PNC USB TRV MET AFL PRU |
| Healthcare | 15 | JNJ LLY ABBV UNH MRK PFE ABT DHR AMGN TMO ISRG CVS MDT SYK BMY |
| Industrials | 15 | CAT HON GE BA UPS RTX LMT DE MMM EMR ETN FDX WM CARR PCAR |
| Information Technology | 15 | AAPL MSFT NVDA ORCL CRM NOW ADBE AMD QCOM IBM INTC TXN AMAT MU ACN |
| Materials | 15 | LIN APD ECL NEM FCX NUE ALB CF MOS PPG RPM IP PKG DOW IFF |
| Real Estate | 12 | AMT PLD CCI EQIX PSA O DLR SPG VICI AVB EQR WY |
| Utilities | 13 | NEE DUK SO D AEP EXC XEL PCG WEC ETR PPL CMS ES |

---

## 5. FMP Budget

| Call | Tickers | Runs/day | Calls/day |
|---|---|---|---|
| Profile batch (chunk 1: 100 tickers) | 100 | 3 | 3 |
| Profile batch (chunk 2: 65 tickers) | 65 | 3 | 3 |
| Insider trading global fetch | all | 3 | 3 |
| **Total** | | | **9 / 200** |

Congressional data comes from free public S3 — zero FMP calls.

---

## 6. File Changes

| File | What changes |
|---|---|
| `config/universe.csv` | New file — 165 tickers replacing `config/top50.csv` |
| `scripts/run_pipeline.py` | Add `fetch_congress_buys()`, `fetch_price_data()`, `score_congress()`, `score_momentum()`; EDGAR lookback 90→180; WEIGHTS key `macro`→`momentum`; FMP profile batch in chunks of 100; write `_raw_<factor>` keys to output |
| `scripts/generate_top_lists.py` | Add cross-sectional `normalize_score()` per factor; read `_raw_<factor>` from status JSON |
| `scripts/send_toplists_discord.py` | Factor emoji map: 🌍 `macro` → 📈 `momentum`; label update only |
| `.github/workflows/edgar_3x.yml` | Default `tickers_file` → `config/universe.csv` |

No new files. No new Python dependencies (yfinance and requests are already installed).

---

## 7. Error Handling & Fallbacks

- **Congress feed unavailable:** `fetch_congress_buys()` catches all exceptions and returns `{}`. Any ticker with no congressional data gets raw score 0.0 (neutral after normalization with the rest of the universe).
- **FMP batch profile fails:** `mktcaps` defaults to `{}`, market caps default to 0.0. Cap-tier assignment falls back to CSV-declared tier.
- **Price fetch fails:** `fetch_price_data()` returns `{"return_20d": 0.0}`. Momentum score defaults to 0.0 (will rank at bottom after normalization — a conservative default).
- **Normalization with all-identical scores:** `normalize_score()` already handles zero-range arrays by returning `out_min` for all — no division-by-zero risk.
- **`fallback_reweight()`** from `regime_trader/scoring/normalize.py` is used in `generate_top_lists.py` when any factor vector is entirely missing (all NaN/zero), redistributing its weight to the remaining factors.

---

## 8. Testing

Each new function needs a unit test in `tests/`:

- `test_fetch_congress_buys`: mock both S3 URLs, verify purchase/sale counting and 90-day cutoff
- `test_score_congress`: verify raw→[0,1] mapping including edge cases (all purchases, all sales, no data)
- `test_fetch_price_data`: mock `yf.download`, verify 20-day return calculation and empty-df fallback
- `test_cross_sectional_normalization`: feed known raw vectors, assert normalized output has correct min/max/median
- `test_fmp_profile_chunks`: verify 165 tickers are split into batches of ≤100

Existing tests in `tests/test_normalize.py` already cover `normalize_score()` and `fallback_reweight()` — no changes needed there.
