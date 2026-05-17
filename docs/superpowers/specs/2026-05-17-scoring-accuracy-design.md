# Smart Money Scoring System — Accuracy Overhaul Design Spec

**Date:** 2026-05-17  
**Author:** Nathan T (regime_trader)  
**Status:** Approved for implementation

---

## Goal

Fix the one failing CI test, then improve every scoring signal to institutional-grade accuracy while keeping the 5-factor architecture. Quiver data moves from display-only metadata to a reinforcing source for the congress and insider factors. Dashboard and Discord reflect the improved signals.

---

## Architecture Decision

**Keep 5-factor model.** Quiver enriches existing factors rather than adding a 6th:
- `congress` factor: Quiver replaces S3 as the primary source (S3 is dead/403). Recency weighting already in place.
- `insider` factor: Quiver insider trades cross-validate EDGAR Form 4 signals.
- All other factors improved in quality without changing the factor count.

**New weights (sum = 1.00):**
```python
WEIGHTS = {
    "edgar":    0.28,   # was 0.30 — EDGAR Form 4 activity
    "insider":  0.23,   # was 0.25 — CEO/CFO open-market purchase value
    "congress": 0.22,   # was 0.20 — net congressional trades (Quiver-primary)
    "news":     0.15,   # unchanged — Finnhub sentiment + yfinance fallback
    "momentum": 0.12,   # was 0.10 — enhanced: SPY-relative + volume confirmation
}
```

---

## File Map

| Action | File | Change |
|--------|------|--------|
| Fix test | `tests/test_congress_fetcher.py` | Patch `QuiverClient.congress_by_ticker` not `requests.get` |
| Update weights | `scripts/run_pipeline.py` | New WEIGHTS dict; update `score_insider()`, `score_news()`, `score_momentum()` |
| Update weights | `backend/market_intel/generate_top_lists.py` | New WEIGHTS dict; update FACTOR_FIELDS comment |
| New scorer | `scripts/run_pipeline.py` | `score_news_finnhub()` — Finnhub primary, yfinance fallback |
| Enhance scorer | `scripts/run_pipeline.py` | `score_momentum()` — SPY-relative return + volume spike |
| Enhance scorer | `scripts/run_pipeline.py` | `score_insider_value()` — dollar value as % mktcap with decay |
| Quiver evidence | `scripts/run_pipeline.py` | Attach `quiver_evidence` dict per ticker in `_score_ticker()` result |
| Dashboard | `regime_trader/ui/streamlit_app.py` | Show 5 factor scores in Market Intel; congress shows Quiver source |
| Discord | `scripts/send_toplists_discord.py` | Show updated factor weights in embed footer |
| Workflows | `edgar_3x.yml`, `canary.yml`, etc. | Add `FINNHUB_API_KEY` secret injection |
| Tests | `tests/test_scoring_signals.py` | New test file for all 5 scoring functions |

---

## Signal Specifications

### 1. `edgar` score (weight 0.28)

**Source:** SEC EDGAR submissions API — Form 4 filing count in last 90 days.

**Formula (unchanged, already correct):**
```python
def score_edgar(form4_count: int) -> float:
    if form4_count <= 0:
        return 0.30   # floor: company exists but no recent filings
    return round(min(0.90, 0.30 + form4_count * 0.12), 4)
```

**Interpretation:** Counts insider filing *activity* (any Form 4). Higher activity = more insider interest. Capped at 0.90 so a single very active ticker doesn't dominate cross-sectionally.

**No change needed.**

---

### 2. `insider` score (weight 0.23)

**Source:** EDGAR Form 4 XML parse — open-market purchases (`code="P"`) by key officers (CEO, CFO, COO, Director, Chairman, Founder) in last 90 days.

**Problem with current formula:** `0.50 + count × 0.08` ignores dollar magnitude. A CEO buying $5,000 and a CEO buying $5,000,000 get the same score per transaction.

**New formula — dollar value as % of market cap:**
```python
def score_insider_value(
    key_purchases_usd: float,
    market_cap: float,
    days_since_most_recent: int = 0,
) -> float:
    """Score insider conviction by purchase size relative to company size.
    
    $conviction = total_purchase_usd / market_cap$, capped at 1% (0.01).
    Maps 0% → 0.0, 0.01% → 0.30, 0.10% → 0.65, 1.00%+ → 0.90.
    Recency decay: purchases older than 30 days decay toward 0.50 neutral.
    """
    if key_purchases_usd <= 0 or market_cap <= 0:
        return 0.0   # no purchases = dead signal, not neutral
    
    pct = key_purchases_usd / market_cap  # 0.001 = 0.1% of mktcap
    # Log scale: small buys still count, large buys don't explode
    import math
    raw = min(1.0, math.log1p(pct * 10000) / math.log1p(100))
    base_score = round(0.30 + 0.60 * raw, 4)
    
    # Recency decay (same logic as congress)
    if days_since_most_recent > 30:
        decay = max(0.70, 1.0 - 0.30 * min(days_since_most_recent - 30, 150) / 150)
        base_score = round(0.5 + (base_score - 0.5) * decay, 4)
    
    return base_score
```

**Rationale (Stiglitz 2001):** A $5M open-market purchase by a CEO is a costly, credible signal. A $5K purchase is noise. Dollar magnitude as % of market cap is the correct normalization — it filters for conviction, not frequency.

**Cross-validation with Quiver:** After scoring EDGAR, attach Quiver insider trades in `quiver_evidence["insider"]` for the explainability layer. If Quiver shows large insider acquisitions not yet in EDGAR (< 2-day reporting lag), note them in evidence but do not double-count in score.

---

### 3. `congress` score (weight 0.22)

**Source:** Quiver Quantitative `/beta/live/congresstrading` (primary). S3 Stock Watcher (fallback, currently 403-dead). 

**Formula (already correct — keep as-is):**
```python
def score_congress(data: Optional[Dict]) -> float:
    # data = {"purchases": int, "sales": int, "total": int, "recency_days": optional[int]}
    if not data:
        return 0.0   # dead feed → penalised, not neutral
    purchases = int(data.get("purchases", 0))
    sales     = int(data.get("sales", 0))
    total     = purchases + sales
    if total == 0:
        return 0.50  # data present, no net activity = genuine neutral
    raw = (purchases - sales) / (total + 1)  # ∈ (-1, 1)
    base_score = round((raw + 1) / 2, 4)     # → (0, 1)
    # Recency decay: full credit ≤30 days, 0.70× at 180 days
    recency_days = data.get("recency_days")
    if recency_days is not None and recency_days > 30:
        decay = max(0.70, 1.0 - 0.30 * min(recency_days - 30, 150) / 150)
        base_score = round(0.5 + (base_score - 0.5) * decay, 4)
    return base_score
```

**Key fix:** `fetch_congress_buys()` must pass `recency_days` from Quiver's `congress_by_ticker()` into the per-ticker dict. Currently when Quiver is the source, `recency_days` IS populated. When S3 is the source, it is NOT. Fix: populate `recency_days` in `_parse_congress_transactions()` as well.

**Quiver as primary source:** Congress data now always goes through `QuiverClient.congress_by_ticker()`. S3 is kept as fallback for when Quiver fails.

---

### 4. `news` score (weight 0.15)

**Source:** Finnhub `/news-sentiment` endpoint (primary). yfinance word-count (fallback).

**Problem with current formula:** Counting "beat", "surge" in headlines is low-precision. Finnhub computes professional sentiment scores from news aggregation.

**New formula:**
```python
def score_news_finnhub(ticker: str, api_key: str) -> float:
    """Engle (2003 Nobel) — Finnhub pre-computed sentiment score ∈ [0, 1].
    
    Finnhub /news-sentiment returns:
      buzz.weeklyAverage    — normalized buzz volume (0-1)
      sentiment.bullishPercent — fraction of bullish articles (0-1)
      sentiment.bearishPercent — fraction of bearish articles (0-1)
    
    Score = 0.60 × bullishPercent + 0.40 × min(1.0, weeklyAverage / 0.5)
    
    Returns 0.0 (not 0.5) if API fails — dead feed is penalised.
    """
    import requests
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        bullish = float(d.get("sentiment", {}).get("bullishPercent", 0.5))
        buzz    = float(d.get("buzz", {}).get("weeklyAverage", 0.0))
        buzz_norm = min(1.0, buzz / 0.5)
        return round(0.60 * bullish + 0.40 * buzz_norm, 4)
    except Exception:
        return _score_news_yfinance(ticker)  # fallback


def _score_news_yfinance(ticker: str) -> float:
    """Fallback: existing yfinance word-count sentiment. Returns 0.0 on failure."""
    # (existing score_news() logic, renamed)
    # Changed: returns 0.0 on exception instead of 0.50 — dead feed penalised
```

**CI isolation:** Finnhub calls must be mocked in tests. Add `FINNHUB_API_KEY` to workflow env blocks.

---

### 5. `momentum` score (weight 0.12)

**Source:** yfinance — 20-day price return vs SPY, plus volume spike vs 90-day average.

**Problem with current formula:** Absolute 20-day return doesn't tell you if the move is market-driven or stock-specific. A ticker up 5% while SPY is up 6% is actually underperforming.

**New formula — SPY-relative + volume confirmation:**
```python
def score_momentum(ticker_return_20d: float, spy_return_20d: float, volume_spike: float) -> float:
    """Thaler (2017 Nobel) — relative momentum with volume confirmation.
    
    relative_return = ticker_return_20d - spy_return_20d
    vol_score       = min(1.0, max(0.0, (volume_spike - 1.0) / 4.0))
    
    Combined: 0.65 × return_score + 0.35 × vol_score
    
    Returns 0.0 if data unavailable — penalised, not neutral.
    """
    r = max(-0.30, min(0.30, ticker_return_20d - spy_return_20d))
    return_score = round((r + 0.30) / 0.60, 4)   # maps (-0.30, +0.30) → (0, 1)
    vol_score    = round(min(1.0, max(0.0, (volume_spike - 1.0) / 4.0)), 4)
    return round(0.65 * return_score + 0.35 * vol_score, 4)
```

**Data fetching:** `fetch_price_data()` extended to also fetch SPY 20d return and volume spike (5d avg / 90d avg). SPY is fetched once per pipeline run and cached.

---

## Quiver Evidence Persistence

**Problem:** `_score_ticker()` returns congress score but discards the raw congress metadata (purchases, sales, recency_days, representatives). `generate_top_lists.py` reads `quiver_evidence` from results but it's never populated.

**Fix:** Add `quiver_evidence` to every result row in `_score_ticker()`:
```python
return {
    # ... existing fields ...
    "quiver_evidence": {
        "congress": {
            "purchases":       congress_raw.get("purchases", 0),
            "sales":           congress_raw.get("sales", 0),
            "net":             congress_raw.get("net", 0),
            "recency_days":    congress_raw.get("recency_days"),
            "representatives": congress_raw.get("representatives", []),
        },
        "source": "quiver" if quiver_was_used else "s3",
    }
}
```

This data flows through `intel_source_status.json` → `generate_top_lists.py` → `top_lists.json` → `send_toplists_discord.py` and Streamlit. No additional API calls.

---

## Discovery Scanner

**Quiver remains enrichment-only** in the discovery scanner. Scoring stays at:
- 45% EDGAR/yfinance insider (dollar value vs market cap)
- 35% FMP institutional accumulation
- 20% momentum (volume spike + price change)

Quiver `quiver["congress"]`, `quiver["insider"]`, `quiver["f13"]` are shown in Streamlit evidence expander but do not affect the `smart_money_score`. This is correct: the discovery scanner is a separate pipeline from the 5-factor model.

---

## CI Fix

**Failing test:** `tests/test_congress_fetcher.py::TestFetchCongressBuys::test_s3_403_falls_back_to_quiver`

**Root cause:** `_fetch_quiver_congress()` delegates to `QuiverClient.congress_by_ticker()` which uses `session.get` (its own session), not `requests.get`. The test patches `requests.get` — which intercepts S3 calls but not QuiverClient's private session. Without a mock, the test hits the real Quiver API which returns 500.

**Fix:** Patch `QuiverClient.congress_by_ticker` directly:
```python
def test_s3_403_falls_back_to_quiver(self, tmp_path, monkeypatch):
    monkeypatch.setenv("QUIVER_API_KEY", "test-key")
    monkeypatch.setattr("scripts.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")
    
    quiver_result = {
        "TSLA": {"purchases": 1, "sales": 0, "total": 1, "net": 1,
                 "representatives": ["Test Rep"], "recency_days": 5}
    }
    
    def mock_s3_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = Exception("403")
        return resp
    
    with patch("requests.get", side_effect=mock_s3_get), \
         patch("regime_trader.services.quiver_client.QuiverClient.congress_by_ticker",
               return_value=quiver_result):
        result = fetch_congress_buys(lookback_days=90)
    
    assert "TSLA" in result
    assert result["TSLA"]["purchases"] == 1
```

---

## Dashboard Updates

**Streamlit Market Intel tab:**
- Factor columns already show as ProgressColumn for smart_money, insider, institutional, momentum
- Add: updated weight labels to reflect new 28/23/22/15/12 split
- Congress column: show `{net} net ({recency}d ago)` from `quiver_evidence` if available
- Quiver evidence expander: show all 5 Quiver data types (already implemented, keep as-is)
- News column: show Finnhub source label vs yfinance fallback in evidence

**Discord embed:**
- Footer: update weight percentages (28/23/22/15/12)
- `send_toplists_discord.py` reads weights from `top_lists.json["weights"]` so it auto-updates if weights change

---

## GitHub Workflows

Add `FINNHUB_API_KEY` to all workflows that run the pipeline:

| Workflow | Needs FINNHUB_API_KEY? |
|----------|----------------------|
| `edgar_3x.yml` | ✅ Add to EDGAR fetch step env |
| `canary.yml` | ✅ Add to Run EDGAR pipeline step env |
| `nightly_edgar.yml` | ✅ Add to EDGAR full-universe fetch step env |
| `hybrid_pipeline.yml` | ✅ Add to Run EDGAR pipeline step env |
| `ci.yml` | ❌ Tests mock HTTP; no key needed |
| `market_intel.yml` | ❌ Tests only |

Finnhub calls must be mocked in CI tests (same pattern as existing mocks).

---

## Test Strategy

**New file: `tests/test_scoring_signals.py`**
```
TestScoreInsiderValue
  - zero purchases returns 0.0 (not 0.5)
  - large CEO purchase scores near 0.90
  - small purchase scores between 0.30 and 0.65
  - recency decay: 120-day-old purchase scores less than 5-day-old
  - decay toward neutral, not zero

TestScoreNewsFinnhub
  - all-bullish returns > 0.5
  - all-bearish returns < 0.5
  - API failure falls back to yfinance
  - yfinance failure returns 0.0 (not 0.5)

TestScoreMomentum (enhanced)
  - ticker beats SPY → score > 0.5
  - ticker lags SPY → score < 0.5
  - high volume spike boosts score
  - missing data returns 0.0

TestQuiverEvidenceInResults
  - _score_ticker() result contains quiver_evidence key
  - quiver_evidence.congress matches fetch_congress_buys() output
  - quiver_evidence persists through intel_source_status.json
```

**Update: `tests/test_congress_fetcher.py`**
- Fix `test_s3_403_falls_back_to_quiver` to patch `QuiverClient.congress_by_ticker`

---

## Consistency Checklist

| Check | Before | After |
|-------|--------|-------|
| `WEIGHTS` in `run_pipeline.py` | 0.30/0.25/0.20/0.15/0.10 | 0.28/0.23/0.22/0.15/0.12 |
| `WEIGHTS` in `generate_top_lists.py` | 0.30/0.25/0.20/0.15/0.10 | 0.28/0.23/0.22/0.15/0.12 |
| `insider` score formula | count × 0.08 from 0.50 | dollar value % mktcap, log-scaled |
| `news` score source | yfinance word-count | Finnhub API + yfinance fallback |
| `momentum` score | absolute 20d return | SPY-relative return + volume spike |
| `congress` evidence | discarded after scoring | persisted in quiver_evidence |
| `news` failure default | 0.50 (neutral) | 0.0 (dead feed penalised) |
| Dead-factor weight redistribution | ✅ existing | ✅ unchanged |
| VIX overlay | ✅ existing | ✅ unchanged |
| Cross-sectional normalisation | ✅ existing | ✅ unchanged |
| CI test passing | ❌ 1 failing | ✅ all passing |
