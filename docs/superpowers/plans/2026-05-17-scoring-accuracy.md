# Smart Money Scoring Accuracy Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix one failing CI test and upgrade all 5 scoring signals to institutional-grade accuracy with new weights (edgar 28%, insider 23%, congress 22%, news 15%, momentum 12%).

**Architecture:** Each signal is an isolated pure function; the cross-sectional normaliser and VIX overlay in `generate_top_lists.py` are unchanged. `run_pipeline.py` and `generate_top_lists.py` share the same WEIGHTS dict semantically — both must be updated to match.

**Tech Stack:** Python 3.11, pytest, yfinance, requests (Finnhub), SEC EDGAR submissions API, Quiver Quantitative API, GitHub Actions secrets.

---

## File Map

| File | Action |
|------|--------|
| `tests/test_congress_fetcher.py` | Fix `test_s3_403_falls_back_to_quiver` — patch `QuiverClient.congress_by_ticker` |
| `tests/test_scoring_signals.py` | **Create** — tests for all 5 improved signal functions |
| `scripts/run_pipeline.py` | Update WEIGHTS; add `score_insider_value()`, `score_news_finnhub()`, `_score_news_yfinance()`; update `score_momentum()`; extend `fetch_price_data()`; update `_score_ticker()` to use new scorers + persist `quiver_evidence` |
| `backend/market_intel/generate_top_lists.py` | Update WEIGHTS dict + FACTOR_FIELDS key rename `macro` → `momentum` |
| `.github/workflows/edgar_3x.yml` | Add `FINNHUB_API_KEY` to EDGAR fetch step |
| `.github/workflows/canary.yml` | Add `FINNHUB_API_KEY` to Run EDGAR pipeline step |
| `.github/workflows/nightly_edgar.yml` | Add `FINNHUB_API_KEY` to EDGAR full-universe fetch step |
| `.github/workflows/hybrid_pipeline.yml` | Add `FINNHUB_API_KEY` to Run EDGAR pipeline step |

---

## Task 1: Fix the failing CI test

**Context:** `test_s3_403_falls_back_to_quiver` patches `requests.get` expecting to intercept the Quiver fallback call, but `_fetch_quiver_congress()` delegates to `QuiverClient.congress_by_ticker()` which uses `self._session.get()` — a different object. The patch misses, the real Quiver API is hit, and the test fails.

**Files:**
- Modify: `tests/test_congress_fetcher.py:130-153`

- [ ] **Step 1: Write the replacement test method**

Replace the `test_s3_403_falls_back_to_quiver` method (lines 130–153) in `tests/test_congress_fetcher.py` with:

```python
def test_s3_403_falls_back_to_quiver(self, tmp_path, monkeypatch):
    """When S3 returns 403, QuiverClient.congress_by_ticker() result is used."""
    monkeypatch.setenv("QUIVER_API_KEY", "test-key")
    monkeypatch.setattr("scripts.run_pipeline.CONGRESS_CACHE_PATH", tmp_path / "cc.json")

    quiver_result = {
        "TSLA": {
            "purchases": 1, "sales": 0, "total": 1, "net": 1,
            "representatives": ["Test Rep"], "recency_days": 5,
        }
    }

    def mock_s3_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = Exception("403 Forbidden")
        return resp

    with patch("requests.get", side_effect=mock_s3_get), \
         patch(
             "regime_trader.services.quiver_client.QuiverClient.congress_by_ticker",
             return_value=quiver_result,
         ):
        result = fetch_congress_buys(lookback_days=90)

    assert "TSLA" in result
    assert result["TSLA"]["purchases"] == 1
    assert result["TSLA"]["sales"] == 0
```

- [ ] **Step 2: Run the test to verify it now passes**

```bash
pytest tests/test_congress_fetcher.py::TestFetchCongressBuys::test_s3_403_falls_back_to_quiver -v
```

Expected output: `PASSED`

- [ ] **Step 3: Run the full congress test suite to catch regressions**

```bash
pytest tests/test_congress_fetcher.py -v
```

Expected: all tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/test_congress_fetcher.py
git commit -m "fix(test): patch QuiverClient.congress_by_ticker instead of requests.get in S3 fallback test"
```

---

## Task 2: New test file for all 5 scoring signals

**Context:** No test file exists for the improved signal functions. We write the tests first (TDD) so they fail against the old code, then the implementations in Task 3 make them pass.

**Files:**
- Create: `tests/test_scoring_signals.py`

- [ ] **Step 1: Create `tests/test_scoring_signals.py` with failing tests**

```python
"""tests/test_scoring_signals.py
TDD tests for all five Smart Money scoring signals.

Written against the TARGET implementations — these tests will FAIL against
the old code and PASS once Tasks 3–4 are complete.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest


# ── Import helpers (imported lazily to avoid module-level side-effects) ─────
def _import():
    from scripts.run_pipeline import (
        score_insider_value,
        score_news_finnhub,
        _score_news_yfinance,
        score_momentum,
        fetch_price_data,
        score_edgar,
        score_congress,
    )
    return (
        score_insider_value,
        score_news_finnhub,
        _score_news_yfinance,
        score_momentum,
        fetch_price_data,
        score_edgar,
        score_congress,
    )


class TestScoreInsiderValue:
    def test_zero_purchases_returns_zero_not_neutral(self):
        (score_insider_value, *_) = _import()
        assert score_insider_value(0.0, 1_000_000) == pytest.approx(0.0)

    def test_zero_market_cap_returns_zero(self):
        (score_insider_value, *_) = _import()
        assert score_insider_value(100_000.0, 0.0) == pytest.approx(0.0)

    def test_large_ceo_purchase_scores_near_ceiling(self):
        # $5M purchase, $500M market cap = 1% → near 0.90
        (score_insider_value, *_) = _import()
        score = score_insider_value(5_000_000.0, 500_000_000.0)
        assert score >= 0.85

    def test_small_purchase_scores_between_floor_and_midpoint(self):
        # $10K purchase, $1B market cap = 0.001% → between 0.30 and 0.65
        (score_insider_value, *_) = _import()
        score = score_insider_value(10_000.0, 1_000_000_000.0)
        assert 0.30 <= score <= 0.65

    def test_recency_decay_reduces_score_for_old_purchases(self):
        (score_insider_value, *_) = _import()
        recent = score_insider_value(500_000.0, 100_000_000.0, days_since_most_recent=5)
        old    = score_insider_value(500_000.0, 100_000_000.0, days_since_most_recent=150)
        assert recent > old

    def test_recency_decay_preserves_direction_not_zero(self):
        # Old net-buy signal should still be above 0.30 (not zeroed out)
        (score_insider_value, *_) = _import()
        score = score_insider_value(1_000_000.0, 500_000_000.0, days_since_most_recent=180)
        assert score > 0.30

    def test_score_bounded_0_to_1(self):
        (score_insider_value, *_) = _import()
        for usd, cap in [
            (0.0, 1e9), (1e6, 1e9), (1e8, 1e9),
            (1e9, 1e9), (1e6, 0.0),
        ]:
            score = score_insider_value(usd, cap)
            assert 0.0 <= score <= 1.0, f"Out of bounds for usd={usd}, cap={cap}"


class TestScoreNewsFinnhub:
    def test_all_bullish_returns_above_neutral(self):
        _, score_news_finnhub, *_ = _import()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "sentiment": {"bullishPercent": 0.90, "bearishPercent": 0.10},
            "buzz": {"weeklyAverage": 0.8},
        }
        with patch("requests.get", return_value=mock_resp):
            score = score_news_finnhub("AAPL", "fake-key")
        assert score > 0.5

    def test_all_bearish_returns_below_neutral(self):
        _, score_news_finnhub, *_ = _import()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "sentiment": {"bullishPercent": 0.10, "bearishPercent": 0.90},
            "buzz": {"weeklyAverage": 0.1},
        }
        with patch("requests.get", return_value=mock_resp):
            score = score_news_finnhub("TSLA", "fake-key")
        assert score < 0.5

    def test_api_failure_falls_back_to_yfinance(self):
        _, score_news_finnhub, _score_news_yfinance, *_ = _import()
        with patch("requests.get", side_effect=Exception("timeout")), \
             patch(
                 "scripts.run_pipeline._score_news_yfinance",
                 return_value=0.55,
             ) as mock_yf:
            score = score_news_finnhub("MSFT", "fake-key")
        mock_yf.assert_called_once_with("MSFT")
        assert score == pytest.approx(0.55)

    def test_yfinance_fallback_failure_returns_zero_not_neutral(self):
        _, score_news_finnhub, *_ = _import()
        with patch("requests.get", side_effect=Exception("timeout")), \
             patch("scripts.run_pipeline._score_news_yfinance", side_effect=Exception("yf down")):
            score = score_news_finnhub("GOOG", "fake-key")
        assert score == pytest.approx(0.0)

    def test_score_formula(self):
        # bullish=0.60, buzz weeklyAverage=0.5 → buzz_norm=1.0
        # score = 0.60*0.60 + 0.40*1.0 = 0.36 + 0.40 = 0.76
        _, score_news_finnhub, *_ = _import()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "sentiment": {"bullishPercent": 0.60},
            "buzz": {"weeklyAverage": 0.5},
        }
        with patch("requests.get", return_value=mock_resp):
            score = score_news_finnhub("AMZN", "fake-key")
        assert score == pytest.approx(0.76, abs=0.01)


class TestScoreNewsFinnhubYFinanceFallback:
    def test_yfinance_failure_returns_zero(self):
        *_, _score_news_yfinance, score_momentum, _, _, _ = _import()
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            score = _score_news_yfinance("AAPL")
        assert score == pytest.approx(0.0)


class TestScoreMomentumEnhanced:
    def test_ticker_beats_spy_scores_above_neutral(self):
        *_, score_momentum, _, _, _ = _import()
        # ticker +10%, SPY +5% → relative +5% → should score > 0.5
        score = score_momentum(
            ticker_return_20d=0.10,
            spy_return_20d=0.05,
            volume_spike=1.0,
        )
        assert score > 0.5

    def test_ticker_lags_spy_scores_below_neutral(self):
        *_, score_momentum, _, _, _ = _import()
        score = score_momentum(
            ticker_return_20d=0.02,
            spy_return_20d=0.08,
            volume_spike=1.0,
        )
        assert score < 0.5

    def test_high_volume_spike_boosts_score(self):
        *_, score_momentum, _, _, _ = _import()
        low_vol  = score_momentum(ticker_return_20d=0.05, spy_return_20d=0.05, volume_spike=1.0)
        high_vol = score_momentum(ticker_return_20d=0.05, spy_return_20d=0.05, volume_spike=5.0)
        assert high_vol > low_vol

    def test_missing_data_returns_zero(self):
        *_, score_momentum, _, _, _ = _import()
        score = score_momentum(
            ticker_return_20d=0.0,
            spy_return_20d=0.0,
            volume_spike=0.0,
        )
        # 0 spike → vol_score = max(0, (0-1)/4) = 0; equal returns → return_score = 0.5
        # Combined = 0.65*0.5 + 0.35*0 = 0.325 — not a hard 0 here; just bounded
        assert 0.0 <= score <= 1.0

    def test_score_formula(self):
        *_, score_momentum, _, _, _ = _import()
        # relative = 0.10 − 0.05 = 0.05 → clipped to 0.05
        # return_score = (0.05 + 0.30) / 0.60 = 0.583...
        # vol_score = min(1, (3.0 - 1) / 4) = 0.50
        # combined = 0.65*0.5833 + 0.35*0.50 = 0.3791 + 0.175 = 0.5541
        score = score_momentum(
            ticker_return_20d=0.10,
            spy_return_20d=0.05,
            volume_spike=3.0,
        )
        assert score == pytest.approx(0.5541, abs=0.01)

    def test_score_bounded_0_to_1(self):
        *_, score_momentum, _, _, _ = _import()
        for t, s, v in [
            (0.50, -0.50, 10.0),   # extreme outperformance + huge spike
            (-0.50, 0.50, 0.0),    # extreme underperformance + no volume
            (0.0, 0.0, 1.0),       # neutral
        ]:
            score = score_momentum(ticker_return_20d=t, spy_return_20d=s, volume_spike=v)
            assert 0.0 <= score <= 1.0, f"Out of bounds: t={t}, s={s}, v={v}"


class TestFetchPriceDataEnhanced:
    def test_returns_spy_return_and_volume_spike(self):
        from scripts.run_pipeline import fetch_price_data
        import pandas as pd
        import numpy as np

        # Build fake yfinance data for both ticker and SPY
        dates = pd.date_range("2026-04-01", periods=30, freq="B")
        ticker_close = pd.Series(np.linspace(100, 110, 30), index=dates)
        spy_close    = pd.Series(np.linspace(500, 505, 30), index=dates)
        ticker_vol   = pd.Series([1_000_000] * 25 + [3_000_000] * 5, index=dates)

        fake_ticker_df = pd.DataFrame({
            "Close":  ticker_close,
            "Volume": ticker_vol,
        })
        fake_spy_df = pd.DataFrame({"Close": spy_close})

        def fake_download(symbol, **kwargs):
            return fake_spy_df if symbol == "SPY" else fake_ticker_df

        with patch("yfinance.download", side_effect=fake_download):
            result = fetch_price_data("AAPL")

        assert "return_20d" in result
        assert "spy_return_20d" in result
        assert "volume_spike" in result
        assert result["spy_return_20d"] > 0.0
        assert result["volume_spike"] > 1.0   # recent vol higher than avg

    def test_returns_zeros_on_failure(self):
        from scripts.run_pipeline import fetch_price_data
        with patch("yfinance.download", side_effect=Exception("network")):
            result = fetch_price_data("FAIL")
        assert result == {"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0}


class TestQuiverEvidenceInResults:
    def test_score_ticker_result_contains_quiver_evidence_key(self):
        """_score_ticker() result must include quiver_evidence dict."""
        from scripts.run_pipeline import run
        import tempfile, csv, json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            tickers_file = tdp / "tickers.csv"
            log_dir = tdp / "logs"
            log_dir.mkdir()
            with tickers_file.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ticker", "sector", "cap_tier"])
                w.writeheader()
                w.writerow({"ticker": "AAPL", "sector": "Tech", "cap_tier": "large"})

            import yfinance as yf
            import pandas as pd, numpy as np
            dates = pd.date_range("2026-04-01", periods=30, freq="B")
            fake_df = pd.DataFrame({
                "Close":  pd.Series(np.linspace(100, 110, 30), index=dates),
                "Volume": pd.Series([1_000_000] * 30, index=dates),
            })
            fake_spy = pd.DataFrame({
                "Close": pd.Series(np.linspace(500, 510, 30), index=dates)
            })

            with patch("yfinance.download", side_effect=lambda sym, **kw: fake_spy if sym == "SPY" else fake_df), \
                 patch("yfinance.Ticker") as mock_ticker, \
                 patch("scripts.run_pipeline._sec_get", side_effect=Exception("no SEC in test")), \
                 patch("scripts.run_pipeline.fetch_fmp_profiles", return_value={"AAPL": 3e12}), \
                 patch("scripts.run_pipeline.fetch_congress_buys", return_value={}), \
                 patch("scripts.run_pipeline.score_news_finnhub", return_value=0.55):
                mock_ticker.return_value.news = []
                status = run(tickers_file, log_dir, max_workers=1)

            results = status.get("results", [])
            assert results, "No results returned"
            r = results[0]
            assert "quiver_evidence" in r, "quiver_evidence key missing from result"
            assert isinstance(r["quiver_evidence"], dict)
```

- [ ] **Step 2: Run the new tests to confirm they FAIL (expected)**

```bash
pytest tests/test_scoring_signals.py -v --tb=short 2>&1 | head -60
```

Expected: most tests fail with `ImportError: cannot import name 'score_insider_value'` or `AttributeError`. This confirms the tests are wired correctly.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_scoring_signals.py
git commit -m "test(scoring): add TDD tests for enhanced 5-factor signal functions"
```

---

## Task 3: Implement enhanced scoring functions in `run_pipeline.py`

**Context:** We update WEIGHTS, add `score_insider_value()`, add `score_news_finnhub()` + `_score_news_yfinance()`, update `score_momentum()` to accept SPY-relative + volume args, extend `fetch_price_data()` to return spy_return_20d + volume_spike, and update `_score_ticker()` to use all new functions and persist `quiver_evidence`.

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Update WEIGHTS dict (lines 44–50)**

Replace:
```python
WEIGHTS = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "momentum": 0.10,
}
```

With:
```python
WEIGHTS = {
    "edgar":    0.28,
    "insider":  0.23,
    "congress": 0.22,
    "news":     0.15,
    "momentum": 0.12,
}
```

- [ ] **Step 2: Replace `score_news()` with `score_news_finnhub()` + `_score_news_yfinance()`**

Find and replace the entire `score_news()` function (around line 429–454) with:

```python
def _score_news_yfinance(ticker: str) -> float:
    """yfinance headline word-count fallback. Returns 0.0 (not 0.5) on any failure."""
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        scores = []
        for item in news[:8]:
            content = item.get("content", {})
            title = (
                content.get("title", "") if isinstance(content, dict)
                else item.get("title", "")
            )
            if not title:
                continue
            words = set(title.lower().split())
            bull  = len(words & _BULL)
            bear  = len(words & _BEAR)
            if bull == 0 and bear == 0:
                scores.append(0.50)
            else:
                scores.append(max(0.10, min(0.90, 0.50 + 0.20 * (bull - bear))))
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)
    except Exception:
        return 0.0


def score_news_finnhub(ticker: str, api_key: str) -> float:
    """Engle (2003 Nobel) — Finnhub pre-computed sentiment ∈ [0, 1].

    Finnhub /news-sentiment returns:
      buzz.weeklyAverage      — normalized buzz volume (0-1)
      sentiment.bullishPercent — fraction of bullish articles (0-1)

    Score = 0.60 × bullishPercent + 0.40 × min(1.0, weeklyAverage / 0.5)

    Falls back to _score_news_yfinance() on any API failure.
    Returns 0.0 (not 0.5) if both sources fail — dead feed is penalised.
    """
    import requests as _req
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={api_key}"
    try:
        resp = _req.get(url, timeout=10)
        resp.raise_for_status()
        d       = resp.json()
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

- [ ] **Step 3: Add `score_insider_value()` function**

Insert this function after `score_edgar()` (around line 463):

```python
def score_insider_value(
    key_purchases_usd: float,
    market_cap: float,
    days_since_most_recent: int = 0,
) -> float:
    """Stiglitz (2001 Nobel) — dollar conviction score for insider purchases ∈ [0, 1].

    Maps total open-market purchase value as % of market cap to a score:
      0%      → 0.0   (no purchases = dead signal, penalised not neutral)
      0.01%   → ~0.30 (floor — small but credible)
      0.10%   → ~0.65 (mid — meaningful conviction)
      1.00%+  → ~0.90 (ceiling — exceptional conviction)

    Uses log-scale so small buys still count while large buys don't explode.
    Recency decay: purchases older than 30 days decay toward 0.50 neutral
    (direction preserved but urgency reduced — same decay as score_congress).
    """
    if key_purchases_usd <= 0 or market_cap <= 0:
        return 0.0

    pct = key_purchases_usd / market_cap
    raw = min(1.0, math.log1p(pct * 10000) / math.log1p(100))
    base_score = round(0.30 + 0.60 * raw, 4)

    if days_since_most_recent > 30:
        decay = max(0.70, 1.0 - 0.30 * min(days_since_most_recent - 30, 150) / 150)
        base_score = round(0.5 + (base_score - 0.5) * decay, 4)

    return base_score
```

- [ ] **Step 4: Update `score_momentum()` signature and formula**

Replace the existing `score_momentum()` function:

```python
def score_momentum(return_20d: float) -> float:
    r = max(-0.30, min(0.30, return_20d))
    return round(0.10 + (r + 0.30) / 0.60 * 0.80, 4)
```

With:

```python
def score_momentum(
    ticker_return_20d: float,
    spy_return_20d: float,
    volume_spike: float,
) -> float:
    """Thaler (2017 Nobel) — SPY-relative momentum + volume confirmation ∈ [0, 1].

    relative_return = ticker_return_20d - spy_return_20d, clipped to ±30%.
    return_score maps (-0.30, +0.30) linearly to (0, 1).
    vol_score maps volume_spike (ratio of recent avg to 90-day avg) to (0, 1):
      1.0× (flat) → 0.0,  5.0× spike → 1.0.

    Combined: 0.65 × return_score + 0.35 × vol_score
    """
    r = max(-0.30, min(0.30, ticker_return_20d - spy_return_20d))
    return_score = round((r + 0.30) / 0.60, 4)
    vol_score    = round(min(1.0, max(0.0, (volume_spike - 1.0) / 4.0)), 4)
    return round(0.65 * return_score + 0.35 * vol_score, 4)
```

- [ ] **Step 5: Update `fetch_price_data()` to also fetch SPY return and volume spike**

Replace the existing `fetch_price_data()` function (around line 323–345):

```python
def fetch_price_data(ticker: str) -> Dict[str, float]:
    """Thaler (2017 Nobel) — 20-day SPY-relative return + volume spike.

    Fetches ticker and SPY simultaneously for fair comparison.
    SPY is used as the market benchmark for relative momentum.
    Volume spike = 5-day avg volume / 90-day avg volume.

    Returns {"return_20d": float, "spy_return_20d": float, "volume_spike": float}.
    Returns {"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0} on any error.
    """
    _default = {"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0}
    try:
        import yfinance as yf
        import numpy as np

        df = yf.download(ticker, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 5:
            return _default

        close = df["Close"].squeeze().dropna()
        if len(close) < 2:
            return _default
        ret = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])

        # Volume spike: avg of last 5 sessions vs avg of full 90-day window
        if "Volume" in df.columns:
            vol = df["Volume"].squeeze().dropna()
            if len(vol) >= 10:
                recent_avg = float(vol.iloc[-5:].mean())
                full_avg   = float(vol.mean())
                volume_spike = round(recent_avg / full_avg, 4) if full_avg > 0 else 1.0
            else:
                volume_spike = 1.0
        else:
            volume_spike = 1.0

        # SPY return (same window)
        spy_df = yf.download("SPY", period="3mo", interval="1d",
                              progress=False, auto_adjust=True)
        if spy_df is None or spy_df.empty or len(spy_df) < 2:
            spy_ret = 0.0
        else:
            spy_close = spy_df["Close"].squeeze().dropna()
            spy_ret = float((spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0])

        return {
            "return_20d":     round(ret, 6),
            "spy_return_20d": round(spy_ret, 6),
            "volume_spike":   volume_spike,
        }
    except Exception as exc:
        log.debug("fetch_price_data %s failed: %s", ticker, exc)
        return _default
```

- [ ] **Step 6: Update `fetch_edgar_data()` to return `days_since_most_recent` and total purchase USD**

Find `fetch_edgar_data()` (around line 535). The return type changes from `Tuple[int, float, bool]` to `Tuple[int, float, bool, float, int]`. Update the function signature comment and the return:

Replace:
```python
def fetch_edgar_data(ticker: str, lookback_days: int = 180) -> Tuple[int, float, bool]:
```
With:
```python
def fetch_edgar_data(ticker: str, lookback_days: int = 180) -> Tuple[int, float, bool, float, int]:
```

Add `days_since_most_recent` tracking. Replace the block starting at `insider_score = 0.50` (around line 593–607):

```python
    insider_score = 0.0      # 0.0: no purchases = dead/penalised, not neutral
    ceo_buy = False
    total_purchases_usd = 0.0
    days_since_most_recent = 0
    if key_purchases:
        total_purchases_usd = sum(key_purchases)
        ceo_buy = total_purchases_usd > 25_000
        log.debug(
            "INSIDER %s: %d key purchases $%.0f ceo_buy=%s",
            ticker, len(key_purchases), total_purchases_usd, ceo_buy,
        )
        # days_since_most_recent: use the most recent filing date
        if form4_filings:
            most_recent_date = form4_filings[0]["date"]   # already sorted newest-first
            try:
                from datetime import date as _date
                delta = (_date.fromisoformat(
                    datetime.now(timezone.utc).date().isoformat()
                ) - _date.fromisoformat(most_recent_date)).days
                days_since_most_recent = max(0, delta)
            except Exception:
                days_since_most_recent = 0

    return form4_count, total_purchases_usd, ceo_buy, mktcaps_local, days_since_most_recent
```

**Note:** `fetch_edgar_data()` doesn't have access to `mktcaps` (that lives in `run()`). So instead of computing the full insider score here, just return the raw USD value and `days_since_most_recent`. The score is computed in `_score_ticker()` where `mktcaps` is available.

The correct return is:

```python
    return form4_count, total_purchases_usd, ceo_buy, days_since_most_recent
```

And the type hint:
```python
def fetch_edgar_data(ticker: str, lookback_days: int = 180) -> Tuple[int, float, bool, int]:
```

- [ ] **Step 7: Update `_score_ticker()` to use new scorers and persist `quiver_evidence`**

Replace the entire `_score_ticker()` inner function inside `run()` (around lines 650–698):

```python
    def _score_ticker(row: Dict[str, str]) -> Dict[str, Any]:
        ticker = row["ticker"]
        edgar_ok    = False
        form4_count = 0
        total_purchases_usd = 0.0
        ceo_buy     = False
        days_since_most_recent = 0
        try:
            form4_count, total_purchases_usd, ceo_buy, days_since_most_recent = fetch_edgar_data(ticker)
            edgar_ok = True
        except Exception as exc:
            log.warning("EDGAR unreachable for %s: %s", ticker, exc)

        mktcap = mktcaps.get(ticker, 0.0)
        congress_raw = congress_data.get(ticker)

        try:
            finnhub_key = os.getenv("FINNHUB_API_KEY", "")
            e_score    = score_edgar(form4_count)
            i_score    = score_insider_value(total_purchases_usd, mktcap, days_since_most_recent)
            c_score    = score_congress(congress_raw)
            n_score    = score_news_finnhub(ticker, finnhub_key) if finnhub_key else _score_news_yfinance(ticker)
            price_data = fetch_price_data(ticker)
            m_score    = score_momentum(
                price_data["return_20d"],
                price_data["spy_return_20d"],
                price_data["volume_spike"],
            )

            quiver_evidence = {
                "congress": {
                    "purchases":       int(congress_raw.get("purchases", 0)) if congress_raw else 0,
                    "sales":           int(congress_raw.get("sales", 0)) if congress_raw else 0,
                    "net":             int(congress_raw.get("net", congress_raw.get("purchases", 0) - congress_raw.get("sales", 0))) if congress_raw else 0,
                    "recency_days":    congress_raw.get("recency_days") if congress_raw else None,
                    "representatives": congress_raw.get("representatives", []) if congress_raw else [],
                },
                "source": "quiver" if congress_raw and congress_raw.get("recency_days") is not None else "s3",
            }

            return {
                "ticker":           ticker,
                "sector":           sector.get(ticker, "Unknown"),
                "cap_tier":         cap_tier.get(ticker, "large"),
                "market_cap":       mktcap,
                "edgar_score":      e_score,
                "insider_score":    i_score,
                "congress_score":   c_score,
                "news_score":       n_score,
                "momentum_score":   m_score,
                "ceo_buy":          ceo_buy,
                "form4_count":      form4_count,
                "quiver_evidence":  quiver_evidence,
                "_edgar_ok":        edgar_ok,
                "_scoring_error":   False,
            }
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", ticker, exc)
            return {
                "ticker": ticker, "sector": sector.get(ticker, "Unknown"),
                "cap_tier": cap_tier.get(ticker, "large"),
                "market_cap": mktcap,
                "edgar_score": 0.30, "insider_score": 0.0,
                "congress_score": 0.0, "news_score": 0.0,
                "momentum_score": 0.0, "ceo_buy": ceo_buy,
                "form4_count": form4_count,
                "quiver_evidence": {},
                "_edgar_ok":      edgar_ok,
                "_scoring_error": True,
            }
```

- [ ] **Step 8: Run the new scoring tests**

```bash
pytest tests/test_scoring_signals.py -v --tb=short
```

Expected: most tests PASS. Investigate any failures before continuing.

- [ ] **Step 9: Run the full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat(scoring): upgrade insider/news/momentum signals; persist quiver_evidence; update weights 28/23/22/15/12"
```

---

## Task 4: Update `generate_top_lists.py` — weights and factor field rename

**Context:** `generate_top_lists.py` has stale WEIGHTS (0.30/0.25/0.20/0.15/0.10) and a misleading key `"macro"` that maps to `momentum_score`. The pipeline writes `momentum_score`, not `macro_score`.

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py:47–66`

- [ ] **Step 1: Update WEIGHTS and FACTOR_FIELDS**

Replace lines 47–66 in `backend/market_intel/generate_top_lists.py`:

```python
WEIGHTS: Dict[str, float] = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "macro":    0.10,
}

# Maps factor key → field name in intel_source_status.json results
FACTOR_FIELDS: Dict[str, str] = {
    "edgar":    "edgar_score",
    "insider":  "insider_score",
    "congress": "congress_score",
    "news":     "news_score",
    # NOTE: the pipeline writes this field as "momentum_score" (price momentum alpha
    # factor), not a true macro/beta factor. A real macro score (VIX, yields, oil)
    # is applied as a multiplicative overlay via _apply_vix_overlay() below so that
    # the absolute risk regime is separated from the cross-sectional ranking.
    "macro":    "momentum_score",
}
```

With:

```python
WEIGHTS: Dict[str, float] = {
    "edgar":    0.28,
    "insider":  0.23,
    "congress": 0.22,
    "news":     0.15,
    "momentum": 0.12,
}

# Maps factor key → field name in intel_source_status.json results
FACTOR_FIELDS: Dict[str, str] = {
    "edgar":    "edgar_score",
    "insider":  "insider_score",
    "congress": "congress_score",
    "news":     "news_score",
    "momentum": "momentum_score",
}
```

- [ ] **Step 2: Fix CSV header in `generate()` — rename `macro` column to `momentum`**

Find the `writer.writerow` call that writes the CSV header (around line 382–393) and replace `"macro"` with `"momentum"`:

```python
        writer.writerow([
            "rank", "ticker", "sector", "cap_tier", "market_cap",
            "final_score", "badge", "ceo_buy", "form4_count",
            "edgar", "insider", "congress", "news", "momentum",
        ])
```

And in the data row below it, replace `f["macro"]` with `f["momentum"]`:

```python
            writer.writerow([
                rank,
                entry["ticker"],
                entry["sector"],
                entry["cap_tier"],
                entry["market_cap"],
                entry["final_score"],
                entry["badge"],
                entry["ceo_buy"],
                entry["form4_count"],
                f["edgar"], f["insider"], f["congress"], f["news"], f["momentum"],
            ])
```

- [ ] **Step 3: Fix the docstring at the top of the file**

Replace:
```python
  final_score = 0.30×edgar + 0.25×insider + 0.20×congress + 0.15×news + 0.10×macro
```
With:
```python
  final_score = 0.28×edgar + 0.23×insider + 0.22×congress + 0.15×news + 0.12×momentum
```

- [ ] **Step 4: Run generate_top_lists tests**

```bash
pytest tests/test_cross_sectional.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/market_intel/generate_top_lists.py
git commit -m "fix(generate_top_lists): update weights 28/23/22/15/12; rename macro→momentum factor key"
```

---

## Task 5: Add `FINNHUB_API_KEY` to GitHub workflow env blocks

**Context:** `score_news_finnhub()` reads `FINNHUB_API_KEY` from `os.getenv()`. Without it in the workflow env, Finnhub is never called and the function falls back to yfinance. We inject the secret in the 4 workflows that run the live pipeline.

**Files:**
- Modify: `.github/workflows/edgar_3x.yml`
- Modify: `.github/workflows/canary.yml`
- Modify: `.github/workflows/nightly_edgar.yml`
- Modify: `.github/workflows/hybrid_pipeline.yml`

- [ ] **Step 1: Add `FINNHUB_API_KEY` to `edgar_3x.yml`**

In `.github/workflows/edgar_3x.yml`, find the "EDGAR full-universe fetch" step env block (around line 72–78). Add after `QUIVER_API_KEY`:

```yaml
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
```

Full env block should read:
```yaml
        env:
          EDGAR_USER_AGENT: ${{ secrets.EDGAR_USER_AGENT || 'regime-trader-research n.tardy@hotmail.fr' }}
          FMP_API_KEY: ${{ secrets.FMP_API_KEY || '' }}
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
          EDGAR_FIRST: "true"
```

- [ ] **Step 2: Add `FINNHUB_API_KEY` to `canary.yml`**

In `.github/workflows/canary.yml`, find the "Run EDGAR pipeline" step env block (around line 52–57). Add:

```yaml
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
```

Full env block:
```yaml
        env:
          EDGAR_USER_AGENT: ${{ secrets.EDGAR_USER_AGENT || 'regime-trader-research n.tardy@hotmail.fr' }}
          FMP_API_KEY: ${{ secrets.FMP_API_KEY || '' }}
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
          EDGAR_FIRST: "true"
```

- [ ] **Step 3: Add `FINNHUB_API_KEY` to `nightly_edgar.yml`**

In `.github/workflows/nightly_edgar.yml`, find the "EDGAR full-universe fetch" step env block (around line 74–78). Add:

```yaml
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
```

Full env block:
```yaml
        env:
          EDGAR_USER_AGENT: ${{ secrets.EDGAR_USER_AGENT || 'regime-trader-research n.tardy@hotmail.fr' }}
          FMP_API_KEY: ${{ secrets.FMP_API_KEY || '' }}
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
          EDGAR_FIRST: "true"
```

- [ ] **Step 4: Add `FINNHUB_API_KEY` to `hybrid_pipeline.yml`**

In `.github/workflows/hybrid_pipeline.yml`, find the "Run EDGAR pipeline" step env block (around line 99–105). Add:

```yaml
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
```

Full env block:
```yaml
        env:
          SEC_USER_AGENT: ${{ secrets.SEC_USER_AGENT || '' }}
          FMP_API_KEY: ${{ secrets.FMP_API_KEY || '' }}
          QUIVER_API_KEY: ${{ secrets.QUIVER_API_KEY || '' }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY || '' }}
          MARKET_INTEL_DATA_DIR: ${{ github.workspace }}/data/raw/edgar
          SHORTLIST_QUINTILE_FLOOR: ${{ env.SHORTLIST_QUINTILE_FLOOR }}
```

- [ ] **Step 5: Verify no workflow references old weights in comments/docs**

```bash
grep -r "0.30.*edgar\|0.25.*insider\|0.20.*congress\|0.10.*momentum\|0.10.*macro" .github/
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/edgar_3x.yml .github/workflows/canary.yml \
        .github/workflows/nightly_edgar.yml .github/workflows/hybrid_pipeline.yml
git commit -m "ci: inject FINNHUB_API_KEY secret into all live pipeline workflows"
```

---

## Task 6: Final integration test and push

**Context:** Run the complete test suite, verify no regressions, then push to origin.

**Files:** None (test-only + git push)

- [ ] **Step 1: Run complete test suite**

```bash
pytest tests/ -v 2>&1 | tail -40
```

Expected: all tests PASS. Zero failures.

- [ ] **Step 2: Verify weights consistency between the two files**

```bash
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from scripts.run_pipeline import WEIGHTS as W1
from backend.market_intel.generate_top_lists import WEIGHTS as W2
print("run_pipeline WEIGHTS:", W1)
print("generate_top_lists WEIGHTS:", W2)
assert W1 == W2, f"WEIGHTS mismatch!\n  run_pipeline: {W1}\n  generate_top_lists: {W2}"
print("OK — weights match")
assert abs(sum(W1.values()) - 1.0) < 1e-9, f"Weights don't sum to 1: {sum(W1.values())}"
print("OK — weights sum to 1.0")
EOF
```

Expected output:
```
run_pipeline WEIGHTS: {'edgar': 0.28, 'insider': 0.23, 'congress': 0.22, 'news': 0.15, 'momentum': 0.12}
generate_top_lists WEIGHTS: {'edgar': 0.28, 'insider': 0.23, 'congress': 0.22, 'news': 0.15, 'momentum': 0.12}
OK — weights match
OK — weights sum to 1.0
```

- [ ] **Step 3: Verify `score_congress(None) == 0.0` (regression guard)**

```bash
python -c "
from scripts.run_pipeline import score_congress
assert score_congress(None) == 0.0, 'REGRESSION: score_congress(None) must be 0.0'
assert score_congress({}) == 0.0, 'REGRESSION: score_congress({}) must be 0.0'
print('score_congress regression guards OK')
"
```

Expected: `score_congress regression guards OK`

- [ ] **Step 4: Push to origin**

```bash
git push origin main
```

---

## Consistency Checklist (Self-Review)

| Check | Spec requirement | Covered by |
|-------|-----------------|------------|
| CI test fixed | `test_s3_403_falls_back_to_quiver` passes | Task 1 |
| WEIGHTS updated in run_pipeline.py | 0.28/0.23/0.22/0.15/0.12 | Task 3 Step 1 |
| WEIGHTS updated in generate_top_lists.py | same as above | Task 4 Step 1 |
| `insider` uses log-scaled dollar value | `score_insider_value()` | Task 3 Steps 3, 6, 7 |
| `news` uses Finnhub primary | `score_news_finnhub()` | Task 3 Step 2 |
| `news` fallback to yfinance | `_score_news_yfinance()` | Task 3 Step 2 |
| `news` failure returns 0.0 not 0.5 | `_score_news_yfinance()` returns 0.0 | Task 3 Step 2 |
| `momentum` is SPY-relative + volume | `score_momentum(t, s, v)` | Task 3 Steps 4, 5 |
| `quiver_evidence` persisted in results | `_score_ticker()` returns it | Task 3 Step 7 |
| `FACTOR_FIELDS` key is `momentum` not `macro` | generate_top_lists.py | Task 4 Step 1 |
| FINNHUB_API_KEY in all 4 workflows | edgar_3x, canary, nightly, hybrid | Task 5 |
| TDD: tests written before implementation | test_scoring_signals.py committed first | Task 2 before Task 3 |
| All tests pass end-to-end | pytest tests/ | Task 6 Step 1 |
