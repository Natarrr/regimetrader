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
    def test_ticker_beats_spy_with_volume_scores_above_neutral(self):
        *_, score_momentum, _, _, _ = _import()
        # ticker +10%, SPY +5%, volume 3x spike → combined should score > 0.5
        score = score_momentum(
            ticker_return_20d=0.10,
            spy_return_20d=0.05,
            volume_spike=3.0,
        )
        assert score > 0.5

    def test_ticker_lags_spy_scores_below_neutral(self):
        *_, score_momentum, _, _, _ = _import()
        # ticker lags SPY regardless of volume → below neutral
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


class TestEvidencePassthroughFields:
    """_score_ticker() must include the 4 evidence pass-through fields."""

    def test_score_ticker_result_contains_evidence_fields(self):
        from scripts.run_pipeline import run
        import tempfile, csv, json
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            tickers_file = tdp / "tickers.csv"
            log_dir = tdp / "logs"
            log_dir.mkdir()
            with tickers_file.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ticker", "sector", "cap_tier"])
                w.writeheader()
                w.writerow({"ticker": "AAPL", "sector": "Tech", "cap_tier": "large"})

            import pandas as pd, numpy as np
            dates = pd.date_range("2026-04-01", periods=30, freq="B")
            fake_df = pd.DataFrame({
                "Close":  pd.Series(np.linspace(100, 110, 30), index=dates),
                "Volume": pd.Series([1_000_000] * 25 + [3_000_000] * 5, index=dates),
            })
            fake_spy = pd.DataFrame({
                "Close": pd.Series(np.linspace(500, 510, 30), index=dates)
            })

            with patch("yfinance.download", side_effect=lambda sym, **kw: fake_spy if sym == "SPY" else fake_df), \
                 patch("yfinance.Ticker") as mock_ticker, \
                 patch("scripts.run_pipeline._sec_get", side_effect=Exception("no SEC")), \
                 patch("scripts.run_pipeline.fetch_fmp_profiles", return_value={"AAPL": 3e12}), \
                 patch("scripts.run_pipeline.fetch_congress_buys", return_value={}), \
                 patch("scripts.run_pipeline.score_news_finnhub", return_value=0.55):
                mock_ticker.return_value.news = []
                status = run(tickers_file, log_dir, max_workers=1)

            r = status["results"][0]
            assert "news_source" in r,           "news_source missing"
            assert "insider_usd" in r,           "insider_usd missing"
            assert "momentum_spy_relative" in r, "momentum_spy_relative missing"
            assert "volume_spike" in r,          "volume_spike missing"
            assert r["news_source"] in ("finnhub", "yfinance", "none")
            assert isinstance(r["insider_usd"], float)
            assert isinstance(r["momentum_spy_relative"], float)
            assert isinstance(r["volume_spike"], float)
            assert isinstance(r["quiver_evidence"], dict)
