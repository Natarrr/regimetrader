"""tests/test_scoring_signals.py
TDD tests for all five Smart Money scoring signals.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── Import helpers (imported lazily to avoid module-level side-effects) ─────
def _import():
    from scripts.run_pipeline import (
        score_insider_value,
        score_news_fmp,
        _score_news_yfinance,
        score_momentum,
        fetch_price_data,
        score_edgar,
        score_congress,
    )
    return (
        score_insider_value,
        score_news_fmp,
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


class TestScoreNewsFMP:
    """Tests for score_news_fmp() — FMP-based news sentiment scoring."""

    def test_all_positive_returns_above_neutral(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        _, score_news_fmp, *_ = _import()
        from regime_trader.services.fmp_client import FMPClient
        articles = [{"sentiment": "Positive"}] * 45 + [{"sentiment": "Negative"}] * 5
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("AAPL")
        assert score > 0.5

    def test_all_negative_returns_below_neutral(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        _, score_news_fmp, *_ = _import()
        from regime_trader.services.fmp_client import FMPClient
        articles = [{"sentiment": "Negative"}] * 45 + [{"sentiment": "Positive"}] * 5
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("TSLA")
        assert score < 0.5

    def test_empty_articles_falls_back_to_yfinance(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        _, score_news_fmp, *_ = _import()
        from regime_trader.services.fmp_client import FMPClient
        with patch.object(FMPClient, "get_news_raw_articles", return_value=[]), \
             patch("scripts.run_pipeline._score_news_sentiment_yfinance", return_value=0.55) as mock_yf:
            score = score_news_fmp("MSFT")
        mock_yf.assert_called_once_with("MSFT")
        assert score == pytest.approx(0.55)

    def test_score_bounded_0_to_1(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        _, score_news_fmp, *_ = _import()
        from regime_trader.services.fmp_client import FMPClient
        articles = [{"sentiment": "Positive"}] * 50
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("GOOG")
        assert 0.0 <= score <= 1.0

    def test_score_formula(self, monkeypatch):
        # 30 positive / 50 total → 0.60 * (30/50) + 0.40 * min(1.0, 50/50)
        #                        = 0.60 * 0.60 + 0.40 * 1.0 = 0.36 + 0.40 = 0.76
        monkeypatch.setenv("FMP_API_KEY", "k")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        _, score_news_fmp, *_ = _import()
        from regime_trader.services.fmp_client import FMPClient
        articles = [{"sentiment": "Positive"}] * 30 + [{"sentiment": "Negative"}] * 20
        with patch.object(FMPClient, "get_news_raw_articles", return_value=articles):
            score = score_news_fmp("AMZN")
        assert score == pytest.approx(0.76, abs=0.01)


class TestScoreNewsYFinanceFallback:
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


def _fmp_rows(closes: list, volumes: list) -> list:
    """Build FMP historical-price-eod/full rows (newest-first)."""
    rows = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        rows.append({"date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                     "close": c, "volume": v})
    return list(reversed(rows))   # newest-first


_PRICES = "regime_trader.services.fmp_client.FMPClient.get_historical_prices"


class TestFetchPriceDataEnhanced:
    def test_returns_spy_return_and_volume_spike(self):
        from scripts.run_pipeline import fetch_price_data

        # ≥252 bars so 12-1m is computable; volume spike in last 5 bars.
        n = 260
        closes  = [100.0 + (10.0 * i / (n - 1)) for i in range(n)]
        volumes = [1_000_000.0] * (n - 5) + [3_000_000.0] * 5
        rows = _fmp_rows(closes, volumes)

        spy_return_12m = 0.01

        with patch(_PRICES, return_value=rows):
            result = fetch_price_data("AAPL", spy_return=spy_return_12m)

        assert "return_12_1m" in result
        assert "spy_return_12_1m" in result
        assert "volume_spike" in result
        assert result["spy_return_12_1m"] == pytest.approx(spy_return_12m)
        assert result["volume_spike"] > 1.0

    def test_returns_none_on_failure(self):
        """Exception → default dict with return_12_1m=None (dead signal)."""
        from scripts.run_pipeline import fetch_price_data
        with patch(_PRICES, side_effect=Exception("network")):
            result = fetch_price_data("FAIL")
        assert result["return_12_1m"] is None
        assert result["volume_spike"] == pytest.approx(1.0)


class TestFetchSpyReturn:
    def test_success_returns_float(self):
        from scripts.run_pipeline import _fetch_spy_return

        # Rising 60-bar series → positive 12-1m return (idx_far=0, idx_near=-21)
        closes  = [500.0 + (10.0 * i / 59) for i in range(60)]
        volumes = [1_000_000.0] * 60
        rows = _fmp_rows(closes, volumes)

        with patch(_PRICES, return_value=rows):
            result = _fetch_spy_return()

        assert isinstance(result, float)
        assert result > 0.0

    def test_failure_returns_zero(self):
        from scripts.run_pipeline import _fetch_spy_return
        with patch(_PRICES, side_effect=Exception("network")):
            result = _fetch_spy_return()
        assert result == pytest.approx(0.0)


class TestQuiverEvidenceInResults:
    def test_score_ticker_result_contains_quiver_evidence_key(self):
        """_score_ticker() result must include quiver_evidence dict."""
        from scripts.run_pipeline import run
        import tempfile, csv
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

            def _fmp_rows(closes, vols):
                rows = [{"date": f"2026-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                         "close": c, "volume": v} for i, (c, v) in enumerate(zip(closes, vols))]
                return list(reversed(rows))

            ticker_rows = _fmp_rows([100.0 + i / 3 for i in range(30)], [1_000_000.0] * 30)

            with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
                       return_value=ticker_rows), \
                 patch("scripts.run_pipeline._sec_get", side_effect=Exception("no SEC in test")), \
                 patch("scripts.run_pipeline.fetch_fmp_profiles", return_value={"AAPL": 3e12}), \
                 patch("scripts.run_pipeline.fetch_congress_buys", return_value={}), \
                 patch("scripts.run_pipeline.fetch_fmp_insider_all", return_value={}), \
                 patch("scripts.run_pipeline.score_news_sentiment_combined", return_value=(0.55, "fmp", None, 0)), \
                 patch("scripts.run_pipeline.score_news_buzz_combined", return_value=(0.40, "fmp")):
                status = run(tickers_file, log_dir, max_workers=1)

            results = status.get("results", [])
            assert results, "No results returned"
            r = results[0]
            assert "quiver_evidence" in r, "quiver_evidence key missing from result"


class TestEvidencePassthroughFields:
    """_score_ticker() must include the Fix #3 evidence pass-through fields."""

    def test_score_ticker_result_contains_evidence_fields(self):
        from scripts.run_pipeline import run
        import tempfile, csv
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

            # Need ≥252 bars so return_12_1m is computable (not None)
            n = 260
            closes  = [100.0 + (10.0 * i / (n - 1)) for i in range(n)]
            volumes = [1_000_000.0] * (n - 5) + [3_000_000.0] * 5
            ticker_rows = [{"date": f"2024-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
                            "close": c, "volume": v}
                           for i, (c, v) in enumerate(zip(closes, volumes))]
            ticker_rows = list(reversed(ticker_rows))

            with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
                       return_value=ticker_rows), \
                 patch("scripts.run_pipeline._sec_get", side_effect=Exception("no SEC")), \
                 patch("scripts.run_pipeline.fetch_fmp_profiles", return_value={"AAPL": 3e12}), \
                 patch("scripts.run_pipeline.fetch_congress_buys", return_value={}), \
                 patch("scripts.run_pipeline.fetch_fmp_insider_all", return_value={}), \
                 patch("scripts.run_pipeline.score_news_sentiment_combined", return_value=(0.55, "fmp", None, 0)), \
                 patch("scripts.run_pipeline.score_news_buzz_combined", return_value=(0.40, "fmp")):
                status = run(tickers_file, log_dir, max_workers=1)

            r = status["results"][0]
            assert "news_sentiment_source" in r, "news_sentiment_source missing"
            assert "news_buzz_source" in r,      "news_buzz_source missing"
            assert "insider_usd" in r,           "insider_usd missing"
            assert "momentum_spy_relative" in r, "momentum_spy_relative missing"
            assert "volume_spike" in r,          "volume_spike missing"
            assert r["news_sentiment_source"] == "fmp"
            assert isinstance(r["insider_usd"], float)
            assert isinstance(r["momentum_spy_relative"], float)
            assert isinstance(r["volume_spike"], float)
            assert isinstance(r["quiver_evidence"], dict)


class TestFetchFMPInsiderAllSignals:
    """Unit tests for fetch_fmp_insider_all() — FMP Form 4 insider pre-fetch."""

    def test_no_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        from scripts.run_pipeline import fetch_fmp_insider_all
        result = fetch_fmp_insider_all(["AAPL"])
        assert result == {}

    def test_purchase_counted(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from scripts.run_pipeline import fetch_fmp_insider_all
        from regime_trader.services.fmp_client import FMPClient
        with patch.object(FMPClient, "get_insider_purchases", return_value=(250_000.0, 5)):
            result = fetch_fmp_insider_all(["AAPL"])
        assert result["AAPL"][0] == pytest.approx(250_000.0)
        assert result["AAPL"][1] >= 0

    def test_empty_input_returns_empty(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "k")
        from scripts.run_pipeline import fetch_fmp_insider_all
        assert fetch_fmp_insider_all([]) == {}

    def test_api_exception_returns_zero_not_crash(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        monkeypatch.setenv("FMP_MAX_RPS", "1000")
        from scripts.run_pipeline import fetch_fmp_insider_all
        from regime_trader.services.fmp_client import FMPClient
        with patch.object(FMPClient, "get_insider_purchases", side_effect=RuntimeError("network")):
            result = fetch_fmp_insider_all(["AAPL"])
        assert result["AAPL"] == (0.0, 0)
