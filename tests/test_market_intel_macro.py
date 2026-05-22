"""tests/test_market_intel_macro.py
Unit tests for regime_trader.scanners.market_intel_macro.

Coverage:
  - calc_term_structure_score : backwardation, contango, neutral, all-zero
  - calc_cot_proxy_score      : strongly bullish, neutral, strongly bearish, clamp
  - calc_sentiment_score      : extreme bull, neutral, extreme bear, missing key
  - calc_trend_score          : golden cross, death cross, oversold, overbought
  - calc_macro_conviction     : composite label thresholds, output keys
  - check_macro_shocks        : oil spike, wheat spike, gold/copper divergence, quiet
  - generate_macro_synthesis  : crude backwardation path, no data fallback
  - safe_download             : mock yfinance, short df, import error path
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.scanners.market_intel_macro import (
    calc_cot_proxy_score,
    calc_macro_conviction,
    calc_sentiment_score,
    calc_term_structure_score,
    calc_trend_score,
    check_macro_shocks,
    generate_macro_synthesis,
    safe_download,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_price_data(
    price: float = 100.0,
    ret_5d: float = 0.0,
    ret_20d: float = 0.0,
    pct_52: float = 0.5,
    rsi14: float = 50.0,
    sma50: float = 95.0,
    sma200: float = 90.0,
    etf: str = "GLD",
) -> Dict[str, Any]:
    return {
        "price": price,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "pct_52": pct_52,
        "rsi14": rsi14,
        "sma50": sma50,
        "sma200": sma200,
        "etf": etf,
    }


# ── calc_term_structure_score ──────────────────────────────────────────────────

class TestCalcTermStructureScore:
    def test_backwardation_golden_cross(self):
        """Price > SMA200 and SMA50 > SMA200, with positive acceleration → high score."""
        data = _make_price_data(price=105.0, ret_5d=0.04, ret_20d=0.02,
                                sma50=103.0, sma200=95.0)
        score, label = calc_term_structure_score(data)
        assert score >= 0.58
        assert "Backwardation" in label or "Flat" in label

    def test_contango_death_cross(self):
        """Price below both MAs, negative momentum → low score."""
        data = _make_price_data(price=80.0, ret_5d=-0.04, ret_20d=-0.01,
                                sma50=90.0, sma200=95.0)
        score, label = calc_term_structure_score(data)
        assert score <= 0.45

    def test_neutral_flat(self):
        """Zero returns, price between MAs → mid-range score."""
        data = _make_price_data(price=92.0, ret_5d=0.0, ret_20d=0.0,
                                sma50=90.0, sma200=95.0)
        score, _ = calc_term_structure_score(data)
        assert 0.25 <= score <= 0.75

    def test_all_zero_returns_score_in_range(self):
        data = _make_price_data()
        score, label = calc_term_structure_score(data)
        assert 0.0 <= score <= 1.0
        assert isinstance(label, str)

    def test_score_bounded(self):
        """Extreme inputs must not produce score outside [0, 1]."""
        data = _make_price_data(price=1000.0, ret_5d=0.5, ret_20d=0.001,
                                sma50=1.0, sma200=1.0)
        score, _ = calc_term_structure_score(data)
        assert 0.0 <= score <= 1.0


# ── calc_cot_proxy_score ───────────────────────────────────────────────────────

class TestCalcCotProxyScore:
    def test_strongly_bullish_low_pct_52(self):
        data = _make_price_data(pct_52=0.10, rsi14=40.0, ret_5d=0.01)
        score, label = calc_cot_proxy_score(data)
        assert score >= 0.80
        assert "BULLISH" in label.upper()

    def test_neutral_mid_range(self):
        data = _make_price_data(pct_52=0.50, rsi14=50.0, ret_5d=0.0)
        score, label = calc_cot_proxy_score(data)
        assert 0.40 <= score <= 0.65
        assert "Neutral" in label

    def test_strongly_bearish_high_pct_52(self):
        data = _make_price_data(pct_52=0.90, rsi14=70.0, ret_5d=-0.02)
        score, label = calc_cot_proxy_score(data)
        assert score <= 0.25
        assert "BEARISH" in label.upper()

    def test_score_clamped_to_0_1(self):
        """ret_5d boost must not push score above 1.0."""
        data = _make_price_data(pct_52=0.05, rsi14=20.0, ret_5d=0.99)
        score, _ = calc_cot_proxy_score(data)
        assert score <= 1.0

    def test_score_floor_at_zero(self):
        data = _make_price_data(pct_52=0.99, ret_5d=-0.99)
        score, _ = calc_cot_proxy_score(data)
        assert score >= 0.0


# ── calc_sentiment_score ───────────────────────────────────────────────────────

class TestCalcSentimentScore:
    def test_extreme_retail_bullish_is_contrarian_sell(self):
        score, label = calc_sentiment_score("GLD", {"GLD": 0.95})
        assert score < 0.25
        assert "Contrarian Sell" in label

    def test_extreme_retail_bearish_is_strong_buy(self):
        score, label = calc_sentiment_score("GLD", {"GLD": 0.05})
        assert score > 0.90
        assert "Strong Buy" in label

    def test_neutral_sentiment(self):
        score, label = calc_sentiment_score("GLD", {"GLD": 0.50})
        assert 0.5 <= score <= 0.65
        assert "Neutral" in label

    def test_missing_etf_defaults_to_0_5(self):
        """Ticker not in map → raw=0.5, score = 1 - 0.8*0.5 = 0.6."""
        score, _ = calc_sentiment_score("MISSING", {})
        assert score == pytest.approx(0.60, abs=1e-4)

    def test_score_bounded(self):
        for raw in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score, _ = calc_sentiment_score("X", {"X": raw})
            assert 0.0 <= score <= 1.0


# ── calc_trend_score ───────────────────────────────────────────────────────────

class TestCalcTrendScore:
    def test_golden_cross_oversold_max_score(self):
        """Price > SMA200 > SMA50 and RSI < 30 → near-maximum score."""
        data = _make_price_data(price=105.0, sma50=103.0, sma200=90.0, rsi14=25.0)
        score, label = calc_trend_score(data)
        assert score >= 0.80
        assert "Golden Cross" in label
        assert "Oversold" in label

    def test_death_cross_overbought_low_score(self):
        data = _make_price_data(price=80.0, sma50=90.0, sma200=95.0, rsi14=80.0)
        score, label = calc_trend_score(data)
        assert score <= 0.25
        assert "Death Cross" in label

    def test_score_bounded(self):
        for price, sma50, sma200, rsi in [
            (105, 103, 90, 25), (80, 90, 95, 80), (92, 90, 95, 50),
        ]:
            data = _make_price_data(price=price, sma50=sma50, sma200=sma200, rsi14=rsi)
            score, _ = calc_trend_score(data)
            assert 0.0 <= score <= 1.0


# ── calc_macro_conviction ──────────────────────────────────────────────────────

class TestCalcMacroConviction:
    def test_output_keys_present(self):
        data = _make_price_data()
        result = calc_macro_conviction(data, {})
        required = {
            "composite", "conviction_label", "conviction_clr",
            "ts_score", "ts_label", "cot_score", "cot_label",
            "sent_score", "sent_label", "tr_score", "tr_label",
        }
        assert required.issubset(result.keys())

    def test_composite_bounded(self):
        for pct_52 in [0.05, 0.5, 0.95]:
            data = _make_price_data(pct_52=pct_52)
            result = calc_macro_conviction(data, {})
            assert 0.0 <= result["composite"] <= 1.0

    def test_strong_buy_label_at_high_score(self):
        """Force all sub-scores high by using backwardation + low pct_52 + low RSI."""
        data = _make_price_data(
            price=105.0, ret_5d=0.05, ret_20d=0.01,
            pct_52=0.08, rsi14=28.0,
            sma50=103.0, sma200=90.0, etf="USO",
        )
        sentiment_map = {"USO": 0.10}  # extreme retail bearish → contrarian buy
        result = calc_macro_conviction(data, sentiment_map)
        assert result["conviction_label"] in ("Strong Buy", "Buy")

    def test_avoid_label_at_low_score(self):
        data = _make_price_data(
            price=80.0, ret_5d=-0.05, ret_20d=-0.03,
            pct_52=0.95, rsi14=78.0,
            sma50=90.0, sma200=95.0, etf="USO",
        )
        sentiment_map = {"USO": 0.95}  # extreme retail bullish → contrarian sell
        result = calc_macro_conviction(data, sentiment_map)
        assert result["conviction_label"] in ("Avoid", "Reduce")


# ── check_macro_shocks ─────────────────────────────────────────────────────────

class TestCheckMacroShocks:
    def test_no_alerts_when_quiet(self):
        prices = {
            "CL=F": {"ret_5d": 0.01},
            "ZW=F": {"ret_5d": 0.02},
            "GC=F": {"ret_5d": 0.00},
            "HG=F": {"ret_5d": 0.00},
            "NG=F": {"ret_5d": 0.05},
        }
        alerts = check_macro_shocks(prices)
        assert alerts == []

    def test_oil_spike_triggers_error(self):
        prices = {"CL=F": {"ret_5d": 0.07}}
        alerts = check_macro_shocks(prices)
        assert any(a["level"] == "error" for a in alerts)
        assert any("Crude Oil" in a["message"] for a in alerts)

    def test_oil_moderate_triggers_warning(self):
        prices = {"CL=F": {"ret_5d": 0.04}}
        alerts = check_macro_shocks(prices)
        assert any(a["level"] == "warning" for a in alerts)

    def test_wheat_spike_triggers_error(self):
        prices = {"ZW=F": {"ret_5d": 0.12}}
        alerts = check_macro_shocks(prices)
        assert any("Wheat" in a["message"] for a in alerts)
        assert any(a["level"] == "error" for a in alerts)

    def test_gold_copper_divergence_error(self):
        prices = {
            "GC=F": {"ret_5d": 0.04},
            "HG=F": {"ret_5d": -0.05},
        }
        alerts = check_macro_shocks(prices)
        assert any("Recession Warning" in a["message"] for a in alerts)

    def test_gold_copper_mild_divergence_warning(self):
        prices = {
            "GC=F": {"ret_5d": 0.015},
            "HG=F": {"ret_5d": -0.015},
        }
        alerts = check_macro_shocks(prices)
        assert any(a["level"] == "warning" for a in alerts)

    def test_empty_prices_no_crash(self):
        alerts = check_macro_shocks({})
        assert isinstance(alerts, list)

    def test_none_values_no_crash(self):
        alerts = check_macro_shocks({"CL=F": None, "GC=F": None})
        assert isinstance(alerts, list)


# ── generate_macro_synthesis ───────────────────────────────────────────────────

class TestGenerateMacroSynthesis:
    def test_returns_non_empty_list(self):
        paras = generate_macro_synthesis({}, {}, {})
        assert isinstance(paras, list)
        assert len(paras) >= 1

    def test_no_data_fallback_message(self):
        paras = generate_macro_synthesis({}, {}, {})
        assert any("Insufficient" in p for p in paras)

    def test_crude_backwardation_paragraph(self):
        prices = {"CL=F": {"ret_5d": 0.03, "rsi14": 55.0}}
        convictions = {"CL=F": {
            "ts_label": "Backwardation (up-up)",
            "cot_label": "Bullish — Commercial Buying",
            "conviction_label": "Buy",
        }}
        paras = generate_macro_synthesis(prices, convictions, {})
        assert any("CRUDE OIL" in p for p in paras)
        assert any("Backwardation" in p or "backwardation" in p for p in paras)

    def test_macro_backdrop_included_with_vix(self):
        indicators = {"^VIX": {"price": 35.0, "ret_5d": 0.10}}
        paras = generate_macro_synthesis({}, {}, indicators)
        assert any("VIX" in p for p in paras)


# ── safe_download ──────────────────────────────────────────────────────────────

class TestSafeDownload:
    def test_returns_none_on_empty_df(self):
        import pandas as pd
        empty_df = pd.DataFrame()
        with patch("yfinance.download", return_value=empty_df):
            result = safe_download("MISSING")
        assert result is None

    def test_returns_none_on_exception(self):
        with patch("yfinance.download", side_effect=RuntimeError("network")):
            result = safe_download("AAPL")
        assert result is None

    def test_returns_dataframe_on_success(self):
        import pandas as pd
        import numpy as np
        n = 15
        idx = pd.date_range("2026-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Close":  np.linspace(100, 110, n),
            "Volume": [1_000_000.0] * n,
            "Open":   np.linspace(99, 109, n),
            "High":   np.linspace(101, 111, n),
            "Low":    np.linspace(98, 108, n),
        }, index=idx)
        with patch("yfinance.download", return_value=df):
            result = safe_download("AAPL")
        assert result is not None
        assert len(result) == n
