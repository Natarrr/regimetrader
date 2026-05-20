"""tests/test_satellite_factors.py
Unit tests for backend.market_intel.satellite_factors.
All yfinance and FMP calls are monkeypatched — no network access.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.market_intel.satellite_factors import (
    MIN_MONTHLY_OBSERVATIONS,
    PE_MAX,
    PRICE_VS_52W_LOW_MAX,
    TOP_N,
    get_top_cannibals,
    get_top_cyclical,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_monthly_df(closes: list[float], month: int = 5) -> pd.DataFrame:
    """Build a single-ticker monthly DataFrame with all rows in `month`.

    One observation per year (annual cadence) so that filtering by
    ``df.index.month == month`` retains every row.
    """
    dates = pd.DatetimeIndex(
        [pd.Timestamp(year=2015 + i, month=month, day=1) for i in range(len(closes))]
    )
    opens = [c * 0.95 for c in closes]  # open slightly below close → always a win
    return pd.DataFrame({"Open": opens, "Close": closes}, index=dates)


def _make_batch_df(ticker_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create multi-column batch DataFrame as yf.download returns for multiple tickers."""
    frames = {}
    for ticker, df in ticker_dfs.items():
        for col in ("Open", "Close"):
            frames[(col, ticker)] = df[col]
    return pd.DataFrame(frames)


# ── Cyclicality tests ─────────────────────────────────────────────────────────

class TestGetTopCyclical:
    def test_filters_insufficient_history(self):
        """Ticker with fewer than MIN_MONTHLY_OBSERVATIONS rows is excluded."""
        short_df = _make_monthly_df([10.0] * (MIN_MONTHLY_OBSERVATIONS - 1), month=5)
        batch = _make_batch_df({"AAPL": short_df})

        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors.datetime"
            ) as mock_dt:
                mock_dt.now.return_value = MagicMock(month=5)
                result = get_top_cyclical(["AAPL"])

        assert result == [], "ticker with insufficient history must be excluded"

    def test_win_rate_calculation(self):
        """Known OHLC data produces expected win_rate."""
        # 10 months: 8 where close > open, 2 where close < open → win_rate = 0.80
        closes = [105.0] * 8 + [95.0] * 2
        df = _make_monthly_df(closes, month=5)
        # Override last 2 rows: open above close → losses
        df.iloc[-2:, df.columns.get_loc("Open")] = 100.0
        df.iloc[-2:, df.columns.get_loc("Close")] = 95.0
        batch = _make_batch_df({"PLTR": df})

        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors.datetime"
            ) as mock_dt:
                mock_dt.now.return_value = MagicMock(month=5)
                result = get_top_cyclical(["PLTR"])

        assert len(result) == 1
        assert result[0]["ticker"] == "PLTR"
        assert math.isclose(result[0]["win_rate"], 0.8, rel_tol=1e-3)

    def test_returns_at_most_top_n(self):
        """Result list is capped at TOP_N entries."""
        tickers = [f"T{i}" for i in range(TOP_N + 2)]
        dfs = {t: _make_monthly_df([100.0 + i] * MIN_MONTHLY_OBSERVATIONS, month=5)
               for i, t in enumerate(tickers)}
        batch = _make_batch_df(dfs)

        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors.datetime"
            ) as mock_dt:
                mock_dt.now.return_value = MagicMock(month=5)
                result = get_top_cyclical(tickers)

        assert len(result) <= TOP_N

    def test_returns_empty_on_yfinance_exception(self):
        """Any exception from yfinance.download returns []."""
        with patch("yfinance.download", side_effect=RuntimeError("timeout")):
            result = get_top_cyclical(["AAPL"])
        assert result == []


# ── Cannibal tests ────────────────────────────────────────────────────────────

class TestGetTopCannibals:
    _FMP_KEY = "test_key"
    _MARKET_CAPS = {"PLTR": 50_000_000_000.0, "SQ": 30_000_000_000.0}

    def _good_info(self, pe: float = 18.0, price: float = 20.0, low_52w: float = 18.0) -> dict:
        return {
            "trailingPE":      pe,
            "currentPrice":    price,
            "fiftyTwoWeekLow": low_52w,
        }

    def _fmp_quarters(self, repurchased: float = 500_000_000.0) -> list[dict]:
        return [{"repurchasedCommonStock": -repurchased / 4}] * 4

    def test_filters_high_pe(self):
        """Ticker with P/E >= PE_MAX is excluded."""
        info = self._good_info(pe=PE_MAX + 1)
        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            result = get_top_cannibals(["PLTR"], self._FMP_KEY, self._MARKET_CAPS)
        assert result == []

    def test_filters_price_above_52w_band(self):
        """Ticker priced above PRICE_VS_52W_LOW_MAX * 52w-low is excluded."""
        info = self._good_info(price=100.0, low_52w=50.0)  # ratio = 2.0 > 1.25
        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            result = get_top_cannibals(["PLTR"], self._FMP_KEY, self._MARKET_CAPS)
        assert result == []

    def test_zero_market_cap_skipped(self):
        """Ticker with market_cap = 0 is skipped — no ZeroDivisionError."""
        info = self._good_info()
        quarters = self._fmp_quarters()
        mock_resp = MagicMock()
        mock_resp.json.return_value = quarters
        mock_resp.raise_for_status = lambda: None

        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            with patch("requests.get", return_value=mock_resp):
                result = get_top_cannibals(["PLTR"], self._FMP_KEY, {"PLTR": 0.0})
        assert result == []

    def test_missing_fmp_key_returns_empty(self):
        """No FMP key → return [] without any HTTP call."""
        result = get_top_cannibals(["PLTR"], "", {"PLTR": 1e10})
        assert result == []

    def test_buyback_yield_calculated_correctly(self):
        """Correct buyback_yield = total_repurchased / market_cap."""
        info = self._good_info(pe=10.0, price=19.0, low_52w=18.0)
        # 4 quarters × 250M = 1B repurchased; market_cap = 50B → yield = 0.02
        quarters = [{"repurchasedCommonStock": -250_000_000}] * 4
        mock_resp = MagicMock()
        mock_resp.json.return_value = quarters
        mock_resp.raise_for_status = lambda: None

        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            with patch("requests.get", return_value=mock_resp):
                result = get_top_cannibals(
                    ["PLTR"], self._FMP_KEY, {"PLTR": 50_000_000_000.0}
                )

        assert len(result) == 1
        assert math.isclose(result[0]["buyback_yield"], 0.02, rel_tol=1e-3)


# ── Integration: main() ────────────────────────────────────────────────────────

class TestMain:
    def _make_top_lists(self, tickers: list[str]) -> dict:
        return {
            "generated_at": "2026-05-20T08:00:00+00:00",
            "top_buys": [
                {"ticker": t, "final_score": 0.7, "badge": "HIGH BUY",
                 "market_cap": 1e10, "factors": {}}
                for t in tickers
            ],
            "mid_caps":   [],
            "small_caps": [],
        }

    def test_main_writes_satellite_json(self, tmp_path, monkeypatch):
        """main() reads top_lists.json and writes satellite_insights.json."""
        top_lists = self._make_top_lists(["PLTR"])
        (tmp_path / "top_lists.json").write_text(
            json.dumps(top_lists), encoding="utf-8"
        )

        monkeypatch.setenv("FMP_API_KEY", "")  # no FMP key → cannibals = []
        with patch("yfinance.download", side_effect=RuntimeError("no network")):
            import sys
            monkeypatch.setattr(sys, "argv", ["satellite_factors", "--log-dir", str(tmp_path)])
            from backend.market_intel import satellite_factors
            satellite_factors.main()

        out = json.loads((tmp_path / "satellite_insights.json").read_text())
        assert "generated_at" in out
        assert "cyclicals" in out
        assert "cannibals" in out
        assert out["status"] in ("success", "partial", "error")

    def test_satellite_status_error_when_both_fail(self, tmp_path, monkeypatch):
        """status='error' when both cyclicals and cannibals return []."""
        tickers = ["PLTR"]
        top_lists = self._make_top_lists(tickers)
        (tmp_path / "top_lists.json").write_text(
            json.dumps(top_lists), encoding="utf-8"
        )

        # Both fail: yfinance exception + FMP key present but yf.info fails
        monkeypatch.setenv("FMP_API_KEY", "present_key")
        with patch("yfinance.download", side_effect=RuntimeError("timeout")):
            with patch(
                "backend.market_intel.satellite_factors._fetch_yf_info",
                side_effect=RuntimeError("timeout"),
            ):
                import sys
                monkeypatch.setattr(
                    sys, "argv",
                    ["satellite_factors", "--log-dir", str(tmp_path)],
                )
                from backend.market_intel import satellite_factors
                satellite_factors.main()

        out = json.loads((tmp_path / "satellite_insights.json").read_text())
        assert out["status"] == "error"
        assert out["cyclicals"] == []
        assert out["cannibals"] == []
