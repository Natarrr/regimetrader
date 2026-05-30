"""tests/test_satellite_factors.py
Unit tests for backend.market_intel.satellite_factors.
All FMP calls are monkeypatched — no network access.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from backend.market_intel.satellite_factors import (
    MIN_MONTHLY_OBSERVATIONS,
    PE_MAX,
    TOP_N,
    get_top_cannibals,
    get_top_cyclical,
)


# ── FMP price row factory ─────────────────────────────────────────────────────

def _fmp_rows(closes: list[float], month: int = 5) -> list[dict]:
    """Build FMP historical-price-eod/full rows (newest-first) for a given month."""
    rows = []
    for i, c in enumerate(reversed(closes)):
        year = 2025 - i // 1   # all same year is fine for tests
        rows.append({
            "date":   f"{year}-{month:02d}-{(i % 28) + 1:02d}",
            "close":  c,
            "volume": 1_000_000.0,
        })
    return rows   # newest-first as FMP returns


# ── Cyclicality tests ─────────────────────────────────────────────────────────

class TestGetTopCyclical:
    def _patch_fmp(self, rows: list[dict]):
        """Patch FMPClient.get_historical_prices to return the given rows."""
        return patch(
            "regime_trader.services.fmp_client.FMPClient.get_historical_prices",
            return_value=rows,
        )

    def test_filters_insufficient_history(self):
        """Ticker with fewer than MIN_MONTHLY_OBSERVATIONS month-samples is excluded."""
        # Only MIN-1 rows all in month 5 → not enough monthly observations
        short_rows = _fmp_rows([100.0] * (MIN_MONTHLY_OBSERVATIONS - 1), month=5)
        with self._patch_fmp(short_rows):
            with patch("backend.market_intel.satellite_factors.datetime") as mock_dt:
                mock_dt.now.return_value = MagicMock(month=5, tzinfo=timezone.utc)
                result = get_top_cyclical(["AAPL"])
        assert result == [], "ticker with insufficient history must be excluded"

    def test_win_rate_calculation(self):
        """Known prices: 8 months close > open, 2 months close < open → win_rate ≈ 0.80."""
        # Build rows with distinct year-months all in May (month 5).
        # get_top_cyclical groups by year-month and uses first/last close.
        rows = []
        for i in range(8):   # wins: last close (28th) > first close (1st)
            year = 2010 + i
            rows.append({"date": f"{year}-05-01", "close": 100.0, "volume": 1e6})
            rows.append({"date": f"{year}-05-28", "close": 105.0, "volume": 1e6})
        for i in range(2):   # losses: last close < first close
            year = 2018 + i
            rows.append({"date": f"{year}-05-01", "close": 100.0, "volume": 1e6})
            rows.append({"date": f"{year}-05-28", "close": 95.0,  "volume": 1e6})
        rows.sort(key=lambda r: r["date"], reverse=True)   # newest-first

        with self._patch_fmp(rows):
            with patch("backend.market_intel.satellite_factors.datetime") as mock_dt:
                mock_dt.now.return_value = MagicMock(month=5, tzinfo=timezone.utc)
                result = get_top_cyclical(["PLTR"])

        assert len(result) == 1
        assert result[0]["ticker"] == "PLTR"
        assert math.isclose(result[0]["win_rate"], 0.8, rel_tol=0.1)

    def test_returns_at_most_top_n(self):
        """Result list is capped at TOP_N entries."""
        tickers = [f"T{i}" for i in range(TOP_N + 2)]

        def _side(ticker, limit=280):
            rows = []
            for y in range(MIN_MONTHLY_OBSERVATIONS + 2):
                rows.append({"date": f"201{y % 9}-05-01", "close": 100.0, "volume": 1e6})
                rows.append({"date": f"201{y % 9}-05-28", "close": 105.0, "volume": 1e6})
            rows.sort(key=lambda r: r["date"], reverse=True)
            return rows

        with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
                   side_effect=_side):
            with patch("backend.market_intel.satellite_factors.datetime") as mock_dt:
                mock_dt.now.return_value = MagicMock(month=5, tzinfo=timezone.utc)
                result = get_top_cyclical(tickers)

        assert len(result) <= TOP_N

    def test_returns_empty_on_fmp_exception(self):
        """Any exception from FMP returns []."""
        with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
                   side_effect=RuntimeError("network error")):
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
        with patch(
            "backend.market_intel.satellite_factors._fetch_fmp_info",
            return_value=self._good_info(pe=PE_MAX + 1),
        ):
            result = get_top_cannibals(["PLTR"], self._FMP_KEY, self._MARKET_CAPS)
        assert result == []

    def test_filters_price_above_52w_band(self):
        """Ticker priced above PRICE_VS_52W_LOW_MAX * 52w-low is excluded."""
        with patch(
            "backend.market_intel.satellite_factors._fetch_fmp_info",
            return_value=self._good_info(price=100.0, low_52w=50.0),
        ):
            result = get_top_cannibals(["PLTR"], self._FMP_KEY, self._MARKET_CAPS)
        assert result == []

    def test_zero_market_cap_skipped(self):
        """Ticker with market_cap = 0 is skipped — no ZeroDivisionError."""
        quarters = self._fmp_quarters()
        with patch(
            "backend.market_intel.satellite_factors._fetch_fmp_info",
            return_value=self._good_info(),
        ):
            with patch(
                "regime_trader.services.fmp_client.FMPClient.get_cash_flow_statements",
                return_value=quarters,
            ):
                result = get_top_cannibals(["PLTR"], self._FMP_KEY, {"PLTR": 0.0})
        assert result == []

    def test_missing_fmp_key_returns_empty(self):
        """No FMP key → return [] without any HTTP call."""
        result = get_top_cannibals(["PLTR"], "", {"PLTR": 1e10})
        assert result == []

    def test_buyback_yield_calculated_correctly(self):
        """Correct buyback_yield = total_repurchased / market_cap."""
        quarters = [{"repurchasedCommonStock": -250_000_000}] * 4
        with patch(
            "backend.market_intel.satellite_factors._fetch_fmp_info",
            return_value=self._good_info(pe=10.0, price=19.0, low_52w=18.0),
        ):
            with patch(
                "regime_trader.services.fmp_client.FMPClient.get_cash_flow_statements",
                return_value=quarters,
            ):
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
                 "sector": "Technology", "market_cap": 1e12, "cap_tier": "large",
                 "market": "USA"}
                for t in tickers
            ],
        }

    def test_main_writes_satellite_json(self, tmp_path, monkeypatch):
        """main() with mock data writes satellite_insights.json."""
        from backend.market_intel.satellite_factors import main as sat_main

        top_lists = self._make_top_lists(["AAPL", "MSFT"])
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "top_lists.json").write_text(
            json.dumps(top_lists), encoding="utf-8"
        )

        monkeypatch.setenv("FMP_API_KEY", "fake_key")

        with patch("backend.market_intel.satellite_factors.get_top_cyclical", return_value=[]):
            with patch("backend.market_intel.satellite_factors.get_top_cannibals", return_value=[]):
                import sys
                sys.argv = ["satellite_factors", "--log-dir", str(log_dir), "--verbose"]
                try:
                    sat_main()
                except SystemExit:
                    pass

        assert (log_dir / "satellite_insights.json").exists()
