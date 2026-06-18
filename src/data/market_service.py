# Path: src/data/market_service.py
"""Market price data service — historical OHLCV via yfinance."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


class MarketData:
    """Thin wrapper around yfinance for historical price bars."""

    @staticmethod
    def get_historical_bars(
        symbol: str,
        years_back: int = 5,
    ) -> pd.DataFrame:
        """Return daily OHLCV DataFrame for *symbol* going back *years_back* years.

        Columns: Open, High, Low, Close, Volume, Adj Close (yfinance default).
        Returns empty DataFrame on failure — callers must handle gracefully.
        """
        try:
            import yfinance as yf  # noqa: PLC0415
            end   = pd.Timestamp.now()
            start = end - pd.DateOffset(years=years_back)
            df = yf.download(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                log.warning("MarketData: no data for %s", symbol)
                return pd.DataFrame()
            return df.sort_index()
        except Exception as exc:
            log.warning("MarketData %s failed: %s", symbol, exc)
            return pd.DataFrame()

    @staticmethod
    def get_log_returns(symbol: str, years_back: int = 5) -> pd.Series:
        """Convenience: daily log returns for *symbol*."""
        import numpy as np  # noqa: PLC0415
        df = MarketData.get_historical_bars(symbol, years_back)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return pd.Series(dtype=float)
        return np.log(closes / closes.shift(1)).dropna()
