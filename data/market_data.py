from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


class MarketData:
    """Fetch historical OHLCV bars for a single symbol via yfinance."""

    def get_historical_bars(
        self,
        symbol: str,
        years_back: int = 3,
        interval: str = "1d",
    ) -> pd.DataFrame:
        end   = datetime.today()
        start = end - timedelta(days=int(years_back * 365.25))
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"yfinance returned no data for {symbol}")

        # Flatten MultiIndex columns produced by yfinance ≥ 0.2.x
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
