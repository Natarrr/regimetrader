from __future__ import annotations

import logging

import yfinance as yf

from .base import BaseMarketFetcher, MarketEnum, TickerEntry

logger = logging.getLogger(__name__)

_RELIABILITY = 0.6


class AsianMarketFetcher(BaseMarketFetcher):
    """Asian equities via yfinance (Yahoo Finance scraping)."""

    @property
    def market(self) -> MarketEnum:
        return MarketEnum.ASIA

    def source_reliability(self, ticker: str) -> float:
        return _RELIABILITY

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        entries: list[TickerEntry] = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                fi = t.fast_info
                info = t.info
                price = float(fi.last_price or 0)
                mktcap = float(fi.market_cap or 0)
                avg_vol = float(fi.three_month_average_volume or 1)
                vol = float(info.get("regularMarketVolume") or 0)
                momentum = (vol / max(avg_vol, 1)) - 1.0
                eps = float(info.get("trailingEps") or 0)
                entries.append(TickerEntry(
                    ticker=ticker,
                    market=MarketEnum.ASIA,
                    sector="",
                    cap_tier="",
                    source_reliability=_RELIABILITY,
                    raw_factors={
                        "price": price,
                        "market_cap": mktcap,
                        "momentum": momentum,
                        "eps": eps,
                    },
                ))
            except Exception as exc:
                logger.warning("AsianFetcher skip %s: %s", ticker, exc)
        return entries
