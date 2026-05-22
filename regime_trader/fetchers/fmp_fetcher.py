from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .base import BaseMarketFetcher, MarketEnum, TickerEntry

logger = logging.getLogger(__name__)

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_RELIABILITY = 0.75
_RATE_LIMIT_DELAY = 0.25


class FMPFetcher(BaseMarketFetcher):
    """European equities via Financial Modeling Prep API."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def market(self) -> MarketEnum:
        return MarketEnum.EUROPE

    def source_reliability(self, ticker: str) -> float:
        return _RELIABILITY

    def _fetch_quote(self, ticker: str) -> dict[str, Any]:
        url = f"{_FMP_BASE}/quote/{ticker}"
        resp = requests.get(url, params={"apikey": self._api_key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError(f"Empty FMP response for {ticker}")
        return data[0]

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        entries: list[TickerEntry] = []
        for ticker in tickers:
            try:
                quote = self._fetch_quote(ticker)
                price = float(quote.get("price") or 0)
                mktcap = float(quote.get("marketCap") or 0)
                momentum = (float(quote.get("volume") or 0) /
                            max(float(quote.get("avgVolume") or 1), 1)) - 1.0
                entries.append(TickerEntry(
                    ticker=ticker,
                    market=MarketEnum.EUROPE,
                    sector="",
                    cap_tier="",
                    source_reliability=_RELIABILITY,
                    raw_factors={
                        "price": price,
                        "market_cap": mktcap,
                        "momentum": momentum,
                        "eps": float(quote.get("eps") or 0),
                    },
                ))
                time.sleep(_RATE_LIMIT_DELAY)
            except Exception as exc:
                logger.warning("FMPFetcher skip %s: %s", ticker, exc)
        return entries
