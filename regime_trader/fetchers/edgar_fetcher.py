from __future__ import annotations

from .base import BaseMarketFetcher, MarketEnum, TickerEntry


class EDGARFetcher(BaseMarketFetcher):
    """Wrapper for the existing US EDGAR/Quiver pipeline."""

    @property
    def market(self) -> MarketEnum:
        return MarketEnum.USA

    def source_reliability(self, ticker: str) -> float:
        return 1.0

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        return [
            TickerEntry(
                ticker=f"{t}.US",
                market=MarketEnum.USA,
                sector="",
                cap_tier="",
                source_reliability=self.source_reliability(t),
                raw_factors={},
            )
            for t in tickers
        ]
