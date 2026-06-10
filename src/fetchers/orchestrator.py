from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.fetchers_base import BaseMarketFetcher, MarketEnum, TickerEntry

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs all fetchers in parallel; failures are logged and skipped."""

    def __init__(self, fetchers: list[BaseMarketFetcher], max_workers: int = 4) -> None:
        self._fetchers = {f.market: f for f in fetchers}
        self._max_workers = max_workers

    def run(self, ticker_map: dict[str, list[str]]) -> list[TickerEntry]:
        """
        ticker_map: {"USA": [...], "EUROPE": [...], "ASIA": [...]}
        Returns merged TickerEntry list across all markets.
        """
        futures: dict = {}
        results: list[TickerEntry] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for market_str, tickers in ticker_map.items():
                market = MarketEnum(market_str)
                fetcher = self._fetchers.get(market)
                if fetcher is None:
                    logger.warning("No fetcher registered for market %s", market_str)
                    continue
                futures[pool.submit(fetcher.prepare, tickers)] = market_str

            for fut in as_completed(futures):
                market_str = futures[fut]
                try:
                    entries = fut.result()
                    results.extend(entries)
                    logger.info("Orchestrator: %d entries from %s", len(entries), market_str)
                except Exception as exc:
                    logger.error("Orchestrator: %s fetcher failed — skipping: %s", market_str, exc)

        return results
