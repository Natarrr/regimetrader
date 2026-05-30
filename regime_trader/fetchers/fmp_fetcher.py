"""regime_trader.fetchers.fmp_fetcher — EU/Asia market data via FMP stable/.

Phase-0 smoke-test (2026-05-30) confirmed FMP Ultimate stable/ routes work
for EU (SAP.DE) and Asia (7203.T): historical-price-eod/full, quote, ratios-ttm.
Previously used yfinance for EU/Asia price/volume — now fully on FMP.

Available factors per ticker:
  - return_12_1m:   Jegadeesh-Titman 12-1 month momentum (FMP historical prices)
  - volume_spike:   5d / 90d volume ratio (FMP historical prices)

Structurally absent (no FMP source):
  - insider data (EU MAR / Japan EDINET not integrated)
  - news sentiment/buzz (FMP news/stock returns empty for non-US tickers)
  - congress disclosures (no STOCK Act equivalent outside US)
"""
from __future__ import annotations

import logging
from .base import BaseMarketFetcher, MarketEnum, TickerEntry

logger = logging.getLogger(__name__)

_VOL_BASELINE_BARS = 90
_VOL_BASELINE_SKIP = 5
_VOL_MAX_SPIKE     = 20.0
_MIN_BARS_MOMENTUM = 252
_PRICE_LIMIT       = 280   # 13 months of trading days


class FMPFetcher(BaseMarketFetcher):
    """EU/Asia market data fetcher using FMP stable/historical-price-eod/full.

    Class name preserved for backward compatibility with Orchestrator.
    FMP Ultimate confirmed live for EU and Asia symbols (Phase-0 smoke-test).
    """

    def __init__(self, api_key: str = "", market: MarketEnum = MarketEnum.EUROPE) -> None:
        self._api_key = api_key
        self._market  = market

    @property
    def market(self) -> MarketEnum:
        return self._market

    def source_reliability(self, ticker: str) -> float:
        return 0.85   # FMP exchange data — higher than yfinance scraping

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        """Fetch price/volume for EU/Asia tickers via FMP historical prices.

        Returns TickerEntry list with raw_factors:
          - return_12_1m: float | None  (None if < 252 bars — recent IPO)
          - volume_spike: float         (0.0 if no volume data)
        """
        from regime_trader.services.fmp_client import (  # noqa: PLC0415
            FMPClient, fmp_prices_to_arrays,
        )

        client  = FMPClient(api_key=self._api_key)
        entries: list[TickerEntry] = []

        for ticker in tickers:
            try:
                rows = client.get_historical_prices(ticker, limit=_PRICE_LIMIT)
                closes, volumes, _ = fmp_prices_to_arrays(rows)

                if len(closes) < 5:
                    logger.warning("FMPFetcher: no price data for %s — skipping", ticker)
                    continue

                # 12-1m momentum (Jegadeesh-Titman 1993)
                if len(closes) >= _MIN_BARS_MOMENTUM:
                    idx_far   = max(0, len(closes) - _MIN_BARS_MOMENTUM)
                    idx_near  = max(1, len(closes) - 21)
                    p_far, p_near = closes[idx_far], closes[idx_near]
                    return_12_1m  = (p_near - p_far) / p_far if p_far != 0 else None
                else:
                    return_12_1m = None

                # Volume attention spike
                volume_spike = 0.0
                n_vol = len(volumes)
                if n_vol > _VOL_BASELINE_SKIP + 5:
                    baseline_end   = max(0, n_vol - _VOL_BASELINE_SKIP)
                    baseline_start = max(0, baseline_end - _VOL_BASELINE_BARS)
                    avg_vol  = sum(volumes[baseline_start:baseline_end]) / max(1, baseline_end - baseline_start)
                    last_vol = volumes[-1]
                    if avg_vol > 0:
                        volume_spike = min(last_vol / avg_vol, _VOL_MAX_SPIKE)

                entries.append(TickerEntry(
                    ticker=ticker,
                    market=self._market,
                    sector="",
                    cap_tier="",
                    source_reliability=self.source_reliability(ticker),
                    raw_factors={
                        "return_12_1m": return_12_1m,
                        "volume_spike": volume_spike,
                    },
                ))

            except Exception as exc:
                logger.warning("FMPFetcher: skip %s: %s", ticker, exc)

        return entries
