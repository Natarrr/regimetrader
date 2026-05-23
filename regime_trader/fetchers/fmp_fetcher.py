"""regime_trader.fetchers.fmp_fetcher — EU/Asia market data via yfinance.

Fix #5 (2026-05-23): FMP Ultimate plan returns 403 Forbidden for all non-US
symbols (quote, insider, news). yfinance provides universal coverage for all
major EU/Asia exchange-listed symbols (price, volume).

FMPFetcher now uses yfinance for EUROPE and ASIA markets. The class name is
preserved for backward compatibility with the Orchestrator wiring in run_pipeline.py.

Available factors per ticker:
  - return_12_1m:   Jegadeesh-Titman 12-1 month momentum (yfinance)
  - volume_spike:   Last-day volume / 90-day avg volume (yfinance)
  These map to momentum_long_score and volume_attention_score in _score_ticker_international.

Structurally absent (no source):
  - insider data (EU MAR / Japan EDINET not integrated)
  - news sentiment/buzz (FMP 403, yfinance English coverage unreliable for non-US)
  - congress disclosures (no STOCK Act equivalent outside US)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .base import BaseMarketFetcher, MarketEnum, TickerEntry

logger = logging.getLogger(__name__)

# Volume baseline: 90-bar rolling mean, skipping the 5 most recent bars to avoid
# look-ahead contamination on the current trading day.
_VOL_BASELINE_BARS = 90
_VOL_BASELINE_SKIP = 5
_VOL_MAX_SPIKE = 20.0  # cap at 20× to prevent outlier distortion

# Minimum bars required to compute 12-1m return (Jegadeesh-Titman formation window).
_MIN_BARS_FOR_MOMENTUM = 252

# yfinance download period — 13 months covers the full 12-1m formation window.
_YF_PERIOD = "13mo"

# Delay between yfinance downloads to avoid triggering rate limits.
_YF_RATE_DELAY = 0.3


class FMPFetcher(BaseMarketFetcher):
    """EU/Asia market data fetcher using yfinance.

    Class name preserved for backward compatibility with Orchestrator.
    api_key parameter accepted but unused (FMP returns 403 for non-US symbols).
    """

    def __init__(self, api_key: str = "", market: MarketEnum = MarketEnum.EUROPE) -> None:
        self._api_key = api_key  # preserved but unused for EU/Asia
        self._market = market

    @property
    def market(self) -> MarketEnum:
        return self._market

    def source_reliability(self, ticker: str) -> float:
        # yfinance: scraped from Yahoo Finance — lower confidence than SEC EDGAR.
        # Preserved as informative metadata; no longer used as a score multiplier (Fix #5).
        return 0.60

    def prepare(self, tickers: list[str]) -> list[TickerEntry]:
        """Fetch price/volume data for EU/Asia tickers via yfinance.

        Returns TickerEntry list with raw_factors:
          - return_12_1m: float | None  (None if < 252 bars — recent IPO)
          - volume_spike: float         (0.0 if no volume data)

        Structurally absent factors are NOT included in raw_factors (None key
        is handled by _score_ticker_international → factor output = None).
        """
        import yfinance as yf

        entries: list[TickerEntry] = []
        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    period=_YF_PERIOD,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if raw is None or raw.empty:
                    logger.warning("FMPFetcher: no yfinance data for %s — skipping", ticker)
                    continue

                # yfinance returns multi-level columns when a single ticker is downloaded
                # in newer versions: (field, ticker). Squeeze to flat.
                if hasattr(raw.columns, "levels"):
                    raw.columns = raw.columns.droplevel(1)

                close = raw["Close"].dropna()
                vol   = raw["Volume"].dropna() if "Volume" in raw.columns else None

                # ── 12-1m momentum (Jegadeesh-Titman 1993) ────────────────────
                if len(close) >= _MIN_BARS_FOR_MOMENTUM:
                    # Formation: bar at -252 to bar at -21 (skip most recent month)
                    idx_far  = max(0, len(close) - _MIN_BARS_FOR_MOMENTUM)
                    idx_near = max(1, len(close) - 21)
                    p_far    = float(close.iloc[idx_far])
                    p_near   = float(close.iloc[idx_near])
                    return_12_1m = (p_near - p_far) / p_far if p_far != 0 else None
                else:
                    return_12_1m = None

                # ── Volume attention spike ─────────────────────────────────────
                if vol is not None and len(vol) > _VOL_BASELINE_SKIP + 5:
                    baseline_end   = max(0, len(vol) - _VOL_BASELINE_SKIP)
                    baseline_start = max(0, baseline_end - _VOL_BASELINE_BARS)
                    avg_vol = float(vol.iloc[baseline_start:baseline_end].mean())
                    last_vol = float(vol.iloc[-1])
                    if avg_vol > 0:
                        volume_spike = min(last_vol / avg_vol, _VOL_MAX_SPIKE)
                    else:
                        volume_spike = 0.0
                else:
                    volume_spike = 0.0

                entries.append(TickerEntry(
                    ticker=ticker,
                    market=self._market,
                    sector="",    # filled by orchestrator from ticker_registry.json
                    cap_tier="",  # filled by orchestrator from ticker_registry.json
                    source_reliability=self.source_reliability(ticker),
                    raw_factors={
                        "return_12_1m":  return_12_1m,  # None → dead signal for momentum
                        "volume_spike":  volume_spike,
                    },
                ))
                time.sleep(_YF_RATE_DELAY)

            except Exception as exc:
                logger.warning("FMPFetcher: skip %s: %s", ticker, exc)

        return entries
