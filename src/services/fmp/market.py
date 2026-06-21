# Path: src/services/fmp/market.py
"""MarketDataMixin — market endpoint methods for FMPClient."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class MarketDataMixin:

    def get_company_screener(
        self,
        *,
        exchange: Optional[str] = None,
        market_cap_more_than: Optional[float] = None,
        market_cap_lower_than: Optional[float] = None,
        volume_more_than: Optional[float] = None,
        is_actively_trading: Optional[bool] = None,
        limit: int = 100,
        include_etf: bool = False,
    ) -> List[Dict]:
        """Dynamic universe candidates from stable/company-screener.

        Thin wrapper (CLAUDE.md §2 — client carries zero trading math): returns
        the raw screener rows; liquidity/cap filtering and ranking belong to
        src/ingestion/universe_screener.py. ETFs and funds are excluded by
        default so the universe holds operating companies only.

        ``market_cap_lower_than`` applies a server-side cap ceiling (isolating the
        small/mid band without fetching mega-caps); ``is_actively_trading`` drops
        halted/delisted shells (a small/mid-cap liquidity-trap guard). Beta is not
        screened here — it rides along in each row for downstream soft-rank use.

        Returns [] when the API key is absent or the route yields no rows.
        """
        if not self._api_key:
            return []
        params: Dict[str, Any] = {
            "limit": limit,
            "isEtf": str(include_etf).lower(),
            "isFund": "false",
        }
        if exchange:
            params["exchange"] = exchange
        if market_cap_more_than is not None:
            params["marketCapMoreThan"] = int(market_cap_more_than)
        if market_cap_lower_than is not None:
            params["marketCapLowerThan"] = int(market_cap_lower_than)
        if volume_more_than is not None:
            params["volumeMoreThan"] = int(volume_more_than)
        if is_actively_trading is not None:
            params["isActivelyTrading"] = str(is_actively_trading).lower()

        cache_key = (f"{exchange}_{market_cap_more_than}_{market_cap_lower_than}"
                     f"_{volume_more_than}_{is_actively_trading}"
                     f"_{limit}_{include_etf}")
        cached = self._cache_read("screener", cache_key)
        if cached is not None:
            return cached
        data = self._get("company-screener", params, bucket="screener") or []
        result = data if isinstance(data, list) else []
        if result:
            self._cache_write("screener", cache_key, result)
        return result

    def get_historical_prices(self, ticker: str, limit: int = 280) -> List[Dict]:
        """Daily OHLCV from stable/historical-price-eod/full.

        Returns list of dicts sorted newest-first:
            [{symbol, date, open, high, low, close, volume, change, changePercent, vwap}, ...]

        Works for US, EU (SAP.DE), Asia (7203.T), indices (^VIX, ^TNX),
        ETFs (SPY, GLD), and futures (CL=F, GC=F) — confirmed in Phase-0 tests.

        Args:
            limit: Number of trading days to return. 280 ≈ 13 months (12-1m window).
        """
        if not self._api_key:
            return []
        cached = self._cache_read("quote", f"hist_{ticker}_{limit}")
        if cached is not None:
            return cached
        data = self._get("historical-price-eod/full",
                         {"symbol": ticker, "limit": limit},
                         bucket="quote") or []
        result = data if isinstance(data, list) else []
        if result:
            self._cache_write("quote", f"hist_{ticker}_{limit}", result)
        return result

    def get_quote(self, ticker: str, bypass_cache: bool = False) -> Dict:
        """Return quote dict {price, marketCap, volume, avgVolume, eps, ...}.

        International suffixes (SAP.DE, 7203.T) confirmed live on Ultimate
        per Phase-0 smoke-test (2026-05-30).
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("quote", ticker, bypass_cache=bypass_cache)
        if cached is not None:
            return cached
        data = self._get("quote", {"symbol": ticker}, bucket="quote") or []
        result: Dict = data[0] if isinstance(data, list) and data else {}
        if result:
            self._cache_write("quote", ticker, result)
        return result

    def get_batch_quotes(self, tickers: List[str]) -> Dict[str, Dict]:
        """Batch quote (stable/batch-quote). PASS in smoke-test.

        One call for up to 100 tickers instead of N serial calls.
        Uses Ultimate bulk capability.
        """
        if not self._api_key or not tickers:
            return {}
        out: Dict[str, Dict] = {}
        CHUNK = 100
        for i in range(0, len(tickers), CHUNK):
            chunk = tickers[i:i + CHUNK]
            data = self._get("batch-quote", {"symbols": ",".join(chunk)},
                             bucket="quote") or []
            for row in (data if isinstance(data, list) else []):
                sym = row.get("symbol")
                if sym:
                    out[sym] = row
                    # Populate the per-ticker "quote" bucket so a downstream
                    # get_quote(sym) is a cache hit within the 5m TTL — the
                    # batch row IS the stable/quote schema, so this removes
                    # redundant per-ticker quote round-trips at no data cost.
                    self._cache_write("quote", sym, row)
        return out
