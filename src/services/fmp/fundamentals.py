# Path: src/services/fmp/fundamentals.py
"""FundamentalsMixin — fundamentals endpoint methods for FMPClient."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class FundamentalsMixin:

    def get_ratios_ttm(self, ticker: str) -> Dict:
        """TTM financial ratios (stable/ratios-ttm). Falls back to base symbol for EU/Asia."""
        if not self._api_key:
            return {}
        cached = self._cache_read("ratios", ticker)
        if cached is not None:
            return cached
        data = self._get(
            "ratios-ttm", {"symbol": ticker}, bucket="ratios") or []
        result = data[0] if isinstance(data, list) and data else {}
        if not result and "." in ticker:
            base = ticker.split(".")[0]
            cached_base = self._cache_read("ratios", base)
            if cached_base is not None:
                return cached_base
            data2 = self._get("ratios-ttm", {"symbol": base}, bucket="ratios") or []
            result = data2[0] if isinstance(data2, list) and data2 else {}
            if result:
                self._cache_write("ratios", base, result)
        if result:
            self._cache_write("ratios", ticker, result)
        return result

    def get_enterprise_value(self, ticker: str) -> Optional[float]:
        """Most recent enterprise value in USD from stable/enterprise-values.

        Falls back to base symbol for dotted EU/Asia tickers (e.g., ASML.AS → ASML).
        Returns None when FMP has no coverage (not 0.0 — absence is distinct from zero EV).

        Reference: Damodaran (2006) — FCF Yield denominator.
        """
        if not self._api_key:
            return None
        cached = self._cache_read("ev", ticker)
        if cached is not None:
            return float(cached) if cached else None

        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])

        for sym in symbols_to_try:
            data = self._get("enterprise-values", {"symbol": sym, "limit": 1},
                             bucket="key_metrics") or []
            if data and isinstance(data, list):
                ev = data[0].get("enterpriseValue")
                if ev is not None:
                    ev_float = float(ev) or None
                    self._cache_write("ev", ticker, ev_float)
                    return ev_float
        self._cache_write("ev", ticker, None)
        return None

    def get_levered_dcf(self, ticker: str) -> Optional[float]:
        """Model-implied levered-DCF fair value (stable/levered-discounted-cash-flow).

        CANDIDATE-factor source (weight-0): feeds score_dcf_upside. Returns the
        `dcf` field of the most recent record in USD, or None when FMP has no
        coverage. Falls back to the base symbol for dotted EU/Asia tickers
        (e.g. ASML.AS → ASML), matching get_enterprise_value.

        Reference: Damodaran (2012), "Investment Valuation".
        """
        if not self._api_key:
            return None
        cached = self._cache_read("dcf", ticker)
        if cached is not None:
            return float(cached) if cached else None

        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])

        for sym in symbols_to_try:
            data = self._get("levered-discounted-cash-flow", {"symbol": sym},
                             bucket="dcf") or []
            if data and isinstance(data, list):
                dcf = data[0].get("dcf")
                if dcf is not None:
                    dcf_float = float(dcf) or None
                    self._cache_write("dcf", ticker, dcf_float)
                    return dcf_float
        self._cache_write("dcf", ticker, None)
        return None

    def get_sector_pe(
        self, sector: str, exchange: str = "NASDAQ", date: Optional[str] = None
    ) -> Optional[float]:
        """Sector P/E for a peer group (stable/sector-pe-snapshot).

        CANDIDATE-factor source (weight-0): feeds score_sector_relative_value.
        The full snapshot for an (exchange, date) is fetched ONCE and cached as a
        {sector_lower: pe} map, so scoring an N-ticker universe costs one call per
        exchange/date rather than N. `date` defaults to today (UTC); pass an
        explicit as-of date for look-ahead-safe backfill (CLAUDE.md §3).

        Returns the matching sector's P/E (case-insensitive), or None when the
        sector is absent or the snapshot is empty.

        Reference: Asness, Porter & Stevens (2000) — within-industry value.
        """
        if not self._api_key or not sector:
            return None
        d = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = f"{exchange}_{d}"
        snap = self._cache_read("sector_pe", cache_key)
        if not isinstance(snap, dict):
            data = self._get(
                "sector-pe-snapshot", {"date": d, "exchange": exchange},
                bucket="sector_pe") or []
            snap = {}
            for row in (data if isinstance(data, list) else []):
                s = row.get("sector")
                pe = row.get("pe")
                if s and pe is not None:
                    try:
                        snap[str(s).lower()] = float(pe)
                    except (TypeError, ValueError):
                        continue
            self._cache_write("sector_pe", cache_key, snap)
        val = snap.get(str(sector).lower())
        return float(val) if val is not None else None

    def get_quality_score(self, ticker: str) -> tuple[float, int]:
        """Piotroski F-score quality gate from cached ratios-ttm data.

        Calls get_ratios_ttm(ticker) — already cached in "ratios" bucket (24h TTL).
        Zero additional API calls.

        Returns (score, raw_count) where score is in [0, 1] and raw_count is the
        integer F-score (0–8). The raw count is used by _piotroski_gate_multiplier
        to apply the suppress/discount gate independently of the normalised score.

        Dead signal is (0.0, 0) — NOT Optional. This differs from get_upside_to_target
        (which returns None for missing analyst coverage) because quality data is
        universally available for any listed company. A missing ratios response means
        a broken endpoint, not "no quality data for this ticker."

        Returns (0.0, 0) on exception or when get_ratios_ttm() returns empty dict.

        References: Piotroski (2000) JAR; Novy-Marx (2013) JFE.
        """
        if not self._api_key:
            return 0.0, 0
        try:
            from src.scoring.momentum_signals import score_quality_piotroski  # noqa: PLC0415
            ratios = self.get_ratios_ttm(ticker)
            score, raw_count = score_quality_piotroski(ratios)
            return score, raw_count
        except Exception as exc:
            log.debug("get_quality_score %s failed: %s", ticker, exc)
            return 0.0, 0

    def get_cash_flow_statements(self, ticker: str, limit: int = 4) -> List[Dict]:
        """Quarterly cash flow statements (stable/cash-flow-statement). PASS in smoke-test.

        Used by satellite_factors cannibal filter (buyback yield).
        """
        if not self._api_key:
            return []
        # Key by limit — satellite_factors / v3_shadow share limit=4, but the
        # key stays explicit so a future caller with a different limit cannot
        # collide on a truncated row set.
        cache_key = f"{ticker}:cf:q{limit}"
        cached = self._cache_read("key_metrics", cache_key)
        if cached is not None:
            return cached
        data = self._get("cash-flow-statement",
                         {"symbol": ticker, "period": "quarter", "limit": limit},
                         bucket="key_metrics") or []
        result = data if isinstance(data, list) else []
        if result:  # never cache an empty/transient miss for the 24h TTL
            self._cache_write("key_metrics", cache_key, result)
        return result

    def get_income_statements(
        self, ticker: str, period: str = "quarter", limit: int = 8
    ) -> List[Dict]:
        """Income statements (stable/income-statement), quarterly or annual.

        v3.0 margin_expansion input: 8 quarters for the TTM-vs-prior-TTM
        operating-margin delta, or 2 annual rows for the fallback track
        (HKEX semi-annual mandates / JP tanshin make 8 clean quarters rare
        ex-US). Rows carry filingDate for look-ahead-safe ordering; the
        discrete-quarter validation lives in the scorer, not here.
        """
        if not self._api_key:
            return []
        # Key by period AND limit: margin_expansion fetches both (quarter, 8)
        # and (annual, 2) for the SAME ticker in one run — a ticker-only key
        # would alias the two and corrupt the discrete-quarter validation.
        cache_key = f"{ticker}:is:{period}:{limit}"
        cached = self._cache_read("key_metrics", cache_key)
        if cached is not None:
            return cached
        data = self._get("income-statement",
                         {"symbol": ticker, "period": period, "limit": limit},
                         bucket="key_metrics") or []
        result = data if isinstance(data, list) else []
        if result:  # never cache an empty/transient miss for the 24h TTL
            self._cache_write("key_metrics", cache_key, result)
        return result

    def get_balance_sheet(
        self, ticker: str, period: str = "quarter", limit: int = 1
    ) -> List[Dict]:
        """Balance-sheet statements (stable/balance-sheet-statement), newest-first.

        Candidate accruals factor (Sloan 1996) input: totalAssets as the
        deflator for (netIncome − operatingCashFlow). Rows carry filingDate for
        look-ahead-safe anchoring (CLAUDE.md §3). 24h TTL (key_metrics bucket),
        keyed by period+limit like get_income_statements to avoid aliasing.
        """
        if not self._api_key:
            return []
        cache_key = f"{ticker}:bs:{period}:{limit}"
        cached = self._cache_read("key_metrics", cache_key)
        if cached is not None:
            return cached
        data = self._get("balance-sheet-statement",
                         {"symbol": ticker, "period": period, "limit": limit},
                         bucket="key_metrics") or []
        result = data if isinstance(data, list) else []
        if result:  # never cache an empty/transient miss for the 24h TTL
            self._cache_write("key_metrics", cache_key, result)
        return result
