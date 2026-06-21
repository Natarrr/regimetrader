# Path: src/services/fmp/ownership.py
"""OwnershipFlowMixin — ownership endpoint methods for FMPClient."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone
from src.services.fmp.core import FMPEndpointError

log = logging.getLogger(__name__)


class OwnershipFlowMixin:

    def get_congress_trades(self, ticker: str, lookback_days: int = 180) -> Dict:
        """Congressional trading data from public S3 Stock Watcher feeds.

        FMP stable/ senate-trading and house-trading routes return HTTP 404 —
        FMP has not migrated these endpoints from the deprecated v4 paths to
        stable/ as of 2026-05-30. Contact FMP support to request migration.

        Fallback: fetches directly from the free public S3 feeds maintained
        by House/Senate Stock Watcher (no API key required, same source that
        run_pipeline.fetch_congress_buys uses as primary).

        Returns dict matching the 7-factor pipeline contract:
            {purchases, sales, total, net, recency_days, representatives}
        Returns {} when no trades found in the lookback window.
        """
        from datetime import timedelta as _td
        import requests as _req

        cutoff = (datetime.now(timezone.utc) -
                  _td(days=lookback_days)).date().isoformat()
        purchases = sales = total = 0
        recency_days = 9999
        reps: set[str] = set()
        now_date = datetime.now(timezone.utc).date()

        _HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
        _SENATE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"

        cached = self._cache_read("congress", ticker)
        if cached is not None:
            return cached

        # Probe FMP congress routes once per process lifetime.
        # These return HTTP 404 (not in current plan per Phase-0 smoke test).
        # We track the failure in endpoint_failures so fmp_health.json is
        # accurate — the S3 fallback below is the actual data source.
        # Class-level flag avoids per-ticker API calls (probe only needed once).
        if self._api_key and not type(self)._fmp_congress_probe_done:
            type(self)._fmp_congress_probe_done = True
            try:
                self._get(
                    "senate-trading",
                    {"symbol": ticker, "page": 0, "limit": 1},
                    bucket="congress",
                )
                log.info("FMP senate-trading is LIVE — route may have been migrated")
            except FMPEndpointError as exc:
                if exc.status == 404:
                    log.debug(
                        "FMP senate-trading: HTTP 404 (not in plan). "
                        "S3 Stock Watcher fallback active. "
                        "Failure recorded in health_report()."
                    )
            except Exception as exc:
                log.debug("FMP congress probe failed (non-4xx): %s", exc)

        for url, name_key in [(_SENATE_URL, "senator"), (_HOUSE_URL, "representative")]:
            try:
                resp = _req.get(url, timeout=30)
                if resp.status_code == 403:
                    log.warning(
                        "S3 congress feed %s returned 403 — bucket restricted", name_key)
                    continue
                resp.raise_for_status()
                for rec in resp.json():
                    ticker_field = str(rec.get("ticker", "")
                                       or "").upper().strip()
                    if ticker_field != ticker.upper():
                        continue
                    disclosure = (rec.get("disclosure_date")
                                  or rec.get("transaction_date") or "")
                    if not disclosure or disclosure[:10] < cutoff:
                        continue
                    tx_type = (rec.get("type") or rec.get(
                        "transaction_type") or "").lower()
                    if "purchase" in tx_type or "buy" in tx_type:
                        purchases += 1
                    elif "sale" in tx_type or "sold" in tx_type or "sell" in tx_type:
                        sales += 1
                    else:
                        continue
                    total += 1
                    rep = str(rec.get(name_key) or rec.get(
                        "name") or "").strip()
                    if rep:
                        reps.add(rep)
                    try:
                        from datetime import date as _date
                        d = _date.fromisoformat(disclosure[:10])
                        recency_days = min(recency_days, (now_date - d).days)
                    except Exception:
                        pass
            except Exception as exc:
                log.debug("S3 congress feed %s failed: %s", name_key, exc)

        if total == 0:
            self._cache_write("congress", ticker, {})
            return {}

        result = {
            "purchases":      purchases,
            "sales":          sales,
            "total":          total,
            "net":            purchases - sales,
            "recency_days":   recency_days if recency_days < 9999 else None,
            "representatives": sorted(reps),
        }
        self._cache_write("congress", ticker, result)
        return result

    def get_insider_purchases(
        self, ticker: str, lookback_days: int = 180
    ) -> Tuple[float, int]:
        """Return (total_acquisition_usd, days_since_most_recent) from Form 4.

        Filters acquisitionOrDisposition == 'A' only. Empty 200 -> (0.0, 0).
        """
        if not self._api_key:
            return 0.0, 0
        cached = self._cache_read("insider", ticker)
        if cached is not None:
            return tuple(cached)  # type: ignore[return-value]

        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=lookback_days)).date().isoformat()
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])
        data: list = []
        for sym in symbols_to_try:
            data = self._get("insider-trading/search",
                             {"symbol": sym, "page": 0, "limit": 500},
                             bucket="insider") or []
            if data:
                break

        total_usd = 0.0
        most_recent_days = 0
        now_date = datetime.now(timezone.utc).date()

        for r in data:
            # Handle historical FMP typo: acquistionOrDisposition (missing 'i')
            aod = (r.get("acquisitionOrDisposition")
                   or r.get("acquistionOrDisposition") or "")
            if aod != "A":
                continue
            shares = float(r.get("securitiesTransacted") or 0)
            price = float(r.get("price") or 0)
            if shares <= 0 or price <= 0:
                continue
            tx_date = r.get("transactionDate", "")
            if tx_date and tx_date < cutoff:
                continue
            total_usd += shares * price
            try:
                d = datetime.fromisoformat(tx_date[:10]).date()
                most_recent_days = max(most_recent_days, (now_date - d).days)
            except Exception:
                pass

        result = (round(total_usd, 2), most_recent_days)
        self._cache_write("insider", ticker, list(result))
        return result

    def get_insider_transactions(self, ticker: str, lookback_days: int = 90) -> Dict[str, List[Dict]]:
        """Return {'P': [...], 'S': [...]} for the breadth signal.

        score_insider_breadth needs P vs S by distinct insider_id.
        Uses the same stable/insider-trading/search route as get_insider_purchases.
        Falls back to base symbol for EU/Asia tickers.
        """
        if not self._api_key:
            return {"P": [], "S": []}
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])
        data: List[Dict] = []
        for sym in symbols_to_try:
            data = self._get("insider-trading/search",
                             {"symbol": sym, "page": 0, "limit": 500},
                             bucket="insider") or []
            if data:
                break
        cutoff = (datetime.now(timezone.utc) -
                  timedelta(days=lookback_days)).date().isoformat()
        out: Dict[str, List[Dict]] = {"P": [], "S": []}
        for r in data:
            tx_date = str(r.get("transactionDate", ""))[:10]
            if tx_date and tx_date < cutoff:
                continue
            aod = (r.get("acquisitionOrDisposition")
                   or r.get("acquistionOrDisposition") or "")
            entry = {
                "insider_id": r.get("reportingCik") or r.get("reportingName"),
                "title": r.get("typeOfOwner", ""),
                "date": tx_date,
            }
            if aod == "A":
                out["P"].append(entry)
            elif aod == "D":
                out["S"].append(entry)
        return out

    def get_insider_statistics(self, ticker: str) -> List[Dict]:
        """Quarterly insider buy/sell aggregates (stable/insider-trading/statistics).

        Returns the raw quarterly records newest-first (year desc, quarter desc),
        each carrying acquiredTransactions / disposedTransactions /
        acquiredDisposedRatio. Drives the acquired-vs-disposed spike overlay
        (Net Purchase Ratio [Lakonishok & Lee 2001]) — a display/badge signal,
        NOT a weighted scoring factor. Cached under a distinct key so it never
        collides with the (insider, ticker) get_insider_purchases entry.
        Returns [] when the key is absent or the route yields no rows.
        """
        if not self._api_key:
            return []
        cache_key = f"stats_{ticker}"
        cached = self._cache_read("insider", cache_key)
        if cached is not None:
            return cached
        symbols_to_try = [ticker]
        if "." in ticker:
            symbols_to_try.append(ticker.split(".")[0])
        data: List[Dict] = []
        for sym in symbols_to_try:
            data = self._get("insider-trading/statistics",
                             {"symbol": sym}, bucket="insider") or []
            if data:
                break
        result = data if isinstance(data, list) else []
        # newest-first → the spike scorer reads the latest quarter at index 0
        result.sort(
            key=lambda r: (int(r.get("year") or 0), int(r.get("quarter") or 0)),
            reverse=True,
        )
        if result:
            self._cache_write("insider", cache_key, result)
        return result

    def get_institutional_ownership(self, ticker: str) -> Dict:
        """13F institutional holdings summary.

        Uses stable/institutional-ownership/symbol-positions-summary with
        year + quarter params (required — returns HTTP 400 without them).
        Fetches the most recently completed quarter automatically.

        Returns aggregate fields: investorsHolding, investorsHoldingChange,
        increasedPositions, reducedPositions, newPositions, closedPositions,
        numberOf13FsharesChange, ownershipPercent, ownershipPercentChange.
        Returns {} if no data or key absent.
        """
        if not self._api_key:
            return {}
        cached = self._cache_read("f13", ticker)
        if cached is not None:
            return cached

        # Determine the target quarter (13F filings lag ~45 days). The naive
        # now−45d frequently lands inside the CURRENT, not-yet-ended quarter
        # (e.g., June 11 → Apr 27 → Q2, unfiled) — so on an empty response
        # retry the PREVIOUS quarter before concluding "no coverage".
        now = datetime.now(timezone.utc)
        as_of = now.date() - timedelta(days=45)
        year = as_of.year
        quarter = (as_of.month - 1) // 3 + 1
        prev_year, prev_quarter = (
            (year, quarter - 1) if quarter > 1 else (year - 1, 4)
        )

        # 13F filings are SEC-mandated, keyed to US listings — EU/Asia local
        # lines often only have data under the base (ADR) symbol.
        symbols = [ticker]
        if "." in ticker:
            symbols.append(ticker.split(".")[0])

        result: Dict = {}
        for y, q in ((year, quarter), (prev_year, prev_quarter)):
            for sym in symbols:
                data = self._get(
                    "institutional-ownership/symbol-positions-summary",
                    {"symbol": sym, "year": y,
                        "quarter": q, "page": 0, "limit": 1},
                    bucket="f13",
                ) or []
                result = data[0] if isinstance(data, list) and data else {}
                if result:
                    break
            if result:
                break
        if result:
            self._cache_write("f13", ticker, result)
        return result
