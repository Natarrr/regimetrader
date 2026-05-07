"""backend/market_intel/fmp_fallback.py — FMP as authoritative-data fallback.

FMP's /v4/insider-trading endpoint aggregates the same Form-4 data that EDGAR
publishes, but with broader coverage (multiple roles, parsed price/value) and
without requiring per-CIK navigation. We use it when EDGAR returns nothing —
typically for foreign issuers, recent IPOs, or transient SEC outages.

Public function:
    fetch_fmp_for_ticker(ticker, limit=50) -> List[InsiderEvent]
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from . import config

log = logging.getLogger("market_intel.fmp")


# ── Role mapping (FMP's typeOfOwner string → our canonical role) ──────────────

def _map_role(type_of_owner: str) -> str:
    s = (type_of_owner or "").upper()
    if "CEO" in s or "CHIEF EXECUTIVE" in s:
        return "CEO"
    if "CFO" in s or "CHIEF FINANCIAL" in s:
        return "CFO"
    if any(k in s for k in ("PRESIDENT", "COO", "CHAIRMAN")):
        return "Officer-Senior"
    if "OFFICER" in s:
        return "Officer"
    if "DIRECTOR" in s:
        return "Director"
    if "10 PERCENT" in s or "10% OWNER" in s:
        return "10%Owner"
    return "Unknown"


def _to_iso_date(s: Any) -> Optional[str]:
    if not s:
        return None
    try:
        return str(s)[:10]
    except Exception:
        return None


# ── HTTP wrapper ──────────────────────────────────────────────────────────────

def _fmp_get(path: str, params: Dict[str, Any]) -> Any:
    if not config.FMP_API_KEY:
        return None
    url = f"{config.FMP_BASE}{path}"
    params = {**params, "apikey": config.FMP_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_S)
        if resp.status_code == 429:
            time.sleep(2.0)
            resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("fmp_fail path=%s err=%s", path, str(exc)[:200])
        return None


# ── Public ────────────────────────────────────────────────────────────────────

def fetch_fmp_for_ticker(ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent insider transactions for a ticker from FMP, normalize them.

    Returns a list of *raw* event dicts (matching the parser output schema)
    so they can be passed through the same normalizer as EDGAR data.
    """
    ticker = str(ticker).upper().strip()
    data = _fmp_get("/v4/insider-trading", {"symbol": ticker, "limit": limit})
    if not isinstance(data, list):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=config.INSIDER_WINDOW_DAYS)
    out: List[Dict[str, Any]] = []
    for tx in data:
        try:
            tx_date_s = _to_iso_date(tx.get("transactionDate") or tx.get("filingDate"))
            if tx_date_s:
                try:
                    tx_dt = datetime.strptime(tx_date_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if tx_dt < cutoff:
                        continue
                except ValueError:
                    pass

            shares = abs(float(tx.get("securitiesTransacted") or 0) or 0)
            price = abs(float(tx.get("price") or 0) or 0)
            value = round(shares * price, 2) if shares and price else None

            tx_type = str(tx.get("transactionType") or "").upper()
            # FMP transactionType values like "P-Purchase", "S-Sale", "A-Award", "F-Tax", "M-Exempt"
            code = tx_type[:1] if tx_type else None
            if tx_type.startswith("P"):
                code, ad = "P", "A"
            elif tx_type.startswith("S"):
                code, ad = "S", "D"
            else:
                ad = None

            out.append({
                "type":              "Form-4",
                "issuer_ticker":     ticker,
                "issuer_name":       tx.get("companyName"),
                "reporting_person":  tx.get("reportingName"),
                "reporting_role":    _map_role(str(tx.get("typeOfOwner") or "")),
                "transaction_date":  tx_date_s,
                "transaction_code":  code,
                "shares":            shares or None,
                "price":             price or None,
                "value":             value,
                "acquired_disposed": ad,
                "filing_accession":  str(tx.get("link") or tx.get("filingLink") or ""),
                "is_amendment":      False,
            })
        except Exception:
            continue
    return out
