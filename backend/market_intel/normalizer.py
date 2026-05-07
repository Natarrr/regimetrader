"""backend/market_intel/normalizer.py — common event schema.

Stiglitz (2001 Nobel) — Information aggregation: heterogeneous source feeds
(EDGAR XML, FMP JSON, Finnhub) only become useful once they share a common
schema. This module is the single source of truth for that schema.

InsiderEvent  — one row per Form-4 transaction.
InstitutionHolding — one row per 13F-HR holding.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional


SourceTag = Literal["EDGAR", "FMP", "FINNHUB", "YFINANCE"]


@dataclass
class InsiderEvent:
    type: str = "Form-4"
    issuer_ticker: Optional[str] = None
    issuer_name: Optional[str] = None
    reporting_person: Optional[str] = None
    reporting_role: str = "Unknown"
    transaction_date: Optional[str] = None        # ISO YYYY-MM-DD
    transaction_code: Optional[str] = None        # P (purchase), S (sale), A (award), F, M, etc.
    shares: Optional[float] = None
    price: Optional[float] = None
    value: Optional[float] = None                 # USD = shares × price
    acquired_disposed: Optional[str] = None       # A | D
    filing_accession: Optional[str] = None
    is_amendment: bool = False
    source: SourceTag = "EDGAR"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstitutionHolding:
    type: str = "13F-HR"
    issuer_name: Optional[str] = None
    issuer_ticker: Optional[str] = None
    cusip: Optional[str] = None
    title_class: Optional[str] = None
    value_usd: Optional[float] = None
    shares: Optional[float] = None
    filing_accession: Optional[str] = None
    holder_name: Optional[str] = None
    is_amendment: bool = False
    source: SourceTag = "EDGAR"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Normalizer ────────────────────────────────────────────────────────────────

def _normalize_one(raw: Dict[str, Any], source: SourceTag) -> Dict[str, Any]:
    """Coerce a single parsed dict into the canonical schema with source tag."""
    if raw.get("type") == "13F-HR" or "title_class" in raw:
        ev = InstitutionHolding(
            issuer_name=raw.get("issuer_name"),
            issuer_ticker=raw.get("issuer_ticker"),
            cusip=raw.get("cusip"),
            title_class=raw.get("title_class"),
            value_usd=raw.get("value_usd"),
            shares=raw.get("shares"),
            filing_accession=raw.get("filing_accession"),
            holder_name=raw.get("holder_name"),
            is_amendment=bool(raw.get("is_amendment", False)),
            source=source,
        )
        return ev.to_dict()

    ev_i = InsiderEvent(
        issuer_ticker=raw.get("issuer_ticker"),
        issuer_name=raw.get("issuer_name"),
        reporting_person=raw.get("reporting_person"),
        reporting_role=raw.get("reporting_role", "Unknown"),
        transaction_date=raw.get("transaction_date"),
        transaction_code=raw.get("transaction_code"),
        shares=raw.get("shares"),
        price=raw.get("price"),
        value=raw.get("value"),
        acquired_disposed=raw.get("acquired_disposed"),
        filing_accession=raw.get("filing_accession"),
        is_amendment=bool(raw.get("is_amendment", False)),
        source=source,
    )
    return ev_i.to_dict()


def normalize_events(raw_events: List[Dict[str, Any]],
                     source: SourceTag = "EDGAR") -> List[Dict[str, Any]]:
    """Normalize a list of raw parsed events to canonical schema. Stable order."""
    return [_normalize_one(r, source) for r in raw_events if isinstance(r, dict)]
