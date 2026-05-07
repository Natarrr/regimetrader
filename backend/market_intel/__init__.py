"""backend/market_intel — EDGAR-first market intelligence pipeline.

Public surface:
    fetch_intel(ticker)           — adapter, EDGAR primary / FMP fallback
    fetch_edgar_for_ticker(ticker) — direct EDGAR ingest
    parse_form4_file(path)         — Form-4 XML parser
    normalize_events(events, src)  — schema normalizer
    score_events(events)           — buy/sell cluster → 0–1 score
"""
from __future__ import annotations

from .adapter import fetch_intel
from .edgar_ingest import fetch_edgar_for_ticker
from .edgar_parse import parse_form4_file
from .normalizer import normalize_events, InsiderEvent, InstitutionHolding
from .scorer import score_events

__all__ = [
    "fetch_intel",
    "fetch_edgar_for_ticker",
    "parse_form4_file",
    "normalize_events",
    "score_events",
    "InsiderEvent",
    "InstitutionHolding",
]
