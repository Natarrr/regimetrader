"""tests/market_intel/test_parse_form4.py — Form-4 XML parser unit tests.

Validates the parser against three fixture variants:
  1. Single CEO purchase (acquired)
  2. Multi-row CFO sale (disposed)
  3. Amendment (4/A) with namespace + 10% owner
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.market_intel.edgar_parse import parse_form4_file
from backend.market_intel.normalizer import normalize_events
from backend.market_intel.scorer import score_events


_FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_ceo_purchase():
    events = parse_form4_file(_FIXTURES / "form4_purchase.xml")
    assert len(events) == 1
    e = events[0]
    assert e["issuer_ticker"] == "AAPL"
    assert e["reporting_person"] == "COOK TIMOTHY D"
    assert e["reporting_role"] == "CEO"
    assert e["transaction_date"] == "2026-05-03"
    assert e["transaction_code"] == "P"
    assert e["acquired_disposed"] == "A"
    assert e["shares"] == 1000.0
    assert e["price"] == 150.25
    assert e["value"] == 150250.0
    assert e["is_amendment"] is False


def test_parse_cfo_sale_multi_row():
    events = parse_form4_file(_FIXTURES / "form4_sale.xml")
    assert len(events) == 2, "expected two non-derivative transactions"
    assert all(ev["transaction_code"] == "S" for ev in events)
    assert all(ev["acquired_disposed"] == "D" for ev in events)
    assert events[0]["reporting_role"] == "CFO"
    # Row 1: 2500 × 420.50 = 1,051,250
    assert events[0]["value"] == pytest.approx(1_051_250.0, rel=1e-9)
    # Row 2: 1500 × 421.10 = 631,650
    assert events[1]["value"] == pytest.approx(631_650.0, rel=1e-9)


def test_parse_amendment_with_namespace():
    events = parse_form4_file(_FIXTURES / "form4_amendment.xml")
    assert len(events) == 1
    e = events[0]
    assert e["issuer_ticker"] == "TSLA"
    assert e["reporting_role"] == "CEO"   # officer + CEO title wins over director/10%
    assert e["transaction_code"] == "P"
    assert e["is_amendment"] is True
    assert e["value"] == pytest.approx(50000 * 185.40, rel=1e-9)


def test_normalize_attaches_source():
    events = parse_form4_file(_FIXTURES / "form4_purchase.xml")
    norm = normalize_events(events, source="EDGAR")
    assert all(n["source"] == "EDGAR" for n in norm)
    assert all("reporting_role" in n for n in norm)


def test_score_ceo_buy_lifts_above_neutral():
    events = parse_form4_file(_FIXTURES / "form4_purchase.xml")
    breakdown = score_events(normalize_events(events, source="EDGAR"))
    # CEO buy with $150k value → ceo_buy floor of 0.62 applies
    assert breakdown["ceo_buy"] is True
    assert breakdown["score"] >= 0.62
    assert breakdown["buy_count"] == 1


def test_score_cfo_sales_below_neutral():
    events = parse_form4_file(_FIXTURES / "form4_sale.xml")
    breakdown = score_events(normalize_events(events, source="EDGAR"))
    assert breakdown["sell_count"] == 2
    assert breakdown["buy_count"] == 0
    assert breakdown["net_value"] < 0
    # CFO sells totaling ~$1.7M → score below neutral but above 0
    assert breakdown["score"] < 0.50
    assert breakdown["score"] >= 0.0


def test_score_empty_returns_neutral():
    breakdown = score_events([])
    assert breakdown["score"] == 0.50
    assert breakdown["events_in_window"] == 0
    assert breakdown["ceo_buy"] is False
