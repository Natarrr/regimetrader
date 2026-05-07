"""tests/market_intel/test_adapter.py — adapter integration tests with mocks.

Validates priority logic (EDGAR > FMP > NONE) and schema invariants without
hitting the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.market_intel import adapter
from backend.market_intel.edgar_parse import parse_form4_file


_FIXTURES = Path(__file__).parent / "fixtures"


def _fake_edgar_present(*args, **kwargs):
    """Stand-in for _try_edgar that returns a parsed sample."""
    return parse_form4_file(_FIXTURES / "form4_purchase.xml")


def _fake_edgar_empty(*args, **kwargs):
    return []


def _fake_fmp_present(ticker: str, limit: int = 50):
    return [{
        "type":              "Form-4",
        "issuer_ticker":     ticker,
        "issuer_name":       "FMP Test Co",
        "reporting_person":  "DIRECTOR JANE",
        "reporting_role":    "Director",
        "transaction_date":  "2026-04-30",
        "transaction_code":  "P",
        "shares":            500.0,
        "price":             45.0,
        "value":             22_500.0,
        "acquired_disposed": "A",
        "filing_accession":  "0001234567-26-000001",
        "is_amendment":      False,
    }]


def _fake_fmp_empty(ticker: str, limit: int = 50):
    return []


# ── Priority: EDGAR present → use EDGAR ──────────────────────────────────────

def test_adapter_uses_edgar_when_present(monkeypatch):
    monkeypatch.setattr(adapter, "_try_edgar", _fake_edgar_present)
    monkeypatch.setattr(adapter, "fetch_fmp_for_ticker", _fake_fmp_present)
    res = adapter.fetch_intel("AAPL")
    assert res["source"] == "EDGAR"
    assert res["is_authoritative"] is True
    assert res["presence"] is True
    assert res["activity_count"] == 1
    assert res["events"][0]["source"] == "EDGAR"
    assert res["events"][0]["reporting_role"] == "CEO"


# ── Fallback: EDGAR empty → use FMP ──────────────────────────────────────────

def test_adapter_falls_back_to_fmp(monkeypatch):
    monkeypatch.setattr(adapter, "_try_edgar", _fake_edgar_empty)
    monkeypatch.setattr(adapter, "fetch_fmp_for_ticker", _fake_fmp_present)
    monkeypatch.setattr(adapter.config, "FMP_API_KEY", "fake-key")
    res = adapter.fetch_intel("XYZ")
    assert res["source"] == "FMP"
    assert res["is_authoritative"] is False
    assert res["presence"] is True
    assert res["activity_count"] == 1
    assert res["events"][0]["source"] == "FMP"
    assert res["events"][0]["reporting_role"] == "Director"


# ── Both empty → NONE ────────────────────────────────────────────────────────

def test_adapter_returns_none_when_no_data(monkeypatch):
    monkeypatch.setattr(adapter, "_try_edgar", _fake_edgar_empty)
    monkeypatch.setattr(adapter, "fetch_fmp_for_ticker", _fake_fmp_empty)
    monkeypatch.setattr(adapter.config, "FMP_API_KEY", "fake-key")
    res = adapter.fetch_intel("UNKNOWN")
    assert res["source"] == "NONE"
    assert res["presence"] is False
    assert res["activity_count"] == 0
    assert res["score"] == 0.50
    assert res["events"] == []


# ── FMP fallback disabled ────────────────────────────────────────────────────

def test_adapter_skips_fmp_when_disabled(monkeypatch):
    monkeypatch.setattr(adapter, "_try_edgar", _fake_edgar_empty)
    monkeypatch.setattr(adapter, "fetch_fmp_for_ticker", _fake_fmp_present)
    res = adapter.fetch_intel("XYZ", use_fmp_fallback=False)
    assert res["source"] == "NONE"


# ── Schema invariants ────────────────────────────────────────────────────────

@pytest.mark.parametrize("source_fn,fmp_fn", [
    (_fake_edgar_present, _fake_fmp_empty),
    (_fake_edgar_empty,   _fake_fmp_present),
    (_fake_edgar_empty,   _fake_fmp_empty),
])
def test_adapter_schema_invariant(monkeypatch, source_fn, fmp_fn):
    monkeypatch.setattr(adapter, "_try_edgar", source_fn)
    monkeypatch.setattr(adapter, "fetch_fmp_for_ticker", fmp_fn)
    monkeypatch.setattr(adapter.config, "FMP_API_KEY", "fake-key")
    res = adapter.fetch_intel("TEST")
    required = {"ticker", "source", "presence", "is_authoritative",
                "activity_count", "events", "score", "score_breakdown",
                "last_updated", "errors"}
    assert required.issubset(res.keys())
    assert res["source"] in ("EDGAR", "FMP", "NONE")
    assert 0.0 <= res["score"] <= 1.0


# ── write_summary_files round-trip ───────────────────────────────────────────

def test_write_summary_files(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_try_edgar", _fake_edgar_present)
    monkeypatch.setattr(adapter.config, "FMP_API_KEY", "")
    results = [adapter.fetch_intel(t) for t in ("AAPL", "MSFT")]
    paths = adapter.write_summary_files(results, tmp_path)
    assert paths["form4_csv"].exists()
    assert paths["debug_json"].exists()
    assert paths["events_json"].exists()

    import json
    debug = json.loads(paths["debug_json"].read_text(encoding="utf-8"))
    assert debug["ticker_count"] == 2
    assert debug["edgar_present_count"] == 2
    assert debug["missing_count"] == 0
