# Path: tests/test_source_diagnostics.py
"""Sprint 4: source_diagnostics tagging — api_error vs no_coverage.

Tests _schema_gate() annotates missing_sources with the reason tag when
the row carries a source_diagnostics dict (produced by fmp_fetcher.py).
"""
from __future__ import annotations

import pytest


def _make_row(factor_fields: dict, source_diagnostics: dict = None) -> dict:
    """Build a minimal raw row for _schema_gate."""
    from backend.market_intel.generate_top_lists import FACTOR_FIELDS
    # All factors default to 1.0 (present), caller can override to 0.0/None
    row = {field: 1.0 for field in FACTOR_FIELDS.values()}
    row.update(factor_fields)
    if source_diagnostics is not None:
        row["source_diagnostics"] = source_diagnostics
    row.setdefault("ticker", "TEST")
    row.setdefault("esg_flag", False)
    return row


def _gate(rows):
    from backend.market_intel.generate_top_lists import _schema_gate
    _schema_gate(rows, universe_size=len(rows), enforce_circuit_breaker=False)
    return rows


class TestSchemaGateDiagnostics:
    def test_api_error_annotation_in_missing_sources(self):
        row = _make_row(
            {"news_sentiment_score": None},
            source_diagnostics={"news_sentiment": "api_error"},
        )
        _gate([row])
        assert "news_sentiment:api_error" in row["_validation"]["missing_sources"]

    def test_no_coverage_annotation_in_missing_sources(self):
        row = _make_row(
            {"analyst_revision_score": 0.0},
            source_diagnostics={"analyst_revision": "no_coverage"},
        )
        _gate([row])
        assert "analyst_revision:no_coverage" in row["_validation"]["missing_sources"]

    def test_factor_with_no_diagnosis_keeps_plain_name(self):
        # Factor is zero but no source_diagnostics entry → stays plain (legacy)
        row = _make_row({"congress_score": 0.0})
        _gate([row])
        assert "congress" in row["_validation"]["missing_sources"]
        # Ensure no accidental colon appended
        for m in row["_validation"]["missing_sources"]:
            if m.startswith("congress"):
                assert ":" not in m

    def test_present_factors_not_in_missing_sources(self):
        row = _make_row(
            {},
            source_diagnostics={"news_sentiment": "no_coverage"},
        )
        _gate([row])
        # news_sentiment_score = 1.0 → not missing even if diagnostics present
        for m in row["_validation"]["missing_sources"]:
            assert not m.startswith("news_sentiment")

    def test_multiple_mixed_diagnostics(self):
        row = _make_row(
            {"news_sentiment_score": None, "fcf_yield_score": 0.0},
            source_diagnostics={
                "news_sentiment": "api_error",
                "fcf_yield": "no_coverage",
            },
        )
        _gate([row])
        ms = row["_validation"]["missing_sources"]
        assert "news_sentiment:api_error" in ms
        assert "fcf_yield:no_coverage" in ms
