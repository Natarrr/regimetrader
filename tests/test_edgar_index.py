"""tests/test_edgar_index.py
Unit tests for regime_trader.services.edgar_index.

Stiglitz (2001 Nobel) — the daily bulk index must parse, cache, and expose
filings reliably. Every test mocks HTTP; no live SEC requests in CI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from regime_trader.services.edgar_index import (
    EdgarDailyIndex,
    DailyFilingRef,
    _parse_company_idx,
    _evidence_id,
    _quarter,
)
from datetime import date


# ── Sample data ────────────────────────────────────────────────────────────────

# Columns at offsets (0, 62, 74, 86, 98) matching _COL_OFFSETS in edgar_index.py
# [0:62] company  [62:74] form  [74:86] CIK  [86:98] date  [98:] filename
_SAMPLE_IDX = (
    "Company Name                                                  Form Type   CIK         Date Filed  Filename\n"
    + "-" * 100 + "\n"
    + "APPLE INC                                                     4           0000320193  2026-01-15  edgar/data/320193/0000320193-26-000001-index.htm\n"
    + "MICROSOFT CORP                                                4           0000789019  2026-01-15  edgar/data/789019/0000789019-26-000001-index.htm\n"
    + "TESLA INC                                                     13F-HR      0001318605  2026-01-15  edgar/data/1318605/0001318605-26-000001-index.htm\n"
)


def _make_resp(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture()
def fast_idx(tmp_path: Path) -> EdgarDailyIndex:
    """EdgarDailyIndex with high rate (no delay) and temp cache dir."""
    return EdgarDailyIndex(rate_per_sec=1000.0, cache_root=tmp_path / "edgar_idx")


# ── Parser ─────────────────────────────────────────────────────────────────────

class TestParseCompanyIdx:
    def test_parses_expected_number_of_rows(self):
        rows = _parse_company_idx(_SAMPLE_IDX, "2026-01-15")
        assert len(rows) == 3

    def test_row_fields_present(self):
        rows = _parse_company_idx(_SAMPLE_IDX, "2026-01-15")
        r = rows[0]
        assert r["company"]
        assert r["form"]
        assert r["cik"]
        assert r["date"] == "2026-01-15"
        assert r["url"].startswith("https://www.sec.gov/")
        assert len(r["evidence_id"]) == 16

    def test_cik_zero_padded_to_10_digits(self):
        rows = _parse_company_idx(_SAMPLE_IDX, "2026-01-15")
        for r in rows:
            assert len(r["cik"]) == 10
            assert r["cik"].isdigit()

    def test_empty_text_returns_empty_list(self):
        assert _parse_company_idx("", "2026-01-15") == []

    def test_header_only_returns_empty_list(self):
        hdr = "Company Name   Form Type   CIK   Date Filed  Filename\n" + "-" * 80
        assert _parse_company_idx(hdr, "2026-01-15") == []

    def test_form_types_preserved(self):
        rows = _parse_company_idx(_SAMPLE_IDX, "2026-01-15")
        forms = {r["form"] for r in rows}
        assert "4" in forms
        assert "13F-HR" in forms


# ── Evidence ID ────────────────────────────────────────────────────────────────

class TestEvidenceId:
    def test_stable_for_same_inputs(self):
        eid1 = _evidence_id("2026-01-15", "edgar/data/320193/abc.htm")
        eid2 = _evidence_id("2026-01-15", "edgar/data/320193/abc.htm")
        assert eid1 == eid2

    def test_different_for_different_filenames(self):
        e1 = _evidence_id("2026-01-15", "edgar/data/320193/a.htm")
        e2 = _evidence_id("2026-01-15", "edgar/data/320193/b.htm")
        assert e1 != e2

    def test_length_16(self):
        assert len(_evidence_id("2026-01-15", "any/path")) == 16


# ── Quarter helper ────────────────────────────────────────────────────────────

class TestQuarter:
    def test_jan_is_q1(self):
        assert _quarter(date(2026, 1, 15)) == 1

    def test_apr_is_q2(self):
        assert _quarter(date(2026, 4, 1)) == 2

    def test_jul_is_q3(self):
        assert _quarter(date(2026, 7, 31)) == 3

    def test_oct_is_q4(self):
        assert _quarter(date(2026, 10, 1)) == 4


# ── EdgarDailyIndex.list_filings ──────────────────────────────────────────────

class TestListFilings:
    def test_weekend_returns_empty(self, fast_idx: EdgarDailyIndex):
        # 2026-01-17 is a Saturday
        result = fast_idx.list_filings("2026-01-17")
        assert result == []

    def test_sunday_returns_empty(self, fast_idx: EdgarDailyIndex):
        assert fast_idx.list_filings("2026-01-18") == []

    def test_cache_miss_fetches_http(self, fast_idx: EdgarDailyIndex):
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            rows = fast_idx.list_filings("2026-01-15")
        assert len(rows) == 3

    def test_cache_hit_skips_http(self, fast_idx: EdgarDailyIndex):
        # Prime the cache
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            fast_idx.list_filings("2026-01-15")

        # Second call should NOT hit HTTP
        mock_get = MagicMock(return_value=_make_resp(_SAMPLE_IDX))
        with patch.object(fast_idx._session, "get", mock_get):
            fast_idx.list_filings("2026-01-15")

        mock_get.assert_not_called()

    def test_cache_is_atomic(self, fast_idx: EdgarDailyIndex, tmp_path: Path):
        """No temp files should remain after successful cache write."""
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            fast_idx.list_filings("2026-01-15")

        cache_dir = fast_idx._cache_root
        leftovers = [p for p in cache_dir.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_http_failure_returns_empty(self, fast_idx: EdgarDailyIndex):
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("timeout")
        with patch.object(fast_idx._session, "get", side_effect=Exception("timeout")):
            rows = fast_idx.list_filings("2026-01-14")
        assert rows == []

    def test_form_type_filter(self, fast_idx: EdgarDailyIndex):
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            rows = fast_idx.list_filings("2026-01-15", form_types=["4"])
        assert all(r["form"] == "4" for r in rows)
        assert len(rows) == 2  # AAPL + MSFT have form 4

    def test_all_form_types_when_no_filter(self, fast_idx: EdgarDailyIndex):
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            rows = fast_idx.list_filings("2026-01-15", form_types=None)
        assert len(rows) == 3

    def test_accepts_date_object(self, fast_idx: EdgarDailyIndex):
        d = date(2026, 1, 15)
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            rows = fast_idx.list_filings(d)
        assert len(rows) == 3


# ── list_filings_range ────────────────────────────────────────────────────────

class TestListFilingsRange:
    def test_range_aggregates_multiple_days(self, fast_idx: EdgarDailyIndex):
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            # 2026-01-12 Mon, 2026-01-13 Tue, 2026-01-14 Wed → 3 weekdays
            rows = fast_idx.list_filings_range("2026-01-12", "2026-01-14")
        # 3 rows × 3 weekdays (weekends skipped)
        assert len(rows) == 9

    def test_range_skips_weekends(self, fast_idx: EdgarDailyIndex):
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            # Full week Mon-Sun (5 weekdays)
            rows = fast_idx.list_filings_range("2026-01-12", "2026-01-18")
        assert len(rows) == 15  # 5 weekdays × 3 rows each


# ── TTL expiry ────────────────────────────────────────────────────────────────

class TestCacheTtl:
    def test_expired_cache_triggers_refetch(self, fast_idx: EdgarDailyIndex, monkeypatch):
        """After TTL expires, a fresh HTTP call must be made."""
        with patch.object(fast_idx._session, "get", return_value=_make_resp(_SAMPLE_IDX)):
            fast_idx.list_filings("2026-01-15")

        # Expire the cache by backdating _ts
        cache_file = fast_idx._cache_root / "2026-01-15.json"
        data = json.loads(cache_file.read_text())
        data["_ts"] = time.time() - (25 * 3600)  # 25 h ago > 24 h TTL
        cache_file.write_text(json.dumps(data))

        mock_get = MagicMock(return_value=_make_resp(_SAMPLE_IDX))
        with patch.object(fast_idx._session, "get", mock_get):
            fast_idx.list_filings("2026-01-15")

        mock_get.assert_called_once()
