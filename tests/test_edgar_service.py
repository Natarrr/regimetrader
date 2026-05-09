"""tests/test_edgar_service.py
Unit tests for regime_trader.services.edgar_service.

Leontief (1973 Nobel) — systematic, cached access to EDGAR is as important
as the filings themselves; a broken data pipeline produces silent failures.

Coverage:
  - _RateLimiter: interval enforcement
  - _parse_company_idx: fixed-width format parsing
  - _cache_read / _cache_write: TTL and round-trip
  - EdgarService.quarterly_index: cache hit, cache miss → HTTP, failure path
  - EdgarService.list_filings: cache hit, Atom XML parsing
  - EdgarService.fetch_filing: cache hit, live fetch, failure
  - _parse_atom_feed: valid XML, malformed XML
  - Rate limiting: total HTTP calls bounded under repeated requests
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from regime_trader.services.edgar_service import (
    EdgarService,
    FilingRef,
    IndexRow,
    _RateLimiter,
    _cache_read,
    _cache_write,
    _parse_company_idx,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all EDGAR cache operations to a temp directory."""
    import regime_trader.services.edgar_service as mod
    monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_path / "edgar")
    return tmp_path / "edgar"


@pytest.fixture()
def svc(tmp_cache: Path) -> EdgarService:
    """EdgarService with very high rate (no delay) for unit tests."""
    return EdgarService(rate_per_sec=1000.0, cache_root=tmp_cache)


# ── Sample data ───────────────────────────────────────────────────────────────

def _make_idx_row(company: str, form: str, cik: str, date: str, filename: str) -> str:
    """Build one fixed-width company.idx row matching the parser's column offsets."""
    return f"{company:<62}{form:<12}{cik:<12}{date:<12}{filename}"


_SAMPLE_IDX = "\n".join([
    "Full-Index of EDGAR (company.idx)",
    "Description: Master Index to EDGAR full-text submissions",
    "Last-Modified: 2026-04-01 00:00:00",
    "---",
    "",
    "Company Name                                                  Form Type   CIK         Date Filed  Filename",
    "------------------------------------------------------------  ----------  ----------  ----------  --------",
    "",
    "",
    _make_idx_row("APPLE INC", "4", "0000320193", "2026-03-01",
                  "edgar/data/320193/0000320193-26-000010-index.htm"),
    _make_idx_row("MICROSOFT CORP", "10-K", "0000789019", "2026-02-15",
                  "edgar/data/789019/0000789019-26-000005-index.htm"),
    _make_idx_row("TINY STARTUP INC", "8-K", "0001234567", "2026-01-10",
                  "edgar/data/1234567/0001234567-26-000001-index.htm"),
    "",
])

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>EDGAR Full-Text Search</title>
  <entry>
    <title>Form 4 — APPLE INC — 2026-03-01</title>
    <updated>2026-03-01T00:00:00Z</updated>
    <id>https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0000320193&amp;type=4</id>
    <link href="https://www.sec.gov/Archives/edgar/data/320193/form4.htm"/>
  </entry>
  <entry>
    <title>Form 4 — APPLE INC — 2026-02-15</title>
    <updated>2026-02-15T00:00:00Z</updated>
    <id>https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0000320193&amp;type=4</id>
    <link href="https://www.sec.gov/Archives/edgar/data/320193/form4b.htm"/>
  </entry>
</feed>"""


# ── _RateLimiter ──────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_high_rate_does_not_block(self) -> None:
        rl = _RateLimiter(rate_per_sec=1000.0)
        t0 = time.monotonic()
        for _ in range(5):
            rl.acquire()
        assert time.monotonic() - t0 < 0.5

    def test_interval_is_inverse_of_rate(self) -> None:
        rl = _RateLimiter(rate_per_sec=0.5)
        assert rl._interval == pytest.approx(2.0, rel=0.01)

    def test_zero_rate_clamped(self) -> None:
        """Rate of 0 should not divide by zero."""
        rl = _RateLimiter(rate_per_sec=0.0)
        assert rl._interval >= 0.0


# ── _parse_company_idx ────────────────────────────────────────────────────────

class TestParseCompanyIdx:
    def test_parses_known_entries(self) -> None:
        rows = _parse_company_idx(_SAMPLE_IDX)
        companies = [r["company_name"] for r in rows]
        assert any("APPLE" in c for c in companies)
        assert any("MICROSOFT" in c for c in companies)

    def test_form_type_extracted(self) -> None:
        rows = _parse_company_idx(_SAMPLE_IDX)
        apple = next((r for r in rows if "APPLE" in r["company_name"]), None)
        assert apple is not None
        assert apple["form_type"] == "4"

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_company_idx("") == []

    def test_short_lines_skipped(self) -> None:
        rows = _parse_company_idx("too short\nalso short\n")
        assert rows == []


# ── File cache ────────────────────────────────────────────────────────────────

class TestEdgarFileCache:
    def test_round_trip(self, tmp_cache: Path) -> None:
        _cache_write("index", "2026_Q1", "raw index content")
        assert _cache_read("index", "2026_Q1", ttl=3600) == "raw index content"

    def test_ttl_expiry(self, tmp_cache: Path) -> None:
        _cache_write("index", "OLD_Q4", "stale")
        p = tmp_cache / "index" / "OLD_Q4.txt"
        old_t = time.time() - 48 * 3600
        os.utime(p, (old_t, old_t))
        assert _cache_read("index", "OLD_Q4", ttl=24 * 3600) is None

    def test_missing_returns_none(self, tmp_cache: Path) -> None:
        assert _cache_read("index", "NOEXIST", ttl=3600) is None


# ── quarterly_index ───────────────────────────────────────────────────────────

class TestQuarterlyIndex:
    def test_cache_hit_skips_http(self, svc: EdgarService, tmp_cache: Path) -> None:
        _cache_write("index", "2026_Q1", _SAMPLE_IDX)
        with patch.object(svc, "_get_text") as mock_get:
            rows = svc.quarterly_index(2026, 1)
        mock_get.assert_not_called()
        assert len(rows) > 0

    def test_cache_miss_fetches_and_caches(
        self, svc: EdgarService, tmp_cache: Path
    ) -> None:
        with patch.object(svc, "_get_text", return_value=_SAMPLE_IDX):
            rows = svc.quarterly_index(2026, 2)
        assert len(rows) >= 2
        # Verify it was cached
        cached = _cache_read("index", "2026_Q2", ttl=86400)
        assert cached == _SAMPLE_IDX

    def test_http_failure_returns_empty(self, svc: EdgarService) -> None:
        with patch.object(svc, "_get_text", return_value=None):
            rows = svc.quarterly_index(2025, 4)
        assert rows == []

    def test_rate_limiting_bounds_calls(self, svc: EdgarService, tmp_cache: Path) -> None:
        """10 calls for the same quarter should hit the cache after the first."""
        call_count = 0

        def _counted_get(url: str, **kw) -> str:
            nonlocal call_count
            call_count += 1
            return _SAMPLE_IDX

        with patch.object(svc, "_get_text", side_effect=_counted_get):
            for _ in range(10):
                svc.quarterly_index(2026, 3)

        assert call_count == 1, (
            f"_get_text called {call_count} times; caching should limit it to 1"
        )


# ── list_filings ──────────────────────────────────────────────────────────────

class TestListFilings:
    def test_cache_hit_skips_http(self, svc: EdgarService, tmp_cache: Path) -> None:
        import json
        payload = [{"cik": "320193", "company_name": "APPLE INC",
                    "form_type": "4", "date_filed": "2026-03-01",
                    "filename": "form4.htm", "url": "https://sec.gov/form4.htm"}]
        _cache_write("filings", "0000320193_4_40", json.dumps(payload))
        with patch.object(svc, "_get_text") as mock_get:
            result = svc.list_filings("0000320193", form_type="4")
        mock_get.assert_not_called()
        assert len(result) == 1
        assert result[0]["cik"] == "320193"

    def test_atom_feed_parsed(self, svc: EdgarService) -> None:
        with patch.object(svc, "_get_text", return_value=_SAMPLE_ATOM):
            result = svc.list_filings("320193", form_type="4")
        assert len(result) == 2
        assert all(r["form_type"] == "4" for r in result)
        assert "form4.htm" in result[0]["filename"]

    def test_http_failure_returns_empty(self, svc: EdgarService) -> None:
        with patch.object(svc, "_get_text", return_value=None):
            assert svc.list_filings("0000000000") == []


# ── fetch_filing ──────────────────────────────────────────────────────────────

class TestFetchFiling:
    def test_cache_hit(self, svc: EdgarService, tmp_cache: Path) -> None:
        url = "https://www.sec.gov/Archives/edgar/data/320193/form4.htm"
        _cache_write("docs", url, "<html>filing content</html>")
        with patch.object(svc, "_get_text") as mock_get:
            result = svc.fetch_filing(url)
        mock_get.assert_not_called()
        assert result == "<html>filing content</html>"

    def test_cache_miss_fetches_and_caches(self, svc: EdgarService) -> None:
        url = "https://www.sec.gov/Archives/edgar/data/320193/form4b.htm"
        with patch.object(svc, "_get_text", return_value="<html>new</html>"):
            result = svc.fetch_filing(url)
        assert result == "<html>new</html>"

    def test_http_failure_returns_none(self, svc: EdgarService) -> None:
        with patch.object(svc, "_get_text", return_value=None):
            assert svc.fetch_filing("https://www.sec.gov/missing.htm") is None


# ── _parse_atom_feed ──────────────────────────────────────────────────────────

class TestParseAtomFeed:
    def test_valid_feed(self) -> None:
        results = EdgarService._parse_atom_feed(_SAMPLE_ATOM, "4")
        assert len(results) == 2
        assert results[0]["url"].endswith("form4.htm")

    def test_malformed_xml_returns_empty(self) -> None:
        assert EdgarService._parse_atom_feed("not xml at all", "4") == []

    def test_empty_feed_returns_empty(self) -> None:
        xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        assert EdgarService._parse_atom_feed(xml, "10-K") == []
