"""tests/test_edgar_fetcher.py
Unit tests for regime_trader.services.edgar_fetcher.

Stiglitz (2001 Nobel) — XBRL-first parsing reduces information asymmetry.
All HTTP interactions are mocked; no live SEC traffic in CI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from regime_trader.services.edgar_fetcher import (
    EdgarFetcher,
    ParsedFiling,
    _extract_form4_rows,
    _extract_xbrl_facts,
    _extract_8k_items,
    _pick_best_doc,
    _cb_allows_calls,
    _cb_record_failure,
    _cb_record_success,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def fast_fetcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EdgarFetcher:
    import regime_trader.services.edgar_fetcher as mod
    monkeypatch.setattr(mod, "_CB_PATH", tmp_path / "cb.json")
    monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_path / "filings")
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "10")
    monkeypatch.setenv("EDGAR_CB_COOLDOWN_MIN", "1")
    return EdgarFetcher(rate_per_sec=1000.0, cache_root=tmp_path / "filings")


@pytest.fixture()
def reset_cb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import regime_trader.services.edgar_fetcher as mod
    monkeypatch.setattr(mod, "_CB_PATH", tmp_path / "cb_test.json")
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "3")
    monkeypatch.setenv("EDGAR_CB_COOLDOWN_MIN", "1")


# ── Sample XBRL / Form-4 / 8-K ────────────────────────────────────────────────

_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner><rptOwnerName>Tim Cook</rptOwnerName></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate>2026-01-10</transactionDate>
      <transactionCode>S</transactionCode>
      <transactionShares>50000</transactionShares>
      <transactionPricePerShare>175.50</transactionPricePerShare>
      <transactionAcquiredDisposedCode>D</transactionAcquiredDisposedCode>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_XBRL_XML = """<?xml version="1.0"?>
<xbrl xmlns:dei="http://xbrl.sec.gov/dei/2023"
      xmlns:us-gaap="http://fasb.org/us-gaap/2023">
  <dei:EntityCommonStockSharesOutstanding>15000000000</dei:EntityCommonStockSharesOutstanding>
  <dei:CurrentFiscalYearEndDate>--09-30</dei:CurrentFiscalYearEndDate>
  <us-gaap:Assets>335038000000</us-gaap:Assets>
</xbrl>"""

_8K_TEXT = """Item 1.01 Entry into a Material Definitive Agreement.
On January 10, 2026, Apple Inc. entered into ...
Item 8.01 Other Events.
The company announces ..."""

_INDEX_HTML_XBRL = """<!DOCTYPE html>
<html><body>
<a href="aapl-20260110_htm.xml">XBRL Instance</a>
<a href="aapl-20260110.htm">HTML Filing</a>
</body></html>"""

_INDEX_HTML_ONLY = """<!DOCTYPE html>
<html><body>
<a href="aapl-20260110.htm">HTML Filing</a>
</body></html>"""


def _mock_resp(text: str, status: int = 200) -> MagicMock:
    m = MagicMock()
    m.text = text
    m.status_code = status
    m.raise_for_status = MagicMock()
    return m


# ── Parser unit tests ─────────────────────────────────────────────────────────

class TestExtractForm4:
    def test_parses_transaction_row(self):
        rows = _extract_form4_rows(_FORM4_XML)
        assert len(rows) == 1
        r = rows[0]
        assert r["issuerTicker"] == "AAPL"
        assert r["reportingOwner"] == "Tim Cook"
        assert r["transactionCode"] == "S"
        assert r["shares"] == "50000"

    def test_malformed_xml_returns_empty(self):
        assert _extract_form4_rows("<<not xml>>") == []

    def test_empty_string_returns_empty(self):
        assert _extract_form4_rows("") == []


class TestExtractXbrlFacts:
    def test_extracts_facts(self):
        facts = _extract_xbrl_facts(_XBRL_XML)
        assert len(facts) > 0

    def test_strips_namespaces(self):
        facts = _extract_xbrl_facts(_XBRL_XML)
        # All keys should be plain tag names without braces
        for k in facts:
            assert "{" not in k

    def test_malformed_xml_returns_empty(self):
        assert _extract_xbrl_facts("<<bad>>") == {}


class TestExtract8kItems:
    def test_extracts_items(self):
        items = _extract_8k_items(_8K_TEXT)
        assert "1.01" in items
        assert "8.01" in items

    def test_empty_text_returns_empty(self):
        assert _extract_8k_items("") == []


class TestPickBestDoc:
    def test_prefers_xbrl_over_html(self):
        base = "https://www.sec.gov/Archives/edgar/data/320193/0001/"
        url, src = _pick_best_doc(_INDEX_HTML_XBRL, base)
        assert src == "xbrl"
        assert "htm.xml" in url.lower() or "xbrl" in url.lower()

    def test_falls_back_to_html(self):
        base = "https://www.sec.gov/Archives/edgar/data/320193/0001/"
        url, src = _pick_best_doc(_INDEX_HTML_ONLY, base)
        assert src == "html"
        assert url.endswith(".htm")


# ── EdgarFetcher integration (mocked HTTP) ────────────────────────────────────

class TestEdgarFetcherIntegration:
    def test_fetch_form4_returns_parsed_filing(self, fast_fetcher: EdgarFetcher):
        idx_resp  = _mock_resp(_INDEX_HTML_XBRL)
        doc_resp  = _mock_resp(_FORM4_XML)
        fast_fetcher._session.get = MagicMock(side_effect=[idx_resp, doc_resp])

        result = fast_fetcher.fetch_and_parse_filing(
            "https://www.sec.gov/Archives/edgar/data/320193/idx.htm",
            evidence_id="test_ev_001",
            form_type="4",
        )
        assert result["source"] in ("xbrl", "html", "text")
        assert "evidence_id" in result
        assert "fetched_at" in result

    def test_cache_hit_skips_http(self, fast_fetcher: EdgarFetcher):
        idx_resp = _mock_resp(_INDEX_HTML_XBRL)
        doc_resp = _mock_resp(_FORM4_XML)
        fast_fetcher._session.get = MagicMock(side_effect=[idx_resp, doc_resp])
        url = "https://www.sec.gov/Archives/edgar/data/320193/idx.htm"

        # First call
        fast_fetcher.fetch_and_parse_filing(url, evidence_id="ev_cache_test", form_type="4")

        # Second call — HTTP must NOT be called again
        mock2 = MagicMock()
        fast_fetcher._session.get = mock2
        fast_fetcher.fetch_and_parse_filing(url, evidence_id="ev_cache_test", form_type="4")
        mock2.assert_not_called()

    def test_cache_is_atomic(self, fast_fetcher: EdgarFetcher, tmp_path: Path):
        idx_resp = _mock_resp(_INDEX_HTML_ONLY)
        doc_resp = _mock_resp(_FORM4_XML)
        fast_fetcher._session.get = MagicMock(side_effect=[idx_resp, doc_resp])

        fast_fetcher.fetch_and_parse_filing(
            "https://www.sec.gov/Archives/edgar/data/320193/idx.htm",
            evidence_id="ev_atomic",
            form_type="4",
        )
        cache_dir = fast_fetcher._cache_root
        temps = [p for p in cache_dir.iterdir() if p.suffix == ".tmp"]
        assert temps == []

    def test_network_failure_returns_error_result(self, fast_fetcher: EdgarFetcher):
        fast_fetcher._session.get = MagicMock(side_effect=Exception("timeout"))
        result = fast_fetcher.fetch_and_parse_filing(
            "https://www.sec.gov/Archives/edgar/data/320193/idx.htm",
            evidence_id="ev_err",
        )
        assert result["source"] == "error"
        assert "error" in result["parsed"]

    def test_evidence_id_derived_from_url_if_empty(self, fast_fetcher: EdgarFetcher):
        idx_resp = _mock_resp(_INDEX_HTML_ONLY)
        doc_resp = _mock_resp(_FORM4_XML)
        fast_fetcher._session.get = MagicMock(side_effect=[idx_resp, doc_resp])
        result = fast_fetcher.fetch_and_parse_filing(
            "https://www.sec.gov/Archives/edgar/data/320193/somefile.htm"
        )
        assert len(result["evidence_id"]) == 16


# ── Circuit breaker ────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_cb_open_returns_error_result(
        self, fast_fetcher: EdgarFetcher, monkeypatch: pytest.MonkeyPatch
    ):
        import regime_trader.services.edgar_fetcher as mod
        monkeypatch.setattr(mod, "_cb_allows_calls", lambda: False)

        result = fast_fetcher.fetch_and_parse_filing(
            "https://www.sec.gov/Archives/edgar/data/320193/idx.htm",
            evidence_id="ev_cb",
        )
        assert result["source"] == "error"
        assert "circuit_breaker_open" in result["parsed"].get("error", "")

    def test_cb_records_failure_on_network_error(
        self, reset_cb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import regime_trader.services.edgar_fetcher as mod
        monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "10")
        fetcher = EdgarFetcher(
            rate_per_sec=1000.0,
            cache_root=tmp_path / "f_cb_err",
        )
        fetcher._session.get = MagicMock(side_effect=Exception("500"))

        fetcher.fetch_and_parse_filing(
            "https://www.sec.gov/Archives/edgar/data/320193/idx.htm",
            evidence_id="ev_fail_rec",
        )
        # Fail count should be >= 1
        state = mod._cb_state()
        assert state["fail_count"] >= 1

    def test_cb_success_resets_fail_count(
        self, reset_cb, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import regime_trader.services.edgar_fetcher as mod
        monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "10")
        _cb_record_failure()
        assert mod._cb_state()["fail_count"] == 1
        _cb_record_success()
        assert mod._cb_state()["fail_count"] == 0
