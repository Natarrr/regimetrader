"""tests/test_fmp_bulk_prefetch.py
Unit tests for the FMP bulk response parser.

FMP bulk routes serve text/csv as of 2026-06-09 (confirmed live for both
endpoints). Earlier snapshots were NDJSON. A format the parser does
not recognize must never silently yield 0 records again — that masked the
entire bulk pipeline being dead.
"""
from __future__ import annotations

from src.ingestion.fmp_bulk_prefetch import (
    _coerce_csv_value,
    _parse_response,
    build_ticker_index_with_ambiguous,
    is_endpoint_usable,
    read_prefetch_status,
)


class TestParseResponseCSV:
    # Verbatim shape of the live upgrades-downgrades-consensus-bulk route.
    CSV = (
        '"symbol","strongBuy","buy","hold","sell","strongSell","consensus"\n'
        '"000550.SZ",1,14,4,0,0,"Buy"\n'
        '"AAPL",10,20,5,1,0,"Buy"\n'
    )

    def test_csv_parsed_to_records(self):
        records = _parse_response("upgrades-downgrades-consensus-bulk", self.CSV)
        assert len(records) == 2
        assert records[1]["symbol"] == "AAPL"

    def test_csv_numeric_cells_coerced(self):
        records = _parse_response("upgrades-downgrades-consensus-bulk", self.CSV)
        rec = records[1]
        assert rec["strongBuy"] == 10 and isinstance(rec["strongBuy"], int)
        assert rec["consensus"] == "Buy"

    def test_csv_ratios_floats_coerced(self):
        csv_text = (
            '"symbol","grossProfitMarginTTM","debtToEquityRatioTTM"\n'
            '"MSFT",0.6852,0.31\n'
        )
        records = _parse_response("ratios-ttm-bulk", csv_text)
        assert records[0]["grossProfitMarginTTM"] == 0.6852
        assert isinstance(records[0]["debtToEquityRatioTTM"], float)

    def test_csv_empty_cell_becomes_none(self):
        csv_text = '"symbol","currentRatioTTM"\n"TSLA",\n'
        records = _parse_response("ratios-ttm-bulk", csv_text)
        assert records[0]["currentRatioTTM"] is None

    def test_rows_without_symbol_dropped(self):
        csv_text = '"symbol","buy"\n"",3\n"NVDA",7\n'
        records = _parse_response("upgrades-downgrades-consensus-bulk", csv_text)
        assert len(records) == 1
        assert records[0]["symbol"] == "NVDA"


class TestParseResponseLegacyFormats:
    def test_ndjson_still_parsed(self):
        text = '{"symbol": "AAPL", "consensus": "Buy"}\n{"symbol": "MSFT", "consensus": "Hold"}\n'
        records = _parse_response("upgrades-downgrades-consensus-bulk", text)
        assert [r["symbol"] for r in records] == ["AAPL", "MSFT"]

    def test_json_array_still_parsed(self):
        text = '[{"symbol": "AAPL"}, {"symbol": "MSFT"}]'
        records = _parse_response("upgrades-downgrades-consensus-bulk", text)
        assert len(records) == 2

    def test_empty_text_returns_empty(self):
        assert _parse_response("ratios-ttm-bulk", "") == []

    def test_garbage_returns_empty_with_warning(self, caplog):
        with caplog.at_level("WARNING"):
            records = _parse_response("ratios-ttm-bulk", "<html>502 Bad Gateway</html>")
        assert records == []
        assert any("0 records" in r.message for r in caplog.records)


class TestCoerceCsvValue:
    def test_int(self):
        assert _coerce_csv_value("14") == 14

    def test_float(self):
        assert _coerce_csv_value("0.6852") == 0.6852

    def test_empty_is_none(self):
        assert _coerce_csv_value("") is None
        assert _coerce_csv_value(None) is None

    def test_string_passthrough(self):
        assert _coerce_csv_value("Strong Buy") == "Strong Buy"


class TestEndToEndIndexFromCSVSnapshot:
    def test_csv_snapshot_builds_nonempty_index(self, tmp_path):
        """prefetch-format cache written from a CSV response → usable index."""
        import json
        from datetime import datetime, timezone

        records = _parse_response(
            "upgrades-downgrades-consensus-bulk", TestParseResponseCSV.CSV
        )
        payload = {
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "_ttl_hours": 7,
            "_record_count": len(records),
            "data": records,
        }
        (tmp_path / "upgrades-downgrades-consensus-bulk.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        idx, ambiguous = build_ticker_index_with_ambiguous(
            tmp_path, "upgrades-downgrades-consensus-bulk"
        )
        assert "AAPL" in idx
        assert idx["AAPL"]["consensus"] == "Buy"


class TestPiotroskiFromBulkSnapshot:
    """End-to-end: a ratios-ttm-bulk CSV snapshot feeds score_quality_piotroski."""

    RATIOS_CSV = (
        '"symbol","operatingProfitMarginTTM","debtToEquityRatioTTM",'
        '"currentRatioTTM","grossProfitMarginTTM","netProfitMarginTTM","assetTurnoverTTM"\n'
        '"NVDA",0.62,0.22,4.1,0.75,0.55,0.45\n'
    )

    def _write_snapshot(self, tmp_path):
        import json
        from datetime import datetime, timezone

        records = _parse_response("ratios-ttm-bulk", self.RATIOS_CSV)
        payload = {
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "_ttl_hours": 7,
            "_record_count": len(records),
            "data": records,
        }
        (tmp_path / "ratios-ttm-bulk.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_bulk_record_scores_full_piotroski(self, tmp_path):
        from src.scoring.momentum_signals import score_quality_piotroski

        self._write_snapshot(tmp_path)
        idx, _ = build_ticker_index_with_ambiguous(tmp_path, "ratios-ttm-bulk")
        assert "NVDA" in idx
        score, raw = score_quality_piotroski(idx["NVDA"])
        assert raw == 8
        assert score == 1.0

    def test_junk_record_yields_dead_signal_for_fallback(self):
        """A record with no recognizable ratio fields must return (0.0, 0) so
        run_pipeline falls back to per-ticker FMP instead of scoring garbage."""
        from src.scoring.momentum_signals import score_quality_piotroski

        score, raw = score_quality_piotroski({"symbol": "X", "unknownField": 1.0})
        assert (score, raw) == (0.0, 0)


class TestPrefetchStatusVisibility:
    """A fetch failure must NOT be reported as success. Serving a prior cache is
    allowed but must be marked "stale" (loud ERROR + status marker) so downstream
    freshness never silently treats old data as fresh."""

    def _boom(self, *a, **k):
        raise RuntimeError("simulated FMP outage")

    def test_fetch_failure_with_cache_is_stale_and_logged(self, tmp_path, monkeypatch, caplog):
        import json
        from datetime import datetime, timezone, timedelta
        from src.ingestion import fmp_bulk_prefetch as bp

        ep = "upgrades-downgrades-consensus-bulk"
        old = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
        (tmp_path / f"{ep}.json").write_text(
            json.dumps({"_cached_at": old, "_ttl_hours": 1,
                        "_record_count": 1, "data": [{"symbol": "AAPL"}]}),
            encoding="utf-8",
        )
        monkeypatch.setattr(bp, "_fetch_endpoint", self._boom)
        with caplog.at_level("ERROR"):
            results = bp.prefetch([ep], tmp_path, ttl_hours=0, api_key="x", rps=50)

        assert results[ep] == "stale"  # not True / not "fresh"
        assert any("STALE CACHE" in r.message for r in caplog.records)
        marker = json.loads((tmp_path / "bulk_prefetch_status.json").read_text(encoding="utf-8"))
        assert marker["endpoints"][ep]["status"] == "stale"
        assert marker["endpoints"][ep]["cached_at"] == old

    def test_fetch_failure_without_cache_is_failed(self, tmp_path, monkeypatch):
        import json
        from src.ingestion import fmp_bulk_prefetch as bp

        ep = "ratios-ttm-bulk"
        monkeypatch.setattr(bp, "_fetch_endpoint", self._boom)
        results = bp.prefetch([ep], tmp_path, ttl_hours=0, api_key="x", rps=50)

        assert results[ep] == "failed"
        marker = json.loads((tmp_path / "bulk_prefetch_status.json").read_text(encoding="utf-8"))
        assert marker["endpoints"][ep]["status"] == "failed"

    def test_within_ttl_cache_is_fresh(self, tmp_path):
        import json
        from datetime import datetime, timezone
        from src.ingestion import fmp_bulk_prefetch as bp

        ep = "upgrades-downgrades-consensus-bulk"
        (tmp_path / f"{ep}.json").write_text(
            json.dumps({"_cached_at": datetime.now(timezone.utc).isoformat(),
                        "_ttl_hours": 23, "_record_count": 1,
                        "data": [{"symbol": "AAPL"}]}),
            encoding="utf-8",
        )
        results = bp.prefetch([ep], tmp_path, ttl_hours=23, api_key="x", rps=50)
        assert results[ep] == "fresh"  # served from valid cache, no fetch attempted


class TestReadPrefetchStatus:
    """The consumer (run_pipeline) must be able to read the status marker so a
    stale/failed bulk snapshot is gated out instead of scored as fresh (F3)."""

    def _write_status(self, tmp_path, endpoints):
        import json
        (tmp_path / "bulk_prefetch_status.json").write_text(
            json.dumps({"_written_at": "2026-06-22T00:00:00+00:00",
                        "endpoints": endpoints}),
            encoding="utf-8",
        )

    def test_reads_endpoint_statuses(self, tmp_path):
        self._write_status(tmp_path, {
            "ratios-ttm-bulk": {"status": "stale", "cached_at": "x", "age_hours": 50},
            "upgrades-downgrades-consensus-bulk": {"status": "fresh", "cached_at": "y"},
        })
        status = read_prefetch_status(tmp_path)
        assert status["ratios-ttm-bulk"] == "stale"
        assert status["upgrades-downgrades-consensus-bulk"] == "fresh"

    def test_absent_file_returns_empty(self, tmp_path):
        assert read_prefetch_status(tmp_path) == {}

    def test_malformed_file_returns_empty(self, tmp_path):
        (tmp_path / "bulk_prefetch_status.json").write_text("<not json>", encoding="utf-8")
        assert read_prefetch_status(tmp_path) == {}


class TestIsEndpointUsable:
    """Gate predicate: stale/failed ⇒ skip bulk (per-ticker fallback engages).
    Unknown/missing status ⇒ usable, preserving prior behavior when no marker
    exists (e.g. an old cache dir written before status tracking)."""

    def test_fresh_is_usable(self):
        assert is_endpoint_usable({"ratios-ttm-bulk": "fresh"}, "ratios-ttm-bulk") is True

    def test_stale_is_not_usable(self):
        assert is_endpoint_usable({"ratios-ttm-bulk": "stale"}, "ratios-ttm-bulk") is False

    def test_failed_is_not_usable(self):
        assert is_endpoint_usable({"ratios-ttm-bulk": "failed"}, "ratios-ttm-bulk") is False

    def test_missing_status_is_usable(self):
        assert is_endpoint_usable({}, "ratios-ttm-bulk") is True
