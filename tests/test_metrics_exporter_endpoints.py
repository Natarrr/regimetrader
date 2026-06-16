"""WS4 — metrics_exporter rolls FMP telemetry into additive metrics.json keys
without disturbing the six legacy canary keys."""
from __future__ import annotations

import json

import pytest

from monitoring.metrics_exporter import export_metrics

_LEGACY_KEYS = {
    "last_run", "run_duration_seconds", "ticker_count",
    "edgar_count", "fmp_count", "error_count",
}


def _write_status(log_dir, *, fmp_endpoints=None):
    status = {
        "_edgar_meta": {
            "last_run": "2026-06-16T00:00:00+00:00",
            "run_duration_seconds": 120.0,
            "ticker_count": 500,
            "edgar_count": 480,
            "fmp_count": 500,
            "error_count": 0,
        },
    }
    if fmp_endpoints is not None:
        status["_fmp_endpoints"] = fmp_endpoints
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "intel_source_status.json").write_text(json.dumps(status), encoding="utf-8")


class TestExporterRollup:
    def test_rollup_math(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_COST_PER_CALL_USD", raising=False)
        _write_status(tmp_path, fmp_endpoints={
            "totals": {"calls": 1000, "failures": 20,
                       "cache_hits": 700, "cache_misses": 300},
        })
        m = export_metrics(tmp_path)
        assert m["calls_per_run"] == 1000
        assert m["error_rate"] == pytest.approx(0.02)        # 20 / 1000
        assert m["cache_hit_rate"] == pytest.approx(0.70)    # 700 / 1000 lookups
        assert m["cache_lookups"] == 1000
        assert m["cost_estimate_per_run"] == 0.0             # flat plan, unset rate

    def test_legacy_keys_survive(self, tmp_path):
        _write_status(tmp_path, fmp_endpoints={
            "totals": {"calls": 10, "failures": 0, "cache_hits": 5, "cache_misses": 5},
        })
        m = export_metrics(tmp_path)
        assert _LEGACY_KEYS.issubset(m.keys())
        assert m["ticker_count"] == 500
        assert m["edgar_count"] == 480

    def test_missing_telemetry_block_is_zeroed(self, tmp_path):
        _write_status(tmp_path, fmp_endpoints=None)
        m = export_metrics(tmp_path)
        assert _LEGACY_KEYS.issubset(m.keys())
        assert m["calls_per_run"] == 0
        assert m["error_rate"] == 0.0
        assert m["cache_hit_rate"] == 0.0
        assert m["cost_estimate_per_run"] == 0.0

    def test_cost_estimate_honours_env_rate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FMP_COST_PER_CALL_USD", "0.001")
        _write_status(tmp_path, fmp_endpoints={
            "totals": {"calls": 1500, "failures": 0, "cache_hits": 0, "cache_misses": 0},
        })
        m = export_metrics(tmp_path)
        assert m["cost_estimate_per_run"] == pytest.approx(1.5)  # 1500 × 0.001
        assert m["cache_hit_rate"] == 0.0                        # no lookups

    def test_written_file_matches_return(self, tmp_path):
        _write_status(tmp_path, fmp_endpoints={
            "totals": {"calls": 4, "failures": 1, "cache_hits": 2, "cache_misses": 2},
        })
        m = export_metrics(tmp_path)
        on_disk = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert on_disk == m
        assert on_disk["error_rate"] == pytest.approx(0.25)
