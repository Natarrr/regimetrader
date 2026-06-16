"""WS4 — check_metrics.check_fmp_telemetry advisory soft-gate logic.

These are warnings only: they must fire on genuine degradation and stay silent
on healthy runs, cold caches, and older artifacts without a telemetry block."""
from __future__ import annotations

from monitoring.check_metrics import check_fmp_telemetry


class TestErrorRateGate:
    def test_high_error_rate_warns(self):
        warns = check_fmp_telemetry({"calls_per_run": 1000, "error_rate": 0.10})
        assert any("error_rate" in w for w in warns)

    def test_low_error_rate_silent(self):
        warns = check_fmp_telemetry({"calls_per_run": 1000, "error_rate": 0.01})
        assert not any("error_rate" in w for w in warns)

    def test_no_calls_is_silent(self):
        # older artifact / on-demand run with no telemetry block
        assert check_fmp_telemetry({"calls_per_run": 0, "error_rate": 1.0}) == []
        assert check_fmp_telemetry({}) == []


class TestCacheHitRateGate:
    def test_low_hit_rate_with_enough_lookups_warns(self):
        warns = check_fmp_telemetry({
            "calls_per_run": 500, "error_rate": 0.0,
            "cache_hit_rate": 0.10, "cache_lookups": 400,
        })
        assert any("cache_hit_rate" in w for w in warns)

    def test_low_hit_rate_but_cold_cache_silent(self):
        # Few lookups (cold cache / first run) → no false positive.
        warns = check_fmp_telemetry({
            "calls_per_run": 30, "error_rate": 0.0,
            "cache_hit_rate": 0.0, "cache_lookups": 5,
        })
        assert not any("cache_hit_rate" in w for w in warns)

    def test_healthy_cache_silent(self):
        warns = check_fmp_telemetry({
            "calls_per_run": 500, "error_rate": 0.0,
            "cache_hit_rate": 0.85, "cache_lookups": 400,
        })
        assert warns == []

    def test_custom_thresholds_respected(self):
        metrics = {"calls_per_run": 100, "error_rate": 0.03,
                   "cache_hit_rate": 0.6, "cache_lookups": 100}
        # Tighten error threshold below the observed rate → now warns.
        warns = check_fmp_telemetry(metrics, max_error_rate=0.02)
        assert any("error_rate" in w for w in warns)
