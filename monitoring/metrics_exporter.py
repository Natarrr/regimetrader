"""monitoring/metrics_exporter.py — derive metrics.json from intel_source_status.json.

Reads `<log-dir>/intel_source_status.json` (written by run_pipeline.py with the
`_edgar_meta` block) and emits `<log-dir>/metrics.json` containing exactly the
six keys the canary check expects:

    {
        "last_run":             ISO8601,
        "run_duration_seconds": float,
        "ticker_count":         int,
        "edgar_count":          int,
        "fmp_count":            int,
        "error_count":          int
    }

Usage:
    python -m monitoring.metrics_exporter
    python -m monitoring.metrics_exporter --log-dir logs --duration-seconds 124
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.io import atomic_write_json

log = logging.getLogger("monitoring.metrics_exporter")

# Amortized per-call cost (USD). FMP Ultimate is a flat $139/mo plan, so the true
# marginal per-call cost is 0; set FMP_COST_PER_CALL_USD to a derived amortized
# rate (e.g. 139 / expected_monthly_calls) to surface cost_estimate_per_run.
_DEFAULT_COST_PER_CALL_USD = 0.0


def _fmp_endpoint_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Roll the ``_fmp_endpoints`` telemetry block into additive metrics keys.

    Returns calls_per_run / error_rate / cache_hit_rate / cost_estimate_per_run.
    An absent block yields all-zeros so older artifacts stay forward-compatible
    and the six legacy canary keys are never disturbed.
    """
    totals = ((raw.get("_fmp_endpoints") or {}).get("totals")) or {}
    calls    = int(totals.get("calls")        or 0)
    failures = int(totals.get("failures")     or 0)
    hits     = int(totals.get("cache_hits")   or 0)
    misses   = int(totals.get("cache_misses") or 0)
    try:
        per_call = float(os.getenv("FMP_COST_PER_CALL_USD", _DEFAULT_COST_PER_CALL_USD))
    except ValueError:
        per_call = _DEFAULT_COST_PER_CALL_USD
    cache_total = hits + misses
    return {
        "calls_per_run":         calls,
        "error_rate":            round(failures / calls, 6) if calls else 0.0,
        "cache_hit_rate":        round(hits / cache_total, 6) if cache_total else 0.0,
        "cache_lookups":         cache_total,   # guards the soft cache-rate gate
        "cost_estimate_per_run": round(calls * per_call, 6),
    }


def export_metrics(
    log_dir: Path,
    *,
    duration_override_s: float | None = None,
) -> Dict[str, Any]:
    """Derive metrics dict + write metrics.json. Returns the written dict.

    duration_override_s wins over `_edgar_meta.run_duration_seconds` when set —
    use this when timing the run from a CI step that wraps run_pipeline.

    When intel_source_status.json is absent (pipeline aborted before writing it),
    a tombstone metrics.json is written with pipeline_failed=True so downstream
    check_metrics can report the failure without this step itself erroring.
    """
    src_path = log_dir / "intel_source_status.json"
    if not src_path.exists():
        tombstone: Dict[str, Any] = {
            "last_run":             datetime.now(timezone.utc).isoformat(),
            "run_duration_seconds": round(float(duration_override_s or 0.0), 2),
            "ticker_count":         0,
            "edgar_count":          0,
            "fmp_count":            0,
            "error_count":          0,
            "pipeline_failed":      True,
        }
        out_path = log_dir / "metrics.json"
        log_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path, tombstone)
        log.warning("intel_source_status.json not found — wrote tombstone metrics.json")
        return tombstone

    raw = json.loads(src_path.read_text(encoding="utf-8"))
    meta = raw.get("_edgar_meta") or {}

    duration = duration_override_s
    if duration is None:
        duration = float(meta.get("run_duration_seconds") or 0.0)

    metrics: Dict[str, Any] = {
        "last_run":             meta.get("last_run") or datetime.now(timezone.utc).isoformat(),
        "run_duration_seconds": round(float(duration), 2),
        "ticker_count":         int(meta.get("ticker_count") or 0),
        "edgar_count":          int(meta.get("edgar_count")  or 0),
        "fmp_count":            int(meta.get("fmp_count")    or 0),
        "error_count":          int(meta.get("error_count")  or 0),
    }
    # WS4: additive FMP per-endpoint rollup — never disturbs the six legacy keys.
    metrics.update(_fmp_endpoint_metrics(raw))

    out_path = log_dir / "metrics.json"
    atomic_write_json(out_path, metrics)
    log.info("Wrote %s — %s", out_path, json.dumps(metrics))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export canary metrics from intel_source_status.json")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing intel_source_status.json (default: logs)")
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="Override run_duration_seconds (e.g. measured by the CI step)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        export_metrics(args.log_dir, duration_override_s=args.duration_seconds)
        return 0
    except Exception as exc:
        log.exception("metrics export failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
