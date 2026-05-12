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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from regime_trader.utils.io import atomic_write_json

log = logging.getLogger("monitoring.metrics_exporter")


def export_metrics(
    log_dir: Path,
    *,
    duration_override_s: float | None = None,
) -> Dict[str, Any]:
    """Derive metrics dict + write metrics.json. Returns the written dict.

    duration_override_s wins over `_edgar_meta.run_duration_seconds` when set —
    use this when timing the run from a CI step that wraps run_pipeline.
    """
    src_path = log_dir / "intel_source_status.json"
    if not src_path.exists():
        raise FileNotFoundError(f"intel_source_status.json not found at {src_path}")

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
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except Exception as exc:
        log.exception("metrics export failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
