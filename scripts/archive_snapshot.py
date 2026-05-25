"""scripts/archive_snapshot.py
Wrapper CLI for Fix #4 daily snapshot archiving.

Copies logs/intel_source_status.json to logs/historical/YYYY-MM-DD/
using the idempotent archive_current_run() from historical_loader.

Used for manual local runs. In CI, archive_current_run() is called
directly from run_pipeline.py (Fix #4 wire-up).

Usage:
    python scripts/archive_snapshot.py [--log-dir logs]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("archive_snapshot")


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive intel_source_status.json snapshot (Fix #4)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing intel_source_status.json (default: logs)")
    args = parser.parse_args()

    try:
        from regime_trader.research.historical_loader import archive_current_run
    except ImportError as exc:
        log.error("archive_current_run not importable — check regime_trader.research.historical_loader: %s", exc)
        return 1

    try:
        dest = archive_current_run(args.log_dir)
        if dest is None:
            log.warning("Nothing to archive — %s/intel_source_status.json not found", args.log_dir)
            return 0
        log.info("Snapshot archived: %s", dest)
        return 0
    except Exception as exc:
        log.error("Archive failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
