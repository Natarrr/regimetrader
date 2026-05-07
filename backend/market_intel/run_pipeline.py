"""backend/market_intel/run_pipeline.py — CLI entry for scheduled runs.

Usage:
    python -m backend.market_intel.run_pipeline --tickers-file top50.csv --limit-forms 5
    python -m backend.market_intel.run_pipeline --tickers AAPL MSFT NVDA --limit-forms 3

Exit codes:
    0 — success
    1 — partial failure (some tickers errored but pipeline completed)
    2 — fatal failure (no tickers processed)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from . import config
from .adapter import fetch_intel_universe, write_summary_files

log = logging.getLogger("market_intel.cli")


def _load_tickers_csv(path: Path) -> List[str]:
    """Read tickers from a one-column or first-column CSV. Skips header if present."""
    out: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if not row:
                continue
            cell = row[0].strip().upper()
            if i == 0 and cell.lower() in ("ticker", "symbol"):
                continue   # header
            if cell:
                out.append(cell)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDGAR-first market intel pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", nargs="+", help="Inline ticker list")
    src.add_argument("--tickers-file", type=Path, help="CSV file with tickers (one per row)")
    parser.add_argument("--limit-forms", type=int, default=config.DEFAULT_LIMIT_FORMS,
                        help="Per-form-type cap on filings fetched (default: %(default)s)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers (SEC rate limit still enforced)")
    parser.add_argument("--log-dir", type=Path, default=config.LOG_DIR,
                        help="Output directory for summary JSON/CSV")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true",
                           help="Debug-level logging (every HTTP call, parse outcome)")
    verbosity.add_argument("--quiet", "-q", action="store_true",
                           help="Warning-level logging only (production cron)")
    args = parser.parse_args(argv)

    if args.verbose:
        _level = logging.DEBUG
    elif args.quiet:
        _level = logging.WARNING
    else:
        _level = logging.INFO
    logging.basicConfig(level=_level, format="%(asctime)s %(levelname)s %(message)s")
    # Ensure package loggers respect verbosity at source (their own file handlers
    # default to INFO; child level filtering must match --quiet/--verbose).
    for _name in ("market_intel.edgar", "market_intel.fmp",
                  "market_intel.adapter", "market_intel.cli", "market_intel.import_db"):
        logging.getLogger(_name).setLevel(_level)

    if args.tickers_file:
        if not args.tickers_file.exists():
            log.error("tickers file not found: %s", args.tickers_file)
            return 2
        tickers = _load_tickers_csv(args.tickers_file)
    else:
        tickers = [str(t).upper() for t in args.tickers]

    if not tickers:
        log.error("no tickers to process")
        return 2

    log.info("Starting fetch_intel_universe — %d tickers, limit_forms=%d, workers=%d",
             len(tickers), args.limit_forms, args.max_workers)

    _started_at = datetime.now(timezone.utc)
    _t_start = _started_at.timestamp()

    def _progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            log.info("  progress: %d / %d", done, total)

    results = fetch_intel_universe(
        tickers,
        max_workers=args.max_workers,
        limit_forms=args.limit_forms,
        progress_cb=_progress,
    )

    paths = write_summary_files(results, args.log_dir)
    edgar_n = sum(1 for r in results if r["source"] == "EDGAR")
    fmp_n   = sum(1 for r in results if r["source"] == "FMP")
    none_n  = sum(1 for r in results if r["source"] == "NONE")
    err_n   = sum(1 for r in results if r.get("errors"))

    # Augment intel_source_status.json so the existing UI badge reader picks it up.
    # The reader expects {<pillar>: {"count": N, "present": M}}; we add EDGAR keys.
    src_status_path = args.log_dir / "intel_source_status.json"
    try:
        existing = (
            json.loads(src_status_path.read_text(encoding="utf-8"))
            if src_status_path.exists() else {}
        )
    except Exception:
        existing = {}
    existing["edgar_insider"]       = {"count": len(results), "present": edgar_n}
    existing["fmp_insider_fallback"] = {"count": len(results), "present": fmp_n}
    _completed_at = datetime.now(timezone.utc)

    try:
        from .edgar_ingest import cb_state
        _cb = cb_state()
        _cb_open = _cb.get("state") == "open"
    except Exception:
        _cb = {}
        _cb_open = False

    existing["_edgar_meta"] = {
        "started_at":             _started_at.isoformat(),
        "last_run":               _completed_at.isoformat(),
        "run_duration_seconds":   round(_completed_at.timestamp() - _t_start, 2),
        "ticker_count":           len(results),
        "edgar_count":            edgar_n,
        "fmp_count":              fmp_n,
        "none_count":             none_n,
        "error_count":            err_n,
        "edgar_cb_open":          _cb_open,
        "edgar_cb_state":         _cb,
    }
    try:
        from utils.atomic_write import atomic_write_json
        atomic_write_json(src_status_path, existing)
    except Exception as exc:
        log.warning("could not write intel_source_status.json: %s", exc)

    log.info("Done. EDGAR=%d  FMP=%d  NONE=%d  with_errors=%d", edgar_n, fmp_n, none_n, err_n)
    log.info("Wrote: %s", json.dumps({k: str(v) for k, v in paths.items()}))

    if edgar_n + fmp_n == 0:
        return 2
    if err_n > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
