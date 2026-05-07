"""backend/market_intel/import_to_sqlite.py — Load form4_summary.csv into SQLite.

Idempotent: filings keyed by (ticker, filing_accession, transaction_date,
reporting_person, transaction_code) so re-running ingestion never duplicates rows.

Usage:
    python -m backend.market_intel.import_to_sqlite \
        --csv logs/form4_summary.csv \
        --db  data/market_intel.db

Schema:
    insider_events(
        id INTEGER PRIMARY KEY,
        ticker TEXT, source TEXT,
        transaction_date TEXT, reporting_person TEXT, reporting_role TEXT,
        transaction_code TEXT, shares REAL, price REAL, value REAL,
        acquired_disposed TEXT, filing_accession TEXT,
        is_amendment INTEGER, imported_at TEXT,
        UNIQUE(ticker, filing_accession, transaction_date, reporting_person, transaction_code)
    )
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("market_intel.import_db")


_DDL = """
CREATE TABLE IF NOT EXISTS insider_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT NOT NULL,
    source            TEXT NOT NULL,
    transaction_date  TEXT,
    reporting_person  TEXT,
    reporting_role    TEXT,
    transaction_code  TEXT,
    shares            REAL,
    price             REAL,
    value             REAL,
    acquired_disposed TEXT,
    filing_accession  TEXT,
    is_amendment      INTEGER NOT NULL DEFAULT 0,
    imported_at       TEXT NOT NULL,
    UNIQUE(ticker, filing_accession, transaction_date, reporting_person, transaction_code)
);
CREATE INDEX IF NOT EXISTS idx_insider_ticker      ON insider_events(ticker);
CREATE INDEX IF NOT EXISTS idx_insider_date        ON insider_events(transaction_date);
CREATE INDEX IF NOT EXISTS idx_insider_role_code   ON insider_events(reporting_role, transaction_code);
"""


def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_bool_int(s: str | None) -> int:
    return 1 if str(s or "").strip().lower() in ("1", "true", "yes") else 0


def import_csv(csv_path: Path, db_path: Path) -> dict:
    """Load CSV rows into SQLite; returns counts {inserted, skipped, total}."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_DDL)
        conn.commit()

        inserted = skipped = total = 0
        ts = datetime.now(timezone.utc).isoformat()

        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total += 1
                try:
                    conn.execute(
                        """INSERT INTO insider_events (
                            ticker, source, transaction_date, reporting_person,
                            reporting_role, transaction_code, shares, price, value,
                            acquired_disposed, filing_accession, is_amendment, imported_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row.get("ticker"),
                            row.get("source"),
                            row.get("transaction_date") or None,
                            row.get("reporting_person") or None,
                            row.get("reporting_role") or None,
                            row.get("transaction_code") or None,
                            _to_float(row.get("shares")),
                            _to_float(row.get("price")),
                            _to_float(row.get("value")),
                            row.get("acquired_disposed") or None,
                            row.get("filing_accession") or None,
                            _to_bool_int(row.get("is_amendment")),
                            ts,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    skipped += 1   # duplicate row — idempotency guarantee
        conn.commit()
    finally:
        conn.close()
    return {"inserted": inserted, "skipped": skipped, "total": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import form4_summary.csv into SQLite")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to form4_summary.csv produced by run_pipeline.py")
    parser.add_argument("--db", type=Path, required=True,
                        help="Path to SQLite database file (created if missing)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.csv.exists():
        log.error("CSV not found: %s", args.csv)
        return 2

    counts = import_csv(args.csv, args.db)
    log.info("Done. inserted=%d  skipped(duplicate)=%d  total_rows=%d  db=%s",
             counts["inserted"], counts["skipped"], counts["total"], args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
