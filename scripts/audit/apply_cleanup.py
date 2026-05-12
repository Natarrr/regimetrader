"""Delete orphaned .py files and dead/duplicate .md files.

Usage:
    python scripts/audit/apply_cleanup.py --dry-run
    python scripts/audit/apply_cleanup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(dry_run: bool) -> None:
    root   = Path(".")
    report = json.loads((root / "docs/superpowers/specs/audit-report.json").read_text())
    tag    = "[DRY] " if dry_run else ""

    for rel in report["orphans"]:
        p = root / rel
        if p.exists():
            print(f"{tag}Delete orphan    {rel}")
            if not dry_run:
                p.unlink()

    for rel in report["dead_docs"]:
        p = root / rel
        if p.exists():
            print(f"{tag}Delete dead doc  {rel}")
            if not dry_run:
                p.unlink()

    for a, b, score in report["duplicate_docs"]:
        pa, pb = root / a, root / b
        if pa.exists() and pb.exists():
            delete = b if pa.stat().st_size >= pb.stat().st_size else a
            keep   = a if delete == b else b
            print(f"{tag}Delete dup ({score:.2f}) {delete}  [keeping {keep}]")
            if not dry_run:
                (root / delete).unlink()


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
