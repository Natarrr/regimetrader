"""Read audit-report.json and apply structural keep/delete decisions.

Usage:
    python scripts/audit/apply_structural.py --dry-run   # preview
    python scripts/audit/apply_structural.py             # apply
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path


def main(dry_run: bool) -> None:
    root   = Path(".")
    report = json.loads((root / "docs/superpowers/specs/audit-report.json").read_text())
    tag    = "[DRY] " if dry_run else ""

    for d in report["structural_decisions"]:
        if d["winner"] == "both_orphaned":
            continue

        keep_path = d["left"]  if d["winner"] == "left"  else d["right"]
        lose_path = d["right"] if d["winner"] == "left"  else d["left"]
        keep_mod  = keep_path.replace("/", ".").replace("\\", ".")
        lose_mod  = lose_path.replace("/", ".").replace("\\", ".")

        if d["unique_in_loser"]:
            print(f"SKIP  {lose_path}: unique symbols {d['unique_in_loser'][:3]} — merge manually first")
            continue

        for py_file in root.rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            new_text = re.sub(rf"\bfrom\s+{re.escape(lose_mod)}\b",   f"from {keep_mod}",   text)
            new_text = re.sub(rf"\bimport\s+{re.escape(lose_mod)}\b", f"import {keep_mod}", new_text)
            if new_text != text:
                print(f"{tag}Rewrite imports  {py_file.relative_to(root)}")
                if not dry_run:
                    py_file.write_text(new_text, encoding="utf-8")

        lose_dir = root / lose_path
        if lose_dir.is_dir():
            print(f"{tag}Delete dir       {lose_path}/")
            if not dry_run:
                shutil.rmtree(lose_dir)
        elif lose_dir.with_suffix(".py").exists():
            print(f"{tag}Delete file      {lose_path}.py")
            if not dry_run:
                lose_dir.with_suffix(".py").unlink()


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
