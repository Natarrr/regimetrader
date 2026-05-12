"""Run all three audit phases and write the report + decision log."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.audit.audit_imports import (
    build_import_graph, collect_entry_points, discover_local_packages,
    find_broken_imports, find_orphans,
)
from scripts.audit.audit_docs import find_dead_docs, find_duplicate_docs
from scripts.audit.audit_structure import score_pairs, PairDecision


def main(root_str: str = ".", out_str: str = "docs/superpowers/specs") -> None:
    root = Path(root_str).resolve()
    out  = root / out_str
    out.mkdir(parents=True, exist_ok=True)

    print("Phase 1a — import graph...")
    graph   = build_import_graph(root)
    entries = collect_entry_points(root)
    orphans = find_orphans(graph, entries)

    print("Phase 1b — broken imports...")
    local  = discover_local_packages(root)
    broken = find_broken_imports(root, local)

    print("Phase 1c — dead/duplicate docs...")
    dead_docs = find_dead_docs(root)
    dup_docs  = find_duplicate_docs(root)

    print("Phase 2  — scoring module pairs...")
    decisions = score_pairs(root, graph)

    report = {
        "generated":            date.today().isoformat(),
        "orphans":              sorted(orphans),
        "broken_imports":       {k: v for k, v in sorted(broken.items())},
        "dead_docs":            sorted(dead_docs),
        "duplicate_docs":       [(a, b, round(s, 3)) for a, b, s in dup_docs],
        "structural_decisions": [_to_dict(d) for d in decisions],
    }

    (out / "audit-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "audit-report.md").write_text(_render_md(report), encoding="utf-8")
    today = date.today().isoformat()
    (out / f"{today}-consistency-audit-decisions.md").write_text(
        _render_decisions(decisions), encoding="utf-8"
    )

    print(f"\nReport → {out}/audit-report.md")
    print(f"  Orphaned .py files : {len(orphans)}")
    print(f"  Broken imports     : {len(broken)}")
    print(f"  Dead docs          : {len(dead_docs)}")
    print(f"  Duplicate doc pairs: {len(dup_docs)}")
    print(f"  Structural pairs   : {len(decisions)}")


def _to_dict(d: PairDecision) -> dict:
    return {
        "left": d.left, "right": d.right,
        "left_score": d.left_score, "right_score": d.right_score,
        "winner": d.winner, "unique_in_loser": d.unique_in_loser,
    }


def _render_md(r: dict) -> str:
    def section(title: str, items: list, fmt) -> list:
        lines = [f"\n## {title} ({len(items)})"]
        lines += [fmt(i) for i in items] if items else ["_none_"]
        return lines

    lines = [f"# Audit Report — {r['generated']}"]
    lines += section("Orphaned Files",       r["orphans"],       lambda f: f"- `{f}`")
    lines += section("Broken Imports",       list(r["broken_imports"].items()),
                     lambda kv: f"- `{kv[0]}`: {', '.join(f'`{b}`' for b in kv[1])}")
    lines += section("Dead Docs",            r["dead_docs"],     lambda f: f"- `{f}`")
    lines += section("Duplicate Doc Pairs",  r["duplicate_docs"],
                     lambda t: f"- `{t[0]}` ↔ `{t[1]}` (Jaccard={t[2]})")

    lines += ["\n## Structural Decisions",
              "| Left | Right | L | R | Winner | Unique in loser |",
              "|------|-------|---|---|--------|-----------------|"]
    for d in r["structural_decisions"]:
        u = ", ".join(d["unique_in_loser"][:4]) + ("…" if len(d["unique_in_loser"]) > 4 else "")
        lines.append(
            f"| `{d['left']}` | `{d['right']}` | {d['left_score']} "
            f"| {d['right_score']} | **{d['winner']}** | {u} |"
        )
    return "\n".join(lines) + "\n"


def _render_decisions(decisions: list) -> str:
    lines = ["# Structural Decisions\n"]
    for d in decisions:
        keep = d.left  if d.winner == "left"  else d.right
        lose = d.right if d.winner == "left"  else d.left
        lines += [
            f"## {d.left} vs {d.right}",
            f"- Keep  : `{keep}` (score {max(d.left_score, d.right_score):.3f})",
            f"- Delete: `{lose}` (score {min(d.left_score, d.right_score):.3f})",
        ]
        if d.unique_in_loser:
            lines.append(f"- **Merge first**: {', '.join(d.unique_in_loser)}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out",  default="docs/superpowers/specs")
    args = p.parse_args()
    main(args.root, args.out)
