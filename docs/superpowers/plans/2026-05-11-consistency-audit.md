# Consistency Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a three-phase consistency audit that eliminates duplicate modules, removes orphaned/broken code, cleans dead docs, and enforces naming/config uniformity across the `regime_trader` project.

**Architecture:** Four discovery scripts (`audit_imports`, `audit_docs`, `audit_structure`, `run_audit`) emit a JSON report and decision log with zero file changes; two apply scripts (`apply_structural`, `apply_cleanup`) read that report and execute deletions + import rewrites in targeted commits; a naming checker produces a third uniformity commit.

**Tech Stack:** Python 3.11+ stdlib only (`ast`, `re`, `pathlib`, `json`, `shutil`); `pytest` for audit script tests.

---

## File Map

**Create:**
- `scripts/__init__.py` (empty — makes `scripts` importable in tests)
- `scripts/audit/__init__.py` (empty)
- `scripts/audit/audit_imports.py` — import graph, orphan finder, broken-import finder
- `scripts/audit/audit_docs.py` — dead doc finder, duplicate doc finder
- `scripts/audit/audit_structure.py` — module pair scorer (5 weighted criteria)
- `scripts/audit/run_audit.py` — CLI runner, writes report + decision log
- `scripts/audit/apply_structural.py` — reads report, rewrites imports, deletes loser dirs
- `scripts/audit/apply_cleanup.py` — reads report, deletes orphaned .py and dead .md files
- `scripts/audit/check_naming.py` — flags non-snake_case .py files and duplicate requirements
- `tests/audit/__init__.py` (empty)
- `tests/audit/test_audit_imports.py`
- `tests/audit/test_audit_docs.py`
- `tests/audit/test_audit_structure.py`

**Generated (gitignore these — they change every run):**
- `docs/superpowers/specs/audit-report.json`
- `docs/superpowers/specs/audit-report.md`
- `docs/superpowers/specs/2026-05-11-consistency-audit-decisions.md`

---

### Task 1: Import graph scanner

**Files:**
- Create: `scripts/__init__.py`, `scripts/audit/__init__.py`, `tests/audit/__init__.py`
- Create: `scripts/audit/audit_imports.py`
- Create: `tests/audit/test_audit_imports.py`

- [ ] **Step 1: Create package init files**

```bash
python -c "
from pathlib import Path
for p in ['scripts/__init__.py', 'scripts/audit/__init__.py', 'tests/audit/__init__.py']:
    Path(p).touch()
"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/audit/test_audit_imports.py`:

```python
from pathlib import Path
from scripts.audit.audit_imports import (
    build_import_graph, find_orphans, find_broken_imports, discover_local_packages,
)


def test_graph_resolves_package_import(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "mod.py").write_text("x = 1")
    (tmp_path / "main.py").write_text("from pkg import mod\n")
    graph = build_import_graph(tmp_path)
    assert any("pkg" in p for p in graph["main.py"])


def test_orphan_file_detected(tmp_path):
    (tmp_path / "used.py").write_text("x = 1")
    (tmp_path / "unused.py").write_text("y = 2")
    (tmp_path / "consumer.py").write_text("from used import x\n")
    graph = build_import_graph(tmp_path)
    orphans = find_orphans(graph, entry_points={"consumer.py"})
    assert "unused.py" in orphans
    assert "used.py" not in orphans


def test_entry_points_excluded_from_orphans(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "1_Home.py").write_text("x = 1")
    graph = build_import_graph(tmp_path)
    orphans = find_orphans(graph, entry_points={"pages/1_Home.py"})
    assert "pages/1_Home.py" not in orphans


def test_init_files_never_flagged_as_orphans(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    graph = build_import_graph(tmp_path)
    orphans = find_orphans(graph, entry_points=set())
    assert not any("__init__.py" in o for o in orphans)


def test_broken_import_detected(tmp_path):
    (tmp_path / "regime_trader").mkdir()
    (tmp_path / "regime_trader" / "__init__.py").write_text("")
    (tmp_path / "main.py").write_text("from regime_trader import nonexistent\n")
    broken = find_broken_imports(tmp_path, local_packages={"regime_trader"})
    assert "main.py" in broken


def test_stdlib_not_flagged_as_broken(tmp_path):
    (tmp_path / "main.py").write_text("import os\nimport sys\n")
    broken = find_broken_imports(tmp_path, local_packages=set())
    assert "main.py" not in broken


def test_discover_local_packages(tmp_path):
    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text("")
    pkgs = discover_local_packages(tmp_path)
    assert "mypkg" in pkgs
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/audit/test_audit_imports.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.audit.audit_imports'`

- [ ] **Step 4: Implement `scripts/audit/audit_imports.py`**

```python
"""Import graph builder, orphan finder, and broken-import scanner."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set

ENTRY_PATTERNS = [
    "pages/*.py",
    "scripts/*.py",
    "cloud/**/*.py",
    "tests/**/*.py",
    "backend/tests/**/*.py",
]


def collect_entry_points(root: Path) -> Set[str]:
    entries: Set[str] = set()
    for pat in ENTRY_PATTERNS:
        for p in root.glob(pat):
            entries.add(str(p.relative_to(root)))
    for p in root.rglob("conftest.py"):
        entries.add(str(p.relative_to(root)))
    return entries


def discover_local_packages(root: Path) -> Set[str]:
    return {
        p.parent.name
        for p in root.glob("*/__init__.py")
        if not p.parent.name.startswith(".")
    }


def _module_to_rel(module: str, root: Path) -> str | None:
    parts = module.split(".")
    pkg = root.joinpath(*parts, "__init__.py")
    if pkg.exists():
        return str(pkg.relative_to(root))
    mod = root.joinpath(*parts).with_suffix(".py")
    if mod.exists():
        return str(mod.relative_to(root))
    return None


def build_import_graph(root: Path) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for py_file in root.rglob("*.py"):
        rel = str(py_file.relative_to(root))
        graph[rel] = set()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = _module_to_rel(alias.name, root)
                    if resolved:
                        graph[rel].add(resolved)
            elif isinstance(node, ast.ImportFrom) and node.module:
                resolved = _module_to_rel(node.module, root)
                if resolved:
                    graph[rel].add(resolved)
    return graph


def find_orphans(graph: Dict[str, Set[str]], entry_points: Set[str]) -> Set[str]:
    imported: Set[str] = set()
    for deps in graph.values():
        imported.update(deps)
    return {
        f for f in graph
        if f not in imported
        and f not in entry_points
        and not f.endswith("__init__.py")
    }


def find_broken_imports(root: Path, local_packages: Set[str]) -> Dict[str, List[str]]:
    broken: Dict[str, List[str]] = {}
    for py_file in root.rglob("*.py"):
        rel = str(py_file.relative_to(root))
        bad: List[str] = []
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in local_packages and not _module_to_rel(alias.name, root):
                        bad.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in local_packages and not _module_to_rel(node.module, root):
                    bad.append(node.module)
        if bad:
            broken[rel] = bad
    return broken
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/audit/test_audit_imports.py -v
```

Expected: 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/audit/__init__.py scripts/audit/audit_imports.py tests/audit/__init__.py tests/audit/test_audit_imports.py
git commit -m "feat(audit): import graph scanner with orphan and broken-import detection"
```

---

### Task 2: Dead doc and duplicate doc scanner

**Files:**
- Create: `scripts/audit/audit_docs.py`
- Create: `tests/audit/test_audit_docs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audit/test_audit_docs.py`:

```python
from pathlib import Path
from scripts.audit.audit_docs import (
    collect_symbols, extract_doc_symbols, find_dead_docs, find_duplicate_docs, jaccard,
)


def test_collect_symbols_finds_class_and_function(tmp_path):
    (tmp_path / "mod.py").write_text("class Foo:\n    pass\ndef bar(): pass\n")
    syms = collect_symbols(tmp_path)
    assert "Foo" in syms
    assert "bar" in syms


def test_extract_symbols_from_backtick_identifiers():
    text = "Use `FmpClient` to fetch data. Call `get_profile()` for details."
    syms = extract_doc_symbols(text)
    assert "FmpClient" in syms
    assert "get_profile" in syms


def test_dead_doc_flagged_when_symbols_missing(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "old.md").write_text(
        "Use `OldClass` to call `deprecated_func` and `another_gone`."
    )
    dead = find_dead_docs(tmp_path, threshold=0.5)
    assert "docs/old.md" in dead


def test_live_doc_not_flagged(tmp_path):
    (tmp_path / "pkg.py").write_text("class RealClass:\n    pass\ndef real_func(): pass\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "live.md").write_text("Use `RealClass` and `real_func`.")
    dead = find_dead_docs(tmp_path, threshold=0.5)
    assert "docs/live.md" not in dead


def test_jaccard_identical():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_duplicate_docs_detected(tmp_path):
    (tmp_path / "docs").mkdir()
    content = "# Overview\n## Setup\n```python\nx = 1\n```\n"
    (tmp_path / "docs" / "a.md").write_text(content)
    (tmp_path / "docs" / "b.md").write_text(content)
    dups = find_duplicate_docs(tmp_path, threshold=0.7)
    pairs = {frozenset([a, b]) for a, b, _ in dups}
    assert frozenset(["docs/a.md", "docs/b.md"]) in pairs
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/audit/test_audit_docs.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.audit.audit_docs'`

- [ ] **Step 3: Implement `scripts/audit/audit_docs.py`**

```python
"""Dead doc finder and duplicate doc detector."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]+)`|`([A-Za-z_][A-Za-z0-9_]+)\(")


def collect_symbols(root: Path) -> Set[str]:
    symbols: Set[str] = set()
    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
    return symbols


def extract_doc_symbols(text: str) -> List[str]:
    return [
        (m.group(1) or m.group(2)).rstrip("(.")
        for m in _SYMBOL_RE.finditer(text)
    ]


def find_dead_docs(root: Path, threshold: float = 0.5) -> List[str]:
    symbols = collect_symbols(root)
    dead: List[str] = []
    for md_file in root.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        refs = extract_doc_symbols(text)
        if not refs:
            continue
        missing_frac = sum(1 for r in refs if r not in symbols) / len(refs)
        if missing_frac > threshold:
            dead.append(str(md_file.relative_to(root)))
    return dead


def _heading_set(text: str) -> Set[str]:
    return {
        line.lstrip("#").strip().lower()
        for line in text.splitlines()
        if line.startswith("#") and line.strip() != "#"
    }


def _code_fingerprints(text: str) -> Set[str]:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return {b.strip()[:120] for b in blocks if b.strip()}


def jaccard(a: Set, b: Set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def find_duplicate_docs(root: Path, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    profiles: List[Tuple[str, Set[str]]] = []
    for md_file in root.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        profile = _heading_set(text) | _code_fingerprints(text)
        profiles.append((str(md_file.relative_to(root)), profile))

    dups: List[Tuple[str, str, float]] = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            score = jaccard(profiles[i][1], profiles[j][1])
            if score > threshold:
                dups.append((profiles[i][0], profiles[j][0], score))
    return sorted(dups, key=lambda x: -x[2])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/audit/test_audit_docs.py -v
```

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/audit_docs.py tests/audit/test_audit_docs.py
git commit -m "feat(audit): dead-doc and duplicate-doc scanner"
```

---

### Task 3: Module pair auto-scorer

**Files:**
- Create: `scripts/audit/audit_structure.py`
- Create: `tests/audit/test_audit_structure.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/audit/test_audit_structure.py`:

```python
from pathlib import Path
from scripts.audit.audit_structure import score_pairs, _list_symbols


def _make_pkg(root: Path, name: str, functions=("foo",)) -> None:
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"def {fn}():\n    pass" for fn in functions)
    (pkg / "__init__.py").write_text(body)


def test_more_functions_wins(tmp_path):
    _make_pkg(tmp_path, "left",  functions=["a"])
    _make_pkg(tmp_path, "right", functions=["a", "b", "c"])
    decisions = score_pairs(tmp_path, graph={}, pairs=[("left", "right")])
    assert decisions[0].winner == "right"


def test_missing_left_package_picks_right(tmp_path):
    _make_pkg(tmp_path, "right", functions=["a", "b"])
    decisions = score_pairs(tmp_path, graph={}, pairs=[("missing", "right")])
    assert decisions[0].winner == "right"


def test_unique_symbols_in_loser_flagged(tmp_path):
    _make_pkg(tmp_path, "left",  functions=["shared", "unique_left"])
    _make_pkg(tmp_path, "right", functions=["shared", "r1", "r2", "r3"])
    decisions = score_pairs(tmp_path, graph={}, pairs=[("left", "right")])
    d = decisions[0]
    assert d.winner == "right"
    assert "unique_left" in d.unique_in_loser


def test_list_symbols_finds_functions(tmp_path):
    _make_pkg(tmp_path, "mypkg", functions=["alpha", "beta"])
    syms = _list_symbols(tmp_path, "mypkg")
    assert "alpha" in syms and "beta" in syms


def test_both_missing_skipped(tmp_path):
    decisions = score_pairs(tmp_path, graph={}, pairs=[("gone_a", "gone_b")])
    assert decisions == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/audit/test_audit_structure.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.audit.audit_structure'`

- [ ] **Step 3: Implement `scripts/audit/audit_structure.py`**

```python
"""Module pair scorer: 5-criteria auto-decision engine."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

MODULE_PAIRS: List[Tuple[str, str]] = [
    ("backend/market_intel",  "regime_trader/services"),
    ("regime",                "regime_trader"),
    ("utils",                 "regime_trader/utils"),
    ("utils",                 "backend/utils"),
    ("backend/quant_models",  "hmm_engine"),
    ("analysis",              "feature_engineering"),
    ("data",                  "backend/data"),
]

_WEIGHTS = {
    "inbound_imports": 0.30,
    "test_coverage":   0.25,
    "completeness":    0.25,
    "recency":         0.10,
    "claude_md":       0.10,
}

_LAUREATES = frozenset({
    "Nobel", "Markowitz", "Fama", "Stiglitz", "Minsky",
    "Sharpe", "Black", "Scholes", "Arrow", "Heckman",
})


@dataclass
class PairDecision:
    left: str
    right: str
    left_score: float
    right_score: float
    winner: str
    unique_in_loser: List[str] = field(default_factory=list)


def _py_files(root: Path, pkg: str) -> List[Path]:
    d = root / pkg
    return list(d.rglob("*.py")) if d.exists() else []


def _count_inbound(pkg: str, graph: Dict[str, Set[str]]) -> int:
    prefix = pkg.replace("\\", "/")
    count = 0
    for importer, deps in graph.items():
        if not importer.replace("\\", "/").startswith(prefix):
            for dep in deps:
                if dep.replace("\\", "/").startswith(prefix):
                    count += 1
                    break
    return count


def _count_tests(root: Path, pkg: str) -> int:
    pkg_module = pkg.replace("/", ".").replace("\\", ".")
    pkg_leaf = pkg.split("/")[-1]
    count = 0
    for test_file in [*root.glob("tests/**/*.py"), *root.glob("backend/tests/**/*.py")]:
        try:
            text = test_file.read_text(encoding="utf-8", errors="replace")
            if pkg_module in text or pkg_leaf in text:
                count += 1
        except OSError:
            pass
    return count


def _count_functions(root: Path, pkg: str) -> int:
    count = 0
    for py_file in _py_files(root, pkg):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        count += sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    return count


def _latest_mtime(root: Path, pkg: str) -> float:
    files = _py_files(root, pkg)
    return max((f.stat().st_mtime for f in files), default=0.0)


def _claude_md_fraction(root: Path, pkg: str) -> float:
    total = with_laureate = 0
    for py_file in _py_files(root, pkg):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total += 1
                doc = ast.get_docstring(node) or ""
                if any(lau in doc for lau in _LAUREATES):
                    with_laureate += 1
    return with_laureate / total if total > 0 else 0.0


def _raw_metrics(root: Path, pkg: str, graph: Dict, max_mtime: float) -> Dict[str, float]:
    return {
        "inbound_imports": float(_count_inbound(pkg, graph)),
        "test_coverage":   float(_count_tests(root, pkg)),
        "completeness":    float(_count_functions(root, pkg)),
        "recency":         _latest_mtime(root, pkg) / max_mtime if max_mtime > 0 else 0.0,
        "claude_md":       _claude_md_fraction(root, pkg),
    }


def _weighted_score(m_l: Dict, m_r: Dict) -> Tuple[float, float]:
    sl = sr = 0.0
    for key, w in _WEIGHTS.items():
        l, r = m_l[key], m_r[key]
        denom = max(l, r, 1e-9)
        sl += w * (l / denom)
        sr += w * (r / denom)
    return sl, sr


def _list_symbols(root: Path, pkg: str) -> Set[str]:
    syms: Set[str] = set()
    for py_file in _py_files(root, pkg):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                syms.add(n.name)
    return syms


def score_pairs(
    root: Path,
    graph: Dict[str, Set[str]],
    pairs: List[Tuple[str, str]] | None = None,
) -> List[PairDecision]:
    pairs = pairs or MODULE_PAIRS
    all_mtimes = [_latest_mtime(root, p) for pair in pairs for p in pair]
    max_mtime = max(all_mtimes, default=1.0)

    decisions: List[PairDecision] = []
    for left, right in pairs:
        l_exists = (root / left).exists()
        r_exists = (root / right).exists()

        if not l_exists and not r_exists:
            continue
        if not l_exists:
            decisions.append(PairDecision(left, right, 0.0, 1.0, "right"))
            continue
        if not r_exists:
            decisions.append(PairDecision(left, right, 1.0, 0.0, "left"))
            continue

        m_l = _raw_metrics(root, left,  graph, max_mtime)
        m_r = _raw_metrics(root, right, graph, max_mtime)
        sl, sr = _weighted_score(m_l, m_r)

        winner = "left" if sl >= sr else "right"
        loser  = right if winner == "left" else left
        keeper = left  if winner == "left" else right
        unique = sorted(_list_symbols(root, loser) - _list_symbols(root, keeper))

        decisions.append(PairDecision(left, right, round(sl, 3), round(sr, 3), winner, unique))

    return decisions
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/audit/test_audit_structure.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/audit_structure.py tests/audit/test_audit_structure.py
git commit -m "feat(audit): module pair auto-scorer (5 weighted criteria)"
```

---

### Task 4: Audit CLI runner

**Files:**
- Create: `scripts/audit/run_audit.py`

- [ ] **Step 1: Implement `scripts/audit/run_audit.py`**

```python
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
```

- [ ] **Step 2: Run all audit tests to confirm nothing broke**

```bash
python -m pytest tests/audit/ -v
```

Expected: 19 tests PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/audit/run_audit.py
git commit -m "feat(audit): CLI runner — phases 1a/1b/1c + phase 2, emits report + decisions"
```

---

### Task 5: Run Phase 1 + Phase 2 discovery

**Files:** None created — reads the live codebase, writes report files.

- [ ] **Step 1: Add generated report files to .gitignore**

Append to `.gitignore`:

```
# Generated audit reports
docs/superpowers/specs/audit-report.json
docs/superpowers/specs/audit-report.md
docs/superpowers/specs/*-consistency-audit-decisions.md
```

```bash
git add .gitignore
git commit -m "chore: gitignore generated audit report files"
```

- [ ] **Step 2: Execute the audit runner**

```bash
python scripts/audit/run_audit.py --root . --out docs/superpowers/specs
```

Expected output form (counts vary):
```
Phase 1a — import graph...
Phase 1b — broken imports...
Phase 1c — dead/duplicate docs...
Phase 2  — scoring module pairs...

Report → .../docs/superpowers/specs/audit-report.md
  Orphaned .py files : N
  Broken imports     : N
  Dead docs          : N
  Duplicate doc pairs: N
  Structural pairs   : N
```

- [ ] **Step 3: Review the human-readable report**

```bash
cat docs/superpowers/specs/audit-report.md
```

Check the **Orphaned Files** section for false positives. A file that uses dynamic imports (e.g. `importlib.import_module(...)`) will look orphaned even though it is used. Remove any such false positives from `audit-report.json` under the `"orphans"` key before proceeding.

- [ ] **Step 4: Review structural decisions**

```bash
cat docs/superpowers/specs/2026-05-11-consistency-audit-decisions.md
```

For any decision with **Merge first** symbols listed: open both the loser and keeper in the IDE and verify those symbols are truly absent from the keeper (not just renamed). If a symbol is present under a different name, remove it from `unique_in_loser` in `audit-report.json` — otherwise the apply script will skip that entire pair.

---

### Task 6: Apply structural cleanup (Commit 1)

**Files:**
- Create: `scripts/audit/apply_structural.py`

- [ ] **Step 1: Create `scripts/audit/apply_structural.py`**

```python
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
```

- [ ] **Step 2: Dry-run to preview**

```bash
python scripts/audit/apply_structural.py --dry-run
```

Verify every `[DRY] Delete dir` and `[DRY] Rewrite imports` line is intentional. Edit `audit-report.json` to adjust any incorrect decision before continuing.

- [ ] **Step 3: Apply changes**

```bash
python scripts/audit/apply_structural.py
```

- [ ] **Step 4: Run the test suite**

```bash
python -m pytest tests/ backend/tests/ -q --tb=short
```

Fix any residual import errors (dynamic imports, `__all__` re-exports) manually. Re-run until green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: structural cleanup — delete losing duplicate modules, rewrite imports"
```

---

### Task 7: Apply orphan + dead-doc cleanup (Commit 2)

**Files:**
- Create: `scripts/audit/apply_cleanup.py`

- [ ] **Step 1: Create `scripts/audit/apply_cleanup.py`**

```python
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
```

- [ ] **Step 2: Dry-run to preview**

```bash
python scripts/audit/apply_cleanup.py --dry-run
```

Remove any false positives from `audit-report.json` before applying.

- [ ] **Step 3: Apply changes**

```bash
python scripts/audit/apply_cleanup.py
```

- [ ] **Step 4: Run the test suite**

```bash
python -m pytest tests/ backend/tests/ -q --tb=short
```

Expected: same or better pass count as after Task 6.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove orphaned .py files and dead/duplicate docs"
```

---

### Task 8: Naming and config uniformity (Commit 3)

**Files:**
- Create: `scripts/audit/check_naming.py`

- [ ] **Step 1: Create `scripts/audit/check_naming.py`**

```python
"""Flag non-snake_case .py module files and overlapping requirements entries.

Usage: python scripts/audit/check_naming.py
"""
from __future__ import annotations

import re
from pathlib import Path

_SNAKE = re.compile(r"^[a-z][a-z0-9_]*\.py$")
_SKIP  = {".venv", ".git", "__pycache__", "node_modules", ".claude"}


def find_non_snake_case(root: Path) -> list[str]:
    bad: list[str] = []
    for py in root.rglob("*.py"):
        if any(part in _SKIP for part in py.parts):
            continue
        name = py.name
        if name in ("__init__.py", "conftest.py") or name.startswith("test_"):
            continue
        if not _SNAKE.match(name):
            bad.append(str(py.relative_to(root)))
    return sorted(bad)


def find_shared_requirements(root: Path) -> list[str]:
    def _parse(path: Path) -> set[str]:
        if not path.exists():
            return set()
        return {
            re.split(r"[=><!]", line)[0].strip().lower()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(("#", "-r", "-"))
        }

    prod = _parse(root / "requirements.txt")
    ci   = _parse(root / "requirements-ci.txt")
    return sorted(prod & ci)


def main() -> None:
    root = Path(".")
    bad_names  = find_non_snake_case(root)
    shared_req = find_shared_requirements(root)

    print(f"Non-snake_case .py files ({len(bad_names)}):")
    for f in bad_names:
        print(f"  {f}")

    print(f"\nDuplicate entries in requirements.txt + requirements-ci.txt ({len(shared_req)}):")
    for p in shared_req:
        print(f"  {p}")

    if not bad_names and not shared_req:
        print("All clean.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the naming checker**

```bash
python scripts/audit/check_naming.py
```

- [ ] **Step 3: Fix non-snake_case files**

For each file reported (example — actual names depend on output):

```bash
git mv scripts/SomeFile.py scripts/some_file.py
```

Then find and update any imports referencing the old name:

```bash
python -c "
import re, sys
from pathlib import Path
old, new = 'SomeFile', 'some_file'
for f in Path('.').rglob('*.py'):
    t = f.read_text(encoding='utf-8', errors='replace')
    t2 = re.sub(rf'\b{old}\b', new, t)
    if t2 != t:
        print(f); f.write_text(t2, encoding='utf-8')
"
```

- [ ] **Step 4: Fix duplicate requirements**

If `requirements-ci.txt` duplicates entries from `requirements.txt`, replace the duplicates with a reference line. Open `requirements-ci.txt` and add `-r requirements.txt` as the first line, then remove any packages already listed in `requirements.txt`.

Example `requirements-ci.txt` after fix:
```
-r requirements.txt
# CI-only extras
pytest
pytest-anyio
yamllint
```

- [ ] **Step 5: Run the test suite**

```bash
python -m pytest tests/ backend/tests/ -q --tb=short
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: enforce snake_case module names and deduplicate requirements"
```

---

### Task 9: Final validation

- [ ] **Step 1: Full test suite**

```bash
python -m pytest tests/ backend/tests/ -q --tb=short
```

Expected: all tests PASS, 0 errors, 0 collection errors.

- [ ] **Step 2: Smoke imports**

```bash
python -c "import regime_trader; import hmm_engine; import monitoring; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Re-run audit to confirm improvement**

```bash
python scripts/audit/run_audit.py --root . --out docs/superpowers/specs
```

Expected: `Orphaned .py files: 0`, `Broken imports: 0`. Dead/duplicate docs should be 0 or only reference the audit scripts themselves (which is acceptable).

- [ ] **Step 4: Commit audit tooling**

```bash
git add scripts/audit/ tests/audit/
git commit -m "chore: retain audit tooling for future consistency checks"
```
