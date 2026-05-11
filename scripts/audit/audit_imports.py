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
                if top in local_packages:
                    # For ImportFrom, check if what we're importing exists
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        import_path = f"{node.module}.{alias.name}"
                        if not _module_to_rel(import_path, root):
                            bad.append(import_path)
        if bad:
            broken[rel] = bad
    return broken
