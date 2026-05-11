"""Dead doc finder and duplicate doc detector."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]+)`|`([A-Za-z_][A-Za-z0-9_]+)\(")

_SKIP_DIRS = frozenset({".venv", ".venv_old", "node_modules", "__pycache__", ".git", ".claude"})


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS or (part.startswith(".") and part != ".")
               for part in p.relative_to(root).parts):
            continue
        yield p


def _iter_md_files(root: Path):
    for p in root.rglob("*.md"):
        if any(part in _SKIP_DIRS or (part.startswith(".") and part != ".")
               for part in p.relative_to(root).parts):
            continue
        yield p


def collect_symbols(root: Path) -> Set[str]:
    """Collect all function/class names from .py files."""
    symbols: Set[str] = set()
    for py_file in _iter_py_files(root):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
    return symbols


def extract_doc_symbols(text: str) -> List[str]:
    """Extract backtick-wrapped identifiers from markdown text."""
    results = []
    for m in _SYMBOL_RE.finditer(text):
        raw = (m.group(1) or m.group(2)).rstrip("(.")
        # Take only the final component of dotted paths (e.g. "regime_trader.Foo" -> "Foo")
        name = raw.rsplit(".", 1)[-1]
        if name:
            results.append(name)
    return results


def find_dead_docs(root: Path, threshold: float = 0.5) -> List[str]:
    """Return .md files where >threshold fraction of referenced symbols don't exist."""
    symbols = collect_symbols(root)
    dead: List[str] = []
    for md_file in _iter_md_files(root):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        refs = extract_doc_symbols(text)
        if not refs:
            continue
        missing_frac = sum(1 for r in refs if r not in symbols) / len(refs)
        if missing_frac > threshold:
            dead.append(md_file.relative_to(root).as_posix())
    return dead


def _heading_set(text: str) -> Set[str]:
    result = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip()
        if heading:  # skip empty headings like "## "
            result.add(heading.lower())
    return result


def _code_fingerprints(text: str) -> Set[str]:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return {b.strip()[:120] for b in blocks if b.strip()}


def jaccard(a: Set, b: Set) -> float:
    """Jaccard similarity between two sets."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def find_duplicate_docs(root: Path, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Return pairs of .md files with Jaccard similarity > threshold."""
    profiles: List[Tuple[str, Set[str]]] = []
    for md_file in _iter_md_files(root):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        profile = _heading_set(text) | _code_fingerprints(text)
        profiles.append((md_file.relative_to(root).as_posix(), profile))

    dups: List[Tuple[str, str, float]] = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            if not profiles[i][1] and not profiles[j][1]:
                continue  # both empty — no meaningful content to compare
            score = jaccard(profiles[i][1], profiles[j][1])
            if score > threshold:
                dups.append((profiles[i][0], profiles[j][0], score))
    return sorted(dups, key=lambda x: -x[2])
