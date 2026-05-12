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
    prefix = pkg.replace("\\", "/").rstrip("/") + "/"
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
    max_mtime = max(all_mtimes) if any(t > 0 for t in all_mtimes) else 1.0

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
