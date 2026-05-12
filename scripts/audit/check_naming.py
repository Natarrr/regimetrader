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
