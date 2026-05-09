"""scripts/check_imports.py
Sanity import probe — fails fast in CI before pytest spends 90 s collecting.

Run:
    python scripts/check_imports.py

Exit codes:
    0  all required modules import successfully
    1  one or more imports failed (printed to stderr)

Used by:
    .github/workflows/ci.yml  (sanity job, before pytest)
"""
from __future__ import annotations

import importlib
import sys
from typing import Iterable


# (module_name, package_name_in_pip)
REQUIRED: tuple[tuple[str, str], ...] = (
    ("pydantic",     "pydantic"),
    ("requests",     "requests"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("anthropic",    "anthropic"),
    ("hmmlearn",     "hmmlearn"),
    ("sklearn",      "scikit-learn"),
    ("scipy",        "scipy"),
    ("statsmodels",  "statsmodels"),
    ("arch",         "arch"),
    ("yfinance",     "yfinance"),
)


def check(modules: Iterable[tuple[str, str]] = REQUIRED) -> int:
    failures: list[str] = []
    for mod, pkg in modules:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  OK   {pkg:<14} {ver}")
        except ImportError as exc:
            failures.append(f"  FAIL {pkg:<14} {exc}")

    if failures:
        print("\nMissing or broken packages:", file=sys.stderr)
        for line in failures:
            print(line, file=sys.stderr)
        print(
            "\nFix: pip install -r requirements-ci.txt",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(REQUIRED)} sanity imports passed.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
