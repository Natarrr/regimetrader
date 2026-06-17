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
from pathlib import Path
from typing import Iterable

# Script runs as `python scripts/check_imports.py`, so sys.path[0] is scripts/;
# project-module probes need the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# (module_name, package_name_in_pip)
REQUIRED: tuple[tuple[str, str], ...] = (
    ("pydantic",     "pydantic"),
    ("requests",     "requests"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("sklearn",      "scikit-learn"),
    ("scipy",        "scipy"),
    ("statsmodels",  "statsmodels"),
    ("yfinance",     "yfinance"),
)

# Project modules — fail fast if the src/ package layout breaks.
PROJECT: tuple[str, ...] = (
    "src.config.weights",
    "src.risk.regime",
    "src.risk.exit_rules",
    "src.scoring.normalize",
    "src.services.fmp_client",
    "src.utils.io",
    "src.fetchers.orchestrator",
    "src.monitoring.factor_orthogonality",
    "src.ingestion.run_pipeline",
    "src.delivery.send_discord",
    "src.delivery.cook_toplists",
    "backend.market_intel.generate_top_lists",
    "monitoring.metrics_exporter",
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

    for mod in PROJECT:
        try:
            importlib.import_module(mod)
            print(f"  OK   {mod}")
        except ImportError as exc:
            failures.append(f"  FAIL {mod:<40} {exc}")

    if failures:
        print("\nMissing or broken packages:", file=sys.stderr)
        for line in failures:
            print(line, file=sys.stderr)
        print(
            "\nFix: pip install -r requirements-ci.txt",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(REQUIRED) + len(PROJECT)} sanity imports passed.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
