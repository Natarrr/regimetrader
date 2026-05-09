"""scripts/check_secrets.py
CI secret presence checker — verifies required API keys are in the environment.

Stiglitz (2001 Nobel) — information asymmetry: the CI environment must have
complete credentials before running the full test suite.  This script is the
fast gate that prevents wasted runner time.

Run:
    python scripts/check_secrets.py

Exit codes:
    0  all required secrets are present
    1  one or more required secrets are missing

Output format:
    FMP_API_KEY            present: True
    ALPACA_API_KEY         present: False    ← missing → exit 1
    ...

Values are NEVER printed.  Only presence (True/False) is shown.

Used by:
    .github/workflows/ci.yml  (secrets-check step, before smoke tests)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Sequence, Tuple

# ── Required secrets ───────────────────────────────────────────────────────────

REQUIRED: Tuple[str, ...] = (
    "FMP_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
)

# Optional — checked and reported but do NOT cause a non-zero exit.
OPTIONAL: Tuple[str, ...] = (
    "POLYGON_API_KEY",
    "ANTHROPIC_API_KEY",
)


def check_secrets(
    required: Sequence[str] = REQUIRED,
    optional: Sequence[str] = OPTIONAL,
) -> Tuple[Dict[str, bool], int]:
    """Check presence of each secret; return result map and exit code.

    Args:
        required: Keys whose absence causes exit code 1.
        optional: Keys reported but not enforced.

    Returns:
        (results_dict, exit_code)  where results_dict maps key → present(bool).
    """
    results: Dict[str, bool] = {}
    missing: list[str] = []

    col_width = max(len(k) for k in list(required) + list(optional)) + 4

    print("-" * (col_width + 20))
    print("CI secret presence check")
    print("-" * (col_width + 20))

    for key in required:
        present = bool(os.environ.get(key, "").strip())
        results[key] = present
        tag = "REQUIRED"
        status = "OK" if present else "MISSING"
        print(f"  {key:<{col_width}} present: {present!s:<5}  [{tag}]  {status}")
        if not present:
            missing.append(key)

    for key in optional:
        present = bool(os.environ.get(key, "").strip())
        results[key] = present
        tag = "optional"
        print(f"  {key:<{col_width}} present: {present!s:<5}  [{tag}]")

    print("-" * (col_width + 20))

    if missing:
        print(f"\nFAIL: {len(missing)} required secret(s) missing: {', '.join(missing)}")
        print(
            "  Set them in GitHub -> Settings -> Secrets -> Actions\n"
            "  or in your local .env file and re-source it before running CI.\n"
            "  See docs/CI_SECRETS.md for setup instructions."
        )
        return results, 1

    print(f"\nOK: All {len(required)} required secrets present.")
    return results, 0


def main() -> int:
    _, code = check_secrets()
    return code


if __name__ == "__main__":
    sys.exit(main())
