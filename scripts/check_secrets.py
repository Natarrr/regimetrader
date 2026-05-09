"""scripts/check_secrets.py
CI-friendly secret presence checker.

Stiglitz (2001 Nobel) — information asymmetry: CI runners need to know
*whether* credentials are present without ever revealing their values.

Exit behaviour
--------------
- Protected run (main/master push, or REQUIRE_SECRETS=true):
    exit 1 when any REQUIRED key is missing.
- All other contexts (fork PR, feature branch, local dev):
    exit 0 even when keys are absent — CI continues in degraded mode.

A run is "protected" when ANY of these is true:
  1. REQUIRE_SECRETS=true  (explicit override)
  2. GITHUB_REF ends in /main or /master  (protected-branch push)
  3. GITHUB_ACTIONS is unset AND a local developer ran the script
     with --strict (not yet implemented — reserved for future gate)

Usage:
    python scripts/check_secrets.py
    REQUIRE_SECRETS=true python scripts/check_secrets.py   # hard gate

Exit codes:
    0  all required secrets present  OR  non-protected context
    1  required secrets missing AND protected context
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Sequence, Tuple


# ── Secret lists ───────────────────────────────────────────────────────────────

REQUIRED: Tuple[str, ...] = (
    "FMP_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
)

OPTIONAL: Tuple[str, ...] = (
    "POLYGON_API_KEY",
    "ANTHROPIC_API_KEY",
)


# ── Protection detection ───────────────────────────────────────────────────────

def _is_protected_run() -> bool:
    """Return True when missing secrets should cause a hard failure.

    Keynes (1936) — expectations matter: CI should only hard-fail when it
    is running in a context where secrets *should* be present.
    """
    # Explicit override takes priority
    if os.environ.get("REQUIRE_SECRETS", "").lower() in ("1", "true", "yes"):
        return True

    # Not in GitHub Actions at all (local dev) — soft by default
    if not os.environ.get("GITHUB_ACTIONS"):
        return False

    # PR events never have secrets for fork PRs; don't hard-fail either way
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        return False

    # Protected branches: main / master
    ref = os.environ.get("GITHUB_REF", "")
    if ref in ("refs/heads/main", "refs/heads/master"):
        return True

    return False


# ── Checker ────────────────────────────────────────────────────────────────────

def check_secrets(
    required: Sequence[str] = REQUIRED,
    optional: Sequence[str] = OPTIONAL,
) -> Tuple[Dict[str, bool], int]:
    """Check presence of each secret; return result map and exit code.

    Modigliani / Miller (1958/1985) — separation principle: the presence
    check is structurally independent of the values it guards.

    Args:
        required: Keys whose absence causes exit 1 on protected runs.
        optional: Keys reported but never enforced.

    Returns:
        (results_dict, exit_code) where results_dict maps key -> present(bool).
    """
    results: Dict[str, bool] = {}
    missing: list[str] = []

    col_width = max(len(k) for k in list(required) + list(optional)) + 4

    protected = _is_protected_run()

    print("-" * (col_width + 20))
    print("CI secret presence check")
    context = "PROTECTED" if protected else "degraded (non-protected)"
    print(f"Context: {context}")
    print("-" * (col_width + 20))

    for key in required:
        present = bool(os.environ.get(key, "").strip())
        results[key] = present
        status = "OK" if present else "MISSING"
        print(f"  {key:<{col_width}} present: {present!s:<5}  [REQUIRED]  {status}")
        if not present:
            missing.append(key)

    for key in optional:
        present = bool(os.environ.get(key, "").strip())
        results[key] = present
        print(f"  {key:<{col_width}} present: {present!s:<5}  [optional]")

    print("-" * (col_width + 20))

    if missing and protected:
        print(
            f"\nFAIL: {len(missing)} required secret(s) missing on protected run: "
            f"{', '.join(missing)}"
        )
        print(
            "  Set them in GitHub -> Settings -> Secrets -> Actions\n"
            "  or in your local .env file and re-source it before running CI.\n"
            "  See docs/CI_SECRETS.md for setup instructions."
        )
        return results, 1

    if missing:
        print(
            f"\nWARN: {len(missing)} secret(s) absent — "
            "running in degraded mode (non-protected context)."
        )
    else:
        print(f"\nOK: All {len(required)} required secrets present.")

    return results, 0


def main() -> int:
    _, code = check_secrets()
    return code


if __name__ == "__main__":
    sys.exit(main())
