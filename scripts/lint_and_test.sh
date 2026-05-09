#!/usr/bin/env bash
# scripts/lint_and_test.sh
# Local pre-push gate. Runs:
#   1. import sanity probe
#   2. ruff (if installed)
#   3. mypy (if installed) — non-blocking
#   4. pytest with short tracebacks
#
# Usage:
#   bash scripts/lint_and_test.sh
#   bash scripts/lint_and_test.sh tests/test_regime_detector.py   # subset
#
# Exit codes:
#   0  everything passed
#   1  any blocking step failed

set -euo pipefail

cd "$(dirname "$0")/.."

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"

step() {
    printf "${GREEN}── %s ──${NC}\n" "$*"
}
warn() {
    printf "${YELLOW}! %s${NC}\n" "$*"
}
fail() {
    printf "${RED}✗ %s${NC}\n" "$*" >&2
    exit 1
}

# ── 1. Sanity imports ─────────────────────────────────────────────────────────
step "Sanity imports"
python scripts/check_imports.py || fail "sanity imports failed"

# ── 2. Ruff (optional) ────────────────────────────────────────────────────────
if command -v ruff >/dev/null 2>&1; then
    step "Ruff"
    ruff check regime_trader/ analysis/ regime/ tests/ scripts/ \
        --exit-non-zero-on-fix \
        || fail "ruff reported issues"
else
    warn "ruff not installed — skipping (pip install ruff)"
fi

# ── 3. mypy (non-blocking) ────────────────────────────────────────────────────
if command -v mypy >/dev/null 2>&1; then
    step "mypy (non-blocking)"
    mypy regime_trader/ analysis/ regime/ --ignore-missing-imports || warn "mypy found type issues (non-blocking)"
else
    warn "mypy not installed — skipping (pip install mypy)"
fi

# ── 4. Pytest ─────────────────────────────────────────────────────────────────
step "pytest"
TARGETS="${*:-tests/ backend/tests/}"
pytest $TARGETS -q --tb=short || fail "pytest failed"

step "All checks passed."
