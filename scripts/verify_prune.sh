#!/usr/bin/env bash
# scripts/verify_prune.sh
# Verify that the pruned folders are absent from the live tree and that
# the full test suite still passes after the archive commits.
#
# Usage:
#   bash scripts/verify_prune.sh
#
# Exit code 0 = all checks pass; non-zero = at least one check failed.

set -euo pipefail

ARCHIVE_BASE="archive/prune-chore-prune-unused-folders-20260509"
PRUNED=(frontend infra intelligence log_manager)
PASS=0
FAIL=0

check() {
    local label="$1"; local ok="$2"
    if [ "$ok" -eq 0 ]; then
        echo "  PASS  $label"
        PASS=$(( PASS + 1 ))
    else
        echo "  FAIL  $label"
        FAIL=$(( FAIL + 1 ))
    fi
}

echo "=== verify_prune.sh ==="
echo ""

echo "1. Pruned folders absent from project root:"
for dir in "${PRUNED[@]}"; do
    if [ -d "$dir" ]; then
        check "$dir/ NOT present at root" 1
    else
        check "$dir/ absent from root" 0
    fi
done

echo ""
echo "2. Archive copies present:"
for dir in "${PRUNED[@]}"; do
    if [ -d "${ARCHIVE_BASE}/${dir}" ]; then
        check "${ARCHIVE_BASE}/${dir} exists" 0
    else
        check "${ARCHIVE_BASE}/${dir} missing" 1
    fi
done

echo ""
echo "3. Core package importable:"
python -c "import regime_trader" 2>/dev/null && check "regime_trader importable" 0 || check "regime_trader importable" 1

echo ""
echo "4. pytest (unit tests, excluding slow network tests):"
if python -m pytest tests/ -q \
    --ignore=tests/test_discovery_scanner.py \
    --tb=short 2>&1 | tail -3; then
    check "pytest passed" 0
else
    check "pytest failed" 1
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
