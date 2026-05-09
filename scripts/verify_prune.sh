#!/usr/bin/env bash
# scripts/verify_prune.sh
# Reproduce the static and dynamic checks used to validate the 2026-05-09 prune.
#
# Checks:
#   1. Archived folders absent from project root.
#   2. Archive copies present under ARCHIVE_BASE.
#   3. Static reference scan (ripgrep) in *.py for archived module names.
#   4. .github/workflows and .claude do not reference archived folders.
#   5. python scripts/check_imports.py passes.
#   6. pytest -q exits 0 (unit suite, excluding live-network tests).
#
# Output: human-readable log + JSON summary printed at end.
# Exit: 0 = all clear, 1 = at least one check failed.
#
# Usage:
#   bash scripts/verify_prune.sh
#   bash scripts/verify_prune.sh 2>&1 | tee prune_verify_$(date +%Y%m%d).log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE_BASE="archive/prune-chore-prune-unused-folders-20260509"
PRUNED=(frontend infra intelligence log_manager)
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u)"
PASS=0
FAIL=0
STATIC_REFS_FOUND=0
IMPORTS_OK="false"
TESTS_OK="false"

pass() { echo "  [PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

cd "$REPO_ROOT"

echo "================================================================"
echo " verify_prune.sh  —  regime_trader  —  $TIMESTAMP"
echo "================================================================"
echo ""

# ── 1. Archived folders absent from project root ──────────────────────────────
echo "1. Archived folders absent from project root:"
for dir in "${PRUNED[@]}"; do
    # A directory may still exist on Windows if node_modules is locked;
    # pass if it is absent OR contains only untracked/gitignored artifacts.
    tracked=$(git ls-files "$dir" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$tracked" -eq 0 ]; then
        pass "$dir/ has 0 tracked files at root"
    else
        fail "$dir/ still has $tracked tracked files at root"
    fi
done
echo ""

# ── 2. Archive copies present ─────────────────────────────────────────────────
echo "2. Archive copies present:"
for dir in "${PRUNED[@]}"; do
    if [ -d "${ARCHIVE_BASE}/${dir}" ]; then
        pass "${ARCHIVE_BASE}/${dir}/ exists"
    else
        fail "${ARCHIVE_BASE}/${dir}/ MISSING"
    fi
done
echo ""

# ── 3. Static reference scan ──────────────────────────────────────────────────
echo "3. Static reference scan (Python import statements in *.py):"
SCAN_PATTERNS=("from intelligence" "import intelligence" "from log_manager" "import log_manager")

RG_CMD="rg"
if ! command -v rg >/dev/null 2>&1; then
    RG_CMD="grep -r"
fi

for pat in "${SCAN_PATTERNS[@]}"; do
    if command -v rg >/dev/null 2>&1; then
        hits=$($RG_CMD --glob "*.py" --ignore-file .gitignore \
               -l "$pat" . \
               --ignore ".venv" --ignore "archive" 2>/dev/null || true)
    else
        hits=$($RG_CMD --include="*.py" "$pat" . \
               --exclude-dir=.venv --exclude-dir=archive -l 2>/dev/null || true)
    fi

    if [ -n "$hits" ]; then
        STATIC_REFS_FOUND=$((STATIC_REFS_FOUND + 1))
        fail "'$pat' still referenced outside archive:"
        echo "$hits" | sed 's/^/    /'
    else
        pass "No live Python imports of '$pat'"
    fi
done
echo ""

# ── 4. CI and .claude do not reference archived folders ───────────────────────
echo "4. CI workflows / .claude do not reference archived folders:"
CI_DIRS=(".github/workflows" ".claude")
for ci_dir in "${CI_DIRS[@]}"; do
    [ -d "$ci_dir" ] || continue
    for dir in "${PRUNED[@]}"; do
        if command -v rg >/dev/null 2>&1; then
            hits=$(rg -l "$dir" "$ci_dir" 2>/dev/null || true)
        else
            hits=$(grep -rl "$dir" "$ci_dir" 2>/dev/null || true)
        fi
        if [ -n "$hits" ]; then
            STATIC_REFS_FOUND=$((STATIC_REFS_FOUND + 1))
            fail "'$dir' referenced in $ci_dir: $hits"
        else
            pass "$ci_dir does not reference '$dir'"
        fi
    done
done
echo ""

# ── 5. check_imports.py ───────────────────────────────────────────────────────
echo "5. python scripts/check_imports.py:"
if python scripts/check_imports.py 2>&1; then
    pass "check_imports.py exited 0"
    IMPORTS_OK="true"
else
    fail "check_imports.py failed"
    IMPORTS_OK="false"
fi
echo ""

# ── 6. pytest ─────────────────────────────────────────────────────────────────
echo "6. pytest (unit suite — excluding live-network screener tests):"
if python -m pytest tests/ -q \
    --ignore=tests/test_discovery_scanner.py \
    --tb=short 2>&1 | tail -4; then
    pass "pytest exited 0"
    TESTS_OK="true"
else
    fail "pytest failed"
    TESTS_OK="false"
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "================================================================"
echo ""

cat <<JSON
{
  "archived_folders": ["frontend", "infra", "intelligence", "log_manager"],
  "archive_path": "${ARCHIVE_BASE}",
  "static_refs_found": ${STATIC_REFS_FOUND},
  "imports_ok": ${IMPORTS_OK},
  "tests_ok": ${TESTS_OK},
  "checks_passed": ${PASS},
  "checks_failed": ${FAIL},
  "timestamp": "${TIMESTAMP}"
}
JSON

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
