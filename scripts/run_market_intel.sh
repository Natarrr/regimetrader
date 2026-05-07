#!/usr/bin/env bash
# scripts/run_market_intel.sh — POSIX wrapper for scheduled market-intel runs.
#
# Usage (cron):
#   30 10 * * 1-5 /path/to/regime_trader/scripts/run_market_intel.sh top50.csv
#
# Lock file prevents overlapping runs (idempotence at scheduler level).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TICKERS_FILE="${1:-top50.csv}"
LOCK_FILE="/tmp/market_intel.lock"
LOG_FILE="${REPO_ROOT}/logs/market_intel_runner.log"

mkdir -p "${REPO_ROOT}/logs"

# Atomic lock acquisition
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "[$(date -u +%FT%TZ)] another run is in progress — skipping" >> "${LOG_FILE}"
    exit 0
fi

cd "${REPO_ROOT}"
echo "[$(date -u +%FT%TZ)] starting run with tickers=${TICKERS_FILE}" >> "${LOG_FILE}"

# Activate venv if present
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python -m backend.market_intel.run_pipeline \
    --tickers-file "${TICKERS_FILE}" \
    --limit-forms 5 \
    --max-workers 4 \
    >> "${LOG_FILE}" 2>&1

echo "[$(date -u +%FT%TZ)] run complete" >> "${LOG_FILE}"
