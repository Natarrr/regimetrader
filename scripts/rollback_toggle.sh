#!/usr/bin/env bash
# scripts/rollback_toggle.sh — flip EDGAR_FIRST in .env (interactive confirmation).
#
# Usage:
#   bash scripts/rollback_toggle.sh           # interactive
#   bash scripts/rollback_toggle.sh --yes     # non-interactive (CI / scripted use)
#   bash scripts/rollback_toggle.sh --set off # force disable (rollback to FMP-only)
#   bash scripts/rollback_toggle.sh --set on  # force re-enable (EDGAR primary)

set -euo pipefail

# ── Locate repo root + .env ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found. Create it first:"
    echo "    cp .env.example .env"
    exit 1
fi

# ── Parse args ────────────────────────────────────────────────────────────────
AUTO_CONFIRM="no"
FORCE_VALUE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y)        AUTO_CONFIRM="yes"; shift ;;
        --set)           FORCE_VALUE="${2:-}"; shift 2 ;;
        --set=*)         FORCE_VALUE="${1#--set=}"; shift ;;
        -h|--help)
            grep -E "^# " "$0" | sed -E "s/^# ?//"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1 (use --help)"
            exit 2
            ;;
    esac
done

# ── Read current value (default true) ─────────────────────────────────────────
CURRENT_RAW="$(grep -E "^EDGAR_FIRST=" "${ENV_FILE}" 2>/dev/null | head -n1 | cut -d= -f2- || true)"
CURRENT_LOWER="$(echo "${CURRENT_RAW:-true}" | tr "[:upper:]" "[:lower:]" | tr -d "[:space:]")"

case "${CURRENT_LOWER}" in
    true|1|yes|y) CURRENT="true"  ;;
    *)            CURRENT="false" ;;
esac

# ── Determine new value ───────────────────────────────────────────────────────
if [[ -n "${FORCE_VALUE}" ]]; then
    case "$(echo "${FORCE_VALUE}" | tr "[:upper:]" "[:lower:]")" in
        on|true|1|yes|y)   NEW="true"  ;;
        off|false|0|no|n)  NEW="false" ;;
        *)
            echo "ERROR: --set must be 'on'/'true' or 'off'/'false', got: ${FORCE_VALUE}"
            exit 2
            ;;
    esac
else
    NEW="$([[ "${CURRENT}" == "true" ]] && echo "false" || echo "true")"
fi

# ── Confirm ───────────────────────────────────────────────────────────────────
echo "==> Current  EDGAR_FIRST = ${CURRENT}"
echo "==> Will set EDGAR_FIRST = ${NEW}"
if [[ "${NEW}" == "false" ]]; then
    echo "    (ROLLBACK — adapter will skip EDGAR and use FMP only)"
else
    echo "    (RE-ENABLE — adapter will try EDGAR first)"
fi

if [[ "${AUTO_CONFIRM}" != "yes" ]]; then
    read -rp "Proceed? [y/N] " ans
    if [[ ! "${ans}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ── Write .env (with timestamped backup) ──────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
cp "${ENV_FILE}" "${ENV_FILE}.bak.${TS}"

if grep -qE "^EDGAR_FIRST=" "${ENV_FILE}"; then
    # Portable in-place edit (works on both GNU and BSD sed)
    sed -E "s|^EDGAR_FIRST=.*|EDGAR_FIRST=${NEW}|" "${ENV_FILE}" > "${ENV_FILE}.tmp"
    mv "${ENV_FILE}.tmp" "${ENV_FILE}"
else
    printf "\nEDGAR_FIRST=%s\n" "${NEW}" >> "${ENV_FILE}"
fi

echo "==> Done. Backup: ${ENV_FILE}.bak.${TS}"
echo "==> Restart the pipeline / Streamlit app for the change to take effect."
