#!/usr/bin/env bash
# run_backend.sh
# Start the FastAPI backend from repo root.
# Usage: ./run_backend.sh

set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/backend"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  "$ROOT_DIR/.venv/bin/python" -m uvicorn main:app --reload --port 8000
else
  uvicorn main:app --reload --port 8000
fi
