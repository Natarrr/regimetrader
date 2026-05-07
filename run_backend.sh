#!/usr/bin/env bash
# run_backend.sh
# Start the FastAPI backend from repo root.
# Usage: ./run_backend.sh

set -e
cd "$(dirname "$0")/backend"
uvicorn main:app --reload --port 8000
