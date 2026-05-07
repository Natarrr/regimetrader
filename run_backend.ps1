# run_backend.ps1
# Start the FastAPI backend from repo root.
# Usage: .\run_backend.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location "$ProjectRoot\backend"
try {
    uvicorn main:app --reload --port 8000
} finally {
    Pop-Location
}
