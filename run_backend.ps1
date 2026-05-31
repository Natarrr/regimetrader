# run_backend.ps1
# Start the FastAPI backend from repo root.
# Usage: .\run_backend.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: Python executable not found at $Python" -ForegroundColor Red
    exit 1
}
Push-Location "$ProjectRoot\backend"
try {
    & $Python -m uvicorn main:app --reload --port 8000
} finally {
    Pop-Location
}
