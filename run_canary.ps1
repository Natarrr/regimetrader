Write-Host "=== Canary script started ==="

# Resolve project root — three fallbacks for any PS context
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }
Set-Location -LiteralPath $root
Write-Host "Project root: $root"

# Activate venv
$venv = Join-Path $root ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venv)) {
    Write-Host "ERROR: venv not found at $venv" -ForegroundColor Red
    exit 1
}

Write-Host "Activating venv..."
& $venv

# Ensure PYTHONPATH — append, never overwrite
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$root;$env:PYTHONPATH" } else { $env:PYTHONPATH = $root }

# Run pipeline
Write-Host "Running EDGAR pipeline..."
python -m backend.market_intel.run_pipeline --tickers-file config/canary_top10.csv --limit-forms 5 --verbose
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline failed" -ForegroundColor Red
    exit 2
}

# Export metrics
Write-Host "Exporting metrics..."
python -m monitoring.metrics_exporter
if ($LASTEXITCODE -ne 0) {
    Write-Host "metrics_exporter failed" -ForegroundColor Red
    exit 3
}

# Check metrics
Write-Host "Checking metrics..."
python -m monitoring.check_metrics
if ($LASTEXITCODE -ne 0) {
    Write-Host "Canary FAILED" -ForegroundColor Red
    exit 4
}

Write-Host "=== Canary PASSED ===" -ForegroundColor Green
exit 0
