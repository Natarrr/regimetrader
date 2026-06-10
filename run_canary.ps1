Write-Host "=== Canary script started ==="

# Resolve project root — three fallbacks for any PS context
$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $root) { $root = (Get-Location).Path }
Set-Location -LiteralPath $root
Write-Host "Project root: $root"

# Activate venv
$venv = Join-Path $root ".venv\Scripts\Activate.ps1"
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venv)) {
    Write-Host "ERROR: venv not found at $venv" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $python)) {
    Write-Host "ERROR: Python executable not found at $python" -ForegroundColor Red
    exit 1
}

Write-Host "Activating venv..."
. $venv

# Load local .env if present so FMP_API_KEY and other secrets are available
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Write-Host "Loading .env variables from $envFile"
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) { return }
        $idx = $line.IndexOf('=')
        if ($idx -lt 0) { return }
        $name = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1)
        if ($name) { Set-Item -Path "env:$name" -Value $value }
    }
    Write-Host "FMP_API_KEY loaded:" ([bool]$env:FMP_API_KEY)
} else {
    Write-Host "No .env file found at $envFile"
}

# Ensure PYTHONPATH — append, never overwrite
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$root;$env:PYTHONPATH" } else { $env:PYTHONPATH = $root }

# Run pipeline
Write-Host "Running EDGAR pipeline..."
& $python src\ingestion\run_pipeline.py --tickers-file config/canary_top10.csv --max-workers 4 --verbose
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline failed" -ForegroundColor Red
    exit 2
}

# Export metrics
Write-Host "Exporting metrics..."
& $python -m monitoring.metrics_exporter
if ($LASTEXITCODE -ne 0) {
    Write-Host "metrics_exporter failed" -ForegroundColor Red
    exit 3
}

# Check metrics
Write-Host "Checking metrics..."
& $python -m monitoring.check_metrics
if ($LASTEXITCODE -ne 0) {
    Write-Host "Canary FAILED" -ForegroundColor Red
    exit 4
}

Write-Host "=== Canary PASSED ===" -ForegroundColor Green
exit 0
