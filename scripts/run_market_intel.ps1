# scripts/run_market_intel.ps1 — Windows Task Scheduler wrapper.
#
# Register example (PowerShell, run as admin):
#   $A = New-ScheduledTaskAction -Execute "powershell.exe" `
#        -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\path\to\regime_trader\scripts\run_market_intel.ps1"
#   $T = New-ScheduledTaskTrigger -Daily -At 6:30AM
#   Register-ScheduledTask -Action $A -Trigger $T -TaskName "MarketIntel-PreOpen"

param(
    [string]$TickersFile = "top50.csv"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $RepoRoot "logs"
$LogFile  = Join-Path $LogDir "market_intel_runner.log"
$LockFile = Join-Path $env:TEMP "market_intel.lock"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Lightweight lock (skips if another run is active)
if (Test-Path $LockFile) {
    $age = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    if ($age.TotalMinutes -lt 60) {
        Add-Content $LogFile "[$(Get-Date -Format o)] another run is in progress — skipping"
        exit 0
    }
}
"$PID" | Out-File -FilePath $LockFile -Encoding ascii

try {
    Set-Location $RepoRoot
    Add-Content $LogFile "[$(Get-Date -Format o)] starting run with tickers=$TickersFile"

    $Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
    & $Python -m backend.market_intel.run_pipeline `
        --tickers-file $TickersFile `
        --limit-forms 5 `
        --max-workers 4 *>> $LogFile

    Add-Content $LogFile "[$(Get-Date -Format o)] run complete (exit=$LASTEXITCODE)"
}
finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $LockFile
}
