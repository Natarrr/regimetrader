Write-Host "=== test.ps1 started ===" -ForegroundColor Cyan
Write-Host "TEST OK - PowerShell executes scripts"
Write-Host "  PS version : $($PSVersionTable.PSVersion)"
Write-Host "  Policy     : $(Get-ExecutionPolicy)"
Write-Host "  User       : $env:USERNAME"
Write-Host "=== test.ps1 done ===" -ForegroundColor Cyan
