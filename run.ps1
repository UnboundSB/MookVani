<#
  run.ps1

  Starts the live ISL pipeline without reinstalling dependencies.
#>

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Write-Host 'Virtual environment not found. Running setup first...' -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'setup_and_run.ps1')
    exit
}

& $venvPython live_inference.py