<#
  setup_and_run.ps1

  Creates .venv, installs dependencies, and starts the live ISL pipeline.

  Run from PowerShell:

      powershell -ExecutionPolicy Bypass -File setup_and_run.ps1

  Close the OpenCV window with Q or Esc.
#>

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$pythonLauncher = $null

if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonLauncher = 'py'
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonLauncher = 'python'
}
else {
    throw 'Python was not found. Install Python 3.10-3.12 from https://www.python.org/downloads/windows/ and retry.'
}

if (-not (Test-Path '.venv')) {
    Write-Host 'Creating virtual environment...' -ForegroundColor Cyan
    & $pythonLauncher -m venv .venv
}

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    throw '.venv was not created correctly.'
}

Write-Host 'Installing dependencies...' -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host 'Validating setup...' -ForegroundColor Cyan
& $venvPython check_setup.py
if ($LASTEXITCODE -ne 0) {
    throw 'Setup validation failed. Fix the [FAIL] lines above, then rerun.'
}

Write-Host 'Starting live ISL pipeline...' -ForegroundColor Green
Write-Host 'Local web stream: http://127.0.0.1:5000' -ForegroundColor Green
Write-Host 'Press Q or Esc to exit.' -ForegroundColor Green

& $venvPython live_inference.py