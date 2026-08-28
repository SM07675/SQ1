$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Set-Location $ProjectRoot
python -m venv "backend\.venv"
& "backend\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "backend\.venv\Scripts\python.exe" -m pip install -e "backend[dev]"
& "backend\.venv\Scripts\python.exe" "scripts\make_demo_data.py"

Push-Location "frontend"
npm install
Pop-Location

Write-Host ""
Write-Host "SatQuery GeoProof setup is complete." -ForegroundColor Green
Write-Host "Run: powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1"

