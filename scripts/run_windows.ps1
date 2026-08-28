$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $ProjectRoot "backend"
$Frontend = Join-Path $ProjectRoot "frontend"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Backend'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Frontend'; npm run dev"

Write-Host "Backend: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Dashboard: http://localhost:5173" -ForegroundColor Cyan

