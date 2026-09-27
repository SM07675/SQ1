# SatQuery Modal Backend Deployment Script
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "SatQuery AI - Modal Serverless Backend Deployment" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonExe = Join-Path $RepoRoot "backend\.venv-integrated\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Set-Location $RepoRoot

Write-Host "[1/2] Verifying Modal Authentication..." -ForegroundColor Yellow
& $PythonExe -m modal profile current

Write-Host "[2/2] Deploying modal_app.py to Modal..." -ForegroundColor Yellow
& $PythonExe -m modal deploy modal_app.py

Write-Host "`n Deployment complete!" -ForegroundColor Green
Write-Host "Production URL: https://sarveshm4444--satquery-api-fastapi-app.modal.run" -ForegroundColor Cyan
Write-Host "Swagger Docs:   https://sarveshm4444--satquery-api-fastapi-app.modal.run/docs" -ForegroundColor Cyan
Write-Host "Health Check:   https://sarveshm4444--satquery-api-fastapi-app.modal.run/health" -ForegroundColor Cyan
