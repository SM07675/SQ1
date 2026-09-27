# SatQuery Vercel Frontend Deployment Script
$ErrorActionPreference = "Continue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "SatQuery AI - Vercel Frontend Deployment" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendDir = Join-Path $RepoRoot "frontend"

Set-Location $FrontendDir

Write-Host "[1/2] Building Frontend for Production..." -ForegroundColor Yellow
npm run build

Write-Host "`n[2/2] Deploying to Vercel (Production)..." -ForegroundColor Yellow
npx --yes vercel --prod --yes

Write-Host "`n Deployment process finished!" -ForegroundColor Green
