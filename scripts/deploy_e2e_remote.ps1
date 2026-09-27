# ==============================================================================
# SatQuery AI - Remote 1-Click Deployment to E2E Networks Linux Node
# Run from Windows PowerShell:
#   powershell .\scripts\deploy_e2e_remote.ps1 -NodeIp "164.52.xxx.xxx"
# ==============================================================================
[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, HelpMessage = "Public IP of your E2E Networks Linux Node")]
    [string]$NodeIp,

    [Parameter(Mandatory = $false)]
    [string]$SshUser = "root",

    [Parameter(Mandatory = $false)]
    [string]$KeyPath = "",

    [Parameter(Mandatory = $false)]
    [int]$SshPort = 22
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   SatQuery AI -> E2E Networks Remote Deployment Engine   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Target Host: $SshUser@$NodeIp (Port: $SshPort)" -ForegroundColor Yellow

# Build standard SSH argument list
$BaseSshArgs = @("-o", "StrictHostKeyChecking=no", "-p", "$SshPort")
if ($KeyPath -and (Test-Path $KeyPath)) {
    $BaseSshArgs += @("-i", $KeyPath)
}

# 1. Test SSH connectivity
Write-Host "`n[1/5] Testing SSH connectivity to $NodeIp..." -ForegroundColor Cyan
try {
    $testArgs = $BaseSshArgs + @("-o", "ConnectTimeout=8", "$SshUser@$NodeIp", "uname -a")
    $sshTest = & ssh.exe $testArgs
    Write-Host "Connected: $sshTest" -ForegroundColor Green
} catch {
    Write-Host "Failed to connect to $SshUser@$NodeIp via SSH." -ForegroundColor Red
    Write-Host "Please ensure:" -ForegroundColor Yellow
    Write-Host " 1. The E2E node is running and Public IP is correct."
    Write-Host " 2. Port 22 is allowed in E2E Networks Security Group / Firewall."
    Write-Host " 3. Your SSH Key or password authentication is valid."
    exit 1
}

# 2. Package required deployment files
Write-Host "`n[2/5] Creating deployment bundle for E2E node..." -ForegroundColor Cyan
$TempZip = Join-Path $env:TEMP "satquery_e2e_bundle.tar.gz"
if (Test-Path $TempZip) { Remove-Item $TempZip -Force }

Write-Host "Archiving backend source, configs, and models..." -ForegroundColor Gray
$TarArgs = @(
    "--exclude=*.venv*",
    "--exclude=__pycache__",
    "--exclude=.pytest_cache",
    "--exclude=node_modules",
    "--exclude=.git",
    "-czf",
    $TempZip,
    "backend",
    "models",
    "Dockerfile.e2e",
    "docker-compose.e2e.yml",
    "scripts/deploy_e2e.sh"
)
& tar.exe $TarArgs

$bundleSizeMB = [math]::Round((Get-Item $TempZip).Length / 1MB, 2)
Write-Host "Bundle created successfully: $bundleSizeMB MB ($TempZip)" -ForegroundColor Green

# 3. Transfer archive to E2E node
Write-Host "`n[3/5] Uploading deployment package to /opt/satquery on remote server..." -ForegroundColor Cyan
$mkdirArgs = $BaseSshArgs + @("$SshUser@$NodeIp", "mkdir -p /opt/satquery")
& ssh.exe $mkdirArgs

$scpArgs = @("-P", "$SshPort", "-o", "StrictHostKeyChecking=no")
if ($KeyPath -and (Test-Path $KeyPath)) {
    $scpArgs += @("-i", $KeyPath)
}
$scpArgs += @($TempZip, "$SshUser@$NodeIp`:/opt/satquery/bundle.tar.gz")
& scp.exe $scpArgs

# 4. Unpack and trigger deploy script on remote server
Write-Host "`n[4/5] Executing automated deployment script on E2E Node..." -ForegroundColor Cyan
$remoteCmd = "cd /opt/satquery && tar -xzf bundle.tar.gz && chmod +x scripts/deploy_e2e.sh && bash scripts/deploy_e2e.sh"
$runArgs = $BaseSshArgs + @("$SshUser@$NodeIp", $remoteCmd)
& ssh.exe $runArgs

# 5. Verify live remote health check from this machine
Write-Host "`n[5/5] Verifying remote API health from Windows local environment..." -ForegroundColor Cyan
$ApiUrl = "http://${NodeIp}:8000"
Start-Sleep -Seconds 5

try {
    $HealthResponse = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 10 -Method Get
    Write-Host "Health Check Succeeded: $($HealthResponse | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not connect to $ApiUrl/health directly from this machine." -ForegroundColor Yellow
    Write-Host "Please ensure Inbound Port 8000 is open in E2E Networks Security Group / Firewall." -ForegroundColor Yellow
}

# 6. Update frontend environment variables
Write-Host "`nUpdating frontend environment configurations..." -ForegroundColor Cyan
$FrontendEnv = Join-Path $ProjectRoot "frontend\.env"
$FrontendEnvProd = Join-Path $ProjectRoot "frontend\.env.production"

Set-Content -Path $FrontendEnv -Value "VITE_API_URL=$ApiUrl" -Encoding utf8
Set-Content -Path $FrontendEnvProd -Value "VITE_API_URL=$ApiUrl" -Encoding utf8

Write-Host "Updated frontend\.env with VITE_API_URL=$ApiUrl" -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "        E2E NETWORKS DEPLOYMENT COMPLETE!                 " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "Live Backend Endpoint : $ApiUrl" -ForegroundColor Cyan
Write-Host "API Documentation     : $ApiUrl/docs" -ForegroundColor Cyan
Write-Host "Frontend Configured   : $FrontendEnv" -ForegroundColor Cyan
Write-Host "`nYou can now deploy or test the frontend with:" -ForegroundColor White
Write-Host "  cd frontend && npm run dev" -ForegroundColor Yellow
Write-Host "  powershell .\scripts\deploy_vercel.ps1" -ForegroundColor Yellow
