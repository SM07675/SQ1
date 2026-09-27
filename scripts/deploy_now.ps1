# ==============================================================================
# SatQuery AI - Direct E2E Networks Deployment Pipeline
# ==============================================================================
$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $ProjectRoot

$NodeIp = "151.185.58.96"
$KeyPath = "C:\Users\sarve\key"
$SshArgs = @("-o", "StrictHostKeyChecking=no", "-i", $KeyPath)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   SatQuery AI -> Deploying to E2E Networks ($NodeIp)    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check if models are already present on remote server
Write-Host "[1/5] Checking remote model status on server..." -ForegroundColor Cyan
$remoteCheck = & ssh.exe @SshArgs root@$NodeIp "test -f /opt/satquery/models/manifests/models.yaml && echo 'MODELS_EXIST' || echo 'MODELS_MISSING'"

if ($remoteCheck -match "MODELS_MISSING") {
    Write-Host "Models missing on server. Uploading pre-packaged model bundle (~2.3 GB)..." -ForegroundColor Yellow
    Write-Host "This will take ~2-3 minutes at ~13 MB/s. Please stand by..." -ForegroundColor Gray
    
    $bundleZip = Join-Path $ProjectRoot "deployment\huggingface-space.zip"
    if (-not (Test-Path $bundleZip)) {
        throw "Could not find $bundleZip"
    }

    # Upload zip bundle
    & scp.exe -P 22 -o StrictHostKeyChecking=no -i $KeyPath $bundleZip "root@${NodeIp}:/opt/satquery/bundle.zip"
    
    # Extract on server
    Write-Host "Extracting model bundle on server..." -ForegroundColor Cyan
    & ssh.exe @SshArgs root@$NodeIp "cd /opt/satquery && unzip -q -o bundle.zip && cp -rn huggingface-space/* . && rm -rf huggingface-space bundle.zip"
} else {
    Write-Host "Models already present on remote server! Skipping 2.3 GB upload." -ForegroundColor Green
}

# 2. Package and upload latest source code & Docker configs
Write-Host "`n[2/5] Packaging and syncing latest backend code and Dockerfile..." -ForegroundColor Cyan
$codeTar = Join-Path $env:TEMP "satquery_latest_code.tar.gz"
if (Test-Path $codeTar) { Remove-Item $codeTar -Force }

$TarArgs = @(
    "--exclude=*.venv*",
    "--exclude=__pycache__",
    "--exclude=.pytest_cache",
    "-czf",
    $codeTar,
    "-C",
    (Join-Path $ProjectRoot "deployment\huggingface-space"),
    "app",
    "satquery_engine",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml"
)
& tar.exe $TarArgs

Write-Host "Uploading latest application code..." -ForegroundColor Gray
& scp.exe -P 22 -o StrictHostKeyChecking=no -i $KeyPath $codeTar "root@${NodeIp}:/opt/satquery/code.tar.gz"
& ssh.exe @SshArgs root@$NodeIp "cd /opt/satquery && tar -xzf code.tar.gz && rm -f code.tar.gz"
Remove-Item $codeTar -Force

# 3. Build and launch Docker container
Write-Host "`n[3/5] Building and launching SatQuery backend container on E2E Node..." -ForegroundColor Cyan
$dockerCmd = "cd /opt/satquery && docker compose down --remove-orphans 2>/dev/null || true && docker compose up -d --build"
& ssh.exe @SshArgs root@$NodeIp $dockerCmd

# 4. Wait for health check
Write-Host "`n[4/5] Polling container health on http://$NodeIp:8000/health..." -ForegroundColor Cyan
$maxAttempts = 30
$attempt = 0
$healthy = $false

while ($attempt -lt $maxAttempts) {
    try {
        $res = Invoke-RestMethod -Uri "http://${NodeIp}:8000/health" -TimeoutSec 5 -Method Get -ErrorAction Stop
        if ($res.status -eq "ok" -or $res.status -eq "healthy") {
            $healthy = $true
            break
        }
    } catch {
        # Container starting
    }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 4
    $attempt++
}
Write-Host ""

if ($healthy) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "   SATQUERY AI BACKEND IS LIVE ON E2E NETWORKS!           " -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "Public API URL : http://${NodeIp}:8000" -ForegroundColor Cyan
    Write-Host "API Health     : http://${NodeIp}:8000/health" -ForegroundColor Cyan
    Write-Host "Swagger Docs   : http://${NodeIp}:8000/docs" -ForegroundColor Cyan

    # 5. Update frontend configuration
    Write-Host "`n[5/5] Updating frontend configuration..." -ForegroundColor Cyan
    $FrontendEnv = Join-Path $ProjectRoot "frontend\.env"
    $FrontendEnvProd = Join-Path $ProjectRoot "frontend\.env.production"
    
    Set-Content -Path $FrontendEnv -Value "VITE_API_URL=http://${NodeIp}:8000" -Encoding utf8
    Set-Content -Path $FrontendEnvProd -Value "VITE_API_URL=http://${NodeIp}:8000" -Encoding utf8
    Write-Host "Updated frontend\.env with VITE_API_URL=http://${NodeIp}:8000" -ForegroundColor Green
} else {
    Write-Host "`nContainer build finished but health check did not return 200 within timeout." -ForegroundColor Yellow
    Write-Host "Fetching container logs from server..." -ForegroundColor Cyan
    & ssh.exe @SshArgs root@$NodeIp "docker logs --tail 50 satquery-backend"
}
