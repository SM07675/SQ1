$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $ProjectRoot

$NodeIp = "151.185.58.96"
$KeyPath = "C:\Users\sarve\key"
$SshArgs = @("-o", "StrictHostKeyChecking=no", "-i", $KeyPath)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   SatQuery AI -> Deploying Complete Backend to E2E       " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Upload bundle.zip
$bundleZip = Join-Path $ProjectRoot "deployment\huggingface-space.zip"
Write-Host "[1/4] Uploading complete model & app bundle (2.3 GB)..." -ForegroundColor Cyan
& scp.exe -P 22 -o StrictHostKeyChecking=no -i $KeyPath $bundleZip "root@${NodeIp}:/opt/satquery/bundle.zip"
Write-Host "Upload complete!" -ForegroundColor Green

# 2. Extract on server & setup files
Write-Host "`n[2/4] Unpacking bundle and configuring container environment..." -ForegroundColor Cyan
$setupScript = @'
cd /opt/satquery
unzip -q -o bundle.zip
cp -rn huggingface-space/* .
rm -rf huggingface-space bundle.zip
mkdir -p /opt/satquery/artifacts
'@
& ssh.exe @SshArgs root@$NodeIp $setupScript

# 3. Upload latest Dockerfile and compose
Write-Host "`n[3/4] Uploading production Docker configs..." -ForegroundColor Cyan
$df = Join-Path $ProjectRoot "deployment\huggingface-space\Dockerfile"
$dc = Join-Path $ProjectRoot "deployment\huggingface-space\docker-compose.yml"
& scp.exe -P 22 -o StrictHostKeyChecking=no -i $KeyPath $df "root@${NodeIp}:/opt/satquery/Dockerfile"
& scp.exe -P 22 -o StrictHostKeyChecking=no -i $KeyPath $dc "root@${NodeIp}:/opt/satquery/docker-compose.yml"

# 4. Launch container
Write-Host "`n[4/4] Starting SatQuery production backend container..." -ForegroundColor Cyan
$launchScript = @'
cd /opt/satquery
docker compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.yml up -d --build
'@
& ssh.exe @SshArgs root@$NodeIp $launchScript

Write-Host "Container launched. Verifying healthcheck..." -ForegroundColor Green
