$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $ProjectRoot "backend"
$Frontend = Join-Path $ProjectRoot "frontend"

function Test-SatQueryPython([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    Push-Location $Backend
    try {
        & $Path -c "import fastapi, rasterio, torch, transformers, onnxruntime, cv2, app.main, app.services.orchestrator" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $Frontend "node_modules"))) {
    throw "Frontend packages are missing. Run scripts\setup_windows.ps1 first."
}

$DriveLetter = @("Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z") |
    Where-Object { -not (Test-Path -LiteralPath "${_}:\") } | Select-Object -First 1
if (-not $DriveLetter) { throw "No free drive letter for the Windows short-path runtime." }
$Drive = "${DriveLetter}:"
subst $Drive $ProjectRoot
try {
    $ProjectPython = "$Drive\backend\.venv-integrated\Scripts\python.exe"
    if (-not (Test-SatQueryPython $ProjectPython)) {
        throw "No working SatQuery Python runtime found. Run scripts\setup_windows.ps1 first."
    }
} finally {
    subst $Drive /D
}

$ApiProcess = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    "`"$(Join-Path $PSScriptRoot 'run_api_windows.ps1')`"",
    "-ProjectRoot", "`"$ProjectRoot`"", "-DriveLetter", $DriveLetter
) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
$WebProcess = Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev") -WorkingDirectory $Frontend -WindowStyle Hidden -PassThru

Write-Host "SatQuery API: http://127.0.0.1:8000/docs (process $($ApiProcess.Id))" -ForegroundColor Cyan
Write-Host "SatQuery dashboard: http://localhost:5173 (process $($WebProcess.Id))" -ForegroundColor Cyan
Write-Host "Local model status: http://127.0.0.1:8000/api/v1/models/status" -ForegroundColor Cyan
