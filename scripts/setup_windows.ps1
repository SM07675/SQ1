$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Set-Location $ProjectRoot
python -m venv "backend\.venv-integrated"
$DriveLetter = @("Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z") |
    Where-Object { -not (Test-Path -LiteralPath "${_}:\") } | Select-Object -First 1
if (-not $DriveLetter) { throw "No free drive letter for the Windows short-path installer." }
$Drive = "${DriveLetter}:"
subst $Drive $ProjectRoot
try {
    $ShortPython = "$Drive\backend\.venv-integrated\Scripts\python.exe"
    & $ShortPython -m pip install --no-cache-dir --upgrade pip
    & $ShortPython -m pip install --no-cache-dir -e "$Drive\backend[dev]"
    & $ShortPython "$Drive\scripts\make_demo_data.py"
} finally {
    subst $Drive /D
}

Push-Location "frontend"
npm install
Pop-Location

Write-Host ""
Write-Host "SatQuery GeoProof setup is complete." -ForegroundColor Green
Write-Host "Run: powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1"
