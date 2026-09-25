param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$DriveLetter
)

$ErrorActionPreference = "Stop"
$Drive = "${DriveLetter}:"
subst $Drive $ProjectRoot
try {
    $ShortBackend = "$Drive\backend"
    Set-Location $ShortBackend
    & (Join-Path $ShortBackend ".venv-integrated\Scripts\python.exe") -m uvicorn app.main:app --host 127.0.0.1 --port 8000
} finally {
    subst $Drive /D
}
