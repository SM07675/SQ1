param([switch]$Apply)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Targets = @(
    "models/cache",
    "models/earthdial",
    "models/pretrained/vlm",
    "models/vlm",
    "models/croma",
    "models/pretrained/fusion",
    "models/fusion",
    "models/remoteclip",
    "models/pretrained/remoteclip",
    "models/water/prithvi-sen1floods11",
    "models/pretrained/change",
    "models/land_cover/bigearthnet-s1-resnet50",
    "models/land_cover/bigearthnet-s1s2-resnet101",
    "models/land_cover/bigearthnet-s2-resnet50",
    "models/buildings/dinov3s-buildings",
    "models/waternet",
    "models/satquery_buildings_bundle",
    "models/satquery_landcover_v1_bundle",
    "model",
    "backend/models",
    "backend/.pip-cache",
    "backend/.pip-wheels",
    "models/finetuned/water/satquery_water_shadow_v1/satquery_water_state_dict.pt",
    "deployment/vercel-frontend/node_modules",
    "deployment/vercel-frontend/dist",
    "deployment/vercel-frontend/tsconfig.app.tsbuildinfo"
)

foreach ($Relative in $Targets) {
    $Target = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Relative))
    if (-not $Target.StartsWith($ProjectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to touch a path outside the workspace: $Target"
    }
    if (-not (Test-Path -LiteralPath $Target)) { continue }
    if ($Apply) {
        Remove-Item -LiteralPath $Target -Recurse -Force
        Write-Output "Removed: $Relative"
    } else {
        Write-Output "Would remove: $Relative"
    }
}
