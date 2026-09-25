"""Build clean, standalone Hugging Face and Vercel upload directories."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).absolute().parents[1]
OUTPUT = ROOT / "deployment"
SPACE = OUTPUT / "huggingface-space"
WEB = OUTPUT / "vercel-frontend"

# Keep only checkpoints reachable by the local inference paths. Endpoint-only
# EarthDial/CROMA/RemoteCLIP weights and unimplemented research adapters are excluded.
MODEL_KEYS = (
    "water_s2_surface",
    "land_deepness",
    "building_satellite",
    "building_primary",
    "building_secondary",
    "water_finetuned",
    "land_rgb",
    "satlas_aerial_swinb_si",
    "land_flair_hub",
    "landcover_candidate",
    "water_shadow_rgb",
)


def copy_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".cache", "tests", "notebooks", "*.ipynb"),
    )


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing deployment files: {OUTPUT}")
    SPACE.mkdir(parents=True)
    WEB.mkdir()

    for name in ("Dockerfile", "README.md", ".dockerignore", ".gitattributes"):
        shutil.copy2(ROOT / "infra" / "huggingface" / name, SPACE / name)
    shutil.copy2(ROOT / "backend" / "pyproject.toml", SPACE / "pyproject.toml")
    copy_directory(ROOT / "backend" / "app", SPACE / "app")
    copy_directory(ROOT / "backend" / "satquery_engine", SPACE / "satquery_engine")

    source_manifest = yaml.safe_load((ROOT / "models" / "manifests" / "models.yaml").read_text(encoding="utf-8"))
    entries = source_manifest["models"]
    selected = {key: entries[key] for key in MODEL_KEYS}
    for key, entry in selected.items():
        relative_path = Path(entry["path"])
        source = ROOT / "models" / relative_path
        target = SPACE / "models" / relative_path
        print(f"Copying model {key}: {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_directory(source, target)
    # This file duplicates the ten-band water checkpoint in satquery_water_bundle.
    extra_water = SPACE / "models" / "finetuned" / "water" / "satquery_water_shadow_v1" / "satquery_water_state_dict.pt"
    extra_water.unlink(missing_ok=True)
    manifest_dir = SPACE / "models" / "manifests"
    manifest_dir.mkdir(parents=True)
    source_manifest["models"] = selected
    (manifest_dir / "models.yaml").write_text(yaml.safe_dump(source_manifest, sort_keys=False), encoding="utf-8")

    frontend = ROOT / "frontend"
    for name in ("package.json", "package-lock.json", "index.html", "vite.config.ts",
                 "tsconfig.json", "tsconfig.app.json", "vercel.json", ".env.example"):
        shutil.copy2(frontend / name, WEB / name)
    copy_directory(frontend / "src", WEB / "src")
    if (frontend / "public").is_dir():
        copy_directory(frontend / "public", WEB / "public")

    summary = {
        "space_files": sum(1 for file in SPACE.rglob("*") if file.is_file()),
        "space_gb": round(sum(file.stat().st_size for file in SPACE.rglob("*") if file.is_file()) / 2**30, 2),
        "web_files": sum(1 for file in WEB.rglob("*") if file.is_file()),
        "web_mb": round(sum(file.stat().st_size for file in WEB.rglob("*") if file.is_file()) / 2**20, 2),
        "included_models": list(MODEL_KEYS),
    }
    (OUTPUT / "package-manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
