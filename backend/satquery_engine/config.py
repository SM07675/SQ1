from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Search workspace root and backend directory for .env
_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
_backend_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env)
if _backend_env.exists():
    load_dotenv(_backend_env)
load_dotenv()

_default_model_dir = Path(__file__).resolve().parents[2] / "models"
_configured_model_dir = Path(os.getenv("SATQUERY_MODEL_DIR", str(_default_model_dir)))
os.environ.setdefault("HF_HOME", str(_configured_model_dir / "cache" / "huggingface"))


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def default_building_checkpoint(model_dir: Path) -> Path:
    """Prefer the measured satellite instance model; retain bundle-only installs."""
    satellite = _registered_model_path(model_dir, "building_satellite") or model_dir / "buildings/rf-detr-seg-satellite-buildings"
    if all((satellite / name).is_file() for name in ("model.safetensors", "config.json", "preprocessor_config.json")):
        return satellite
    if not (model_dir / "manifests/models.yaml").is_file():
        return model_dir / "buildings/rf-detr-seg-satellite-buildings"
    registered = _registered_model_path(model_dir, "building_primary")
    if registered is not None:
        return registered
    for bundle in (model_dir / "finetuned/buildings/satquery_buildings_v1",
                   model_dir / "satquery_buildings_bundle"):
        if (bundle / "satquery_buildings_config.json").is_file():
            return bundle
    return model_dir / "buildings/rf-detr-seg-satellite-buildings"


def _registered_model_path(model_dir: Path, key: str) -> Path | None:
    manifest = model_dir / "manifests/models.yaml"
    if not manifest.is_file():
        return None
    try:
        import yaml
        raw = (yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}).get("models", {}).get(key, {})
        configured = Path(str(raw["path"]))
        path = configured if configured.is_absolute() else model_dir / configured
        if path.resolve().is_relative_to(model_dir.resolve()):
            return path
    except (KeyError, OSError, ValueError, TypeError):
        pass
    return None


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("SATQUERY_ENV", "development")
    artifact_dir: Path = Path(os.getenv("SATQUERY_ARTIFACT_DIR", "artifacts"))
    max_upload_mb: int = int(os.getenv("SATQUERY_MAX_UPLOAD_MB", "256"))
    cors_origins: tuple[str, ...] = _split_csv(
        os.getenv("SATQUERY_CORS_ORIGINS", "http://localhost:5173,http://localhost:8080")
    )
    model_dir: Path = _configured_model_dir
    offline_mode: bool = os.getenv("SATQUERY_OFFLINE_MODE", "true").lower() in ("true", "1", "yes")
    earthdial_endpoint: str | None = (
        os.getenv("SATQUERY_EARTHDIAL_ENDPOINT") or None
    )
    earthdial_rgb_endpoint: str | None = (
        os.getenv("SATQUERY_EARTHDIAL_RGB_ENDPOINT") or os.getenv("SATQUERY_EARTHDIAL_ENDPOINT") or None
    )
    earthdial_ms_endpoint: str | None = (
        os.getenv("SATQUERY_EARTHDIAL_MS_ENDPOINT") or os.getenv("SATQUERY_EARTHDIAL_ENDPOINT") or None
    )
    croma_endpoint: str | None = (
        os.getenv("SATQUERY_CROMA_ENDPOINT") or None
    )
    remoteclip_endpoint: str | None = (
        os.getenv("SATQUERY_REMOTECLIP_ENDPOINT") or None
    )
    change_endpoint: str | None = (
        os.getenv("SATQUERY_CHANGE_ENDPOINT") or None
    )
    vlm_endpoint: str | None = (
        os.getenv("SATQUERY_VLM_ENDPOINT")
        or os.getenv("SATQUERY_EARTHDIAL_ENDPOINT")
        or None
    )
    vlm_api_key: str | None = os.getenv("SATQUERY_VLM_API_KEY") or None
    vlm_model_name: str = os.getenv("SATQUERY_VLM_MODEL_NAME", "managed-geospatial-vlm")
    building_checkpoint: Path = Path(
        os.getenv(
            "SATQUERY_BUILDING_CHECKPOINT",
            str(default_building_checkpoint(model_dir)),
        )
    )
    water_checkpoint: Path = Path(
        os.getenv(
            "SATQUERY_SURFACE_WATER_CHECKPOINT",
            str(_registered_model_path(model_dir, "water_finetuned") or model_dir / "satquery_water_bundle"),
        )
    )
    s2_water_checkpoint: Path = Path(
        os.getenv("SATQUERY_S2_WATER_CHECKPOINT",
                  str(model_dir / "water/s2-water-unetplusplus-efficientnet-b4/model.pth"))
    )
    tile_size: int = int(os.getenv("SATQUERY_TILE_SIZE", "448"))
    tile_overlap: int = int(os.getenv("SATQUERY_TILE_OVERLAP", "64"))
    max_model_tiles: int = int(os.getenv("SATQUERY_MAX_MODEL_TILES", "32"))
    database_path: Path = Path(os.getenv("SATQUERY_DATABASE_PATH", "artifacts/satquery.sqlite3"))


settings = Settings()
