from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _resolve_file_path(env_var: str, *fallbacks: str) -> Path | None:
    backend_root = Path(__file__).resolve().parent.parent
    workspace_root = backend_root.parent

    candidates: list[str] = []
    val = os.getenv(env_var)
    if val:
        candidates.append(val)
    candidates.extend(fallbacks)

    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p.resolve()
        p_b = (backend_root / c).resolve()
        if p_b.is_file():
            return p_b
        p_w = (workspace_root / c).resolve()
        if p_w.is_file():
            return p_w
        p_top = (workspace_root.parent / c).resolve()
        if p_top.is_file():
            return p_top
    return None


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("SATQUERY_ENV", "development")
    artifact_dir: Path = Path(os.getenv("SATQUERY_ARTIFACT_DIR", "artifacts"))
    max_upload_mb: int = int(os.getenv("SATQUERY_MAX_UPLOAD_MB", "256"))
    upload_bucket: str = os.getenv("SATQUERY_UPLOAD_BUCKET", "")
    cors_origins: tuple[str, ...] = _split_csv(
        os.getenv("SATQUERY_CORS_ORIGINS", "http://localhost:5173,http://localhost:8080")
    )
    earthdial_endpoint: str | None = os.getenv("SATQUERY_EARTHDIAL_ENDPOINT") or None
    earthdial_rgb_endpoint: str | None = (
        os.getenv("SATQUERY_EARTHDIAL_RGB_ENDPOINT") or os.getenv("SATQUERY_EARTHDIAL_ENDPOINT") or None
    )
    earthdial_ms_endpoint: str | None = (
        os.getenv("SATQUERY_EARTHDIAL_MS_ENDPOINT") or os.getenv("SATQUERY_EARTHDIAL_ENDPOINT") or None
    )
    croma_endpoint: str | None = os.getenv("SATQUERY_CROMA_ENDPOINT") or None
    remoteclip_endpoint: str | None = os.getenv("SATQUERY_REMOTECLIP_ENDPOINT") or None
    change_endpoint: str | None = os.getenv("SATQUERY_CHANGE_ENDPOINT") or None
    vlm_endpoint: str | None = os.getenv("SATQUERY_VLM_ENDPOINT") or None
    vlm_api_key: str | None = os.getenv("SATQUERY_VLM_API_KEY") or None
    vlm_model_name: str = os.getenv("SATQUERY_VLM_MODEL_NAME", "managed-geospatial-vlm")
    tile_size: int = int(os.getenv("SATQUERY_TILE_SIZE", "448"))
    tile_overlap: int = int(os.getenv("SATQUERY_TILE_OVERLAP", "64"))
    max_model_tiles: int = int(os.getenv("SATQUERY_MAX_MODEL_TILES", "32"))
    database_path: Path = Path(os.getenv("SATQUERY_DATABASE_PATH", "artifacts/satquery.sqlite3"))
    water_model_checkpoint: Path | None = _resolve_file_path(
        "SATQUERY_WATER_CHECKPOINT",
        "models/satquery_water_bundle/satquery_water_state_dict.pt",
        "satquery_water_state_dict.pt",
    )
    water_model_config: Path | None = _resolve_file_path(
        "SATQUERY_WATER_CONFIG",
        "models/satquery_water_bundle/satquery_water_config.json",
        "satquery_water_config.json",
    )
    water_model_metrics: Path | None = _resolve_file_path(
        "SATQUERY_WATER_METRICS",
        "models/satquery_water_bundle/satquery_water_metrics.json",
        "satquery_water_metrics.json",
    )
    water_model_enabled: bool = os.getenv("SATQUERY_WATER_MODEL_ENABLED", "true").lower() in ("true", "1", "yes")
    landcover_model_checkpoint: Path | None = _resolve_file_path(
        "SATQUERY_LANDCOVER_CHECKPOINT",
        "models/finetuned/landcover/satquery_landcover_v1/best_checkpoint.pt",
        "models/landcover/best_checkpoint.pt",
    )
    buildings_model_checkpoint: Path | None = _resolve_file_path(
        "SATQUERY_BUILDINGS_CHECKPOINT",
        "models/finetuned/buildings/satquery_buildings_v1/satquery_buildings_state_dict.pt",
        "models/buildings/satquery_buildings_state_dict.pt",
    )
    landcover_model_enabled: bool = os.getenv("SATQUERY_LANDCOVER_MODEL_ENABLED", "true").lower() in ("true", "1", "yes")
    buildings_model_enabled: bool = os.getenv("SATQUERY_BUILDINGS_MODEL_ENABLED", "true").lower() in ("true", "1", "yes")


settings = Settings()
