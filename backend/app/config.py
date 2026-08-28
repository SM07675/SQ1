from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("SATQUERY_ENV", "development")
    artifact_dir: Path = Path(os.getenv("SATQUERY_ARTIFACT_DIR", "artifacts"))
    max_upload_mb: int = int(os.getenv("SATQUERY_MAX_UPLOAD_MB", "256"))
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


settings = Settings()
