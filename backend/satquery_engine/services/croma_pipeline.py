from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from satquery_engine.services.model_registry import ModelRegistry
from satquery_engine.services.spectral import BAND_ALIASES, _canonical_band_map, read_index


@dataclass(frozen=True)
class CROMAFusedResult:
    producer: str
    sensor_agreement_score: float
    radar_optical_iou: float
    confirmed_area_m2: float | None
    region_count: int
    sar_channels_prepared: list[str]
    optical_channels_prepared: list[str]
    confidence: float
    agreement_mask_path: Path
    sar_db_preview_path: Path
    optical_preview_path: Path
    geojson_path: Path
    croma_features: dict[str, Any]
    cosine_similarity: float = 0.88
    confirmed_percent: float = 0.0

    @property
    def sensor_agreement_iou(self) -> float:
        return self.radar_optical_iou

    @property
    def area_m2(self) -> float | None:
        return self.confirmed_area_m2


def is_sar_image(path: Path) -> bool:
    """Detects whether a raster is a SAR image (Sentinel-1 VV/VH)."""
    p_str = path.name.lower()
    if any(k in p_str for k in ("sar", "radar", "s1", "sentinel1", "sentinel-1")):
        return True
    try:
        with rasterio.open(path) as src:
            descriptions = [str(d).lower() for d in (src.descriptions or ())]
            if any(d in ("vv", "vh", "hh", "hv") for d in descriptions):
                return True
            if src.count <= 2:
                sample = src.read(1, out_shape=(32, 32), resampling=Resampling.nearest)
                finite = sample[np.isfinite(sample)]
                if finite.size and np.min(finite) < -5.0:
                    return True
    except Exception:
        pass
    return False


def _area_m2(geometry: dict[str, Any], crs: Any) -> float | None:
    polygon = shape(geometry)
    if polygon.is_empty or crs is None:
        return None
    try:
        transformer = Transformer.from_crs(crs, "EPSG:6933", always_xy=True)
        projected = shapely_transform(transformer.transform, polygon)
        return abs(float(projected.area))
    except Exception:
        return None


def _polygonize(
    mask: np.ndarray,
    transform: Affine,
    crs: Any,
    kind: str,
    output_path: Path,
) -> tuple[int, float | None]:
    features: list[dict[str, Any]] = []
    total_area = 0.0
    area_available = crs is not None
    minimum_native_area = abs(transform.a * transform.e - transform.b * transform.d) * 4

    for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if int(value) != 1:
            continue
        if shape(geometry).area < minimum_native_area:
            continue
        area = _area_m2(geometry, crs)
        if area is None:
            area_available = False
        else:
            total_area += area
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {"kind": kind, "area_m2": round(area, 3) if area is not None else None},
        })

    features.sort(key=lambda f: f["properties"]["area_m2"] or 0, reverse=True)
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"crs": crs.to_string() if crs else None, "producer": "croma_sensor_fusion_v1"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(features), round(total_area, 3) if area_available else None


def prepare_sentinel1_sar(sar_path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Converts Sentinel-1 SAR bands (VV, VH) to calibrated normalized dB representations."""
    with rasterio.open(sar_path) as src:
        band_names = [d.lower() if d else f"band_{i}" for i, d in enumerate(src.descriptions or (), start=1)]
        arr = src.read(
            out_shape=(src.count, target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear,
        ).astype("float32")

    prepared_channels = []
    normalized_bands = []

    for i in range(arr.shape[0]):
        band_data = arr[i]
        bname = band_names[i] if i < len(band_names) else f"band_{i+1}"
        prepared_channels.append(bname)

        finite = band_data[np.isfinite(band_data)]
        if not finite.size:
            normalized_bands.append(np.zeros_like(band_data))
            continue

        # Check if already in dB (typical values between -40 and +5) or linear intensity (> 0)
        if np.min(finite) >= 0 and np.max(finite) > 1.0:
            # Linear power intensity -> convert to dB
            db = 10.0 * np.log10(np.clip(band_data, 1e-6, None))
        else:
            db = band_data

        # Normalize typical SAR range [-30 dB, 0 dB] -> [0.0, 1.0]
        norm = np.clip((db - (-30.0)) / 30.0, 0.0, 1.0)
        normalized_bands.append(norm)

    return np.stack(normalized_bands, axis=0), prepared_channels


def prepare_sentinel2_optical(optical_path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Prepares and normalizes Sentinel-2 multispectral reflectance bands."""
    with rasterio.open(optical_path) as src:
        band_names = [d.lower() if d else f"b{i}" for i, d in enumerate(src.descriptions or (), start=1)]
        arr = src.read(
            out_shape=(src.count, target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear,
        ).astype("float32")

    prepared_channels = []
    normalized_bands = []

    for i in range(arr.shape[0]):
        bdata = arr[i]
        bname = band_names[i] if i < len(band_names) else f"b{i+1}"
        prepared_channels.append(bname)

        finite = bdata[np.isfinite(bdata)]
        if not finite.size:
            normalized_bands.append(np.zeros_like(bdata))
            continue

        p2, p98 = np.percentile(finite, [2, 98])
        norm = np.clip((bdata - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
        normalized_bands.append(norm)

    return np.stack(normalized_bands, axis=0), prepared_channels


async def run_croma_fusion(
    optical_path: Path,
    sar_path: Path,
    output_dir: Path,
    registry: ModelRegistry,
) -> CROMAFusedResult:
    raise ValueError("The CROMA fusion specialist is unavailable; no fused result was generated.")
