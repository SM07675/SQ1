from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom


def write_raster(path: Path, array: np.ndarray, profile: dict[str, Any], valid: np.ndarray, nodata: float | int | None = None) -> Path:
    data = array[None] if array.ndim == 2 else array
    output_profile = dict(profile)
    output_profile.update(driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0], dtype=str(data.dtype), compress="deflate", predictor=2)
    output_profile.pop("photometric", None)
    if nodata is not None:
        output_profile["nodata"] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(data)
        dst.write_mask(valid.astype("uint8") * 255)
    return path


def building_geojson(path: Path, labels: np.ndarray, probability: np.ndarray, transform, crs) -> int:
    features = []
    for geometry, value in shapes(labels.astype("int32"), mask=labels > 0, transform=transform):
        instance_id = int(value)
        mask = labels == instance_id
        output_geometry = transform_geom(crs, "EPSG:4326", geometry, precision=7) if crs else geometry
        features.append({
            "type": "Feature",
            "geometry": output_geometry,
            "properties": {"instance_id": instance_id, "mean_probability": float(probability[mask].mean()), "pixel_area": int(mask.sum())},
        })
    collection = {"type": "FeatureCollection", "features": features, "properties": {"crs": "EPSG:4326" if crs else None}}
    path.write_text(json.dumps(collection, indent=2, allow_nan=False), encoding="utf-8")
    return len(features)

