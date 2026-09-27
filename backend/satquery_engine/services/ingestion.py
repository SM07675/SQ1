from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.shutil import copy as raster_copy
from rasterio.windows import Window, bounds as window_bounds

from satquery_engine.schemas import PreprocessingReport


NETCDF_SUFFIXES = {".nc", ".nc4", ".cdf", ".netcdf"}


@dataclass(frozen=True)
class PreparedSource:
    raster_path: Path
    source_format: str
    selected_variable: str | None
    available_variables: list[str]


def _variable_name(identifier: str) -> str:
    match = re.search(r":([^:]+)$", identifier)
    return (match.group(1) if match else Path(identifier).stem).strip('"')


def prepare_source(path: Path, output_dir: Path, variable: str | None = None) -> PreparedSource:
    """Convert a GeoTIFF-compatible or NetCDF source into one analysis GeoTIFF."""
    if path.suffix.lower() not in NETCDF_SUFFIXES:
        with rasterio.open(path) as src:
            if src.count < 1 or src.width < 1 or src.height < 1:
                raise ValueError("Raster contains no readable image bands")
        return PreparedSource(path, "geospatial_raster", None, [])

    with rasterio.open(path) as root:
        subdatasets = list(root.subdatasets)
    if not subdatasets:
        return PreparedSource(path, "netcdf", None, [])

    variables = [_variable_name(item) for item in subdatasets]
    candidates: list[tuple[int, str, str]] = []
    for identifier, name in zip(subdatasets, variables, strict=True):
        try:
            with rasterio.open(identifier) as src:
                score = src.width * src.height * max(1, min(src.count, 16))
                if src.width > 1 and src.height > 1:
                    candidates.append((score, name, identifier))
        except rasterio.errors.RasterioIOError:
            continue
    if not candidates:
        raise ValueError("The NetCDF file has no readable two-dimensional raster variable")
    if variable:
        selected = next((item for item in candidates if item[1].lower() == variable.lower()), None)
        if selected is None:
            raise ValueError(f"NetCDF variable '{variable}' not found. Available variables: {variables}")
    else:
        selected = max(candidates, key=lambda item: item[0])

    _, selected_name, identifier = selected
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = output_dir / f"netcdf_{re.sub(r'[^a-zA-Z0-9_-]+', '_', selected_name)}.tif"
    raster_copy(identifier, converted, driver="GTiff", compress="deflate", tiled=True)
    return PreparedSource(converted, "netcdf", selected_name, variables)


def _starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if values[-1] != last:
        # If the gap between the last stepped tile and the edge is tiny (< 64px),
        # adjust the last start to `last` while maintaining safe overlap (>= 64px)
        # with the preceding tile. This avoids nearly 100% redundant border tiles.
        if len(values) >= 2 and (last - values[-2]) <= (tile_size - 64) and (last - values[-1]) < 64:
            values[-1] = last
        else:
            values.append(last)
    return values


def _normalize(array: np.ndarray) -> np.ndarray:
    values = array.astype("float32")
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values, dtype="uint8")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        if finite.size and finite.max() <= 1.0:
            return (np.clip(values, 0, 1) * 255).astype("uint8")
        return np.clip(values, 0, 255).astype("uint8")
    return (np.clip((values - low) / (high - low), 0, 1) * 255).astype("uint8")


def build_model_tiles(
    path: Path,
    output_dir: Path,
    *,
    public_manifest_url: str,
    source_format: str,
    selected_variable: str | None = None,
    available_variables: list[str] | None = None,
    tile_size: int = 448,
    overlap: int = 64,
    max_tiles: int = 32,
) -> tuple[PreprocessingReport, list[Path]]:
    """Materialize evenly sampled, overlapping RGB tiles with world-coordinate provenance."""
    if tile_size < 64 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be >=64 and overlap must satisfy 0 <= overlap < tile_size")
    step = tile_size - overlap
    tile_dir = output_dir / "model_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    tile_paths: list[Path] = []
    with rasterio.open(path) as src:
        windows = [
            Window(col, row, min(tile_size, src.width - col), min(tile_size, src.height - row))
            for row in _starts(src.height, tile_size, step)
            for col in _starts(src.width, tile_size, step)
        ]
        total = len(windows)
        keep = min(total, max_tiles)
        selected_indices = sorted(set(np.linspace(0, total - 1, keep, dtype=int).tolist())) if total else []
        band_indexes = [1, 2, 3] if src.count >= 3 else [1]
        for sequence, window_index in enumerate(selected_indices):
            window = windows[window_index]
            data = src.read(
                band_indexes,
                window=window,
                out_shape=(len(band_indexes), tile_size, tile_size),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=src.nodata or 0,
            )
            if len(band_indexes) == 1:
                gray = _normalize(data[0])
                rgb = np.stack([gray, gray, gray], axis=-1)
            else:
                if data.dtype == np.uint8:
                    rgb = np.transpose(data[:3], (1, 2, 0))
                elif np.nanmax(data) <= 1.0 and np.nanmin(data) >= 0.0:
                    rgb = np.clip(np.transpose(data[:3], (1, 2, 0)) * 255.0, 0, 255).astype("uint8")
                else:
                    finite = data[np.isfinite(data)]
                    if finite.size:
                        low, high = np.percentile(finite, [2, 98])
                        if high > low:
                            norm_data = np.clip((data[:3] - low) / (high - low), 0, 1) * 255.0
                            rgb = np.transpose(norm_data, (1, 2, 0)).astype("uint8")
                        else:
                            rgb = np.stack([_normalize(data[idx]) for idx in range(3)], axis=-1)
                    else:
                        rgb = np.zeros((tile_size, tile_size, 3), dtype="uint8")
            tile_path = tile_dir / f"tile_{sequence:04d}.png"
            Image.fromarray(rgb, mode="RGB").save(tile_path)
            left, bottom, right, top = window_bounds(window, src.transform)
            records.append({
                "tile_id": sequence,
                "candidate_index": window_index,
                "file": str(tile_path.relative_to(output_dir)),
                "pixel_window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
                "source_x": int(window.col_off),
                "source_y": int(window.row_off),
                "source_width": int(window.width),
                "source_height": int(window.height),
                "model_width": tile_size,
                "model_height": tile_size,
                "source_transform": list(src.transform)[:6],
                "bounds": [left, bottom, right, top],
                "crs": src.crs.to_string() if src.crs else None,
            })
            tile_paths.append(tile_path)

    manifest = {
        "schema_version": "1.0",
        "source": path.name,
        "tile_size": tile_size,
        "overlap": overlap,
        "total_candidate_tiles": total,
        "materialized_tiles": len(tile_paths),
        "complete_coverage": len(tile_paths) == total,
        "selection": "all" if len(tile_paths) == total else "even_spatial_sample",
        "normalization": "per-band percentile 2-98 to uint8",
        "tiles": records,
    }
    manifest_path = output_dir / "tiles_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = PreprocessingReport(
        filename=path.name,
        source_format=source_format,
        selected_variable=selected_variable,
        available_variables=available_variables or [],
        tile_size=tile_size,
        overlap=overlap,
        total_candidate_tiles=total,
        materialized_tiles=len(tile_paths),
        complete_coverage=len(tile_paths) == total,
        normalization="per-band percentile 2-98 to uint8",
        manifest_url=public_manifest_url,
    )
    return report, tile_paths
