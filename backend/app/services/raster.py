from __future__ import annotations

import json
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

from app.schemas import QualityReport, RasterMetadata
from app.services.spectral import available_indices


def _normalize(array: np.ndarray) -> np.ndarray:
    arr = array.astype("float32")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype="float32")
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - low) / (high - low), 0, 1)


def inspect_raster(path: Path) -> RasterMetadata:
    with rasterio.open(path) as src:
        sample_h = min(src.height, 512)
        sample_w = min(src.width, 512)
        mask = src.read_masks(1, out_shape=(sample_h, sample_w), resampling=Resampling.nearest)
        nodata_percent = float((mask == 0).mean() * 100)
        descriptions = list(src.descriptions or ())
        band_names = [descriptions[i] or f"band_{i + 1}" for i in range(src.count)]

        # Sensor modality and format inspection
        from app.services.croma_pipeline import inspect_modality
        mod_report = inspect_modality(path)

        return RasterMetadata(
            filename=path.name,
            width=src.width,
            height=src.height,
            bands=src.count,
            dtype=str(src.dtypes[0]),
            crs=src.crs.to_string() if src.crs else None,
            bounds=[float(v) for v in src.bounds],
            resolution=[abs(float(src.res[0])), abs(float(src.res[1]))],
            nodata=float(src.nodata) if src.nodata is not None else None,
            nodata_percent=round(nodata_percent, 3),
            band_names=band_names,
            available_indices=available_indices(path),
            source_format=mod_report.file_format,
            file_format=mod_report.file_format,
            modality=mod_report.modality,
            sensor_verified=mod_report.sensor_verified,
            georeferenced=mod_report.georeferenced,
            metadata_available=mod_report.metadata_available,
            analysis_capabilities=mod_report.analysis_capabilities,
            limitations=mod_report.limitations,
        )


def inspect_image_quality(path: Path, max_size: int = 1024) -> dict:
    """
    Stage 1 quality diagnostics: compute brightness, dynamic range,
    potential cloud/shadow coverage, and saturation statistics.

    Returns a dict with quality metrics that feed the arbiter.
    """
    try:
        with rasterio.open(path) as src:
            ratio = min(1.0, max_size / max(src.width, src.height))
            out_h = max(1, round(src.height * ratio))
            out_w = max(1, round(src.width * ratio))

            # Read first 3 bands (or fewer)
            n_bands = min(src.count, 3)
            bands = []
            for i in range(1, n_bands + 1):
                band = src.read(i, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                bands.append(band)

            if not bands:
                return {"quality_score": 0.5, "diagnostics": "No readable bands"}

            stack = np.stack(bands, axis=0)

            # Normalize to [0, 1]
            max_val = float(stack.max())
            if max_val > 1.5:
                stack = stack / (255.0 if max_val <= 255.0 else max_val)
            stack = np.clip(stack, 0.0, 1.0)

            # Compute luminance (mean across channels)
            lum = stack.mean(axis=0)
            mean_brightness = float(lum.mean())
            std_brightness = float(lum.std())

            # Dynamic range (2nd to 98th percentile)
            p2, p98 = np.percentile(lum, [2, 98])
            dynamic_range = float(p98 - p2)

            # Potential cloud pixels (very bright, low contrast)
            cloud_suspect = float((lum > 0.90).mean() * 100)

            # Potential shadow pixels (very dark)
            shadow_suspect = float((lum < 0.05).mean() * 100)

            # Saturated pixels (any channel at max)
            saturated = float((stack.max(axis=0) > 0.99).mean() * 100)

            # Quality score: penalize clouds, shadows, low dynamic range, saturation
            quality_score = 1.0
            quality_score -= min(0.3, cloud_suspect / 100 * 0.5)
            quality_score -= min(0.2, shadow_suspect / 100 * 0.4)
            quality_score -= max(0, 0.3 - dynamic_range) * 0.5
            quality_score -= min(0.1, saturated / 100 * 0.2)
            quality_score = round(max(0.1, min(1.0, quality_score)), 3)

            return {
                "quality_score": quality_score,
                "mean_brightness": round(mean_brightness, 4),
                "std_brightness": round(std_brightness, 4),
                "dynamic_range": round(dynamic_range, 4),
                "cloud_suspect_percent": round(cloud_suspect, 2),
                "shadow_suspect_percent": round(shadow_suspect, 2),
                "saturated_percent": round(saturated, 2),
                "image_size": [out_w, out_h],
                "n_bands": n_bands,
            }
    except Exception:
        return {"quality_score": 0.5, "diagnostics": "Could not compute quality metrics"}

def validate_inputs(paths: list[Path]) -> tuple[list[RasterMetadata], QualityReport]:
    metadata = [inspect_raster(path) for path in paths]
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {"file_count": len(paths)}

    has_any_crs = any(item.crs is not None for item in metadata)
    all_have_crs = all(item.crs is not None for item in metadata)
    checks["geospatial"] = all_have_crs

    for item in metadata:
        if item.crs is None:
            warnings.append(
                f"{item.filename}: missing CRS — pixel-space analysis only; "
                "area/distance values will not be available"
            )
        if item.nodata_percent > 40:
            warnings.append(f"{item.filename}: {item.nodata_percent:.1f}% NoData")
        if item.bands < 1:
            blockers.append(f"{item.filename}: no readable bands")
        if not item.available_indices:
            warnings.append(
                f"{item.filename}: no deterministic spectral index is available; name Red/Green/NIR/SWIR bands for NDVI/NDWI/NDBI"
            )

    compatible = not blockers
    if len(metadata) == 2:
        a, b = metadata
        same_shape = (a.width, a.height) == (b.width, b.height)

        if all_have_crs:
            # Full geospatial validation when both images have CRS
            same_crs = a.crs == b.crs
            same_bounds = bool(np.allclose(a.bounds, b.bounds, rtol=0, atol=max(a.resolution + b.resolution)))
            resolution_ratio = max(a.resolution[0], b.resolution[0]) / max(min(a.resolution[0], b.resolution[0]), 1e-9)
            intersection_width = max(0.0, min(a.bounds[2], b.bounds[2]) - max(a.bounds[0], b.bounds[0]))
            intersection_height = max(0.0, min(a.bounds[3], b.bounds[3]) - max(a.bounds[1], b.bounds[1]))
            intersection_area = intersection_width * intersection_height
            a_area = max(1e-9, (a.bounds[2] - a.bounds[0]) * (a.bounds[3] - a.bounds[1]))
            b_area = max(1e-9, (b.bounds[2] - b.bounds[0]) * (b.bounds[3] - b.bounds[1]))
            overlap_ratio = intersection_area / min(a_area, b_area)
            alignment_score = 1.0 if same_crs and same_shape and same_bounds else max(0.0, overlap_ratio / resolution_ratio)
            checks.update({
                "same_crs": same_crs,
                "same_shape": same_shape,
                "same_bounds": same_bounds,
                "resolution_ratio": round(resolution_ratio, 4),
                "geographic_overlap_ratio": round(overlap_ratio, 4),
                "alignment_score": round(alignment_score, 4),
            })
            if not same_crs:
                warnings.append("Paired rasters use different CRS values — will reproject to primary raster CRS")
            if not same_shape:
                warnings.append(f"Paired rasters have different pixel dimensions ({a.width}×{a.height} vs {b.width}×{b.height}); auto-normalized")
            if not same_bounds and overlap_ratio < 0.1:
                blockers.append("Paired rasters do not share sufficient geographic overlap (< 10%)")
            elif not same_bounds:
                warnings.append(f"Paired rasters have partial geographic overlap ({overlap_ratio * 100:.1f}%)")
        elif has_any_crs:
            # A single georeferenced image cannot establish where the other
            # image belongs. Reject paired measurements instead of inventing
            # pixel alignment from matching dimensions.
            blockers.append("One image has a CRS and the other does not; cannot verify alignment")
            alignment_score = 0.0
            checks.update({
                "same_crs": False,
                "same_shape": same_shape,
                "same_bounds": False,
                "alignment_score": alignment_score,
            })
        else:
            # Neither has CRS — standard pixel-space image analysis (PNG / JPG / etc.)
            alignment_score = 1.0 if same_shape else 0.85
            checks.update({
                "same_crs": True,
                "same_shape": same_shape,
                "same_bounds": True,
                "alignment_score": alignment_score,
            })
            if not same_shape:
                warnings.append(f"Paired images have different dimensions ({a.width}×{a.height} vs {b.width}×{b.height}); auto-normalized")
            warnings.append(
                "Standard image analysis (no CRS) — coordinates in pixel-space; "
                "area measurements will be reported in pixels/percentages"
            )

        compatible = not blockers

    penalty = 0.20 * len(blockers) + 0.04 * len(warnings)
    score = max(0.0, min(1.0, 1.0 - penalty))
    return metadata, QualityReport(
        score=round(score, 3),
        compatible=compatible,
        blockers=blockers,
        warnings=warnings,
        checks=checks,
    )


def render_preview(path: Path, output_path: Path, max_size: int = 1024) -> Path:
    with rasterio.open(path) as src:
        ratio = min(1.0, max_size / max(src.width, src.height))
        out_w = max(1, round(src.width * ratio))
        out_h = max(1, round(src.height * ratio))
        indexes = [1, 2, 3] if src.count >= 3 else [1]
        data = src.read(indexes, out_shape=(len(indexes), out_h, out_w), resampling=Resampling.bilinear)
        if len(indexes) == 1:
            gray = (_normalize(data[0]) * 255).astype("uint8")
            rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            rgb = np.stack([(_normalize(data[i]) * 255).astype("uint8") for i in range(3)], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
    return output_path


def _read_gray(path: Path, max_size: int = 1024) -> tuple[np.ndarray, Affine, rasterio.crs.CRS | None]:
    with rasterio.open(path) as src:
        ratio = min(1.0, max_size / max(src.width, src.height))
        out_w = max(1, round(src.width * ratio))
        out_h = max(1, round(src.height * ratio))
        data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        return _normalize(data), transform, src.crs


def _denoise_binary(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    neighbour_count = uniform_filter(mask.astype("float32"), size=3, mode="constant", cval=0.0) * 9
    return mask & (neighbour_count >= 4)



def _area_square_meters(geom: Any, crs: rasterio.crs.CRS | None) -> float | None:
    polygon = shape(geom)
    if polygon.is_empty or crs is None:
        return None
    transformer = Transformer.from_crs(crs, "EPSG:6933", always_xy=True)
    projected = shapely_transform(transformer.transform, polygon)
    return abs(float(projected.area))


def deterministic_change_detection(before: Path, after: Path, output_dir: Path) -> dict[str, Any]:
    """Create a generic radiometric change mask.

    This fallback detects changed pixels only. It does not claim what semantic
    class changed or whether built-up/water/vegetation increased.
    """
    a, transform, crs = _read_gray(before)
    b, _, _ = _read_gray(after)
    if a.shape != b.shape:
        raise ValueError("Rasters must be aligned to the same sampled grid")

    delta = np.abs(a - b)
    median = float(np.median(delta))
    mad = float(np.median(np.abs(delta - median)))
    threshold = max(0.08, median + 3.0 * max(mad, 0.01))
    mask = _denoise_binary(delta > threshold)
    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    changed_percent = (changed_pixels / total_pixels * 100) if total_pixels else 0.0

    rgba = np.zeros((*mask.shape, 4), dtype="uint8")
    rgba[mask] = [239, 68, 68, 190]
    mask_path = output_dir / "change_mask.png"
    Image.fromarray(rgba, mode="RGBA").save(mask_path, format="PNG")

    features = []
    total_area_m2 = 0.0
    area_available = True
    for geom, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if int(value) != 1:
            continue
        area = _area_square_meters(geom, crs)
        if area is None:
            area_available = False
        else:
            total_area_m2 += area
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"kind": "generic_change", "area_m2": round(area, 3) if area is not None else None},
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"crs": crs.to_string() if crs else None, "semantic_class": "unknown"},
    }
    geojson_path = output_dir / "change_regions.geojson"
    geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    return {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_percent": round(changed_percent, 3),
        "threshold": round(threshold, 4),
        "region_count": len(features),
        "area_m2": round(total_area_m2, 3) if area_available else None,
        "semantic_supported": False,
        "mask_path": mask_path,
        "geojson_path": geojson_path,
    }
