from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from satquery_engine.schemas import QualityReport, RasterMetadata
from satquery_engine.services.spectral import available_indices


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
    import hashlib
    from scipy.ndimage import laplace
    from satquery_engine.services.bands import detect_band_map
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream,"sha256").hexdigest()
    with rasterio.open(path) as src:
        sample_h = min(src.height, 512)
        sample_w = min(src.width, 512)
        sample = src.read(out_shape=(src.count,sample_h,sample_w),masked=True,resampling=Resampling.nearest).astype("float32").filled(np.nan)
        valid = np.all(np.isfinite(sample),axis=0)
        nodata_percent = float((~valid).mean()*100)
        band_map = detect_band_map(src)
        gray = np.mean(sample,axis=0)
        finite = gray[valid]
        span = float(np.ptp(finite)) if finite.size else 0.0
        normalized = np.where(valid,(gray-(float(finite.min()) if finite.size else 0))/max(span,1e-9),0)
        quality_metrics = {"sample_shape":[sample_h,sample_w],"valid_fraction":float(valid.mean()),
            "dynamic_range":span,"laplacian_variance":float(laplace(normalized).var()),
            "cloud_mask_available":False,"quality_method":"valid_fraction_and_dynamic_range_v1"}
        descriptions = list(src.descriptions or ())
        band_names = [descriptions[i] or f"band_{i + 1}" for i in range(src.count)]
        tags = {k.lower(): v for k, v in src.tags().items()}
        sensor = tags.get("sensor") or tags.get("satellite") or tags.get("platform")
        polarizations = [n.upper() for n in band_names if n.lower() in {"vv", "vh", "hh", "hv"}]
        tag_pol = tags.get("polarization", tags.get("polarisation", ""))
        polarizations += re.findall(r"\b(?:VV|VH|HH|HV)\b", tag_pol.upper())
        radar_tag = tags.get("modality", "").lower() in {"sar", "radar"} or bool(sensor and re.search(r"sentinel.?1|radarsat|terrasar", sensor, re.I))
        roles = set(band_map.indices)
        has_spectral_names = bool(roles & {"nir", "swir", "swir2"})
        is_rgb = {"red", "green", "blue"} <= roles
        ordinary_rgb = src.count == 3 and not any(src.descriptions) and not band_map.warnings
        modality = "sar" if polarizations or radar_tag else "multispectral" if has_spectral_names else "optical" if is_rgb or ordinary_rgb else "unknown"
        date = next((tags[k] for k in ("acquisition_date", "datetime", "sensing_time", "date_acquired", "acquisition_datetime") if tags.get(k)), None)
        def optional_float(*names):
            for name in names:
                if tags.get(name) is not None:
                    try: return float(tags[name])
                    except (TypeError, ValueError): return None
            return None
        sun_azimuth = optional_float("sun_azimuth", "solar_azimuth", "sunazimuth")
        sun_elevation = optional_float("sun_elevation", "solar_elevation", "sunelevation")
        cloud_cover = optional_float("cloud_cover", "cloudcover", "cloud_coverage")
        return RasterMetadata(
            file_hash=digest, band_map=band_map.to_dict(), wavelengths=band_map.wavelengths_nm,
            quality_metrics=quality_metrics,quality_score=float(valid.mean())*(1.0 if span>0 else 0.25),
            filename=path.name,
            width=src.width,
            height=src.height,
            bands=src.count,
            dtype=str(src.dtypes[0]),
            crs=src.crs.to_string() if src.crs else None,
            bounds=[float(v) for v in src.bounds],
            resolution=[abs(float(src.res[0])), abs(float(src.res[1]))],
            nodata=float(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else None,
            nodata_percent=round(nodata_percent, 3),
            band_names=band_names,
            available_indices=available_indices(path),
            transform=list(src.transform)[:6], modality=modality, acquisition_date=date,
            sensor=sensor, polarization=sorted(set(polarizations)), tags=tags,
            sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
            cloud_information={"cover_percent": cloud_cover} if cloud_cover is not None else {},
            wgs84_bounds=list(transform_bounds(src.crs,"EPSG:4326",*src.bounds,densify_pts=21)) if src.crs else None,
        )


def validate_inputs(paths: list[Path]) -> tuple[list[RasterMetadata], QualityReport]:
    metadata = [inspect_raster(path) for path in paths]
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {"file_count": len(paths)}

    has_any_crs = any(item.crs is not None for item in metadata)
    all_have_crs = all(item.crs is not None for item in metadata)
    checks["geospatial"] = all_have_crs

    for item in metadata:
        warnings.extend(f"{item.filename}: {w}" for w in item.band_map.get("warnings",[]))
        if item.crs is None:
            warnings.append(
                f"{item.filename}: missing CRS — pixel-space analysis only; "
                "area/distance values will not be available"
            )
        if item.nodata_percent > 40:
            warnings.append(f"{item.filename}: {item.nodata_percent:.1f}% NoData")
        if item.nodata_percent >= 99.9:
            blockers.append(f"{item.filename}: no usable image pixels")
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
            b_bounds = list(transform_bounds(b.crs, a.crs, *b.bounds, densify_pts=21))
            same_bounds = bool(np.allclose(a.bounds, b_bounds, rtol=0, atol=max(a.resolution)))
            resolution_ratio = max(a.resolution[0], b.resolution[0]) / max(min(a.resolution[0], b.resolution[0]), 1e-9)
            intersection_width = max(0.0, min(a.bounds[2], b_bounds[2]) - max(a.bounds[0], b_bounds[0]))
            intersection_height = max(0.0, min(a.bounds[3], b_bounds[3]) - max(a.bounds[1], b_bounds[1]))
            intersection_area = intersection_width * intersection_height
            a_area = max(1e-9, (a.bounds[2] - a.bounds[0]) * (a.bounds[3] - a.bounds[1]))
            b_area = max(1e-9, (b_bounds[2] - b_bounds[0]) * (b_bounds[3] - b_bounds[1]))
            overlap_ratio = intersection_area / max(a_area, b_area)
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
            if overlap_ratio < 0.8:
                blockers.append("These two images do not overlap enough for a reliable comparison (at least 80% of both footprints is required).")
            elif not same_bounds:
                warnings.append(f"Paired rasters have partial geographic overlap ({overlap_ratio * 100:.1f}%)")
        elif has_any_crs:
            # Mixed: one has CRS, one doesn't — block because alignment is ambiguous
            blockers.append("One image has a CRS and the other does not — cannot verify alignment")
            checks.update({"same_crs": False, "same_shape": same_shape, "alignment_score": 0.0})
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
    score = max(0.0, min(1.0, 1.0 - penalty)) * min((a.quality_score if a.quality_score is not None else 1 for a in metadata),default=0)
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
        from satquery_engine.services.radiometry import rgb_indexes
        try:
            indexes = rgb_indexes(src)
        except ValueError:
            indexes = [1]
        data = src.read(indexes, out_shape=(len(indexes), out_h, out_w), resampling=Resampling.bilinear,masked=True).astype("float32").filled(np.nan)
        if len(indexes) == 1:
            gray = (np.nan_to_num(_normalize(data[0])) * 255).astype("uint8")
            rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            rgb = np.stack([(np.nan_to_num(_normalize(data[i])) * 255).astype("uint8") for i in range(3)], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
    return output_path


def _read_gray(path: Path, max_size: int = 1024) -> tuple[np.ndarray, Affine, rasterio.crs.CRS | None]:
    with rasterio.open(path) as src:
        ratio = min(1.0, max_size / max(src.width, src.height))
        out_w = max(1, round(src.width * ratio))
        out_h = max(1, round(src.height * ratio))
        data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear,masked=True).astype("float32").filled(np.nan)
        transform = src.transform @ Affine.scale(src.width / out_w, src.height / out_h)
        return _normalize(data), transform, src.crs


def _denoise_binary(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    neighbour_count = uniform_filter(mask.astype("float32"), size=3, mode="constant", cval=0.0) * 9
    return mask & (neighbour_count >= 4)



def _area_square_meters(geom: Any, crs: rasterio.crs.CRS | None) -> float | None:
    polygon = shape(geom)
    if polygon.is_empty or crs is None:
        return None
    from pyproj import CRS, Geod
    if CRS(crs).is_geographic:
        from shapely.geometry.polygon import orient
        geographic = shapely_transform(Transformer.from_crs(crs,4326,always_xy=True).transform, polygon)
        parts = list(geographic.geoms) if geographic.geom_type == "MultiPolygon" else [geographic]
        # Geodesic area handles longitude wrapping and respects interior holes.
        return sum(abs(Geod(ellps="WGS84").geometry_area_perimeter(orient(part,sign=1))[0]) for part in parts)
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

    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        raise ValueError("No shared valid pixels are available for change measurement")

    std_a = float(np.std(a[valid]))
    std_b = float(np.std(b[valid]))
    if std_a > 0.02 and std_b > 0.02:
        ratio = std_a / std_b
        if 0.25 < ratio < 4.0:
            mean_a = float(np.mean(a[valid]))
            mean_b = float(np.mean(b[valid]))
            b_norm = np.clip((b - mean_b) * ratio + mean_a, 0.0, 1.0)
        else:
            b_norm = b
    else:
        b_norm = b

    delta = np.abs(a - b_norm)
    median = float(np.median(delta[valid]))
    mad = float(np.median(np.abs(delta[valid] - median)))
    threshold = max(0.08, median + 3.0 * max(mad, 0.01))
    mask = _denoise_binary(valid & (delta > threshold))
    changed_pixels = int(mask.sum())
    total_pixels = int(valid.sum())
    changed_percent = (changed_pixels / total_pixels * 100) if total_pixels else 0.0

    from scipy import ndimage
    from satquery_engine.services.spatial_outputs import export_labels
    spatial=export_labels(ndimage.label(mask)[0],before,output_dir,"change",transform=transform,color=(239,68,68))
    mask_path=output_dir/"change_mask.png"
    geojson_path=output_dir/"change.geojson"

    return {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_percent": round(changed_percent, 3),
        "threshold": round(threshold, 4),
        "region_count": spatial["region_count"],
        "area_m2": spatial["area_m2"],
        "semantic_supported": False,
        "mask_path": mask_path,
        "geojson_path": geojson_path,
        "paths": spatial["paths"],
    }
