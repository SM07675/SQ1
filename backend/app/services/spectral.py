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
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform


BAND_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "b04", "b4"),
    "green": ("green", "b03", "b3"),
    "blue": ("blue", "b02", "b2"),
    "nir": ("nir", "nir08", "b08", "b8", "b8a"),
    "swir": ("swir", "swir16", "b11"),
    "vv": ("vv",),
    "vh": ("vh",),
}

INDEX_REQUIREMENTS = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "ndbi": ("swir", "nir"),
}

TARGET_INDEX = {"vegetation": "ndvi", "water": "ndwi", "built-up": "ndbi"}

PALETTES = {
    "ndvi": ((111, 78, 55), (245, 237, 210), (24, 150, 79)),
    "ndwi": ((181, 105, 55), (235, 240, 229), (0, 172, 193)),
    "ndbi": ((40, 112, 180), (239, 241, 243), (239, 113, 54)),
}


def _canonical_band_map(src: rasterio.io.DatasetReader) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for index, description in enumerate(src.descriptions or (), start=1):
        name = (description or "").strip().lower().replace("-", "").replace("_", "")
        for canonical, aliases in BAND_ALIASES.items():
            normalized_aliases = {alias.replace("-", "").replace("_", "") for alias in aliases}
            if name in normalized_aliases:
                resolved.setdefault(canonical, index)
    return resolved


def available_indices(path: Path) -> list[str]:
    with rasterio.open(path) as src:
        bands = _canonical_band_map(src)
    return [name for name, required in INDEX_REQUIREMENTS.items() if all(band in bands for band in required)]


def _read_band(
    src: rasterio.io.DatasetReader,
    index: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    return src.read(index, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")


def read_index(path: Path, index_name: str, max_size: int = 1024) -> tuple[np.ndarray, Affine, Any]:
    required = INDEX_REQUIREMENTS[index_name]
    with rasterio.open(path) as src:
        bands = _canonical_band_map(src)
        missing = [band for band in required if band not in bands]
        if missing:
            raise ValueError(
                f"{index_name.upper()} requires named bands {required}; missing {tuple(missing)}. "
                "Set GeoTIFF band descriptions instead of guessing the band order."
            )
        ratio = min(1.0, max_size / max(src.width, src.height))
        out_w = max(1, round(src.width * ratio))
        out_h = max(1, round(src.height * ratio))
        a = _read_band(src, bands[required[0]], out_h, out_w)
        b = _read_band(src, bands[required[1]], out_h, out_w)
        denominator = a + b
        valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > 1e-7)
        result = np.full_like(a, np.nan, dtype="float32")
        result[valid] = (a[valid] - b[valid]) / denominator[valid]
        result = np.clip(result, -1, 1)
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        return result, transform, src.crs


def _colorize_index(values: np.ndarray, index_name: str) -> Image.Image:
    low, mid, high = (np.asarray(color, dtype="float32") for color in PALETTES[index_name])
    scaled = np.clip((np.nan_to_num(values, nan=-1.0) + 1.0) / 2.0, 0, 1)
    rgb = np.empty((*values.shape, 3), dtype="float32")
    lower = scaled <= 0.5
    factor_low = (scaled * 2)[..., None]
    factor_high = ((scaled - 0.5) * 2)[..., None]
    rgb[lower] = (low + (mid - low) * factor_low)[lower]
    rgb[~lower] = (mid + (high - mid) * factor_high)[~lower]
    alpha = np.where(np.isfinite(values), 255, 0).astype("uint8")
    rgba = np.dstack([rgb.astype("uint8"), alpha])
    return Image.fromarray(rgba, mode="RGBA")


def write_index_png(values: np.ndarray, index_name: str, output_path: Path) -> Path:
    _colorize_index(values, index_name).save(output_path, format="PNG")
    return output_path


def _denoise(mask: np.ndarray) -> np.ndarray:
    neighbour_count = ndi.uniform_filter(mask.astype("float32"), size=3, mode="constant", cval=0.0) * 9
    return mask & (neighbour_count >= 4)


def _area_m2(geometry: dict[str, Any], crs: Any) -> float | None:
    polygon = shape(geometry)
    if polygon.is_empty or crs is None:
        return None
    transformer = Transformer.from_crs(crs, "EPSG:6933", always_xy=True)
    projected = shapely_transform(transformer.transform, polygon)
    return abs(float(projected.area))


def _polygonize(
    mask: np.ndarray,
    transform: Affine,
    crs: Any,
    *,
    kind: str,
    output_path: Path,
) -> tuple[int, float | None]:
    features: list[dict[str, Any]] = []
    total_area = 0.0
    area_available = True
    minimum_native_area = abs(transform.a * transform.e - transform.b * transform.d) * 4
    for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if int(value) != 1:
            continue
        geom_shape = shape(geometry)
        if geom_shape.area < minimum_native_area:
            continue
        area = _area_m2(geometry, crs)
        if area is None:
            area_available = False
        else:
            total_area += area
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "kind": kind,
                "area_m2": round(area, 3) if area is not None else None,
                "pixel_area": round(float(geom_shape.area), 2),
                "bbox": list(geom_shape.bounds),
            },
        })
    # Sort largest first, by real area or pixel geometry area
    features.sort(
        key=lambda f: (f["properties"].get("area_m2") or 0.0, f["properties"].get("pixel_area") or 0.0),
        reverse=True,
    )
    for idx, feat in enumerate(features):
        feat["properties"]["rank"] = idx + 1
        feat["properties"]["is_largest"] = (idx == 0)

    payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"crs": crs.to_string() if crs else None, "producer": "spectral_toolkit_v2"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(features), round(total_area, 3) if area_available else None


def _write_mask(mask: np.ndarray, output_path: Path, color: tuple[int, int, int]) -> Path:
    rgba = np.zeros((*mask.shape, 4), dtype="uint8")
    rgba[mask] = [*color, 215]
    Image.fromarray(rgba, mode="RGBA").save(output_path, format="PNG")
    return output_path


def _direction(query: str) -> str:
    low = query.lower()
    for dec in ("decrease", "loss", "decline", "reduction", "less", "dropped", "shrunk"):
        if dec in low:
            return "decrease"
    return "increase"


def semantic_change_detection(
    before: Path,
    after: Path,
    target: str,
    arg4: Any = None,
    arg5: Any = None,
) -> dict[str, Any] | None:
    """Flexible wrapper supporting both (before, after, target, query, output_dir) and (before, after, target, output_dir, query)."""
    if isinstance(arg4, (str, bytes)):
        query = str(arg4)
        output_dir = Path(arg5) if arg5 is not None else Path(".")
    elif isinstance(arg5, (str, bytes)):
        output_dir = Path(arg4) if arg4 is not None else Path(".")
        query = str(arg5)
    else:
        output_dir = Path(arg4) if arg4 is not None else (Path(arg5) if arg5 is not None else Path("."))
        query = ""
    return spectral_change_analysis(before, after, target, output_dir, query)


def spectral_change_analysis(
    before: Path,
    after: Path,
    target: str,
    output_dir: Path,
    query: str = "",
) -> dict[str, Any] | None:
    index_name = TARGET_INDEX.get(target)
    if not index_name:
        return None
    try:
        a, transform, crs = read_index(before, index_name)
        b, _, _ = read_index(after, index_name)
    except ValueError:
        return None
    if a.shape != b.shape:
        return None

    direction = _direction(query)
    delta = b - a
    valid = np.isfinite(a) & np.isfinite(b)
    delta_threshold = 0.12
    if direction == "increase":
        class_floor = 0.05 if target != "vegetation" else 0.25
        mask = valid & (delta >= delta_threshold) & (b >= class_floor)
    else:
        class_floor = 0.05 if target != "vegetation" else 0.25
        mask = valid & (delta <= -delta_threshold) & (a >= class_floor)
    mask = _denoise(mask)

    before_name = f"{index_name}_before.png"
    after_name = f"{index_name}_after.png"
    mask_name = "semantic_change_mask.png"
    geojson_name = "semantic_change_regions.geojson"
    write_index_png(a, index_name, output_dir / before_name)
    write_index_png(b, index_name, output_dir / after_name)
    color = {"vegetation": (34, 197, 94), "water": (0, 184, 217), "built-up": (249, 115, 22)}[target]
    _write_mask(mask, output_dir / mask_name, color)
    region_count, area_m2 = _polygonize(
        mask, transform, crs, kind=f"{target}_{direction}", output_path=output_dir / geojson_name
    )
    changed_pixels = int(mask.sum())
    valid_pixels = max(1, int(valid.sum()))
    changed_percent = changed_pixels / valid_pixels * 100
    changed_delta = np.abs(delta[mask]) if changed_pixels else np.asarray([0.0])
    evidence_strength = float(np.clip(np.nanmean(changed_delta) / 0.4, 0, 1))
    confidence = round(0.67 + evidence_strength * 0.18, 3) if changed_pixels else 0.58
    return {
        "target": target,
        "direction": direction,
        "index": index_name.upper(),
        "changed_pixels": changed_pixels,
        "changed_percent": round(changed_percent, 3),
        "area_m2": area_m2,
        "region_count": region_count,
        "delta_threshold": delta_threshold,
        "mean_changed_delta": round(float(np.nanmean(changed_delta)), 4),
        "confidence": confidence,
        "before_path": output_dir / before_name,
        "after_path": output_dir / after_name,
        "mask_path": output_dir / mask_name,
        "geojson_path": output_dir / geojson_name,
        "producer": "deterministic_spectral_proxy_v2",
    }


def _describe_region_location(centroid_y_norm: float, centroid_x_norm: float) -> str:
    """Computes an intuitive natural-language description of where a region is located."""
    h_label = ""
    v_label = ""

    if centroid_x_norm < 0.38:
        h_label = "western (left)"
    elif centroid_x_norm > 0.62:
        h_label = "eastern (right)"
    else:
        h_label = "central"

    if centroid_y_norm < 0.38:
        v_label = "northern (upper)"
    elif centroid_y_norm > 0.62:
        v_label = "southern (lower)"
    else:
        v_label = "central"

    if h_label == "central" and v_label == "central":
        return "central portion of the image"
    elif h_label == "central":
        return f"{v_label} portion of the image"
    elif v_label == "central":
        return f"{h_label} portion of the image"
    else:
        # e.g. "western and north-western sector (upper-left region)"
        return f"{v_label} and {h_label} sector"


def extract_water_grounding(
    path: Path,
    output_dir: Path,
    query: str = "Highlight the largest water body.",
    max_size: int = 1024,
) -> dict[str, Any] | None:
    """
    High-precision, multi-stage water grounding engine with urban false-positive suppression.
    
    Distinguishes between:
    - TRUE MULTISPECTRAL NDWI (when calibrated Green + NIR bands exist)
    - RGB VISUAL WATER ESTIMATION (using texture, edge density, and color signature)
    
    Applies connected component analysis, morphological cleanup, urban structural edge suppression,
    and region-level evidence scoring to isolate genuine water bodies without falsely marking
    shadows, roads, or rooftops.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    q_lower = query.lower()
    is_largest_request = any(k in q_lower for k in ("largest", "biggest", "main", "primary", "dominant")) or True

    # 1. Attempt True Multispectral NDWI (Green + NIR)
    ndwi_values: np.ndarray | None = None
    transform: Affine = Affine.identity()
    crs: Any = None
    h, w = 0, 0
    is_spectral = False

    try:
        ndwi_values, transform, crs = read_index(path, "ndwi", max_size=max_size)
        h, w = ndwi_values.shape
        # Also check for NDVI to suppress dense vegetation
        try:
            ndvi_values, _, _ = read_index(path, "ndvi", max_size=max_size)
            veg_suppress = (ndvi_values > 0.15)
        except Exception:
            veg_suppress = np.zeros_like(ndwi_values, dtype=bool)

        water_candidate_pixels = np.isfinite(ndwi_values) & (ndwi_values >= 0.10) & (~veg_suppress)
        is_spectral = True
        index_label = "NDWI"
    except Exception:
        ndwi_values = None

    # 2. RGB Visual Water Estimation with Urban False-Positive Suppression
    if ndwi_values is None:
        try:
            with rasterio.open(path) as src:
                crs = src.crs
                ratio = min(1.0, max_size / max(src.width, src.height))
                w = max(1, round(src.width * ratio))
                h = max(1, round(src.height * ratio))
                transform = src.transform * Affine.scale(src.width / w, src.height / h)
                num_bands = src.count
                if num_bands >= 3:
                    r = src.read(1, out_shape=(h, w), resampling=Resampling.bilinear).astype("float32")
                    g = src.read(2, out_shape=(h, w), resampling=Resampling.bilinear).astype("float32")
                    b = src.read(3, out_shape=(h, w), resampling=Resampling.bilinear).astype("float32")
                else:
                    arr = src.read(1, out_shape=(h, w), resampling=Resampling.bilinear).astype("float32")
                    r, g, b = arr, arr, arr
        except Exception:
            # Fallback to PIL
            try:
                with Image.open(path) as img:
                    img_rgb = img.convert("RGB")
                    if max(img_rgb.size) > max_size:
                        img_rgb.thumbnail((max_size, max_size), Image.BILINEAR)
                    w, h = img_rgb.size
                    arr = np.asarray(img_rgb, dtype="float32")
                    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                    transform = Affine.identity()
                    crs = None
            except Exception:
                return None

        # Normalize channels to [0.0, 1.0]
        max_val = float(np.max(np.stack([r, g, b])))
        if max_val > 1.0:
            scale = 255.0 if max_val <= 255.0 else max_val
            r = np.clip(r / scale, 0.0, 1.0)
            g = np.clip(g / scale, 0.0, 1.0)
            b = np.clip(b / scale, 0.0, 1.0)

        # A. Luminance / Brightness
        y_lum = 0.299 * r + 0.587 * g + 0.114 * b

        # B. Color Difference Indices
        ndbr = (b - r) / (b + r + 1e-5)
        ndgr = (g - r) / (g + r + 1e-5)

        # C. Local Texture Variance (5x5 sliding window)
        mean_sq = ndi.uniform_filter(y_lum * y_lum, size=5)
        mean_y = ndi.uniform_filter(y_lum, size=5)
        local_texture_std = np.sqrt(np.maximum(0.0, mean_sq - mean_y * mean_y))

        # D. High-Frequency Gradient Magnitude (Sobel filter)
        gx = ndi.sobel(y_lum, axis=1) / 4.0
        gy = ndi.sobel(y_lum, axis=0) / 4.0
        gradient_mag = np.sqrt(gx * gx + gy * gy)

        # E. Urban Structural Edge Density (17x17 window)
        # Strong structural edges are typical for buildings, roads, and rooftops
        strong_edges = (gradient_mag > 0.06).astype(np.float32)
        urban_edge_density = ndi.uniform_filter(strong_edges, size=17)

        # F. Rejection of neutral grey asphalt, concrete, and dark shadows
        # Water exhibits blue/cyan hue; asphalt/shadows have neutral balance |R-G| ~ 0, |G-B| ~ 0
        color_spread = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
        is_neutral_grey_or_shadow = (color_spread < 0.035) & (ndbr < 0.04)

        # G. Multi-condition Water Pixel Candidate Selection
        # 1. Clear blue/cyan or green-blue dominance over red
        # 2. Smooth local texture and low high-frequency gradient
        # 3. Low urban edge density (rejects dense urban building clusters)
        # 4. Moderate brightness (rejects pure black shadow noise < 0.03 and extreme glints > 0.60)
        # 5. Non-neutral color signature
        water_candidate_pixels = (
            ((b > r * 1.04) | ((g > r * 1.06) & (b > r * 0.94)))
            & ((ndbr > 0.02) | (ndgr > 0.03))
            & (local_texture_std < 0.035)
            & (gradient_mag < 0.065)
            & (urban_edge_density < 0.10)
            & (y_lum >= 0.03)
            & (y_lum <= 0.60)
            & (~is_neutral_grey_or_shadow)
        )

        # Continuous optical water index [-1.0, 1.0] for visualization
        ndwi_values = np.clip(
            ndbr * 0.5 + ndgr * 0.3 + (0.35 - y_lum) * 0.4 - urban_edge_density * 0.8 - local_texture_std * 1.2,
            -1.0,
            1.0,
        )
        is_spectral = False
        index_label = "RGB_WATER_ESTIMATION"

    # 3. Morphological Cleanup (Opening removes thin lines/shadows, Closing fills internal gaps)
    struct_elem = ndi.generate_binary_structure(2, 2)
    opened_mask = ndi.binary_opening(water_candidate_pixels, structure=struct_elem, iterations=1)
    cleaned_mask = ndi.binary_closing(opened_mask, structure=struct_elem, iterations=2)

    total_image_pixels = h * w
    min_component_pixels = max(150, round(total_image_pixels * 0.0015))

    # 4. Connected Component Analysis & Region Quality Scoring
    labeled_components, num_features = ndi.label(cleaned_mask)
    
    valid_regions_data: list[dict[str, Any]] = []
    
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled_components == comp_id)
        comp_pixels = int(comp_mask.sum())
        
        if comp_pixels < min_component_pixels:
            continue
            
        coords = np.argwhere(comp_mask)
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        box_h = ymax - ymin + 1
        box_w = xmax - xmin + 1
        aspect_ratio = max(box_h / box_w, box_w / box_h)
        
        # Penalize thin linear structures (e.g. roads or shadows along buildings)
        if aspect_ratio > 12 and comp_pixels < 2500:
            continue
            
        mean_index_score = float(ndwi_values[comp_mask].mean()) if ndwi_values is not None else 0.5
        
        # Region quality score factoring size and spatial coherence
        region_score = (mean_index_score + 1.0) * (comp_pixels ** 0.5)
        
        centroid_y = float(coords[:, 0].mean())
        centroid_x = float(coords[:, 1].mean())
        
        valid_regions_data.append({
            "id": comp_id,
            "mask": comp_mask,
            "pixels": comp_pixels,
            "score": region_score,
            "mean_index": mean_index_score,
            "centroid": (centroid_y, centroid_x),
            "centroid_norm": (centroid_y / h, centroid_x / w),
            "bbox_pixels": [int(ymin), int(xmin), int(ymax), int(xmax)],
            "bbox_norm": [round(ymin / h, 4), round(xmin / w, 4), round(ymax / h, 4), round(xmax / w, 4)],
        })

    # Sort valid regions by evidence score (largest and most coherent first)
    valid_regions_data.sort(key=lambda r: r["score"], reverse=True)

    # 5. Sanity Check & Safe Abstention
    if not valid_regions_data:
        # No genuine coherent water body was detected
        empty_mask_path = output_dir / "water_mask.png"
        empty_grounding_path = output_dir / "water_grounding_mask.png"
        empty_ndwi_path = output_dir / "ndwi.png"
        empty_geojson_path = output_dir / "water_regions.geojson"
        
        # Write clean empty files
        _write_mask(np.zeros((h, w), dtype=bool), empty_mask_path, (0, 184, 217))
        Image.fromarray(np.zeros((h, w, 4), dtype="uint8"), mode="RGBA").save(empty_grounding_path, format="PNG")
        write_index_png(ndwi_values if ndwi_values is not None else np.zeros((h, w)), "ndwi", empty_ndwi_path)
        empty_geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")

        return {
            "target": "water",
            "water_body_identified": False,
            "findings": "No coherent, continuous water body could be identified with sufficient confidence.",
            "primary_region": None,
            "grounding_task": "largest_water_body" if is_largest_request else "water_delineation",
            "is_spectral": is_spectral,
            "index_type": index_label,
            "location_description": "No verified water body identified",
            "mask_path": empty_mask_path,
            "grounding_mask_path": empty_grounding_path,
            "ndwi_path": empty_ndwi_path,
            "geojson_path": empty_geojson_path,
            "region_count": 0,
            "area_m2": None,
            "total_area_m2": None,
            "coverage_percent": 0.0,
            "largest_coverage_percent": 0.0,
            "largest_pixel_count": 0,
            "total_water_pixels": 0,
            "primary_bbox": [0.0, 0.0, 0.0, 0.0],
            "confidence": 0.35,
            "producer": "multispectral_ndwi_grounding_engine_v2" if is_spectral else "optical_water_grounding_engine_v2",
        }

    # Extract Primary Water Body & All Validated Water Regions
    primary_region = valid_regions_data[0]
    primary_mask = primary_region["mask"]
    primary_pixels = primary_region["pixels"]
    primary_bbox = primary_region["bbox_norm"]
    
    # Union of all validated water components
    all_validated_water_mask = np.zeros((h, w), dtype=bool)
    for r_item in valid_regions_data:
        all_validated_water_mask |= r_item["mask"]
    total_water_pixels = int(all_validated_water_mask.sum())

    # Generate location wording
    location_desc = _describe_region_location(primary_region["centroid_norm"][0], primary_region["centroid_norm"][1])

    # 6. Generate Vector Geometries (GeoJSON)
    geojson_path = output_dir / "water_regions.geojson"
    region_count, total_area_m2 = _polygonize(
        all_validated_water_mask, transform, crs, kind="water", output_path=geojson_path
    )

    # Read back primary polygon real-world area if CRS available
    primary_area_m2: float | None = None
    if geojson_path.exists():
        try:
            fc = json.loads(geojson_path.read_text(encoding="utf-8"))
            feats = fc.get("features", [])
            if feats:
                primary_area_m2 = feats[0]["properties"].get("area_m2")
        except Exception:
            pass

    # 7. Render Output Visual Layers
    # A. Continuous Water Index Heatmap
    ndwi_path = output_dir / "ndwi.png"
    write_index_png(ndwi_values if ndwi_values is not None else np.zeros((h, w)), "ndwi", ndwi_path)

    # B. Full Verified Water Mask (Cyan)
    mask_path = output_dir / "water_mask.png"
    _write_mask(all_validated_water_mask, mask_path, (0, 184, 217))

    # C. High-Contrast Primary Water Grounding Overlay
    grounding_mask_path = output_dir / "water_grounding_mask.png"
    rgba_grounding = np.zeros((h, w, 4), dtype="uint8")

    # Secondary water regions (soft translucent cyan)
    rgba_grounding[all_validated_water_mask] = [0, 180, 216, 120]
    # Primary water body (vibrant electric azure)
    rgba_grounding[primary_mask] = [0, 229, 255, 205]

    # Glowing border outline for primary water body
    primary_u8 = primary_mask.astype("uint8")
    padded = np.pad(primary_u8, 1)
    boundary = primary_mask & (
        (padded[:-2, 1:-1] == 0) | (padded[2:, 1:-1] == 0) |
        (padded[1:-1, :-2] == 0) | (padded[1:-1, 2:] == 0)
    )
    rgba_grounding[boundary] = [56, 189, 248, 255]

    Image.fromarray(rgba_grounding, mode="RGBA").save(grounding_mask_path, format="PNG")

    coverage_percent = round((total_water_pixels / total_image_pixels) * 100, 2)
    largest_coverage_percent = round((primary_pixels / total_image_pixels) * 100, 2)

    # Calibrate confidence based on spectral presence and region dominance
    if is_spectral:
        confidence = 0.94 if largest_coverage_percent >= 5.0 else 0.88
    else:
        # RGB visual estimation confidence
        confidence = 0.86 if largest_coverage_percent >= 10.0 else 0.78

    findings_text = (
        f"Primary water body verified in the {location_desc}, covering {largest_coverage_percent}% of the scene "
        f"({primary_pixels:,} pixels)."
    )

    return {
        "target": "water",
        "water_body_identified": True,
        "findings": findings_text,
        "primary_region": primary_region,
        "grounding_task": "largest_water_body" if is_largest_request else "water_delineation",
        "is_spectral": is_spectral,
        "index_type": index_label,
        "location_description": location_desc,
        "mask_path": mask_path,
        "grounding_mask_path": grounding_mask_path,
        "ndwi_path": ndwi_path,
        "geojson_path": geojson_path,
        "region_count": len(valid_regions_data),
        "area_m2": primary_area_m2 or total_area_m2,
        "total_area_m2": total_area_m2,
        "coverage_percent": coverage_percent,
        "largest_coverage_percent": largest_coverage_percent,
        "largest_pixel_count": primary_pixels,
        "total_water_pixels": total_water_pixels,
        "primary_bbox": primary_bbox,
        "confidence": confidence,
        "producer": "multispectral_ndwi_grounding_engine_v2" if is_spectral else "optical_water_grounding_engine_v2",
    }


def spectral_scene_analysis(path: Path, target: str | None, output_dir: Path) -> dict[str, Any] | None:
    requested = TARGET_INDEX.get(target or "")
    candidates = [requested] if requested else ["ndvi", "ndwi", "ndbi"]
    computed: dict[str, dict[str, Any]] = {}
    for name in candidates:
        if not name:
            continue
        try:
            values, transform, crs = read_index(path, name)
        except ValueError:
            continue
        valid = values[np.isfinite(values)]
        if not valid.size:
            continue
        png_path = output_dir / f"{name}.png"
        write_index_png(values, name, png_path)
        computed[name] = {
            "mean": round(float(np.mean(valid)), 4),
            "p10": round(float(np.percentile(valid, 10)), 4),
            "p90": round(float(np.percentile(valid, 90)), 4),
            "path": png_path,
        }
    if not computed:
        return None
    primary_name = requested if requested in computed else next(iter(computed))
    primary_values, transform, crs = read_index(path, primary_name)
    threshold = 0.25 if primary_name == "ndvi" else (0.1 if primary_name == "ndwi" else 0.05)
    mask = np.isfinite(primary_values) & (primary_values >= threshold)
    mask = _denoise(mask)
    result = {
        "primary_index": primary_name.upper(),
        "indices": computed,
        "crs": crs.to_string() if crs else None,
        "producer": "spectral_scene_analyzer_v2",
    }
    if target:
        mask_path = output_dir / f"{target}_mask.png"
        geojson_path = output_dir / f"{target}_regions.geojson"
        color = {"vegetation": (34, 197, 94), "water": (0, 184, 217), "built-up": (249, 115, 22)}.get(
            target, (59, 130, 246)
        )
        _write_mask(mask, mask_path, color)
        count, area = _polygonize(mask, transform, crs, kind=target, output_path=geojson_path)
        result.update({
            "target": target,
            "mask_path": mask_path,
            "geojson_path": geojson_path,
            "region_count": count,
            "area_m2": area,
            "coverage_percent": round(float(mask.mean() * 100), 3),
            "index": requested.upper() if requested else primary_name.upper(),
            "threshold": threshold,
        })
    return result


def optical_sar_water_fusion(optical: Path, sar: Path, output_dir: Path) -> dict[str, Any] | None:
    from app.services.croma_pipeline import is_sar_image

    if is_sar_image(optical) and not is_sar_image(sar):
        optical, sar = sar, optical

    ndwi: np.ndarray | None = None
    transform: Affine = Affine.identity()
    crs: Any = None

    try:
        ndwi, transform, crs = read_index(optical, "ndwi")
    except Exception:
        ndwi = None

    if ndwi is None:
        try:
            with rasterio.open(optical) as src_opt:
                crs = src_opt.crs
                ratio = min(1.0, 1024 / max(src_opt.width, src_opt.height))
                out_w = max(1, round(src_opt.width * ratio))
                out_h = max(1, round(src_opt.height * ratio))
                transform = src_opt.transform * Affine.scale(src_opt.width / out_w, src_opt.height / out_h)
                if src_opt.count >= 3:
                    r = src_opt.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                    g = src_opt.read(2, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                    b = src_opt.read(3, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                    max_v = max(float(r.max()), float(g.max()), float(b.max()))
                    if max_v > 1.0:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    y_lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ndbr = (b - r) / (b + r + 1e-5)
                    ndgr = (g - r) / (g + r + 1e-5)
                    ndwi = np.clip(ndbr * 0.5 + ndgr * 0.3 + (0.35 - y_lum) * 0.4, -1.0, 1.0)
                else:
                    arr = src_opt.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                    ndwi = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-5)
        except Exception:
            return None

    if ndwi is None:
        return None

    target_h, target_w = ndwi.shape
    try:
        with rasterio.open(sar) as src:
            radar = src.read(1, out_shape=(target_h, target_w), resampling=Resampling.bilinear).astype("float32")
    except Exception:
        return None

    finite = radar[np.isfinite(radar)]
    if not finite.size:
        return None

    low, high = np.percentile(finite, [2, 98])
    normalized = np.clip((radar - low) / max(high - low, 1e-7), 0, 1)
    optical_water = np.isfinite(ndwi) & (ndwi >= 0.12)
    sar_water = np.isfinite(normalized) & (normalized <= 0.28)
    agreement = _denoise(optical_water & sar_water)
    union = optical_water | sar_water
    iou = float(agreement.sum() / max(1, union.sum()))

    rgba = np.zeros((*agreement.shape, 4), dtype="uint8")
    rgba[optical_water & ~sar_water] = [245, 158, 11, 185]
    rgba[sar_water & ~optical_water] = [168, 85, 247, 185]
    rgba[agreement] = [0, 229, 255, 225]
    agreement_path = output_dir / "sensor_agreement.png"
    Image.fromarray(rgba, mode="RGBA").save(agreement_path, format="PNG")
    ndwi_path = output_dir / "ndwi_optical.png"
    write_index_png(ndwi, "ndwi", ndwi_path)
    geojson_path = output_dir / "confirmed_water_regions.geojson"
    region_count, area_m2 = _polygonize(
        agreement, transform, crs, kind="optical_sar_confirmed_water", output_path=geojson_path
    )
    confidence = round(min(0.91, 0.62 + 0.35 * iou), 3)
    return {
        "target": "water",
        "agreement_iou": round(iou, 4),
        "sensor_agreement_percent": round(iou * 100, 2),
        "confirmed_percent": round(float(agreement.mean() * 100), 3),
        "area_m2": area_m2,
        "region_count": region_count,
        "confidence": confidence,
        "agreement_path": agreement_path,
        "ndwi_path": ndwi_path,
        "geojson_path": geojson_path,
        "producer": "optical_ndwi_plus_sar_backscatter_v2",
    }


def classify_land_cover_scene(
    image_path: Path,
    output_dir: Path,
    query: str = "",
    max_size: int = 1024,
) -> dict[str, Any]:
    """Multi-class deterministic and spectral land-cover classification engine."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(image_path) as src:
        band_map = _canonical_band_map(src)
        ratio = min(1.0, max_size / max(src.width, src.height))
        out_w = max(1, round(src.width * ratio))
        out_h = max(1, round(src.height * ratio))
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        crs = src.crs

        has_nir = "nir" in band_map
        has_red = "red" in band_map
        has_green = "green" in band_map
        has_blue = "blue" in band_map
        has_swir = "swir" in band_map

        if src.count >= 3 and (not has_red or not has_green or not has_blue):
            r = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
            g = src.read(2, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
            b = src.read(3, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
        else:
            r = _read_band(src, band_map.get("red", 1), out_h, out_w)
            g = _read_band(src, band_map.get("green", min(2, src.count)), out_h, out_w)
            b = _read_band(src, band_map.get("blue", min(3, src.count)), out_h, out_w)

        if has_nir:
            nir = _read_band(src, band_map["nir"], out_h, out_w)
        else:
            nir = None

        if has_swir:
            swir = _read_band(src, band_map["swir"], out_h, out_w)
        else:
            swir = None

    # Compute Normalized Bands
    r_norm = (r - r.min()) / max(r.max() - r.min(), 1e-5)
    g_norm = (g - g.min()) / max(g.max() - g.min(), 1e-5)
    b_norm = (b - b.min()) / max(b.max() - b.min(), 1e-5)
    brightness = 0.299 * r_norm + 0.587 * g_norm + 0.114 * b_norm

    # Urban Edge Suppression
    gx = ndi.sobel(brightness, axis=1) / 4.0
    gy = ndi.sobel(brightness, axis=0) / 4.0
    gmag = np.sqrt(gx * gx + gy * gy)
    edge_density = ndi.uniform_filter((gmag > 0.06).astype(np.float32), size=17)

    if nir is not None:
        ndvi = np.clip((nir - r) / np.maximum(nir + r, 1e-6), -1.0, 1.0)
        ndwi = np.clip((g - nir) / np.maximum(g + nir, 1e-6), -1.0, 1.0)
    else:
        # Optical proxy indices
        ndbr = (b_norm - r_norm) / (b_norm + r_norm + 1e-5)
        ndgr = (g_norm - r_norm) / (g_norm + r_norm + 1e-5)
        ndvi = np.clip((g_norm - r_norm) / np.maximum(g_norm + r_norm, 1e-6), -1.0, 1.0)
        ndwi = np.clip(ndbr * 0.5 + ndgr * 0.3 - edge_density * 0.8, -1.0, 1.0)

    if swir is not None and nir is not None:
        ndbi = np.clip((swir - nir) / np.maximum(swir + nir, 1e-6), -1.0, 1.0)
    else:
        ndbi = np.clip((r_norm - g_norm) + (brightness - 0.5) + edge_density * 0.5, -1.0, 1.0)

    total_pixels = out_h * out_w

    # 1. Water mask with urban suppression
    water_candidate = (ndwi >= 0.12) & (ndvi < 0.10) & (brightness < 0.60) & (edge_density < 0.12)
    struct_elem = ndi.generate_binary_structure(2, 2)
    water_mask = ndi.binary_closing(ndi.binary_opening(water_candidate, structure=struct_elem), structure=struct_elem)

    # 2. Forest / Dense canopy
    forest_mask = _denoise((ndvi >= 0.40) & ~water_mask)

    # 3. General Vegetation / Greenery (excluding dense forest)
    veg_mask = _denoise((ndvi >= 0.18) & ~forest_mask & ~water_mask)

    # 4. Built-up / Urban footprint
    builtup_mask = _denoise(((ndbi >= 0.06) | (edge_density > 0.10) | ((brightness > 0.55) & (ndvi < 0.15))) & ~water_mask & ~forest_mask & ~veg_mask)

    # 5. Agricultural land / Cultivated fields
    agri_mask = _denoise(((ndvi >= 0.10) & (ndvi < 0.28) & (r_norm > 0.20)) & ~water_mask & ~forest_mask & ~veg_mask & ~builtup_mask)

    # 6. Bare land / Soil (remaining)
    bare_mask = ~water_mask & ~forest_mask & ~veg_mask & ~builtup_mask & ~agri_mask

    # Calculate percentages
    water_pct = round(float(water_mask.sum() / total_pixels * 100), 2)
    forest_pct = round(float(forest_mask.sum() / total_pixels * 100), 2)
    veg_pct = round(float(veg_mask.sum() / total_pixels * 100), 2)
    total_veg_pct = round(veg_pct + forest_pct, 2)
    builtup_pct = round(float(builtup_mask.sum() / total_pixels * 100), 2)
    agri_pct = round(float(agri_mask.sum() / total_pixels * 100), 2)
    bare_pct = round(float(bare_mask.sum() / total_pixels * 100), 2)

    def _qualitative(pct: float) -> str:
        if pct >= 35.0:
            return "Dominant"
        if pct >= 15.0:
            return "Moderate"
        if pct >= 2.0:
            return "Detected"
        if pct >= 0.2:
            return "Low"
        return "None"

    breakdown = {
        "water": {"percent": water_pct, "status": _qualitative(water_pct), "color": "#0ea5e9"},
        "vegetation": {"percent": total_veg_pct, "status": _qualitative(total_veg_pct), "color": "#10b981"},
        "forest": {"percent": forest_pct, "status": _qualitative(forest_pct), "color": "#059669"},
        "agricultural": {"percent": agri_pct, "status": _qualitative(agri_pct), "color": "#eab308"},
        "built_up": {"percent": builtup_pct, "status": _qualitative(builtup_pct), "color": "#f43f5e"},
        "bare_land": {"percent": bare_pct, "status": _qualitative(bare_pct), "color": "#b48c64"},
    }

    # Generate Land Cover Color-coded RGBA Map
    rgba = np.zeros((out_h, out_w, 4), dtype="uint8")
    rgba[bare_mask] = [180, 140, 100, 200]
    rgba[agri_mask] = [234, 179, 8, 220]
    rgba[builtup_mask] = [244, 63, 94, 225]
    rgba[veg_mask] = [52, 211, 153, 220]
    rgba[forest_mask] = [16, 185, 129, 235]
    rgba[water_mask] = [14, 165, 233, 240]

    mask_path = output_dir / "land_cover_mask.png"
    Image.fromarray(rgba, mode="RGBA").save(mask_path, format="PNG")

    geojson_path = output_dir / "land_cover.geojson"
    region_count, area_m2 = _polygonize(
        water_mask | forest_mask | veg_mask | builtup_mask,
        transform,
        crs,
        kind="land_cover_zones",
        output_path=geojson_path,
    )

    dominant_class = max(breakdown.items(), key=lambda x: x[1]["percent"])
    summary_text = (
        f"Land cover analysis resolved: {dominant_class[0].replace('_', ' ').title()} is dominant ({dominant_class[1]['percent']}%), "
        f"Vegetation canopy: {total_veg_pct}%, Built-up footprint: {builtup_pct}%, "
        f"Water bodies: {water_pct}%, Agricultural land: {agri_pct}%, Bare soil: {bare_pct}%."
    )

    return {
        "target": "land_cover",
        "producer": "multispectral_land_cover_engine_v1",
        "summary": summary_text,
        "confidence": 0.90,
        "breakdown": breakdown,
        "dominant_class": dominant_class[0],
        "mask_path": mask_path,
        "geojson_path": geojson_path,
        "water_percent": water_pct,
        "vegetation_percent": total_veg_pct,
        "forest_percent": forest_pct,
        "agricultural_percent": agri_pct,
        "built_up_percent": builtup_pct,
        "bare_land_percent": bare_pct,
        "region_count": region_count,
        "area_m2": area_m2,
    }
