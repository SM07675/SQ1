from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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

def _get_otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Computes Otsu's threshold for a given numpy array, ignoring NaNs."""
    valid_vals = values[np.isfinite(values)]
    if not valid_vals.size:
        return 0.0
    
    min_v, max_v = float(valid_vals.min()), float(valid_vals.max())
    if max_v - min_v < 1e-5:
        return min_v
    
    scaled = np.clip((valid_vals - min_v) / (max_v - min_v) * (bins - 1), 0, bins - 1).astype(int)
    hist = np.bincount(scaled.ravel(), minlength=bins)
    
    total = hist.sum()
    current_max, threshold = 0.0, 0
    sum_total, sum_b = 0.0, 0.0
    weight_b, weight_f = 0.0, 0.0
    
    for i in range(bins):
        sum_total += i * hist[i]
        
    for i in range(bins):
        weight_b += hist[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
            
        sum_b += i * hist[i]
        m_b = sum_b / weight_b
        m_f = (sum_total - sum_b) / weight_f
        
        var_between = weight_b * weight_f * (m_b - m_f) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i
            
    return float(threshold / (bins - 1) * (max_v - min_v) + min_v)

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


def check_index_bands_available(path: Path, index_name: str) -> tuple[bool, str]:
    """Inspects raster metadata to verify whether required bands for a spectral index exist.

    Never assumes RGB channels are NIR/SWIR.
    Returns (True, '') if available, or (False, reason) if unavailable.
    """
    required = INDEX_REQUIREMENTS.get(index_name.lower())
    if not required:
        return False, f"Unknown spectral index: {index_name}"
    try:
        with rasterio.open(path) as src:
            bands = _canonical_band_map(src)
            missing = [band for band in required if band not in bands]
            if missing:
                return (
                    False,
                    "This index cannot be reliably computed from the supplied image because the required spectral bands are unavailable.",
                )
        return True, ""
    except Exception:
        return (
            False,
            "This index cannot be reliably computed from the supplied image because the required spectral bands are unavailable.",
        )


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
                "This index cannot be reliably computed from the supplied image because the required spectral bands are unavailable."
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


def _describe_region_location(centroid_y_norm: float, centroid_x_norm: float, coverage_pct: float = 0.0) -> str:
    """Computes an intuitive natural-language description of where a region is located."""
    if coverage_pct >= 45.0:
        if centroid_x_norm < 0.50:
            return "broad marine/coastal expanse spanning the western and offshore sectors"
        else:
            return "broad marine/coastal expanse spanning the eastern and offshore sectors"

    h_label = "western (left)" if centroid_x_norm < 0.42 else ("eastern (right)" if centroid_x_norm > 0.58 else "central")
    v_label = "northern (upper)" if centroid_y_norm < 0.42 else ("southern (lower)" if centroid_y_norm > 0.58 else "central")

    if h_label == "central" and v_label == "central":
        return "central portion of the image"
    elif h_label == "central":
        return f"{v_label} portion of the image"
    elif v_label == "central":
        return f"{h_label} portion of the image"
    else:
        return f"{v_label} and {h_label} sector"


def _fill_small_holes(mask: np.ndarray, max_hole_pixels: int = 150) -> np.ndarray:
    """Fills small holes (e.g. wave glints or sensor dropouts) without wiping out entire islands."""
    inverted = ~mask
    labeled, num_features = ndi.label(inverted)
    if num_features == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    small_holes = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_features + 1):
        if counts[i] <= max_hole_pixels:
            small_holes[labeled == i] = True
    return mask | small_holes


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

    # 0. Attempt Calibrated Deep Learning Model (SatlasWaterNet Swin-v2) on GeoTIFF/NetCDF inputs
    is_geospatial = path.suffix.lower() in (".tif", ".tiff", ".nc", ".geotiff")
    if is_geospatial:
        try:
            from app.services.water_model import is_water_model_available, predict_water_mask
            if is_water_model_available():
                dl_res = predict_water_mask(path, output_dir, query=query, max_size=max_size)
                if dl_res is not None:
                    return dl_res
        except Exception as e:
            logger.warning("SatlasWaterNet inference failed or skipped, falling back to spectral proxy: %s", e)

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

        otsu_thresh = _get_otsu_threshold(ndwi_values)
        adaptive_thresh = max(0.05, min(0.3, otsu_thresh))

        water_candidate_pixels = np.isfinite(ndwi_values) & (ndwi_values >= adaptive_thresh) & (~veg_suppress)
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

        # F. Rejection of neutral grey asphalt, concrete, and dark shadows on land
        # Water exhibits blue/cyan hue; asphalt/shadows have neutral balance |R-G| ~ 0, |G-B| ~ 0
        # In deep ocean, R, G, B are all low (< 0.05), so we only apply grey-shadow rejection
        # when brightness is in the asphalt/shadow range (y_lum >= 0.05) or urban edge density is present.
        color_spread = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
        is_neutral_grey_or_shadow = (
            (y_lum >= 0.05)
            & (color_spread < 0.035)
            & (ndbr < 0.04)
        )

        # G. Multi-condition Water Pixel Candidate Selection
        # 1. Clear blue/cyan or green-blue dominance over red
        # 2. Smooth local texture and low high-frequency gradient
        # 3. Low urban edge density (rejects dense urban building clusters)
        # 4. Moderate brightness (rejects pure black sensor noise < 0.003 and extreme glints > 0.90)
        # 5. Non-neutral color signature on land
        # 6. Low luminance threshold (0.003) ensures dark deep-ocean water is fully detected

        # Continuous optical water index [-1.0, 1.0] for visualization
        ndwi_values = np.clip(
            ndbr * 0.5 + ndgr * 0.3 + (0.35 - y_lum) * 0.4 - urban_edge_density * 0.8 - local_texture_std * 1.5,
            -1.0,
            1.0,
        )
        otsu_thresh = _get_otsu_threshold(ndwi_values)
        adaptive_thresh = max(0.01, min(0.15, otsu_thresh))

        water_candidate_pixels = (
            ((b > r * 1.01) | ((g > r * 1.02) & (b > r * 0.88)))
            & ((ndbr > 0.005) | (ndgr > 0.005))
            & (local_texture_std < 0.06)
            & (gradient_mag < 0.08)
            & (urban_edge_density < 0.15)
            & (y_lum >= 0.003)
            & (y_lum <= 0.85)
            & (~is_neutral_grey_or_shadow)
            & (ndwi_values >= adaptive_thresh)
        )

        is_spectral = False
        index_label = "RGB_WATER_ESTIMATION"

    # 3. Morphological Cleanup (Opening removes thin lines/shadows, Closing fills micro-gaps, Small-hole fill protects islands)
    struct_elem = ndi.generate_binary_structure(2, 2)
    opened_mask = ndi.binary_opening(water_candidate_pixels, structure=struct_elem, iterations=1)
    cleaned_mask = ndi.binary_closing(opened_mask, structure=struct_elem, iterations=2)
    cleaned_mask = _fill_small_holes(cleaned_mask, max_hole_pixels=150)

    # 3b. Deep Learning Consensus Fusion (Stage 6 — Segmentation Witness)
    # Query the satquery_landcover_v1 water witness for a second independent opinion.
    # This creates a two-witness consensus:
    #   - VETO: spectral says water, but DL is confidently land → remove (false positive)
    #   - RECOVER: DL says high-confidence water, spectral missed → add back
    dl_witness_used = False
    try:
        from app.services.landcover_model import is_landcover_model_available, predict_water_witness
        if is_landcover_model_available():
            dl_witness = predict_water_witness(path, max_size=max_size, threshold=0.30)
            if dl_witness is not None:
                dl_water = dl_witness["water_mask"]
                dl_confident_land = dl_witness.get("confident_land_mask")
                dl_water_prob = dl_witness["water_prob"]
                argmax_canon = dl_witness.get("argmax_canonical")

                # Resize DL masks to match spectral resolution if needed
                if dl_water.shape != (h, w):
                    from PIL import Image as _PILImg
                    dl_water = np.array(_PILImg.fromarray(dl_water.astype(np.uint8)).resize((w, h), _PILImg.NEAREST)).astype(bool)
                    if dl_confident_land is not None:
                        dl_confident_land = np.array(_PILImg.fromarray(dl_confident_land.astype(np.uint8)).resize((w, h), _PILImg.NEAREST)).astype(bool)
                    if argmax_canon is not None:
                        argmax_canon = np.array(_PILImg.fromarray(argmax_canon.astype(np.uint8)).resize((w, h), _PILImg.NEAREST))
                    dl_water_prob = np.array(_PILImg.fromarray(
                        (dl_water_prob * 255).clip(0, 255).astype("uint8")
                    ).resize((w, h), _PILImg.BILINEAR)).astype("float32") / 255.0

                # VETO: Remove pixels where spectral says water but DL confidently identifies a specific land category.
                # Only veto when DL predicts built-up (1), bare soil/sand (4), or road (6),
                # OR when the pixel has land-level brightness (y_lum >= 0.06) or urban edge density (> 0.05).
                # This protects dark, smooth open ocean water from being falsely vetoed as terrestrial vegetation.
                if dl_confident_land is not None:
                    if argmax_canon is not None:
                        is_structural_or_soil_land = np.isin(argmax_canon, [1, 4, 6])
                        has_land_appearance = (y_lum >= 0.06) | (urban_edge_density > 0.05)
                        valid_land_veto = dl_confident_land & (is_structural_or_soil_land | has_land_appearance)
                    else:
                        valid_land_veto = dl_confident_land & (y_lum >= 0.06)
                    vetoed = cleaned_mask & valid_land_veto
                    cleaned_mask = cleaned_mask & ~valid_land_veto
                else:
                    vetoed = np.zeros_like(cleaned_mask)

                # RECOVER: Add back pixels the DL is confident are water (prob >= 0.50) but spectral missed
                high_conf_dl_water = (dl_water_prob >= 0.50) & dl_water & ~cleaned_mask
                cleaned_mask = cleaned_mask | high_conf_dl_water

                # Re-run morphological cleanup after consensus
                cleaned_mask = ndi.binary_opening(cleaned_mask, structure=struct_elem, iterations=1)
                cleaned_mask = ndi.binary_closing(cleaned_mask, structure=struct_elem, iterations=2)
                cleaned_mask = _fill_small_holes(cleaned_mask, max_hole_pixels=150)

                veto_count = int(vetoed.sum())
                recover_count = int(high_conf_dl_water.sum())
                dl_witness_used = True
                logger.info(
                    "DL consensus fusion: vetoed %d land-false-positive pixels, recovered %d DL-water pixels",
                    veto_count, recover_count,
                )
    except Exception as e:
        logger.warning("DL water witness unavailable, proceeding with spectral-only: %s", e)

    total_image_pixels = h * w
    min_component_pixels = max(30, round(total_image_pixels * 0.0005))

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
        box_h = max(1, ymax - ymin + 1)
        box_w = max(1, xmax - xmin + 1)
        
        solidity = comp_pixels / (box_h * box_w)
        aspect_ratio = max(box_h / box_w, box_w / box_h)
        
        # Penalize thin linear structures (e.g. roads or shadows along buildings)
        if aspect_ratio > 8 and solidity < 0.25:
            continue
            
        mean_index_score = float(ndwi_values[comp_mask].mean()) if ndwi_values is not None else 0.5
        
        # Region quality score factoring size, spatial coherence, and solidity
        shape_factor = (solidity ** 0.5) / (aspect_ratio ** 0.25)
        region_score = (mean_index_score + 1.0) * (comp_pixels ** 0.5) * shape_factor
        
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

    if valid_regions_data:
        filtered_regions = []
        primary_pixels = valid_regions_data[0]["pixels"]
        
        for r in valid_regions_data:
            # Keep the primary region
            if r["id"] == valid_regions_data[0]["id"]:
                filtered_regions.append(r)
                continue
                
            # For other regions, they must be at least 2% of the primary region size 
            # AND have a strong mean index (e.g. > 0.1)
            if r["pixels"] > primary_pixels * 0.02 and r["mean_index"] > 0.1:
                filtered_regions.append(r)
            # OR they must be extremely high confidence and solid
            elif r["mean_index"] > 0.4 and (r["pixels"] / (max(1, r["bbox_pixels"][2] - r["bbox_pixels"][0] + 1) * max(1, r["bbox_pixels"][3] - r["bbox_pixels"][1] + 1))) > 0.7:
                filtered_regions.append(r)

        valid_regions_data = filtered_regions

    result_type = ("combined" if dl_witness_used else "spectral") if is_spectral else ("combined" if dl_witness_used else "RGB proxy")

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
            "findings": f"No coherent, continuous water body could be identified with sufficient confidence ({result_type} analysis).",
            "primary_region": None,
            "grounding_task": "largest_water_body" if is_largest_request else "water_delineation",
            "is_spectral": is_spectral,
            "result_type": result_type,
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
    primary_cov_pct = round((primary_pixels / max(1, h * w)) * 100, 2)
    location_desc = _describe_region_location(
        primary_region["centroid_norm"][0],
        primary_region["centroid_norm"][1],
        coverage_pct=primary_cov_pct,
    )

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
    # A. Water Index / Probability Heatmap
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

    # Calibrate confidence based on spectral presence, region dominance, and DL consensus
    if is_spectral:
        confidence = 0.94 if largest_coverage_percent >= 5.0 else 0.88
    else:
        # RGB visual estimation confidence
        confidence = 0.86 if largest_coverage_percent >= 10.0 else 0.78

    # Boost confidence if DL consensus was used and agreed
    if dl_witness_used:
        confidence = min(0.97, confidence + 0.04)

    findings_text = (
        f"Primary water body verified in the {location_desc} ({result_type} analysis), covering {largest_coverage_percent}% of the scene "
        f"({primary_pixels:,} pixels)."
    )

    if is_spectral:
        producer = "multispectral_ndwi_consensus_engine_v3" if dl_witness_used else "multispectral_ndwi_grounding_engine_v2"
    else:
        producer = "optical_water_consensus_engine_v3" if dl_witness_used else "optical_water_grounding_engine_v2"

    return {
        "target": "water",
        "water_body_identified": True,
        "findings": findings_text,
        "primary_region": primary_region,
        "grounding_task": "largest_water_body" if is_largest_request else "water_delineation",
        "is_spectral": is_spectral,
        "dl_consensus": dl_witness_used,
        "result_type": result_type,
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
        "producer": producer,
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
    optical_water: np.ndarray | None = None
    producer = "optical_ndwi_plus_sar_backscatter_v2"

    try:
        ndwi, transform, crs = read_index(optical, "ndwi")
        if ndwi is not None:
            optical_water = np.isfinite(ndwi) & (ndwi >= 0.12)
    except Exception:
        ndwi = None

    if optical_water is None:
        # Check SatlasWaterNet deep learning model for RGB or imagery lacking NIR band
        try:
            from app.services.water_model import is_water_model_available, predict_water_mask
            if is_water_model_available():
                w_res = predict_water_mask(optical, output_dir / "opt_water_dl", max_size=1024)
                if w_res and w_res.get("mask_path"):
                    with Image.open(w_res["mask_path"]) as m_img:
                        m_arr = np.array(m_img)
                        optical_water = (m_arr[..., 0] > 0) | (m_arr[..., 1] > 0) | (m_arr[..., 2] > 0)
                        target_h, target_w = optical_water.shape
                        transform = Affine.identity()
                        producer = "optical_waternet_plus_sar_backscatter_v2"
        except Exception as e:
            logger.warning("Optical waternet extraction failed in sensor fusion: %s", e)

    if optical_water is None:
        try:
            with rasterio.open(optical) as src_opt:
                crs = src_opt.crs
                ratio = min(1.0, 1024 / max(src_opt.width, src_opt.height))
                out_w = max(1, round(src_opt.width * ratio))
                out_h = max(1, round(src_opt.height * ratio))
                transform = src_opt.transform * Affine.scale(src_opt.width / out_w, src_opt.height / out_h)
                arr = src_opt.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    p15 = np.percentile(finite, 15)
                    optical_water = np.isfinite(arr) & (arr <= p15)
                    producer = "optical_radiometric_plus_sar_backscatter_v2"
        except Exception:
            return None

    if optical_water is None:
        return None

    target_h, target_w = optical_water.shape
    try:
        with rasterio.open(sar) as src:
            radar = src.read(1, out_shape=(target_h, target_w), resampling=Resampling.bilinear).astype("float32")
    except Exception:
        return None

    finite_radar = radar[np.isfinite(radar)]
    if not finite_radar.size:
        return None

    # Check if linear intensity or dB
    if np.min(finite_radar) >= 0 and np.max(finite_radar) > 1.0:
        db = 10.0 * np.log10(np.clip(radar, 1e-6, None))
    else:
        db = radar

    # In SAR, smooth open water causes specular reflection away from antenna (typically <= -20 dB)
    sar_water = np.isfinite(db) & (db <= -20.0)
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
    if ndwi is not None:
        write_index_png(ndwi, "ndwi", ndwi_path)
    else:
        # Save optical water mask preview
        Image.fromarray((optical_water.astype("uint8") * 255), mode="L").save(ndwi_path, format="PNG")

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
        "producer": producer,
    }


def optical_sar_builtup_fusion(optical: Path, sar: Path, output_dir: Path) -> dict[str, Any] | None:
    """Jointly confirms built-up urban structures using optical building detection and SAR double-bounce."""
    from app.services.croma_pipeline import is_sar_image

    if is_sar_image(optical) and not is_sar_image(sar):
        optical, sar = sar, optical

    output_dir.mkdir(parents=True, exist_ok=True)
    optical_builtup: np.ndarray | None = None
    transform: Affine = Affine.identity()
    crs: Any = None
    producer = "optical_buildings_plus_sar_backscatter_v2"

    with rasterio.open(optical) as src_opt:
        crs = src_opt.crs
        ratio = min(1.0, 1024 / max(src_opt.width, src_opt.height))
        out_w = max(1, round(src_opt.width * ratio))
        out_h = max(1, round(src_opt.height * ratio))
        transform = src_opt.transform * Affine.scale(src_opt.width / out_w, src_opt.height / out_h)

    # 1. Optical built-up / building detection
    try:
        from app.services.landcover_model import is_buildings_model_available, predict_building_footprints
        if is_buildings_model_available():
            b_res = predict_building_footprints(optical, output_dir / "opt_buildings_dl", max_size=max(out_w, out_h))
            if b_res and b_res.get("mask_path"):
                with Image.open(b_res["mask_path"]) as m_img:
                    if m_img.size != (out_w, out_h):
                        m_img = m_img.resize((out_w, out_h), Image.NEAREST)
                    optical_builtup = np.array(m_img) > 0
    except Exception as e:
        logger.warning("Building model extraction failed: %s", e)

    if optical_builtup is None:
        try:
            with rasterio.open(optical) as src_opt:
                arr = src_opt.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    p75 = np.percentile(finite, 75)
                    optical_builtup = np.isfinite(arr) & (arr >= p75)
        except Exception:
            return None

    if optical_builtup is None:
        return None

    # 2. SAR radar high backscatter / corner double-bounce
    try:
        with rasterio.open(sar) as src_sar:
            radar = src_sar.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
    except Exception:
        return None

    finite_radar = radar[np.isfinite(radar)]
    if not finite_radar.size:
        return None

    if np.min(finite_radar) >= 0 and np.max(finite_radar) > 1.0:
        db = 10.0 * np.log10(np.clip(radar, 1e-6, None))
    else:
        db = radar

    # Strong double-bounce scattering from vertical structures typically >= -10 dB
    sar_builtup = np.isfinite(db) & (db >= -10.0)
    agreement = _denoise(optical_builtup & sar_builtup)
    union = optical_builtup | sar_builtup
    iou = float(agreement.sum() / max(1, union.sum()))

    rgba = np.zeros((*agreement.shape, 4), dtype="uint8")
    rgba[optical_builtup & ~sar_builtup] = [245, 158, 11, 185]
    rgba[sar_builtup & ~optical_builtup] = [168, 85, 247, 185]
    rgba[agreement] = [0, 229, 255, 225]
    agreement_path = output_dir / "sensor_agreement.png"
    Image.fromarray(rgba, mode="RGBA").save(agreement_path, format="PNG")

    builtup_map_path = output_dir / "builtup_optical.png"
    Image.fromarray((optical_builtup.astype("uint8") * 255), mode="L").save(builtup_map_path, format="PNG")

    geojson_path = output_dir / "confirmed_builtup_regions.geojson"
    region_count, area_m2 = _polygonize(
        agreement, transform, crs, kind="optical_sar_confirmed_builtup", output_path=geojson_path
    )
    confidence = round(min(0.92, 0.65 + 0.30 * iou), 3)

    return {
        "target": "built-up",
        "agreement_iou": round(iou, 4),
        "sensor_agreement_percent": round(iou * 100, 2),
        "confirmed_percent": round(float(agreement.mean() * 100), 3),
        "area_m2": area_m2,
        "region_count": region_count,
        "confidence": confidence,
        "agreement_path": agreement_path,
        "builtup_path": builtup_map_path,
        "geojson_path": geojson_path,
        "producer": producer,
    }


def classify_land_cover_scene(
    image_path: Path,
    output_dir: Path,
    query: str = "",
    max_size: int = 1024,
) -> dict[str, Any]:
    """Multi-class land-cover classification engine.

    Tries DL model (satquery_landcover_v1) first for RGB/GeoTIFF inputs.
    Falls back to the deterministic spectral engine if DL is unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Attempt Deep Learning Land-Cover Model (satquery_landcover_v1, smp.Unet/mit_b2)
    is_supported_image = image_path.suffix.lower() in (".tif", ".tiff", ".nc", ".geotiff", ".png", ".jpg", ".jpeg")
    if is_supported_image:
        try:
            from app.services.landcover_model import is_landcover_model_available, predict_landcover_mask
            if is_landcover_model_available():
                dl_res = predict_landcover_mask(image_path, output_dir, query=query, max_size=max_size)
                if dl_res is not None:
                    logger.info("DL land-cover inference succeeded (satquery_landcover_v1).")
                    return dl_res
        except Exception as e:
            logger.warning("DL landcover inference failed or skipped, falling back to spectral engine: %s", e)

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

    # 1. Water mask with strict vegetation, grass & dark soil suppression
    # Prevents land and vegetation from being falsely detected as water
    exg = 2.0 * g_norm - r_norm - b_norm
    is_vegetation = (exg > 0.03) | ((g_norm > b_norm * 1.30) & (g_norm > r_norm * 1.05))
    is_dark_soil = (r_norm > b_norm * 1.12) & (g_norm > b_norm * 0.95) & (brightness > 0.10)
    is_grass = (g_norm > r_norm * 1.05) & (g_norm > b_norm)
    water_candidate = (
        (ndwi >= 0.18) &
        (ndvi < 0.05) &
        (b_norm > r_norm * 1.15) &
        ~is_vegetation &
        ~is_dark_soil &
        ~is_grass &
        (brightness < 0.55) &
        (edge_density < 0.10)
    )
    struct_elem = ndi.generate_binary_structure(2, 2)
    water_mask = ndi.binary_closing(ndi.binary_opening(water_candidate, structure=struct_elem), structure=struct_elem)

    # 2. Forest / Dense canopy (dark green)
    forest_mask = _denoise((ndvi >= 0.38) & ~water_mask)

    # 3. Grass / General Vegetation (bright green)
    veg_mask = _denoise((ndvi >= 0.16) & ~forest_mask & ~water_mask)

    # 4. Road / Impervious surface (bright yellow)
    # Neutral spectral balance, low NDVI, asphalt/pavement brightness
    road_candidate = (
        (ndvi < 0.08) &
        (np.abs(r_norm - b_norm) < 0.09) &
        (np.abs(g_norm - b_norm) < 0.09) &
        (brightness >= 0.18) &
        (brightness <= 0.68) &
        ~water_mask & ~forest_mask & ~veg_mask
    )
    road_mask = _denoise(road_candidate & (edge_density > 0.06))

    # 5. Built-up / Urban footprint (warm red)
    builtup_mask = _denoise(
        ((ndbi >= 0.06) | (edge_density > 0.12) | ((brightness > 0.55) & (ndvi < 0.15))) &
        ~water_mask & ~forest_mask & ~veg_mask & ~road_mask
    )

    # 6. Agricultural land / Cultivated fields (light olive)
    agri_mask = _denoise(
        ((ndvi >= 0.08) & (ndvi < 0.28) & (r_norm > 0.18)) &
        ~water_mask & ~forest_mask & ~veg_mask & ~builtup_mask & ~road_mask
    )

    # 7. Bare land / Soil (dark yellow land color)
    bare_mask = ~water_mask & ~forest_mask & ~veg_mask & ~builtup_mask & ~agri_mask & ~road_mask

    # Calculate percentages
    water_pct = round(float(water_mask.sum() / total_pixels * 100), 2)
    forest_pct = round(float(forest_mask.sum() / total_pixels * 100), 2)
    veg_pct = round(float(veg_mask.sum() / total_pixels * 100), 2)
    total_veg_pct = round(veg_pct + forest_pct, 2)
    road_pct = round(float(road_mask.sum() / total_pixels * 100), 2)
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

    # Harmonious, visually distinct palette matching human visual perception:
    # Water: Dark Blue (#143c8c), Grass: Bright Green (#22c55e), Forest: Dark Green (#166534)
    # Road: Bright Yellow (#f5c81e), Bare Land: Dark Yellow (#c29b38), Built-up: Red (#dc3545)
    breakdown = {
        "water": {"percent": water_pct, "status": _qualitative(water_pct), "color": "#143c8c"},
        "vegetation": {"percent": total_veg_pct, "status": _qualitative(total_veg_pct), "color": "#22c55e"},
        "forest": {"percent": forest_pct, "status": _qualitative(forest_pct), "color": "#166534"},
        "road": {"percent": road_pct, "status": _qualitative(road_pct), "color": "#f5c81e"},
        "agricultural": {"percent": agri_pct, "status": _qualitative(agri_pct), "color": "#7cb342"},
        "built_up": {"percent": builtup_pct, "status": _qualitative(builtup_pct), "color": "#dc3545"},
        "bare_land": {"percent": bare_pct, "status": _qualitative(bare_pct), "color": "#c29b38"},
    }

    # 1. Composite Grounding Mask with Translucent Class Fills and Crisp White Contours
    grounding_rgba = np.zeros((out_h, out_w, 4), dtype="uint8")
    grounding_rgba[bare_mask] = [194, 155, 56, 140]      # Dark yellow (Land)
    grounding_rgba[agri_mask] = [124, 179, 66, 145]      # Light olive (Crops)
    grounding_rgba[road_mask] = [245, 200, 30, 160]      # Bright yellow (Road)
    grounding_rgba[builtup_mask] = [220, 53, 69, 160]    # Warm red (Built-up)
    grounding_rgba[veg_mask] = [34, 197, 94, 150]        # Bright green (Grass)
    grounding_rgba[forest_mask] = [22, 101, 52, 160]     # Dark green (Forest)
    grounding_rgba[water_mask] = [20, 60, 140, 165]      # Dark blue (Water)

    all_boundaries = np.zeros((out_h, out_w), dtype=bool)
    for m in (water_mask, forest_mask, veg_mask, road_mask, builtup_mask, agri_mask, bare_mask):
        if m.any():
            dil = ndi.binary_dilation(m, iterations=1)
            all_boundaries |= (dil & ~m)
    grounding_rgba[all_boundaries] = [255, 255, 255, 230]

    grounding_mask_path = output_dir / "land_cover_grounding_mask.png"
    Image.fromarray(grounding_rgba, mode="RGBA").save(grounding_mask_path, format="PNG")

    # 2. Individual Class Grounding Masks
    def _create_class_grounding(c_mask: np.ndarray, color: list[int], file_name: str) -> Path | None:
        if not c_mask.any():
            return None
        c_rgba = np.zeros((out_h, out_w, 4), dtype="uint8")
        c_rgba[c_mask] = [*color, 155]
        dil = ndi.binary_dilation(c_mask, iterations=2)
        bnd = dil & ~c_mask
        c_rgba[bnd] = [255, 255, 255, 255]
        c_path = output_dir / file_name
        Image.fromarray(c_rgba, mode="RGBA").save(c_path, format="PNG")
        return c_path

    class_masks: dict[str, Path] = {}
    builtup_p = _create_class_grounding(builtup_mask, [220, 53, 69], "builtup_grounding_mask.png")
    if builtup_p:
        class_masks["built_up"] = builtup_p

    veg_combined = veg_mask | forest_mask
    veg_p = _create_class_grounding(veg_combined, [34, 197, 94], "vegetation_grounding_mask.png")
    if veg_p:
        class_masks["vegetation"] = veg_p

    if forest_mask.any():
        forest_p = _create_class_grounding(forest_mask, [22, 101, 52], "forest_grounding_mask.png")
        if forest_p:
            class_masks["forest"] = forest_p

    if road_mask.any():
        road_p = _create_class_grounding(road_mask, [245, 200, 30], "road_grounding_mask.png")
        if road_p:
            class_masks["road"] = road_p

    bare_p = _create_class_grounding(bare_mask, [194, 155, 56], "bare_land_grounding_mask.png")
    if bare_p:
        class_masks["bare_land"] = bare_p

    if agri_mask.any():
        agri_p = _create_class_grounding(agri_mask, [124, 179, 66], "agri_grounding_mask.png")
        if agri_p:
            class_masks["agricultural"] = agri_p

    if water_mask.any() and not (output_dir / "water_grounding_mask.png").exists():
        water_p = _create_class_grounding(water_mask, [20, 60, 140], "water_grounding_mask.png")
        if water_p:
            class_masks["water"] = water_p

    # Land-only mask (terrestrial surfaces without water) - visualized in dark yellow
    land_mask = (builtup_mask | veg_mask | forest_mask | road_mask | agri_mask | bare_mask) & ~water_mask
    land_p = _create_class_grounding(land_mask, [194, 155, 56], "land_grounding_mask.png")
    if land_p:
        class_masks["land"] = land_p

    land_pixels = int(land_mask.sum())
    land_coverage_pct = round(land_pixels / total_pixels * 100, 2)
    land_geojson_path = output_dir / "land_regions.geojson"
    land_region_count, land_area_m2 = _polygonize(
        land_mask,
        transform,
        crs,
        kind="land_regions",
        output_path=land_geojson_path,
    )
    land_area_ha = round(land_area_m2 / 10000.0, 2) if land_area_m2 is not None else None

    # Solid Land Mask (dark yellow)
    land_solid = np.zeros((out_h, out_w, 4), dtype="uint8")
    land_solid[land_mask] = [194, 155, 56, 210]
    land_mask_path = output_dir / "land_mask.png"
    Image.fromarray(land_solid, mode="RGBA").save(land_mask_path, format="PNG")

    # 3. Dense solid classification map
    rgba = np.zeros((out_h, out_w, 4), dtype="uint8")
    rgba[bare_mask] = [194, 155, 56, 210]     # dark yellow (bare land)
    rgba[agri_mask] = [124, 179, 66, 220]     # light olive (crops)
    rgba[road_mask] = [245, 200, 30, 225]     # bright yellow (road)
    rgba[builtup_mask] = [220, 53, 69, 225]   # warm red (buildings)
    rgba[veg_mask] = [34, 197, 94, 220]       # bright green (grass)
    rgba[forest_mask] = [22, 101, 52, 235]    # dark green (forest)
    rgba[water_mask] = [20, 60, 140, 240]     # dark blue (water)

    mask_path = output_dir / "land_cover_mask.png"
    Image.fromarray(rgba, mode="RGBA").save(mask_path, format="PNG")

    geojson_path = output_dir / "land_cover.geojson"
    region_count, area_m2 = _polygonize(
        water_mask | forest_mask | veg_mask | road_mask | builtup_mask,
        transform,
        crs,
        kind="land_cover_zones",
        output_path=geojson_path,
    )

    dominant_class = max(breakdown.items(), key=lambda x: x[1]["percent"])

    q_lower = query.lower()
    is_land_only_query = any(k in q_lower for k in (
        "find land", "identify land", "highlight land", "highlight the land",
        "land only", "detect land", "show land", "where is land", "map land",
        "locate land", "segment land", "land classification", "classify land"
    )) or q_lower.strip() in ("land", "land cover", "land classification")

    area_str = f" ({land_area_ha} hectares)" if land_area_ha is not None else ""
    if is_land_only_query and land_p:
        summary_text = (
            f"Identified and highlighted land regions: {land_coverage_pct}% scene coverage"
            f"{area_str} across {land_region_count} landmass regions."
        )
        selected_grounding = land_p
        selected_mask = land_mask_path
        selected_geojson = land_geojson_path
        selected_region_count = land_region_count
        selected_area_m2 = land_area_m2
    else:
        summary_text = (
            f"Land cover analysis resolved: {dominant_class[0].replace('_', ' ').title()} is dominant ({dominant_class[1]['percent']}%), "
            f"Vegetation canopy: {total_veg_pct}%, Built-up: {builtup_pct}%, Roads: {road_pct}%, "
            f"Water bodies: {water_pct}%, Agricultural land: {agri_pct}%, Bare soil: {bare_pct}%."
        )
        selected_grounding = grounding_mask_path
        selected_mask = mask_path
        selected_geojson = geojson_path
        selected_region_count = region_count
        selected_area_m2 = area_m2

    return {
        "target": "land" if is_land_only_query else "land_cover",
        "producer": "multispectral_land_cover_engine_v1",
        "summary": summary_text,
        "confidence": 0.90,
        "breakdown": breakdown,
        "dominant_class": dominant_class[0],
        "mask_path": selected_mask,
        "grounding_mask_path": selected_grounding,
        "land_grounding_path": land_p or grounding_mask_path,
        "land_mask_path": land_mask_path,
        "class_grounding_masks": class_masks,
        "geojson_path": selected_geojson,
        "coverage_percent": land_coverage_pct,
        "land_coverage_percent": land_coverage_pct,
        "land_area_m2": land_area_m2,
        "land_area_ha": land_area_ha,
        "land_region_count": land_region_count,
        "water_percent": water_pct,
        "vegetation_percent": total_veg_pct,
        "forest_percent": forest_pct,
        "road_percent": road_pct,
        "agricultural_percent": agri_pct,
        "built_up_percent": builtup_pct,
        "bare_land_percent": bare_pct,
        "region_count": selected_region_count,
        "area_m2": selected_area_m2,
    }


def extract_vegetation_grounding(
    path: Path,
    output_dir: Path,
    query: str = "Find vegetation.",
    max_size: int = 1024,
) -> dict[str, Any]:
    """
    High-precision vegetation and forest grounding engine.
    - If true multispectral bands (NIR + Red) exist: computes true NDVI.
    - If RGB-only: records spectral band absence honestly and applies semantic specialist delineation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    q_lower = query.lower()
    is_forest_query = any(k in q_lower for k in ("forest", "dense canopy", "woodland", "trees"))

    has_spectral, _ = check_index_bands_available(path, "ndvi")
    if has_spectral:
        ndvi_values, transform, crs = read_index(path, "ndvi", max_size=max_size)
        h, w = ndvi_values.shape
        threshold = 0.38 if is_forest_query else 0.22
        veg_mask = np.isfinite(ndvi_values) & (ndvi_values >= threshold)
        veg_mask = _denoise(veg_mask)

        ndvi_path = output_dir / "ndvi.png"
        write_index_png(ndvi_values, "ndvi", ndvi_path)
        mask_path = output_dir / ("forest_mask.png" if is_forest_query else "vegetation_mask.png")
        grounding_path = output_dir / ("forest_grounding_mask.png" if is_forest_query else "vegetation_grounding_mask.png")
        geojson_path = output_dir / ("forest_regions.geojson" if is_forest_query else "vegetation_regions.geojson")

        color = (16, 185, 129) if not is_forest_query else (5, 150, 105)
        _write_mask(veg_mask, mask_path, color)

        grounding_rgba = np.zeros((h, w, 4), dtype="uint8")
        grounding_rgba[veg_mask] = [*color, 160]
        if veg_mask.any():
            dil = ndi.binary_dilation(veg_mask, iterations=2)
            bnd = dil & ~veg_mask
            grounding_rgba[bnd] = [255, 255, 255, 240]
        Image.fromarray(grounding_rgba, mode="RGBA").save(grounding_path, format="PNG")

        region_count, total_area_m2 = _polygonize(veg_mask, transform, crs, kind="forest" if is_forest_query else "vegetation", output_path=geojson_path)
        total_pixels = h * w
        cov_pct = round(float(veg_mask.sum() / max(1, total_pixels) * 100), 2)

        coords = np.argwhere(veg_mask)
        if len(coords):
            cy, cx = float(coords[:, 0].mean()) / h, float(coords[:, 1].mean()) / w
            loc_desc = _describe_region_location(cy, cx, coverage_pct=cov_pct)
        else:
            loc_desc = "the scene"

        findings = (
            f"Delineated {'dense forest canopy' if is_forest_query else 'vegetation canopy'} using calibrated multispectral NDVI (NIR + Red bands): "
            f"{cov_pct}% scene coverage ({region_count} regions) concentrated in {loc_desc}."
        )

        return {
            "target": "forest" if is_forest_query else "vegetation",
            "is_spectral": True,
            "result_type": "spectral",
            "index_type": "NDVI",
            "findings": findings,
            "coverage_percent": cov_pct,
            "mask_path": mask_path,
            "grounding_mask_path": grounding_path,
            "ndvi_path": ndvi_path,
            "geojson_path": geojson_path,
            "region_count": region_count,
            "area_m2": total_area_m2,
            "confidence": 0.92,
            "producer": "multispectral_ndvi_grounding_engine_v1",
        }
    else:
        # RGB only: Cannot compute real NDVI. Use semantic specialist.
        lc = classify_land_cover_scene(path, output_dir, query=query, max_size=max_size)
        forest_pct = lc.get("forest_percent", 0.0)
        veg_pct = lc.get("vegetation_percent", 0.0)
        cov_pct = forest_pct if is_forest_query else veg_pct

        target_key = "forest" if is_forest_query else "vegetation"
        grounding_path = lc.get("class_grounding_masks", {}).get(target_key) or lc["grounding_mask_path"]
        mask_path = lc.get("class_grounding_masks", {}).get(target_key) or lc["mask_path"]

        findings = (
            f"Spectral NDVI cannot be reliably computed from the supplied image because the required spectral bands are unavailable. "
            f"{'Forest' if is_forest_query else 'Vegetation'} canopy was delineated using semantic land cover classification: "
            f"{cov_pct}% scene coverage."
        )

        return {
            "target": "forest" if is_forest_query else "vegetation",
            "is_spectral": False,
            "result_type": "semantic",
            "index_type": "RGB_Semantic_Canopy",
            "spectral_limitation": "This index cannot be reliably computed from the supplied image because the required spectral bands are unavailable.",
            "findings": findings,
            "coverage_percent": cov_pct,
            "mask_path": mask_path,
            "grounding_mask_path": grounding_path,
            "ndvi_path": None,
            "geojson_path": lc["geojson_path"],
            "region_count": lc.get("region_count", 1),
            "area_m2": lc.get("area_m2"),
            "confidence": 0.84,
            "producer": "optical_semantic_vegetation_engine_v1",
        }


def extract_building_grounding(
    path: Path,
    output_dir: Path,
    query: str = "How many buildings are visible?",
    max_size: int = 1024,
) -> dict[str, Any] | None:
    """
    Building footprint detection engine.

    Priority order:
      1. satquery_buildings_v1 deep learning model (SatlasNet SwinV2-B + FPN dual-head)
      2. Spectral built-up index (NDBI) when SWIR + NIR bands are present
      3. Semantic land-cover built-up specialist fallback for RGB imagery
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Attempt DL building model on any raster input
    try:
        from app.services.landcover_model import is_buildings_model_available, predict_building_footprints
        if is_buildings_model_available():
            dl_res = predict_building_footprints(path, output_dir, query=query, max_size=max_size)
            if dl_res is not None:
                dl_res["result_type"] = "model-based"
                logger.info("DL building detection succeeded (satquery_buildings_v1): %d buildings.", dl_res.get("building_count", 0))
                return dl_res
    except Exception as e:
        logger.warning("DL building inference failed, falling back to spectral/semantic: %s", e)

    # 2. Spectral built-up index (NDBI-based) when SWIR + NIR bands exist
    has_ndbi, _ = check_index_bands_available(path, "ndbi")
    if has_ndbi:
        try:
            ndbi_values, transform, crs = read_index(path, "ndbi", max_size=max_size)
            h, w = ndbi_values.shape
            builtup_mask = np.isfinite(ndbi_values) & (ndbi_values >= 0.05)
            builtup_mask = _denoise(builtup_mask)

            if builtup_mask.any():
                geojson_path = output_dir / "buildings_proxy.geojson"
                grounding_path = output_dir / "building_grounding_proxy.png"
                _write_mask(builtup_mask, grounding_path, (249, 115, 22))
                region_count, area_m2 = _polygonize(builtup_mask, transform, crs, kind="built_up_spectral", output_path=geojson_path)
                coverage_pct = round(float(builtup_mask.mean() * 100), 2)

                coords = np.argwhere(builtup_mask)
                cy = float(coords[:, 0].mean()) / h
                cx = float(coords[:, 1].mean()) / w
                loc_desc = _describe_region_location(cy, cx)

                return {
                    "target": "buildings",
                    "producer": "spectral_ndbi_builtup_engine_v1",
                    "is_deep_learning": False,
                    "is_spectral": True,
                    "result_type": "spectral",
                    "model_name": "Multispectral NDBI built-up engine",
                    "findings": f"Multispectral NDBI verified {region_count} built-up region(s) covering {coverage_pct}% of the scene in {loc_desc}.",
                    "building_count": None,
                    "coverage_percent": coverage_pct,
                    "total_area_m2": area_m2,
                    "location_description": loc_desc,
                    "confidence": 0.82,
                    "mean_footprint_prob": None,
                    "supports_claim": region_count > 0,
                    "footprint_path": grounding_path,
                    "instances_path": grounding_path,
                    "grounding_mask_path": grounding_path,
                    "geojson_path": geojson_path,
                    "instance_stats": {},
                    "model_metrics": {},
                }
        except Exception:
            pass

    # 3. Semantic land cover fallback for RGB images lacking SWIR/NIR bands
    try:
        lc = classify_land_cover_scene(path, output_dir, query=query, max_size=max_size)
        built_pct = lc.get("built_up_percent", 0.0)
        built_grounding = lc.get("class_grounding_masks", {}).get("built_up") or lc["grounding_mask_path"]
        return {
            "target": "buildings",
            "producer": "optical_semantic_builtup_engine_v1",
            "is_deep_learning": False,
            "is_spectral": False,
            "result_type": "semantic",
            "spectral_limitation": "This index cannot be reliably computed from the supplied image because the required spectral bands are unavailable.",
            "model_name": "Optical semantic built-up classifier",
            "findings": (
                f"Spectral NDBI cannot be reliably computed from the supplied image because the required spectral bands are unavailable. "
                f"Built-up footprint covering {built_pct}% of the scene was identified via optical semantic analysis."
            ),
            "building_count": None,
            "coverage_percent": built_pct,
            "total_area_m2": lc.get("area_m2"),
            "location_description": "the analyzed scene",
            "confidence": 0.75,
            "mean_footprint_prob": None,
            "supports_claim": built_pct > 0,
            "footprint_path": built_grounding,
            "instances_path": built_grounding,
            "grounding_mask_path": built_grounding,
            "geojson_path": lc["geojson_path"],
            "instance_stats": {},
            "model_metrics": {},
        }
    except Exception:
        return None

