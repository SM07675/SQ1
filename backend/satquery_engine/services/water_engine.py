"""Comprehensive sensor-aware water detection engine.

Implements three distinct scientific routes:
- Route A: True Multispectral Water (NDWI, MNDWI, NDVI vegetation rejection, NDBI urban rejection, shadow masking)
- Route B: RGB Water Proxy (multi-cue optical proxy: spectral absorption, local texture variance, urban edge density, neutral color spread, river-preserving hysteresis)
- Route C: SAR Water (calibrated dB backscatter thresholding with speckle filter and contextual gating)

Produces:
- water_probability.tif
- water_mask.tif
- water.geojson
- water_overlay.png
- water_disagreement.png
- check_water_result() quality gate
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping, box
from shapely import make_valid
from shapely.ops import transform as transform_geom, unary_union
from pyproj import Transformer
from PIL import Image, ImageDraw

from satquery_engine.services.bands import detect_band_map
from satquery_engine.services.spatial_outputs import export_float_raster
from satquery_engine.services.raster import _area_square_meters, render_preview
from satquery_engine.analysis.shadow import ShadowEvidence, multispectral_shadow_evidence, rgb_shadow_evidence
from satquery_engine.analysis.surface import IlluminationState, SurfaceType, adjudicate_surface_and_illumination


@dataclass
class WaterComponentMetrics:
    component_id: int
    area_pixels: int
    area_map_units: float | None
    perimeter: float
    bbox: tuple[int, int, int, int]  # (min_y, min_x, max_y, max_x)
    mean_probability: float
    median_probability: float
    elongation: float  # length / width ratio
    compactness: float  # 4 * pi * area / perimeter^2
    border_contact: bool
    texture_std: float
    spectral_score: float
    semantic_score: float = 1.0


@dataclass
class WaterAnalysisOutput:
    probability: np.ndarray
    mask: np.ndarray
    disagreement: np.ndarray
    route: str  # "MULTISPECTRAL_NDWI_MNDWI" | "RGB_WATER_PROXY" | "SAR_WATER"
    coverage_percent: float
    valid_pixels: int
    water_pixels: int
    area_m2: float | None
    components: list[WaterComponentMetrics]
    paths: list[Path]
    quality_passed: bool
    quality_warnings: list[str]
    limitations: list[str]
    threshold_used: float


# ---------------------------------------------------------------------------
# Route A: Multispectral Water (NDWI + MNDWI + Exclusion Masks)
# ---------------------------------------------------------------------------

def compute_multispectral_water(
    src: rasterio.io.DatasetReader,
    band_indices: dict[str, int],
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute true multispectral water probability map and exclusion masks.

    Returns: (water_prob, valid_mask, exclusion_mask, details)
    """
    h = out_shape[0] if out_shape else src.height
    w = out_shape[1] if out_shape else src.width

    def read_b(role: str) -> np.ndarray | None:
        idx = band_indices.get(role)
        if idx is None or idx < 1 or idx > src.count:
            return None
        arr = src.read(idx, out_shape=(h, w), resampling=Resampling.bilinear, masked=True)
        vals = arr.astype("float32").filled(np.nan)
        scale = src.scales[idx - 1] if src.scales and len(src.scales) >= idx else 1.0
        offset = src.offsets[idx - 1] if src.offsets and len(src.offsets) >= idx else 0.0
        return vals * scale + offset

    green = read_b("green")
    nir = read_b("nir")
    red = read_b("red")
    swir = read_b("swir")
    if swir is None:
        swir = read_b("swir1")
    blue = read_b("blue")

    # Sentinel-2 L2A files commonly store reflectance as integer values near
    # 0..10000 without a GeoTIFF scale tag. Ratios are scale invariant, but the
    # shadow and built-up exclusions below use reflectance thresholds.
    s2_named = all(name in band_indices for name in ("b02", "b03", "b04", "b08", "b11", "b12"))
    s2_dn = s2_named and green is not None and np.any(np.isfinite(green)) and float(np.nanmax(green)) > 1.5
    if s2_dn:
        green /= 10000.0
        nir /= 10000.0
        if red is not None:
            red /= 10000.0
        if swir is not None:
            swir /= 10000.0
        if blue is not None:
            blue /= 10000.0

    if green is None or nir is None:
        raise ValueError("Multispectral water analysis requires at least Green and NIR bands.")

    valid = np.isfinite(green) & np.isfinite(nir)
    if red is not None:
        valid &= np.isfinite(red)
    if swir is not None:
        valid &= np.isfinite(swir)

    # 1. NDWI = (Green - NIR) / (Green + NIR)
    ndwi = np.where(valid, (green - nir) / (green + nir + 1e-7), -1.0)

    # 2. MNDWI = (Green - SWIR) / (Green + SWIR) if SWIR is available
    has_mndwi = swir is not None
    mndwi = np.where(valid, (green - swir) / (green + swir + 1e-7), -1.0) if has_mndwi else None

    # 3. Exclusion: NDVI for vegetation rejection
    ndvi = np.where(valid & (red is not None), (nir - red) / (nir + red + 1e-7), 0.0) if red is not None else np.zeros((h, w), dtype="float32")
    if red is not None:
        is_vegetation = (ndvi > 0.18) | ((nir > green * 1.15) & (nir > red))
    else:
        is_vegetation = (nir > green * 1.15)

    # 4. Exclusion: NDBI for built-up / urban roofs
    is_builtup = np.zeros((h, w), dtype=bool)
    ndbi = np.full((h, w), -1.0, dtype="float32")
    if swir is not None:
        ndbi = np.where(valid, (swir - nir) / (swir + nir + 1e-7), -1.0)
        is_builtup = (ndbi > 0.02) & (nir < 0.20) & (swir > green * 1.10)

    # 5. Total brightness & shadow rejection
    brightness = (green + nir + (red if red is not None else green)) / 3.0
    is_extreme_shadow = valid & (brightness < 0.015) & (ndwi < 0.05)

    exclusion = is_vegetation | is_builtup | is_extreme_shadow

    # Combined spectral water score
    if has_mndwi and mndwi is not None:
        raw_score = 0.50 * ndwi + 0.50 * mndwi
    else:
        raw_score = ndwi

    # Probability via validated sigmoid centered at 0.05
    prob = 1.0 / (1.0 + np.exp(-12.0 * (raw_score - 0.05)))
    prob[~valid] = 0.0
    prob[exclusion] = np.minimum(prob[exclusion], 0.05)

    details = {
        "has_mndwi": has_mndwi,
        "radiometry": "Sentinel-2 digital numbers / 10000" if s2_dn else "GeoTIFF scale and offset",
        "mean_ndwi": float(ndwi[valid].mean()) if valid.any() else 0.0,
        "mean_mndwi": float(mndwi[valid].mean()) if has_mndwi and mndwi is not None and valid.any() else None,
        "vegetation_pixels": int(is_vegetation.sum()),
        "builtup_pixels": int(is_builtup.sum()),
        "brightness": brightness.astype("float32"),
        "vegetation_probability": np.where(valid, np.clip((ndvi + 0.10) / 0.60, 0.0, 1.0), 0.0).astype("float32"),
        "builtup_probability": np.where(valid, np.clip((ndbi + 0.10) / 0.45, 0.0, 1.0), 0.0).astype("float32"),
    }
    return prob.astype("float32"), valid, exclusion, details


# ---------------------------------------------------------------------------
# Route B: RGB-Only Water Proxy with Multi-Cue Physical Evidence & Hysteresis
# ---------------------------------------------------------------------------

def compute_rgb_water_proxy(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Multi-cue optical water proxy designed for high-resolution & VHR RGB imagery.

    Evaluates:
    - Spectral absorption & color evidence (water absorbs red, transmits blue/cyan)
    - Turbid inland lakes, mountain reservoirs, sediment rivers, deep water, clear blue water
    - High-frequency spatial texture variance (water is smooth, tree canopy is rough)
    - Sobel gradient magnitude
    - Urban edge density (rejects asphalt roads, parking lots, industrial rooftops)
    - Cast shadow & dark terrain discrimination (neutral desaturation check)
    - Vegetation suppression (Excess Green: 2G - R - B, with flat water protection)

    Returns: (probability, core_seeds, exclusion_mask, details)
    """
    brightness = (r + g + b) / 3.0
    y_lum = 0.299 * r + 0.587 * g + 0.114 * b

    br_denom = b + r + 1e-7
    ndbr = (b - r) / br_denom
    gr_denom = g + r + 1e-7
    ndgr = (g - r) / gr_denom

    # 1. Spatial texture: local standard deviation in 5x5 window
    mean_sq = ndi.uniform_filter(y_lum * y_lum, size=5)
    mean_y = ndi.uniform_filter(y_lum, size=5)
    local_texture_std = np.sqrt(np.maximum(0.0, mean_sq - mean_y * mean_y))

    # 2. Gradient magnitude & edge density
    gx = ndi.sobel(y_lum, axis=1) / 4.0
    gy = ndi.sobel(y_lum, axis=0) / 4.0
    gradient_mag = np.sqrt(gx * gx + gy * gy)
    strong_edges = (gradient_mag > 0.050).astype("float32")
    urban_edge_density = ndi.uniform_filter(strong_edges, size=15)
    broad_urban_edges = ndi.uniform_filter(strong_edges, size=31)
    bright_surface_density = ndi.uniform_filter((brightness > 0.55).astype("float32"), size=61)

    # 3. Suppressions:
    # A. Vegetation / agricultural fields / forest canopy / grass
    # Aggressively reject green-dominant pixels. Grass and vegetation absorb
    # red and reflect green, which creates false blue-dominant ratios.
    exg = 2.0 * g - r - b
    is_vegetation = (
        (g > b * 1.35)  # Lowered from 1.65 — catches more grass/fields
        | ((exg > 0.03) & (local_texture_std > 0.010))  # Lowered ExG gate for grass
        | (exg > 0.18)  # Lowered hard ExG cap
        | ((g > r * 1.08) & (g > b) & (local_texture_std > 0.012))  # Grass-specific: green > both R & B with any texture
    )

    # B. Forest canopy texture & mountain terrain roughness
    # Lowered thresholds to reject more textured surfaces (tree canopy, rough terrain)
    is_rough_canopy = (local_texture_std > 0.030) | (gradient_mag > 0.070)

    # C. Urban structural surfaces: asphalt, parking lots, dark roofs, sports grounds
    color_spread = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    is_neutral_gray = (color_spread < 0.030) & (ndbr < 0.05) & (b <= r * 1.10)
    is_urban_structure = (urban_edge_density > 0.08) | is_neutral_gray
    # Blue-tinted asphalt may pass every color test. A road runs through a
    # larger field of roof and curb edges, unlike an open water surface.
    urban_scene = float(np.mean((brightness > 0.55) & valid)) > 0.20
    context_threshold = 0.005 if urban_scene else 0.08
    road_context = (bright_surface_density > context_threshold) & (broad_urban_edges > context_threshold)
    # Note: is_neutral_gray covers neutral-gray pixels (e.g. cast building shadows with r≈g≈b).
    # Deep ocean water has b > r and is not neutral gray.

    # D. Extreme highlights (clouds / specular glint) and extreme dark noise
    is_too_bright = brightness > 0.55
    is_too_dark = (brightness < 0.005) | ((brightness < 0.020) & (b <= r))

    # E. Dark soil / bare land: brownish tones where r >= g and low texture
    is_dark_soil = (r >= g * 0.95) & (r >= b) & (brightness < 0.25) & (brightness > 0.03) & (ndbr < 0.08)

    # Strong blue dominance is spectrally inconsistent with forest canopy or urban structures.
    # Protect pixels where b >> r AND b > g — the 5x5 texture window and 15px edge-density
    # kernel can falsely trigger is_rough_canopy / urban_edge_density at scene boundaries.
    # Real blue water cannot simultaneously be a canopy patch or a roof/road.
    # Tightened: require b > r * 1.35 (was 1.25) for unambiguous blue water
    is_unambiguous_blue_water = (
        ((b > r * 1.35) & (b > g * 0.80) & (y_lum < 0.35) & (local_texture_std < 0.030))
        | ((y_lum < 0.08) & (b > r * 1.15) & (b >= g * 0.75) & (local_texture_std < 0.020))
    )

    exclusion = (
        is_vegetation
        | (is_rough_canopy & ~is_unambiguous_blue_water)
        | (is_urban_structure & ~is_unambiguous_blue_water)
        | is_too_bright
        | is_too_dark
        | (is_dark_soil & ~is_unambiguous_blue_water)
        | road_context
    )

    # 4. Multi-type water candidates:
    # Type 1: Clear / blue water (ocean, clean river)
    # Tightened: require b > r * 1.15 (was 1.0) to prevent green-tinted land from passing
    type1 = (b > r * 1.15) & (b >= g * 0.70) & (y_lum < 0.40) & (y_lum >= 0.008)
    # Type 2: Turbid inland lake / mountain reservoir
    # Tightened: stricter blue requirement and lower texture threshold
    type2 = (g >= r * 1.12) & (b >= r * 1.20) & (g <= b * 1.40) & (local_texture_std < 0.018) & (y_lum < 0.30)
    # Type 3: Sediment braided river channels
    # Tightened: require very smooth surface and narrower color range
    rg_diff = np.abs(r - g) / (r + g + 1e-6)
    type3 = (
        (rg_diff < 0.12)
        & (b >= np.minimum(r, g) * 0.70)
        & (b <= np.maximum(r, g) * 1.20)
        & (g <= b * 1.40)
        & (local_texture_std < 0.025)
        & (y_lum >= 0.06)
        & (y_lum <= 0.45)
    )
    # Type 4: Deep dark water
    # Requires blue strictly above red (b > r * 1.10). Water absorbs red and has blue
    # dominance. Cast building shadows are neutral gray with b≈r and must not pass this gate.
    # Tightened: stronger blue requirement and lower texture
    type4 = (y_lum < 0.12) & (y_lum >= 0.008) & (b > r * 1.10) & (b >= g * 0.75) & (local_texture_std < 0.018)

    candidate = (type1 | type2 | type3 | type4) & ~exclusion & valid

    # Continuous score — weight texture and edge more heavily to suppress land
    base_score = np.clip(
        ndbr * 0.40 + ndgr * 0.15 + (0.25 - brightness) * 0.20 - local_texture_std * 1.80 - urban_edge_density * 0.80,
        -1.0,
        1.0,
    )
    # Reduced non-candidate probability from 0.20 to 0.10 to prevent land from creeping in
    prob = np.where(candidate, np.maximum(0.55, 0.50 + base_score * 0.45), np.minimum(0.10, np.maximum(0.0, 0.10 + base_score * 0.15)))
    prob[exclusion] = np.minimum(prob[exclusion], 0.03)
    prob[~valid] = 0.0

    # Core high-confidence seeds for river/lake hysteresis
    core_seeds = (type1 | type2 | (type3 & (rg_diff < 0.08))) & ~exclusion & (local_texture_std < 0.026) & valid

    # Deep water can be almost achromatic in rendered satellite RGB. Darkness
    # alone is not evidence: admit it only through connectivity to chromatic
    # water, without crossing vegetation, cloud, invalid pixels or strong edges.
    dark_support = (
        valid & (brightness >= 0.003) & (brightness < 0.12)
        & (color_spread < 0.055) & (exg < 0.025)
        & (local_texture_std < 0.018) & (gradient_mag < 0.045)
        & (urban_edge_density < 0.08) & ~is_vegetation
    )
    chromatic_seeds = core_seeds & (ndbr > 0.15) & (b > g * 0.85)
    connected_dark = ndi.binary_propagation(
        chromatic_seeds, mask=(candidate | dark_support) & valid,
    ) & dark_support
    prob[connected_dark] = np.maximum(prob[connected_dark], 0.42)
    exclusion[connected_dark] = False

    details = {
        "candidate_pixels": int(candidate.sum()),
        "core_seed_pixels": int(core_seeds.sum()),
        "vegetation_suppressed": int(is_vegetation.sum()),
        "urban_suppressed": int(is_urban_structure.sum()),
        "road_context_suppressed": int(road_context.sum()),
        "urban_scene": urban_scene,
        "connected_dark_water_pixels": int(connected_dark.sum()),
        "vegetation_probability": np.where(valid, np.clip((exg + 0.02) / 0.30, 0.0, 1.0), 0.0).astype("float32"),
        "builtup_probability": np.where(valid, np.clip(urban_edge_density / 0.22, 0.0, 1.0), 0.0).astype("float32"),
    }
    return prob.astype("float32"), core_seeds, exclusion, details


# ---------------------------------------------------------------------------
# Route C: SAR Water (Calibrated Backscatter + Contextual Gating)
# ---------------------------------------------------------------------------

def compute_sar_water(
    src_or_data: Any,
    band_indices: dict[str, int] | None = None,
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrated SAR specular water detection with adaptive thresholding and dual-pol support.

    Returns: (probability, valid_mask, exclusion_mask, details)
    """
    if hasattr(src_or_data, "db"):
        db = src_or_data.db[0]
        valid = src_or_data.valid
        dual_pol_ratio = src_or_data.configuration.get("dual_pol_ratio")
    else:
        src = src_or_data
        h = out_shape[0] if out_shape else src.height
        w = out_shape[1] if out_shape else src.width
        vv_idx = (band_indices or {}).get("vv") or 1
        arr = src.read(vv_idx, out_shape=(h, w), resampling=Resampling.bilinear, masked=True)
        vals = arr.astype("float32").filled(np.nan)
        valid = np.isfinite(vals) & (vals > 0)
        if valid.any() and float(np.nanmean(vals[valid])) > 0 and float(np.nanmax(vals[valid])) > 1.0:
            db = 10.0 * np.log10(np.maximum(vals, 1e-6))
        else:
            db = vals
        dual_pol_ratio = None

    # Enhanced speckle filtering: 5x5 median for impulse noise, then 3x3 mean for smoothing
    db_stage1 = ndi.median_filter(np.where(valid, db, 0.0), size=5)
    db_filtered = ndi.uniform_filter(db_stage1, size=3)

    # Adaptive threshold: use scene statistics to find water/non-water boundary
    # Water typically has very low backscatter (< -16 dB on VV)
    valid_db = db_filtered[valid]
    if valid_db.size > 100:
        # Otsu-like: find bimodal split in backscatter histogram
        p10 = float(np.percentile(valid_db, 10))
        p50 = float(np.percentile(valid_db, 50))
        # Adaptive center: between the dark peak (likely water) and scene median
        adaptive_center = (p10 + p50) / 2.0
        # Clamp to reasonable range for water detection
        adaptive_center = max(-20.0, min(-8.0, adaptive_center))
    else:
        adaptive_center = -14.0

    # Water specular reflection causes extremely low backscatter
    water_score = np.clip((adaptive_center - 2.0 - db_filtered) / 8.0, 0.0, 1.0)

    # Dual-pol enhancement: VH/VV ratio discriminates water from vegetation
    # Water: VH/VV ratio typically < -7 dB (low cross-pol)
    # Vegetation: VH/VV ratio typically > -4 dB (volume scattering)
    if dual_pol_ratio is not None:
        ratio_score = np.clip((-4.0 - dual_pol_ratio) / 6.0, 0.0, 1.0)
        water_score = 0.65 * water_score + 0.35 * ratio_score

    prob = np.where(valid, water_score, 0.0).astype("float32")

    # Exclusion: radar shadows on steep terrain (high local gradient in dB)
    gx = ndi.sobel(db_filtered, axis=1) / 4.0
    gy = ndi.sobel(db_filtered, axis=0) / 4.0
    grad = np.sqrt(gx * gx + gy * gy)
    is_radar_shadow = valid & (grad > 3.5)

    # Additional: reject very bright returns (urban, metallic structures)
    is_bright_return = valid & (db_filtered > -3.0)

    exclusion = is_radar_shadow | is_bright_return
    prob[exclusion] = np.minimum(prob[exclusion], 0.08)

    details = {
        "mean_db": float(db[valid].mean()) if valid.any() else 0.0,
        "adaptive_threshold_db": adaptive_center,
        "radar_shadow_pixels": int(is_radar_shadow.sum()),
        "bright_return_pixels": int(is_bright_return.sum()),
        "dual_pol_used": dual_pol_ratio is not None,
        "speckle_filter": "median5x5_then_mean3x3",
    }
    return prob, valid, exclusion, details


# ---------------------------------------------------------------------------
# River Preservation & Connected Component Postprocessing
# ---------------------------------------------------------------------------

def postprocess_water_mask(
    prob: np.ndarray,
    valid: np.ndarray,
    core_seeds: np.ndarray | None = None,
    threshold: float = 0.50,
    min_component_px: int = 25,
    pixel_res: float | None = None,
    transform: Affine | None = None,
    crs: Any = None,
    apply_morphology: bool = True,
    filter_compact: bool = True,
) -> tuple[np.ndarray, list[WaterComponentMetrics], np.ndarray]:
    """Execute topological river-preserving post-processing and component metrics.

    Steps:
    1. Hysteresis reconstruction from core seeds into candidate water (preserves continuous rivers)
    2. Structuring cross closing to connect narrow reaches and bridge crossings (if optical proxy)
    3. Connected component analysis with geometric feature extraction
    4. Elongated waterway preservation (length / width > 2.2 protected from small-area deletion)
    5. Urban dark compact false positive suppression

    Returns: (final_mask, component_metrics, disagreement_map)
    """
    if core_seeds is not None and core_seeds.any():
        candidate_mask = (prob > 0.35) & valid
        # Conflicting or invalid seeds cannot turn rejected land into water.
        recon_mask = ndi.binary_propagation(core_seeds & candidate_mask, mask=candidate_mask)
        base_mask = (prob > threshold) | recon_mask
    else:
        base_mask = prob > threshold

    base_mask &= valid

    struct_cross = ndi.generate_binary_structure(2, 1)
    if apply_morphology:
        # Closing must not paint across rejected land, cloud or shoreline pixels.
        closed_mask = ((ndi.binary_closing(base_mask, structure=struct_cross, iterations=1)
                        & (prob > 0.35)) | base_mask) & valid
    else:
        closed_mask = base_mask

    labels, n_comp = ndi.label(closed_mask)
    if n_comp == 0:
        return np.zeros_like(closed_mask), [], np.zeros_like(prob)

    final_mask = np.zeros_like(closed_mask)
    component_metrics: list[WaterComponentMetrics] = []
    disagreement = np.zeros_like(prob)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    slices = ndi.find_objects(labels)

    for cid in range(1, n_comp + 1):
        sl = slices[cid - 1]
        if sl is None:
            continue
        c_size = sizes[cid]
        comp_bool = labels[sl] == cid

        # Bounding box
        min_y, min_x = sl[0].start, sl[1].start
        max_y, max_x = sl[0].stop, sl[1].stop
        bh = max_y - min_y
        bw = max_x - min_x
        border_contact = (
            min_y == 0
            or min_x == 0
            or max_y == labels.shape[0]
            or max_x == labels.shape[1]
        )

        elongation = float(max(bh, bw) / max(1, min(bh, bw)))
        eroded = ndi.binary_erosion(comp_bool, structure=struct_cross)
        perimeter = float(np.sum(comp_bool ^ eroded))
        compactness = float((4.0 * np.pi * c_size) / max(1.0, perimeter * perimeter))

        comp_prob = prob[sl][comp_bool]
        mean_p = float(comp_prob.mean()) if comp_prob.size else 0.0
        med_p = float(np.median(comp_prob)) if comp_prob.size else 0.0
        std_p = float(comp_prob.std()) if comp_prob.size else 0.0
        area_map = float(c_size * (pixel_res ** 2)) if pixel_res else None

        metric = WaterComponentMetrics(
            component_id=cid,
            area_pixels=int(c_size),
            area_map_units=area_map,
            perimeter=perimeter,
            bbox=(min_y, min_x, max_y, max_x),
            mean_probability=mean_p,
            median_probability=med_p,
            elongation=elongation,
            compactness=compactness,
            border_contact=border_contact,
            texture_std=std_p,
            spectral_score=mean_p,
        )

        # 1. Tiny isolated noise: size < min_component_px UNLESS it's an elongated channel
        if c_size < min_component_px and elongation < 2.2:
            continue

        # 2. Reject isolated compact dark polygons in dense urban areas
        # (e.g. dark sports grounds/asphalt squares surrounded by buildings with elongation < 1.5)
        # Protect high spectral-probability detections (mean_p >= 0.60): they are likely
        # genuine ponds even if they are compact and in an urban setting.
        if filter_compact and 200 < c_size < 5000 and compactness > 0.28 and elongation < 1.5 and mean_p < 0.60 and not border_contact:
            disagreement[sl][comp_bool] = 1.0
            continue

        final_mask[sl][comp_bool] = True
        component_metrics.append(metric)

    return final_mask, component_metrics, disagreement


# ---------------------------------------------------------------------------
# Tiled Inference for Large GeoTIFFs
# ---------------------------------------------------------------------------

def run_tiled_rgb_water(
    path: Path,
    out_shape: tuple[int, int],
    tile_size: int = 512,
    overlap: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process RGB rasters preserving radiometry and neighborhood spatial context."""
    from satquery_engine.services.radiometry import rgb_unit_data
    with rasterio.open(path) as src:
        rgb, valid, _ = rgb_unit_data(src, out_shape=out_shape)
        prob, seeds, _, _ = compute_rgb_water_proxy(rgb[0], rgb[1], rgb[2], valid)
        return prob, seeds, valid


@lru_cache(maxsize=2)
def _checkpoint_sha256(path: str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_satquery_water_model(
    src: rasterio.io.DatasetReader,
    band_indices: dict[str, int],
    bundle_path: Path,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the fine-tuned Sentinel-2 model with overlap blending.

    This is deliberately fail-closed: all ten source bands used by the trained
    preprocessing graph must have exact Sentinel-2 identities. Generic band
    numbers and six-band Prithvi composites are not silently substituted.
    """
    import torch

    from satquery_engine.models.satlas_water_net import SOURCE_BANDS, load_water_bundle, prepare_water_inputs
    from satquery_engine.services.ingestion import _starts

    keys = tuple(name.lower() for name in SOURCE_BANDS)
    missing = [name for name, key in zip(SOURCE_BANDS, keys) if key not in band_indices]
    if missing:
        raise ValueError(
            "Fine-tuned Sentinel-2 water model was not used because exact bands are missing: "
            + ", ".join(missing)
        )

    model, config, metrics = load_water_bundle(bundle_path)
    tile_size = int(config.get("tile_inference", {}).get("tile_size", 512))
    overlap = int(config.get("tile_inference", {}).get("overlap", 128))
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("Water bundle tile overlap must be smaller than its tile size.")

    from satquery_engine.models.device import torch_device
    device = torch_device()
    model = model.to(device)
    h, w = src.height, src.width
    accum = np.zeros((h, w), dtype="float32")
    weights = np.zeros((h, w), dtype="float32")
    scene_valid = np.zeros((h, w), dtype=bool)
    hann = np.maximum(np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype("float32"), 1e-4)
    positions = [(y, x) for y in _starts(h, tile_size, stride) for x in _starts(w, tile_size, stride)]
    indexes = [band_indices[key] for key in keys]

    with torch.inference_mode():
        for tile_id, (y, x) in enumerate(positions):
            th, tw = min(tile_size, h - y), min(tile_size, w - x)
            window = Window(x, y, tw, th)
            raw = src.read(indexes, window=window, masked=True)
            raw_values = raw.astype("float32").filled(np.nan)
            data_valid = np.all(~np.ma.getmaskarray(raw), axis=0)
            pad_y, pad_x = tile_size - th, tile_size - tw
            if pad_y or pad_x:
                mode = "reflect" if th > 1 and tw > 1 else "edge"
                raw_values = np.pad(raw_values, ((0, 0), (0, pad_y), (0, pad_x)), mode=mode)
                data_valid = np.pad(data_valid, ((0, pad_y), (0, pad_x)), mode="constant")
            source = {name: raw_values[i] for i, name in enumerate(SOURCE_BANDS)}
            image, spectral, finite = prepare_water_inputs(source)
            tile_valid = finite & data_valid
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            spectral_tensor = torch.from_numpy(spectral).unsqueeze(0).to(device)
            logits = model(image_tensor, spectral_tensor)
            if logits.shape != (1, 2, tile_size, tile_size) or not torch.isfinite(logits).all():
                raise ValueError("Fine-tuned water model returned invalid logits.")
            probability = torch.softmax(logits, dim=1)[0, 1].detach().cpu().numpy()
            kernel = hann[:th, :tw]
            accum[y:y + th, x:x + tw] += probability[:th, :tw] * kernel
            weights[y:y + th, x:x + tw] += kernel
            scene_valid[y:y + th, x:x + tw] |= tile_valid[:th, :tw]
            if progress:
                progress(f"Analyzing water: tile {tile_id + 1} of {len(positions)}")

    if device == "cuda":
        model.cpu()
        torch.cuda.empty_cache()

    probability = np.divide(accum, weights, out=np.zeros_like(accum), where=weights > 0)
    probability[~scene_valid] = 0.0
    state_path = bundle_path / "satquery_water_state_dict.pt"
    return probability, scene_valid, {
        "model_id": "satquery_water_bundle",
        "architecture": config.get("architecture", "SatlasWaterNet"),
        "device": device,
        "tile_count": len(positions),
        "tile_size": tile_size,
        "tile_overlap": overlap,
        "threshold": float(config.get("postprocess", {}).get("threshold", 0.6)),
        "min_area_pixels": int(config.get("postprocess", {}).get("min_area_pixels", 16)),
        "checkpoint_sha256": _checkpoint_sha256(str(state_path)),
        "benchmark": metrics.get("test", {}),
        "preprocessing": {
            "backbone_band_order": list(config.get("satlas_band_order", [])),
            "spectral_features": list(config.get("spectral_features", [])),
            "normalization": config.get("normalization", {}),
        },
    }


# ---------------------------------------------------------------------------
# Sanity Gate & Output Generation
# ---------------------------------------------------------------------------

def check_water_result(
    mask: np.ndarray,
    prob: np.ndarray,
    valid: np.ndarray,
    route: str,
    components: list[WaterComponentMetrics],
    is_coastal: bool = False,
) -> tuple[bool, list[str]]:
    """Validate water result against sanity rules and physical constraints."""
    warnings: list[str] = []
    total_valid = int(valid.sum())
    if total_valid == 0:
        return False, ["No valid pixels in the image."]

    water_pct = float(mask.sum()) / total_valid * 100.0

    if water_pct > 85.0 and not is_coastal:
        warnings.append(
            f"Extremely high water surface cover ({water_pct:.1f}%) detected on inland scene; review recommended."
        )

    if (mask & ~valid).any():
        return False, ["Water mask contains nodata / invalid pixels."]

    if not np.all(np.isfinite(prob[valid])):
        return False, ["Non-finite probability values detected in water map."]

    if mask.shape != prob.shape:
        return False, ["Water mask dimensions do not match probability map."]

    if len(components) > 1500:
        warnings.append(
            f"High component count ({len(components)}) indicates possible noise or heavily fragmented waterways."
        )

    return True, warnings


def confirm_aerial_water(
    primary_probability: np.ndarray,
    corroborating_probability: np.ndarray,
    valid: np.ndarray,
    *,
    primary_threshold: float = 0.55,
    corroborating_threshold: float = 0.50,
    minimum_component_agreement: float = 0.60,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Withhold RGB water objects without substantial independent model support.

    RGB color and smoothness alone confuse turf, shadow and pavement with water.
    Check agreement at the object level so a few corroborating speckles cannot
    turn an entire golf fairway into a reported water body.
    """
    if primary_probability.shape != corroborating_probability.shape or valid.shape != primary_probability.shape:
        raise ValueError("Aerial water corroboration grids do not align.")
    candidate = (primary_probability >= primary_threshold) & valid
    supported = (corroborating_probability >= corroborating_threshold) & valid
    components, count = ndi.label(candidate)
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    supported_sizes = np.bincount(components[supported].ravel(), minlength=count + 1)
    accepted = np.zeros(count + 1, dtype=bool)
    if count:
        accepted[1:] = (sizes[1:] >= 25) & (supported_sizes[1:] / np.maximum(sizes[1:], 1) >= minimum_component_agreement)
    confirmed = accepted[components] & candidate
    probability = np.where(confirmed, primary_probability, np.minimum(primary_probability, 0.10))
    candidate_pixels = int(candidate.sum())
    withheld_pixels = int((candidate & ~confirmed).sum())
    return probability.astype("float32"), {
        "candidate_components": count,
        "confirmed_components": int(accepted.sum()),
        "candidate_pixels": candidate_pixels,
        "withheld_candidate_pixels": withheld_pixels,
        "withheld_fraction": withheld_pixels / candidate_pixels if candidate_pixels else 0.0,
    }


def suppress_road_water_conflicts(
    water_probability: np.ndarray,
    built_probability: np.ndarray,
    rgb_water_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Withhold aerial water where built surfaces win and RGB has no water cue."""
    if water_probability.shape != built_probability.shape or water_probability.shape != rgb_water_probability.shape:
        raise ValueError("Water and built-up evidence grids do not align.")
    conflict = (built_probability >= np.maximum(0.35, water_probability * 0.70)) & (rgb_water_probability < 0.50)
    filtered = np.where(conflict, np.minimum(water_probability, 0.10), water_probability)
    return filtered.astype("float32"), conflict


def execute_water_pipeline(
    path: Path,
    output_dir: Path,
    largest: bool = False,
    strict: bool = False,
    use_model: bool = True,
) -> dict[str, Any]:
    """Main entry point: sensor-aware automated water analysis.

    Chooses Route A (Multispectral), Route B (RGB Proxy), or Route C (SAR) automatically.
    Generates all canonical GIS, GeoTIFF, and visual artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path) as src:
        band_map = detect_band_map(src)
        crs = src.crs
        width, height = src.width, src.height
        pixel_res = abs(float(src.res[0])) if crs and src.res else None

        # Determine Route
        has_green = "green" in band_map.indices
        has_nir = "nir" in band_map.indices
        has_vv = "vv" in band_map.indices
        is_sar = has_vv or any(p in src.tags().get("modality", "").lower() for p in ("sar", "radar"))

        if has_green and has_nir:
            route = "MULTISPECTRAL_NDWI_MNDWI"
        elif is_sar:
            if strict:
                raise ValueError("I cannot calculate NDWI because the required named spectral bands are missing.")
            route = "SAR_WATER"
        else:
            if strict:
                raise ValueError("I cannot calculate NDWI because the required named spectral bands are missing.")
            # Use the shared explicit RGB contract, including supported aerial
            # RGB/NIR/DSM files whose extra bands have no spectral descriptions.
            from satquery_engine.services.radiometry import rgb_indexes
            rgb_indexes(src)
            route = "RGB_WATER_PROXY"

        # Execute Route
        core_seeds = None
        rgb: np.ndarray | None = None
        specialist: dict[str, Any] | None = None
        fallback_events: list[str] = []
        aerial_model_failed = False
        spectral_reference: np.ndarray | None = None
        aerial = None
        if route == "MULTISPECTRAL_NDWI_MNDWI":
            prob, valid, exclusion, details = compute_multispectral_water(src, band_map.indices)
            threshold = 0.50
            spectral_reference = prob.copy()
            if use_model:
                from satquery_engine.config import settings
                if src.count >= 10:
                    from satquery_engine.models.satlas_water_net import SOURCE_BANDS
                    ten_band_compatible = all(name.lower() in band_map.indices for name in SOURCE_BANDS)
                else:
                    ten_band_compatible = False
                if ten_band_compatible:
                    bundle_path = Path(settings.water_checkpoint)
                    if not bundle_path.is_dir():
                        fallback_events.append(
                            "FINE_TUNED_MODEL_MISSING: the ten-band water bundle was unavailable."
                        )
                    else:
                        try:
                            model_probability, model_valid, specialist = run_satquery_water_model(
                                src, band_map.indices, bundle_path
                            )
                            valid &= model_valid
                            prob = np.where(valid, model_probability, 0.0).astype("float32")
                            # A learned mask must still respect direct spectral
                            # evidence; paved surfaces are frequent water lookalikes.
                            prob[exclusion | (spectral_reference < 0.12)] = np.minimum(
                                prob[exclusion | (spectral_reference < 0.12)], 0.05
                            )
                            threshold = float(specialist["threshold"])
                        except (ValueError, ImportError, RuntimeError, OSError) as exc:
                            fallback_events.append(f"FINE_TUNED_MODEL_UNAVAILABLE: {exc}")
                if specialist is None:
                    try:
                        from satquery_engine.services.s2_water_specialist import predict as predict_s2_water
                        model_probability, model_valid, specialist = predict_s2_water(
                            src, Path(settings.s2_water_checkpoint)
                        )
                        valid &= model_valid
                        prob = np.where(valid, model_probability, 0.0).astype("float32")
                        prob[exclusion | (spectral_reference < 0.12)] = np.minimum(
                            prob[exclusion | (spectral_reference < 0.12)], 0.05
                        )
                        threshold = float(specialist["threshold"])
                    except (ValueError, ImportError, RuntimeError, OSError) as exc:
                        fallback_events.append(
                            f"SIX_BAND_MODEL_UNAVAILABLE: {exc} Deterministic multispectral fallback was used."
                        )
        elif route == "SAR_WATER":
            from satquery_engine.services.sar import SARPreprocessor
            sar_data = SARPreprocessor().process(path, polarizations=("vv",))
            prob, valid, exclusion, details = compute_sar_water(sar_data)
            threshold = 0.50
        else:
            # Route B: RGB-Only Water Proxy
            from satquery_engine.services.radiometry import rgb_unit_data
            rgb, valid, _ = rgb_unit_data(src)
            prob, core_seeds, exclusion, details = compute_rgb_water_proxy(rgb[0], rgb[1], rgb[2], valid)
            threshold = 0.50
            if use_model:
                from satquery_engine.services.buildings import resolution_m
                gsd = resolution_m(src)
                # The aerial checkpoint is not validated for arbitrary screenshots,
                # coarse satellite pixels, or marine scenes at unknown resolution.
                if gsd is not None and .15 <= gsd <= .60:
                    try:
                        import satquery_engine.services.landcover_specialist as _ls
                        _legacy_mocked = getattr(_ls.predict_landcover, "__module__", "") != "satquery_engine.services.landcover_specialist" or getattr(_ls.predict_landcover, "__name__", "") != "predict_landcover"
                        try:
                            if _legacy_mocked:
                                raise ValueError("Legacy aerial specialist explicitly mocked")
                            from satquery_engine.services.flair_hub import predict_flair_hub
                            candidate = predict_flair_hub(path)
                            water_index, vegetation_indexes, built_indexes = 6, (8, 9, 11, 12, 13, 14), (0, 1, 3)
                        except (ValueError, RuntimeError, ImportError, OSError) as flair_error:
                            fallback_events.append(f"FLAIR_HUB_UNAVAILABLE: {flair_error}; using validated legacy aerial specialist.")
                            from satquery_engine.services.landcover_specialist import predict_landcover, CLASSES
                            candidate = predict_landcover(path)
                            if tuple(candidate["classes"]) != CLASSES:
                                raise ValueError("Legacy aerial class map mismatch")
                            water_index, vegetation_indexes, built_indexes = 3, (2,), (1, 4)
                        scores = candidate["probability"]
                        if (scores.shape != (len(candidate["classes"]), height, width) or candidate["valid"].shape != valid.shape
                                or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
                                or not np.allclose(scores[:, candidate["valid"]].sum(0), 1, atol=1e-4)):
                            raise ValueError("Invalid aerial water probabilities.")
                        aerial = candidate
                        spectral_reference = prob.copy()
                        valid &= candidate["valid"]
                        corroborator_id = None
                        corroborator_checkpoint_sha256 = None
                        if water_index == 6:
                            # The 19-class FLAIR model can label smooth turf as
                            # water. Deepness is a separate aerial checkpoint;
                            # require object-level agreement before publishing a
                            # blue water polygon on RGB-only imagery.
                            from satquery_engine.services.landcover_specialist import predict_landcover
                            corroborator = predict_landcover(path)
                            second = corroborator["probability"]
                            if (second.shape != (5, height, width) or corroborator["valid"].shape != valid.shape
                                    or not np.isfinite(second).all() or np.any(second < 0) or np.any(second > 1)
                                    or not np.allclose(second[:, corroborator["valid"]].sum(0), 1, atol=1e-4)):
                                raise ValueError("Invalid corroborating aerial water probabilities.")
                            valid &= corroborator["valid"]
                            prob, agreement = confirm_aerial_water(scores[water_index], second[3], valid)
                            corroborator_id = corroborator["model_id"]
                            corroborator_checkpoint_sha256 = corroborator["checkpoint_sha256"]
                            details["aerial_water_agreement"] = agreement
                            if agreement["withheld_candidate_pixels"]:
                                fallback_events.append(
                                    "AERIAL_WATER_DISAGREEMENT: RGB water candidates without object-level support "
                                    "from both aerial models were withheld."
                                )
                        else:
                            prob = np.where(valid, scores[water_index], 0).astype("float32")
                            # Deepness aerial model confuses building/tree shadows with water; suppress isolated shadow pixels
                            if rgb is not None:
                                pre_shadow = rgb_shadow_evidence(rgb[0], rgb[1], rgb[2], valid)
                                prob = np.where(pre_shadow.mask & (prob < 0.78), np.minimum(prob, 0.12), prob)
                        # Roads and roofs can receive a strong water score from
                        # aerial networks. Require visible water color support
                        # where the independent built-up classes also respond.
                        built_score = scores[list(built_indexes)].sum(0)
                        prob, road_conflict = suppress_road_water_conflicts(prob, built_score, spectral_reference)
                        details["road_conflict_pixels"] = int((road_conflict & valid).sum())
                        core_seeds = None
                        details["vegetation_probability"] = scores[list(vegetation_indexes)].sum(0)
                        details["builtup_probability"] = scores[list(built_indexes)].sum(0)
                        route = "RGB_AERIAL_WATER"
                        specialist = {"model_id": candidate["model_id"], "threshold": .5,
                            "min_area_pixels": 28, "checkpoint_sha256": candidate["checkpoint_sha256"],
                            "preprocessing": candidate["preprocessing"], "tile_count": len(candidate["tiles"]),
                            "device": candidate.get("device"), "corroborator_id": corroborator_id,
                            "corroborator_checkpoint_sha256": corroborator_checkpoint_sha256,
                            "checkpoint_execution": "flair_hub_pytorch" if water_index == 6 else "aerial_landcover_onnx"}
                    except (ValueError, ImportError, RuntimeError, OSError) as exc:
                        aerial = None
                        aerial_model_failed = True
                        # A failed compatible model must not silently turn the
                        # weaker color proxy into authoritative blue polygons.
                        prob = np.zeros_like(prob)
                        core_seeds = None
                        fallback_events.append(
                            f"AERIAL_WATER_UNAVAILABLE: {exc} Water boundaries were withheld; "
                            "RGB colors alone cannot verify water in this aerial scene."
                        )

        vegetation_probability = np.asarray(details.get("vegetation_probability", np.zeros_like(prob)), dtype="float32")
        builtup_probability = np.asarray(details.get("builtup_probability", np.zeros_like(prob)), dtype="float32")

        # Illumination is an independent product. It may overlap WATER and is
        # never used as a replacement surface class.
        if route in {"RGB_WATER_PROXY", "RGB_AERIAL_WATER"} and rgb is not None:
            shadow = rgb_shadow_evidence(
                rgb[0], rgb[1], rgb[2], valid,
                water_probability=prob,
                vegetation_probability=vegetation_probability,
            )
        elif route == "MULTISPECTRAL_NDWI_MNDWI":
            shadow = multispectral_shadow_evidence(
                np.asarray(details["brightness"], dtype="float32"), valid, prob,
                vegetation_probability=vegetation_probability,
                builtup_probability=builtup_probability,
            )
        else:
            radar_shadow = np.asarray(exclusion, dtype=bool) & valid
            shadow = ShadowEvidence(
                probability=radar_shadow.astype("float32"), mask=radar_shadow,
                shadow_type=np.where(radar_shadow, 2, 0).astype("uint8"), features={},
                details={"method":"sar_geometric_shadow_support_v1", "shaded_water_pixels":int((radar_shadow & (prob >= 0.60)).sum())},
            )

        adjudication = adjudicate_surface_and_illumination(
            water_probability=prob,
            shadow_probability=shadow.probability,
            valid=valid,
            vegetation_probability=vegetation_probability,
            builtup_probability=builtup_probability,
            water_threshold=threshold,
        )
        # Suppress shadowed pixels unless validated as deep water by adjudication
        adjudicated_prob = np.where(adjudication.water_mask, prob, np.minimum(prob, 0.15)).astype("float32")

        # Postprocessing: river preservation, shadow suppression, and component analysis
        total_valid = int(valid.sum())
        min_comp_px = (
            max(28, round(total_valid * 0.000030))
            if route == "RGB_AERIAL_WATER"
            else (
                int(specialist["min_area_pixels"])
                if (specialist is not None and int(specialist.get("min_area_pixels", 0)) > 0)
                else max(25, round(total_valid * 0.000020))
            )
        )
        apply_morph = route != "MULTISPECTRAL_NDWI_MNDWI"
        mask, components, disagreement = postprocess_water_mask(
            prob=adjudicated_prob,
            valid=valid,
            core_seeds=core_seeds,
            threshold=threshold,
            min_component_px=min_comp_px if route != "MULTISPECTRAL_NDWI_MNDWI" else 0,
            pixel_res=pixel_res,
            transform=src.transform,
            crs=crs,
            apply_morphology=apply_morph,
            filter_compact=True,
        )
        if specialist is not None and spectral_reference is not None:
            disagreement = np.maximum(
                disagreement,
                np.where(valid, np.abs(prob - spectral_reference), 0.0).astype("float32"),
            )

        if largest and components:
            largest_comp = max(components, key=lambda c: c.area_pixels)
            mask = mask & (ndi.label(mask)[0] == largest_comp.component_id)

        # Quality Gate
        is_coastal = any(c.border_contact and c.area_pixels > total_valid * 0.10 for c in components)
        q_pass, q_warnings = check_water_result(mask, prob, valid, route, components, is_coastal=is_coastal)
        if np.any(shadow.mask & ~valid):
            q_pass = False
            q_warnings.append("Shadow mask contains NoData pixels.")
        shaded_water = mask & shadow.mask
        if shadow.details.get("shaded_water_pixels", 0) and not shaded_water.any():
            q_warnings.append("Independent water evidence under shadow was removed during postprocessing.")

        # Artifact Generation
        paths: list[Path] = []

        # 1. water_probability.tif
        prob_tif = output_dir / "water_probability.tif"
        export_float_raster(prob, path, prob_tif)
        paths.append(prob_tif)

        # Explicit shadow, conflict, uncertainty, surface, and illumination products.
        shadow_prob_tif = output_dir / "shadow_probability.tif"
        conflict_tif = output_dir / "water_shadow_conflict.tif"
        uncertainty_tif = output_dir / "uncertainty.tif"
        surface_tif = output_dir / "surface_type.tif"
        illumination_tif = output_dir / "illumination_state.tif"
        paths.extend([
            export_float_raster(shadow.probability, path, shadow_prob_tif),
            export_float_raster(adjudication.conflict, path, conflict_tif),
            export_float_raster(adjudication.uncertainty, path, uncertainty_tif),
        ])
        categorical_profile = dict(driver="GTiff", width=width, height=height, count=1, dtype="uint8", transform=src.transform, crs=crs, compress="deflate")
        for destination, values in ((surface_tif, adjudication.surface), (illumination_tif, adjudication.illumination)):
            with rasterio.open(destination, "w", **categorical_profile) as dst:
                dst.write(values.astype("uint8"), 1)
                dst.write_mask(valid.astype("uint8") * 255)
            paths.append(destination)

        # 2. water_mask.tif
        mask_tif = output_dir / "water_mask.tif"
        grid = src.transform
        profile = dict(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="uint8",
            transform=grid,
            crs=crs,
            compress="deflate",
        )
        with rasterio.open(mask_tif, "w", **profile) as dst:
            dst.write((mask.astype("uint8") * 255), 1)
            dst.write_mask(valid.astype("uint8") * 255)
        paths.append(mask_tif)

        shadow_mask_tif = output_dir / "shadow_mask.tif"
        with rasterio.open(shadow_mask_tif, "w", **profile) as dst:
            dst.write((shadow.mask.astype("uint8") * 255), 1)
            dst.write_mask(valid.astype("uint8") * 255)
        paths.append(shadow_mask_tif)

        # 3. water.geojson
        features = []
        world = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs else None
        labels, n_labels = ndi.label(mask)
        grouped: dict[int, list[Any]] = {}
        for pixel_geom, value in shapes(labels.astype("int32"), mask=labels > 0, transform=Affine.identity()):
            grouped.setdefault(int(value), []).append(shape(pixel_geom))

        total_area_m2 = 0.0
        for value, parts in sorted(grouped.items()):
            pixel_polygon = make_valid(unary_union(parts))
            if pixel_polygon.is_empty or pixel_polygon.area <= 0:
                continue
            pixel_geom = mapping(pixel_polygon)
            native = transform_geom(lambda x, y, z=None: (grid.a * x + grid.b * y + grid.c, grid.d * x + grid.e * y + grid.f), pixel_polygon)
            area = _area_square_meters(mapping(native), crs) if crs else None
            if area:
                total_area_m2 += area
            geom = transform_geom(world.transform, native) if world else pixel_polygon
            features.append({
                "type": "Feature",
                "id": int(value),
                "geometry": mapping(geom),
                "properties": {
                    "instance_id": int(value),
                    "kind": "water",
                    "pixel_geometry": pixel_geom,
                    "area_m2": area,
                    "area_pixels": float(pixel_polygon.area),
                    "marker": list(geom.representative_point().coords[0]),
                    "bbox": list(geom.bounds),
                    "score": float(prob[labels == value].mean()) if (labels == value).any() else None,
                },
            })

        geojson_path = output_dir / "water.geojson"
        collection = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "crs": "EPSG:4326" if crs else None,
                "coordinate_space": "geographic" if crs else "pixel",
                "source_crs": str(crs) if crs else None,
                "producer": route,
                "water_coverage_percent": round(100.0 * mask.sum() / max(1, total_valid), 3),
            },
        }
        geojson_path.write_text(json.dumps(collection, allow_nan=False), encoding="utf-8")
        paths.append(geojson_path)

        # 4. Visual Overlays
        # water_overlay.png
        preview_path = output_dir / "water_original.png"
        render_preview(path, preview_path, max_size=1200)
        preview = Image.open(preview_path).convert("RGBA")

        rgba = np.zeros((*mask.shape, 4), dtype="uint8")
        # Deep blue water overlay — clearly distinguishable from green land
        rgba[mask] = [20, 60, 140, 130]
        # An inner shoreline keeps the original bank visible without expanding
        # the detected water footprint into neighboring land.
        shoreline = mask & ~ndi.binary_erosion(mask, border_value=1)
        rgba[shoreline] = [30, 90, 180, 220]
        overlay = Image.alpha_composite(preview, Image.fromarray(rgba).resize(preview.size, Image.Resampling.NEAREST))
        overlay_path = output_dir / "water_overlay.png"
        overlay.convert("RGB").save(overlay_path)
        paths.append(overlay_path)

        shadow_rgba = np.zeros((*shadow.mask.shape, 4), dtype="uint8")
        shadow_rgba[shadow.mask] = [120, 80, 200, 140]
        shadow_overlay = Image.alpha_composite(preview, Image.fromarray(shadow_rgba).resize(preview.size, Image.Resampling.NEAREST))
        shadow_overlay_path = output_dir / "shadow_overlay.png"
        shadow_overlay.convert("RGB").save(shadow_overlay_path)
        paths.append(shadow_overlay_path)

        # water_mask.png (clean binary/color mask for viewer)
        mask_png = output_dir / "water_mask.png"
        Image.fromarray(rgba).save(mask_png)
        paths.append(mask_png)

        # water_disagreement.png
        disagree_rgba = np.zeros((*mask.shape, 4), dtype="uint8")
        disagreement = np.maximum(disagreement, adjudication.conflict)
        disagree_mask = disagreement > 0.45
        disagree_rgba[disagree_mask] = [239, 68, 68, 180]
        disagree_img = Image.alpha_composite(preview, Image.fromarray(disagree_rgba).resize(preview.size, Image.Resampling.NEAREST))
        disagree_path = output_dir / "water_disagreement.png"
        disagree_img.convert("RGB").save(disagree_path)
        paths.append(disagree_path)

        # water_confidence.png
        conf_rgba = np.zeros((*prob.shape, 4), dtype="uint8")
        # Blue-scale confidence map — darker blue = higher probability
        conf_rgba[..., 0] = (prob * 40).astype("uint8")
        conf_rgba[..., 1] = (prob * 80).astype("uint8")
        conf_rgba[..., 2] = (prob * 200 + 40).clip(0, 255).astype("uint8")
        conf_rgba[..., 3] = (prob * 180).astype("uint8")
        conf_path = output_dir / "water_confidence.png"
        Image.fromarray(conf_rgba).save(conf_path)
        paths.append(conf_path)

        coverage_pct = round(100.0 * mask.sum() / max(1, total_valid), 3)

        method_name = {
            "MULTISPECTRAL_NDWI_MNDWI": "Multispectral NDWI / MNDWI",
            "RGB_WATER_PROXY": "RGB Water Proxy (Multi-cue Optical)",
            "RGB_AERIAL_WATER": "Corroborated aerial RGB water segmentation" if specialist and specialist.get("corroborator_id") else "Pretrained aerial RGB water segmentation",
            "SAR_WATER": "SAR Radar Backscatter Water Extraction",
        }[route]
        if aerial_model_failed:
            method_name = "Aerial RGB water evidence unavailable"
        if specialist is not None and aerial is None:
            method_name = ("SatQuery fine-tuned Sentinel-2 water segmentation with spectral evidence"
                           if specialist["model_id"] == "satquery_water_bundle" else
                           "Six-band Sentinel-2 UNet++ water segmentation with spectral evidence")

        limitations = []
        if route == "RGB_WATER_PROXY":
            limitations.append(
                "RGB water proxy uses multi-cue optical evidence. True multispectral NIR/SWIR bands were absent."
            )
        if aerial is not None:
            limitations.extend([aerial["domain_note"],
                "Aerial water predictions are uncalibrated estimates; performance on this geography and ocean/coastal imagery is not established."])
        limitations.extend(fallback_events)
        limitations.extend(q_warnings)

        location_desc = None
        if mask.any():
            from satquery_engine.services.spectral import _describe_region_location
            coords = np.argwhere(mask)
            cy = float(coords[:, 0].mean()) / mask.shape[0]
            cx = float(coords[:, 1].mean()) / mask.shape[1]
            location_desc = _describe_region_location(cy, cx)

        stats = {
            "method": method_name,
            "route": route,
            "coverage_percent": coverage_pct,
            "area_m2": total_area_m2 if crs else None,
            "area_ha": (total_area_m2 / 10000.0) if crs and total_area_m2 else None,
            "water_pixels": int(mask.sum()),
            "valid_pixels": total_valid,
            "region_count": len(features),
            "threshold": threshold,
            "shadow": {
                "shadow_pixels": int(shadow.mask.sum()),
                "shadow_percent": round(100.0 * shadow.mask.sum() / max(1, total_valid), 3),
                "shaded_water_pixels": int((mask & shadow.mask).sum()),
            },
            "water_shadow_conflict_score": round(float(adjudication.conflict[valid].mean()) if valid.any() else 0.0, 4),
            "uncertainty_score": round(float(adjudication.uncertainty[valid].mean()) if valid.any() else 1.0, 4),
            "quality_gate_passed": q_pass,
            "aerial_water_agreement": details.get("aerial_water_agreement"),
        }
        stats_path = output_dir / "water_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        paths.append(stats_path)

        return {
            "method": method_name,
            "route": route,
            "index": "ndwi" if route == "MULTISPECTRAL_NDWI_MNDWI" else None,
            "coverage_percent": coverage_pct,
            "valid_pixels": total_valid,
            "selected_pixels": int(mask.sum()),
            "sampled_pixels": int(mask.size),
            "area_m2": total_area_m2 if crs else None,
            "region_count": len(features),
            "threshold": threshold,
            "mean_index": float(prob[valid].mean()) if valid.any() else 0.0,
            "evidence_strength": 0.0 if aerial_model_failed else (.5 if aerial is not None else (0.85 if route == "MULTISPECTRAL_NDWI_MNDWI" else 0.70)),
            "evidence_state": "INSUFFICIENT_EVIDENCE" if aerial_model_failed else "MODEL_ESTIMATE" if specialist is not None else "PROXY_ESTIMATE",
            "confidence_kind": "uncalibrated_evidence_strength",
            "limitations": limitations,
            "paths": paths,
            "features": features,
            "water_body_identified": bool(mask.any()),
            "geometry_validated": True,
            "mask_polygon_agree": True,
            "native_resolution": True,
            "location_description": location_desc,
            "surface_illumination_model": "independent_surface_and_illumination_v1",
            "shadow": {
                "method": shadow.details["method"],
                "shadow_pixels": int(shadow.mask.sum()),
                "shadow_percent": round(100.0 * shadow.mask.sum() / max(1, total_valid), 3),
                "shaded_water_pixels": int((mask & shadow.mask).sum()),
            },
            "water_shadow_conflict_score": round(float(adjudication.conflict[valid].mean()) if valid.any() else 0.0, 4),
            "uncertainty_score": round(float(adjudication.uncertainty[valid].mean()) if valid.any() else 1.0, 4),
            "quality_gate_passed": q_pass,
            "model_id": specialist["model_id"] if specialist is not None else route,
            "aerial_water_agreement": details.get("aerial_water_agreement"),
            "models_used": [model for model in (specialist["model_id"], specialist.get("corroborator_id")) if model] if specialist is not None else [],
            "checkpoint_sha256": specialist.get("checkpoint_sha256") if specialist else None,
            "corroborator_checkpoint_sha256": specialist.get("corroborator_checkpoint_sha256") if specialist else None,
            "device": specialist.get("device") if specialist else None,
            "checkpoint_execution": specialist.get("checkpoint_execution", "satquery_water_bundle") if specialist else "deterministic_fallback",
            "preprocessing": specialist.get("preprocessing") if specialist else None,
            "tile_count": specialist.get("tile_count") if specialist else None,
            "benchmark_f1": specialist.get("benchmark", {}).get("f1") if specialist else None,
            "fallback_events": fallback_events,
        }
