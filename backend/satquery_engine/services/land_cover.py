"""Compose native spatial evidence; never impersonate a scene classifier with heuristics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from scipy import ndimage as ndi

from satquery_engine.services.bands import detect_band_map, BandMap
from satquery_engine.services.spatial_outputs import export_labels


# ---------------------------------------------------------------------------
# BigEarthNet v2.0 (19 Corine Land Cover Taxonomy)
# ---------------------------------------------------------------------------

BIGEARTHNET_19_CLASSES: list[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Official v0.2.0 channel orders
BIGEARTHNET_S2_CHANNELS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
BIGEARTHNET_S1_CHANNELS = ["VV", "VH"]
BIGEARTHNET_ALL_CHANNELS = ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def validate_bigearthnet_channels(
    band_names: list[str],
    variant: str = "S2",
) -> tuple[bool, str]:
    """Validate that raster bands match the pinned BigEarthNet v0.2.0 channel order."""
    import re
    upper_names = [re.sub(r"^B0+(\d+)$",r"B\1",b.upper().strip()) for b in band_names]
    required = {"S2":BIGEARTHNET_S2_CHANNELS,"S1":BIGEARTHNET_S1_CHANNELS,"ALL":BIGEARTHNET_ALL_CHANNELS}.get(variant)
    if required is None:
        return False, f"Unknown BigEarthNet variant: {variant}."
    expected = [re.sub(r"^B0+(\d+)$",r"B\1",b) for b in required]
    if upper_names != expected:
        return False, "BigEarthNet channel order must match the complete, unique declared input specification."
    return True, "Channel names and order match; sensor, radiometry, resolution and pretrained runtime still require validation."


# ---------------------------------------------------------------------------
# BigEarthNet Scene Witness
# ---------------------------------------------------------------------------

@dataclass
class BigEarthNetWitnessResult:
    available: bool
    model_name: str
    variant: str
    class_probabilities: dict[str, float]
    top_classes: list[tuple[str, float]]
    water_evidence: float
    urban_evidence: float
    forest_evidence: float
    agriculture_evidence: float
    notes: str


def compute_bigearthnet_witness(
    band_data: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    variant: str = "S2",
) -> BigEarthNetWitnessResult:
    """Abstain until a real, compatible pretrained classifier has executed.

    Spectral rules are not BigEarthNet inference and cannot supply class probabilities.
    """
    return BigEarthNetWitnessResult(
        available=False, model_name="", variant=variant,
        class_probabilities={}, top_classes=[], water_evidence=0.0,
        urban_evidence=0.0, forest_evidence=0.0, agriculture_evidence=0.0,
        notes="BigEarthNet pretrained inference is unavailable; no class probabilities were generated.",
    )


def check_land_result(
    breakdown: dict[str, Any],
    witness: BigEarthNetWitnessResult,
    band_info: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Context-aware sanity check comparing spatial coverage to scene witness."""
    warnings: list[str] = []
    water_pct = breakdown.get("water", {}).get("percent", 0.0)
    urban_pct = breakdown.get("built_up", {}).get("percent", 0.0)
    vegetation_pct = breakdown.get("vegetation", {}).get("percent", 0.0)

    # 1. Review flag: high spatial water but weak BigEarthNet water evidence
    if witness.available and water_pct > 50.0 and witness.water_evidence < 0.12:
        warnings.append(
            f"WATER_RESULT_REVIEW: Water mask occupies {water_pct:.1f}% of scene, "
            f"but BigEarthNet scene witness indicates low water probability ({witness.water_evidence:.2f})."
        )

    # 2. Review flag: high spatial built-up but weak BigEarthNet urban evidence
    if witness.available and urban_pct > 60.0 and witness.urban_evidence < 0.10:
        warnings.append(
            f"BUILT_UP_RESULT_REVIEW: Built-up mask occupies {urban_pct:.1f}% of scene, "
            f"but BigEarthNet scene witness indicates low urban probability ({witness.urban_evidence:.2f})."
        )

    # 3. Pixel-count-based sanity checks — run regardless of BigEarthNet availability.
    # These catch impossible single-class dominance with no supporting evidence from other
    # specialists (e.g. water claims 70% but vegetation and buildings combined are < 5%).
    if water_pct > 65.0 and (vegetation_pct + urban_pct) < 5.0:
        warnings.append(
            f"WATER_COVERAGE_REVIEW: Water mask covers {water_pct:.1f}% of scene with "
            f"very low vegetation ({vegetation_pct:.1f}%) and built-up ({urban_pct:.1f}%); "
            "results should be manually reviewed (coastal or flooded scene?)."
        )
    if vegetation_pct > 85.0 and (water_pct + urban_pct) < 2.0:
        warnings.append(
            f"VEGETATION_REVIEW: Vegetation covers {vegetation_pct:.1f}% of scene with "
            "negligible water and built-up; dense canopy scenes may over-claim vegetation."
        )

    # 4. Sensor & band check
    if not band_info.get("has_nir"):
        warnings.append("Vegetation and water classified using RGB optical proxy; true multispectral bands unavailable.")

    passed = len(warnings) == 0
    return passed, warnings


# ---------------------------------------------------------------------------
# Composite Pixel Land-Cover Pipeline
# ---------------------------------------------------------------------------

def classify_land_cover_composite(
    image_path: Path, output_dir: Path, query: str = "", max_size: int | None = None,
    water_result: dict | None = None, building_result: dict | None = None,
    vegetation_result: dict | None = None,
) -> dict[str, Any]:
    """Compose a native-grid map from the final specialist masks.

    Residual pixels are unknown. RGB brightness does not establish built-up land,
    forest type, agriculture or bare soil. Scene classification cannot supply areas.
    """
    from satquery_engine.services.water_engine import execute_water_pipeline
    from satquery_engine.services.measurements import measure_cover
    from satquery_engine.services.radiometry import rgb_unit_data
    output_dir.mkdir(parents=True, exist_ok=True)
    limitations = []
    with rasterio.open(image_path) as src:
        if src.width * src.height > 32_000_000:
            raise ValueError("Land-cover analysis supports up to 32 million native pixels.")
        grid, crs = src.transform, src.crs
        valid = src.dataset_mask() > 0
        dims = (src.height, src.width)
        band_map = detect_band_map(src)
        try:
            rgb, rgb_valid, _ = rgb_unit_data(src)
            valid &= rgb_valid
        except ValueError:
            rgb = None
        if any(p in band_map.indices for p in ("vv", "vh", "hh", "hv")):
            raise ValueError("A validated SAR land-cover classifier is unavailable; optical proxies cannot analyze radar imagery.")
    if water_result is None:
        try:
            water_result = execute_water_pipeline(image_path, output_dir / "water_evidence")
        except ValueError as exc:
            limitations.append(str(exc))
    if vegetation_result is None:
        try:
            vegetation_result = measure_cover(image_path, output_dir / "vegetation_evidence", "vegetation")
        except ValueError as exc:
            limitations.append(str(exc))

    learned = None
    if rgb is not None:
        try:
            from satquery_engine.services.landcover_specialist import compatibility, predict_landcover
            compatible, note = compatibility(image_path)
            if not compatible:
                raise ValueError(note)
            try:
                # Select for dense land classes, independently from water's
                # specialist. The shared-class aerial benchmark favors Deepness.
                learned = predict_landcover(image_path)
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                limitations.append(f"Deepness land-cover model unavailable; trying FLAIR-HUB: {exc}")
                from satquery_engine.services.flair_hub import predict_flair_hub
                learned = predict_flair_hub(image_path)
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            limitations.append(f"Aerial land-cover specialist unavailable: {exc}")

    def final_mask(result, suffix):
        if result is None:
            return np.zeros(dims, dtype=bool)
        paths = [p for p in result.get("paths", []) if p.name == suffix]
        if not paths:
            raise ValueError("A canonical specialist mask is missing.")
        with rasterio.open(paths[0]) as mask_src:
            if (mask_src.height, mask_src.width) != dims or mask_src.transform != grid or mask_src.crs != crs:
                raise ValueError("Land-cover evidence does not align with the source raster.")
            mask = mask_src.read(1) > 0
            if np.any(mask & ~valid):
                raise ValueError("Specialist evidence contains pixels invalid in the land-cover grid.")
            return mask

    water = final_mask(water_result, "water_mask.tif")
    vegetation = final_mask(vegetation_result, "vegetation_labels.tif")
    buildings = final_mask(building_result, "buildings_labels.tif")
    model_classes = None
    is_flair_hub = False
    model_probability = None
    model_confidence = None
    model_margin = None
    class_statistics = {}
    model_ids = list((water_result or {}).get("models_used", []))
    probability_path = None
    if learned is not None:
        from satquery_engine.services.landcover_specialist import CLASSES
        from satquery_engine.services.flair_hub import MODEL_ID as FLAIR_HUB_ID
        is_flair_hub = learned["model_id"] == FLAIR_HUB_ID
        if not is_flair_hub and tuple(learned["classes"]) != CLASSES:
            raise ValueError("Unexpected land-cover class order; classification withheld.")
        nclasses = 19 if is_flair_hub else 5
        if learned["probability"].shape != (nclasses, *dims) or learned["valid"].shape != dims:
            raise ValueError("Pretrained land-cover output is not aligned to the source grid.")
        if np.any(learned["valid"] & ~valid):
            raise ValueError("Pretrained land-cover output contains invalid source pixels.")
        model_probability = learned["probability"]
        if (not np.isfinite(model_probability).all() or np.any(model_probability < 0)
                or np.any(model_probability > 1)
                or not np.allclose(model_probability[:, valid].sum(0), 1, atol=1e-4)):
            raise ValueError("Invalid land-cover probabilities; classification withheld.")
        ordered = np.sort(model_probability, axis=0)
        model_confidence = ordered[-1]
        model_margin = ordered[-1] - ordered[-2]
        confident = (model_confidence >= .50) & (model_margin >= .10) & learned["valid"] & valid
        raw_classes = model_probability.argmax(0)
        model_classes = np.where(confident, raw_classes, -1)
        if is_flair_hub:
            model_classes[np.isin(model_classes, [15, 16, 17, 18])] = -1
        class_statistics = {
            name: {"class_id": i, "pixels": int(((model_classes == i) & valid).sum()),
                   "percent_valid": float(100.0 * ((model_classes == i) & valid).sum() / max(int(valid.sum()), 1))}
            for i, name in enumerate(learned["classes"]) if not (is_flair_hub and i >= 15)
        }
        model_ids.append(learned["model_id"])
        if building_result is None:
            buildings = np.isin(model_classes, [0, 1]) & valid if is_flair_hub else (model_classes == 1) & valid
        # RGB vegetation rules are weaker than the aerial specialist. Retain
        # index-based vegetation only when a declared NIR band exists.
        if "nir" not in band_map.indices:
            vegetation = np.isin(model_classes, [8, 9, 11, 12, 13, 14]) & valid if is_flair_hub else (model_classes == 2) & valid
        # A supplied canonical water mask is authoritative across answers and maps.
        # The aerial model can fill in water only if that specialist was unavailable.
        learned_water = (model_classes == 6) & valid if is_flair_hub else (model_classes == 3) & valid
        if water_result is None:
            water = learned_water
        elif np.any(learned_water & ~water):
            limitations.append("WATER_MODEL_DISAGREEMENT: aerial water predictions outside the canonical water mask were withheld.")
        probability_path = output_dir / "landcover_probability.tif"
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff", count=nclasses, dtype="float32", nodata=np.nan, compress="deflate")
            profile.pop("photometric", None)
            with rasterio.open(probability_path, "w", **profile) as dst:
                dst.write(np.where(valid[None], model_probability, np.nan).astype("float32"))
                dst.write_mask(valid.astype("uint8") * 255)
                for index, name in enumerate(learned["classes"], 1):
                    dst.set_band_description(index, name)
    supplemental_flair = None
    from satquery_engine.services.landcover_specialist import MODEL_ID as DEEPNESS_ID
    if learned is not None and learned["model_id"] == DEEPNESS_ID:
        try:
            from satquery_engine.services.flair_hub import predict_flair_hub
            supplemental_flair = predict_flair_hub(image_path)
            flair_prob = supplemental_flair["probability"]
            if (flair_prob.shape != (19, *dims) or supplemental_flair["valid"].shape != dims
                    or np.any(supplemental_flair["valid"] & ~valid)
                    or not np.isfinite(flair_prob).all() or np.any(flair_prob < 0) or np.any(flair_prob > 1)
                    or not np.allclose(flair_prob[:, supplemental_flair["valid"]].sum(0), 1, atol=1e-4)):
                raise ValueError("FLAIR-HUB land probabilities are invalid or misaligned.")
            model_ids.append(supplemental_flair["model_id"])
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            supplemental_flair = None
            limitations.append(f"FLAIR-HUB supplemental land classes unavailable: {exc}")

    conflicts = (water & buildings) | (vegetation & buildings) | (water & vegetation)
    # Preserve canonical water exactly. Disputed non-water pixels remain unknown.
    built_surface = (model_classes == 3) & valid if is_flair_hub else np.zeros(dims, bool)
    built = (buildings | built_surface) & ~water & ~vegetation
    veg = vegetation & ~water & ~buildings
    road = ((model_classes == 4) & valid & ~water & ~buildings & ~vegetation) if model_classes is not None and not is_flair_hub else np.zeros(dims, bool)
    bare = np.isin(model_classes, [4, 5]) & valid if is_flair_hub else np.zeros(dims, bool)
    agriculture = np.isin(model_classes, [9, 10, 11]) & valid if is_flair_hub else np.zeros(dims, bool)
    snow = (model_classes == 7) & valid if is_flair_hub else np.zeros(dims, bool)
    pool = (model_classes == 2) & valid if is_flair_hub else np.zeros(dims, bool)
    labels = np.zeros(dims, dtype="int32")
    labels[valid] = 4
    labels[bare & ~water & ~buildings] = 7
    labels[agriculture & ~water & ~buildings] = 8
    labels[snow & ~water & ~buildings] = 9
    labels[pool & ~water & ~buildings] = 10
    labels[veg & ~agriculture] = 2
    labels[built] = 3
    labels[water] = 1
    if model_classes is not None:
        labels[veg & (np.isin(model_classes, [12, 13, 14]) if is_flair_hub else (model_classes == 2))] = 5
        labels[road] = 6
    supplemental_pixels = {}
    if supplemental_flair is not None:
        flair_prob = supplemental_flair["probability"]
        flair_ordered = np.sort(flair_prob, axis=0)
        flair_confidence = flair_ordered[-1]
        flair_margin = flair_ordered[-1] - flair_ordered[-2]
        flair_class = flair_prob.argmax(0)
        eligible = ((labels == 4) & supplemental_flair["valid"] & valid
                    & (flair_confidence >= .62) & (flair_margin >= .12))
        for class_id, class_names in ((7, (4, 5)), (2, (8,)), (8, (9, 10, 11)),
                                      (5, (12, 13, 14))):
            selected = eligible & np.isin(flair_class, class_names)
            labels[selected] = class_id
            supplemental_pixels[{7:"bare_pervious", 2:"vegetation", 8:"agriculture", 5:"woodland"}[class_id]] = int(selected.sum())
            if model_confidence is not None:
                model_confidence[selected] = flair_confidence[selected]
        limitations.append(
            "Bare soil, pervious ground, grass and agricultural classes added only where the "
            "supplemental FLAIR-HUB aerial model is confident; transfer accuracy is unmeasured."
        )
    if not valid.any():
        raise ValueError("No valid land-cover pixels are available.")
    # Remove isolated model speckles without expanding boundaries or assigning
    # neighboring land a guessed class. Canonical water/buildings are preserved.
    removed_speckles = 0
    for class_id in (2, 5, 6, 7, 8, 9, 10):
        components, _ = ndi.label(labels == class_id)
        sizes = np.bincount(components.ravel())
        tiny = sizes < 9
        tiny[0] = False
        noise = tiny[components]
        removed_speckles += int(noise.sum())
        labels[noise] = 4
    spatial = export_labels(labels, image_path, output_dir, "land_cover", valid_mask=valid)
    names = {1:"water", 2:"vegetation", 3:"built_up", 4:"unknown", 5:"woodland", 6:"road",
             7:"bare_pervious", 8:"agriculture", 9:"snow", 10:"swimming_pool"}
    # Human-friendly color palette: each class is clearly distinguishable
    # Water = deep blue, Vegetation/Grass = bright green, Buildings = warm red,
    # Unknown/Land = dark yellow/sandy, Woodland = dark green, Road = bright yellow,
    # Bare soil = tan/brown, Agriculture = light olive, Snow = white, Pool = turquoise
    colors = {1:(20, 60, 140), 2:(34, 197, 94), 3:(220, 53, 69), 4:(194, 155, 56), 5:(22, 101, 52), 6:(245, 200, 30),
              7:(166, 130, 80), 8:(124, 179, 66), 9:(240, 248, 255), 10:(6, 182, 212)}
    scores = {"water":water_result, "vegetation":vegetation_result, "built_up":building_result}
    breakdown = {}
    total = int(valid.sum())
    for ident, name in names.items():
        count = int((labels == ident).sum())
        area = sum(f["properties"]["area_m2"] or 0 for f in spatial["features"] if f["id"] == ident) if crs else None
        specialist_strength = float(model_confidence[labels == ident].mean()) if model_confidence is not None and count and name != "unknown" else 0.0
        strength = float((scores.get(name) or {}).get("evidence_strength", specialist_strength))
        breakdown[name] = {"pixels":count, "percent":100.0*count/total, "area_m2":area,
            "confidence":min(strength, .5), "status":"unclassified" if name == "unknown" else "estimated",
            "color":"#%02x%02x%02x" % colors[ident]}
    for feature in spatial["features"]:
        feature["properties"]["class"] = names[feature["id"]]
        feature["properties"]["color"] = breakdown[names[feature["id"]]]["color"]
    geojson = output_dir / "land_cover.geojson"
    collection = json.loads(geojson.read_text()); collection["features"] = spatial["features"]
    geojson.write_text(json.dumps(collection, allow_nan=False))
    # Replace the generic binary rendering with the actual class palette.
    rgba = np.zeros((*dims,4), dtype="uint8")
    for ident, color in colors.items():
        # Unclassified surface is visibly amber, but distinct from the tan
        # bare-soil class. Transparency preserves the source image for review.
        rgba[labels == ident] = [*color, 100 if ident == 4 else 140]
    Image.fromarray(rgba).save(output_dir / "land_cover_mask.png")
    preview = Image.open(output_dir / "land_cover_original.png").convert("RGBA")
    Image.alpha_composite(preview, Image.fromarray(rgba).resize(preview.size, Image.Resampling.NEAREST)).convert("RGB").save(output_dir / "land_cover_overlay.png")
    # A land-only view must not paint water or unresolved pixels as land.
    land_mask = valid & np.isin(labels, [2, 3, 5, 6, 7, 8, 9])
    land_rgba = rgba.copy()
    land_rgba[~land_mask] = 0
    land_overlay = output_dir / "land_only_overlay.png"
    Image.alpha_composite(preview, Image.fromarray(land_rgba).resize(preview.size, Image.Resampling.NEAREST)).convert("RGB").save(land_overlay)
    land_mask_path = output_dir / "land_only_mask.tif"
    with rasterio.open(land_mask_path, "w", driver="GTiff", width=dims[1], height=dims[0], count=1,
                       dtype="uint8", crs=crs, transform=grid, nodata=None, compress="deflate") as dst:
        dst.write(land_mask.astype("uint8"), 1)
        dst.write_mask(valid.astype("uint8") * 255)
    spatial["paths"].extend([land_overlay, land_mask_path])
    map_path = output_dir / "land_cover_map.tif"
    with rasterio.open(image_path) as src:
        with rasterio.open(map_path, "w", driver="GTiff", width=dims[1], height=dims[0], count=1,
                           dtype="uint8", crs=crs, transform=grid, nodata=0, compress="deflate") as dst:
            dst.write(labels.astype("uint8"), 1); dst.write_mask(valid.astype("uint8")*255)
            dst.update_tags(classes=json.dumps(names))
    limitations += ["Land classes are estimates from available spatial evidence. Unknown includes land whose type cannot be established.",
                    "BigEarthNet scene-level inference is unavailable; dense aerial classes come from the spatial land-cover specialist when compatible.",
                    "Built-up includes FLAIR impervious surface when compatible; it is not a count of buildings." if is_flair_hub else "Built-up coverage here measures non-disputed building footprints, not all roads or paved surfaces."]
    if learned is not None:
        limitations.append(learned["domain_note"])
    if conflicts.any():
        limitations.append("LAND_COVER_DISAGREEMENT: specialist masks overlap; canonical water is retained and conflicting non-water pixels remain unknown.")
    if crs is None:
        limitations.append("Area cannot be calculated because this image does not contain a reliable geographic scale.")
    veg_mode = (vegetation_result or {}).get("method", "unavailable")
    known = {k:v for k,v in breakdown.items() if k != "unknown"}
    dominant = max(breakdown, key=lambda k:breakdown[k]["pixels"])
    summary = "Estimated image coverage: " + ", ".join(f"{v['percent']:.2f}% {k.replace('_',' ')}" for k,v in breakdown.items()) + "."
    result = {**spatial, "method":"Aerial land-cover specialist with deterministic evidence fusion" if learned is not None else "Specialist mask composition", "target":"land_cover", "summary":summary,
        "breakdown":breakdown, "dominant_class":dominant, "valid_pixels":total,
        "coverage_percent":100.0, "water_percent":breakdown["water"]["percent"],
        "classified_land_percent":100.0 * int(land_mask.sum()) / total,
        "classified_land_pixels":int(land_mask.sum()),
        "noise_pixels_withheld":removed_speckles,
        "vegetation_percent":breakdown["vegetation"]["percent"] + breakdown["woodland"]["percent"], "built_up_percent":breakdown["built_up"]["percent"],
        "vegetation_mode":veg_mode, "built_up_mode":"BUILDING_FOOTPRINTS", "native_resolution":True,
        "evidence_strength":min([v["confidence"] for v in known.values() if v["pixels"]] or [0.0]),
        "confidence_kind":"uncalibrated_evidence_strength", "quality_gate_passed":True,
        "quality_warnings":limitations, "limitations":limitations, "disputed_pixels":int(conflicts.sum()),
        "evidence_state":"DISAGREEMENT" if conflicts.any() else "MODEL_ESTIMATE" if learned is not None else "INSUFFICIENT_EVIDENCE",
        "bigearthnet_witness":{"available":False,"model":None,"class_probabilities":{}},
        "trained_class_statistics": class_statistics,
        "supplemental_flair_pixels": supplemental_pixels,
        "models_used":list(dict.fromkeys(model_ids)),
        "mask_path":output_dir / "land_cover_mask.png", "geojson_path":geojson}
    result["paths"].append(map_path)
    if probability_path is not None:
        result["paths"].append(probability_path)
        result["checkpoint_sha256"] = learned["checkpoint_sha256"]
        result["preprocessing"] = learned["preprocessing"]
        result["device"] = learned.get("device")
    stats = output_dir / "land_cover_stats.json"
    stats.write_text(json.dumps({k:v for k,v in result.items() if k not in {"paths","features","mask_path","geojson_path"}}, indent=2, allow_nan=False))
    result["paths"].append(stats)
    return result
