from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer

from .fusion import aggregate_landcover, fuse
from .geoio import building_geojson, write_raster
from .models import ModelCache
from .postprocess import filter_binary, separate_buildings
from .preprocess import normalized_rgb, rgb_unit, spectral_features, water_spectral_support
from .registry import ModelRegistry
from .tiling import blended_predict


@dataclass(frozen=True)
class InferenceOptions:
    device: str | None = None
    mixed_precision: bool = True
    require_all_models: bool = False
    building_overrides_landcover: bool = True
    deterministic: bool = True


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(registry: ModelRegistry, spec) -> dict[str, Any] | None:
    relative = spec.raw.get("metrics_file")
    path = (registry.root / relative).resolve() if relative else None
    return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None


def _band_names(src, override: list[str] | None) -> list[str]:
    if override:
        if len(override) != src.count:
            raise ValueError(f"--band-order contains {len(override)} names but raster has {src.count} bands.")
        return override
    descriptions = [item for item in src.descriptions]
    return descriptions if all(descriptions) else []


def _set_deterministic(enabled: bool) -> None:
    if not enabled:
        return
    np.random.seed(20260922)
    try:
        import torch
        torch.manual_seed(20260922)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260922)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _resolution_metres(width: int, height: int, transform, crs) -> float | None:
    if not crs:
        return None
    parsed = CRS.from_user_input(crs)
    if parsed.is_projected:
        factor = parsed.axis_info[0].unit_conversion_factor if parsed.axis_info else 1.0
        return float(max(np.hypot(transform.a, transform.d), np.hypot(transform.b, transform.e)) * factor)
    transformer = Transformer.from_crs(parsed, "EPSG:4326", always_xy=True)
    cx, cy = width / 2, height / 2
    x, y = transform * (cx, cy)
    lon, lat = transformer.transform(x, y)
    distances = []
    for dx, dy in ((transform.a, transform.d), (transform.b, transform.e)):
        lon2, lat2 = transformer.transform(x + dx, y + dy)
        distances.append(abs(Geod(ellps="WGS84").inv(lon, lat, lon2, lat2)[2]))
    return float(max(distances))


def run_inference(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    band_order: list[str] | None = None,
    cloud_mask_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    options: InferenceOptions | None = None,
) -> dict[str, Any]:
    options = options or InferenceOptions()
    _set_deterministic(options.deterministic)
    registry = ModelRegistry(registry_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with rasterio.open(source) as src:
        names = _band_names(src, band_order)
        sensor = registry.detect_sensor(src.count, names)
        if not names:
            names = ["red", "green", "blue"] if sensor == "RGB_VHR" else []
        image = src.read().astype("float32")
        valid = np.all(src.read_masks() > 0, axis=0) & np.all(np.isfinite(image), axis=0)
        profile = src.profile.copy()
        transform, crs = src.transform, src.crs
        resolution_m = _resolution_metres(src.width, src.height, src.transform, src.crs)
        source_metadata = {
            "path": str(source), "sensor": sensor, "band_order": names, "width": src.width, "height": src.height,
            "crs": str(src.crs) if src.crs else None, "transform": list(src.transform)[:6], "resolution": list(src.res), "nodata": src.nodata,
            "approximate_resolution_m": resolution_m,
        }
    if cloud_mask_path is not None:
        with rasterio.open(cloud_mask_path) as cloud_src:
            expected_grid = (profile["width"], profile["height"], profile.get("crs"), profile["transform"])
            if (cloud_src.width, cloud_src.height, cloud_src.crs, cloud_src.transform) != expected_grid:
                raise ValueError("Cloud mask must have exactly the same dimensions, CRS, and affine transform as the input.")
            cloud = cloud_src.read(1) != 0
        valid &= ~cloud
        source_metadata["cloud_mask"] = str(Path(cloud_mask_path).resolve())
    if not valid.any():
        raise ValueError("Input contains no valid pixels after nodata masking.")
    if sensor == "RGB_VHR" and resolution_m is not None and not (0.01 <= resolution_m <= 1.5):
        raise ValueError(f"RGB model route requires VHR imagery near 0.05-1.5 m/pixel; source is approximately {resolution_m:.3f} m/pixel.")
    if sensor == "SENTINEL2_L1C_13B" and resolution_m is not None and not (5.0 <= resolution_m <= 65.0):
        raise ValueError(f"Sentinel-2 model route expects an aligned 5-65 m grid; source is approximately {resolution_m:.3f} m/pixel.")

    cache = ModelCache(options.device, options.mixed_precision)
    warnings: list[str] = []
    used: dict[str, dict[str, Any]] = {}
    building_probability = boundary_probability = instance_labels = None
    water_probability = water_support = None

    if sensor == "RGB_VHR":
        lc_spec = registry.model("landcover_rgb_v1")
        registry.validate_array(lc_spec, image, names)
        norm = lc_spec.raw["normalization"]
        prepared = normalized_rgb(image, norm["mean"], norm["std"])
        fine_probability = blended_predict(
            prepared, lambda tile: cache.predict_landcover(lc_spec, tile), len(lc_spec.raw["output_classes"]),
            lc_spec.raw["tile_size"], lc_spec.raw["overlap"],
        )
        landcover_probability = aggregate_landcover(fine_probability, lc_spec.raw["fine_to_canonical"])
        water_probability = landcover_probability[5].copy()
        used[lc_spec.key] = {"version": lc_spec.raw["version"], "checkpoint": str(lc_spec.checkpoint), "sha256": _sha256(lc_spec.checkpoint), "reported_metrics": _metrics(registry, lc_spec)}

        building_spec = registry.model("buildings_rgb_v1")
        registry.validate_array(building_spec, image, names)
        building_output = blended_predict(
            rgb_unit(image), lambda tile: cache.predict_buildings(building_spec, tile), 2,
            building_spec.raw["tile_size"], building_spec.raw["overlap"],
        )
        building_probability, boundary_probability = building_output
        bt = building_spec.raw["thresholds"]
        instance_labels, building_stats = separate_buildings(
            building_probability, boundary_probability, valid,
            bt["footprint"], bt["boundary"], bt["minimum_area_pixels"], bt["watershed_h_prominence"], bt["watershed_min_distance"],
        )
        used[building_spec.key] = {"version": building_spec.raw["version"], "checkpoint": str(building_spec.checkpoint), "sha256": _sha256(building_spec.checkpoint), "reported_metrics": _metrics(registry, building_spec)}
        lc_thresholds = lc_spec.raw["thresholds"]
    else:
        water_spec = registry.model("water_sentinel2_v1")
        registry.validate_array(water_spec, image, names)
        backbone, spectral, evidence = spectral_features(image)
        packed = np.concatenate([backbone, spectral], axis=0)
        water_two_class = blended_predict(
            packed, lambda tile: cache.predict_water(water_spec, tile[:9], tile[9:]), 2,
            water_spec.raw["tile_size"], water_spec.raw["overlap"],
        )
        water_probability = water_two_class[1]
        wt = water_spec.raw["thresholds"]
        water_support = water_spectral_support(evidence, wt)
        water_support &= filter_binary((water_probability >= wt["probability"]) & water_support & valid, wt["minimum_area_pixels"])
        used[water_spec.key] = {"version": water_spec.raw["version"], "checkpoint": str(water_spec.checkpoint), "sha256": _sha256(water_spec.checkpoint), "reported_metrics": _metrics(registry, water_spec)}
        lc_spec = registry.model("landcover_sentinel2_v1")
        if options.require_all_models:
            lc_spec.require_available()
        warnings.append("No trained Sentinel-2 land-cover checkpoint is present; base land cover is unknown. Water output remains available from the validated multispectral specialist.")
        landcover_probability = np.zeros((7, *valid.shape), dtype="float32")
        landcover_probability[0] = 1.0
        lc_thresholds = lc_spec.raw["thresholds"]

    water_threshold = registry.model("water_sentinel2_v1").raw["thresholds"]["probability"] if sensor != "RGB_VHR" else 1.1
    building_threshold = registry.model("buildings_rgb_v1").raw["thresholds"]["footprint"]
    fusion = fuse(
        landcover_probability, valid, lc_thresholds["minimum_confidence"], lc_thresholds["ambiguity_margin"],
        water_probability=water_probability if sensor != "RGB_VHR" else None,
        water_support=water_support, water_threshold=water_threshold,
        building_probability=building_probability if options.building_overrides_landcover else None,
        building_threshold=building_threshold,
    )

    class_names = registry.model("landcover_rgb_v1").raw["canonical_classes"]
    outputs: dict[str, str] = {}
    outputs["landcover"] = str(write_raster(output / "landcover_classes.tif", fusion.classes.astype("uint8"), profile, valid, 255))
    outputs["confidence"] = str(write_raster(output / "landcover_confidence.tif", fusion.confidence.astype("float32"), profile, valid, np.nan))
    outputs["ambiguous"] = str(write_raster(output / "ambiguous_pixels.tif", fusion.ambiguous.astype("uint8"), profile, valid, 255))
    for class_id, name in enumerate(class_names):
        safe = name.replace(" ", "_").replace("/", "_")
        outputs[f"probability_{safe}"] = str(write_raster(output / f"probability_{safe}.tif", landcover_probability[class_id].astype("float32"), profile, valid, np.nan))
    outputs["water_probability"] = str(write_raster(output / "water_probability.tif", water_probability.astype("float32"), profile, valid, np.nan))
    if sensor == "SENTINEL2_L1C_13B":
        water_mask = filter_binary((water_probability >= water_threshold) & water_support & valid, registry.model("water_sentinel2_v1").raw["thresholds"]["minimum_area_pixels"])
    else:
        water_mask = (fusion.classes == 5) & valid
    outputs["water_mask"] = str(write_raster(output / "water_mask.tif", water_mask.astype("uint8"), profile, valid, 255))

    building_count = None
    building_stats = None if instance_labels is None else building_stats
    if instance_labels is not None:
        outputs["building_probability"] = str(write_raster(output / "building_probability.tif", building_probability.astype("float32"), profile, valid, np.nan))
        outputs["building_boundary"] = str(write_raster(output / "building_boundary_probability.tif", boundary_probability.astype("float32"), profile, valid, np.nan))
        outputs["building_instances"] = str(write_raster(output / "building_instance_ids.tif", instance_labels.astype("int32"), profile, valid, 0))
        geojson = output / "buildings.geojson"
        building_count = building_geojson(geojson, instance_labels, building_probability, transform, crs)
        outputs["building_geojson"] = str(geojson)
        if building_count != int(instance_labels.max()):
            raise RuntimeError("Building polygon count and compact instance IDs disagree; result withheld.")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "registry_version": registry.data["registry_version"],
        "source": source_metadata,
        "execution": {"device": str(cache.device), "mixed_precision": cache.mixed_precision, "deterministic": options.deterministic, "python": platform.python_version()},
        "models": used,
        "fusion": registry.data["fusion"],
        "class_mapping": {str(index): name for index, name in enumerate(class_names)},
        "building_count": building_count,
        "building_postprocess": building_stats,
        "warnings": warnings,
        "validation": {"structural_pipeline_tests": "see pytest output", "target_domain_accuracy": "not measured for this input", "accuracy_claim": False},
        "outputs": outputs,
    }
    metadata_path = output / "inference_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    outputs["metadata"] = str(metadata_path)
    return metadata
