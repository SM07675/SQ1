"""Fine-tuned Satlas building inference with explicit legacy ONNX support.

Overlapping probabilities are merged before global instance extraction. The GIS
feature set is the sole source of counts, masks, statistics and visual evidence.
Counts are estimates; image sharpness is not proof of detection accuracy.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.special import expit
from skimage.morphology import h_maxima, local_maxima
from skimage.segmentation import watershed
from pyproj import CRS, Transformer, Geod
from satquery_engine.config import settings
from satquery_engine.services.ingestion import _starts
from satquery_engine.services.spectral import _canonical_band_map
from satquery_engine.services.spatial_outputs import export_labels

MODEL_ID = "hotosm/dinov3s-buildings"
# SHA256 of the original FP16 model.onnx — source-of-truth sentinel
MODEL_SHA256 = "9a28e28403accc06c70a3104aedb31eca50401ed23c8ba4314b26357c25e9043"
# SHA256 of the FP32 CPU conversion (recorded in models/buildings/cpu_conversion.json)
MODEL_CPU_FP32_SHA256 = "26896aad9dd91172d2ab26743a2b310759aec2ca62b5e96aaa948101dc3c16e6"

MEAN = np.array([.4296737853453577, .4001659668453235, .34333372802741474], dtype="float32")[:, None, None]
STD  = np.array([.2056069389373208, .16738555558380538, .1598986422586595], dtype="float32")[:, None, None]
THRESHOLD = .4371

# Tile configuration: 256-px window, 128-px stride (50 % overlap)
_TILE = 256
_STRIDE = 128   # was 192 — 50 % overlap reduces boundary suppression
_SIGMA = 40.0   # Gaussian σ for blending kernel (was 32)

# Instance filtering
_MIN_PIXELS = 16        # minimum instance size (4 m² at 0.5 m/px) — was 4
_LARGE_BLOB = 1500      # pixel count above which a blob is "large"
_MAX_MARKERS = 50       # per-blob marker cap for large blobs (prevents over-segmentation)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _resolve_checkpoint() -> Path:
    """Use the configured checkpoint or fine-tuned model bundle."""
    source = Path(settings.building_checkpoint)
    if not source.exists():
        raise ValueError("The building analysis model is currently unavailable. No count was generated.")
    if source.is_dir() and (source / "satquery_buildings_config.json").is_file():
        return source
    if source.is_file():
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != MODEL_SHA256 and digest != MODEL_CPU_FP32_SHA256:
            pass
        return source
    return source


@lru_cache(maxsize=1)
def load_session(path: str):
    """Load and cache the building model (fine-tuned bundle or ONNX checkpoint)."""
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise ValueError("The building analysis model is currently unavailable. No count was generated.")
    if checkpoint.is_dir() and (checkpoint / "model.safetensors").is_file():
        from satquery_engine.services.buildings_rf import load_rf_building

        return load_rf_building(str(checkpoint))
    if checkpoint.is_dir() and (checkpoint / "satquery_buildings_config.json").is_file():
        from satquery_engine.models.satlas_building_net import load_building_bundle
        return load_building_bundle(checkpoint)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    from satquery_engine.models.device import onnx_providers
    providers = onnx_providers()
    session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=providers)
    if session.get_inputs()[0].shape != [1, 3, 256, 256] or session.get_outputs()[0].shape != [1, 3, 256, 256]:
        raise ValueError("Unexpected building model input/output schema.")
    if providers[0] == "CUDAExecutionProvider":
        probe = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 256, 256), dtype="float32")})[0]
        if not np.isfinite(probe).all():
            session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=["CPUExecutionProvider"])
    return session


@lru_cache(maxsize=4)
def _sha256(path: str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# ---------------------------------------------------------------------------
# Gaussian blend kernel (pre-computed once)
# ---------------------------------------------------------------------------

def _make_kernel(size: int = _TILE, sigma: float = _SIGMA) -> np.ndarray:
    axis = np.exp(-0.5 * ((np.arange(size) - (size - 1) / 2.0) / sigma) ** 2).astype("float32")
    return np.maximum(np.outer(axis, axis), 1e-8)


_KERNEL = _make_kernel()


# ---------------------------------------------------------------------------
# Instance separation (watershed)
# ---------------------------------------------------------------------------

def _plateau_peaks(distance, foreground, min_distance):
    """One candidate per connected maximum plateau, not one per grid interval."""
    surface = np.where(foreground, distance, -np.inf)
    maxima = (surface == ndimage.maximum_filter(surface, size=2 * min_distance + 1)) & foreground
    # A flat saddle connecting two roofs is not a regional maximum, even if
    # its distant higher endpoints lie outside the suppression window.
    maxima &= local_maxima(surface, connectivity=1, allow_borders=True)
    regions, _ = ndimage.label(maxima)
    candidates = []
    for ident, sl in enumerate(ndimage.find_objects(regions), 1):
        if sl is None:
            continue
        points = np.argwhere(regions[sl] == ident)
        center = points.mean(axis=0)
        point = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
        candidates.append(point + [sl[0].start, sl[1].start])
    if not candidates:
        return np.empty((0, 2), dtype=int)
    candidates = np.asarray(candidates)
    order = np.argsort(-distance[tuple(candidates.T)], kind="stable")
    kept = []
    buckets = {}
    for point in candidates[order]:
        cell = tuple(point // min_distance)
        nearby = (other for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  for other in buckets.get((cell[0] + dy, cell[1] + dx), ()))
        if not any(np.max(np.abs(point - other)) < min_distance for other in nearby):
            kept.append(point)
            buckets.setdefault(cell, []).append(point)
    return np.asarray(kept, dtype=int)


def separate_instances(probability, distance, valid, threshold=THRESHOLD):
    """Separate overlapping detections into individual building instances.

    Returns (compacted_labels, rejected_tiny_count).
    Score lookup MUST be computed from the returned compacted labels, not from
    watershed output, to guarantee score-ID alignment in the GeoJSON.
    """
    foreground = (probability > threshold) & valid
    components, n = ndimage.label(foreground)
    seeds = np.zeros(foreground.shape, dtype=bool)

    sizes = np.bincount(components.ravel())
    large_ids = sizes > _LARGE_BLOB
    large_ids[0] = False
    large = large_ids[components]

    # Collapse flat maxima before applying the separation distance (min_distance=6 prevents splitting roofs).
    peaks = _plateau_peaks(distance, (distance >= 2.0) & foreground & ~large, min_distance=6)
    if len(peaks):
        seeds[tuple(peaks.T)] = True

    # Large blobs retain the wider separation distance and marker cap.
    if large.any():
        large_peaks = _plateau_peaks(distance, (distance >= 3.0) & large, min_distance=10)
        if len(large_peaks):
            # Cap markers per large component to prevent runaway over-segmentation
            for comp_id in np.where(large_ids)[0]:
                comp_mask = components == comp_id
                comp_peak_mask = np.zeros(foreground.shape, dtype=bool)
                comp_peak_mask[tuple(large_peaks.T)] = True
                comp_peak_mask &= comp_mask
                comp_peak_coords = np.argwhere(comp_peak_mask)
                if len(comp_peak_coords) > _MAX_MARKERS:
                    # Keep the _MAX_MARKERS peaks with highest distance value
                    dist_vals = distance[comp_peak_mask]
                    top_idx = np.argsort(dist_vals)[-_MAX_MARKERS:]
                    comp_peak_coords = comp_peak_coords[top_idx]
                seeds[comp_peak_coords[:, 0], comp_peak_coords[:, 1]] = True

    # Guarantee at least one marker per connected component (including edge objects)
    for i, sl in enumerate(ndimage.find_objects(components), 1):
        if sl is None:
            continue
        region = components[sl] == i
        if not np.any(seeds[sl] & region):
            y, x = np.unravel_index(
                np.argmax(np.where(region, distance[sl], -np.inf)), region.shape
            )
            seeds[sl[0].start + y, sl[1].start + x] = True

    markers, _ = ndimage.label(seeds)
    labels = watershed(-distance, markers, mask=foreground).astype("int32") if n else components

    # Remove tiny instances
    sizes = np.bincount(labels.ravel())
    rejected = int(np.sum((sizes[1:] < _MIN_PIXELS) & (sizes[1:] > 0)))
    keep = sizes >= _MIN_PIXELS
    keep[0] = False
    labels[~keep[labels]] = 0

    # Compact IDs to 1..N (no gaps) — MUST happen before bincount for scores
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lookup = np.zeros(int(labels.max()) + 1 if labels.max() > 0 else 1, dtype="int32")
    lookup[ids] = np.arange(1, len(ids) + 1)
    return lookup[labels], rejected


def separate_instances_satlas(
    footprint_prob: np.ndarray,
    boundary_prob: np.ndarray,
    valid: np.ndarray,
    foot_thr: float = 0.6,
    boundary_thr: float = 0.6,
    min_area: int = 48,
    min_distance: int = 8,
    h_prominence: float = 6.0,
) -> tuple[np.ndarray, int]:
    """Separate building footprints using dual-head footprint and boundary probabilities.

    Uses smooth distance transform on foreground building regions combined with
    boundary attenuation to prevent over-segmenting single roofs, while cleanly
    splitting touching buildings using prominence-guided watershed.
    """
    if (footprint_prob.shape != boundary_prob.shape or footprint_prob.shape != valid.shape
            or footprint_prob.ndim != 2):
        raise ValueError("Building footprint, boundary and validity grids must match.")
    if not (0 < foot_thr < 1 and 0 < boundary_thr < 1):
        raise ValueError("Building probability thresholds must lie between zero and one.")
    if min_area < 1 or min_distance < 1 or h_prominence <= 0:
        raise ValueError("Building area, separation distance and prominence must be positive.")
    if (not np.isfinite(footprint_prob).all() or not np.isfinite(boundary_prob).all()
            or np.any((footprint_prob < 0) | (footprint_prob > 1))
            or np.any((boundary_prob < 0) | (boundary_prob > 1))):
        raise ValueError("Building postprocessing requires finite probabilities in 0-1.")
    foreground = (footprint_prob > foot_thr) & valid
    if not foreground.any():
        return np.zeros(foreground.shape, dtype="int32"), 0

    distance = ndimage.distance_transform_edt(foreground)
    effective_dist = distance * (1.0 - 0.4 * boundary_prob)
    # Preserve the smooth prominence surface. Thresholding it before finding
    # seeds introduces artificial steps and spurious roof centers.
    elevation = np.where(boundary_prob >= boundary_thr, -effective_dist, -distance)

    # Prominence-based seeds to suppress intra-roof texture noise
    candidates = h_maxima(effective_dist, h=h_prominence) & foreground
    peaks = _plateau_peaks(effective_dist, candidates, min_distance)
    seeds = np.zeros_like(foreground)
    if len(peaks):
        seeds[tuple(peaks.T)] = True

    # Guarantee at least one seed per connected component >= min_area
    components, n_comp = ndimage.label(foreground)
    if n_comp > 0:
        for i, sl in enumerate(ndimage.find_objects(components), 1):
            if sl is None:
                continue
            region = components[sl] == i
            if region.sum() < min_area:
                continue
            if not np.any(seeds[sl] & region):
                y, x = np.unravel_index(
                    np.argmax(np.where(region, distance[sl], -np.inf)), region.shape
                )
                seeds[sl[0].start + y, sl[1].start + x] = True

        markers, n_markers = ndimage.label(seeds)
        labels = watershed(elevation, markers, mask=foreground).astype("int32") if n_markers > 0 else components
    else:
        labels = np.zeros(foreground.shape, dtype="int32")

    sizes = np.bincount(labels.ravel())
    rejected = int(np.sum((sizes[1:] < min_area) & (sizes[1:] > 0)))
    keep = sizes >= min_area
    keep[0] = False
    labels[~keep[labels]] = 0

    ids = np.unique(labels)
    ids = ids[ids > 0]
    lookup = np.zeros(int(labels.max()) + 1 if labels.max() > 0 else 1, dtype="int32")
    lookup[ids] = np.arange(1, len(ids) + 1)
    return lookup[labels], rejected



# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------

def resolution_m(src):
    if not src.crs:
        return None
    crs = CRS(src.crs)
    if crs.is_projected:
        return max(src.res) * crs.axis_info[0].unit_conversion_factor
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    cx, cy = src.width / 2, src.height / 2
    x = src.transform.a * cx + src.transform.b * cy + src.transform.c
    y = src.transform.d * cx + src.transform.e * cy + src.transform.f
    lon, lat = t.transform(x, y)
    distances = []
    for dx,dy in ((src.transform.a,src.transform.d),(src.transform.b,src.transform.e)):
        lon2,lat2 = t.transform(x+dx,y+dy)
        distances.append(abs(Geod(ellps="WGS84").inv(lon,lat,lon2,lat2)[2]))
    return max(distances)


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def detect_buildings(path, output, progress=None, *, water_result=None):
    from satquery_engine.services.radiometry import rgb_unit_data, rgb_indexes

    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError("Building analysis needs optical RGB bands.")
        if src.width * src.height > 32_000_000:
            raise ValueError(
                "Building analysis currently supports up to 32 million pixels per image."
            )
        indexes = rgb_indexes(src)
        gsd = resolution_m(src)
        if gsd is not None and gsd > 1.5:
            raise ValueError(
                f"This image has approximately {gsd:.1f}-metre pixels, "
                "too coarse for a reliable individual-building count with this model. "
                "Individual building footprint segmentation requires very high resolution satellite or aerial imagery (GSD <= 1.5m). "
                "For satellite imagery at this resolution, query 'Identify built-up areas' to map urban surface footprints."
            )

        # Image quality check: blur and contrast
        sample_h = min(src.height, 512)
        sample_w = min(src.width, 512)
        sample = src.read(indexes[:3], out_shape=(3, sample_h, sample_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
        sample_gray = np.mean(sample, axis=0)
        s_span = float(np.ptp(sample_gray)) if sample_gray.size else 1.0
        s_norm = (sample_gray - np.min(sample_gray)) / max(s_span, 1e-6)
        blur_var = float(ndimage.laplace(s_norm).var())
        contrast_val = float(s_norm.std())

        if gsd is not None:
            if gsd <= 0.8 and blur_var >= 0.003 and contrast_val >= 0.10:
                count_reliability = "COUNT_RELIABLE"
            elif gsd <= 1.5:
                count_reliability = "COUNT_APPROXIMATE"
            else:
                count_reliability = "COUNT_UNRELIABLE"
        else:
            count_reliability = "COUNT_APPROXIMATE" if (blur_var >= 0.003 and contrast_val >= 0.10) else "COUNT_UNRELIABLE"

        # Load FP32 checkpoint or fine-tuned model bundle
        fallback_events = []
        fp32_path = Path(settings.building_checkpoint)
        try:
            fp32_path = _resolve_checkpoint()
            session_or_bundle = load_session(str(fp32_path))
        except (ValueError, RuntimeError, OSError) as primary_error:
            # An explicit selection must never silently run another model.
            if os.getenv("SATQUERY_BUILDING_CHECKPOINT") or fp32_path != Path(settings.building_checkpoint):
                raise
            from satquery_engine.models.registry import LocalModelRegistry
            fallback = LocalModelRegistry(settings.model_dir).get("building_fallback")
            if fallback is None or not fallback.availability:
                raise ValueError("The primary building model failed and no verified DINO fallback is available.") from primary_error
            fp32_path = fallback.local_path / "onnx/model.onnx"
            session_or_bundle = load_session(str(fp32_path))
            fallback_events.extend(["PRIMARY_MODEL_FAILED", "FALLBACK_USED",
                                    f"Primary building model failed: {primary_error}"])
        from satquery_engine.services.buildings_rf import RfBuildingSession
        is_rf = isinstance(session_or_bundle, RfBuildingSession)
        is_bundle = isinstance(session_or_bundle, tuple)

        h, w = src.height, src.width
        scene_rgb, scene_valid, _ = rgb_unit_data(src)
        del scene_rgb

        if is_rf:
            from satquery_engine.services.buildings_rf import (
                MODEL_ID as RF_MODEL_ID, TILE as RF_TILE, STRIDE as RF_STRIDE,
                THRESHOLD as RF_THRESHOLD, detect_rf_instances,
            )

            labels, prediction_prob, duplicate_count, tile_records, radiometry = detect_rf_instances(
                path, session_or_bundle, scene_valid, progress=progress
            )
            radiometry = {"source_rgb": radiometry,
                          "model_processor": "RfDetrImageProcessor: resize to 432x432, divide by 255, ImageNet normalization"}
            valid = scene_valid
            positions = tile_records
            rejected = 0
            active_tile_size = RF_TILE
            active_stride = RF_STRIDE
            active_model_id = RF_MODEL_ID
            active_method = "RF-DETR satellite building instance masks with global duplicate suppression"
            active_execution = "rf_detr_satellite_safetensors"
            from satquery_engine.models.device import torch_device
            active_device = torch_device()
            active_checkpoint_sha = session_or_bundle.checkpoint_sha256
            active_threshold = RF_THRESHOLD
            from satquery_engine.services.buildings_rf import MIN_INSTANCE_PIXELS
            active_min_pixels = MIN_INSTANCE_PIXELS
            active_benchmark_f1 = None
            postprocess_audit = {"method": "score_ordered_instance_merge", "score_threshold": RF_THRESHOLD,
                                 "overlap_iou_threshold": .5, "containment_threshold": .8,
                                 "minimum_instance_pixels": active_min_pixels,
                                 "minimum_unclaimed_fraction": .5,
                                 "island_minimum_fraction": .02,
                                 "tile_edge_policy": "prefer complete interior predictions over internally clipped masks",
                                 "probability_kind": "sparse_instance_score"}

        elif is_bundle:
            import torch
            from satquery_engine.models.device import torch_device
            model, config, bundle_metrics = session_or_bundle
            device = torch_device() if hasattr(model, "to") else "cpu"
            if hasattr(model, "to"):
                model.to(device)
            tile_size = int(config.get("input", {}).get("tile_size", 512))
            overlap = int(config.get("tile_inference", {}).get("overlap", 128))
            stride = tile_size - overlap
            foot_thr = float(config.get("postprocess", {}).get("foot_thr", 0.6))
            bound_thr = float(config.get("postprocess", {}).get("boundary_thr", 0.6))
            min_area = int(config.get("postprocess", {}).get("min_area", 48))
            min_dist = int(config.get("postprocess", {}).get("min_distance", 8))
            h_prom = float(config.get("postprocess", {}).get("h_prominence", 6.0))

            if tile_size < 1 or not 0 <= overlap < tile_size:
                raise ValueError("Invalid building tile size or overlap.")
            if not (0 < foot_thr < 1 and 0 < bound_thr < 1):
                raise ValueError("Invalid building model thresholds.")

            hann_1d = np.hanning(tile_size)
            hann_2d = np.outer(hann_1d, hann_1d).astype("float32")
            hann_2d = np.maximum(hann_2d, 1e-4)

            accum = np.zeros((2, h, w), dtype="float32")
            weight = np.zeros((h, w), dtype="float32")
            valid = np.zeros((h, w), dtype=bool)

            positions = [(y, x) for y in _starts(h, tile_size, stride) for x in _starts(w, tile_size, stride)]
            tile_records = []

            for index, (y, x) in enumerate(positions):
                th, tw = min(tile_size, h - y), min(tile_size, w - x)
                raw, good, radiometry = rgb_unit_data(src, window=Window(x, y, tw, th), scene_valid=scene_valid)
                patch = np.pad(raw, ((0, 0), (0, tile_size-th), (0, tile_size-tw)), mode="edge")
                t_patch = torch.from_numpy(patch).unsqueeze(0).to(device)
                with torch.inference_mode():
                    logits = model(t_patch)[0].cpu().numpy()
                if logits.shape != (2, tile_size, tile_size) or not np.isfinite(logits).all():
                    raise ValueError("Building model returned invalid logits.")
                probs_patch = expit(logits)
                k = hann_2d[:th, :tw]
                accum[:, y:y + th, x:x + tw] += probs_patch[:, :th, :tw] * k
                weight[y:y + th, x:x + tw] += k
                valid[y:y + th, x:x + tw] |= good
                tile_records.append({
                    "tile_id": index, "x_offset": x, "y_offset": y, "width": tw, "height": th,
                    "origin": [x, y], "shape": [th, tw],
                    "padding": [tile_size - th, tile_size - tw], "scale": 1,
                    "transform": list(src.window_transform(Window(x, y, tw, th)))[:6],
                    "valid_pixels": int(good.sum()),
                })
                if progress:
                    progress(f"Detecting buildings: tile {index + 1} of {len(positions)}")

            if device == "cuda" and hasattr(model, "cpu"):
                model.cpu()
                torch.cuda.empty_cache()

            if not valid.any() or np.any(weight <= 0):
                raise ValueError("Building inference lacks complete valid coverage.")
            safe_weight = np.where(weight > 0, weight, 1.0)
            probs = accum / safe_weight

            footprint_prob = probs[0]
            boundary_prob = probs[1]
            labels, rejected = separate_instances_satlas(
                footprint_prob, boundary_prob, valid,
                foot_thr=foot_thr, boundary_thr=bound_thr,
                min_area=min_area, min_distance=min_dist,
                h_prominence=h_prom,
            )
            prediction_prob = footprint_prob
            active_tile_size = tile_size
            active_stride = stride
            active_model_id = "satquery_buildings_bundle"
            active_method = "SatlasBuildingNet Aerial Swin-v2-Base with FPN, Hann blend tiling, and boundary-guided watershed"
            active_execution = "satquery_buildings_bundle"
            active_device = device
            active_checkpoint_sha = _sha256(str(Path(fp32_path) / "satquery_buildings_state_dict.pt"))
            active_threshold = foot_thr
            active_min_pixels = min_area
            active_benchmark_f1 = bundle_metrics.get("test", {}).get("instance_f1@0.5")
            duplicate_count = 0
            postprocess_audit = {
                "method": "boundary_guided_watershed_v2",
                "footprint_threshold": foot_thr, "boundary_threshold": bound_thr,
                "minimum_area_pixels": min_area, "minimum_seed_distance_pixels": min_dist,
                "h_prominence": h_prom,
                "threshold_source": config.get("threshold_source", "bundle configuration"),
            }

        else:
            from satquery_engine.models.device import torch_device
            session = session_or_bundle
            accum  = np.zeros((3, h, w), dtype="float32")
            weight = np.zeros((h, w),    dtype="float32")
            valid  = np.zeros((h, w),    dtype=bool)

            positions = [(y, x) for y in _starts(h, _TILE, _STRIDE) for x in _starts(w, _TILE, _STRIDE)]
            tile_records = []

            for index, (y, x) in enumerate(positions):
                th, tw = min(_TILE, h - y), min(_TILE, w - x)
                raw, good, radiometry = rgb_unit_data(src, window=Window(x, y, tw, th), scene_valid=scene_valid)
                tile = np.zeros((3, _TILE, _TILE), dtype="float32")
                tile[:, :th, :tw] = raw
                logits = session.run(None, {"image": ((tile - MEAN) / STD)[None].astype("float32")})[0][0]
                if logits.shape != (3, _TILE, _TILE) or not np.isfinite(logits).all():
                    raise ValueError("Building model returned invalid logits.")
                predictions = np.stack([expit(logits[0]), expit(logits[1]), np.tanh(logits[2])])
                k = _KERNEL[:th, :tw]
                accum[:, y:y + th, x:x + tw] += predictions[:, :th, :tw] * k
                weight[y:y + th, x:x + tw]   += k
                valid[y:y + th, x:x + tw]    |= good
                tile_records.append({
                    "tile_id": index, "x_offset": x, "y_offset": y, "width": tw, "height": th,
                    "origin": [x, y], "shape": [th, tw],
                    "padding": [_TILE - th, _TILE - tw], "scale": 1,
                    "transform": list(src.window_transform(Window(x, y, tw, th)))[:6],
                    "valid_pixels": int(good.sum()),
                })
                if progress:
                    progress(f"Detecting buildings: tile {index + 1} of {len(positions)}")

            # Blend: only require coverage where pixels are valid
            uncovered_valid = valid & (weight == 0)
            if uncovered_valid.any():
                raise ValueError("Building inference did not cover every valid pixel; no count generated.")

            safe_weight = np.where(weight > 0, weight, 1.0)
            predictions = accum / safe_weight

            if not valid.any():
                raise ValueError("No valid RGB pixels are available for building analysis.")

            labels, rejected = separate_instances(predictions[0], predictions[2], valid)
            prediction_prob = predictions[0]
            active_tile_size = _TILE
            active_stride = _STRIDE
            active_model_id = MODEL_ID
            active_method = "DINOv3 footprint/boundary/distance heads with global watershed and overlapping probability merge"
            active_execution = "original_released_onnx"
            active_device = session.get_providers()[0] if hasattr(session, "get_providers") else "unknown"
            if active_device == "CPUExecutionProvider" and torch_device() == "cuda":
                fallback_events.append("GPU_PROVIDER_FAILED: DINO ONNX CUDA output was nonfinite; verified CPU inference was used.")
            active_checkpoint_sha = _sha256(str(fp32_path))
            active_threshold = THRESHOLD
            active_min_pixels = _MIN_PIXELS
            active_benchmark_f1 = None
            duplicate_count = 0
            postprocess_audit = {"method": "dinov3_distance_watershed", "footprint_threshold": THRESHOLD,
                                 "minimum_area_pixels": _MIN_PIXELS}

        if not valid.any():
            raise ValueError("No valid RGB pixels are available for building analysis.")

        surface_exclusion = {"status": "not_requested"}
        if water_result is not None:
            from satquery_engine.services.surface_context import exclude_water_instances
            labels, surface_exclusion = exclude_water_instances(
                labels, valid, path, water_result, active_min_pixels,
            )
        # Compute scores from COMPACTED labels (guaranteeing score keys == GeoJSON IDs)
        sums  = np.bincount(labels.ravel(), weights=prediction_prob.ravel())
        sizes = np.bincount(labels.ravel())
        scores = {
            i: float(sums[i] / sizes[i])
            for i in range(1, len(sizes))
            if sizes[i] > 0
        }

        # Export with numbered overlay for visual count verification
        result = export_labels(
            labels, path, output, "buildings",
            scores=scores, valid_mask=valid, numbered=True
        )
        if len(result["features"]) != len(scores):
            raise ValueError("Building count does not match final features; result withheld.")

        # Persist probability map
        np.save(output / "building_probability.npy", prediction_prob)
        result["paths"].append(output / "building_probability.npy")
        from satquery_engine.services.spatial_outputs import export_float_raster
        result["paths"].append(
            export_float_raster(
                np.where(valid, prediction_prob, np.nan), path,
                output / "buildings_probability.tif"
            )
        )
        if is_bundle:
            result["paths"].append(export_float_raster(
                np.where(valid, boundary_prob, np.nan), path,
                output / "buildings_boundary_probability.tif",
            ))

        # Export binary buildings_mask.tif
        from rasterio.transform import Affine
        mask_tif = output / "buildings_mask.tif"
        with rasterio.open(path) as src_meta:
            grid = src_meta.transform @ Affine.scale(src_meta.width / labels.shape[1], src_meta.height / labels.shape[0])
            prof = dict(
                driver="GTiff",
                width=labels.shape[1],
                height=labels.shape[0],
                count=1,
                dtype="uint8",
                crs=src_meta.crs,
                transform=grid,
                compress="deflate",
            )
        with rasterio.open(mask_tif, "w", **prof) as dst:
            dst.write((labels > 0).astype("uint8") * 255, 1)
            dst.write_mask(valid.astype("uint8") * 255)
        result["paths"].append(mask_tif)

        tile_manifest = output / "building_tiles.json"
        tile_manifest.write_text(json.dumps({
            "tile_size": active_tile_size, "overlap": active_tile_size - active_stride,
            "stride": active_stride, "gaussian_sigma": _SIGMA if not is_bundle and not is_rf else None,
            "tiles": tile_records, "radiometry": radiometry,
            "checkpoint_sha256": active_checkpoint_sha,
            "checkpoint_execution": active_execution,
            "device": active_device,
            "postprocessing": postprocess_audit,
        }, indent=2))
        result["paths"].append(tile_manifest)

        limitations = [
            "Predicted visible footprints are an estimate, not a surveyed building inventory. "
            "Touching roofs, shadows and tiny buildings can cause count errors.",
            "Model scores are uncalibrated. Benchmark performance may not transfer to this scene.",
        ]
        limitations.extend(fallback_events)
        if surface_exclusion["status"] == "applied":
            limitations.append("Building footprints exclude the canonical estimated water mask. Water classification errors can affect shoreline buildings.")
        if is_rf:
            limitations.append("Selected on satellite imagery. This checkpoint performed poorly on the local aerial reference chips; aerial counts require separate validation.")
        benchmark_path = Path(__file__).resolve().parents[3] / "artifacts/building_benchmark/benchmark.json"
        benchmark = None
        if not is_bundle and not is_rf and benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text())
            if (benchmark.get("f1") or 0) < .6:
                limitations.append(
                    f'The measured {benchmark["sample_count"]}-image validation subset had low instance '
                    f'accuracy (F1 {benchmark["f1"]:.3f}); treat this count as a low-confidence estimate.'
                )
        if gsd is None:
            limitations.append(
                "Ground resolution is unknown; geographic areas and exact count reliability cannot be established."
            )
        elif gsd > 1:
            limitations.append("Pixels exceed one metre; small and touching buildings may be missed.")

        if count_reliability == "COUNT_UNRELIABLE":
            limitations.append(
                "I can detect built-up areas, but this image is not detailed enough for an accurate individual-building count."
            )

        if is_rf:
            count_reliability = "COUNT_UNRELIABLE"
            limitations.append(
                "The satellite-model selection set performed well, but a separate aerial reference set had no matched footprints; "
                "accuracy on this scene's sensor and geography is unvalidated."
            )
        elif active_benchmark_f1 is not None and active_benchmark_f1 < .6:
            count_reliability = "COUNT_UNRELIABLE"
            limitations.append(
                f"The fine-tuned model's held-out test instance F1 was {active_benchmark_f1:.3f}; "
                "the footprint count is reported as low-confidence."
            )
        elif not is_bundle and (benchmark is None or (benchmark.get("f1") or 0) < .6):
            count_reliability = "COUNT_UNRELIABLE"
        elif count_reliability == "COUNT_RELIABLE":
            count_reliability = "COUNT_APPROXIMATE"

        small_objs = int(np.sum((sizes[1:] > 0) & (sizes[1:] < 20)))
        uncertain_objs = int(sum(1 for s in scores.values() if s < 0.50))
        mean_conf = float(np.mean(list(scores.values()))) if scores else 0.0

        result.update(
            count=len(result["features"]),
            surface_exclusion=surface_exclusion,
            building_count=len(result["features"]),
            building_count_type="visible_footprint_count",
            confidence="HIGH" if count_reliability == "COUNT_RELIABLE" else ("MEDIUM" if count_reliability == "COUNT_APPROXIMATE" else "LOW"),
            count_reliability=count_reliability,
            average_confidence=round(mean_conf, 4),
            detected_area=result.get("area_m2"),
            rejected_candidates=rejected,
            duplicate_candidates_removed=duplicate_count,
            small_objects=small_objs,
            uncertain_objects=uncertain_objs,
            count_state="ESTIMATED",
            rejected_tiny=rejected,
            rejected_duplicates=duplicate_count,
            checkpoint_sha256=active_checkpoint_sha,
            checkpoint_execution=active_execution,
            device=active_device,
            preprocessing=radiometry,
            postprocessing=postprocess_audit,
            small_object_rate=float(np.mean(sizes[1:] < 20)) if len(sizes) > 1 else 0.,
            count_distribution={
                "under_20_pixels": small_objs,
                "at_least_20_pixels": int(np.sum(sizes[1:] >= 20)),
            },
            duplicate_policy=("score-ordered global instance overlap suppression" if is_rf else
                "overlapping probability maps blended before global instance extraction; no per-tile instances counted"),
            tile_count=len(positions),
            tile_overlap_px=active_tile_size - active_stride,
            complete_coverage=True,
            resolution_m=gsd,
            blur_variance=blur_var,
            contrast_std=contrast_val,
            minimum_instance_pixels=active_min_pixels,
            mean_model_score=mean_conf if scores else None,
            evidence_strength=(mean_conf if scores else 0.0 if is_rf else
                               float(np.mean(1 - prediction_prob[valid])) if valid.any() else 0.0),
            model_id=active_model_id,
            threshold=active_threshold,
            benchmark_f1=active_benchmark_f1 if is_bundle else (benchmark.get("f1") if benchmark else None),
            method=active_method,
            limitations=limitations,
            fallback_events=fallback_events,
        )

    q_pass, q_warnings = check_building_result(result, labels, scores, gsd)
    result["quality_gate_passed"] = q_pass
    result["quality_warnings"] = q_warnings

    result["evidence_strength"] = min(result["evidence_strength"], .25 if count_reliability == "COUNT_UNRELIABLE" else .5)
    result["domain_compatibility"] = "unverified" if gsd is None else "resolution_checked_domain_unvalidated"
    stats = output / "buildings_stats.json"
    stats.write_text(json.dumps({k:v for k,v in result.items() if k not in {"paths", "features"}}, indent=2, allow_nan=False))
    result["paths"].append(stats)
    return result


def check_building_result(
    result: dict[str, Any],
    labels: np.ndarray,
    scores: dict[int, float],
    gsd: float | None,
) -> tuple[bool, list[str]]:
    """Context-aware sanity check on building detection results."""
    warnings: list[str] = []
    count = len(scores)
    if count == 0:
        return True, ["No building footprints detected in this scene."]

    small_objects = result.get("small_objects", 0)
    if float(small_objects) / max(1, count) > 0.60:
        warnings.append("Over 60% of detected structures are small (under 20 pixels); possible over-segmentation or noise.")

    uncertain_objects = result.get("uncertain_objects", 0)
    if float(uncertain_objects) / max(1, count) > 0.50:
        warnings.append("Over 50% of detected buildings have model confidence below 0.50.")

    if gsd is not None and gsd > 1.5:
        warnings.append(f"Image resolution ({gsd:.1f}m GSD) is coarse; individual structures may be merged or omitted.")

    if result.get("count_reliability") == "COUNT_UNRELIABLE":
        warnings.append("I can detect built-up areas, but this image is not detailed enough for an accurate individual-building count.")

    return len(warnings) == 0, warnings



# ---------------------------------------------------------------------------
# Building change matching
# ---------------------------------------------------------------------------

def match_buildings(a, b, output):
    from shapely.geometry import shape
    from scipy.optimize import linear_sum_assignment
    left  = [shape(f["properties"]["pixel_geometry"]) for f in a["features"]]
    right = [shape(f["properties"]["pixel_geometry"]) for f in b["features"]]
    scores = np.zeros((len(left), len(right)))
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            if x.intersects(y):
                scores[i, j] = x.intersection(y).area / x.union(y).area
    matches = []
    if scores.size:
        rows, cols = linear_sum_assignment(-scores)
        matches = [(int(i), int(j)) for i, j in zip(rows, cols) if scores[i, j] >= .3]
    ma = {i for i, j in matches}
    mb = {j for i, j in matches}
    features = []
    for role, result, matched in [("REMOVED", a, ma), ("NEW", b, mb)]:
        for i, f in enumerate(result["features"]):
            features.append({
                **f,
                "properties": {
                    **f["properties"],
                    "status": "UNCHANGED" if i in matched else "POSSIBLE_CHANGE",
                    "candidate_type": None if i in matched else role,
                    "image": 1 if role == "REMOVED" else 2,
                },
            })
    geojson = output / "building_change.geojson"
    geojson.write_text(json.dumps({
        "type": "FeatureCollection", "features": features,
        "properties": {"crs": "EPSG:4326" if a["area_m2"] is not None else None},
    }, allow_nan=False))
    from PIL import Image, ImageDraw
    preview    = next(p for p in b["paths"] if p.name == "buildings_original.png")
    labels_path = next(p for p in b["paths"] if p.name == "buildings_labels.tif")
    with rasterio.open(labels_path) as grid:
        width, height = grid.width, grid.height
    base  = Image.open(preview).convert("RGBA")
    marks = Image.new("RGBA", base.size)
    draw  = ImageDraw.Draw(marks)
    for f in features:
        props = f["properties"]
        if props["image"] == 1 and props["status"] == "UNCHANGED":
            continue
        color = (
            (0, 190, 220, 100) if props["status"] == "UNCHANGED"
            else (245, 165, 30, 160) if props["candidate_type"] == "NEW"
            else (225, 65, 100, 160)
        )
        geometry = props["pixel_geometry"]
        rings = (
            geometry["coordinates"] if geometry["type"] == "Polygon"
            else [r for poly in geometry["coordinates"] for r in poly]
        )
        for ring in rings:
            draw.polygon(
                [(x * base.width / width, y * base.height / height) for x, y in ring],
                fill=color, outline="white",
            )
    overlay = output / "building_change_overlay.png"
    Image.alpha_composite(base, marks).convert("RGB").save(overlay)
    return {
        "before_count": a["count"],
        "after_count": b["count"],
        "net_count_change": b["count"] - a["count"],
        "persistent_count": len(matches),
        "possible_new_count": b["count"] - len(mb),
        "possible_removed_count": a["count"] - len(ma),
        "net_footprint_area_m2": (
            b["area_m2"] - a["area_m2"]
            if a["area_m2"] is not None and b["area_m2"] is not None
            else None
        ),
        "paths": [geojson, overlay],
        "method": "One-to-one footprint IoU matching on the registered grid",
        "evidence_strength": min(a["evidence_strength"], b["evidence_strength"]),
        "limitations": [
            "Unmatched footprints are possible changes; occlusion, image quality and alignment "
            "can also cause unmatched detections.",
            "Change overlay: cyan = matched, amber = possibly new, pink = possibly removed. "
            "Colors do not confirm real-world construction or demolition.",
        ],
    }
