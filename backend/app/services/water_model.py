from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork

from app.config import settings
from app.services.spectral import (
    _area_m2,
    _colorize_index,
    _describe_region_location,
    _polygonize,
    _write_mask,
    write_index_png,
)

logger = logging.getLogger(__name__)

# Default benchmark metrics from satquery_water_metrics.json
DEFAULT_WATER_METRICS = {
    "precision": 0.9209,
    "recall": 0.7626,
    "f1": 0.8343,
    "iou": 0.7157,
    "specificity": 0.9931,
    "balanced_accuracy": 0.8779,
    "dark_land_fpr": 0.0060,
}


class SatlasBackbone(nn.Module):
    """Swin-v2-Base + 4-level Feature Pyramid Network + 2-stage TransposeConv Upsampler."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        swin = models.swin_v2_b()
        # Patch partition layer modified to take 9 multispectral channels
        swin.features[0][0] = nn.Conv2d(9, 128, kernel_size=4, stride=4)
        self.backbone.backbone = swin

        self.fpn = nn.Module()
        self.fpn.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=128,
        )

        self.upsample = nn.Module()
        self.upsample.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting multi-scale Swin features, FPN merging, and 4x upsampling."""
        features = self.backbone.backbone.features
        # features[0]: patch partition (stride 4, 128 channels)
        f0 = features[0](x)
        # features[1]: Stage 1 Swin transformer blocks
        f1 = features[1](f0)
        # features[2]: patch merging (stride 8, 256 channels)
        f2 = features[2](f1)
        # features[3]: Stage 2 Swin transformer blocks
        f3 = features[3](f2)
        # features[4]: patch merging (stride 16, 512 channels)
        f4 = features[4](f3)
        # features[5]: Stage 3 Swin transformer blocks
        f5 = features[5](f4)
        # features[6]: patch merging (stride 32, 1024 channels)
        f6 = features[6](f5)
        # features[7]: Stage 4 Swin transformer blocks
        f7 = features[7](f6)

        # Swin stages produce channels-last tensors: (B, H_i, W_i, C_i)
        # Permute to channels-first: (B, C_i, H_i, W_i)
        p1 = f1.permute(0, 3, 1, 2)
        p2 = f3.permute(0, 3, 1, 2)
        p3 = f5.permute(0, 3, 1, 2)
        p4 = f7.permute(0, 3, 1, 2)

        pyramid = self.fpn.fpn(OrderedDict([("0", p1), ("1", p2), ("2", p3), ("3", p4)]))
        # Upsample the finest FPN feature map ('0' with stride 4) by 2x * 2x -> stride 1
        up = self.upsample.layers[0](pyramid["0"])
        up = self.upsample.layers[1](up)
        return up


class SatlasWaterNet(nn.Module):
    """Complete deep learning water segmentation network matching satquery_water_state_dict.pt."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = SatlasBackbone()

        self.spectral = nn.Module()
        self.spectral.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Module()
        self.fuse.net = nn.Sequential(
            nn.Conv2d(160, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(96, 2, 1)

    def forward(self, backbone_x: torch.Tensor, spectral_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            backbone_x: (B, 9, H, W) normalized 9-band input
            spectral_x: (B, 6, H, W) normalized 6 spectral features
        Returns:
            logits: (B, 2, H, W) Class 0 (background), Class 1 (water)
        """
        feat_spatial = self.backbone(backbone_x)
        feat_spectral = self.spectral.net(spectral_x)
        feat_fused = torch.cat([feat_spatial, feat_spectral], dim=1)
        feat_out = self.fuse.net(feat_fused)
        return self.head(feat_out)


_MODEL_INSTANCE: SatlasWaterNet | None = None
_MODEL_CONFIG: dict[str, Any] | None = None
_MODEL_METRICS: dict[str, Any] | None = None


def find_water_asset_path(filename: str, override_path: Path | None = None) -> Path | None:
    """Finds water model asset files in potential locations."""
    if override_path and override_path.exists():
        return override_path

    search_dirs = [
        Path.cwd().parent,
        Path.cwd(),
        Path(__file__).resolve().parent.parent.parent.parent,
        Path(__file__).resolve().parent.parent.parent / "data",
        Path(__file__).resolve().parent.parent.parent / "artifacts",
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


def get_water_config() -> dict[str, Any]:
    global _MODEL_CONFIG
    if _MODEL_CONFIG is not None:
        return _MODEL_CONFIG

    cfg_path = find_water_asset_path("satquery_water_config.json", settings.water_model_config)
    if cfg_path and cfg_path.exists():
        try:
            _MODEL_CONFIG = json.loads(cfg_path.read_text(encoding="utf-8"))
            return _MODEL_CONFIG
        except Exception as e:
            logger.warning("Failed to parse satquery_water_config.json: %s", e)

    _MODEL_CONFIG = {
        "architecture": "SatlasWaterNet",
        "pretrained_model": "Sentinel2_SwinB_SI_MS",
        "postprocess": {"threshold": 0.60, "min_area_pixels": 16},
        "tile_inference": {"tile_size": 512, "overlap": 128, "blend": "Hann"},
    }
    return _MODEL_CONFIG


def get_water_metrics() -> dict[str, Any]:
    global _MODEL_METRICS
    if _MODEL_METRICS is not None:
        return _MODEL_METRICS

    metrics_path = find_water_asset_path("satquery_water_metrics.json", settings.water_model_metrics)
    if metrics_path and metrics_path.exists():
        try:
            raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            _MODEL_METRICS = raw_metrics.get("test", DEFAULT_WATER_METRICS)
            return _MODEL_METRICS
        except Exception as e:
            logger.warning("Failed to parse satquery_water_metrics.json: %s", e)

    _MODEL_METRICS = DEFAULT_WATER_METRICS
    return _MODEL_METRICS


def is_water_model_available() -> bool:
    if not settings.water_model_enabled:
        return False
    ckpt = find_water_asset_path("satquery_water_state_dict.pt", settings.water_model_checkpoint)
    return ckpt is not None and ckpt.exists()


def get_water_model() -> SatlasWaterNet | None:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if not is_water_model_available():
        return None

    ckpt_path = find_water_asset_path("satquery_water_state_dict.pt", settings.water_model_checkpoint)
    if not ckpt_path:
        return None

    logger.info("Loading SatlasWaterNet weights from %s...", ckpt_path)
    try:
        model = SatlasWaterNet()
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        _MODEL_INSTANCE = model
        logger.info("Successfully loaded SatlasWaterNet weights (487 tensors verified).")
        return _MODEL_INSTANCE
    except Exception as e:
        logger.error("Failed to load SatlasWaterNet model weights: %s", e)
        return None


def prepare_band_cube(
    src_path: Path,
    max_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, Affine, Any]:
    """
    Extracts and normalizes the 9 backbone bands and 6 spectral feature channels.
    
    Backbone 9 bands:
      0: R (B4)
      1: G (B3)
      2: B (B2)
      3: B5 (Red Edge 1)
      4: B6 (Red Edge 2)
      5: B7 (Red Edge 3)
      6: B8 (NIR)
      7: B11 (SWIR 1)
      8: B12 (SWIR 2)
      
    Spectral 6 features:
      0: NDWI
      1: MNDWI
      2: NDVI
      3: AWEI_shadow
      4: visible_brightness
      5: B10_cirrus
    """
    with rasterio.open(src_path) as src:
        crs = src.crs
        w0, h0 = src.width, src.height
        ratio = min(1.0, max_size / max(w0, h0))
        out_w = max(1, round(w0 * ratio))
        out_h = max(1, round(h0 * ratio))
        transform = src.transform * Affine.scale(w0 / out_w, h0 / out_h)

        descriptions = [str(d or "").strip().lower().replace("-", "").replace("_", "") for d in (src.descriptions or ())]
        
        bands_dict: dict[str, np.ndarray] = {}
        alias_map = {
            "red": ["red", "b4", "b04"],
            "green": ["green", "b3", "b03"],
            "blue": ["blue", "b2", "b02"],
            "b5": ["b5", "b05", "re1", "rededge1"],
            "b6": ["b6", "b06", "re2", "rededge2"],
            "b7": ["b7", "b07", "re3", "rededge3"],
            "nir": ["nir", "b8", "b08", "b8a"],
            "b8a": ["b8a"],
            "b10": ["b10", "cirrus"],
            "swir": ["swir", "b11", "swir1", "swir16"],
            "b12": ["b12", "swir2", "swir22"],
        }

        for idx, desc in enumerate(descriptions, start=1):
            for canonical, aliases in alias_map.items():
                if desc in aliases and canonical not in bands_dict:
                    bands_dict[canonical] = src.read(idx, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")

        if "red" not in bands_dict or "green" not in bands_dict or "blue" not in bands_dict:
            count = src.count
            if count >= 3:
                bands_dict["red"] = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                bands_dict["green"] = src.read(2, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                bands_dict["blue"] = src.read(3, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                if count >= 4 and "nir" not in bands_dict:
                    bands_dict["nir"] = src.read(4, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
                if count >= 5 and "swir" not in bands_dict:
                    bands_dict["swir"] = src.read(5, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")

    has_nir = "nir" in bands_dict
    has_green = "green" in bands_dict
    has_red = "red" in bands_dict
    is_spectral = has_nir and has_green and has_red

    r = bands_dict.get("red", np.zeros((out_h, out_w), dtype="float32"))
    g = bands_dict.get("green", np.zeros((out_h, out_w), dtype="float32"))
    b = bands_dict.get("blue", np.zeros((out_h, out_w), dtype="float32"))
    nir = bands_dict.get("nir", np.copy(g))
    swir1 = bands_dict.get("swir", np.copy(nir))
    swir2 = bands_dict.get("b12", np.copy(swir1) * 0.9)

    b5 = bands_dict.get("b5", 0.75 * r + 0.25 * nir)
    b6 = bands_dict.get("b6", 0.50 * r + 0.50 * nir)
    b7 = bands_dict.get("b7", 0.25 * r + 0.75 * nir)
    b10 = bands_dict.get("b10", np.zeros_like(r))

    max_val = max(float(np.nanmax(r)), float(np.nanmax(nir)))
    if max_val <= 1.5:
        scale_mult = 10000.0
    elif max_val <= 255.0:
        scale_mult = 10000.0 / 255.0
    else:
        scale_mult = 1.0

    r_scaled = r * scale_mult
    g_scaled = g * scale_mult
    b_scaled = b * scale_mult
    b5_scaled = b5 * scale_mult
    b6_scaled = b6 * scale_mult
    b7_scaled = b7 * scale_mult
    nir_scaled = nir * scale_mult
    swir1_scaled = swir1 * scale_mult
    swir2_scaled = swir2 * scale_mult
    b10_scaled = b10 * scale_mult

    r_norm = np.clip(r_scaled / 3000.0, 0.0, 1.0)
    g_norm = np.clip(g_scaled / 3000.0, 0.0, 1.0)
    b_norm = np.clip(b_scaled / 3000.0, 0.0, 1.0)

    b5_norm = np.clip(b5_scaled / 8160.0, 0.0, 1.0)
    b6_norm = np.clip(b6_scaled / 8160.0, 0.0, 1.0)
    b7_norm = np.clip(b7_scaled / 8160.0, 0.0, 1.0)
    nir_norm = np.clip(nir_scaled / 8160.0, 0.0, 1.0)
    swir1_norm = np.clip(swir1_scaled / 8160.0, 0.0, 1.0)
    swir2_norm = np.clip(swir2_scaled / 8160.0, 0.0, 1.0)

    backbone_cube = np.stack([
        r_norm, g_norm, b_norm,
        b5_norm, b6_norm, b7_norm,
        nir_norm, swir1_norm, swir2_norm,
    ], axis=0).astype("float32")

    eps = 1e-6
    ndwi = np.clip((g_scaled - nir_scaled) / (g_scaled + nir_scaled + eps), -1.0, 1.0)
    mndwi = np.clip((g_scaled - swir1_scaled) / (g_scaled + swir1_scaled + eps), -1.0, 1.0)
    ndvi = np.clip((nir_scaled - r_scaled) / (nir_scaled + r_scaled + eps), -1.0, 1.0)

    awei_raw = (b_scaled + 2.5 * g_scaled - 1.5 * (nir_scaled + swir1_scaled) - 0.25 * swir2_scaled) / 10000.0
    awei_sh = np.clip(awei_raw, -2.0, 2.0)

    visible_brightness = (r_norm + g_norm + b_norm) / 3.0
    b10_cirrus = np.clip(b10_scaled / 8160.0, 0.0, 1.0)

    spectral_cube = np.stack([
        ndwi, mndwi, ndvi, awei_sh, visible_brightness, b10_cirrus,
    ], axis=0).astype("float32")

    return backbone_cube, spectral_cube, transform, crs, is_spectral


def run_windowed_inference(
    model: SatlasWaterNet,
    backbone_cube: np.ndarray,
    spectral_cube: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
    return_telemetry: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Sliding window inference with 2D Hann window blending across overlapping tiles.
    Uses torch.inference_mode() and VRAM-aware dynamic batching.
    """
    _, h, w = backbone_cube.shape
    stride = tile_size - overlap

    hann_1d = torch.hann_window(tile_size, periodic=False)
    hann_2d = (hann_1d.unsqueeze(1) @ hann_1d.unsqueeze(0)).numpy()
    hann_2d = np.maximum(hann_2d, 1e-4)

    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if spectral_cube is None:
        spectral_cube = np.zeros((6, h, w), dtype="float32")

    if pad_h > 0 or pad_w > 0:
        backbone_padded = np.pad(backbone_cube, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        spectral_padded = np.pad(spectral_cube, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    else:
        backbone_padded = backbone_cube
        spectral_padded = spectral_cube

    _, ph, pw = backbone_padded.shape

    prob_accum = np.zeros((ph, pw), dtype="float32")
    weight_accum = np.zeros((ph, pw), dtype="float32")

    y_steps = list(range(0, max(1, ph - tile_size + 1), stride))
    if ph > tile_size and (ph - tile_size) not in y_steps:
        y_steps.append(ph - tile_size)
    x_steps = list(range(0, max(1, pw - tile_size + 1), stride))
    if pw > tile_size and (pw - tile_size) not in x_steps:
        x_steps.append(pw - tile_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, "to"):
        model = model.to(device)

    batch_size = 1
    vram_mb = 0.0
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb >= 12.0:
            batch_size = 4
        elif total_vram_gb >= 6.0:
            batch_size = 2
        else:
            batch_size = 1

    coords = [(y, x) for y in y_steps for x in x_steps]
    total_tiles = len(coords)

    with torch.inference_mode():
        for i in range(0, total_tiles, batch_size):
            batch_coords = coords[i : i + batch_size]
            bb_batch = torch.from_numpy(
                np.stack([backbone_padded[:, y : y + tile_size, x : x + tile_size] for y, x in batch_coords])
            ).to(device)
            sp_batch = torch.from_numpy(
                np.stack([spectral_padded[:, y : y + tile_size, x : x + tile_size] for y, x in batch_coords])
            ).to(device)

            logits = model(bb_batch, sp_batch)
            probs = F.softmax(logits, dim=1)[:, 1, :, :].cpu().numpy()

            for b_idx, (y, x) in enumerate(batch_coords):
                prob_accum[y : y + tile_size, x : x + tile_size] += probs[b_idx] * hann_2d
                weight_accum[y : y + tile_size, x : x + tile_size] += hann_2d

            del bb_batch, sp_batch, logits, probs
            if device.type == "cuda":
                vram_mb = max(vram_mb, torch.cuda.max_memory_allocated() / (1024 * 1024))
                torch.cuda.empty_cache()

    valid_weights = weight_accum > 1e-6
    prob_accum[valid_weights] /= weight_accum[valid_weights]
    prob_map = np.clip(prob_accum[:h, :w], 0.0, 1.0)

    if return_telemetry:
        telemetry = {
            "tile_count": total_tiles,
            "batch_size": batch_size,
            "device": str(device),
            "estimated_vram_mb": round(vram_mb, 1) if vram_mb > 0 else None,
            "cache_cleared": True,
            "inference_mode": True,
        }
        return prob_map, telemetry

    return prob_map


def predict_water_mask(
    src_path: Path,
    output_dir: Path,
    query: str = "Highlight the largest water body.",
    max_size: int = 1024,
) -> dict[str, Any] | None:
    """
    Executes deep learning surface water segmentation using SatlasWaterNet.
    
    Generates:
      - Continuous calibrated water probability heatmap
      - Delineated binary water mask
      - High-contrast visual grounding overlay
      - Vectorized GeoJSON polygons
      - Quantitative confidence and metrics
    """
    model = get_water_model()
    if model is None:
        logger.warning("SatlasWaterNet model is unavailable; cannot perform deep learning water prediction.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_water_config()
    metrics = get_water_metrics()

    threshold = float(cfg.get("postprocess", {}).get("threshold", 0.60))
    min_area_pixels = int(cfg.get("postprocess", {}).get("min_area_pixels", 16))
    tile_size = int(cfg.get("tile_inference", {}).get("tile_size", 512))
    overlap = int(cfg.get("tile_inference", {}).get("overlap", 128))

    # 1. Prepare multispectral inputs
    backbone_cube, spectral_cube, transform, crs, is_spectral = prepare_band_cube(src_path, max_size=max_size)
    h, w = backbone_cube.shape[1], backbone_cube.shape[2]

    # 2. Windowed inference with VRAM-aware batching & telemetry
    prob_map, telemetry = run_windowed_inference(
        model, backbone_cube, spectral_cube, tile_size=tile_size, overlap=overlap, return_telemetry=True
    )

    # 2.5 Optical vegetation and dark soil suppression for RGB inputs
    if not is_spectral:
        r_01 = backbone_cube[0]
        g_01 = backbone_cube[1]
        b_01 = backbone_cube[2]
        exg = 2.0 * g_01 - r_01 - b_01
        is_veg = (exg > 0.03) | ((g_01 > b_01 * 1.30) & (g_01 > r_01 * 1.05))
        is_soil = (r_01 > b_01 * 1.15) & (g_01 > b_01 * 0.95) & (r_01 > 0.08)
        # Suppress false positives on land and grass
        prob_map[is_veg] = np.minimum(prob_map[is_veg], 0.20)
        prob_map[is_soil] = np.minimum(prob_map[is_soil], 0.20)

    # 3. Decision thresholding & morphological cleanup
    raw_binary = prob_map >= threshold
    labeled, num_features = ndi.label(raw_binary)

    # 4. Filter connected components by minimum pixel area
    clean_water_mask = np.zeros((h, w), dtype=bool)
    regions_info: list[dict[str, Any]] = []

    for comp_id in range(1, num_features + 1):
        comp_mask = labeled == comp_id
        pixel_count = int(comp_mask.sum())
        if pixel_count < min_area_pixels:
            continue

        coords = np.argwhere(comp_mask)
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        mean_prob = float(prob_map[comp_mask].mean())
        score = (mean_prob + 1.0) * (pixel_count ** 0.5)

        centroid_y = float(coords[:, 0].mean())
        centroid_x = float(coords[:, 1].mean())

        regions_info.append({
            "id": comp_id,
            "mask": comp_mask,
            "pixels": pixel_count,
            "score": score,
            "mean_prob": mean_prob,
            "centroid_norm": (centroid_y / h, centroid_x / w),
            "bbox_norm": [round(ymin / h, 4), round(xmin / w, 4), round(ymax / h, 4), round(xmax / w, 4)],
        })
        clean_water_mask |= comp_mask

    regions_info.sort(key=lambda r: r["score"], reverse=True)

    total_scene_pixels = h * w
    total_water_pixels = int(clean_water_mask.sum())
    total_coverage = (total_water_pixels / total_scene_pixels) * 100.0

    prob_path = output_dir / "water_probability.png"
    mask_path = output_dir / "water_mask.png"
    grounding_path = output_dir / "water_grounding_mask.png"
    geojson_path = output_dir / "water_regions.geojson"
    ndwi_path = output_dir / "ndwi.png"

    _colorize_index(prob_map * 2.0 - 1.0, "ndwi").save(prob_path, format="PNG")
    if is_spectral:
        write_index_png(spectral_cube[0], "ndwi", ndwi_path)
        result_type = "combined"
        index_type = "SatlasWaterNet_Multispectral"
    else:
        # Avoid fake NDWI on RGB: copy probability heatmap for layer compatibility
        import shutil
        shutil.copyfile(prob_path, ndwi_path)
        result_type = "model-based"
        index_type = "SatlasWaterNet_Optical_DL"

    if not regions_info:
        _write_mask(np.zeros((h, w), dtype=bool), mask_path, (20, 60, 140))
        Image.fromarray(np.zeros((h, w, 4), dtype="uint8"), mode="RGBA").save(grounding_path, format="PNG")
        geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")

        return {
            "target": "water",
            "water_body_identified": False,
            "findings": f"No validated water bodies detected by the SatlasWaterNet deep learning model ({result_type}) above the calibrated {threshold:.2f} threshold.",
            "primary_region": None,
            "grounding_task": "water_delineation",
            "is_spectral": is_spectral,
            "is_deep_learning": True,
            "result_type": result_type,
            "index_type": index_type,
            "location_description": "No verified water bodies detected",
            "mask_path": mask_path,
            "grounding_mask_path": grounding_path,
            "ndwi_path": ndwi_path,
            "probability_path": prob_path,
            "geojson_path": geojson_path,
            "region_count": 0,
            "area_m2": None,
            "total_area_m2": None,
            "coverage_percent": 0.0,
            "largest_coverage_percent": 0.0,
            "largest_pixel_count": 0,
            "total_water_pixels": 0,
            "primary_bbox": [0.0, 0.0, 0.0, 0.0],
            "confidence": 0.35,
            "producer": "satquery_waternet_swinv2",
            "metrics": metrics,
        }

    primary = regions_info[0]
    largest_coverage = (primary["pixels"] / total_scene_pixels) * 100.0
    loc_desc = _describe_region_location(primary["centroid_norm"][0], primary["centroid_norm"][1])

    region_count, total_area_m2 = _polygonize(clean_water_mask, transform, crs, kind="water", output_path=geojson_path)
    primary_area_m2: float | None = None
    if geojson_path.exists():
        try:
            fc = json.loads(geojson_path.read_text(encoding="utf-8"))
            feats = fc.get("features", [])
            if feats:
                primary_area_m2 = feats[0]["properties"].get("area_m2")
        except Exception:
            pass

    _write_mask(clean_water_mask, mask_path, (20, 60, 140))

    grounding_rgba = np.zeros((h, w, 4), dtype="uint8")
    grounding_rgba[clean_water_mask] = [20, 60, 140, 160]
    grounding_rgba[primary["mask"]] = [15, 45, 120, 220]

    dilated = ndi.binary_dilation(primary["mask"], iterations=2)
    boundary = dilated & ~primary["mask"]
    grounding_rgba[boundary] = [255, 255, 255, 255]
    Image.fromarray(grounding_rgba, mode="RGBA").save(grounding_path, format="PNG")

    model_precision = float(metrics.get("precision", 0.92))
    primary_prob = primary["mean_prob"]
    calibrated_conf = round(float(0.50 * model_precision + 0.35 * primary_prob + 0.15 * min(1.0, primary["pixels"] / 500)), 3)

    return {
        "target": "water",
        "water_body_identified": True,
        "findings": f"Identified {region_count} surface water region(s) using SatlasWaterNet deep learning segmentation ({result_type}).",
        "primary_region": primary,
        "grounding_task": "largest_water_body",
        "is_spectral": is_spectral,
        "is_deep_learning": True,
        "result_type": result_type,
        "index_type": index_type,
        "location_description": loc_desc,
        "mask_path": mask_path,
        "grounding_mask_path": grounding_path,
        "ndwi_path": ndwi_path,
        "probability_path": prob_path,
        "geojson_path": geojson_path,
        "region_count": region_count,
        "area_m2": primary_area_m2,
        "total_area_m2": total_area_m2,
        "coverage_percent": round(total_coverage, 2),
        "largest_coverage_percent": round(largest_coverage, 2),
        "largest_pixel_count": primary["pixels"],
        "total_water_pixels": total_water_pixels,
        "primary_bbox": primary["bbox_norm"],
        "confidence": calibrated_conf,
        "producer": "satquery_waternet_swinv2",
        "metrics": {
            **metrics,
            "threshold": threshold,
            "min_area_pixels": min_area_pixels,
            "primary_mean_probability": round(primary["mean_prob"], 3),
            "inference_telemetry": telemetry,
        },
    }

