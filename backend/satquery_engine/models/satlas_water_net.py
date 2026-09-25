from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from satquery_engine.models.satlas_building_net import FullBackbone


BACKBONE_BANDS = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")
SOURCE_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B10", "B11", "B12")
SPECTRAL_FEATURES = ("NDWI", "MNDWI", "NDVI", "AWEI_SHADOW", "VISIBLE_BRIGHTNESS", "B10_CIRRUS")


class SpectralBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FuseBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(160, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SatlasWaterNet(nn.Module):
    """Fine-tuned Sentinel-2 surface-water model.

    The model has a 9-band Satlas image backbone and a distinct six-channel
    deterministic spectral-feature branch.  Its two output channels are
    background and surface-water logits; shadow remains an independent
    illumination product in the SatQuery pipeline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = FullBackbone(input_channels=9)
        self.spectral = SpectralBlock()
        self.fuse = FuseBlock()
        self.head = nn.Conv2d(96, 2, 1)

    def forward(self, image: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(image)
        spectral_features = self.spectral(spectral)
        return self.head(self.fuse(torch.cat((image_features, spectral_features), dim=1)))


def prepare_water_inputs(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact preprocessing contract recorded in the bundle config.

    Arrays are expected in Sentinel-2 digital-number scale (approximately
    0..10000). Float reflectance arrays in 0..1 are converted back to that
    documented scale before normalization.
    """
    missing = [band for band in SOURCE_BANDS if band not in source]
    if missing:
        raise ValueError(f"SatQuery water model is missing required Sentinel-2 bands: {', '.join(missing)}")

    arrays = {name: np.asarray(source[name], dtype="float32") for name in SOURCE_BANDS}
    shape = arrays[SOURCE_BANDS[0]].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("SatQuery water model source bands do not share one pixel grid.")

    valid = np.ones(shape, dtype=bool)
    for value in arrays.values():
        valid &= np.isfinite(value)

    samples = [value[valid][:2048] for value in arrays.values()] if valid.any() else []
    finite_values = np.concatenate(samples) if samples else np.empty(0, dtype="float32")
    if finite_values.size and float(np.nanpercentile(np.abs(finite_values), 99)) <= 2.0:
        arrays = {name: value * 10000.0 for name, value in arrays.items()}

    b02, b03, b04 = arrays["B02"], arrays["B03"], arrays["B04"]
    b05, b06, b07 = arrays["B05"], arrays["B06"], arrays["B07"]
    b08, b10, b11, b12 = arrays["B08"], arrays["B10"], arrays["B11"], arrays["B12"]

    backbone = np.stack(
        (
            np.clip(b04 / 3000.0, 0.0, 1.0),
            np.clip(b03 / 3000.0, 0.0, 1.0),
            np.clip(b02 / 3000.0, 0.0, 1.0),
            np.clip(b05 / 8160.0, 0.0, 1.0),
            np.clip(b06 / 8160.0, 0.0, 1.0),
            np.clip(b07 / 8160.0, 0.0, 1.0),
            np.clip(b08 / 8160.0, 0.0, 1.0),
            np.clip(b11 / 8160.0, 0.0, 1.0),
            np.clip(b12 / 8160.0, 0.0, 1.0),
        )
    ).astype("float32")

    reflectance = {name: value / 10000.0 for name, value in arrays.items()}
    blue, green, red = reflectance["B02"], reflectance["B03"], reflectance["B04"]
    nir, cirrus = reflectance["B08"], reflectance["B10"]
    swir1, swir2 = reflectance["B11"], reflectance["B12"]

    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a - b, a + b + 1e-6, out=np.zeros_like(a), where=valid)

    ndwi = ratio(green, nir)
    mndwi = ratio(green, swir1)
    ndvi = ratio(nir, red)
    # AWEIsh from Feyisa et al.; this feature is intentionally distinct from
    # the pipeline's independent shadow/illumination classification.
    awei_shadow = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    visible_brightness = (red + green + blue) / 3.0
    spectral = np.stack((ndwi, mndwi, ndvi, awei_shadow, visible_brightness, cirrus)).astype("float32")
    backbone[:, ~valid] = 0.0
    spectral[:, ~valid] = 0.0
    return backbone, spectral, valid


@lru_cache(maxsize=1)
def load_water_bundle(bundle_dir: str | Path) -> tuple[SatlasWaterNet, dict[str, Any], dict[str, Any]]:
    bundle_path = Path(bundle_dir)
    config_file = bundle_path / "satquery_water_config.json"
    state_file = bundle_path / "satquery_water_state_dict.pt"
    metrics_file = bundle_path / "satquery_water_metrics.json"
    thresholds_file = bundle_path / "thresholds.json"
    if not config_file.is_file() or not state_file.is_file():
        raise ValueError(f"Water bundle at {bundle_path} is missing config or state_dict.")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    if thresholds_file.is_file():
        thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
        config.setdefault("postprocess", {}).update({
            "threshold": thresholds["water_probability_threshold"],
            "min_area_pixels": thresholds["minimum_component_area_pixels"],
        })
        config["threshold_source"] = str(thresholds_file)
    if config.get("architecture") != "SatlasWaterNet":
        raise ValueError("Water bundle architecture is not SatlasWaterNet.")
    model = SatlasWaterNet()
    state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else {}
    return model, config, metrics
