from __future__ import annotations

import json
import math
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops


class SwinBackbone(nn.Module):
    """Swin-v2-Base backbone extracting multi-scale feature maps at strides 4, 8, 16, 32."""

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.backbone = models.swin_v2_b()
        if input_channels != 3:
            # Torchvision's patch embed is the layer used by the Satlas
            # checkpoints.  Replacing only that convolution preserves the
            # checkpoint key layout while allowing the documented 9-band
            # Sentinel-2 water backbone to load strictly.
            patch = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels,
                patch.out_channels,
                kernel_size=patch.kernel_size,
                stride=patch.stride,
                padding=patch.padding,
                bias=patch.bias is not None,
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class SatlasFPN(nn.Module):
    """Feature Pyramid Network mapping backbone channels [128, 256, 512, 1024] to 128."""

    def __init__(self):
        super().__init__()
        self.fpn = ops.FeaturePyramidNetwork([128, 256, 512, 1024], 128)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        inp = OrderedDict([(f"feat{i}", el) for i, el in enumerate(x)])
        out = self.fpn(inp)
        return list(out.values())


class UpsampleBlock(nn.Module):
    """Two-stage transposed convolution upsampler (4x total) restoring full resolution."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
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
        for layer in self.layers:
            x = layer(x)
        return x


class FullBackbone(nn.Module):
    """Combines SwinBackbone, SatlasFPN, and UpsampleBlock."""

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.backbone = SwinBackbone(input_channels=input_channels)
        self.fpn = SatlasFPN()
        self.upsample = UpsampleBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        fpn_feats = self.fpn(feats)
        return self.upsample(fpn_feats[0])


class RefineBlock(nn.Module):
    """Refinement block matching Satlas decoder head."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SatlasBuildingNet(nn.Module):
    """Fine-tuned building detection model (SatlasBuildingNet on SpaceNet 2).

    Outputs 2 channels:
      Channel 0: footprint_logit
      Channel 1: boundary_logit
    """

    def __init__(self):
        super().__init__()
        self.backbone = FullBackbone()
        self.refine = RefineBlock()
        self.head = nn.Conv2d(64, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.refine(feat)
        return self.head(feat)


@lru_cache(maxsize=1)
def load_building_bundle(bundle_dir: str | Path) -> Tuple[SatlasBuildingNet, dict[str, Any], dict[str, Any]]:
    """Load and cache the SatlasBuildingNet model, config, and metrics from bundle directory."""
    bundle_path = Path(bundle_dir)
    config_file = bundle_path / "satquery_buildings_config.json"
    state_file = bundle_path / "satquery_buildings_state_dict.pt"
    metrics_file = bundle_path / "satquery_buildings_metrics.json"
    thresholds_file = bundle_path / "thresholds.json"

    if not config_file.is_file() or not state_file.is_file():
        raise ValueError(f"Building bundle at {bundle_path} is missing config or state_dict.")

    config = json.loads(config_file.read_text())
    if thresholds_file.is_file():
        thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
        postprocess = config.setdefault("postprocess", {})
        postprocess.update({
            "foot_thr": thresholds["building_probability_threshold"],
            "boundary_thr": thresholds["boundary_probability_threshold"],
            "min_area": thresholds["instance_min_area_pixels"],
            "h_prominence": thresholds["watershed_h_prominence"],
            "min_distance": thresholds["watershed_min_distance"],
        })
        config["threshold_source"] = str(thresholds_file)
    validate_building_config(config)
    metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}

    model = SatlasBuildingNet()
    state_dict = torch.load(state_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, config, metrics


def validate_building_config(config: dict[str, Any]) -> None:
    """Reject incompatible bundle contracts before allocating/loading weights."""
    if config.get("architecture") != "SatlasBuildingNet":
        raise ValueError("Unsupported building bundle architecture.")
    inputs = config.get("input", {})
    if inputs.get("channels") != "RGB" or inputs.get("scale") != "divide_by_255":
        raise ValueError("Building bundle requires the supported RGB / 255 input contract.")
    if config.get("outputs") != ["footprint_logit", "boundary_logit"]:
        raise ValueError("Building bundle output order must be footprint then boundary logits.")
    tile = inputs.get("tile_size", 512)
    overlap = config.get("tile_inference", {}).get("overlap", 128)
    if (not isinstance(tile, int) or tile < 32 or tile % 32
            or not isinstance(overlap, int) or not 0 <= overlap < tile):
        raise ValueError("Building tile size must be a positive multiple of 32 with valid overlap.")
    post = config.get("postprocess", {})
    for name in ("foot_thr", "boundary_thr"):
        value = post.get(name, .6)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"Invalid building {name} probability threshold.")
    for name, default in (("min_area", 48), ("min_distance", 8)):
        value = post.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Building {name} must be a positive integer.")
    prominence = post.get("h_prominence", 6.0)
    if not isinstance(prominence, (int, float)) or not math.isfinite(prominence) or prominence <= 0:
        raise ValueError("Building prominence must be finite and positive.")
