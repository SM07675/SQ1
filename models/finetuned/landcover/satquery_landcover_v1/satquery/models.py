from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .registry import ModelSpec


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models
        import torchvision.ops as tv_ops
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchvision are required for trained-model inference. Install requirements.txt.") from exc
    return torch, nn, tv_models, tv_ops


def _architecture_classes():
    torch, nn, tv_models, tv_ops = _torch()

    class SwinBackbone(nn.Module):
        def __init__(self, channels: int = 3):
            super().__init__()
            self.backbone = tv_models.swin_v2_b(weights=None)
            if channels != 3:
                patch = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(channels, patch.out_channels, kernel_size=patch.kernel_size, stride=patch.stride, padding=patch.padding, bias=patch.bias is not None)

        def forward(self, x):
            outputs = []
            for layer in self.backbone.features:
                x = layer(x)
                outputs.append(x.permute(0, 3, 1, 2))
            return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]

    class SatlasFPN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fpn = tv_ops.FeaturePyramidNetwork([128, 256, 512, 1024], 128)

        def forward(self, values):
            pyramid = self.fpn(OrderedDict((f"feat{i}", value) for i, value in enumerate(values)))
            return list(pyramid.values())

    class UpsampleBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)),
                nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)),
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class FullBackbone(nn.Module):
        def __init__(self, channels: int = 3):
            super().__init__()
            self.backbone = SwinBackbone(channels)
            self.fpn = SatlasFPN()
            self.upsample = UpsampleBlock()

        def forward(self, x):
            return self.upsample(self.fpn(self.backbone(x))[0])

    class RefineBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64, track_running_stats=False), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64, track_running_stats=False), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class BuildingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FullBackbone(3)
            self.refine = RefineBlock()
            self.head = nn.Conv2d(64, 2, 1)

        def forward(self, x):
            return self.head(self.refine(self.backbone(x)))

    class SpectralBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(6, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32, track_running_stats=False), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32, track_running_stats=False), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class FuseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(160, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96, track_running_stats=False), nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96, track_running_stats=False), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class WaterNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FullBackbone(9)
            self.spectral = SpectralBlock()
            self.fuse = FuseBlock()
            self.head = nn.Conv2d(96, 2, 1)

        def forward(self, image, spectral):
            return self.head(self.fuse(torch.cat((self.backbone(image), self.spectral(spectral)), dim=1)))

    return BuildingNet, WaterNet


class ModelCache:
    """Process-local cache; each checkpoint is loaded at most once per device."""

    def __init__(self, device: str | None = None, mixed_precision: bool = True):
        torch, _, _, _ = _torch()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self._models: dict[str, Any] = {}

    def _state(self, path: Path, key: str | None = None):
        torch, _, _, _ = _torch()
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return obj[key] if key else obj

    def landcover(self, spec: ModelSpec):
        if spec.key not in self._models:
            spec.require_available()
            try:
                import segmentation_models_pytorch as smp
            except ImportError as exc:
                raise RuntimeError("RGB land-cover inference requires segmentation-models-pytorch>=0.5.") from exc
            model = smp.Unet(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=15, activation=None)
            model.load_state_dict(self._state(spec.checkpoint, "model"), strict=True)
            self._models[spec.key] = model.eval().to(self.device)
        return self._models[spec.key]

    def building(self, spec: ModelSpec):
        if spec.key not in self._models:
            spec.require_available()
            BuildingNet, _ = _architecture_classes()
            model = BuildingNet()
            model.load_state_dict(self._state(spec.checkpoint), strict=True)
            self._models[spec.key] = model.eval().to(self.device)
        return self._models[spec.key]

    def water(self, spec: ModelSpec):
        if spec.key not in self._models:
            spec.require_available()
            _, WaterNet = _architecture_classes()
            model = WaterNet()
            model.load_state_dict(self._state(spec.checkpoint), strict=True)
            self._models[spec.key] = model.eval().to(self.device)
        return self._models[spec.key]

    def _run(self, model, *arrays: np.ndarray, kind: str) -> np.ndarray:
        torch, _, _, _ = _torch()
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(self.device) for a in arrays]
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            logits = model(*tensors)[0].float()
            probability = torch.sigmoid(logits) if kind == "sigmoid" else torch.softmax(logits, dim=0)
        return probability.cpu().numpy().astype("float32")

    def predict_landcover(self, spec: ModelSpec, normalized_tile: np.ndarray) -> np.ndarray:
        return self._run(self.landcover(spec), normalized_tile, kind="softmax")

    def predict_buildings(self, spec: ModelSpec, unit_tile: np.ndarray) -> np.ndarray:
        return self._run(self.building(spec), unit_tile, kind="sigmoid")

    def predict_water(self, spec: ModelSpec, backbone_tile: np.ndarray, spectral_tile: np.ndarray) -> np.ndarray:
        return self._run(self.water(spec), backbone_tile, spectral_tile, kind="softmax")
