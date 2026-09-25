from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from satquery_engine.models.manifest import ModelManifest


@dataclass(frozen=True)
class AdaptedInput:
    tensor: np.ndarray
    selected_channels: tuple[int, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class AdaptedWaterInput:
    image_tensor: np.ndarray
    spectral_tensor: np.ndarray
    valid_mask: np.ndarray
    selected_channels: tuple[int, ...]
    audit: dict[str, Any]


class GenericModelAdapter:
    """Checkpoint-specific preprocessing contract; never guesses missing bands."""

    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest

    def preprocess(self, image: np.ndarray, selected_channels: tuple[int, ...]) -> AdaptedInput:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError("Model input must be a channel-first 3-D scientific raster.")
        if len(selected_channels) != self.manifest.input_channels:
            raise ValueError(
                f"{self.manifest.model_id} requires {self.manifest.input_channels} ordered channels; "
                f"received {len(selected_channels)}."
            )
        zero_based = tuple(index - 1 for index in selected_channels)
        if min(zero_based, default=0) < 0 or max(zero_based, default=0) >= array.shape[0]:
            raise ValueError("Selected channel is outside the raster band range.")
        tensor = array[list(zero_based)].astype(self.manifest.input_dtype, copy=False)
        norm = self.manifest.normalization
        if norm.get("type") == "mean_std":
            mean = np.asarray(norm["mean"], dtype="float32")[:, None, None]
            std = np.asarray(norm["std"], dtype="float32")[:, None, None]
            tensor = (tensor - mean) / np.maximum(std, 1e-7)
        elif norm.get("type") == "scale":
            tensor = tensor * float(norm.get("factor", 1.0))
        elif norm.get("type") not in {None, "identity", "checkpoint_native"}:
            raise ValueError(f"Unsupported normalization contract: {norm.get('type')}")
        return AdaptedInput(
            tensor=tensor,
            selected_channels=selected_channels,
            audit={
                "model_id": self.manifest.model_id,
                "adapter": type(self).__name__,
                "input_shape": list(array.shape),
                "output_shape": list(tensor.shape),
                "selected_channels": list(selected_channels),
                "normalization": norm,
                "dtype": str(tensor.dtype),
            },
        )

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        result = np.asarray(output, dtype="float32")
        if "probability" in self.manifest.output_type and (result.min(initial=0) < 0 or result.max(initial=0) > 1):
            result = 1.0 / (1.0 + np.exp(-result))
        return np.clip(result, 0.0, 1.0) if "probability" in self.manifest.output_type else result


# Public architectural name retained while concrete adapters stay explicit.
ModelAdapter = GenericModelAdapter


class DinoBuildingAdapter(GenericModelAdapter):
    pass


class FlairLandCoverAdapter(GenericModelAdapter):
    pass


class PrithviWaterAdapter(GenericModelAdapter):
    pass


class BigEarthNetS2Adapter(GenericModelAdapter):
    pass


class BigEarthNetS1Adapter(GenericModelAdapter):
    pass


class BigEarthNetFusionAdapter(GenericModelAdapter):
    pass


class EarthDialAdapter(GenericModelAdapter):
    pass


class RemoteClipAdapter(GenericModelAdapter):
    pass


class CromaAdapter(GenericModelAdapter):
    pass


class ChangeDetectionAdapter(GenericModelAdapter):
    pass


class SatlasBuildingAdapter(GenericModelAdapter):
    def preprocess(self, image: np.ndarray, selected_channels: tuple[int, ...]) -> AdaptedInput:
        array = np.asarray(image)
        if array.ndim != 3 or len(selected_channels) != 3:
            raise ValueError("Satlas building inference requires channel-first RGB imagery.")
        zero_based = tuple(index - 1 for index in selected_channels)
        tensor = array[list(zero_based)].astype("float32", copy=False)
        if np.nanmax(tensor, initial=0.0) > 1.5:
            tensor = tensor / 255.0
        tensor = np.clip(tensor, 0.0, 1.0)
        return AdaptedInput(
            tensor=tensor,
            selected_channels=selected_channels,
            audit={
                "model_id": self.manifest.model_id,
                "adapter": type(self).__name__,
                "input_shape": list(array.shape),
                "output_shape": list(tensor.shape),
                "selected_channels": list(selected_channels),
                "normalization": "uint8/255 or identity for unit RGB",
                "dtype": str(tensor.dtype),
            },
        )


class SatlasWaterAdapter(GenericModelAdapter):
    def preprocess(self, image: np.ndarray, selected_channels: tuple[int, ...]) -> AdaptedWaterInput:
        from satquery_engine.models.satlas_water_net import SOURCE_BANDS, prepare_water_inputs

        array = np.asarray(image)
        if array.ndim != 3 or len(selected_channels) != len(SOURCE_BANDS):
            raise ValueError(
                f"Satlas water inference requires {len(SOURCE_BANDS)} exact Sentinel-2 source bands."
            )
        zero_based = tuple(index - 1 for index in selected_channels)
        if min(zero_based) < 0 or max(zero_based) >= array.shape[0]:
            raise ValueError("Selected Sentinel-2 channel is outside the raster band range.")
        source = {name: array[index] for name, index in zip(SOURCE_BANDS, zero_based)}
        backbone, spectral, valid = prepare_water_inputs(source)
        return AdaptedWaterInput(
            image_tensor=backbone,
            spectral_tensor=spectral,
            valid_mask=valid,
            selected_channels=selected_channels,
            audit={
                "model_id": self.manifest.model_id,
                "adapter": type(self).__name__,
                "input_shape": list(array.shape),
                "image_tensor_shape": list(backbone.shape),
                "spectral_tensor_shape": list(spectral.shape),
                "selected_channels": list(selected_channels),
                "normalization": self.manifest.normalization,
                "dtype": str(backbone.dtype),
            },
        )


class WaterShadowAdapter(GenericModelAdapter):
    pass


BigEarthNetAdapter = BigEarthNetS2Adapter
ChangeModelAdapter = ChangeDetectionAdapter


_ADAPTERS = {cls.__name__: cls for cls in (
    GenericModelAdapter, DinoBuildingAdapter, SatlasBuildingAdapter, SatlasWaterAdapter, FlairLandCoverAdapter, PrithviWaterAdapter,
    BigEarthNetS2Adapter, BigEarthNetS1Adapter, BigEarthNetFusionAdapter, BigEarthNetAdapter,
    EarthDialAdapter, RemoteClipAdapter, CromaAdapter, ChangeDetectionAdapter, ChangeModelAdapter,
    WaterShadowAdapter,
)}


def adapter_for(manifest: ModelManifest) -> GenericModelAdapter:
    try:
        adapter_type = _ADAPTERS[manifest.adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {manifest.adapter!r} for {manifest.model_id}") from exc
    return adapter_type(manifest)
