from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satquery_engine.config import settings
from satquery_engine.models.adapters import SatlasBuildingAdapter, SatlasWaterAdapter
from satquery_engine.models.manifest import HealthStatus
from satquery_engine.models.registry import LocalModelRegistry
from satquery_engine.services.model_registry import ModelRegistry
from satquery_engine.services.bands import detect_band_map


def _registry():
    return LocalModelRegistry(settings.model_dir, ModelRegistry.get_manifest_path())


def test_finetuned_bundles_are_registered_with_distinct_adapters():
    models = _registry().load(refresh=True)
    building = models["building_primary"]
    water = models["water_finetuned"]
    assert building.health_status == HealthStatus.READY
    assert water.health_status == HealthStatus.READY
    assert building.adapter == "SatlasBuildingAdapter"
    assert water.adapter == "SatlasWaterAdapter"
    assert building.input_channels == 3
    assert water.input_channels == 10


def test_water_bundle_loads_strictly_and_runs_a_lightweight_smoke_inference():
    torch = pytest.importorskip("torch")
    from satquery_engine.models.satlas_water_net import load_water_bundle

    bundle = settings.model_dir / "satquery_water_bundle"
    if not bundle.is_dir():
        pytest.skip("fine-tuned water bundle is not installed")
    model, config, _ = load_water_bundle(bundle)
    assert config["architecture"] == "SatlasWaterNet"
    with torch.inference_mode():
        output = model(torch.zeros(1, 9, 64, 64), torch.zeros(1, 6, 64, 64))
    assert tuple(output.shape) == (1, 2, 64, 64)
    assert bool(torch.isfinite(output).all())


def test_water_adapter_builds_backbone_and_spectral_tensors_without_rgb_substitution():
    manifest = _registry().load()["water_finetuned"]
    adapter = SatlasWaterAdapter(manifest)
    raster = np.full((10, 16, 16), 1200, dtype="uint16")
    adapted = adapter.preprocess(raster, tuple(range(1, 11)))
    assert adapted.image_tensor.shape == (9, 16, 16)
    assert adapted.spectral_tensor.shape == (6, 16, 16)
    assert adapted.valid_mask.all()
    assert np.isfinite(adapted.spectral_tensor).all()
    nodata = adapter.preprocess(np.full((10, 4, 4), np.nan, dtype="float32"), tuple(range(1, 11)))
    assert not nodata.valid_mask.any()
    assert not nodata.image_tensor.any()
    assert not nodata.spectral_tensor.any()
    with pytest.raises(ValueError, match="exact Sentinel-2"):
        adapter.preprocess(raster[:3], (1, 2, 3))


def test_building_adapter_preserves_unit_rgb_and_scales_uint8_rgb():
    manifest = _registry().load()["building_primary"]
    adapter = SatlasBuildingAdapter(manifest)
    unit = np.full((3, 8, 8), 0.5, dtype="float32")
    byte = np.full((3, 8, 8), 128, dtype="uint8")
    assert np.allclose(adapter.preprocess(unit, (1, 2, 3)).tensor, 0.5)
    assert np.allclose(adapter.preprocess(byte, (1, 2, 3)).tensor, 128 / 255)


def test_exact_sentinel2_band_names_are_preserved_for_water_routing():
    names = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")

    class Color:
        name = "undefined"

    class Source:
        count = len(names)
        descriptions = names
        colorinterp = tuple(Color() for _ in names)

        @staticmethod
        def tags(index=None):
            return {} if index is None else {"name": names[index - 1]}

    band_map = detect_band_map(Source())
    assert band_map.indices["green"] == 3
    assert band_map.indices["nir"] == 8
    assert [band_map.indices[f"b{i:02d}"] for i in (2, 3, 4, 5, 6, 7, 8, 10, 11, 12)] == [
        2, 3, 4, 5, 6, 7, 8, 11, 12, 13
    ]
