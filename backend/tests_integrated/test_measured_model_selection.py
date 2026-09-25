from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery_engine.config import default_building_checkpoint


def test_complete_satellite_checkpoint_precedes_old_bundle_registration(tmp_path):
    manifest = tmp_path / "manifests/models.yaml"
    manifest.parent.mkdir()
    manifest.write_text("models:\n  building_primary:\n    path: finetuned/old_bundle\n")
    satellite = tmp_path / "buildings/rf-detr-seg-satellite-buildings"
    satellite.mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        (satellite / name).write_text("test placeholder")
    assert default_building_checkpoint(tmp_path) == satellite
    (satellite / "preprocessor_config.json").unlink()
    assert default_building_checkpoint(tmp_path) == tmp_path / "finetuned/old_bundle"


def test_explicit_missing_model_raises_original_error_without_fallback(tmp_path, monkeypatch):
    from satquery_engine.services import buildings
    image = tmp_path / "rgb.tif"
    with rasterio.open(image, "w", driver="GTiff", width=32, height=32, count=3,
                       dtype="uint8", crs="EPSG:32632", transform=from_origin(500000, 4000000, .5, .5)) as dst:
        dst.write(np.full((3, 32, 32), 100, dtype="uint8"))
    missing = tmp_path / "missing-model"
    monkeypatch.setenv("SATQUERY_BUILDING_CHECKPOINT", str(missing))
    monkeypatch.setattr(buildings, "settings", SimpleNamespace(building_checkpoint=missing))
    with pytest.raises(ValueError, match="currently unavailable"):
        buildings.detect_buildings(image, tmp_path / "out")
