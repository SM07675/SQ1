"""Selected-model and instance extraction contracts for the fine-tuned bundle."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery_engine.models.satlas_building_net import validate_building_config
from satquery_engine.services.buildings import separate_instances_satlas


def test_configured_separation_suppresses_duplicate_roof_centers():
    probability = np.zeros((90, 110), dtype="float32")
    probability[20:60, 20:80] = .9
    boundary = np.zeros_like(probability)
    boundary[20:60, 48:52] = .9
    valid = np.ones_like(probability, dtype=bool)
    fine, _ = separate_instances_satlas(probability, boundary, valid,
        min_area=1, min_distance=8, h_prominence=2)
    coarse, _ = separate_instances_satlas(probability, boundary, valid,
        min_area=1, min_distance=40, h_prominence=2)
    assert fine.max() == 2
    assert coarse.max() == 1
    np.testing.assert_array_equal(fine > 0, probability > .6)
    np.testing.assert_array_equal(coarse > 0, fine > 0)


def test_invalid_boundary_probabilities_cannot_create_footprints():
    probability = np.full((20, 20), .9)
    boundary = np.zeros_like(probability)
    boundary[5, 5] = np.nan
    with pytest.raises(ValueError, match="finite probabilities"):
        separate_instances_satlas(probability, boundary, np.ones_like(probability, bool))


@pytest.mark.parametrize("section,key,value", [
    ("input", "scale", "unknown"),
    ("input", "tile_size", 511),
    ("tile_inference", "overlap", 512),
    ("postprocess", "foot_thr", float("nan")),
    ("postprocess", "min_distance", 0),
    ("postprocess", "min_area", -1),
    ("postprocess", "h_prominence", float("inf")),
])
def test_invalid_bundle_settings_fail_before_inference(section, key, value):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "models/satquery_buildings_bundle/satquery_buildings_config.json").read_text())
    validate_building_config(config)
    invalid = copy.deepcopy(config)
    invalid[section][key] = value
    with pytest.raises(ValueError):
        validate_building_config(invalid)


def test_selected_bundle_failure_does_not_run_another_checkpoint(tmp_path, monkeypatch):
    import satquery_engine.services.buildings as buildings
    source = tmp_path / "input.tif"
    with rasterio.open(source, "w", driver="GTiff", width=32, height=32, count=3,
                       dtype="uint8", crs="EPSG:32632", transform=from_origin(500000, 4000000, .5, .5)) as dst:
        dst.write(np.full((3, 32, 32), 100, dtype="uint8"))
    selected = tmp_path / "selected_bundle"
    selected.mkdir()
    monkeypatch.setattr(buildings, "_resolve_checkpoint", lambda: selected)
    calls = []
    def unavailable(path):
        calls.append(path)
        raise ValueError("Selected bundle cannot load")
    monkeypatch.setattr(buildings, "load_session", unavailable)
    with pytest.raises(ValueError, match="Selected bundle cannot load"):
        buildings.detect_buildings(source, tmp_path / "out")
    assert calls == [str(selected)]
