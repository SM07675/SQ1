"""Counterexamples for counting, scene validity and satellite model routing."""
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from scipy import ndimage

from satquery_engine.config import default_building_checkpoint
from satquery_engine.services.buildings import separate_instances
from satquery_engine.services.radiometry import rgb_unit_data
from satquery_engine.services.landcover_specialist import compatibility


@pytest.mark.parametrize("width", [60, 180])
def test_long_single_roof_is_not_counted_once_per_distance_peak(width):
    mask = np.zeros((100, 220), bool)
    mask[30:50, 10:10 + width] = True
    distance = ndimage.distance_transform_edt(mask).astype("float32")
    labels, rejected = separate_instances(mask.astype(float), distance, np.ones_like(mask))
    assert labels.max() == 1
    assert not rejected
    np.testing.assert_array_equal(labels > 0, mask)


def test_connected_roofs_with_distinct_centers_still_separate():
    mask = np.zeros((80, 100), bool)
    mask[20:50, 10:40] = True
    mask[20:50, 60:90] = True
    mask[33:37, 40:60] = True
    labels, _ = separate_instances(mask.astype(float), ndimage.distance_transform_edt(mask), np.ones_like(mask))
    assert labels.max() == 2


def test_tiles_preserve_interior_black_and_exclude_scene_connected_padding():
    data = np.full((3, 32, 40), 90, dtype="uint8")
    data[:, 10:20, 15:25] = 0  # interior dark object, intersects tile edge
    data[:, :5, :] = 0
    data[:, :15, 2:5] = 0  # padding connected beyond the tile boundary
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=40, height=32, count=3, dtype="uint8") as dst:
            dst.write(data)
        with mem.open() as src:
            _, valid, _ = rgb_unit_data(src)
            for window in (Window(15, 10, 10, 10), Window(0, 7, 10, 15)):
                _, direct, _ = rgb_unit_data(src, window=window)
                _, reused, _ = rgb_unit_data(src, window=window, scene_valid=valid)
                np.testing.assert_array_equal(direct, valid[window.toslices()])
                np.testing.assert_array_equal(reused, direct)
            assert valid[10:20, 15:25].all()
            assert not valid[:15, 2:5].any()


def test_satellite_model_is_selected_by_default(tmp_path):
    bundle = tmp_path / "satquery_buildings_bundle"
    bundle.mkdir()
    released = tmp_path / "buildings/model_cpu_fp32.onnx"
    released.parent.mkdir()
    released.touch()
    assert default_building_checkpoint(tmp_path) == tmp_path / "buildings/rf-detr-seg-satellite-buildings"


def test_missing_selected_bundle_does_not_select_other_weights(tmp_path):
    released = tmp_path / "buildings/model_cpu_fp32.onnx"
    released.parent.mkdir()
    released.touch()
    assert default_building_checkpoint(tmp_path) == tmp_path / "buildings/rf-detr-seg-satellite-buildings"


@pytest.mark.parametrize("resolution,crs", [(10, "EPSG:32632"), (1, None)])
def test_aerial_land_model_rejects_coarse_or_unknown_satellite_scale(tmp_path, resolution, crs):
    path = tmp_path / "satellite.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=3, dtype="uint8",
                       crs=crs, transform=rasterio.transform.from_origin(500000, 4000000, resolution, resolution)) as dst:
        dst.write(np.full((3, 20, 20), 80, dtype="uint8"))
    compatible, _ = compatibility(path)
    assert not compatible
