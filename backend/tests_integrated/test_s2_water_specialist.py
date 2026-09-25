"""Real six-band water inference and fail-closed input checks."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin

from satquery_engine.services.bands import detect_band_map
from satquery_engine.services.quality_gate import validate_result_evidence
from satquery_engine.services.s2_water_specialist import MODEL_ID, band_indexes, predict
from satquery_engine.services.water_engine import execute_water_pipeline
from satquery_engine.services.water_engine import compute_multispectral_water


def _six_band_image(path, descriptions):
    with rasterio.open(path, "w", driver="GTiff", width=40, height=40, count=6,
                       dtype="uint16", crs="EPSG:32643", transform=from_origin(500000, 2800000, 10, 10)) as dst:
        dst.write(np.full((6, 40, 40), 1000, dtype="uint16"))
        dst.descriptions = descriptions
    return path


def test_six_band_model_uses_names_and_never_guesses_order(tmp_path):
    names = ("B11", "B04", "B12", "B02", "B08", "B03")
    with rasterio.open(_six_band_image(tmp_path / "named.tif", names)) as src:
        assert band_indexes(src) == (4, 6, 2, 5, 1, 3)
        bands = detect_band_map(src).indices
        assert bands["green"] == 6 and bands["nir"] == 5
    with rasterio.open(_six_band_image(tmp_path / "unnamed.tif", (None,) * 6)) as src:
        with pytest.raises(ValueError, match="band order is not guessed"):
            band_indexes(src)
    with rasterio.open(tmp_path / "unnamed.tif", "r+") as dst:
        dst.update_tags(band_order="B02,B03,B04,B08,B11,B12")
    with rasterio.open(tmp_path / "unnamed.tif") as src:
        assert band_indexes(src) == (1, 2, 3, 4, 5, 6)
        assert detect_band_map(src).indices["nir"] == 4


def test_public_checkpoint_runs_in_production_water_pipeline(tmp_path):
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "models/water/s2-water-unetplusplus-efficientnet-b4/model.pth"
    scene = root / "data/water_validation/s2_water_public/val_scene/S2A_L2A_20190318_N0211_R061_6Bands_S1.tif"
    truth = root / "data/water_validation/s2_water_public/val_truth/S2A_L2A_20190318_N0211_R061_S1_Truth.tif"
    if not all(path.is_file() for path in (checkpoint, scene, truth)):
        pytest.skip("Optional downloaded checkpoint and public validation scene are absent")
    named = tmp_path / "sentinel2_l2a.tif"
    raster_copy(scene, named, driver="GTiff")
    with rasterio.open(named, "r+") as dst:
        dst.descriptions = ("B02", "B03", "B04", "B08", "B11", "B12")
    output = tmp_path / "water"
    result = execute_water_pipeline(named, output)
    validate_result_evidence(result, output)
    assert result["model_id"] == MODEL_ID
    assert not any(event.startswith("FINE_TUNED_") for event in result["fallback_events"])
    assert result["quality_gate_passed"]
    with rasterio.open(output / "water_mask.tif") as src, rasterio.open(truth) as reference:
        prediction = src.read(1) > 0
        ground_truth = reference.read(1) == 1
        # Public truth metadata marks background 0 as NoData; score all labels.
        tp = int(np.count_nonzero(prediction & ground_truth))
        union = int(np.count_nonzero(prediction | ground_truth))
        assert tp / union > .70
        assert result["selected_pixels"] == int(prediction.sum())


def test_float_radiometry_is_decided_for_the_whole_scene(tmp_path, monkeypatch):
    import torch
    import satquery_engine.services.s2_water_specialist as specialist

    path = tmp_path / "mixed_brightness.tif"
    data = np.full((6, 40, 1300), .8, dtype="float32")
    data[:, :, 1100:] = 5000.0
    with rasterio.open(path, "w", driver="GTiff", width=1300, height=40,
                       count=6, dtype="float32", crs="EPSG:32643",
                       transform=from_origin(500000, 2800000, 10, 10)) as dst:
        dst.write(data)
        dst.descriptions = ("B02", "B03", "B04", "B08", "B11", "B12")

    observed = []

    class RecordingModel:
        def to(self, device):
            return self

        def __call__(self, tensor):
            observed.append(float(tensor[0, 0, 0, 0]))
            return torch.zeros((1, 2, 512, 512), device=tensor.device)

    monkeypatch.setattr(specialist, "load_model", lambda _: RecordingModel())
    monkeypatch.setattr("satquery_engine.models.device.torch_device", lambda: "cpu")
    with rasterio.open(path) as src:
        probability, valid, provenance = predict(src, tmp_path / "unused.pth")
    assert valid.all()
    assert np.allclose(probability, .5)
    assert observed and all(np.isclose(value, .8 / 255) for value in observed)
    assert provenance["preprocessing"]["normalization"] == "Sentinel-2 digital numbers / 255"


def test_nonfinite_model_output_is_rejected(tmp_path, monkeypatch):
    import torch
    import satquery_engine.services.s2_water_specialist as specialist

    path = _six_band_image(tmp_path / "scene.tif", ("B02", "B03", "B04", "B08", "B11", "B12"))

    class InvalidModel:
        def to(self, device):
            return self

        def __call__(self, tensor):
            return torch.full((1, 2, 512, 512), float("nan"), device=tensor.device)

    monkeypatch.setattr(specialist, "load_model", lambda _: InvalidModel())
    monkeypatch.setattr("satquery_engine.models.device.torch_device", lambda: "cpu")
    with rasterio.open(path) as src, pytest.raises(ValueError, match="invalid logits"):
        predict(src, tmp_path / "unused.pth")


def test_sentinel2_spectral_crosscheck_uses_reflectance_scale(tmp_path):
    names = ("B02", "B03", "B04", "B08", "B11", "B12")
    dn = np.array([700, 1700, 1100, 900, 400, 300], dtype="uint16")[:, None, None]
    dn = np.broadcast_to(dn, (6, 40, 40)).copy()
    results = []
    for dtype, data in (("uint16", dn), ("float32", dn.astype("float32") / 10000)):
        path = tmp_path / f"s2_{dtype}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=40, height=40,
                           count=6, dtype=dtype, crs="EPSG:32643",
                           transform=from_origin(500000, 2800000, 10, 10)) as dst:
            dst.write(data)
            dst.descriptions = names
        with rasterio.open(path) as src:
            results.append(compute_multispectral_water(src, detect_band_map(src).indices))
    assert np.allclose(results[0][0], results[1][0], atol=1e-5)
    assert np.allclose(results[0][3]["brightness"], results[1][3]["brightness"], atol=1e-5)
    assert results[0][3]["radiometry"] == "Sentinel-2 digital numbers / 10000"
