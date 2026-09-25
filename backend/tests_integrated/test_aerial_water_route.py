"""Model routing, fallback, and thin-water invariants without pretrained weights."""
import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from satquery_engine.services.water_engine import execute_water_pipeline, confirm_aerial_water
from satquery_engine.services.quality_gate import validate_result_evidence
from satquery_engine.services.land_cover import classify_land_cover_composite
from satquery_engine.services.landcover_specialist import CLASSES


def source(tmp_path, resolution=.5, georeferenced=True):
    path = tmp_path / "source.tif"
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=3, dtype="uint8",
                       crs="EPSG:32632" if georeferenced else None,
                       transform=from_origin(500000, 4000000, resolution, resolution)) as dst:
        dst.write(np.full((3, 40, 40), 100, dtype="uint8"))
        dst.descriptions = ("red", "green", "blue")
    return path


def model(monkeypatch, invalid=False):
    import satquery_engine.services.landcover_specialist as specialist
    scores = np.zeros((5, 40, 40), dtype="float32")
    scores[0] = 1
    scores[0, :, 15] = .1
    scores[3, :, 15] = .9
    if invalid:
        scores[3, 0, 0] = np.nan
    monkeypatch.setattr(specialist, "predict_landcover", lambda *a: {
        "probability": scores, "valid": np.ones((40, 40), bool), "classes": CLASSES,
        "model_id": "test-aerial", "checkpoint_sha256": "test", "preprocessing": {},
        "tiles": [{}], "domain_note": "Test model only"})


def test_aerial_water_requires_object_level_model_agreement():
    primary = np.zeros((50, 100), dtype="float32")
    second = np.zeros_like(primary)
    primary[5:35, 5:35] = .9
    primary[5:35, 60:90] = .9
    second[5:35, 5:24] = .8  # 63% support: retain the first coherent object.
    second[5:15, 60:90] = .8  # 33% support: reject the second object.
    probability, audit = confirm_aerial_water(primary, second, np.ones_like(primary, bool))
    assert (probability[5:35, 5:35] >= .5).all()
    assert (probability[5:35, 60:90] < .5).all()
    assert audit == {"candidate_components": 2, "confirmed_components": 1,
                     "candidate_pixels": 1800, "withheld_candidate_pixels": 900, "withheld_fraction": .5}


def test_land_compositor_colors_confirmed_soil_and_visible_unknown(tmp_path, monkeypatch):
    import satquery_engine.services.landcover_specialist as deepness
    import satquery_engine.services.flair_hub as flair

    path = source(tmp_path)
    empty = tmp_path / "water_mask.tif"
    with rasterio.open(path) as src, rasterio.open(empty, "w", **{**src.profile, "count": 1}) as dst:
        dst.write(np.zeros((40, 40), dtype="uint8"), 1)
    deep_scores = np.zeros((5, 40, 40), dtype="float32")
    deep_scores[0] = 1
    flair_scores = np.zeros((19, 40, 40), dtype="float32")
    flair_scores[0] = 1
    flair_scores[0, 10:30, 10:30] = .02
    flair_scores[5, 10:30, 10:30] = .98
    monkeypatch.setattr(deepness, "predict_landcover", lambda *_: {
        "probability": deep_scores, "valid": np.ones((40, 40), bool), "classes": CLASSES,
        "model_id": deepness.MODEL_ID, "checkpoint_sha256": "test", "preprocessing": {},
        "tiles": [{}], "domain_note": "Test model only"})
    monkeypatch.setattr(flair, "predict_flair_hub", lambda *_: {
        "probability": flair_scores, "valid": np.ones((40, 40), bool), "classes": tuple(str(i) for i in range(19)),
        "model_id": flair.MODEL_ID, "checkpoint_sha256": "test", "preprocessing": {},
        "tiles": [{}], "domain_note": "Test model only"})
    result = classify_land_cover_composite(path, tmp_path / "land", water_result={"paths": [empty]})
    assert result["breakdown"]["bare_pervious"]["pixels"] == 400
    assert result["breakdown"]["unknown"]["pixels"] == 1200
    rgba = np.asarray(Image.open(tmp_path / "land/land_cover_mask.png").convert("RGBA"))
    np.testing.assert_array_equal(rgba[20, 20], [166, 130, 80, 140])
    np.testing.assert_array_equal(rgba[0, 0], [194, 155, 56, 100])


def test_aerial_route_preserves_one_pixel_waterway_and_land_consistency(tmp_path, monkeypatch):
    model(monkeypatch)
    path = source(tmp_path)
    result = execute_water_pipeline(path, tmp_path / "water")
    assert result["route"] == "RGB_AERIAL_WATER"
    assert result["models_used"] == ["test-aerial"]
    assert result["selected_pixels"] == 40
    assert result["checkpoint_execution"] == "aerial_landcover_onnx"
    assert result["confidence_kind"] == "uncalibrated_evidence_strength"
    validate_result_evidence(result, tmp_path / "water")
    land = classify_land_cover_composite(path, tmp_path / "land", water_result=result)
    assert land["water_percent"] == pytest.approx(result["coverage_percent"])
    validate_result_evidence(land, tmp_path / "land")


@pytest.mark.parametrize("resolution,georeferenced,use_model", [(10, True, True), (.5, False, True), (.5, True, False)])
def test_unverified_or_disabled_aerial_does_not_run(tmp_path, monkeypatch, resolution, georeferenced, use_model):
    import satquery_engine.services.landcover_specialist as specialist
    def forbidden(*args):
        raise AssertionError("Incompatible aerial model must not run")
    monkeypatch.setattr(specialist, "predict_landcover", forbidden)
    result = execute_water_pipeline(source(tmp_path, resolution, georeferenced), tmp_path / "out", use_model=use_model)
    assert result["route"] == "RGB_WATER_PROXY"
    assert not result["models_used"]


def test_invalid_aerial_output_withholds_unverified_water(tmp_path, monkeypatch):
    model(monkeypatch, invalid=True)
    result = execute_water_pipeline(source(tmp_path), tmp_path / "out")
    assert result["route"] == "RGB_WATER_PROXY"
    assert any("Invalid aerial water probabilities" in event for event in result["fallback_events"])
    assert not result["models_used"]
    assert result["selected_pixels"] == 0
    assert result["evidence_state"] == "INSUFFICIENT_EVIDENCE"
    validate_result_evidence(result, tmp_path / "out")
