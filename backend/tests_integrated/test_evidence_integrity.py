"""Counterexamples for spatial consistency and malformed specialist outputs."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import translate
from shapely.geometry import mapping, shape

from satquery_engine.services.intents import plan_query
from satquery_engine.services.land_cover import classify_land_cover_composite
from satquery_engine.services.quality_gate import validate_result_evidence
from satquery_engine.services.spatial_outputs import export_labels


def write(path, data):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=len(data), dtype=data.dtype, crs="EPSG:32632",
                       transform=from_origin(500000, 4000000, .5, .5)) as dst:
        dst.write(data)
        if len(data) == 3:
            dst.descriptions = ("red", "green", "blue")
    return path


@pytest.mark.parametrize("query", ["Mark the oceans", "Find seas", "Map coastal waters", "Detect floods"])
def test_water_synonyms(query):
    assert "WATER_ANALYSIS" in plan_query(query).intents


def test_gate_rejects_same_area_shifted_footprint(tmp_path):
    source = write(tmp_path / "source.tif", np.full((3, 20, 20), 120, dtype="uint8"))
    labels = np.zeros((20, 20), dtype="int32")
    labels[2:8, 2:8] = 1
    out = tmp_path / "out"
    result = export_labels(labels, source, out, "buildings")
    validate_result_evidence(result, out)
    feature = result["features"][0]
    feature["properties"]["pixel_geometry"] = mapping(translate(shape(feature["properties"]["pixel_geometry"]), xoff=3))
    geo = out / "buildings.geojson"
    content = json.loads(geo.read_text())
    content["features"] = result["features"]
    geo.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="pixel alignment"):
        validate_result_evidence(result, out)


def compose(tmp_path, monkeypatch, *, invalid=False):
    import satquery_engine.services.landcover_specialist as specialist
    dims = (20, 20)
    source = write(tmp_path / "source.tif", np.full((3, *dims), 120, dtype="uint8"))
    water = write(tmp_path / "water_mask.tif", np.zeros((1, *dims), dtype="uint8"))
    vegetation = write(tmp_path / "vegetation_labels.tif", np.zeros((1, *dims), dtype="uint8"))
    probability = np.zeros((5, *dims), dtype="float32")
    probability[3] = 1
    if invalid:
        probability[3, 0, 0] = np.nan
    monkeypatch.setattr(specialist, "predict_landcover", lambda *a: {
        "probability": probability, "valid": np.ones(dims, bool), "model_id": "test",
        "classes": specialist.CLASSES, "checkpoint_sha256": "test", "preprocessing": {}, "domain_note": "Test output",
    })
    return classify_land_cover_composite(source, tmp_path / "out",
        water_result={"paths": [water]}, vegetation_result={"paths": [vegetation]})


def test_landcover_preserves_canonical_water(tmp_path, monkeypatch):
    result = compose(tmp_path, monkeypatch)
    assert result["water_percent"] == 0
    assert result["breakdown"]["unknown"]["percent"] == 100
    assert any("WATER_MODEL_DISAGREEMENT" in item for item in result["limitations"])
    validate_result_evidence(result, tmp_path / "out")


def test_gate_rejects_class_totals_even_if_sum_is_100(tmp_path, monkeypatch):
    result = compose(tmp_path, monkeypatch)
    result["breakdown"]["unknown"].update(percent=75, pixels=300)
    result["breakdown"]["water"].update(percent=25, pixels=100)
    with pytest.raises(ValueError, match="class measurements"):
        validate_result_evidence(result, tmp_path / "out")


def test_invalid_landcover_probability_abstains(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="probabilities"):
        compose(tmp_path, monkeypatch, invalid=True)


@pytest.mark.parametrize("dims", [(8, 8), (8, 90), (90, 8)])
def test_bundle_tiles_small_and_narrow_images(tmp_path, monkeypatch, dims):
    torch = pytest.importorskip("torch")
    import satquery_engine.services.buildings as buildings
    class Model:
        def __call__(self, tensor):
            assert tuple(tensor.shape) == (1, 3, 32, 32)
            output = torch.full((1, 2, 32, 32), -10.)
            return output
    monkeypatch.setattr(buildings, "_resolve_checkpoint", lambda: tmp_path)
    monkeypatch.setattr(buildings, "_sha256", lambda *a: "test")
    monkeypatch.setattr(buildings, "load_session", lambda *a: (Model(), {
        "input": {"tile_size": 32}, "tile_inference": {"overlap": 8}}, {"test": {"instance_f1@0.5": 0.0}}))
    path = write(tmp_path / "source.tif", np.full((3, *dims), 120, dtype="uint8"))
    result = buildings.detect_buildings(path, tmp_path / "out")
    assert result["complete_coverage"] and result["count"] == 0
    assert result["benchmark_f1"] == 0.0
    assert result["count_reliability"] == "COUNT_UNRELIABLE"
    assert result["tile_count"] > 1 if max(dims) > 32 else result["tile_count"] == 1
    validate_result_evidence(result, tmp_path / "out")


@pytest.mark.asyncio
async def test_unreliable_zero_is_not_reported_as_absence(tmp_path, monkeypatch):
    import satquery_engine.services.execution as execution
    from satquery_engine.services.execution import analyze
    source = write(tmp_path / "source.tif", np.full((3, 40, 40), 120, dtype="uint8"))
    def detect(path, output, *args, **kwargs):
        result = export_labels(np.zeros((40, 40), dtype="int32"), path, output, "buildings")
        return {**result, "count": 0, "model_id": "test", "count_reliability": "COUNT_UNRELIABLE"}
    monkeypatch.setattr(execution, "detect_buildings", detect)
    result = await analyze(result_id="zero", query="Count buildings", pair_type="auto",
                           image_paths=[source], output_dir=tmp_path / "out")
    assert result.verdict.status.value == "low_confidence"
    assert "does not establish that no buildings are present" in result.verdict.answer
