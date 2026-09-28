import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery_engine.services.surface_context import SurfaceContext, exclude_water_instances


def scene(path, values, transform=None):
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype=values.dtype, crs="EPSG:32643",
                       transform=transform or from_origin(500000, 2200000, .5, .5)) as dst:
        dst.write(values, 1)
    return path


def test_water_exclusion_rejects_false_building_and_preserves_shoreline(tmp_path):
    labels = np.zeros((20, 30), dtype="int32")
    labels[2:8, 2:8] = 1
    labels[2:8, 12:18] = 2
    labels[12:18, 22:28] = 3
    water = np.zeros_like(labels, dtype="uint8")
    water[2:8, 2:8] = 1
    water[2:8, 12:13] = 1
    source = scene(tmp_path / "source.tif", np.ones_like(water))
    mask = scene(tmp_path / "water_mask.tif", water)
    result, audit = exclude_water_instances(labels, np.ones_like(water, bool), source, {"paths": [mask]}, 16)
    assert set(np.unique(result)) == {0, 1, 2}
    assert not np.any(result[water > 0])
    assert np.count_nonzero(result == 1) == 30
    assert np.count_nonzero(result == 2) == 36
    assert audit["rejected_instances"] == 1
    assert audit["water_overlap_pixels"] == 42


def test_shifted_water_mask_fails_closed(tmp_path):
    values = np.ones((10, 10), dtype="uint8")
    source = scene(tmp_path / "source.tif", values)
    mask = scene(tmp_path / "water_mask.tif", values, from_origin(500001, 2200000, .5, .5))
    with pytest.raises(ValueError, match="same source grid"):
        exclude_water_instances(values, values > 0, source, {"paths": [mask]}, 16)


def test_water_cache_reuses_only_identical_source_and_options(tmp_path, monkeypatch):
    calls = []
    def measure(source, output, target, **options):
        calls.append((source, options))
        return {"paths": [], "evidence_strength": .5}
    monkeypatch.setattr("satquery_engine.services.measurements.measure_cover", measure)
    cache = SurfaceContext(tmp_path)
    first = cache.water(tmp_path / "a.tif")
    assert cache.water(tmp_path / "a.tif") is first
    cache.water(tmp_path / "b.tif")
    cache.water(tmp_path / "a.tif", largest=True)
    cache.water(tmp_path / "a.tif", strict=True)
    assert len(calls) == 4


def test_filtered_count_matches_all_exported_evidence(tmp_path):
    from .test_building_pipeline_invariants import _write_rgb, _FakeSession, _patch_checkpoint, _patch_session
    from satquery_engine.services.buildings import detect_buildings
    from satquery_engine.services.quality_gate import validate_result_evidence
    source = _write_rgb(tmp_path / "source.tif", 256, 256)
    foreground = np.zeros((256, 256), bool)
    foreground[20:40, 20:40] = True
    foreground[80:100, 80:100] = True
    water = np.zeros((256, 256), dtype="uint8")
    water[10:50, 10:50] = 1
    mask = scene(tmp_path / "water_mask.tif", water)
    with _patch_checkpoint(), _patch_session(_FakeSession(lambda _: foreground)):
        result = detect_buildings(source, tmp_path / "out", water_result={"paths": [mask]})
    validate_result_evidence(result, tmp_path / "out")
    assert result["count"] == result["building_count"] == len(result["features"]) == 1
    with rasterio.open(tmp_path / "out/buildings_labels.tif") as src:
        assert not np.any(src.read(1)[water > 0])


def test_land_only_mask_excludes_water_and_unknown(tmp_path, monkeypatch):
    from .test_building_pipeline_invariants import _write_rgb
    from satquery_engine.services.land_cover import classify_land_cover_composite
    source = _write_rgb(tmp_path / "source.tif", 30, 20)
    water = np.zeros((20, 30), dtype="uint8")
    water[:, :10] = 1
    vegetation = np.zeros_like(water)
    vegetation[:, 10:20] = 1
    water_path = scene(tmp_path / "water_mask.tif", water)
    vegetation_path = scene(tmp_path / "vegetation_labels.tif", vegetation)
    monkeypatch.setattr("satquery_engine.services.landcover_specialist.compatibility", lambda _: (False, "Test: no learned evidence"))
    result = classify_land_cover_composite(source, tmp_path / "land",
        water_result={"paths": [water_path]}, vegetation_result={"paths": [vegetation_path]})
    assert result["classified_land_percent"] == pytest.approx(100 / 3)
    with rasterio.open(tmp_path / "land/land_only_mask.tif") as src:
        np.testing.assert_array_equal(src.read(1) > 0, vegetation > 0)


def test_explicit_land_request_does_not_relabel_water_as_land(tmp_path, monkeypatch):
    from .test_building_pipeline_invariants import _write_rgb
    from satquery_engine.services.land_cover import classify_land_cover_composite
    source = _write_rgb(tmp_path / "source.tif", 30, 20)
    water = np.zeros((20, 30), dtype="uint8")
    water[:, :10] = 1
    water_path = scene(tmp_path / "water_mask.tif", water)
    vegetation_path = scene(tmp_path / "vegetation_labels.tif", np.zeros_like(water))
    monkeypatch.setattr("satquery_engine.services.landcover_specialist.compatibility", lambda _: (False, "Test: no learned evidence"))
    result = classify_land_cover_composite(source, tmp_path / "land", query="Find land",
        water_result={"paths": [water_path]}, vegetation_result={"paths": [vegetation_path]})
    assert result["land_pixels"] == 400
    assert result["land_percent"] == pytest.approx(100 * 400 / 600)
    assert result["breakdown"]["water"]["pixels"] == 200
    assert result["breakdown"]["unknown"]["pixels"] == 400
    with rasterio.open(tmp_path / "land/land_only_mask.tif") as src:
        np.testing.assert_array_equal(src.read(1) > 0, water == 0)


@pytest.mark.asyncio
async def test_combined_analysis_shares_water_and_publishes_filtered_counts(tmp_path, monkeypatch):
    from .test_building_pipeline_invariants import _write_rgb, _FakeSession, _patch_checkpoint, _patch_session
    from satquery_engine.services.execution import analyze
    from satquery_engine.services.water_engine import execute_water_pipeline
    source = _write_rgb(tmp_path / "source.tif", 256, 256)
    rgb = np.full((3, 256, 256), 170, dtype="uint8")
    rgb[:, :, :120] = np.array([20, 65, 90], dtype="uint8")[:, None, None]
    with rasterio.open(source, "r+") as dst:
        dst.write(rgb)
    calls = []
    def measure(path, output, target, **options):
        calls.append(target)
        return execute_water_pipeline(path, output, use_model=False, **options)
    monkeypatch.setattr("satquery_engine.services.measurements.measure_cover", measure)
    foreground = np.zeros((256, 256), bool)
    foreground[30:60, 30:60] = True
    foreground[150:180, 170:200] = True
    with _patch_checkpoint(), _patch_session(_FakeSession(lambda _: foreground)):
        result = await analyze(result_id="shared-water", query="Count buildings and measure water area",
            pair_type="auto", image_paths=[source], output_dir=tmp_path / "analysis")
    assert calls == ["water"]
    assert result.statistics["buildings_a"]["count"] == 1
    assert result.statistics["water_measure"]["selected_pixels"] > 0
    assert all(e.artifact_url for e in result.evidence)
    assert not any(step.status == "unavailable" for step in result.trace)


@pytest.mark.parametrize("shape,ready", [([1, 3, 256, 256], True), ([1, 1, 256, 256], False)])
def test_health_checks_fallback_schema_without_running_unused_model(tmp_path, monkeypatch, shape, ready):
    from dataclasses import replace
    from types import SimpleNamespace
    from .test_model_compatibility import _manifest
    from satquery_engine.models.registry import ModelHealthCheck
    import onnxruntime
    checkpoint = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"test graph placeholder")
    class Session:
        def get_inputs(self): return [SimpleNamespace(shape=[1, 3, 256, 256])]
        def get_outputs(self): return [SimpleNamespace(shape=shape)]
        def run(self, *args): raise AssertionError("Health discovery must not run inference")
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: Session())
    manifest = replace(_manifest(), local_path=tmp_path, required_files=("model.onnx",),
        required_packages=(), metadata={"registry_key": "building_fallback"})
    assert ModelHealthCheck().check(manifest).availability is ready
