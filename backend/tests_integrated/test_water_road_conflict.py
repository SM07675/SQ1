import numpy as np
from pathlib import Path

from satquery_engine.services.water_engine import suppress_road_water_conflicts


def test_aerial_road_conflict_withholds_water_but_keeps_supported_water():
    model_water = np.array([[0.90, 0.91, 0.85]], dtype="float32")
    built = np.array([[0.82, 0.10, 0.50]], dtype="float32")
    rgb_water = np.array([[0.08, 0.80, 0.65]], dtype="float32")

    filtered, conflict = suppress_road_water_conflicts(model_water, built, rgb_water)

    assert conflict.tolist() == [[True, False, False]]
    np.testing.assert_allclose(filtered, [[0.10, 0.91, 0.85]], atol=1e-6)


def test_rgb_road_scene_does_not_become_a_water_map(tmp_path):
    from satquery_engine.services.water_engine import execute_water_pipeline

    sample = Path(__file__).resolve().parents[2] / "frontend" / "public" / "SN3_roads_train_AOI_2_Vegas_PS-RGB_img8.tif"
    result = execute_water_pipeline(sample, tmp_path, use_model=False)

    assert result["route"] == "RGB_WATER_PROXY"
    assert result["coverage_percent"] < 0.7
