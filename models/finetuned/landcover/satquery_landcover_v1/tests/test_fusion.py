import numpy as np

from satquery.fusion import fuse
from satquery.preprocess import spectral_features, water_spectral_support


def probabilities(shape=(8, 8)):
    result = np.zeros((7, *shape), "float32")
    result[2] = 0.8
    result[0] = 0.2
    return result


def test_fusion_priority_invalid_then_building_then_water():
    valid = np.ones((8, 8), bool); valid[0, 0] = False
    water = np.ones((8, 8), "float32")
    support = np.ones((8, 8), bool)
    building = np.zeros((8, 8), "float32"); building[2:4, 2:4] = 0.9
    result = fuse(probabilities(), valid, 0.4, 0.05, water, support, 0.6, building, 0.6)
    assert result.classes[0, 0] == 0
    assert np.all(result.classes[2:4, 2:4] == 1)
    assert result.classes[6, 6] == 5
    assert not result.water_override[2:4, 2:4].any()


def test_dark_shadow_fails_water_spectral_gate():
    image = np.full((13, 4, 4), 300.0, "float32")
    _, _, evidence = spectral_features(image)
    thresholds = {"ndwi_min": .05, "mndwi_min": .02, "ndvi_max": .35, "awei_shadow_min": -.05, "brightness_min": .02, "cirrus_max": .03}
    assert not water_spectral_support(evidence, thresholds).any()


def test_ambiguous_and_all_background_are_unknown():
    probability = np.full((7, 5, 5), 1 / 7, "float32")
    result = fuse(probability, np.ones((5, 5), bool), 0.45, 0.1)
    assert np.all(result.classes == 0)
    assert result.ambiguous.all()

