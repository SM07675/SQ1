import numpy as np

from satquery_engine.services.land_cover import reconcile_roofs_and_pavement


def test_roofs_and_paved_surfaces_need_cross_model_support():
    labels = np.array([[3, 4, 4, 1, 2]], dtype="int32")
    primary = np.zeros((5, 1, 5), dtype="float32")
    secondary = np.zeros((19, 1, 5), dtype="float32")
    primary[4, 0, [0, 2]] = 0.55
    primary[1, 0, 1] = 0.55
    secondary[3, 0, [0, 2]] = 0.80
    secondary[0, 0, 1] = 0.80

    refined, audit = reconcile_roofs_and_pavement(labels, primary, secondary, np.ones((1, 5), bool))

    assert refined.tolist() == [[6, 3, 6, 1, 2]]
    assert audit == {"false_roof_to_paved": 1, "additional_roof_pixels": 1, "additional_paved_pixels": 1}
