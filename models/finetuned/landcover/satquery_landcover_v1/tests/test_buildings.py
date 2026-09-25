import numpy as np

from satquery.postprocess import separate_buildings


def test_touching_buildings_are_separated_after_global_blend():
    yy, xx = np.mgrid[:80, :80]
    footprint = (((yy - 40) ** 2 + (xx - 31) ** 2 < 15 ** 2) | ((yy - 40) ** 2 + (xx - 49) ** 2 < 15 ** 2)).astype("float32")
    boundary = np.zeros_like(footprint); boundary[:, 40] = 1
    labels, stats = separate_buildings(footprint, boundary, np.ones_like(footprint, bool), .5, .5, 20, 1, 8)
    assert labels.max() == 2
    assert stats["instances"] == 2


def test_overlap_dedup_is_implicit_in_single_global_watershed():
    footprint = np.zeros((64, 96), "float32"); footprint[20:45, 35:60] = .9
    labels, _ = separate_buildings(footprint, np.zeros_like(footprint), np.ones_like(footprint, bool), .6, .6, 20, 2, 6)
    assert labels.max() == 1


def test_nodata_never_becomes_building():
    footprint = np.ones((20, 20), "float32")
    valid = np.zeros((20, 20), bool)
    labels, stats = separate_buildings(footprint, footprint, valid)
    assert labels.max() == 0 and stats["instances"] == 0

