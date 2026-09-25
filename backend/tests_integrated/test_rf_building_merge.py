"""Instance counting invariants for overlapping satellite inference tiles."""
import numpy as np

from satquery_engine.services.buildings_rf import merge_instance_tiles


def test_duplicate_tiles_count_each_roof_once_and_keep_distinct_roofs():
    first = np.ones((12, 12), dtype=bool)
    shifted = np.ones((12, 12), dtype=bool)
    third = np.ones((10, 10), dtype=bool)
    valid = np.ones((40, 50), dtype=bool)
    labels, confidence, duplicates = merge_instance_tiles([
        (8, 10, first, .9),
        (8, 10, shifted, .8),
        (8, 31, third, .7),
    ], valid.shape, valid)
    assert labels.max() == 2
    assert duplicates == 1
    assert np.count_nonzero(labels) == first.sum() + third.sum()
    assert confidence[10, 12] == np.float32(.9)
    assert confidence[10, 33] == np.float32(.7)


def test_invalid_pixels_are_never_counted_as_buildings():
    valid = np.ones((30, 30), dtype=bool)
    valid[8:16, 8:16] = False
    labels, confidence, _ = merge_instance_tiles([(5, 5, np.ones((15, 15), bool), .9)], valid.shape, valid)
    assert not labels[~valid].any()
    assert not confidence[~valid].any()


def test_joint_overlap_does_not_count_leftover_strip_as_another_roof():
    valid = np.ones((40, 60), bool)
    labels, _, rejected = merge_instance_tiles([
        (10, 10, np.ones((10, 18), bool), .95),
        (10, 30, np.ones((10, 18), bool), .9),
        (10, 10, np.ones((12, 38), bool), .6),
    ], valid.shape, valid)
    assert labels.max() == 2
    assert rejected == 1


def test_detached_speckles_are_removed_but_separate_roofs_remain():
    valid = np.ones((60, 60), bool)
    mask = np.zeros((30, 30), bool)
    mask[:20, :20] = True
    mask[28:, 28:] = True
    labels, _, _ = merge_instance_tiles([
        (0, 0, mask, .9), (35, 35, np.ones((10, 10), bool), .8),
    ], valid.shape, valid)
    assert labels.max() == 2
    assert not labels[28:30, 28:30].any()


def test_complete_roof_wins_over_higher_scoring_tile_edge_fragment():
    valid = np.ones((40, 40), bool)
    labels, _, rejected = merge_instance_tiles([
        (10, 10, np.ones((20, 10), bool), .95, True),
        (10, 10, np.ones((20, 20), bool), .85, False),
    ], valid.shape, valid)
    assert labels.max() == 1 and rejected == 1
    assert np.count_nonzero(labels) == 400
