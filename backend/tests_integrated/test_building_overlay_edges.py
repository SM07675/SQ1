import numpy as np

from satquery_engine.services.spatial_outputs import instance_edges


def test_building_outline_marks_all_sides_without_filling_center():
    labels = np.zeros((5, 5), dtype="int32")
    labels[1:4, 1:4] = 7

    edges = instance_edges(labels)

    assert edges.sum() == 8
    assert edges[1, 2] and edges[3, 2]
    assert edges[2, 1] and edges[2, 3]
    assert not edges[2, 2]


def test_touching_instances_both_receive_a_boundary():
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype="int32")
    edges = instance_edges(labels)
    assert edges[:, 1].all()
    assert edges[:, 2].all()
