import numpy as np

from satquery.evaluation import building_metrics, confusion_matrix, metrics_from_confusion, water_metrics


def test_perfect_metrics():
    target = np.array([[0, 1], [1, 0]], dtype="uint8")
    metrics = metrics_from_confusion(confusion_matrix(target, target, 2))
    assert metrics["macro_iou"] == 1
    assert water_metrics(target, target)["f1"] == 1


def test_instance_and_count_metrics():
    target = np.zeros((20, 20), "int32"); target[2:7, 2:7] = 1; target[10:16, 10:16] = 2
    result = building_metrics(target, target)
    assert result["instance_f1@0.5"] == 1
    assert result["count_mae"] == 0

