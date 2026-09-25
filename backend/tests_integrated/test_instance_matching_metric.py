"""The evaluator must maximize qualifying matches, not subthreshold IoU."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("building_evaluator", Path(__file__).resolve().parents[2] / "scripts/benchmark_buildings.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_threshold_matching_maximizes_true_positives():
    # Raw IoU assignment favors .99+.49, then throws out .49. Both off-diagonal
    # entries qualify, so the correct instance metric has two true positives.
    matrix = np.array([[.99, .51], [.51, .49]])
    assert set(evaluator.match_instances(matrix)) == {(0, 1), (1, 0)}


def test_matching_is_one_to_one_and_handles_empty_sets():
    pairs = evaluator.match_instances(np.full((5, 3), .8))
    assert len(pairs) == 3
    assert len({a for a,b in pairs}) == len({b for a,b in pairs}) == 3
    assert evaluator.match_instances(np.zeros((0, 4))) == []
    assert evaluator.match_instances(np.zeros((4, 0))) == []
