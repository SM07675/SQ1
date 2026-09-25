import numpy as np

from satquery.tiling import blended_predict, starts, windows


def test_edges_are_covered_once_or_more_and_padded():
    win = list(windows(37, 43, 16, 5))
    coverage = np.zeros((37, 43), "int32")
    for y, x, h, w in win:
        coverage[y:y+h, x:x+w] += 1
    assert coverage.min() >= 1
    assert starts(5, 16, 5) == [0]


def test_hann_blending_has_no_seams_for_constant_predictor():
    image = np.random.default_rng(4).random((3, 45, 59), dtype=np.float32)
    result = blended_predict(image, lambda tile: np.stack([np.full(tile.shape[1:], 0.25), np.full(tile.shape[1:], 0.75)]), 2, 16, 8)
    np.testing.assert_allclose(result[0], 0.25, atol=1e-6)
    np.testing.assert_allclose(result[1], 0.75, atol=1e-6)


def test_blended_inference_is_deterministic():
    image = np.random.default_rng(9).random((3, 31, 29), dtype=np.float32)
    predictor = lambda tile: np.stack([tile[0], 1 - tile[0]])
    first = blended_predict(image, predictor, 2, 16, 4)
    second = blended_predict(image, predictor, 2, 16, 4)
    np.testing.assert_array_equal(first, second)

