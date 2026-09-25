from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > overlap >= 0.")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    if values[-1] != length - tile_size:
        values.append(length - tile_size)
    return values


def windows(height: int, width: int, tile_size: int, overlap: int) -> Iterator[tuple[int, int, int, int]]:
    for y in starts(height, tile_size, overlap):
        for x in starts(width, tile_size, overlap):
            yield y, x, min(tile_size, height - y), min(tile_size, width - x)


def hann2d(size: int, floor: float = 1e-3) -> np.ndarray:
    if size == 1:
        return np.ones((1, 1), dtype="float32")
    axis = np.hanning(size).astype("float32")
    return np.maximum(np.outer(axis, axis), floor)


def reflect_pad(tile: np.ndarray, size: int) -> np.ndarray:
    if tile.ndim != 3 or tile.shape[1] > size or tile.shape[2] > size:
        raise ValueError("Expected CxHxW tile no larger than requested size.")
    py, px = size - tile.shape[1], size - tile.shape[2]
    mode = "reflect" if tile.shape[1] > 1 and tile.shape[2] > 1 else "edge"
    return np.pad(tile, ((0, 0), (0, py), (0, px)), mode=mode)


def blended_predict(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    output_channels: int,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """Predict CxHxW patches and Hann-blend probabilities over the full mosaic."""
    if image.ndim != 3:
        raise ValueError("image must be CxHxW")
    height, width = image.shape[1:]
    accum = np.zeros((output_channels, height, width), dtype="float64")
    weights = np.zeros((height, width), dtype="float64")
    kernel = hann2d(tile_size)
    for y, x, h, w in windows(height, width, tile_size, overlap):
        patch = reflect_pad(image[:, y:y+h, x:x+w], tile_size)
        probability = np.asarray(predictor(patch), dtype="float32")
        if probability.shape != (output_channels, tile_size, tile_size):
            raise ValueError(f"Predictor returned {probability.shape}, expected {(output_channels, tile_size, tile_size)}.")
        if not np.isfinite(probability).all():
            raise ValueError("Predictor returned NaN or infinite values.")
        weight = kernel[:h, :w]
        accum[:, y:y+h, x:x+w] += probability[:, :h, :w] * weight
        weights[y:y+h, x:x+w] += weight
    if np.any(weights <= 0):
        raise RuntimeError("Tiling left uncovered pixels.")
    return (accum / weights[None]).astype("float32")

