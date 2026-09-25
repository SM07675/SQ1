from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionResult:
    classes: np.ndarray
    confidence: np.ndarray
    ambiguous: np.ndarray
    water_override: np.ndarray
    building_override: np.ndarray


def aggregate_landcover(fine_probability: np.ndarray, fine_to_canonical: list[int], class_count: int = 7) -> np.ndarray:
    if fine_probability.shape[0] != len(fine_to_canonical):
        raise ValueError("Fine-class probability count does not match registry mapping.")
    result = np.zeros((class_count, *fine_probability.shape[1:]), dtype="float32")
    for fine_id, coarse_id in enumerate(fine_to_canonical):
        result[coarse_id] += fine_probability[fine_id]
    total = result.sum(axis=0, keepdims=True)
    return np.divide(result, total, out=np.zeros_like(result), where=total > 0)


def fuse(
    landcover_probability: np.ndarray,
    valid: np.ndarray,
    minimum_confidence: float,
    ambiguity_margin: float,
    water_probability: np.ndarray | None = None,
    water_support: np.ndarray | None = None,
    water_threshold: float = 0.60,
    building_probability: np.ndarray | None = None,
    building_threshold: float = 0.60,
    water_class_id: int = 5,
    built_class_id: int = 1,
) -> FusionResult:
    order = np.sort(landcover_probability, axis=0)
    best = np.argmax(landcover_probability, axis=0).astype("uint8")
    confidence = order[-1].astype("float32")
    margin = order[-1] - order[-2] if landcover_probability.shape[0] > 1 else order[-1]
    ambiguous = (confidence < minimum_confidence) | (margin < ambiguity_margin)
    result = best.copy()
    result[ambiguous | ~valid] = 0
    water_override = np.zeros_like(valid)
    if water_probability is not None:
        if water_support is None:
            raise ValueError("Specialist water output cannot be fused without spectral support.")
        water_override = (water_probability >= water_threshold) & water_support & valid
    building_override = np.zeros_like(valid)
    if building_probability is not None:
        building_override = (building_probability >= building_threshold) & valid
    water_override &= ~building_override
    result[water_override] = water_class_id
    if water_probability is not None:
        confidence[water_override] = water_probability[water_override]
    result[building_override] = built_class_id
    if building_probability is not None:
        confidence[building_override] = building_probability[building_override]
    result[~valid] = 0
    confidence[~valid] = 0
    ambiguous &= ~(water_override | building_override)
    return FusionResult(result, confidence, ambiguous, water_override, building_override)
