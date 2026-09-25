from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def filter_binary(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    labels, _ = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype="uint8"))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, int(minimum_area))
    keep[0] = False
    return keep[labels]


def separate_buildings(
    footprint_probability: np.ndarray,
    boundary_probability: np.ndarray,
    valid: np.ndarray,
    footprint_threshold: float = 0.60,
    boundary_threshold: float = 0.60,
    minimum_area: int = 48,
    h_prominence: float = 6.0,
    minimum_distance: int = 8,
) -> tuple[np.ndarray, dict[str, int]]:
    """One global boundary-aware watershed pass over blended mosaic probabilities."""
    foreground = (footprint_probability >= footprint_threshold) & valid
    foreground = filter_binary(foreground, minimum_area)
    if not foreground.any():
        return np.zeros(foreground.shape, dtype="int32"), {"instances": 0, "rejected_small": 0}
    distance = ndimage.distance_transform_edt(foreground)
    boundary = np.clip((boundary_probability - boundary_threshold) / max(1e-6, 1 - boundary_threshold), 0, 1)
    relief = distance * (1.0 - 0.65 * boundary)
    prominent = h_maxima(relief, h=h_prominence)
    candidates = peak_local_max(relief, min_distance=minimum_distance, labels=foreground.astype("uint8"), exclude_border=False)
    seeds = np.zeros_like(foreground)
    for y, x in candidates:
        if prominent[y, x]:
            seeds[y, x] = True
    components, component_count = ndimage.label(foreground)
    for component_id, area_slice in enumerate(ndimage.find_objects(components), 1):
        if area_slice is None:
            continue
        region = components[area_slice] == component_id
        if not np.any(seeds[area_slice] & region):
            local = np.where(region, relief[area_slice], -np.inf)
            y, x = np.unravel_index(np.argmax(local), local.shape)
            seeds[area_slice[0].start + y, area_slice[1].start + x] = True
    markers, _ = ndimage.label(seeds)
    labels = watershed(-relief, markers, mask=foreground).astype("int32")
    sizes = np.bincount(labels.ravel())
    rejected = int(np.count_nonzero((sizes[1:] > 0) & (sizes[1:] < minimum_area)))
    keep = sizes >= minimum_area
    keep[0] = False
    labels[~keep[labels]] = 0
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lookup = np.zeros(int(labels.max()) + 1, dtype="int32")
    lookup[ids] = np.arange(1, len(ids) + 1)
    compact = lookup[labels]
    return compact, {"instances": int(len(ids)), "rejected_small": rejected, "foreground_components": int(component_count)}


def instance_scores(labels: np.ndarray, probability: np.ndarray) -> dict[int, float]:
    count = int(labels.max())
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sums = np.bincount(labels.ravel(), weights=probability.ravel(), minlength=count + 1)
    return {index: float(sums[index] / sizes[index]) for index in range(1, count + 1) if sizes[index]}
