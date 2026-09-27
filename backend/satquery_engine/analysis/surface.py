from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SurfaceType(IntEnum):
    NODATA = 0
    WATER = 1
    VEGETATION = 2
    BUILDING = 3
    BUILT_UP = 4
    BARE_LAND = 5
    OTHER = 6
    UNKNOWN = 7


class IlluminationState(IntEnum):
    NODATA = 0
    NORMAL = 1
    SHADOWED = 2
    UNCERTAIN = 3


@dataclass(frozen=True)
class SurfaceIlluminationResult:
    surface: np.ndarray
    illumination: np.ndarray
    water_mask: np.ndarray
    shadow_mask: np.ndarray
    conflict: np.ndarray
    uncertainty: np.ndarray


def adjudicate_surface_and_illumination(
    *,
    water_probability: np.ndarray,
    shadow_probability: np.ndarray,
    valid: np.ndarray,
    vegetation_probability: np.ndarray | None = None,
    builtup_probability: np.ndarray | None = None,
    building_probability: np.ndarray | None = None,
    water_threshold: float = 0.50,
) -> SurfaceIlluminationResult:
    """Resolve physical surface independently from illumination state.

    A high-confidence water pixel remains WATER when shadow evidence is also
    high. Competing non-water evidence can dispute weak water, but darkness is
    never used as a surface class.
    """
    water = np.clip(np.asarray(water_probability, dtype="float32"), 0, 1)
    shadow = np.clip(np.asarray(shadow_probability, dtype="float32"), 0, 1)
    valid = np.asarray(valid, dtype=bool)
    vegetation = np.clip(np.asarray(vegetation_probability, dtype="float32"), 0, 1) if vegetation_probability is not None else np.zeros_like(water)
    builtup = np.clip(np.asarray(builtup_probability, dtype="float32"), 0, 1) if builtup_probability is not None else np.zeros_like(water)
    building = np.clip(np.asarray(building_probability, dtype="float32"), 0, 1) if building_probability is not None else np.zeros_like(water)

    surface = np.full(water.shape, SurfaceType.UNKNOWN, dtype="uint8")
    surface[~valid] = SurfaceType.NODATA
    strongest_nonwater = np.maximum.reduce([vegetation, builtup, building])
    # Pixels where red >> green >> blue (warm tones) are bare soil/rock — never water,
    # even if they appear dark.  This suppresses false water on terracotta rooftops and
    # dry laterite ground that passes through the water probability estimator.
    warm_surface = (water < 0.55) & (strongest_nonwater < 0.35) & valid
    is_shadow_dominant = (shadow >= 0.40) & (shadow >= water)  # lowered from 0.45 for better sensitivity
    strong_water = (water >= max(0.60, water_threshold)) & ~is_shadow_dominant & ~warm_surface & valid
    deep_water_in_shadow = (water >= 0.75) & (water > shadow) & ~warm_surface & valid
    supported_water = (water >= water_threshold) & (water >= strongest_nonwater + 0.08) & ~is_shadow_dominant & ~warm_surface & valid
    water_mask = strong_water | deep_water_in_shadow | supported_water
    surface[valid & ~water_mask] = SurfaceType.OTHER
    surface[valid & ~water_mask & (vegetation >= 0.52) & (vegetation >= builtup)] = SurfaceType.VEGETATION
    surface[valid & ~water_mask & (builtup >= 0.52) & (builtup >= vegetation)] = SurfaceType.BUILT_UP
    surface[valid & ~water_mask & (building >= 0.52)] = SurfaceType.BUILDING
    # Assign BARE_LAND to non-water pixels where no specialist has a strong claim.
    # This converts ambiguous OTHER pixels with genuinely low probability of being
    # vegetated or built-up into the more informative BARE_LAND class.
    bare_land_candidate = (
        valid & ~water_mask
        & (surface == SurfaceType.OTHER)
        & (vegetation < 0.30) & (builtup < 0.30) & (building < 0.25)
        & (water < 0.35)
    )
    surface[bare_land_candidate] = SurfaceType.BARE_LAND
    surface[water_mask] = SurfaceType.WATER

    conflict = np.clip(1.0 - np.abs(water - shadow), 0.0, 1.0) * np.minimum(water + shadow, 1.0)
    conflict *= valid
    class_margin = np.abs(water - strongest_nonwater)
    uncertainty = np.maximum(conflict, 1.0 - np.clip(class_margin * 2.5, 0, 1))
    uncertainty = np.where(valid, uncertainty, 1.0).astype("float32")

    illumination = np.full(water.shape, IlluminationState.NORMAL, dtype="uint8")
    illumination[~valid] = IlluminationState.NODATA
    illumination[valid & (shadow >= 0.52)] = IlluminationState.SHADOWED
    illumination[valid & (shadow >= 0.38) & (shadow < 0.52)] = IlluminationState.UNCERTAIN
    shadow_mask = (illumination == IlluminationState.SHADOWED) & valid
    return SurfaceIlluminationResult(
        surface=surface, illumination=illumination, water_mask=water_mask,
        shadow_mask=shadow_mask, conflict=conflict.astype("float32"), uncertainty=uncertainty,
    )
