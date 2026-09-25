from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy import ndimage as ndi


class ShadowType(IntEnum):
    NONE = 0
    BUILDING_SHADOW = 1
    TERRAIN_SHADOW = 2
    CLOUD_SHADOW = 3
    VEGETATION_SHADOW = 4
    UNKNOWN_SHADOW = 5


@dataclass(frozen=True)
class ShadowEvidence:
    probability: np.ndarray
    mask: np.ndarray
    shadow_type: np.ndarray
    features: dict[str, np.ndarray]
    details: dict[str, Any]


def rgb_shadow_evidence(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
    *,
    water_probability: np.ndarray | None = None,
    building_mask: np.ndarray | None = None,
    vegetation_probability: np.ndarray | None = None,
) -> ShadowEvidence:
    """Estimate illumination shadow independently from surface identity.

    Darkness contributes evidence, but cannot by itself create a shadow mask.
    Local illumination contrast, chromatic continuation, texture/edges, and
    object adjacency provide the additional evidence.
    """
    r, g, b = (np.asarray(channel, dtype="float32") for channel in (r, g, b))
    valid = np.asarray(valid, dtype=bool)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    local_mean = ndi.uniform_filter(np.where(valid, luminance, 0.0), size=21)
    valid_density = ndi.uniform_filter(valid.astype("float32"), size=21)
    local_mean = local_mean / np.maximum(valid_density, 1e-4)
    illumination_drop = np.clip((local_mean - luminance) / np.maximum(local_mean, 0.04), 0.0, 1.0)

    maximum = np.maximum.reduce([r, g, b])
    minimum = np.minimum.reduce([r, g, b])
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-5)
    chromatic_neutrality = 1.0 - np.clip(saturation / 0.35, 0.0, 1.0)
    darkness = np.clip((0.34 - luminance) / 0.34, 0.0, 1.0)
    gradient = np.hypot(ndi.sobel(luminance, axis=0), ndi.sobel(luminance, axis=1)) / 4.0
    edge_support = np.clip(ndi.uniform_filter((gradient > 0.035).astype("float32"), 9) / 0.18, 0.0, 1.0)

    building_adj = np.zeros_like(luminance, dtype="float32")
    if building_mask is not None:
        objects = np.asarray(building_mask, dtype=bool)
        if objects.shape == valid.shape:
            building_adj = (ndi.binary_dilation(objects, iterations=8) & ~objects).astype("float32")
    vegetation_adj = np.zeros_like(luminance, dtype="float32")
    if vegetation_probability is not None:
        vegetation = np.asarray(vegetation_probability, dtype="float32") > 0.55
        if vegetation.shape == valid.shape:
            vegetation_adj = (ndi.binary_dilation(vegetation, iterations=5) & ~vegetation).astype("float32")
    bright_object = (luminance > 0.82) & valid
    bright_adjacency = (ndi.binary_dilation(bright_object, iterations=18) & ~bright_object).astype("float32")

    water_support = np.zeros_like(valid)
    within_water_drop = np.zeros_like(luminance, dtype="float32")
    if water_probability is not None:
        water_support = np.asarray(water_probability, dtype="float32") >= 0.55
        water_density = ndi.uniform_filter(water_support.astype("float32"), size=31)
        water_local = ndi.uniform_filter(np.where(water_support, luminance, 0.0), size=31) / np.maximum(water_density, 1e-4)
        within_water_drop = np.clip((water_local - luminance) / np.maximum(water_local, 0.02), 0.0, 1.0).astype("float32")

    # At least two independent families are required: darkness/local contrast
    # plus geometry/chromatic evidence. A dark pixel alone therefore stays low.
    probability = (
        0.25 * darkness
        + 0.32 * illumination_drop
        + 0.13 * chromatic_neutrality
        + 0.10 * edge_support
        + 0.14 * building_adj
        + 0.06 * vegetation_adj
        + 0.10 * bright_adjacency
    )
    contextual_support = (building_adj > 0) | (vegetation_adj > 0) | (bright_adjacency > 0) | ((edge_support > 0.55) & (chromatic_neutrality > 0.50))
    corroborated = (illumination_drop > 0.20) & contextual_support
    probability = np.where(corroborated, probability, probability * 0.35)
    # Uniform dark water is not a cast shadow. Water may still be shadowed when
    # it has a clear illumination drop relative to neighboring water pixels.
    water_shadow_support = water_support & (within_water_drop > 0.28) & contextual_support
    probability = np.where(water_support & ~water_shadow_support, probability * 0.20, probability)
    probability = np.where(water_shadow_support, np.maximum(probability, 0.42 + 0.45 * within_water_drop), probability)
    probability = np.clip(probability, 0.0, 1.0).astype("float32")
    probability[~valid] = 0.0

    shadow_type = np.full(valid.shape, ShadowType.NONE, dtype="uint8")
    likely = probability >= 0.52
    shadow_type[likely] = ShadowType.UNKNOWN_SHADOW
    shadow_type[likely & (bright_adjacency > 0)] = ShadowType.CLOUD_SHADOW
    shadow_type[likely & (vegetation_adj > 0)] = ShadowType.VEGETATION_SHADOW
    shadow_type[likely & (building_adj > 0)] = ShadowType.BUILDING_SHADOW

    # Water and shadow are allowed to overlap. Strong independent water evidence
    # is never removed from the illumination product.
    shaded_water_pixels = int((likely & (np.asarray(water_probability) >= 0.60)).sum()) if water_probability is not None else 0
    return ShadowEvidence(
        probability=probability,
        mask=likely & valid,
        shadow_type=shadow_type,
        features={
            "darkness": darkness.astype("float32"),
            "illumination_drop": illumination_drop.astype("float32"),
            "chromatic_neutrality": chromatic_neutrality.astype("float32"),
            "edge_support": edge_support.astype("float32"),
            "building_adjacency": building_adj,
            "vegetation_adjacency": vegetation_adj,
            "bright_object_adjacency": bright_adjacency,
            "within_water_illumination_drop": within_water_drop,
        },
        details={"method":"rgb_local_illumination_geometry_v1", "shaded_water_pixels": shaded_water_pixels},
    )


def multispectral_shadow_evidence(
    brightness: np.ndarray,
    valid: np.ndarray,
    spectral_water_probability: np.ndarray,
    *,
    vegetation_probability: np.ndarray | None = None,
    builtup_probability: np.ndarray | None = None,
) -> ShadowEvidence:
    brightness = np.asarray(brightness, dtype="float32")
    valid = np.asarray(valid, dtype=bool)
    local = ndi.uniform_filter(np.where(valid, brightness, 0.0), 21)
    density = ndi.uniform_filter(valid.astype("float32"), 21)
    local = local / np.maximum(density, 1e-4)
    illumination_drop = np.clip((local - brightness) / np.maximum(local, 0.01), 0.0, 1.0)
    finite_values = brightness[valid]
    dark_scale = float(np.percentile(finite_values, 35)) if finite_values.size else 0.0
    darkness = np.clip((dark_scale - brightness) / max(dark_scale, 1e-4), 0.0, 1.0)
    water = np.asarray(spectral_water_probability, dtype="float32")
    nonwater_support = 1.0 - water
    vegetation = np.asarray(vegetation_probability, dtype="float32") if vegetation_probability is not None else np.zeros_like(water)
    builtup = np.asarray(builtup_probability, dtype="float32") if builtup_probability is not None else np.zeros_like(water)
    context = np.maximum(vegetation, builtup)
    probability = np.clip(0.40 * illumination_drop + 0.20 * darkness + 0.22 * nonwater_support + 0.18 * context, 0.0, 1.0)
    corroborated = (illumination_drop > 0.22) & ((nonwater_support > 0.55) | (context > 0.45))
    probability = np.where(corroborated, probability, probability * 0.30).astype("float32")
    probability[~valid] = 0.0
    mask = (probability >= 0.52) & valid
    shadow_type = np.full(valid.shape, ShadowType.NONE, dtype="uint8")
    shadow_type[mask] = ShadowType.UNKNOWN_SHADOW
    shadow_type[mask & (vegetation >= builtup) & (vegetation > 0.45)] = ShadowType.VEGETATION_SHADOW
    shadow_type[mask & (builtup > vegetation) & (builtup > 0.45)] = ShadowType.BUILDING_SHADOW
    return ShadowEvidence(
        probability=probability, mask=mask, shadow_type=shadow_type,
        features={"darkness":darkness.astype("float32"), "illumination_drop":illumination_drop.astype("float32"), "nonwater_support":nonwater_support.astype("float32")},
        details={"method":"multispectral_illumination_spectral_disagreement_v1", "shaded_water_pixels":int((mask & (water >= 0.60)).sum())},
    )
