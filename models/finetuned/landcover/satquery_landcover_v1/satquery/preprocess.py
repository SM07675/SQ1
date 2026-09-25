from __future__ import annotations

import numpy as np


S2_ORDER = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")


def rgb_unit(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype="float32")
    finite = data[np.isfinite(data)]
    if finite.size and float(np.percentile(finite, 99.9)) > 1.5:
        data = data / 255.0
    return np.clip(data, 0.0, 1.0)


def normalized_rgb(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    unit = rgb_unit(image)
    return (unit - np.asarray(mean, dtype="float32")[:, None, None]) / np.asarray(std, dtype="float32")[:, None, None]


def sentinel_reflectance(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype="float32")
    finite = data[np.isfinite(data)]
    if finite.size and float(np.percentile(np.abs(finite), 99.0)) > 2.0:
        data = data / 10000.0
    return data


def spectral_features(image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the exact nine backbone bands, six spectral features, and named evidence."""
    if image.shape[0] != 13:
        raise ValueError("Sentinel-2 spectral preprocessing requires exactly 13 bands.")
    r = sentinel_reflectance(image)
    bands = {name: r[index] for index, name in enumerate(S2_ORDER)}
    blue, green, red = bands["B02"], bands["B03"], bands["B04"]
    nir, cirrus, swir1, swir2 = bands["B08"], bands["B10"], bands["B11"], bands["B12"]

    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a - b) / (a + b + 1e-6)

    evidence = {
        "ndwi": ratio(green, nir),
        "mndwi": ratio(green, swir1),
        "ndvi": ratio(nir, red),
        "awei_shadow": blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2,
        "brightness": (red + green + blue) / 3.0,
        "cirrus": cirrus,
    }
    backbone = np.stack([
        np.clip(red / 0.3, 0, 1), np.clip(green / 0.3, 0, 1), np.clip(blue / 0.3, 0, 1),
        *[np.clip(bands[name] / 0.816, 0, 1) for name in ("B05", "B06", "B07", "B08", "B11", "B12")],
    ]).astype("float32")
    spectral = np.stack([evidence[name] for name in ("ndwi", "mndwi", "ndvi", "awei_shadow", "brightness", "cirrus")]).astype("float32")
    return backbone, spectral, evidence


def water_spectral_support(evidence: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    positive_index = (evidence["ndwi"] >= thresholds["ndwi_min"]) | (evidence["mndwi"] >= thresholds["mndwi_min"])
    not_vegetation = evidence["ndvi"] <= thresholds["ndvi_max"]
    not_deep_shadow = (evidence["awei_shadow"] >= thresholds["awei_shadow_min"]) & (evidence["brightness"] >= thresholds["brightness_min"])
    not_cloud = evidence["cirrus"] <= thresholds["cirrus_max"]
    return positive_index & not_vegetation & not_deep_shadow & not_cloud

