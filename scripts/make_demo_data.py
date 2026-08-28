from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


TRANSFORM = from_origin(500000, 2200000, 10, 10)
CRS = "EPSG:32643"


def write_multispectral(path: Path, data: np.ndarray) -> None:
    descriptions = ("red", "green", "blue", "nir", "swir")
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": CRS,
        "transform": TRANSFORM,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for band, description in enumerate(descriptions, start=1):
            dst.write(data[band - 1].astype("float32"), band)
            dst.set_band_description(band, description)
            dst.update_tags(band, wavelength_role=description)
        dst.update_tags(sensor="SYNTHETIC-S2-LIKE", purpose="deterministic SIH demonstration")


def write_sar(path: Path, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": 2,
        "dtype": "float32",
        "crs": CRS,
        "transform": TRANSFORM,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for band, description in enumerate(("vv", "vh"), start=1):
            dst.write(data[band - 1].astype("float32"), band)
            dst.set_band_description(band, description)
        dst.update_tags(sensor="SYNTHETIC-S1-LIKE", purpose="deterministic SIH demonstration")


def base_scene(size: int = 384) -> np.ndarray:
    # Red, Green, Blue, NIR and SWIR reflectance-like values.
    scene = np.empty((5, size, size), dtype="float32")
    scene[:] = np.asarray([0.19, 0.31, 0.16, 0.62, 0.24], dtype="float32")[:, None, None]
    # Existing urban block.
    scene[:, 210:320, 215:350] = np.asarray([0.41, 0.36, 0.32, 0.27, 0.59])[:, None, None]
    # Permanent water body.
    scene[:, 45:145, 35:150] = np.asarray([0.08, 0.22, 0.18, 0.04, 0.03])[:, None, None]
    # Bare-soil strip.
    scene[:, 165:205, 25:175] = np.asarray([0.34, 0.31, 0.24, 0.29, 0.39])[:, None, None]
    return scene


def add_noise(scene: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = scene + rng.normal(0, 0.008, size=scene.shape).astype("float32")
    return np.clip(noisy, 0.005, 0.95)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(parents=True, exist_ok=True)

    before = add_noise(base_scene(), 26167)
    after_base = base_scene()
    # New construction replaces vegetation. This creates a measurable NDBI rise.
    after_base[:, 105:205, 180:285] = np.asarray([0.43, 0.37, 0.33, 0.26, 0.61])[:, None, None]
    # Small water expansion provides a second interpretable signal.
    after_base[:, 120:168, 65:150] = np.asarray([0.08, 0.22, 0.18, 0.04, 0.03])[:, None, None]
    after = add_noise(after_base, 26168)
    write_multispectral(out / "demo_before_multispectral.tif", before)
    write_multispectral(out / "demo_after_multispectral.tif", after)

    # Co-registered SAR evidence: water is intentionally low-backscatter.
    rng = np.random.default_rng(26169)
    vv = rng.normal(-8.0, 1.3, size=(384, 384)).astype("float32")
    vh = rng.normal(-14.0, 1.5, size=(384, 384)).astype("float32")
    vv[45:145, 35:150] = rng.normal(-22.0, 0.8, size=(100, 115))
    vh[45:145, 35:150] = rng.normal(-28.0, 0.9, size=(100, 115))
    write_sar(out / "demo_sar_water.tif", np.stack([vv, vh]))
    write_multispectral(out / "demo_optical_water.tif", before)

    readme = out / "DEMO_QUERIES.txt"
    readme.write_text(
        "BI-TEMPORAL\n"
        "A: demo_before_multispectral.tif\n"
        "B: demo_after_multispectral.tif\n"
        "Query: Has built-up area increased between these two dates?\n\n"
        "OPTICAL + SAR\n"
        "A: demo_optical_water.tif\n"
        "B: demo_sar_water.tif\n"
        "Query: Use optical and SAR evidence together to identify water-covered regions.\n\n"
        "SINGLE IMAGE\n"
        "A: demo_before_multispectral.tif\n"
        "Query: Highlight the largest water body.\n",
        encoding="utf-8",
    )
    for path in sorted(out.glob("demo_*.tif")):
        print(path)
    print(readme)


if __name__ == "__main__":
    main()
