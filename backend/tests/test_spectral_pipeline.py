from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.services.spectral import (
    available_indices,
    optical_sar_water_fusion,
    semantic_change_detection,
    spectral_scene_analysis,
)


def _write_ms(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=5,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 2200000, 10, 10),
    ) as dst:
        for index, name in enumerate(("red", "green", "blue", "nir", "swir"), start=1):
            dst.write(data[index - 1], index)
            dst.set_band_description(index, name)


def _write_sar(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=2,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 2200000, 10, 10),
    ) as dst:
        for index, name in enumerate(("vv", "vh"), start=1):
            dst.write(data[index - 1], index)
            dst.set_band_description(index, name)


def test_spectral_built_up_change_produces_evidence(tmp_path: Path) -> None:
    before = np.zeros((5, 96, 96), dtype="float32")
    before[:] = np.asarray([0.2, 0.3, 0.15, 0.65, 0.22])[:, None, None]
    after = before.copy()
    after[:, 20:70, 30:80] = np.asarray([0.4, 0.36, 0.32, 0.25, 0.62])[:, None, None]
    a, b = tmp_path / "before.tif", tmp_path / "after.tif"
    _write_ms(a, before)
    _write_ms(b, after)

    assert set(available_indices(a)) == {"ndvi", "ndwi", "ndbi"}
    result = semantic_change_detection(a, b, "built-up", "Has built-up area increased?", tmp_path)
    assert result is not None
    assert result["changed_percent"] > 10
    assert result["area_m2"] and result["area_m2"] > 0
    assert result["mask_path"].exists()
    assert result["geojson_path"].exists()


def test_single_image_grounding_and_optical_sar_agreement(tmp_path: Path) -> None:
    optical = np.zeros((5, 96, 96), dtype="float32")
    optical[:] = np.asarray([0.2, 0.3, 0.15, 0.62, 0.24])[:, None, None]
    optical[:, 20:70, 25:75] = np.asarray([0.08, 0.24, 0.18, 0.03, 0.02])[:, None, None]
    radar = np.stack([
        np.full((96, 96), -8.0, dtype="float32"),
        np.full((96, 96), -14.0, dtype="float32"),
    ])
    radar[:, 20:70, 25:75] = -26.0
    optical_path, sar_path = tmp_path / "optical.tif", tmp_path / "sar.tif"
    _write_ms(optical_path, optical)
    _write_sar(sar_path, radar)

    scene = spectral_scene_analysis(optical_path, "water", tmp_path)
    assert scene is not None and scene["coverage_percent"] > 10
    fusion = optical_sar_water_fusion(optical_path, sar_path, tmp_path)
    assert fusion is not None
    assert fusion["sensor_agreement_percent"] > 50
    assert fusion["agreement_path"].exists()


def test_extract_water_grounding_rgb_png(tmp_path: Path) -> None:
    from PIL import Image
    from app.services.spectral import extract_water_grounding

    # Create an RGB synthetic satellite image: left half ocean (dark blue/cyan), right half land/desert (bright sand)
    h, w = 128, 128
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Left half: water (R=25, G=60, B=90)
    img_arr[:, :64] = [25, 60, 90]
    # Right half: desert/urban (R=210, G=190, B=160)
    img_arr[:, 64:] = [210, 190, 160]

    png_path = tmp_path / "scene_rgb.png"
    Image.fromarray(img_arr).save(png_path)

    res = extract_water_grounding(png_path, tmp_path / "water_out", query="Highlight the largest water body.")
    assert res is not None
    assert res["target"] == "water"
    assert res["region_count"] >= 1
    assert res["largest_coverage_percent"] >= 40.0
    assert res["mask_path"].exists()
    assert res["grounding_mask_path"].exists()
    assert res["ndwi_path"].exists()
    assert res["geojson_path"].exists()
    assert res["is_spectral"] is False
    assert res["confidence"] >= 0.85
    assert "left" in res["location_description"].lower() or "west" in res["location_description"].lower()

