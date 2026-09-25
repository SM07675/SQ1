from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from satquery_engine.services.spectral import (
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
    from satquery_engine.services.spectral import extract_water_grounding

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


def test_water_vs_shadow_vs_green_land_discrimination(tmp_path: Path, monkeypatch) -> None:
    """Verify that water, cast shadow, green meadow, and bare land are cleanly separated."""
    from PIL import Image
    from satquery_engine.services.spectral import classify_land_cover_scene
    import satquery_engine.services.landcover_specialist as learned
    # Isolate the scientific RGB fallback; pretrained models have separate labeled benchmarks.
    monkeypatch.setattr(learned, "predict_landcover", lambda *a, **k: (_ for _ in ()).throw(ValueError("Model unavailable in fallback test")))

    h, w = 128, 128
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Q1 (top-left): Water (R=20, G=60, B=95)
    img[:64, :64] = [20, 60, 95]
    # Q2 (top-right): Cast shadow (dark neutral grey R=28, G=28, B=30)
    img[:64, 64:] = [28, 28, 30]
    # Q3 (bottom-left): Green meadow (R=45, G=140, B=50)
    img[64:, :64] = [45, 140, 50]
    # Q4 (bottom-right): Bare soil / sand (R=200, G=180, B=150)
    img[64:, 64:] = [200, 180, 150]

    img_path = tmp_path / "discrimination_quadrants.png"
    Image.fromarray(img).save(img_path)

    res = classify_land_cover_scene(img_path, tmp_path / "lc_out")
    assert res is not None
    breakdown = res["breakdown"]

    # Water should be detected in Q1 (~20-25%)
    assert breakdown["water"]["percent"] >= 20.0
    # Neither shadow nor bare soil can be established from brightness alone.
    assert breakdown["unknown"]["percent"] >= 45.0
    assert breakdown["built_up"]["percent"] == 0
    assert breakdown["vegetation"]["percent"] >= 20.0
    assert not res["bigearthnet_witness"]["available"]
    # Total water should NOT exceed the Q1 quadrant significantly (i.e., shadow and grass not falsely counted as water)
    assert breakdown["water"]["percent"] <= 26.0


def test_extract_water_grounding_rejects_shadows_and_greenery(tmp_path: Path) -> None:
    """Ensure extract_water_grounding isolates only genuine water, rejecting shadows and lawns."""
    from PIL import Image
    from satquery_engine.services.spectral import extract_water_grounding

    h, w = 128, 128
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Top half: bare ground (R=190, G=170, B=140) with a building shadow (R=25, G=25, B=28) and a lawn (R=40, G=130, B=45)
    img[:64, :] = [190, 170, 140]
    img[10:30, 10:40] = [25, 25, 28]    # Cast shadow
    img[10:30, 50:80] = [40, 130, 45]   # Lawn / green park

    # Bottom half: genuine lake (R=18, G=55, B=90)
    img[64:, :] = [18, 55, 90]

    img_path = tmp_path / "lake_with_park_and_shadow.png"
    Image.fromarray(img).save(img_path)

    res = extract_water_grounding(img_path, tmp_path / "grounding_out")
    assert res is not None
    assert res["target"] == "water"
    assert res["region_count"] == 1
    # Only bottom half (~50%) should be water, shadow and lawn should not be detected
    assert 40.0 <= res["largest_coverage_percent"] <= 55.0
    assert "bottom" in res["location_description"].lower() or "south" in res["location_description"].lower()

