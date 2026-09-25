from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from satquery_engine.services.raster import deterministic_change_detection, validate_inputs


def _write(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 2200000, 10, 10),
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_change_mask_and_area_are_computed(tmp_path: Path) -> None:
    before = np.zeros((96, 96), dtype="float32")
    after = before.copy()
    after[25:60, 30:72] = 100
    a = tmp_path / "before.tif"
    b = tmp_path / "after.tif"
    _write(a, before)
    _write(b, after)

    _, quality = validate_inputs([a, b])
    assert quality.compatible
    result = deterministic_change_detection(a, b, tmp_path)
    assert result["changed_percent"] > 5
    assert result["area_m2"] is not None and result["area_m2"] > 0
    assert result["mask_path"].exists()
    assert result["geojson_path"].exists()
    assert result["semantic_supported"] is False


def _write_no_crs(path: Path, data: np.ndarray) -> None:
    """Write a plain raster without CRS or geotransform (like a PNG opened via rasterio)."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
    ) as dst:
        dst.write(data.astype("float32"), 1)


def test_png_pair_without_crs_is_compatible(tmp_path: Path) -> None:
    """Two images without CRS but same dimensions should be compatible for pixel-space analysis."""
    before = np.zeros((64, 64), dtype="float32")
    after = before.copy()
    after[10:30, 15:45] = 200
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_no_crs(a, before)
    _write_no_crs(b, after)

    _, quality = validate_inputs([a, b])
    assert quality.compatible is True
    assert not quality.blockers
    assert quality.checks.get("geospatial") is False
    assert quality.checks.get("alignment_score") == 1.0
    # Should have warnings about missing CRS but no blockers
    assert any("missing CRS" in warning for warning in quality.warnings)


def test_mixed_crs_pair_is_blocked(tmp_path: Path) -> None:
    """One image with CRS and one without should be blocked."""
    a = tmp_path / "with_crs.tif"
    b = tmp_path / "without_crs.tif"
    _write(a, np.zeros((64, 64), dtype="float32"))
    _write_no_crs(b, np.zeros((64, 64), dtype="float32"))

    _, quality = validate_inputs([a, b])
    assert quality.compatible is False
    assert any("cannot verify alignment" in blocker for blocker in quality.blockers)


def test_single_image_without_crs_is_compatible(tmp_path: Path) -> None:
    """A single image without CRS should produce warnings, not blockers."""
    a = tmp_path / "single.tif"
    _write_no_crs(a, np.random.default_rng(42).random((48, 48), dtype=np.float32))

    _, quality = validate_inputs([a])
    assert quality.compatible is True
    assert not quality.blockers
    assert any("missing CRS" in warning for warning in quality.warnings)


def test_change_detection_works_without_crs(tmp_path: Path) -> None:
    """Pixel-space change detection should produce results with null area."""
    before = np.zeros((64, 64), dtype="float32")
    after = before.copy()
    after[10:40, 10:50] = 150
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    _write_no_crs(a, before)
    _write_no_crs(b, after)

    result = deterministic_change_detection(a, b, tmp_path)
    assert result["changed_percent"] > 5
    # Area is None because no CRS is available for real-world measurement
    assert result["area_m2"] is None
    assert result["mask_path"].exists()
    assert result["geojson_path"].exists()


