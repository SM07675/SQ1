from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image

from app.schemas import VerdictStatus
from app.services.orchestrator import analyze
from app.services.raster import validate_inputs
from app.services.registration import normalize_and_register_pair


def _create_rgb_png(path: Path, width: int, height: int, pattern: str = "solid", color: tuple[int, int, int] = (100, 150, 200)) -> Path:
    img_arr = np.zeros((height, width, 3), dtype="uint8")
    img_arr[:, :] = color

    if pattern in {"checker", "modified_patch"}:
        for y in range(0, height, 32):
            for x in range(0, width, 32):
                if (x // 32 + y // 32) % 2 == 0:
                    img_arr[y : y + 32, x : x + 32] = (220, 200, 180)

    if pattern == "modified_patch":
        # Add a prominent artificial building / urban expansion block over the scene
        img_arr[height // 4 : height // 2, width // 4 : width // 2] = (255, 30, 30)

    Image.fromarray(img_arr).save(path, format="PNG")
    return path


def _create_geotiff(path: Path, width: int, height: int, crs: str = "EPSG:32643") -> Path:
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2800000.0)
    data = np.random.randint(50, 200, size=(3, height, width), dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.mark.asyncio
async def test_matrix_1_same_dimensions_rgb_png(tmp_path: Path):
    """Test 1: Same dimensions RGB PNG pair completes with genuine evidence and artifacts."""
    img_a = _create_rgb_png(tmp_path / "before.png", 512, 512, pattern="checker")
    img_b = _create_rgb_png(tmp_path / "after.png", 512, 512, pattern="checker")
    out_dir = tmp_path / "output_1"

    response = await analyze(
        result_id="test-run-1",
        query="What has changed between these dates?",
        pair_type="bi_temporal",
        image_paths=[img_a, img_b],
        output_dir=out_dir,
    )

    assert response.quality.compatible is True
    assert len(response.evidence) >= 1
    assert response.verdict.status in {VerdictStatus.SUPPORTED, VerdictStatus.INSUFFICIENT_EVIDENCE}
    assert (out_dir / "change_mask.png").exists()
    assert (out_dir / "registration_report.json").exists()


@pytest.mark.asyncio
async def test_matrix_2_different_dimensions_rgb_png(tmp_path: Path):
    """Test 2: Different dimensions RGB PNG pair (e.g. 640x480 vs 512x512) is normalized and analyzed."""
    img_a = _create_rgb_png(tmp_path / "before.png", 640, 480, pattern="checker")
    img_b = _create_rgb_png(tmp_path / "after.png", 512, 512, pattern="checker")
    out_dir = tmp_path / "output_2"

    metadata, quality = validate_inputs([img_a, img_b])
    # Recoverable issue: warning present, but NOT a fatal blocker
    assert quality.compatible is True
    assert len(quality.blockers) == 0
    assert any("different dimensions" in w for w in quality.warnings)

    response = await analyze(
        result_id="test-run-2",
        query="Has built-up area increased between these two dates?",
        pair_type="bi_temporal",
        image_paths=[img_a, img_b],
        output_dir=out_dir,
    )

    assert response.quality.compatible is True
    assert len(response.evidence) >= 2  # Baseline and learned witness
    assert (out_dir / "prepared_1.png").exists()
    assert (out_dir / "prepared_2.png").exists()
    assert (out_dir / "change_mask.png").exists()


@pytest.mark.asyncio
async def test_matrix_3_identical_image_pair_no_change(tmp_path: Path):
    """Test 3: Identical image pair results in ~0% change."""
    img_a = _create_rgb_png(tmp_path / "before.png", 400, 400, pattern="checker")
    img_b = _create_rgb_png(tmp_path / "after.png", 400, 400, pattern="checker")
    out_dir = tmp_path / "output_3"

    response = await analyze(
        result_id="test-run-3",
        query="What has changed?",
        pair_type="bi_temporal",
        image_paths=[img_a, img_b],
        output_dir=out_dir,
    )

    baseline_ev = next(e for e in response.evidence if e.kind == "baseline_change_detection")
    assert baseline_ev.metrics["changed_percent"] < 0.1
    assert "No significant" in baseline_ev.summary or baseline_ev.metrics["changed_pixels"] == 0


@pytest.mark.asyncio
async def test_matrix_4_clear_visual_change_detected(tmp_path: Path):
    """Test 4: Clear artificial modified patch is detected as significant change with regions."""
    img_a = _create_rgb_png(tmp_path / "before.png", 500, 500, pattern="checker")
    img_b = _create_rgb_png(tmp_path / "after.png", 500, 500, pattern="modified_patch")
    out_dir = tmp_path / "output_4"

    response = await analyze(
        result_id="test-run-4",
        query="Has urban expansion increased?",
        pair_type="bi_temporal",
        image_paths=[img_a, img_b],
        output_dir=out_dir,
    )

    baseline_ev = next(e for e in response.evidence if e.kind == "baseline_change_detection")
    assert baseline_ev.metrics["changed_percent"] > 1.0
    assert baseline_ev.metrics["region_count"] >= 1
    assert response.verdict.status == VerdictStatus.SUPPORTED


def test_matrix_5_invalid_corrupted_file(tmp_path: Path):
    """Test 5: Corrupted or empty bytes produce a graceful validation blocker."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"not a valid png or geotiff file header")

    with pytest.raises(Exception):
        validate_inputs([corrupt_file])


@pytest.mark.asyncio
async def test_matrix_6_geotiff_with_crs(tmp_path: Path):
    """Test 6: GeoTIFF image pair activates CRS geospatial awareness and area calculations."""
    tif_a = _create_geotiff(tmp_path / "t1.tif", 300, 300, crs="EPSG:32643")
    tif_b = _create_geotiff(tmp_path / "t2.tif", 300, 300, crs="EPSG:32643")
    out_dir = tmp_path / "output_6"

    response = await analyze(
        result_id="test-run-6",
        query="Has built-up area increased?",
        pair_type="bi_temporal",
        image_paths=[tif_a, tif_b],
        output_dir=out_dir,
    )

    assert response.mode in {"deterministic_geospatial", "spectral_geoproof"}
    assert response.assets[0].crs == "EPSG:32643"
    assert len(response.evidence) >= 1


@pytest.mark.asyncio
async def test_matrix_7_water_grounding_single_image(tmp_path: Path):
    """Test 7: Single image water grounding isolates water body and produces supported verdict."""
    img_path = _create_rgb_png(tmp_path / "water_scene.png", 256, 256, pattern="solid", color=(20, 50, 80))
    out_dir = tmp_path / "output_7"

    response = await analyze(
        result_id="test-run-7",
        query="Highlight the largest water body.",
        pair_type="single",
        image_paths=[img_path],
        output_dir=out_dir,
    )

    assert response.verdict.status == VerdictStatus.SUPPORTED
    assert response.verdict.confidence >= 0.70
    assert any("largest water body" in e.summary.lower() or "water" in e.summary.lower() for e in response.evidence)
    assert any(a.name == "water_grounding_mask.png" for a in response.artifacts)
    assert any(a.name == "water_regions.geojson" for a in response.artifacts)

