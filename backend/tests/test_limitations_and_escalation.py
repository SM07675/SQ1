import numpy as np
import pytest
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from app.schemas import VerdictStatus
from app.services.orchestrator import analyze
from app.services.registration import normalize_and_register_pair
from app.services.croma_pipeline import is_calibrated_sar, sar_backscatter_profile
from app.services.spectral import check_index_bands_available
from app.services.water_model import run_windowed_inference


def _create_rgb_jpeg(path: Path, width: int = 128, height: int = 128) -> Path:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:height // 2, :] = [34, 139, 34]  # Forest green
    arr[height // 2:, :] = [20, 50, 180]   # Water blue
    Image.fromarray(arr).save(path, format="JPEG")
    return path


def _create_rgb_geotiff(path: Path, width: int = 128, height: int = 128) -> Path:
    transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
    crs = "EPSG:32633"
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
        arr = np.full((3, height, width), 120, dtype=np.uint8)
        dst.write(arr)
        dst.descriptions = ("Red", "Green", "Blue")
    return path


def _create_multispectral_geotiff(path: Path, width: int = 128, height: int = 128) -> Path:
    transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
    crs = "EPSG:32633"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        # B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
        data = np.zeros((4, height, width), dtype=np.float32)
        data[0] = 0.05  # Blue
        data[1] = 0.10  # Green
        data[2] = 0.08  # Red
        data[3] = 0.55  # NIR (high for vegetation)
        dst.write(data)
        dst.descriptions = ("B02", "B03", "B04", "B08")
    return path


def _create_uncalibrated_sar_png(path: Path, width: int = 128, height: int = 128) -> Path:
    arr = np.random.randint(20, 220, size=(height, width), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path


# =========================================================================
# TEST A: RGB JPEG
# =========================================================================
@pytest.mark.asyncio
async def test_scenario_a_rgb_jpeg(tmp_path: Path):
    """Test A: RGB JPEG cannot calculate physical NDVI; routes to semantic vegetation with structured limitation."""
    jpeg_path = _create_rgb_jpeg(tmp_path / "scene.jpg")

    has_ndvi, reason = check_index_bands_available(jpeg_path, "ndvi")
    assert has_ndvi is False

    res = await analyze(
        result_id="test_scen_a",
        query="Calculate NDVI for vegetation",
        pair_type="single",
        image_paths=[jpeg_path],
        output_dir=tmp_path / "out_scen_a",
    )

    assert res.verdict.status == VerdictStatus.SUPPORTED_WITH_LIMITATIONS
    assert any("physical NDVI calculation is not available" in lim for lim in res.verdict.limitations)
    assert any(sl.type == "missing_required_band" and sl.task == "NDVI" for sl in res.verdict.structured_limitations)
    # Check that visual/semantic alternative was used
    assert any(e.kind == "vegetation_grounding_evidence" for e in res.evidence)


# =========================================================================
# TEST B: RGB GeoTIFF
# =========================================================================
@pytest.mark.asyncio
async def test_scenario_b_rgb_geotiff(tmp_path: Path):
    """Test B: RGB GeoTIFF lacks NIR/SWIR bands, enforcing honest spectral limitation."""
    tif_path = _create_rgb_geotiff(tmp_path / "rgb_geo.tif")

    has_ndvi, _ = check_index_bands_available(tif_path, "ndvi")
    assert has_ndvi is False

    has_ndbi, _ = check_index_bands_available(tif_path, "ndbi")
    assert has_ndbi is False

    res = await analyze(
        result_id="test_scen_b",
        query="Detect buildings using NDBI",
        pair_type="single",
        image_paths=[tif_path],
        output_dir=tmp_path / "out_scen_b",
    )

    assert res.verdict.status in {VerdictStatus.SUPPORTED, VerdictStatus.SUPPORTED_WITH_LIMITATIONS}
    assert any(sl.type == "missing_required_band" and sl.task == "NDBI" for sl in res.verdict.structured_limitations)


# =========================================================================
# TEST C: Multispectral GeoTIFF
# =========================================================================
@pytest.mark.asyncio
async def test_scenario_c_multispectral_geotiff(tmp_path: Path):
    """Test C: Multispectral GeoTIFF with B04 (Red) and B08 (NIR) computes true NDVI."""
    ms_path = _create_multispectral_geotiff(tmp_path / "sentinel2_ms.tif")

    has_ndvi, msg = check_index_bands_available(ms_path, "ndvi")
    assert has_ndvi is True
    assert msg == ""

    res = await analyze(
        result_id="test_scen_c",
        query="Map forest canopy",
        pair_type="single",
        image_paths=[ms_path],
        output_dir=tmp_path / "out_scen_c",
    )

    veg_ev = next(e for e in res.evidence if e.kind == "vegetation_grounding_evidence")
    assert veg_ev.metrics.get("result_type") == "spectral"
    assert (tmp_path / "out_scen_c" / "ndvi.png").exists()


# =========================================================================
# TEST D: Large Raster Memory & Tiling Telemetry
# =========================================================================
def test_scenario_d_large_raster_telemetry():
    """Test D: Windowed inference executes with dynamic batching and returns telemetry without VRAM exhaustion."""
    large_tile = np.random.rand(3, 768, 768).astype(np.float32)

    import torch

    class DummyModel(torch.nn.Module):
        def forward(self, bb, sp=None):
            return torch.zeros((bb.shape[0], 2, bb.shape[2], bb.shape[3]), dtype=torch.float32)

    model = DummyModel()

    pred, telemetry = run_windowed_inference(
        model,
        large_tile,
        spectral_cube=None,
        tile_size=256,
        overlap=64,
        return_telemetry=True,
    )

    assert pred.shape == (768, 768)
    assert telemetry["tile_count"] >= 16
    assert telemetry["batch_size"] in {1, 2, 4}
    assert "device" in telemetry
    assert "inference_mode" in telemetry


# =========================================================================
# TEST E: Difficult Temporal Alignment & Registration Escalation
# =========================================================================
def test_scenario_e_registration_escalation(tmp_path: Path):
    """Test E: High affine distortion triggers feature matching RANSAC escalation."""
    import cv2

    base = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(base, (100, 100), 40, (255, 255, 255), -1)
    cv2.rectangle(base, (150, 150), (220, 220), (200, 200, 200), -1)
    for i in range(10, 240, 20):
        cv2.line(base, (i, 10), (i, 50), (255, 255, 255), 2)

    path_a = tmp_path / "img_a.png"
    Image.fromarray(base).save(path_a)

    # Apply 15 degree rotation + shear
    rot_mat = cv2.getRotationMatrix2D((128, 128), 15, 1.0)
    rot_mat[0, 2] += 12.0
    rot_mat[1, 2] -= 8.0
    distorted = cv2.warpAffine(base, rot_mat, (256, 256))
    path_b = tmp_path / "img_b.png"
    Image.fromarray(distorted).save(path_b)

    out_dir = tmp_path / "reg_out"
    _, _, report = normalize_and_register_pair(path_a, path_b, out_dir, max_size=256)

    assert report.attempted is True
    # Verify that registration inspected and ran
    assert report.alignment_score > 0.0
    assert report.transform is not None or report.shift_x != 0 or report.shift_y != 0


# =========================================================================
# TEST F: Uncalibrated SAR
# =========================================================================
def test_scenario_f_uncalibrated_sar(tmp_path: Path):
    """Test F: Raw 8-bit SAR PNG is identified as uncalibrated and labeled relative SAR backscatter."""
    sar_path = _create_uncalibrated_sar_png(tmp_path / "raw_sar.png")

    is_calib, source = is_calibrated_sar(sar_path)
    assert is_calib is False
    assert "8-bit" in source or "uncalibrated" in source

    prof = sar_backscatter_profile(sar_path)
    assert prof["is_calibrated"] is False
    assert "relative" in prof.get("label", "").lower() or "relative" in prof.get("summary", "").lower()


# =========================================================================
# TEST G: Valid Optical + SAR Pair with Multispectral
# =========================================================================
@pytest.mark.asyncio
async def test_scenario_g_valid_multispectral_sar(tmp_path: Path):
    """Test G: Sentinel-2 Multispectral + Sentinel-1 SAR runs dual-sensor pipeline."""
    opt_ms = _create_multispectral_geotiff(tmp_path / "opt_ms.tif")
    sar = tmp_path / "sar.tif"
    with rasterio.open(
        sar,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=2,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(500000.0, 4000000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.full((2, 128, 128), -14.0, dtype=np.float32))
        dst.descriptions = ("VV", "VH")

    res = await analyze(
        result_id="test_scen_g",
        query="Compare optical and SAR sensors.",
        pair_type="optical_sar",
        image_paths=[opt_ms, sar],
        output_dir=tmp_path / "out_scen_g",
    )

    assert res.verdict.status in {VerdictStatus.SUPPORTED, VerdictStatus.SUPPORTED_WITH_LIMITATIONS}
    assert any(e.kind == "croma_cross_attention_fusion" for e in res.evidence)
    assert any(e.kind == "sar_structural_evidence" for e in res.evidence)


# =========================================================================
# TEST H: RGB + SAR where CROMA Requirements Not Met
# =========================================================================
@pytest.mark.asyncio
async def test_scenario_h_rgb_sar_skips_croma(tmp_path: Path):
    """Test H: RGB + SAR skips CROMA with explicit limitation and routes to physical consensus."""
    rgb_opt = _create_rgb_png = _create_rgb_jpeg(tmp_path / "opt_rgb.jpg")
    sar = tmp_path / "sar_raw.png"
    Image.fromarray(np.full((128, 128), 100, dtype=np.uint8)).save(sar)

    res = await analyze(
        result_id="test_scen_h",
        query="Find water using optical and SAR.",
        pair_type="optical_sar",
        image_paths=[rgb_opt, sar],
        output_dir=tmp_path / "out_scen_h",
    )

    # Must have structured limitation for incompatible modality
    assert any(sl.type == "incompatible_modality" and sl.task == "croma_cross_attention_fusion" for sl in res.verdict.structured_limitations)
    # Execution trace explicitly explains why CROMA was skipped
    assert any("CROMA skipped because required input channels/modalities were unavailable" in t.action for t in res.trace)
    # Verdict correctly records supported_with_limitations
    assert res.verdict.status == VerdictStatus.SUPPORTED_WITH_LIMITATIONS
