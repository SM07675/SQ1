from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from app.schemas import VerdictStatus
from app.services.croma_pipeline import (
    ModalityReport,
    inspect_modality,
    is_multispectral_optical,
    is_sar_image,
    run_croma_fusion,
    sar_backscatter_profile,
)
from app.services.model_registry import ModelRegistry
from app.services.orchestrator import analyze
from app.services.planner import plan_query



def _write_multiband(path: Path, data: np.ndarray, descriptions: list[str]) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 2200000, 10, 10),
    ) as dst:
        for i, desc in enumerate(descriptions, start=1):
            dst.write(data[i - 1].astype("float32"), i)
            dst.set_band_description(i, desc)


@pytest.fixture
def test_optical_sar_data(tmp_path: Path):
    sar_path = tmp_path / "sentinel1_sar_scene.tif"
    opt_ms_path = tmp_path / "sentinel2_optical_ms.tif"
    opt_rgb_path = tmp_path / "drone_optical_rgb.tif"

    # Synthetic SAR with water (-25 dB in center) and built-up (-8 dB in corner)
    vv = np.full((64, 64), -16.0, dtype=np.float32)
    vv[10:30, 10:30] = -26.0  # low specular return (water)
    vv[40:60, 40:60] = -7.0   # strong double-bounce (urban/built-up)
    vh = np.full((64, 64), -22.0, dtype=np.float32)
    vh[10:30, 10:30] = -30.0
    vh[40:60, 40:60] = -12.0
    _write_multiband(sar_path, np.stack([vv, vh]), ["vv", "vh"])

    # Synthetic Multispectral Optical (B2, B3, B4, B8 - Blue, Green, Red, NIR)
    blue = np.full((64, 64), 0.12, dtype=np.float32)
    green = np.full((64, 64), 0.14, dtype=np.float32)
    green[10:30, 10:30] = 0.35  # water high green
    red = np.full((64, 64), 0.10, dtype=np.float32)
    red[40:60, 40:60] = 0.45    # urban bright
    nir = np.full((64, 64), 0.35, dtype=np.float32)
    nir[10:30, 10:30] = 0.04    # water low NIR
    nir[40:60, 40:60] = 0.42    # urban high NIR
    _write_multiband(opt_ms_path, np.stack([blue, green, red, nir]), ["b2", "b3", "b4", "b8"])

    # Synthetic 3-band RGB Optical
    rgb_arr = np.stack([red, green, blue])
    _write_multiband(opt_rgb_path, rgb_arr, ["red", "green", "blue"])

    return {
        "sar": sar_path,
        "opt_ms": opt_ms_path,
        "opt_rgb": opt_rgb_path,
    }


@pytest.fixture
def grayscale_sar_like_png(tmp_path: Path) -> Path:
    """A grayscale PNG that LOOKS like SAR but has no SAR metadata."""
    path = tmp_path / "grayscale_visual.png"
    # Uniform grayscale with some texture variation (speckle-like)
    rng = np.random.default_rng(42)
    arr = rng.integers(40, 180, size=(64, 64), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)
    return path


@pytest.fixture
def verified_sar_geotiff(tmp_path: Path) -> Path:
    """A float32 GeoTIFF with VV/VH band descriptions and dB values — verified SAR."""
    path = tmp_path / "sentinel1_vv_vh.tif"
    vv = np.full((64, 64), -15.0, dtype=np.float32)
    vv[10:30, 10:30] = -25.0
    vh = np.full((64, 64), -20.0, dtype=np.float32)
    _write_multiband(path, np.stack([vv, vh]), ["vv", "vh"])
    return path


def test_sar_backscatter_profile(test_optical_sar_data):
    sar_path = test_optical_sar_data["sar"]
    profile = sar_backscatter_profile(sar_path)
    assert "vv" in profile["polarizations"]
    assert profile["mean_db"] < 0
    assert profile["specular_percent"] > 0
    assert profile["structural_percent"] > 0
    assert "double-bounce" in profile["summary"] or "specular" in profile["summary"]


def test_is_multispectral_optical(test_optical_sar_data):
    assert is_multispectral_optical(test_optical_sar_data["opt_ms"]) is True
    assert is_multispectral_optical(test_optical_sar_data["opt_rgb"]) is False
    assert is_sar_image(test_optical_sar_data["sar"]) is True
    assert is_sar_image(test_optical_sar_data["opt_ms"]) is False


# ─── inspect_modality unit tests (Cases A-D) ─────────────────────────────────

def test_inspect_modality_case_a_rgb_optical_png(tmp_path):
    """Case A: RGB optical PNG → optical, sensor_unverified."""
    path = tmp_path / "scene_optical.png"
    arr = np.stack([
        np.full((32, 32), 120, dtype=np.uint8),
        np.full((32, 32), 150, dtype=np.uint8),
        np.full((32, 32), 90, dtype=np.uint8),
    ], axis=-1)
    Image.fromarray(arr, mode="RGB").save(path)
    mr = inspect_modality(path)
    assert mr.modality == "optical"
    assert mr.sensor_verified is False
    assert mr.sar_calibrated is False
    assert mr.band_count == 3


def test_inspect_modality_case_b_optical_geotiff_with_band_metadata(test_optical_sar_data):
    """Case B: Multispectral GeoTIFF with Sentinel-2 band names → optical, sensor_verified."""
    mr = inspect_modality(test_optical_sar_data["opt_ms"])
    assert mr.modality == "optical"
    assert mr.sensor_verified is True  # Sentinel-2 band descriptions present
    assert mr.sar_calibrated is False


def test_inspect_modality_case_c_verified_sar_geotiff(verified_sar_geotiff):
    """Case C: Float32 GeoTIFF with VV/VH band descriptions → sar, sensor_verified."""
    mr = inspect_modality(verified_sar_geotiff)
    assert mr.modality == "sar"
    assert mr.sensor_verified is True
    assert "vv" in mr.sar_polarizations or "VV" in [p.upper() for p in mr.sar_polarizations]
    # is_sar_image must also return True
    assert is_sar_image(verified_sar_geotiff) is True


def test_inspect_modality_case_d_grayscale_sar_like_png(grayscale_sar_like_png):
    """Case D: Grayscale PNG without SAR metadata → sar_like, NOT sar."""
    mr = inspect_modality(grayscale_sar_like_png)
    # Must NOT be classified as verified SAR
    assert mr.modality in ("sar_like", "optical", "unknown")  # never "sar"
    assert mr.modality != "sar", "Grayscale PNG with no SAR metadata must not be classified as verified SAR"
    assert mr.sensor_verified is False
    assert mr.sar_calibrated is False
    # is_sar_image must return False (only True for verified SAR)
    assert is_sar_image(grayscale_sar_like_png) is False


def test_planner_optical_sar_routing():
    # 1. "What does optical show?"
    p_opt = plan_query("What does optical show?", 2, "optical_sar")
    assert p_opt.specific_task == "optical_focus_analysis"

    # 2. "What does SAR show?"
    p_sar = plan_query("What does SAR show?", 2, "optical_sar")
    assert p_sar.specific_task == "sar_focus_analysis"

    # 3. "What additional information does SAR provide?"
    p_comp = plan_query("What additional information does SAR provide?", 2, "optical_sar")
    assert p_comp.specific_task == "sar_complementary_analysis"

    # 4. "Identify built-up areas using both."
    p_built = plan_query("Identify built-up areas using both.", 2, "optical_sar")
    assert p_built.specific_task == "cross_modal_builtup"

    # 5. "Identify water using optical and SAR."
    p_wat = plan_query("Identify water using optical and SAR.", 2, "optical_sar")
    assert p_wat.specific_task == "cross_modal_water"

    # 6. "Compare optical and SAR."
    p_comp2 = plan_query("Compare optical and SAR.", 2, "optical_sar")
    assert p_comp2.specific_task == "cross_modal_comparative"


@pytest.mark.asyncio
async def test_optical_sar_what_does_optical_show(test_optical_sar_data, tmp_path: Path):
    res = await analyze(
        result_id="test_what_optical_shows",
        query="What does optical show?",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_ms"], test_optical_sar_data["sar"]],
        output_dir=tmp_path / "out_opt_show",
    )
    assert res.verdict.status == VerdictStatus.SUPPORTED
    assert "Optical" in res.verdict.answer
    assert any(e.kind == "optical_scene_evidence" for e in res.evidence)
    assert any(e.kind == "sar_structural_evidence" for e in res.evidence)


@pytest.mark.asyncio
async def test_optical_sar_what_does_sar_show(test_optical_sar_data, tmp_path: Path):
    res = await analyze(
        result_id="test_what_sar_shows",
        query="What does SAR show?",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_ms"], test_optical_sar_data["sar"]],
        output_dir=tmp_path / "out_sar_show",
    )
    assert res.verdict.status == VerdictStatus.SUPPORTED
    assert "SAR analysis indicates" in res.verdict.answer or "backscatter" in res.verdict.answer.lower()
    assert any(e.kind == "sar_structural_evidence" for e in res.evidence)


@pytest.mark.asyncio
async def test_optical_sar_additional_information(test_optical_sar_data, tmp_path: Path):
    res = await analyze(
        result_id="test_sar_info_add",
        query="What additional information does SAR provide?",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_ms"], test_optical_sar_data["sar"]],
        output_dir=tmp_path / "out_sar_info",
    )
    assert res.verdict.status == VerdictStatus.SUPPORTED
    assert "penetrates" in res.verdict.answer.lower() or "double-bounce" in res.verdict.answer.lower()
    assert any(e.kind == "sar_structural_evidence" for e in res.evidence)


@pytest.mark.asyncio
async def test_optical_sar_builtup_using_both(test_optical_sar_data, tmp_path: Path):
    res = await analyze(
        result_id="test_builtup_both",
        query="Identify built-up areas using both.",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_ms"], test_optical_sar_data["sar"]],
        output_dir=tmp_path / "out_builtup",
    )
    assert res.verdict.status == VerdictStatus.SUPPORTED
    assert "built-up" in res.verdict.answer.lower()
    assert any(e.kind == "building_detection_evidence" or e.kind == "croma_cross_attention_fusion" for e in res.evidence)


@pytest.mark.asyncio
async def test_optical_sar_rgb_only_skips_croma_vit(test_optical_sar_data, tmp_path: Path):
    # Tests rule: "Do NOT use CROMA on RGB-only data"
    res = await analyze(
        result_id="test_rgb_sar",
        query="Identify water using optical and SAR.",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_rgb"], test_optical_sar_data["sar"]],
        output_dir=tmp_path / "out_rgb_sar",
    )
    assert res.verdict.status in {VerdictStatus.SUPPORTED, VerdictStatus.SUPPORTED_WITH_LIMITATIONS}
    # Limitation clearly marks that CROMA ViT was skipped due to 3-band RGB
    assert any("3-band RGB" in lim for lim in res.verdict.limitations)
    # Evidence uses physical sensor consensus rather than pretending deep CROMA ViT was run
    croma_ev = next(e for e in res.evidence if e.kind == "croma_cross_attention_fusion")
    assert croma_ev.producer == "optical_sar_sensor_consensus_v1"


@pytest.mark.asyncio
async def test_optical_sar_sar_like_png_is_insufficient_for_calibrated_fusion(
    test_optical_sar_data, grayscale_sar_like_png, tmp_path: Path
):
    """Case G (exact current failure): optical + grayscale SAR-like PNG.

    Expected: SUPPORTED_WITH_LIMITATIONS (NOT INSUFFICIENT_EVIDENCE).
    The system must identify the second image as sar_like (not optical),
    explain why calibrated SAR fusion is unavailable, and offer visual comparison.
    The old wrong result was 'Both uploaded images are optical data'.
    """
    res = await analyze(
        result_id="test_sar_like_png",
        query="Analyze these optical and SAR images together.",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_rgb"], grayscale_sar_like_png],
        output_dir=tmp_path / "out_sar_like",
    )
    # Must NOT produce the old misleading error
    assert "Both uploaded images are optical data" not in res.verdict.answer
    # Must produce supported_with_limitations (visual comparison available)
    assert res.verdict.status == VerdictStatus.SUPPORTED_WITH_LIMITATIONS
    # Must have a structured limitation for unverified SAR identity
    assert any(
        sl.type == "unverified_sar_identity" for sl in res.verdict.structured_limitations
    ), "Expected unverified_sar_identity structured limitation"
    # Must have CROMA skipped limitation
    assert any(
        sl.type == "incompatible_modality" and sl.task == "croma_cross_attention_fusion"
        for sl in res.verdict.structured_limitations
    ), "Expected CROMA incompatible_modality structured limitation"
    # Must have evidence for both optical scene and sar_like comparison
    assert any(e.kind == "optical_scene_evidence" for e in res.evidence)
    assert any(e.kind == "sar_like_visual_comparison" for e in res.evidence)
    # Modality reports must be in quality.checks
    assert "modality_image_0" in res.quality.checks
    assert "modality_image_1" in res.quality.checks
    # Image 2 must be identified as sar_like (or at worst unknown), NOT optical or sar
    mr1 = res.quality.checks["modality_image_1"]
    assert mr1["modality"] != "sar", "Grayscale PNG must not be classified as verified SAR"
    assert mr1["sensor_verified"] is False


@pytest.mark.asyncio
async def test_missing_sar_error_handling(test_optical_sar_data, tmp_path: Path):
    # Both images are optical: missing SAR image → still INSUFFICIENT_EVIDENCE
    res = await analyze(
        result_id="test_missing_sar",
        query="Compare optical and SAR.",
        pair_type="optical_sar",
        image_paths=[test_optical_sar_data["opt_ms"], test_optical_sar_data["opt_rgb"]],
        output_dir=tmp_path / "out_missing_sar",
    )
    assert res.verdict.status == VerdictStatus.INSUFFICIENT_EVIDENCE
    assert "Both uploaded images are optical data" in res.verdict.answer
    assert any("Missing complementary SAR" in lim for lim in res.verdict.limitations)


@pytest.mark.asyncio
async def test_auto_detect_pair_type(test_optical_sar_data, tmp_path: Path):
    # When user passes pair_type="auto" with optical + SAR, it auto-detects optical_sar
    res = await analyze(
        result_id="test_auto_pair_detect",
        query="What additional information does SAR provide?",
        pair_type="auto",
        image_paths=[test_optical_sar_data["sar"], test_optical_sar_data["opt_ms"]],
        output_dir=tmp_path / "out_auto_detect",
    )
    assert res.verdict.status == VerdictStatus.SUPPORTED
    assert res.task_plan.application == "optical_sar"


@pytest.mark.asyncio
async def test_auto_detect_pair_type_with_sar_like(test_optical_sar_data, grayscale_sar_like_png, tmp_path: Path):
    """Auto-detection must route optical+sar_like to optical_sar workflow (not single/bi-temporal)."""
    res = await analyze(
        result_id="test_auto_sar_like",
        query="Analyze these images together.",
        pair_type="auto",
        image_paths=[test_optical_sar_data["opt_rgb"], grayscale_sar_like_png],
        output_dir=tmp_path / "out_auto_sar_like",
    )
    # Should route to optical_sar workflow and produce supported_with_limitations
    assert res.verdict.status == VerdictStatus.SUPPORTED_WITH_LIMITATIONS
    # Must not silently misroute as bi-temporal change detection
    assert res.task_plan.task.value != "bi_temporal_change"

