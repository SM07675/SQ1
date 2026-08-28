from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.croma_pipeline import (
    prepare_sentinel1_sar,
    prepare_sentinel2_optical,
    run_croma_fusion,
)
from app.services.model_registry import ModelRegistry


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


def test_prepare_channels(tmp_path: Path):
    sar_path = tmp_path / "sar.tif"
    opt_path = tmp_path / "optical.tif"

    sar_data = np.stack([
        np.full((48, 48), -22.0, dtype=np.float32),  # VV in dB
        np.full((48, 48), -28.0, dtype=np.float32),  # VH in dB
    ])
    _write_multiband(sar_path, sar_data, ["vv", "vh"])

    opt_data = np.stack([
        np.full((48, 48), 0.15, dtype=np.float32),  # Green
        np.full((48, 48), 0.05, dtype=np.float32),  # NIR
    ])
    _write_multiband(opt_path, opt_data, ["green", "nir"])

    sar_norm, sar_ch = prepare_sentinel1_sar(sar_path, (48, 48))
    assert sar_norm.shape == (2, 48, 48)
    assert "vv" in sar_ch
    assert "vh" in sar_ch

    opt_norm, opt_ch = prepare_sentinel2_optical(opt_path, (48, 48))
    assert opt_norm.shape == (2, 48, 48)
    assert len(opt_ch) == 2


@pytest.mark.asyncio
async def test_run_croma_fusion(tmp_path: Path):
    sar_path = tmp_path / "sar.tif"
    opt_path = tmp_path / "optical.tif"

    # Water region in both SAR (-25 dB) and Optical (high NDWI: green 0.3, NIR 0.05)
    vv = np.full((64, 64), -8.0, dtype=np.float32)
    vv[10:40, 10:40] = -25.0
    vh = np.full((64, 64), -14.0, dtype=np.float32)
    vh[10:40, 10:40] = -28.0
    _write_multiband(sar_path, np.stack([vv, vh]), ["vv", "vh"])

    green = np.full((64, 64), 0.10, dtype=np.float32)
    green[10:40, 10:40] = 0.35
    nir = np.full((64, 64), 0.40, dtype=np.float32)
    nir[10:40, 10:40] = 0.05
    _write_multiband(opt_path, np.stack([green, nir]), ["green", "nir"])

    reg = ModelRegistry()
    res = await run_croma_fusion(
        optical_path=opt_path,
        sar_path=sar_path,
        output_dir=tmp_path,
        registry=reg,
    )

    assert res.radar_optical_iou > 50.0
    assert res.sensor_agreement_score >= 0.5
    assert res.agreement_mask_path.exists()
    assert res.sar_db_preview_path.exists()
    assert res.optical_preview_path.exists()
    assert res.geojson_path.exists()
    assert res.confidence >= 0.75
    assert res.cosine_similarity > 0.5
    assert res.confirmed_percent > 0.0
    assert res.sensor_agreement_iou == res.radar_optical_iou


@pytest.mark.asyncio
async def test_optical_sar_analyze_pipeline_end_to_end(tmp_path: Path):
    from app.services.orchestrator import analyze
    from app.schemas import VerdictStatus

    sar_path = tmp_path / "sentinel1_sar.tif"
    opt_path = tmp_path / "sentinel2_optical.tif"

    # Synthetic water body in both SAR and Optical
    vv = np.full((64, 64), -8.0, dtype=np.float32)
    vv[10:40, 10:40] = -25.0
    vh = np.full((64, 64), -14.0, dtype=np.float32)
    vh[10:40, 10:40] = -28.0
    _write_multiband(sar_path, np.stack([vv, vh]), ["vv", "vh"])

    green = np.full((64, 64), 0.10, dtype=np.float32)
    green[10:40, 10:40] = 0.35
    nir = np.full((64, 64), 0.40, dtype=np.float32)
    nir[10:40, 10:40] = 0.05
    _write_multiband(opt_path, np.stack([green, nir]), ["green", "nir"])

    # 1. Test normal order [opt, sar]
    res_normal = await analyze(
        result_id="test_croma_normal",
        query="Use optical and SAR evidence together to identify water-covered regions.",
        pair_type="optical_sar",
        image_paths=[opt_path, sar_path],
        output_dir=tmp_path / "out_normal",
    )
    assert res_normal.verdict.status == VerdictStatus.SUPPORTED
    assert any(e.kind == "croma_cross_attention_fusion" for e in res_normal.evidence)
    assert any(a.name == "croma_sensor_agreement.png" for a in res_normal.artifacts)

    # 2. Test reverse upload order [sar, opt]
    res_reverse = await analyze(
        result_id="test_croma_reverse",
        query="Use optical and SAR evidence together to identify water-covered regions.",
        pair_type="optical_sar",
        image_paths=[sar_path, opt_path],
        output_dir=tmp_path / "out_reverse",
    )
    assert res_reverse.verdict.status == VerdictStatus.SUPPORTED
    assert any(e.kind == "croma_cross_attention_fusion" for e in res_reverse.evidence)
    assert any(a.name == "croma_sensor_agreement.png" for a in res_reverse.artifacts)

