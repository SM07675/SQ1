from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.change_detector import compute_learned_change_probability, run_change_detection_witness
from app.services.model_registry import ModelRegistry


def _write_raster(path: Path, data: np.ndarray) -> None:
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


def test_learned_change_probability():
    before = np.zeros((64, 64), dtype=np.float32)
    after = before.copy()
    after[15:45, 15:45] = 1.0  # Clear square change

    prob = compute_learned_change_probability(before, after)
    assert prob.shape == (64, 64)
    assert np.mean(prob[15:45, 15:45]) > np.mean(prob[:10, :10])
    assert np.max(prob) >= 0.5


@pytest.mark.asyncio
async def test_run_change_detection_witness(tmp_path: Path):
    a_path = tmp_path / "before.tif"
    b_path = tmp_path / "after.tif"

    before = np.zeros((64, 64), dtype=np.float32)
    after = before.copy()
    after[20:40, 20:40] = 0.8
    _write_raster(a_path, before)
    _write_raster(b_path, after)

    reg = ModelRegistry()
    res = await run_change_detection_witness(
        before_path=a_path,
        after_path=b_path,
        output_dir=tmp_path,
        registry=reg,
    )

    assert res.changed_percent > 0
    assert res.changed_pixels > 0
    assert res.mask_path.exists()
    assert res.heatmap_path.exists()
    assert res.geojson_path.exists()
    assert res.confidence >= 0.7
    assert res.f1_proxy_score >= 0.8
