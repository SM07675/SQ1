from pathlib import Path
import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from app.services.ingestion import build_model_tiles
from app.services.model_registry import ModelRegistry
from app.services.remoteclip_retrieval import (
    apply_spatial_nms,
    compute_tile_semantic_score,
    retrieve_hierarchical_tiles,
)


def _write_rgb_raster(path: Path, h: int = 512, w: int = 512) -> None:
    data = np.zeros((3, h, w), dtype="float32")
    # Region 1: Green vegetation
    data[1, :256, :] = 0.8
    # Region 2: Blue water
    data[2, 256:, :] = 0.9

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=3,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 2200000, 10, 10),
    ) as dst:
        for i in range(3):
            dst.write(data[i], i + 1)


def test_tile_semantic_score():
    # Pure blue tile -> High score for water query
    blue_img = Image.new("RGB", (64, 64), (10, 20, 220))
    water_score = compute_tile_semantic_score(blue_img, "locate water reservoir")
    assert water_score >= 0.70

    # Pure green tile -> High score for vegetation query
    green_img = Image.new("RGB", (64, 64), (20, 200, 30))
    veg_score = compute_tile_semantic_score(green_img, "dense canopy vegetation")
    assert veg_score >= 0.70


def test_spatial_nms():
    candidates = [
        {"tile_id": 1, "pixel_window": [0, 0, 448, 448], "similarity_score": 0.92},
        {"tile_id": 2, "pixel_window": [32, 32, 448, 448], "similarity_score": 0.90},  # Heavy overlap with tile 1
        {"tile_id": 3, "pixel_window": [500, 500, 448, 448], "similarity_score": 0.85}, # Disjoint
    ]
    selected = apply_spatial_nms(candidates, iou_threshold=0.40, top_k=5)
    selected_ids = [c["tile_id"] for c in selected]
    assert 1 in selected_ids
    assert 2 not in selected_ids  # Suppressed due to high overlap with tile 1
    assert 3 in selected_ids


@pytest.mark.asyncio
async def test_retrieve_hierarchical_tiles(tmp_path: Path):
    img_path = tmp_path / "test_scene.tif"
    _write_rgb_raster(img_path, h=512, w=512)

    prep_dir = tmp_path / "prep"
    _, tile_paths = build_model_tiles(
        img_path,
        prep_dir,
        public_manifest_url="/artifacts/prep/tiles_manifest.json",
        source_format="raster",
        tile_size=256,
        overlap=32,
    )
    assert len(tile_paths) > 0

    manifest_p = prep_dir / "tiles_manifest.json"
    reg = ModelRegistry()
    res = await retrieve_hierarchical_tiles(
        image_path=img_path,
        tile_paths=tile_paths,
        manifest_path=manifest_p,
        query="water reservoir and lake",
        output_dir=tmp_path / "retrieval",
        registry=reg,
        top_k=4,
    )

    assert res.total_candidate_tiles == len(tile_paths)
    assert res.top_k <= 4
    assert len(res.ranked_tiles) > 0
    assert res.mosaic_path.exists()
    assert res.geojson_roi_path.exists()
    assert res.manifest_path.exists()
    assert res.confidence >= 0.60
