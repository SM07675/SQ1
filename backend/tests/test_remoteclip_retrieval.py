from pathlib import Path
import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from app.services.ingestion import build_model_tiles
from app.services.model_registry import ModelRegistry
from app.services.remoteclip_retrieval import (
    RankedTile,
    apply_spatial_nms,
    compute_tile_semantic_score,
    generate_retrieval_mosaic,
    merge_adjacent_matching_tiles,
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
    # Pure blue tile -> High score for water query (>= 0.70)
    blue_img = Image.new("RGB", (64, 64), (10, 20, 220))
    water_score = compute_tile_semantic_score(blue_img, "locate water reservoir")
    assert water_score >= 0.70

    # Desert / urban tile -> Low score for water query (< 0.50)
    sand_img = Image.new("RGB", (64, 64), (210, 195, 160))
    non_water_score = compute_tile_semantic_score(sand_img, "locate water reservoir")
    assert non_water_score < 0.50

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
    selected = apply_spatial_nms(candidates, iou_threshold=0.40, max_keep=5)
    selected_ids = [c["tile_id"] for c in selected]
    assert 1 in selected_ids
    assert 2 not in selected_ids  # Suppressed due to high overlap with tile 1
    assert 3 in selected_ids


def test_merge_adjacent_matching_tiles():
    # Construct 3 adjacent tiles and 1 isolated tile
    tiles = [
        RankedTile(tile_id=1, file_path=Path("t1.png"), relative_url="", pixel_window=[0, 0, 256, 256], bounds=[0, 0, 256, 256], similarity_score=0.92, rank=1),
        RankedTile(tile_id=2, file_path=Path("t2.png"), relative_url="", pixel_window=[240, 0, 256, 256], bounds=[240, 0, 496, 256], similarity_score=0.91, rank=2), # touches t1
        RankedTile(tile_id=3, file_path=Path("t3.png"), relative_url="", pixel_window=[0, 240, 256, 256], bounds=[0, 240, 256, 496], similarity_score=0.88, rank=3), # touches t1
        RankedTile(tile_id=4, file_path=Path("t4.png"), relative_url="", pixel_window=[800, 800, 256, 256], bounds=[800, 800, 1056, 1056], similarity_score=0.95, rank=4), # isolated
    ]
    
    # When query is "Highlight the largest water body", area should dominate ranking
    merged_largest = merge_adjacent_matching_tiles(tiles, query="Highlight the largest water body")
    assert len(merged_largest) == 2
    # Group containing 1, 2, 3 has larger area than isolated tile 4
    assert merged_largest[0]["tile_count"] == 3
    assert set(merged_largest[0]["tile_ids"]) == {1, 2, 3}
    assert merged_largest[1]["tile_count"] == 1
    assert merged_largest[1]["tile_ids"] == [4]


def test_generate_retrieval_mosaic_abstention(tmp_path: Path):
    out_p = tmp_path / "mosaic_empty.png"
    # When no matching tiles, creates abstention placeholder
    res = generate_retrieval_mosaic([], out_p, query="locate water body")
    assert res.exists()
    assert res.stat().st_size > 0


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
        relevance_threshold=0.70,
    )

    assert res.total_candidate_tiles == len(tile_paths)
    assert res.relevant_tiles_count == len(res.ranked_tiles)
    assert res.excluded_tiles_count == len(res.excluded_tiles)
    assert res.total_candidate_tiles == res.relevant_tiles_count + res.excluded_tiles_count
    
    # All ranked tiles must meet the relevance threshold
    for tile in res.ranked_tiles:
        assert tile.similarity_score >= res.relevance_threshold
        assert tile.is_relevant is True

    # Excluded tiles must be marked as not relevant (either below threshold or suppressed by NMS)
    for tile in res.excluded_tiles:
        assert tile.is_relevant is False
        assert (tile.similarity_score < res.relevance_threshold) or ("NMS" in tile.relevance_reason)

    assert res.mosaic_path.exists()
    assert res.geojson_roi_path.exists()
    assert res.manifest_path.exists()
    assert res.has_relevant_regions is True
    assert res.confidence >= 0.60
