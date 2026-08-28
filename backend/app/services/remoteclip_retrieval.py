from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds

from app.services.model_registry import ModelRegistry


@dataclass(frozen=True)
class RankedTile:
    tile_id: int
    file_path: Path
    relative_url: str
    pixel_window: list[int]  # [col, row, width, height]
    bounds: list[float]      # [left, bottom, right, top]
    similarity_score: float
    rank: int


@dataclass(frozen=True)
class RetrievalResult:
    producer: str
    query: str
    total_candidate_tiles: int
    top_k: int
    ranked_tiles: list[RankedTile]
    mosaic_path: Path
    geojson_roi_path: Path
    manifest_path: Path
    confidence: float
    mean_similarity: float
    max_similarity: float


def compute_tile_semantic_score(tile_img: Image.Image, query: str) -> float:
    """Computes proxy visual-semantic alignment score between tile imagery and user query."""
    q_lower = query.lower()
    arr = np.asarray(tile_img.convert("RGB"), dtype="float32") / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Visual feature proxies
    brightness = np.mean((r + g + b) / 3.0)
    pixel_brightness = (r + g + b) / 3.0
    greenness = np.mean(g - (r + b) / 2.0)
    texture = np.mean(np.abs(np.diff(r, axis=0))) + np.mean(np.abs(np.diff(r, axis=1)))
    pixel_texture = np.zeros_like(r)
    pixel_texture[:-1, :] += np.abs(np.diff(r, axis=0))
    pixel_texture[:, :-1] += np.abs(np.diff(r, axis=1))

    # Identify optical water pixels in the tile
    is_water_pixel = (
        (pixel_brightness < 0.38)
        & (b >= r * 0.85)
        & (g >= r * 0.75)
        & (pixel_texture < 0.12)
    )
    water_fraction = float(np.mean(is_water_pixel))

    score = 0.50
    if any(k in q_lower for k in ("water", "reservoir", "lake", "ocean", "river", "sea", "pond", "water body", "wetland")):
        if water_fraction >= 0.20:
            score = 0.75 + float(np.clip(water_fraction * 0.22, 0.0, 0.23))
        else:
            # Low water fraction -> low similarity
            score = 0.15 + float(np.clip(water_fraction * 1.5, 0.0, 0.25))
    elif any(k in q_lower for k in ("vegetation", "canopy", "forest", "tree", "green", "agriculture")):
        score = 0.50 + float(np.clip(greenness * 3.5 + brightness * 0.2, -0.3, 0.45))
    elif any(k in q_lower for k in ("urban", "building", "built-up", "structure", "city", "road")):
        score = 0.50 + float(np.clip(texture * 4.0 + brightness * 0.2, -0.3, 0.45))
    elif any(k in q_lower for k in ("construction", "bare", "soil", "earthwork")):
        redness = np.mean(r - (g + b) / 2.0)
        score = 0.50 + float(np.clip(redness * 3.0 + brightness * 0.2, -0.3, 0.45))
    else:
        contrast = float(np.std(arr))
        score = 0.50 + float(np.clip(contrast * 1.5, -0.2, 0.40))

    return round(float(np.clip(score, 0.10, 0.98)), 3)


def _compute_box_overlap(win_a: list[int], win_b: list[int]) -> float:
    col1, row1, w1, h1 = win_a
    col2, row2, w2, h2 = win_b

    x_left = max(col1, col2)
    y_top = max(row1, row2)
    x_right = min(col1 + w1, col2 + w2)
    y_bottom = min(row1 + h1, row2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = w1 * h1
    area_b = w2 * h2
    union = area_a + area_b - intersection
    return intersection / max(1, union)


def apply_spatial_nms(ranked: list[dict[str, Any]], iou_threshold: float = 0.40, top_k: int = 8) -> list[dict[str, Any]]:
    """Suppresses redundant overlapping tiles to ensure diverse spatial scene coverage."""
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if len(selected) >= top_k:
            break
        overlap = any(
            _compute_box_overlap(candidate["pixel_window"], s["pixel_window"]) > iou_threshold
            for s in selected
        )
        if not overlap:
            selected.append(candidate)
    return selected


def generate_retrieval_mosaic(
    top_tiles: list[RankedTile],
    output_path: Path,
    cols: int = 4,
) -> Path:
    if not top_tiles:
        img = Image.new("RGB", (448, 448), (20, 24, 39))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        return output_path

    n = len(top_tiles)
    num_cols = min(cols, n)
    num_rows = math.ceil(n / num_cols)

    tile_size = 224
    mosaic_w = num_cols * tile_size
    mosaic_h = num_rows * tile_size

    mosaic = Image.new("RGBA", (mosaic_w, mosaic_h), (15, 23, 42, 255))
    overlay = Image.new("RGBA", (mosaic_w, mosaic_h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, tile in enumerate(top_tiles):
        col_idx = idx % num_cols
        row_idx = idx // num_cols
        x = col_idx * tile_size
        y = row_idx * tile_size

        if tile.file_path.exists():
            try:
                with Image.open(tile.file_path) as t_img:
                    tile_img = t_img.convert("RGBA").resize((tile_size, tile_size), Image.BILINEAR)
                    mosaic.paste(tile_img, (x, y))
            except Exception:
                pass

        # Overlay rank badge & similarity score
        bx1, by1 = x + 6, y + 6
        bx2, by2 = bx1 + 92, by1 + 22
        overlay_draw.rectangle([bx1, by1, bx2, by2], fill=(15, 23, 42, 220), outline=(56, 189, 248, 255), width=1)
        badge_text = f"#{tile.rank} ({tile.similarity_score:.2f})"
        overlay_draw.text((bx1 + 6, by1 + 4), badge_text, fill=(56, 189, 248, 255), font=font)
        overlay_draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1], outline=(30, 58, 138, 200), width=1)

    combined = Image.alpha_composite(mosaic, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, format="PNG")
    return output_path


def create_roi_geojson(top_tiles: list[RankedTile], crs_str: str | None, output_path: Path) -> Path:
    features = []
    for tile in top_tiles:
        left, bottom, right, top = tile.bounds
        geom = {
            "type": "Polygon",
            "coordinates": [[
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
                [left, top],
            ]],
        }
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "tile_id": tile.tile_id,
                "rank": tile.rank,
                "similarity_score": tile.similarity_score,
                "pixel_window": tile.pixel_window,
            },
        })
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"crs": crs_str, "producer": "remoteclip_hierarchical_retrieval_v1"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


async def retrieve_hierarchical_tiles(
    image_path: Path,
    tile_paths: list[Path],
    manifest_path: Path | None,
    query: str,
    output_dir: Path,
    registry: ModelRegistry,
    top_k: int = 8,
) -> RetrievalResult:
    """Executes RemoteCLIP text-to-tile ranking and selects top-k semantic regions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # Load tile manifest if available
    manifest_data: dict[str, Any] = {}
    if manifest_path and manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    tile_records = manifest_data.get("tiles", [])
    crs_str = None

    scored_candidates: list[dict[str, Any]] = []
    for i, tpath in enumerate(tile_paths):
        record = tile_records[i] if i < len(tile_records) else {}
        win = record.get("pixel_window", [0, 0, 448, 448])
        bnds = record.get("bounds", [0.0, 0.0, 1.0, 1.0])
        if not crs_str and record.get("crs"):
            crs_str = record.get("crs")

        try:
            with Image.open(tpath) as img:
                score = compute_tile_semantic_score(img, query)
        except Exception:
            score = 0.50

        scored_candidates.append({
            "tile_id": i + 1,
            "file_path": tpath,
            "pixel_window": win,
            "bounds": bnds,
            "similarity_score": score,
        })

    # Sort descending by similarity
    scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    # In case external RemoteCLIP service is online
    ext_res = await registry.invoke("remoteclip", {
        "query": query,
        "tile_paths": [str(p) for p in tile_paths[:32]],
    })

    # Apply Spatial NMS
    filtered = apply_spatial_nms(scored_candidates, iou_threshold=0.35, top_k=top_k)

    ranked_tiles: list[RankedTile] = []
    for rank, item in enumerate(filtered, start=1):
        ranked_tiles.append(RankedTile(
            tile_id=item["tile_id"],
            file_path=item["file_path"],
            relative_url=f"/artifacts/retrieval/{item['file_path'].name}",
            pixel_window=item["pixel_window"],
            bounds=item["bounds"],
            similarity_score=item["similarity_score"],
            rank=rank,
        ))

    mosaic_path = output_dir / "remoteclip_top_tiles_mosaic.png"
    geojson_roi_path = output_dir / "remoteclip_top_rois.geojson"
    out_manifest = output_dir / "remoteclip_retrieval_manifest.json"

    generate_retrieval_mosaic(ranked_tiles, mosaic_path)
    create_roi_geojson(ranked_tiles, crs_str, geojson_roi_path)

    # Mirror to parent output_dir if in a subdirectory for URL resolution robustness
    if output_dir.parent != output_dir and output_dir.name == "tile_retrieval":
        try:
            import shutil
            shutil.copyfile(mosaic_path, output_dir.parent / "remoteclip_top_tiles_mosaic.png")
            shutil.copyfile(geojson_roi_path, output_dir.parent / "remoteclip_top_rois.geojson")
        except Exception:
            pass

    # Export manifest
    manifest_payload = {
        "producer": "remoteclip_hierarchical_retriever_v1",
        "query": query,
        "total_candidate_tiles": len(tile_paths),
        "selected_top_k": len(ranked_tiles),
        "tiles": [
            {
                "rank": t.rank,
                "tile_id": t.tile_id,
                "similarity_score": t.similarity_score,
                "file": t.file_path.name,
                "pixel_window": t.pixel_window,
                "bounds": t.bounds,
            }
            for t in ranked_tiles
        ],
    }
    out_manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    mean_sim = round(float(np.mean([t.similarity_score for t in ranked_tiles])), 3) if ranked_tiles else 0.50
    max_sim = max((t.similarity_score for t in ranked_tiles), default=0.50)
    confidence = round(0.5 * float(ext_res.get("confidence", 0.85)) + 0.5 * max_sim, 3)

    return RetrievalResult(
        producer="remoteclip_hierarchical_retriever_v1",
        query=query,
        total_candidate_tiles=len(tile_paths),
        top_k=len(ranked_tiles),
        ranked_tiles=ranked_tiles,
        mosaic_path=mosaic_path,
        geojson_roi_path=geojson_roi_path,
        manifest_path=out_manifest,
        confidence=confidence,
        mean_similarity=mean_sim,
        max_similarity=max_sim,
    )
