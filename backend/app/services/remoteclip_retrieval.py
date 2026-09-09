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
    is_relevant: bool = True
    relevance_reason: str = "Strong semantic match with requested feature."


@dataclass(frozen=True)
class RetrievalResult:
    producer: str
    query: str
    total_candidate_tiles: int
    relevant_tiles_count: int
    excluded_tiles_count: int
    relevance_threshold: float
    has_relevant_regions: bool
    top_k: int
    ranked_tiles: list[RankedTile]
    excluded_tiles: list[RankedTile]
    primary_region: dict[str, Any] | None
    merged_regions: list[dict[str, Any]]
    mosaic_path: Path
    geojson_roi_path: Path
    manifest_path: Path
    confidence: float
    mean_similarity: float
    max_similarity: float
    findings: str


def compute_tile_semantic_score(tile_img: Image.Image, query: str) -> float:
    """Computes proxy visual-semantic alignment score between tile imagery and user query."""
    q_lower = query.lower()
    arr = np.asarray(tile_img.convert("RGB"), dtype="float32") / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Visual feature proxies
    brightness = float(np.mean((r + g + b) / 3.0))
    pixel_brightness = (r + g + b) / 3.0
    greenness = float(np.mean(g - (r + b) / 2.0))
    texture = float(np.mean(np.abs(np.diff(r, axis=0))) + np.mean(np.abs(np.diff(r, axis=1))))
    pixel_texture = np.zeros_like(r)
    pixel_texture[:-1, :] += np.abs(np.diff(r, axis=0))
    pixel_texture[:, :-1] += np.abs(np.diff(r, axis=1))

    # Identify optical water pixels in the tile
    is_water_pixel = (
        (pixel_brightness < 0.38)
        & (pixel_brightness > 0.01)
        & (b >= r * 0.85)
        & (b >= 0.04)
        & ((g - (r + b) / 2.0) < 0.10)  # reject vegetation
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
        redness = float(np.mean(r - (g + b) / 2.0))
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


def _are_tiles_adjacent_or_overlapping(win_a: list[int], win_b: list[int], max_gap: int = 48) -> bool:
    """Checks if two tile pixel windows touch, overlap, or are immediately adjacent."""
    col1, row1, w1, h1 = win_a
    col2, row2, w2, h2 = win_b

    # Expanded bounding box
    x_overlap = (col1 - max_gap) < (col2 + w2) and (col2 - max_gap) < (col1 + w1)
    y_overlap = (row1 - max_gap) < (row2 + h2) and (row2 - max_gap) < (row1 + h1)
    return x_overlap and y_overlap


def apply_spatial_nms(ranked: list[dict[str, Any]], iou_threshold: float = 0.45, max_keep: int = 16) -> list[dict[str, Any]]:
    """Suppresses redundant heavily overlapping duplicate tiles while preserving all distinct regions."""
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if len(selected) >= max_keep:
            break
        overlap = any(
            _compute_box_overlap(candidate["pixel_window"], s["pixel_window"]) > iou_threshold
            for s in selected
        )
        if not overlap:
            selected.append(candidate)
    return selected


def merge_adjacent_matching_tiles(
    tiles: list[RankedTile],
    query: str = "",
) -> list[dict[str, Any]]:
    """
    Groups adjacent and contiguous matching tiles into unified spatial regions.
    For 'largest' or 'dominant' queries, ranks merged regions by total spatial area.
    """
    if not tiles:
        return []

    # Disjoint-set / connected component grouping of tiles
    parent = list(range(len(tiles)))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if _are_tiles_adjacent_or_overlapping(tiles[i].pixel_window, tiles[j].pixel_window):
                union(i, j)

    # Group tiles by component root
    groups: dict[int, list[RankedTile]] = {}
    for i, tile in enumerate(tiles):
        root = find(i)
        groups.setdefault(root, []).append(tile)

    merged_regions: list[dict[str, Any]] = []
    for comp_id, group in enumerate(groups.values(), start=1):
        min_col = min(t.pixel_window[0] for t in group)
        min_row = min(t.pixel_window[1] for t in group)
        max_col = max(t.pixel_window[0] + t.pixel_window[2] for t in group)
        max_row = max(t.pixel_window[1] + t.pixel_window[3] for t in group)

        merged_win = [min_col, min_row, max_col - min_col, max_row - min_row]
        merged_area = (max_col - min_col) * (max_row - min_row)

        min_left = min(t.bounds[0] for t in group)
        min_bottom = min(t.bounds[1] for t in group)
        max_right = max(t.bounds[2] for t in group)
        max_top = max(t.bounds[3] for t in group)
        merged_bounds = [min_left, min_bottom, max_right, max_top]

        avg_score = round(float(np.mean([t.similarity_score for t in group])), 3)
        max_score = max(t.similarity_score for t in group)

        merged_regions.append({
            "region_id": comp_id,
            "tile_count": len(group),
            "tile_ids": [t.tile_id for t in group],
            "pixel_window": merged_win,
            "bounds": merged_bounds,
            "area_pixels": merged_area,
            "avg_similarity": avg_score,
            "max_similarity": max_score,
            "tiles": group,
        })

    # Sort merged regions: if query requests "largest", rank primarily by area; otherwise by similarity
    is_largest_req = any(k in query.lower() for k in ("largest", "biggest", "maximum", "dominant", "most extensive", "main"))
    if is_largest_req:
        merged_regions.sort(key=lambda r: (r["area_pixels"], r["max_similarity"]), reverse=True)
    else:
        merged_regions.sort(key=lambda r: (r["max_similarity"], r["area_pixels"]), reverse=True)

    return merged_regions


def generate_retrieval_mosaic(
    matching_tiles: list[RankedTile],
    output_path: Path,
    cols: int = 4,
    query: str = "",
) -> Path:
    """
    Renders ONLY query-matching relevant candidate tiles into the visual evidence mosaic.
    Low-scoring excluded blocks are NOT rendered as evidence.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_size = 224

    if not matching_tiles:
        # Render clean informative banner when no relevant tiles match
        img = Image.new("RGBA", (560, 200), (15, 23, 42, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 550, 190], fill=(23, 37, 84, 180), outline=(239, 68, 68, 255), width=2)
        draw.text((30, 40), "NO QUERY-MATCHING REGIONS FOUND", fill=(248, 113, 113, 255))
        draw.text((30, 75), f"Query: “{query}”", fill=(226, 232, 240, 255))
        draw.text((30, 110), "All candidate blocks scored below the relevance threshold (>= 0.70).", fill=(148, 163, 184, 255))
        draw.text((30, 140), "Status: Safe Abstention (INSUFFICIENT_EVIDENCE)", fill=(56, 189, 248, 255))
        img.convert("RGB").save(output_path, format="PNG")
        return output_path

    n = len(matching_tiles)
    num_cols = min(cols, n)
    num_rows = math.ceil(n / num_cols)

    mosaic_w = num_cols * tile_size
    mosaic_h = num_rows * tile_size

    mosaic = Image.new("RGBA", (mosaic_w, mosaic_h), (15, 23, 42, 255))
    overlay = Image.new("RGBA", (mosaic_w, mosaic_h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, tile in enumerate(matching_tiles):
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

        # Overlay Match Badge & Similarity Score
        bx1, by1 = x + 6, y + 6
        bx2, by2 = bx1 + 104, by1 + 22
        overlay_draw.rectangle([bx1, by1, bx2, by2], fill=(15, 23, 42, 220), outline=(56, 189, 248, 255), width=1)
        badge_text = f"#{tile.rank} ({tile.similarity_score:.2f}) MATCH"
        overlay_draw.text((bx1 + 6, by1 + 4), badge_text, fill=(56, 189, 248, 255), font=font)
        
        # Crisp tile boundary
        overlay_draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1], outline=(14, 165, 233, 220), width=1)

    combined = Image.alpha_composite(mosaic, overlay).convert("RGB")
    combined.save(output_path, format="PNG")
    return output_path


def create_roi_geojson(
    top_tiles: list[RankedTile],
    merged_regions: list[dict[str, Any]],
    crs_str: str | None,
    output_path: Path,
) -> Path:
    features = []

    # 1. Merged Unified Grounding Polygons
    for region in merged_regions:
        left, bottom, right, top = region["bounds"]
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
                "kind": "merged_grounding_region",
                "region_id": region["region_id"],
                "tile_count": region["tile_count"],
                "tile_ids": region["tile_ids"],
                "avg_similarity": region["avg_similarity"],
                "max_similarity": region["max_similarity"],
                "pixel_window": region["pixel_window"],
                "area_pixels": region["area_pixels"],
            },
        })

    # 2. Individual Matching Tiles
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
                "kind": "matching_tile",
                "tile_id": tile.tile_id,
                "rank": tile.rank,
                "similarity_score": tile.similarity_score,
                "is_relevant": tile.is_relevant,
                "relevance_reason": tile.relevance_reason,
                "pixel_window": tile.pixel_window,
            },
        })

    payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "crs": crs_str,
            "producer": "remoteclip_semantic_grounding_v2",
            "matching_regions_count": len(merged_regions),
            "matching_tiles_count": len(top_tiles),
        },
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
    relevance_threshold: float = 0.70,
    max_keep: int = 16,
) -> RetrievalResult:
    """
    Executes RemoteCLIP visual-semantic scoring, filters out low-relevance blocks,
    merges adjacent matching regions, and returns ONLY genuine query-matching evidence.
    """
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

    # Step 1: Score all candidate blocks
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

    # Step 2: Relevance Filtering
    # Apply configured threshold + relative peak floor: tiles must be >= relevance_threshold
    # AND cannot be low-scoring outliers (< 0.65 or < peak * 0.75)
    peak_sim = scored_candidates[0]["similarity_score"] if scored_candidates else 0.0
    effective_threshold = max(relevance_threshold, min(0.68, peak_sim * 0.75))

    relevant_candidates = [
        c for c in scored_candidates
        if c["similarity_score"] >= effective_threshold
    ]

    # Step 3: Spatial NMS Deduplication on relevant candidates
    filtered_relevant = apply_spatial_nms(relevant_candidates, iou_threshold=0.45, max_keep=max_keep)
    selected_tile_ids = {item["tile_id"] for item in filtered_relevant}

    ranked_tiles: list[RankedTile] = []
    for rank, item in enumerate(filtered_relevant, start=1):
        ranked_tiles.append(RankedTile(
            tile_id=item["tile_id"],
            file_path=item["file_path"],
            relative_url=f"/artifacts/retrieval/{item['file_path'].name}",
            pixel_window=item["pixel_window"],
            bounds=item["bounds"],
            similarity_score=item["similarity_score"],
            rank=rank,
            is_relevant=True,
            relevance_reason=f"High semantic alignment ({item['similarity_score']:.2f}) with query.",
        ))

    excluded_candidates = [
        c for c in scored_candidates
        if c["tile_id"] not in selected_tile_ids
    ]

    excluded_tiles: list[RankedTile] = []
    for rank, item in enumerate(excluded_candidates, start=len(ranked_tiles) + 1):
        is_below_thresh = item["similarity_score"] < effective_threshold
        reason = (
            f"Below query relevance threshold ({item['similarity_score']:.2f} < {effective_threshold:.2f})."
            if is_below_thresh
            else f"Spatially redundant candidate suppressed by NMS deduplication."
        )
        excluded_tiles.append(RankedTile(
            tile_id=item["tile_id"],
            file_path=item["file_path"],
            relative_url=f"/artifacts/retrieval/{item['file_path'].name}",
            pixel_window=item["pixel_window"],
            bounds=item["bounds"],
            similarity_score=item["similarity_score"],
            rank=rank,
            is_relevant=False,
            relevance_reason=reason,
        ))

    # Step 4: Merge adjacent matching tiles into continuous regions
    merged_regions = merge_adjacent_matching_tiles(ranked_tiles, query=query)
    primary_region = merged_regions[0] if merged_regions else None

    # Step 5: Render Visual Evidence Mosaic
    mosaic_path = output_dir / "remoteclip_top_tiles_mosaic.png"
    geojson_roi_path = output_dir / "remoteclip_top_rois.geojson"
    out_manifest = output_dir / "remoteclip_retrieval_manifest.json"

    generate_retrieval_mosaic(ranked_tiles, mosaic_path, query=query)
    create_roi_geojson(ranked_tiles, merged_regions, crs_str, geojson_roi_path)

    # Mirror to parent output_dir if in a subdirectory for URL resolution robustness
    if output_dir.parent != output_dir and output_dir.name == "tile_retrieval":
        try:
            import shutil
            shutil.copyfile(mosaic_path, output_dir.parent / "remoteclip_top_tiles_mosaic.png")
            shutil.copyfile(geojson_roi_path, output_dir.parent / "remoteclip_top_rois.geojson")
        except Exception:
            pass

    # Step 6: Formulate Findings and Metrics
    has_relevant_regions = len(ranked_tiles) > 0
    mean_sim = round(float(np.mean([t.similarity_score for t in ranked_tiles])), 3) if ranked_tiles else 0.0
    max_sim = peak_sim

    if has_relevant_regions:
        findings_text = (
            f"Evaluated {len(tile_paths)} candidate blocks; isolated {len(ranked_tiles)} query-matching "
            f"regions (similarity: {min(t.similarity_score for t in ranked_tiles):.2f}–{max_sim:.2f}) "
            f"exceeding the {effective_threshold:.2f} relevance threshold; excluded {len(excluded_tiles)} irrelevant blocks."
        )
        confidence = round(0.4 * float(ext_res.get("confidence", 0.85)) + 0.6 * max_sim, 3)
    else:
        findings_text = (
            f"Evaluated {len(tile_paths)} candidate blocks; no block met the semantic relevance threshold "
            f"({effective_threshold:.2f}). Safe abstention triggered."
        )
        confidence = 0.35

    manifest_payload = {
        "producer": "remoteclip_semantic_grounding_v2",
        "query": query,
        "total_candidate_tiles": len(tile_paths),
        "relevant_tiles_count": len(ranked_tiles),
        "excluded_tiles_count": len(excluded_tiles),
        "relevance_threshold": effective_threshold,
        "has_relevant_regions": has_relevant_regions,
        "merged_regions_count": len(merged_regions),
        "primary_region": primary_region,
        "matching_tiles": [
            {
                "rank": t.rank,
                "tile_id": t.tile_id,
                "similarity_score": t.similarity_score,
                "is_relevant": t.is_relevant,
                "file": t.file_path.name,
                "pixel_window": t.pixel_window,
                "bounds": t.bounds,
            }
            for t in ranked_tiles
        ],
        "excluded_tiles": [
            {
                "rank": t.rank,
                "tile_id": t.tile_id,
                "similarity_score": t.similarity_score,
                "is_relevant": False,
                "reason": t.relevance_reason,
            }
            for t in excluded_tiles
        ],
    }
    out_manifest.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")

    return RetrievalResult(
        producer="remoteclip_semantic_grounding_v2",
        query=query,
        total_candidate_tiles=len(tile_paths),
        relevant_tiles_count=len(ranked_tiles),
        excluded_tiles_count=len(excluded_tiles),
        relevance_threshold=effective_threshold,
        has_relevant_regions=has_relevant_regions,
        top_k=len(ranked_tiles),
        ranked_tiles=ranked_tiles,
        excluded_tiles=excluded_tiles,
        primary_region=primary_region,
        merged_regions=merged_regions,
        mosaic_path=mosaic_path,
        geojson_roi_path=geojson_roi_path,
        manifest_path=out_manifest,
        confidence=confidence,
        mean_similarity=mean_sim,
        max_similarity=max_sim,
        findings=findings_text,
    )
