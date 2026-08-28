#!/usr/bin/env python3
"""RemoteCLIP Hierarchical Semantic Tile Retrieval CLI Runner.

Indexes large GeoTIFF scenes into spatial candidate tiles and ranks them
against natural language queries using RemoteCLIP text-to-tile similarity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.ingestion import build_model_tiles
from app.services.model_registry import registry
from app.services.remoteclip_retrieval import retrieve_hierarchical_tiles


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="RemoteCLIP Semantic Tile Retrieval Runner")
    parser.add_argument(
        "--image",
        type=str,
        default="data/demo_before_multispectral.tif",
        help="Path to source GeoTIFF image",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Highlight the permanent water body",
        help="Natural language text query to search for in scene tiles",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top semantic tiles to retrieve (default: 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/retrieval_eval",
        help="Output directory for mosaic, manifest, and GeoJSON",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[-] Image not found: {img_path}")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SatQuery GeoProof - RemoteCLIP Hierarchical Tile Retrieval")
    print("=" * 80)
    print(f"Source Image : {img_path}")
    print(f"Query        : '{args.query}'")
    print(f"Top-K Target : {args.top_k}")
    print(f"Output Path  : {out_dir}")
    print("-" * 80)

    print("[*] Generating candidate spatial model tiles...")
    prep_dir = out_dir / "preprocess"
    manifest_url = "/artifacts/retrieval_eval/preprocess/tiles_manifest.json"
    report, tile_paths = build_model_tiles(
        img_path,
        prep_dir,
        public_manifest_url=manifest_url,
        source_format="raster",
        tile_size=448,
        overlap=64,
        max_tiles=32,
    )
    print(f"[+] Materialized {len(tile_paths)} candidate tiles across scene.")

    print("[*] Executing RemoteCLIP text-to-tile similarity ranking & NMS...")
    manifest_p = prep_dir / "tiles_manifest.json"
    result = await retrieve_hierarchical_tiles(
        image_path=img_path,
        tile_paths=tile_paths,
        manifest_path=manifest_p,
        query=args.query,
        output_dir=out_dir,
        registry=registry,
        top_k=args.top_k,
    )

    print("\n--- TOP-RANKED SEMANTIC TILES ---")
    print(f"{'RANK':<5} | {'TILE ID':<8} | {'SIMILARITY':<11} | {'PIXEL WINDOW (X, Y, W, H)':<26} | {'FILE'}")
    print("-" * 80)
    for t in result.ranked_tiles:
        win_str = f"[{t.pixel_window[0]}, {t.pixel_window[1]}, {t.pixel_window[2]}, {t.pixel_window[3]}]"
        print(
            f"#{t.rank:<4} | "
            f"Tile-{t.tile_id:<3} | "
            f"{t.similarity_score:<11.3f} | "
            f"{win_str:<26} | "
            f"{t.file_path.name}"
        )

    print("\n" + "=" * 80)
    print(" RETRIEVAL SUMMARY")
    print("=" * 80)
    print(f"Total Tiles Evaluated : {result.total_candidate_tiles}")
    print(f"Top-K Tiles Selected  : {result.top_k}")
    print(f"Max Similarity Score  : {result.max_similarity:.3f}")
    print(f"Mean Similarity Score : {result.mean_similarity:.3f}")
    print(f"Retrieval Confidence  : {result.confidence:.3f}")
    print(f"Mosaic Generated      : {result.mosaic_path}")
    print(f"GeoJSON ROI Bounds    : {result.geojson_roi_path}")
    print("=" * 80)
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
