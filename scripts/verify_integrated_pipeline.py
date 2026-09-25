"""Run a real image and check the reported surface outputs for consistency."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.integrated_analysis import analyze


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/integrated-verification")
    args = parser.parse_args()
    output = args.output.resolve()
    result = await analyze(
        result_id=output.name,
        query="Count buildings, measure water area, and map land cover",
        pair_type="auto",
        image_paths=[args.image.resolve()],
        output_dir=output,
    )
    for key in ("buildings_a", "water_measure", "land_cover"):
        if key not in result.statistics:
            raise RuntimeError(f"{key} failed: {result.verdict.limitations}")
    with rasterio.open(next((output / "surface_context").rglob("water_mask.tif"))) as src:
        water = src.read(1) > 0
    with rasterio.open(output / "buildings_a/buildings_labels.tif") as src:
        buildings = src.read(1)
    with rasterio.open(output / "land_cover/land_only_mask.tif") as src:
        land = src.read(1) > 0
    ids = np.unique(buildings[buildings > 0])
    assert not np.any(water & (buildings > 0)), "Buildings overlap estimated water"
    assert not np.any(water & land), "Land-only mask overlaps estimated water"
    assert len(ids) == result.statistics["buildings_a"]["count"], "Building count disagrees with labels"
    saved = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert saved["models_used"] == result.models_used
    assert saved["summary"]["headline"] == result.verdict.answer
    report = {
        "passed": True,
        "source": str(args.image.resolve()),
        "models_used": result.models_used,
        "building_count": len(ids),
        "water_percent": result.statistics["water_measure"].get("coverage_percent"),
        "land_percent": result.statistics["land_cover"].get("classified_land_percent"),
        "verdict": result.verdict.status.value,
        "scope": "Real inference and artifact consistency, not ground-truth accuracy.",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
