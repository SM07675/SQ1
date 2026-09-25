"""Request-scoped water evidence shared by surface specialists."""
from pathlib import Path

import numpy as np
import rasterio


class SurfaceContext:
    def __init__(self, output: Path):
        self.output = output
        self.results = {}

    def water(self, source, *, largest=False, strict=False):
        from satquery_engine.services.measurements import measure_cover
        from satquery_engine.services.quality_gate import validate_result_evidence

        key = (str(Path(source).resolve()), largest, strict)
        if key not in self.results:
            folder = self.output / f"water_{len(self.results)}"
            result = measure_cover(source, folder, "water", largest=largest, strict=strict)
            validate_result_evidence(result, folder)
            self.results[key] = result
        return self.results[key]


def exclude_water_instances(labels, valid, source, water_result, minimum_pixels):
    """Reject water-dominated instances; trim shoreline overlap without new IDs."""
    if water_result.get("quality_gate_passed") is False:
        raise ValueError("Water evidence failed validation; building count withheld.")
    paths = [Path(p) for p in water_result.get("paths", []) if Path(p).name == "water_mask.tif"]
    if len(paths) != 1:
        raise ValueError("A unique canonical water mask is required for building exclusion.")
    with rasterio.open(source) as src, rasterio.open(paths[0]) as mask:
        if ((mask.height, mask.width) != labels.shape or mask.shape != src.shape
                or mask.transform != src.transform or mask.crs != src.crs):
            raise ValueError("Water and building evidence must use the same source grid.")
        water = mask.read(1) > 0
        if np.any(water & ~(mask.read_masks(1) > 0)) or np.any(water & ~valid):
            raise ValueError("Water exclusion evidence contains invalid pixels.")
    sizes = np.bincount(labels.ravel())
    overlaps = np.bincount(labels[water], minlength=len(sizes))
    reject = (overlaps >= sizes * .5) | ((sizes - overlaps) < minimum_pixels)
    reject[0] = False
    filtered = np.where(water | reject[labels] | ~valid, 0, labels)
    ids = np.unique(filtered)
    ids = ids[ids > 0]
    mapping = np.zeros(len(sizes), dtype="int32")
    mapping[ids] = np.arange(1, len(ids) + 1)
    return mapping[filtered], {
        "status": "applied", "water_model": water_result.get("model_id", water_result.get("method")),
        "water_overlap_pixels": int(overlaps[1:].sum()),
        "rejected_instances": int(np.count_nonzero(reject[1:] & (sizes[1:] > 0))),
        "removed_building_pixels": int(np.count_nonzero(labels) - np.count_nonzero(filtered)),
        "policy": "Reject instances with at least 50% water overlap; trim other water pixels and remove undersized remnants.",
    }
