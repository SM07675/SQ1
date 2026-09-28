"""Fail closed before a specialist output becomes a finding or downloadable evidence."""
import math
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import shape


def validate_result_evidence(result,output):
    for path in result.get("paths",[]):
        if not path.resolve().is_relative_to(output.resolve()) or not path.is_file() or not path.stat().st_size:
            raise ValueError("A required evidence artifact is missing or outside its analysis directory.")
    score=result.get("evidence_strength",0)
    if not math.isfinite(score) or not 0<=score<=1: raise ValueError("Invalid evidence strength; finding withheld.")
    features=result.get("features")
    if features is not None:
        ids=[f.get("id") for f in features]
        if len(ids)!=len(set(ids)): raise ValueError("Duplicate evidence feature IDs; finding withheld.")
        for key in ("count", "building_count"):
            if key in result and result[key]!=len(features): raise ValueError("Count does not match final features.")
        for feature in features:
            geom=shape(feature["geometry"])
            if not geom.is_valid or geom.is_empty or geom.area<=0: raise ValueError("Invalid evidence geometry.")
        for path in result.get("paths",[]):
            if path.suffix==".geojson" and path.stem not in {"water_gain","water_loss","vegetation_gain","vegetation_loss","built_up_gain","built_up_loss"} and json.loads(path.read_text())["features"]!=json.loads(json.dumps(features)):
                raise ValueError("Exported GeoJSON differs from canonical features.")
    for key in ("area_m2","selected_pixels","count","region_count"):
        value=result.get(key)
        if value is not None and (not math.isfinite(value) or value<0): raise ValueError("Invalid spatial measurement.")
    if result.get("quality_gate_passed") is False and not result.get("count_reliability"):
        raise ValueError("Specialist quality validation failed.")
    finals = [p for p in result.get("paths",[]) if p.name in {"water_mask.tif","buildings_labels.tif","land_cover_map.tif","change_labels.tif"}]
    for path in finals:
        with rasterio.open(path) as src:
            labels = src.read(1); selected = labels > 0; valid = src.read_masks(1) > 0
            if np.any(selected & ~valid): raise ValueError("Final mask includes NoData pixels.")
            if "valid_pixels" in result and result["valid_pixels"] != int(valid.sum()):
                raise ValueError("Final mask and valid-pixel denominator disagree.")
            if result.get("selected_pixels") != int(selected.sum()): raise ValueError("Final mask and pixel count disagree.")
            if result.get("area_m2") is not None and src.crs is None: raise ValueError("Area is unavailable without a CRS.")
            if features is not None:
                polygon_pixels = sum(f["properties"].get("area_pixels",0) for f in features)
                if not math.isclose(polygon_pixels,float(selected.sum()),abs_tol=1e-5):
                    raise ValueError("Final mask and polygon areas disagree.")
                # Equal totals cannot detect shifted footprints or swapped instance IDs.
                expected_labels = ndimage.label(selected)[0] if path.name == "water_mask.tif" else labels
                identifiers, counts = np.unique(expected_labels[selected], return_counts=True)
                pixels_by_id = dict(zip(identifiers.tolist(), counts.tolist()))
                label_ids = set(pixels_by_id)
                if label_ids != {f["id"] for f in features}:
                    raise ValueError("Final mask and feature IDs disagree.")
                geometries = []
                for feature in features:
                    pixel_geometry = feature["properties"].get("pixel_geometry")
                    if pixel_geometry is None:
                        raise ValueError("Pixel geometry is required to verify evidence alignment.")
                    pixel_shape = shape(pixel_geometry)
                    pixel_count = pixels_by_id[feature["id"]]
                    if (not pixel_shape.is_valid or pixel_shape.is_empty
                            or not math.isclose(pixel_shape.area, pixel_count, abs_tol=1e-5)
                            or not math.isclose(feature["properties"].get("area_pixels", -1), pixel_count, abs_tol=1e-5)):
                        raise ValueError("Final mask and individual feature areas disagree.")
                    geometries.append((pixel_geometry, feature["id"]))
                reconstructed = (rasterize(geometries, out_shape=labels.shape, transform=Affine.identity(), dtype="int32")
                                 if geometries else np.zeros_like(labels))
                if not np.array_equal(reconstructed, expected_labels):
                    raise ValueError("Final mask and polygon pixel alignment disagree.")
            if path.name == "land_cover_map.tif":
                classes = json.loads(src.tags().get("classes", "{}"))
                breakdown = result.get("breakdown", {})
                if set(classes.values()) != set(breakdown) or np.any(valid & ~selected):
                    raise ValueError("Land-cover classes do not cover the valid grid.")
                for ident, name in classes.items():
                    pixels = int(np.count_nonzero(labels == int(ident)))
                    part = breakdown[name]
                    if (part.get("pixels") != pixels or not valid.any()
                            or not math.isclose(part.get("percent", -1), 100*pixels/int(valid.sum()), abs_tol=.001)):
                        raise ValueError("Land-cover class measurements disagree with the final mask.")
            if "coverage_percent" in result:
                denominator = result.get("valid_pixels", int(valid.sum()))
                if not denominator or not math.isclose(result["coverage_percent"],100*selected.sum()/denominator,abs_tol=.001):
                    raise ValueError("Final mask and coverage percentage disagree.")
    if "breakdown" in result:
        parts = result["breakdown"].values()
        if not math.isclose(sum(p["percent"] for p in parts),100,abs_tol=.001):
            raise ValueError("Land-cover class percentages do not sum to 100%.")
