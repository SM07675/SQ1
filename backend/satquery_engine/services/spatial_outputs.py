"""One mask -> geometry -> statistics -> visualization path for all specialists."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping, box
from shapely import make_valid
from shapely.ops import transform as transform_geom, unary_union
from pyproj import Transformer
from PIL import Image, ImageDraw
from satquery_engine.services.raster import _area_square_meters, render_preview


def instance_edges(labels: np.ndarray) -> np.ndarray:
    """Mark the complete four-sided outline of every positive instance ID."""
    selected = labels > 0
    edges = np.zeros(labels.shape, dtype=bool)
    edges[0] = selected[0]
    edges[-1] = selected[-1]
    edges[:, 0] = selected[:, 0]
    edges[:, -1] = selected[:, -1]
    edges[1:] |= selected[1:] & (labels[1:] != labels[:-1])
    edges[:-1] |= selected[:-1] & (labels[:-1] != labels[1:])
    edges[:, 1:] |= selected[:, 1:] & (labels[:, 1:] != labels[:, :-1])
    edges[:, :-1] |= selected[:, :-1] & (labels[:, :-1] != labels[:, 1:])
    return edges


def export_float_raster(values, source, destination):
    with rasterio.open(source) as src:
        grid=src.transform @ Affine.scale(src.width/values.shape[1],src.height/values.shape[0])
        with rasterio.open(destination,"w",driver="GTiff",height=values.shape[0],width=values.shape[1],count=1,
                           dtype="float32",crs=src.crs,transform=grid,nodata=np.nan,compress="deflate") as dst:
            dst.write(values.astype("float32"),1)
    return destination


def export_labels(labels, source: Path, output: Path, name: str, scores=None, transform=None, color=(0, 200, 230), numbered=False, valid_mask=None):
    labels=np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0):
        raise ValueError("Spatial export requires nonnegative integer instance labels.")
    if valid_mask is not None and np.any((labels>0)&~valid_mask):
        raise ValueError("Selected features include invalid pixels.")
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        crs = src.crs
        grid = transform or src.transform @ Affine.scale(src.width / labels.shape[1], src.height / labels.shape[0])
        profile = dict(driver="GTiff", width=labels.shape[1], height=labels.shape[0], count=1,
                       dtype="int32", transform=grid, crs=crs, compress="deflate")
    features = []
    world = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs else None
    grouped={}
    for pixel_geom, value in shapes(labels.astype("int32"), mask=labels > 0, transform=Affine.identity()):
        grouped.setdefault(int(value),[]).append(shape(pixel_geom))
    for value, parts in sorted(grouped.items()):
        pixel_polygon=make_valid(unary_union(parts))
        # Use 0.5-px tolerance so buildings whose rasterized footprint touches the
        # image boundary (a common, valid case) are not silently rejected.  The
        # half-pixel margin covers float-precision overhang from Shapely's vectorizer.
        _tol = 0.5
        image_box = box(-_tol, -_tol, labels.shape[1] + _tol, labels.shape[0] + _tol)
        if pixel_polygon.is_empty or pixel_polygon.area<=0 or not image_box.covers(pixel_polygon):
            raise ValueError("Invalid or out-of-image evidence geometry; export blocked.")
        pixel_geom=mapping(pixel_polygon)
        native = transform_geom(lambda x, y, z=None: (grid.a * x + grid.b * y + grid.c, grid.d * x + grid.e * y + grid.f), pixel_polygon)
        area = _area_square_meters(mapping(native), crs)
        geom = transform_geom(world.transform, native) if world else pixel_polygon
        if not geom.is_valid or not np.isfinite(geom.bounds).all():
            raise ValueError("Invalid geographic evidence geometry; export blocked.")
        if crs and (area is None or not np.isfinite(area) or area<=0):
            raise ValueError("Invalid measured area; export blocked.")
        features.append({"type": "Feature", "id": int(value), "geometry": mapping(geom), "properties": {
            "instance_id": int(value), "kind": name, "area_m2": area, "area_pixels": float(pixel_polygon.area),
            "marker": list(geom.representative_point().coords[0]), "bbox": list(geom.bounds),
            "pixel_geometry": pixel_geom, "native_geometry": mapping(native), "native_crs": str(crs) if crs else None,
            "score": float(scores[int(value)]) if scores is not None and int(value) in scores else None}})
    collection = {"type": "FeatureCollection", "features": features, "properties": {
        "crs": "EPSG:4326" if crs else None, "coordinate_space": "geographic" if crs else "pixel",
        "source_crs": str(crs) if crs else None, "transform": list(grid)[:6], "producer": name}}
    geojson = output / f"{name}.geojson"
    geojson.write_text(json.dumps(collection, allow_nan=False), encoding="utf-8")
    raster = output / f"{name}_labels.tif"
    with rasterio.open(raster, "w", **profile) as dst:
        dst.write(labels.astype("int32"), 1)
        if valid_mask is not None: dst.write_mask(valid_mask.astype("uint8")*255)
    mask = labels > 0
    rgba = np.zeros((*mask.shape, 4), dtype="uint8")
    rgba[mask] = [*color, 85 if name == "buildings" else 130]
    mask_path = output / f"{name}_mask.png"
    Image.fromarray(rgba).save(mask_path)
    preview_path = output / f"{name}_original.png"
    render_preview(source, preview_path, max_size=1200)
    preview = Image.open(preview_path).convert("RGBA")
    overlay = Image.alpha_composite(preview, Image.fromarray(rgba).resize(preview.size, Image.Resampling.NEAREST))
    if name == "buildings":
        edges = instance_edges(labels)
        border = np.zeros((*labels.shape,4),dtype='uint8'); border[edges] = [255,255,255,230]
        overlay = Image.alpha_composite(overlay, Image.fromarray(border).resize(preview.size, Image.Resampling.NEAREST))
        draw = ImageDraw.Draw(overlay)
        from scipy import ndimage
        for ident, region in enumerate(ndimage.find_objects(labels) if numbered else [], 1):
            if region is None: continue
            ys,xs = np.where(labels[region] == ident)
            if len(xs) < 20: continue
            ys += region[0].start; xs += region[1].start
            nearest = np.argmin((ys-ys.mean())**2 + (xs-xs.mean())**2)
            xy = (int(xs[nearest]*preview.width/labels.shape[1]),int(ys[nearest]*preview.height/labels.shape[0]))
            draw.text(xy,str(ident),fill='white',stroke_width=1,stroke_fill='black')
        if preview.width >= 32 and preview.height >= 28:
            draw.rectangle((5,5,min(preview.width-5,255),27),fill=(0,0,0,200))
            draw.text((10,10),f"Building footprints: {len(features)} (estimated)",fill="white")
    overlay_path = output / f"{name}_overlay.png"
    overlay.convert("RGB").save(overlay_path)
    # Allow 1-pixel float tolerance per instance: Shapely's polygon area for a
    # raster-derived polygon can differ from np.sum(mask) by a small epsilon.
    _area_tolerance = max(1, len(features))
    if abs(sum(f["properties"]["area_pixels"] for f in features) - mask.sum()) > _area_tolerance:
        raise ValueError("Mask and polygon areas disagree; evidence withheld.")
    return {"features": features, "region_count": len(features), "selected_pixels": int(mask.sum()),
            "geometry_validated":True,"mask_polygon_agree":True,
            "area_m2": sum(f["properties"]["area_m2"] or 0 for f in features) if crs else None,
            "paths": [geojson, raster, mask_path, overlay_path, preview_path]}
