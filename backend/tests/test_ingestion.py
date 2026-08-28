from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin

from app.services.ingestion import build_model_tiles, prepare_source


def _write(path: Path, size: int = 900) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3,
        dtype="float32", crs="EPSG:4326", transform=from_origin(70, 20, 0.001, 0.001)
    ) as dst:
        for band, name in enumerate(("red", "green", "blue"), start=1):
            dst.write(np.full((size, size), band / 10, dtype="float32"), band)
            dst.set_band_description(band, name)


def test_overlapping_model_tiles_include_spatial_manifest(tmp_path: Path) -> None:
    source = tmp_path / "large.tif"
    _write(source)
    report, paths = build_model_tiles(
        source, tmp_path / "tiles", public_manifest_url="/manifest.json",
        source_format="geospatial_raster", tile_size=448, overlap=64, max_tiles=4
    )
    assert report.total_candidate_tiles > 4
    assert report.materialized_tiles == 4
    assert report.complete_coverage is False
    assert all(path.exists() for path in paths)
    assert (tmp_path / "tiles" / "tiles_manifest.json").exists()


def test_netcdf_is_materialized_through_gdal(tmp_path: Path) -> None:
    tif = tmp_path / "source.tif"
    nc = tmp_path / "source.nc"
    _write(tif, 64)
    raster_copy(tif, nc, driver="netCDF")
    prepared = prepare_source(nc, tmp_path / "prepared")
    assert prepared.source_format == "netcdf"
    assert prepared.raster_path.exists()
    with rasterio.open(prepared.raster_path) as src:
        assert src.width == 64 and src.height == 64
