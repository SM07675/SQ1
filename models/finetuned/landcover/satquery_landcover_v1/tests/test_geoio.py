import numpy as np
import rasterio
from rasterio.transform import from_origin

from satquery.geoio import write_raster
from satquery.inference import _resolution_metres


def test_crs_transform_dimensions_and_mask_are_preserved(tmp_path):
    transform = from_origin(500000, 4200000, 0.5, 0.5)
    profile = {"driver": "GTiff", "width": 11, "height": 7, "count": 3, "dtype": "uint8", "crs": "EPSG:32632", "transform": transform}
    valid = np.ones((7, 11), bool); valid[0, 0] = False
    path = write_raster(tmp_path / "out.tif", np.ones((7, 11), "uint8"), profile, valid, 255)
    with rasterio.open(path) as result:
        assert result.crs.to_epsg() == 32632
        assert result.transform == transform
        assert (result.height, result.width, result.count) == (7, 11, 1)
        assert result.dataset_mask()[0, 0] == 0


def test_projected_resolution_is_converted_to_metres():
    transform = from_origin(500000, 4200000, 0.5, 0.5)
    assert _resolution_metres(20, 20, transform, "EPSG:32632") == 0.5
