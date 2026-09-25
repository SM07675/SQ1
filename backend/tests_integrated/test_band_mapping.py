import numpy as np
import rasterio
from rasterio.io import MemoryFile

from satquery_engine.services.bands import detect_band_map


def test_sentinel_narrow_nir_is_distinct_from_broad_nir():
    profile={"driver":"GTiff","height":4,"width":4,"count":2,"dtype":"uint16"}
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.ones((2,4,4),dtype="uint16"))
            dst.update_tags(sensor="Sentinel-2")
            dst.set_band_description(1,"B8")
            dst.set_band_description(2,"B8A")
        with mem.open() as src:
            mapping=detect_band_map(src)
    assert mapping.indices["nir"] == 1
    assert mapping.indices["narrow_nir"] == 2
