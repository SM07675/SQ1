import numpy as np

from satquery_engine.services.radiometry import rgb_unit_data
from rasterio.io import MemoryFile


def test_black_border_is_valid_data_masked_before_classification():
    data=np.full((3,16,16),80,dtype="uint8"); data[:,:,0]=0
    with MemoryFile() as mem:
        with mem.open(driver="GTiff",height=16,width=16,count=3,dtype="uint8") as dst:
            dst.write(data)
        with mem.open() as src:
            _,valid,_=rgb_unit_data(src)
    assert not valid[:,0].any()
    assert valid[:,1:].all()
