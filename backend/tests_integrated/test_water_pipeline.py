from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from satquery_engine.services.water_engine import execute_water_pipeline


def test_water_pipeline_emits_surface_and_illumination_artifacts(tmp_path: Path):
    image=np.full((3,96,96),190,dtype="uint8")
    image[:,20:70,8:28]=np.array([25,55,95],dtype="uint8")[:,None,None]
    image[:,30:65,55:82]=45
    source=tmp_path/"rgb.tif"
    with rasterio.open(source,"w",driver="GTiff",width=96,height=96,count=3,dtype="uint8",transform=from_origin(0,96,1,1)) as dst:
        dst.write(image); dst.colorinterp=(rasterio.enums.ColorInterp.red,rasterio.enums.ColorInterp.green,rasterio.enums.ColorInterp.blue)
    result=execute_water_pipeline(source,tmp_path/"out")
    names={p.name for p in result["paths"]}
    assert {"water_probability.tif","water_mask.tif","shadow_probability.tif","shadow_mask.tif","water_shadow_conflict.tif","uncertainty.tif","surface_type.tif","illumination_state.tif"} <= names
    with rasterio.open(tmp_path/"out"/"water_mask.tif") as water, rasterio.open(tmp_path/"out"/"shadow_mask.tif") as shadow:
        assert water.shape == shadow.shape == (96,96)
        assert not np.any((water.read_masks(1)==0) & (water.read(1)>0))
