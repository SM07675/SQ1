from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from satquery_engine.services.water_engine import execute_water_pipeline


def test_water_and_shadow_overlays_share_preview_dimensions(tmp_path: Path):
    source=tmp_path/"image.tif"; data=np.full((3,40,60),128,dtype="uint8")
    with rasterio.open(source,"w",driver="GTiff",height=40,width=60,count=3,dtype="uint8",transform=from_origin(0,40,1,1)) as dst: dst.write(data)
    execute_water_pipeline(source,tmp_path/"out")
    assert Image.open(tmp_path/"out"/"water_overlay.png").size == Image.open(tmp_path/"out"/"shadow_overlay.png").size
