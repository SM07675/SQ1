from satquery_engine.config import settings
from satquery_engine.models.compatibility import ModelCompatibility
from satquery_engine.models.registry import LocalModelRegistry
from satquery_engine.schemas import RasterMetadata


def test_ready_building_specialist_accepts_documented_rgb_only():
    manifest=LocalModelRegistry(settings.model_dir).get("building_primary")
    assert manifest is not None and manifest.availability
    rgb=RasterMetadata(filename="rgb.tif",width=256,height=256,bands=3,dtype="uint8",crs=None,
        bounds=[0,0,256,256],resolution=[.3,.3],nodata=None,nodata_percent=0,band_names=["band_1","band_2","band_3"],modality="optical")
    result=ModelCompatibility.evaluate(manifest,rgb,"building_detection")
    assert result.compatible and result.selected_channels == (1,2,3)


def test_building_specialist_rejects_sar():
    manifest=LocalModelRegistry(settings.model_dir).get("building_primary")
    sar=RasterMetadata(filename="sar.tif",width=256,height=256,bands=2,dtype="float32",crs=None,
        bounds=[0,0,256,256],resolution=[10,10],nodata=None,nodata_percent=0,band_names=["VH","VV"],modality="sar",
        band_map={"indices":{"vh":1,"vv":2}})
    assert manifest is not None and not ModelCompatibility.evaluate(manifest,sar,"building_detection").compatible
