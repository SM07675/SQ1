from pathlib import Path

from satquery_engine.models.compatibility import ModelCompatibility
from satquery_engine.models.manifest import HealthStatus, ModelManifest
from satquery_engine.schemas import RasterMetadata


def _profile(modality="multispectral", bands=None):
    indices = bands or {"blue":1,"green":2,"red":3,"narrow_nir":4,"swir1":5,"swir2":6}
    return RasterMetadata(filename="scene.tif",width=32,height=32,bands=6,dtype="uint16",crs=None,
        bounds=[0,0,32,32],resolution=[10,10],nodata=None,nodata_percent=0,band_names=list(indices),
        modality=modality,band_map={"indices":indices})


def _manifest():
    return ModelManifest(model_id="prithvi",local_path=Path("model.pt"),task=("water_segmentation",),
        modality=("multispectral_s2",),expected_bands=("BLUE","GREEN","RED","NARROW_NIR","SWIR1","SWIR2"),
        expected_band_order=("BLUE","GREEN","RED","NARROW_NIR","SWIR1","SWIR2"),input_channels=6,
        expected_resolution="Sentinel-2 10/20 m",input_dtype="float32",input_range="reflectance",
        normalization={"type":"checkpoint_native"},input_size=(512,512,6),output_type="pixel_water_probability",
        availability=True,health_status=HealthStatus.READY,adapter="PrithviWaterAdapter")


def test_exact_bands_are_required_not_just_matching_tensor_shape():
    assert ModelCompatibility.evaluate(_manifest(),_profile(),"water_segmentation").compatible
    result=ModelCompatibility.evaluate(_manifest(),_profile(bands={"blue":1,"green":2,"red":3,"nir":4,"swir1":5,"swir2":6}),"water_segmentation")
    assert not result.compatible
    assert any("Narrow NIR" in reason for reason in result.reasons)


def test_unready_model_is_never_compatible():
    manifest=_manifest().with_health(HealthStatus.DEGRADED,False)
    assert not ModelCompatibility.evaluate(manifest,_profile(),"water_segmentation").compatible
