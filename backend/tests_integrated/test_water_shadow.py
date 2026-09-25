import numpy as np

from satquery_engine.analysis.shadow import ShadowType, rgb_shadow_evidence
from satquery_engine.analysis.surface import IlluminationState, SurfaceType, adjudicate_surface_and_illumination


def test_building_shadow_is_not_forced_to_water():
    r=np.full((64,64),.65,dtype="float32"); g=r.copy(); b=r.copy()
    r[24:44,24:48]=.07; g[24:44,24:48]=.07; b[24:44,24:48]=.07
    buildings=np.zeros((64,64),bool); buildings[18:24,24:48]=True
    water=np.full((64,64),.08,dtype="float32")
    shadow=rgb_shadow_evidence(r,g,b,np.ones((64,64),bool),water_probability=water,building_mask=buildings)
    result=adjudicate_surface_and_illumination(water_probability=water,shadow_probability=shadow.probability,valid=np.ones((64,64),bool),builtup_probability=buildings.astype("float32"))
    assert not result.water_mask[30,30]
    assert result.illumination[30,30] == IlluminationState.SHADOWED
    assert shadow.shadow_type[30,30] == ShadowType.BUILDING_SHADOW


def test_water_under_shadow_keeps_water_surface():
    water=np.zeros((32,32),dtype="float32"); water[8:24,4:28]=.91
    shadow=np.zeros_like(water); shadow[8:24,14:20]=.78
    result=adjudicate_surface_and_illumination(water_probability=water,shadow_probability=shadow,valid=np.ones_like(water,bool))
    assert result.surface[12,16] == SurfaceType.WATER
    assert result.illumination[12,16] == IlluminationState.SHADOWED
    assert result.water_mask[12,16] and result.shadow_mask[12,16]


def test_nodata_is_neither_water_nor_shadow():
    valid=np.ones((8,8),bool); valid[:,0]=False
    result=adjudicate_surface_and_illumination(water_probability=np.ones((8,8),dtype="float32"),shadow_probability=np.ones((8,8),dtype="float32"),valid=valid)
    assert not result.water_mask[:,0].any()
    assert not result.shadow_mask[:,0].any()
    assert np.all(result.surface[:,0] == SurfaceType.NODATA)
