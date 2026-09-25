"""Scientific counterexamples and production-result invariants (no training)."""
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin, Affine
from PIL import Image
from satquery_engine.services.water_engine import execute_water_pipeline, compute_rgb_water_proxy, run_tiled_rgb_water, postprocess_water_mask, check_water_result
from satquery_engine.services.radiometry import rgb_unit_data
from satquery_engine.services.raster import inspect_raster, _area_square_meters
from satquery_engine.services.intents import plan_query, validate_dag
from satquery_engine.services.land_cover import compute_bigearthnet_witness, classify_land_cover_composite
from satquery_engine.services.quality_gate import validate_result_evidence


def raster(path, values, names, **tags):
    with rasterio.open(path,"w",driver="GTiff",width=values.shape[2],height=values.shape[1],count=len(values),
                       dtype=values.dtype,transform=from_origin(500000,4000000,.5,.5),crs="EPSG:32632") as dst:
        dst.write(values);dst.descriptions=names;dst.update_tags(**tags)
    return path


@pytest.mark.parametrize("query,tools",[
    ("How many buildings are there?",{"buildings"}),
    ("Show me all the buildings.",{"buildings"}),
    ("Count and mark all buildings.",{"buildings"}),
    ("Find the river.",{"spectral"}),
    ("How much water is there?",{"spectral"}),
    ("Find all water bodies.",{"spectral"}),
    ("What type of land is visible?",{"land_cover"}),
    ("What land types are visible?",{"land_cover"}),
    ("Find land",{"land_cover"}),
    ("How much area is built-up?",{"spectral"}),
    ("Show vegetation.",{"spectral"}),
    ("Analyze this image completely.",{"buildings","spectral","land_cover","vlm"}),
])
def test_user_acceptance_routing(query,tools):
    plan = plan_query(query); validate_dag(plan)
    assert tools <= set(plan.tools)
    if "land_cover" in tools and "completely" not in query:
        assert "vlm" not in plan.tools


def test_sar_negative_db_and_linear_equivalence(tmp_path):
    db=np.full((1,50,50),-8,dtype="float32");db[:,10:40,10:40]=-22
    masks=[]
    for units,values in (("dB",db),("intensity",10**(db/10)),("amplitude",10**(db/20))):
        path=raster(tmp_path/f"{units}.tif",values,("vv",),units=units)
        result=execute_water_pipeline(path,tmp_path/units)
        assert result["selected_pixels"] > 500 and result["valid_pixels"]==2500
        with rasterio.open(tmp_path/units/"water_mask.tif") as src:masks.append(src.read(1))
    assert all(np.array_equal(masks[0],m) for m in masks[1:])


def test_sar_unknown_units_fail_closed(tmp_path):
    path=raster(tmp_path/"unknown.tif",np.full((1,30,30),-20,dtype="float32"),("vv",))
    with pytest.raises(ValueError,match="units/representation"):
        execute_water_pipeline(path,tmp_path/"out")


def test_water_tiles_preserve_source_colors_and_neighborhoods(tmp_path):
    rng=np.random.default_rng(7)
    values=rng.integers(15,180,size=(3,190,230),dtype="uint8")
    values[:,30:160,60:130]=np.array([20,60,95])[:,None,None]
    path=raster(tmp_path/"tiles.tif",values,("red","green","blue"))
    with rasterio.open(path) as src:
        rgb,valid,_=rgb_unit_data(src)
        expected,seeds,_,_=compute_rgb_water_proxy(*rgb,valid)
    actual,actual_seeds,actual_valid=run_tiled_rgb_water(path,(190,230),tile_size=80,overlap=24)
    np.testing.assert_allclose(actual,expected,atol=1e-5)
    np.testing.assert_array_equal(actual_seeds,seeds)
    np.testing.assert_array_equal(actual_valid,valid)


def test_water_closing_never_fills_nodata_or_deletes_border():
    prob=np.full((50,50),.95,dtype="float32");valid=np.ones((50,50),bool);valid[24,24]=False
    mask,components,_=postprocess_water_mask(prob,valid,min_component_px=0)
    assert not mask[24,24] and mask[0].all() and mask[:,0].all()
    assert check_water_result(mask,prob,valid,"RGB_WATER_PROXY",components)[0]
    mask[24,24]=True
    assert not check_water_result(mask,prob,valid,"RGB_WATER_PROXY",components)[0]


def test_largest_water_uses_final_component_ids(tmp_path):
    values=np.full((2,80,80),.3,dtype="float32")
    values[0,1:4,1:4]=.7;values[1,1:4,1:4]=.01
    values[0,30:65,30:65]=.7;values[1,30:65,30:65]=.01
    path=raster(tmp_path/"largest.tif",values,("green","nir"))
    result=execute_water_pipeline(path,tmp_path/"out",largest=True)
    assert result["selected_pixels"] == 35*35 and result["region_count"]==1


def test_geographic_area_crosses_dateline_and_keeps_hole():
    geo={"type":"Polygon","coordinates":[[(179.99,0),(180.01,0),(180.01,.01),(179.99,.01),(179.99,0)]]}
    area=_area_square_meters(geo,rasterio.crs.CRS.from_epsg(4326))
    assert 2_000_000 < area < 3_000_000
    assert _area_square_meters(geo,None) is None


def test_named_three_band_spectral_is_not_rgb(tmp_path):
    path=raster(tmp_path/"not_rgb.tif",np.ones((3,20,20),dtype="float32"),("red","nir","swir"))
    assert inspect_raster(path).modality == "multispectral"


def test_flair_style_five_band_vhr_uses_fixed_rgb_contract(tmp_path):
    path=tmp_path/"flair.tif"
    values=np.zeros((5,24,32),dtype="uint8")
    values[:3]=np.array([80,110,70],dtype="uint8")[:,None,None]
    values[3]=150
    values[4]=12
    with rasterio.open(path,"w",driver="GTiff",width=32,height=24,count=5,dtype="uint8",
                       transform=from_origin(500000,4000000,.2,.2),crs="EPSG:2154") as dst:
        dst.write(values)
    with rasterio.open(path) as src:
        rgb,valid,audit=rgb_unit_data(src)
    assert rgb.shape == (3,24,32) and valid.all()
    assert audit["indexes"] == [1,2,3]
    assert audit["selection"] == "FLAIR-style VHR RGB/NIR/DSM contract"


def test_bigearthnet_rules_cannot_impersonate_model():
    result=compute_bigearthnet_witness({"red":np.ones((10,10))},np.ones((10,10),bool))
    assert not result.available and result.class_probabilities=={} and not result.model_name


def test_final_gate_rejects_changed_count_and_invalid_pixels(tmp_path):
    path=raster(tmp_path/"lake.tif",np.stack([np.full((30,30),.5),np.full((30,30),.01)]).astype("float32"),("green","nir"))
    result=execute_water_pipeline(path,tmp_path/"out")
    validate_result_evidence(result,tmp_path/"out")
    result["selected_pixels"]+=1
    with pytest.raises(ValueError,match="pixel count"):
        validate_result_evidence(result,tmp_path/"out")


def test_land_cover_mask_percentages_and_nodata_share_grid(tmp_path,monkeypatch):
    import satquery_engine.services.landcover_specialist as learned
    monkeypatch.setattr(learned,"predict_landcover",lambda *a,**k:(_ for _ in ()).throw(ValueError("Test isolates non-model fallback")))
    values=np.full((3,70,90),120,dtype="uint8")
    values[:,:30,:40]=np.array([20,60,95])[:,None,None]
    path=raster(tmp_path/"land.tif",values,("red","green","blue"))
    with rasterio.open(path,"r+") as dst:
        good=np.ones((70,90),dtype="uint8")*255;good[-4:]=0;dst.write_mask(good)
    water=execute_water_pipeline(path,tmp_path/"water",use_model=False)
    result=classify_land_cover_composite(path,tmp_path/"land",water_result=water)
    validate_result_evidence(result,tmp_path/"land")
    assert sum(v["pixels"] for v in result["breakdown"].values()) == 66*90
    assert sum(v["percent"] for v in result["breakdown"].values()) == pytest.approx(100)
    assert result["water_percent"] == pytest.approx(water["coverage_percent"],abs=.001)
    with rasterio.open(tmp_path/"land/land_cover_map.tif") as src:
        assert src.shape == (70,90) and not src.read(1)[-4:].any()
    geo=json.loads((tmp_path/"land/land_cover.geojson").read_text())
    assert all("class" in f["properties"] for f in geo["features"])


def test_land_cover_compositor_uses_spatial_specialist(tmp_path,monkeypatch):
    import satquery_engine.services.landcover_specialist as learned
    height,width=40,60
    path=raster(tmp_path/"rgb.tif",np.full((3,height,width),120,dtype="uint8"),("red","green","blue"))
    empty=np.zeros((1,height,width),dtype="uint8")
    water_path=raster(tmp_path/"water_mask.tif",empty,("water",))
    vegetation_path=raster(tmp_path/"vegetation_labels.tif",empty,("vegetation",))
    probability=np.full((5,height,width),.01,dtype="float32")
    probability[2,:,:30]=.92
    probability[1,:20,30:]=.90
    probability[4,20:,30:]=.88
    probability[0]=1-probability[1:].sum(axis=0)
    monkeypatch.setattr(learned,"predict_landcover",lambda *a,**k:{
        "probability":probability,
        "valid":np.ones((height,width),dtype=bool),
        "classes":("unknown","building","woodland","water","road"),
        "model_id":"test-spatial-specialist",
        "checkpoint_sha256":"0"*64,
        "preprocessing":{"test":True},
        "domain_note":"Test specialist output.",
    })
    result=classify_land_cover_composite(
        path,tmp_path/"land",
        water_result={"paths":[water_path],"models_used":[],"evidence_strength":0.0},
        vegetation_result={"paths":[vegetation_path],"evidence_strength":0.0},
    )
    assert result["breakdown"]["woodland"]["percent"] == pytest.approx(50)
    assert result["breakdown"]["built_up"]["percent"] == pytest.approx(25)
    assert result["breakdown"]["road"]["percent"] == pytest.approx(25)
    assert result["breakdown"]["unknown"]["percent"] == 0
    assert "test-spatial-specialist" in result["models_used"]
    assert (tmp_path/"land/landcover_probability.tif").exists()
