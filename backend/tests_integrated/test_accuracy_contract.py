"""Counterexamples for scientific, routing and spatial invariants; no model training."""
import asyncio
import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from satquery_engine.services.bands import detect_band_map
from satquery_engine.services.spectral import available_indices,read_index
from satquery_engine.services.radiometry import rgb_unit_data
from satquery_engine.services.sar import SARPreprocessor
from satquery_engine.services.raster import inspect_raster
from satquery_engine.services.measurements import measure_cover
from satquery_engine.services.alignment import align_pair
from satquery_engine.services.spatial_outputs import export_labels
from satquery_engine.services.intents import plan_query,validate_dag
from satquery_engine.schemas import PlanNode


def write(path,data,names=None,**tags):
    with rasterio.open(path,"w",driver="GTiff",width=data.shape[2],height=data.shape[1],count=data.shape[0],dtype=data.dtype,
                       crs="EPSG:32643",transform=from_origin(500000,2200000,.5,.5),nodata=np.nan if data.dtype.kind=="f" else None) as dst:
        dst.write(data)
        if names: dst.descriptions=names
        dst.update_tags(**tags)
    return path


def test_sensor_specific_numbered_bands(tmp_path):
    p=write(tmp_path/"landsat.tif",np.ones((2,12,12),dtype="float32"),("B4","B5"),sensor="Landsat-8")
    assert available_indices(p)==["ndvi"]
    with rasterio.open(p,"r+") as dst: dst.update_tags(sensor="Sentinel-2")
    assert available_indices(p)==[]
    with rasterio.open(p,"r+") as dst: dst.update_tags(sensor="unknown")
    assert available_indices(p)==[]


def test_wavelength_mapping_and_conflicts(tmp_path):
    p=write(tmp_path/"wave.tif",np.ones((2,12,12),dtype="float32"))
    with rasterio.open(p,"r+") as dst:
        dst.update_tags(1,wavelength="0.665",wavelength_units="um")
        dst.update_tags(2,wavelength="842",wavelength_units="nm")
    assert available_indices(p)==["ndvi"]
    with rasterio.open(p,"r+") as dst: dst.set_band_description(2,"red")
    assert available_indices(p)==[]


def test_duplicate_nir_abstains(tmp_path):
    p=write(tmp_path/"duplicate.tif",np.ones((3,12,12),dtype="float32"),("red","nir","nir"))
    assert not available_indices(p)


def test_native_measurement_keeps_single_pixel_river(tmp_path):
    data=np.full((2,20,2048),.5,dtype="float32")
    data[0,:,1001]=.8; data[1,:,1001]=.1
    p=write(tmp_path/"river.tif",data,("green","nir"))
    result=measure_cover(p,tmp_path/"result","water",strict=True)
    assert result["selected_pixels"]==20 and result["sampled_pixels"]==40960
    assert result["mask_polygon_agree"] and result["native_resolution"]


def test_rgb_scaling_does_not_change_between_dark_and_bright_tiles(tmp_path):
    data=np.ones((3,16,32),dtype="uint8"); data[:,:,16:]=200
    p=write(tmp_path/"rgb.tif",data,("red","green","blue"))
    with rasterio.open(p) as src:
        dark,_,_=rgb_unit_data(src,window=rasterio.windows.Window(0,0,16,16))
        light,_,_=rgb_unit_data(src,window=rasterio.windows.Window(16,0,16,16))
    np.testing.assert_allclose(dark,1/255)
    np.testing.assert_allclose(light,200/255)


@pytest.mark.parametrize("units,value,expected",[("amplitude",.1,-20),("intensity",.1,-10),("dB",-12,-12)])
def test_sar_units(units,value,expected,tmp_path):
    p=write(tmp_path/"sar.tif",np.full((2,12,12),value,dtype="float32"),("vv","vh"),units=units)
    result=SARPreprocessor().process(p)
    np.testing.assert_allclose(result.db,expected,atol=1e-5)


def test_sar_unknown_units_abstains(tmp_path):
    p=write(tmp_path/"sar.tif",np.ones((2,12,12),dtype="float32"),("vv","vh"))
    with pytest.raises(ValueError,match="unknown"): SARPreprocessor().process(p)


def test_metadata_quality_checks_all_bands(tmp_path):
    data=np.ones((3,16,16),dtype="float32"); data[2,:,:8]=np.nan
    p=write(tmp_path/"mask.tif",data,("red","green","blue"))
    metadata=inspect_raster(p)
    assert metadata.nodata_percent==50 and len(metadata.file_hash)==64


def test_one_instance_exports_one_feature_even_with_disjoint_parts(tmp_path):
    p=write(tmp_path/"source.tif",np.ones((3,16,16),dtype="uint8"),("red","green","blue"))
    labels=np.zeros((16,16),dtype="int32"); labels[1:4,1:4]=1; labels[10:13,10:13]=1
    result=export_labels(labels,p,tmp_path/"out","buildings")
    assert len(result["features"])==1 and result["selected_pixels"]==18
    assert result["features"][0]["geometry"]["type"]=="MultiPolygon"


def test_georeferenced_residual_shift_is_corrected(tmp_path):
    rng=np.random.default_rng(25)
    data=rng.uniform(.1,.9,(3,128,128)).astype("float32")
    shifted=np.full_like(data,np.nan); shifted[:,3:,2:]=data[:,:-3,:-2]
    a=write(tmp_path/"a.tif",data,("red","green","blue"))
    b=write(tmp_path/"b.tif",shifted,("red","green","blue"))
    aa,bb,report=align_pair(a,b,tmp_path/"aligned")
    assert abs(report.shift_y+3)<.2 and abs(report.shift_x+2)<.2
    with rasterio.open(aa) as x,rasterio.open(bb) as y:
        np.testing.assert_allclose(x.read(),y.read(),atol=.05,equal_nan=True)
    assert json.loads((tmp_path/"aligned/registration_report.json").read_text())["residual_error_pixels"]<=1


def test_flat_pair_cannot_establish_registration(tmp_path):
    data=np.ones((3,32,32),dtype="float32")
    a=write(tmp_path/"a.tif",data); b=write(tmp_path/"b.tif",data)
    with pytest.raises(ValueError,match="texture"): align_pair(a,b,tmp_path/"out")


@pytest.mark.parametrize("query",["Analyze this image completely.","Give me a comprehensive analysis","Show everything in this scene"])
def test_comprehensive_query_has_independent_specialists(query):
    plan=plan_query(query)
    assert {"buildings","spectral","vlm"}<=set(plan.tools)
    assert not plan.requires_pair
    validate_dag(plan)


def test_policy_rejects_arbitrary_model_parameters():
    plan=plan_query("Count houses")
    plan.nodes[1].parameters["checkpoint"]="untrusted.onnx"
    with pytest.raises(ValueError,match="parameters"): validate_dag(plan)


def test_policy_rejects_missing_registration_dependency():
    plan=plan_query("What changed?",2)
    next(n for n in plan.nodes if n.tool=="change").depends_on=["validate"]
    with pytest.raises(ValueError,match="registration"): validate_dag(plan)
