"""Register on a common grid while retaining bands, radiometry and NoData."""
import json
from dataclasses import asdict
import numpy as np
import rasterio
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from rasterio.warp import reproject, Resampling
from satquery_engine.services.registration import RegistrationReport, _compute_phase_correlation_shift, _apply_shift, _normalize_luminance, compute_alignment_quality


def _luminance(data, src):
    from satquery_engine.services.bands import detect_band_map
    roles=detect_band_map(src).indices
    indexes=[roles[k]-1 for k in ("red","green","blue") if k in roles]
    if not indexes:
        indexes=[int(np.argmax([np.nanstd(b) for b in data]))]
    return np.mean(data[indexes],axis=0)


def _similarity(a,b,valid):
    if valid.sum()<64: return 0.0
    av,bv=a[valid],b[valid]
    if min(av.std(),bv.std())<1e-6: return 0.0
    return float(np.clip(np.corrcoef(av,bv)[0,1],0,1))


def residual_translation(da,db,a,b):
    """Estimate then independently check the remaining shift on shared pixels."""
    ga,gb=_normalize_luminance(_luminance(da,a)),_normalize_luminance(_luminance(db,b))
    valid=np.isfinite(ga)&np.isfinite(gb)
    if valid.sum()<64 or min(ga[valid].std(),gb[valid].std())<1e-6:
        raise ValueError("The images lack shared texture to verify registration.")
    aa=np.where(valid,ga-float(ga[valid].mean()),0)
    bb=np.where(valid,gb-float(gb[valid].mean()),0)
    coarse_y,coarse_x,peak=_compute_phase_correlation_shift(aa,bb)
    if peak<.02:
        raise ValueError("Image alignment has no reliable correlation peak; change analysis was blocked.")
    window=np.outer(np.hanning(a.height),np.hanning(a.width))
    shift,_,_=phase_cross_correlation(aa*window,bb*window,upsample_factor=10)
    sy,sx=map(float,shift)
    if abs(sy)>a.height*.1 or abs(sx)>a.width*.1:
        raise ValueError("Required residual registration displacement exceeds the safe limit.")
    shifted_valid=ndimage.shift(np.all(np.isfinite(db),axis=0).astype("float32"),(sy,sx),order=1,mode="constant",cval=0,prefilter=False)>.999
    moved=ndimage.shift(np.nan_to_num(db),(0,sy,sx),order=1,mode="constant",cval=0,prefilter=False)
    moved[:,~shifted_valid]=np.nan
    gm=_normalize_luminance(_luminance(moved,b))
    shared=valid & np.isfinite(gm)
    original_score=_similarity(ga,gb,shared)
    score=_similarity(ga,gm,shared)
    if abs(sy)+abs(sx)>.1 and score<original_score+.005:
        if abs(sy)>1 or abs(sx)>1:
            raise ValueError("Residual alignment could not be verified; change analysis was blocked.")
        moved=db; gm=gb; sy=sx=0.; score=original_score
    residual_y,residual_x,residual_peak=_compute_phase_correlation_shift(np.where(shared,ga,0),np.where(shared,gm,0))
    residual=float(np.hypot(residual_y,residual_x))
    if residual>1 or residual_peak<.02 or score<.35:
        raise ValueError("The images are not aligned well enough for reliable change analysis.")
    return moved,sy,sx,score,{"residual_error_pixels":residual,"phase_peak":float(peak),"correlation_before":original_score,"correlation_after":score,"method_version":"phase_subpixel_checked_v1"}


def align_pair(path_a, path_b, output_dir, cross_modal=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "aligned_a.tif", output_dir / "aligned_b.tif"]
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if bool(a.crs) != bool(b.crs):
            raise ValueError("Both images need georeferencing for a map-space comparison.")
        if a.width*a.height > 32_000_000:
            raise ValueError("This pair exceeds the current 32-million-pixel registration limit.")
        da = a.read(masked=True).astype("float32").filled(np.nan)
        if a.crs:
            db = np.full((b.count, a.height, a.width), np.nan, dtype="float32")
            for i in range(b.count):
                reproject(rasterio.band(b, i+1), db[i], src_transform=b.transform, src_crs=b.crs,
                          dst_transform=a.transform, dst_crs=a.crs, src_nodata=b.nodata,
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
            method = "geospatial_reprojection"
            sy = sx = 0.0
        elif (a.width, a.height) != (b.width, b.height):
            # Auto-resample non-georeferenced pair to common reference dimensions (a.width, a.height)
            try:
                import cv2
                raw_b = b.read(masked=True).astype("float32").filled(np.nan)
                resized_bands = [
                    cv2.resize(raw_b[i], (a.width, a.height), interpolation=cv2.INTER_LINEAR)
                    for i in range(b.count)
                ]
                db = np.stack(resized_bands, axis=0)
            except Exception:
                from scipy.ndimage import zoom
                raw_b = b.read(masked=True).astype("float32").filled(np.nan)
                zoom_factors = (1.0, a.height / b.height, a.width / b.width)
                db = zoom(raw_b, zoom_factors, order=1).astype("float32")
            method = "pixel_grid_resampled"
            sy = sx = 0.0
        else:
            db = b.read(masked=True).astype("float32").filled(np.nan)
            method = "pixel_grid"
            sy = sx = 0.0
        residual_details={"residual_error_pixels":None}
        if not cross_modal:
            db,sy,sx,score,residual_details=residual_translation(da,db,a,b)
            method += "+phase_subpixel_checked"
        shared = np.all(np.isfinite(da),axis=0) & np.all(np.isfinite(db),axis=0)
        overlap = float(shared.mean())
        if overlap < 0.8:
            raise ValueError("Less than 80% of the image contains shared valid pixels after alignment.")
        # Cross-modal intensity correlation is not a registration accuracy score.
        score = 0.0 if cross_modal else score
        for src, data, path in [(a,da,paths[0]),(b,db,paths[1])]:
            data[:,~shared] = np.nan
            with rasterio.open(path,"w",driver="GTiff",height=a.height,width=a.width,count=src.count,
                               dtype="float32",crs=a.crs,transform=a.transform,nodata=np.nan,compress="deflate") as dst:
                dst.write(data)
                dst.descriptions = src.descriptions
                dst.scales = src.scales
                dst.offsets = src.offsets
                dst.update_tags(**src.tags())
                for i in range(1,src.count+1): dst.update_tags(i,**src.tags(i))
                if all(d=="uint8" for d in src.dtypes): dst.update_tags(satquery_rgb_encoding="uint8")
        report = RegistrationReport(True,method,score,"acceptable" if score >= .35 else "low_confidence",sx,sy,
                                    (a.width,a.height),(b.width,b.height),(a.width,a.height),
                                    "Map-grid compatibility; residual cross-sensor registration is unverified." if cross_modal else "Measured image similarity after alignment; not calibrated registration accuracy.")
    (output_dir / "registration_report.json").write_text(json.dumps({**asdict(report),"shared_valid_fraction":overlap,**residual_details},indent=2))
    return paths[0],paths[1],report
