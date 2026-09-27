"""Released Deepness/LandCover.ai model adapter. No weight conversion or training."""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from satquery_engine.services.ingestion import _starts
from satquery_engine.services.radiometry import rgb_unit_data
from satquery_engine.config import settings

MODEL_ID = "PUTvision/Deepness-LandCoverAI-DeepLabV3plus"
SHA256 = "3c82806dfe6a582ba26642427ef5533595981ec2bb6d4d3a3c4a3f06ac4753c4"
CHECKPOINT = settings.model_dir / "landcover/deeplabv3_landcover_4c.onnx"
CLASSES = ("unknown", "building", "woodland", "water", "road")


@lru_cache(maxsize=1)
def load_model():
    import onnxruntime as ort
    if not CHECKPOINT.is_file():
        raise ValueError("The pretrained aerial land-cover specialist is unavailable.")
    with CHECKPOINT.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
            raise ValueError("Land-cover checkpoint identity could not be verified.")
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    from satquery_engine.models.device import onnx_providers
    model = ort.InferenceSession(str(CHECKPOINT), sess_options=options, providers=onnx_providers())
    if model.get_inputs()[0].shape != [1,3,512,512] or model.get_outputs()[0].shape != [1,5,512,512]:
        raise ValueError("Unexpected land-cover model schema.")
    return model


def compatibility(path):
    from satquery_engine.services.buildings import resolution_m
    with rasterio.open(path) as src:
        gsd = resolution_m(src)
        try:
            from satquery_engine.services.radiometry import rgb_indexes
            rgb_indexes(src)
        except ValueError as exc:
            return False, str(exc)
    if gsd is None:
        return False, "Aerial land-cover inference requires known geographic resolution; image colors alone do not establish model compatibility."
    if not .15 <= gsd <= .6:
        return False, "Aerial specialist requires verified 0.15-0.60 metre RGB pixels; domain compatibility is unverified."
    return True, "Resolution-compatible aerial RGB; geography and appearance remain uncalibrated."


def predict_landcover(path, *, benchmark_domain_override=False):
    """Return native-grid class probabilities, validity and model provenance.

    The explicit override is for labeled research chips stripped of georeferencing;
    it is never set from a query, request parameter, or production caller.
    """
    resolved = str(Path(path).resolve())
    return _predict_landcover_cached(resolved, benchmark_domain_override=benchmark_domain_override)


@lru_cache(maxsize=8)
def _predict_landcover_cached(resolved_path: str, *, benchmark_domain_override: bool = False):
    path = Path(resolved_path)
    compatible, note = compatibility(path)
    if not compatible and not benchmark_domain_override:
        raise ValueError(note)
    model = load_model()
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        if h*w > 32_000_000:
            raise ValueError("Aerial specialist supports up to 32 million native pixels.")
        scene_rgb, scene_valid, _ = rgb_unit_data(src)
        del scene_rgb
        accum = np.zeros((5,h,w), dtype="float32")
        weight = np.zeros((h,w), dtype="float32")
        valid = np.zeros((h,w), dtype=bool)
        axis = np.exp(-.5*((np.arange(512)-255.5)/96)**2)
        kernel = np.maximum(np.outer(axis,axis),1e-6).astype("float32")
        tiles = []
        for y in _starts(h,512,384):
            for x in _starts(w,512,384):
                th,tw = min(512,h-y),min(512,w-x)
                rgb,good,radiometry = rgb_unit_data(src,window=Window(x,y,tw,th),scene_valid=scene_valid)
                tile = np.pad(rgb,((0,0),(0,512-th),(0,512-tw)),mode="edge")
                scores = model.run(None,{model.get_inputs()[0].name:tile[None]})[0][0]
                if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1 or not np.allclose(scores.sum(0),1,atol=1e-4):
                    raise ValueError("Invalid pretrained class probabilities; evidence withheld.")
                k = kernel[:th,:tw]
                accum[:,y:y+th,x:x+tw] += scores[:,:th,:tw]*k
                weight[y:y+th,x:x+tw] += k
                valid[y:y+th,x:x+tw] |= good
                tiles.append({"tile_id":len(tiles),"x_offset":x,"y_offset":y,"width":tw,"height":th,
                              "padding":[512-th,512-tw],"valid_pixels":int(good.sum())})
        if not valid.any() or np.any(weight <= 0):
            raise ValueError("Aerial inference lacks complete valid coverage.")
        return {"probability":accum/weight,"valid":valid,"model_id":MODEL_ID,"checkpoint_sha256":SHA256,
                "classes":CLASSES,"tiles":tiles,"preprocessing":radiometry,"domain_note":note,
                "device":model.get_providers()[0]}
