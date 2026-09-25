"""One explicit RGB scientific-to-model contract, shared by every tile."""
import numpy as np
from scipy import ndimage
from rasterio.enums import Resampling
from satquery_engine.services.bands import detect_band_map


def rgb_indexes(src):
    mapping = detect_band_map(src)
    if all(k in mapping.indices for k in ("red","green","blue")):
        return [mapping.indices[k] for k in ("red","green","blue")]
    if src.count == 3 and not mapping.warnings and not any(src.descriptions):
        return [1,2,3]  # documented ordinary RGB upload convention
    # FLAIR-style VHR chips have a fixed five-band layout (RGB, NIR, DSM),
    # uint8 storage, and 20 cm georeferencing but commonly omit descriptions.
    # Keep this deliberately narrow so arbitrary multispectral rasters are not
    # silently routed into RGB models.
    if (
        src.count == 5
        and all(dtype == "uint8" for dtype in src.dtypes)
        and not any(src.descriptions)
        and not mapping.warnings
        and 0.15 <= max(abs(src.res[0]), abs(src.res[1])) <= 0.25
    ):
        return [1,2,3]
    raise ValueError("RGB analysis requires unambiguous red, green and blue bands.")


def rgb_unit_data(src, *, window=None, out_shape=None, scene_valid=None):
    indexes = rgb_indexes(src)
    shape = (3,*out_shape) if out_shape else None
    masked = src.read(indexes, window=window, out_shape=shape, masked=True, resampling=Resampling.nearest).astype("float32")
    values = masked.filled(np.nan)
    good = np.all(np.isfinite(values) & ~np.ma.getmaskarray(masked),axis=0)
    scales = np.asarray([src.scales[i-1] for i in indexes],dtype="float32")[:,None,None]
    offsets = np.asarray([src.offsets[i-1] for i in indexes],dtype="float32")[:,None,None]
    values = values*scales+offsets
    byte = all(src.dtypes[i-1]=="uint8" for i in indexes) and np.all(scales==1) and np.all(offsets==0)
    # Registered rasters retain the explicit original encoding in metadata.
    encoding = src.tags().get("satquery_rgb_encoding")
    if byte or encoding == "uint8":
        values /= 255.0
        method = "uint8 / 255"
    else:
        method = "reflectance using metadata scales and offsets"
    if good.any() and (np.min(values[:,good])<0 or np.max(values[:,good])>1.00001):
        raise ValueError("RGB radiometry is unsupported: use 8-bit RGB or metadata-scaled reflectance in 0-1. No per-tile scaling was guessed.")
    # Exact black, border-connected padding is NoData evidence, not water or
    # shadow. Interior dark surfaces and nonzero dark water remain valid.
    exact_black = np.all(values == 0, axis=0) & good
    if window is not None:
        # Tile edges are not image edges: an interior black roof or pond must
        # retain the same validity in every overlapping inference window.
        if scene_valid is None:
            _, scene_valid, _ = rgb_unit_data(src)
        if scene_valid.shape != (src.height, src.width):
            raise ValueError("RGB scene validity must match the native source grid.")
        rows = np.floor(window.row_off + (np.arange(good.shape[0]) + .5) * window.height / good.shape[0]).astype(int)
        cols = np.floor(window.col_off + (np.arange(good.shape[1]) + .5) * window.width / good.shape[1]).astype(int)
        good &= scene_valid[np.ix_(rows, cols)]
    elif exact_black.any():
        labels, count = ndimage.label(exact_black)
        border_ids = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
        border_ids.discard(0)
        if border_ids:
            padding = np.isin(labels, list(border_ids))
            good[padding] = False
    values[:,~good] = 0
    selection = "explicit band metadata"
    if not all(k in detect_band_map(src).indices for k in ("red", "green", "blue")):
        selection = "ordinary three-band RGB convention" if src.count == 3 else "FLAIR-style VHR RGB/NIR/DSM contract"
    return values.astype("float32"), good, {"indexes":indexes,"selection":selection,"method":method,"scales":scales.ravel().tolist(),"offsets":offsets.ravel().tolist()}
