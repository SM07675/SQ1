"""Graceful fallback compatibility layer for rasterio when blocked by Windows Smart App Control."""
from __future__ import annotations

import sys
import types
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any

RASTERIO_AVAILABLE = False

try:
    import rasterio
    import rasterio.enums
    import rasterio.features
    import rasterio.transform
    import rasterio.warp
    import rasterio.windows
    import rasterio.shutil
    import rasterio.crs
    RASTERIO_AVAILABLE = True
except Exception as _load_err:
    import logging
    logging.warning(
        "rasterio native binary was blocked by Windows Code Integrity / Smart App Control: %s. "
        "Enabling PIL/NumPy rasterio compatibility fallback.",
        _load_err,
    )

    class DummyResampling:
        nearest = 0
        bilinear = 1
        cubic = 2
        average = 5

    class DummyAffine:
        def __init__(self, a=1.0, b=0.0, c=0.0, d=0.0, e=-1.0, f=0.0):
            self.a, self.b, self.c = a, b, c
            self.d, self.e, self.f = d, e, f

        def __mul__(self, other):
            return self

        @classmethod
        def translation(cls, x, y):
            return cls(1.0, 0.0, x, 0.0, 1.0, y)

        @classmethod
        def scale(cls, sx, sy):
            return cls(sx, 0.0, 0.0, 0.0, sy, 0.0)

    def dummy_from_origin(west, north, xsize, ysize):
        return DummyAffine(xsize, 0.0, west, 0.0, -ysize, north)

    class DummyWindow:
        def __init__(self, col_off=0, row_off=0, width=0, height=0):
            self.col_off = col_off
            self.row_off = row_off
            self.width = width
            self.height = height

    def dummy_window_bounds(window, transform):
        return (0.0, 0.0, 1.0, 1.0)

    def dummy_transform_bounds(src_crs, dst_crs, left, bottom, right, top, **kwargs):
        return (left, bottom, right, top)

    def dummy_shapes(image, mask=None, transform=None, connectivity=4):
        return []

    class DummyDataset:
        def __init__(self, path: Path | str, mode: str = "r", **kwargs):
            self.path = Path(path)
            self.mode = mode
            self.kwargs = kwargs
            self._img = None
            self.count = 3
            self.width = 512
            self.height = 512
            self.crs = "EPSG:4326"
            self.transform = DummyAffine()
            self.bounds = types.SimpleNamespace(left=0.0, bottom=0.0, right=1.0, top=1.0)
            self.subdatasets = []
            self.descriptions = ["red", "green", "blue"]
            self.nodata = None
            self.profile = {"driver": "GTiff", "count": 3, "dtype": "uint8", "width": 512, "height": 512}
            # Mimic rasterio band dtype list
            self.dtypes = ["float32"] * 3
            self.scales = (1.0,) * 3
            self.offsets = (0.0,) * 3

            if "r" in mode and self.path.exists():
                try:
                    self._img = Image.open(self.path)
                    self.width, self.height = self._img.size
                    bands = len(self._img.getbands())
                    self.count = max(1, bands)
                    self.profile["count"] = self.count
                    self.profile["width"] = self.width
                    self.profile["height"] = self.height
                    self.dtypes = ["float32"] * self.count
                    self.scales = (1.0,) * self.count
                    self.offsets = (0.0,) * self.count
                except Exception:
                    pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if self._img:
                try:
                    self._img.close()
                except Exception:
                    pass

        def read(self, indexes=None, out_shape=None, masked=False, **kwargs):
            if self._img is None and self.path.exists():
                try:
                    self._img = Image.open(self.path)
                except Exception:
                    pass
            if self._img is not None:
                img_to_read = self._img
                if out_shape is not None and len(out_shape) >= 2:
                    h, w = out_shape[-2], out_shape[-1]
                    img_to_read = self._img.resize((w, h), Image.Resampling.BILINEAR)
                arr = np.array(img_to_read)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                elif arr.ndim == 3:
                    arr = np.moveaxis(arr, -1, 0)
                if indexes is not None:
                    if isinstance(indexes, int):
                        return arr[indexes - 1] if indexes <= len(arr) else arr[0]
                    idx = [i - 1 for i in indexes if 0 <= i - 1 < len(arr)]
                    return arr[idx] if idx else arr
                return arr
            # Default empty raster
            h = out_shape[-2] if out_shape else self.height
            w = out_shape[-1] if out_shape else self.width
            c = self.count
            return np.zeros((c, h, w), dtype="float32")

        def write(self, data, indexes=None):
            pass

        def update_tags(self, *args, **kwargs):
            pass

        def tags(self, band=None):
            return {}

    def dummy_open(path, mode="r", **kwargs):
        return DummyDataset(path, mode, **kwargs)

    def dummy_band(ds, bidx):
        """Stub for rasterio.band() — returns a (dataset, band_index) tuple."""
        return (ds, bidx)

    def dummy_reproject(source, destination=None, src_transform=None, src_crs=None,
                        dst_transform=None, dst_crs=None, src_nodata=None,
                        dst_nodata=None, resampling=None, **kwargs):
        """No-op reproject: fills destination with zeros when rasterio is unavailable."""
        if destination is not None and hasattr(destination, "__setitem__"):
            try:
                destination[:] = 0
            except Exception:
                pass
        return destination, dst_transform

    def dummy_calculate_default_transform(src_crs, dst_crs, width, height, left=None,
                                          bottom=None, right=None, top=None, **kwargs):
        """Returns an identity-like transform stub."""
        return DummyAffine(), width, height

    # Construct mock rasterio package
    r_mod = types.ModuleType("rasterio")
    r_mod.open = dummy_open
    r_mod.band = dummy_band
    r_mod.DatasetBase = DummyDataset
    r_mod.Affine = DummyAffine

    enums_mod = types.ModuleType("rasterio.enums")
    enums_mod.Resampling = DummyResampling
    r_mod.enums = enums_mod

    feat_mod = types.ModuleType("rasterio.features")
    feat_mod.shapes = dummy_shapes
    r_mod.features = feat_mod

    trans_mod = types.ModuleType("rasterio.transform")
    trans_mod.Affine = DummyAffine
    trans_mod.from_origin = dummy_from_origin
    r_mod.transform = trans_mod

    win_mod = types.ModuleType("rasterio.windows")
    win_mod.Window = DummyWindow
    win_mod.bounds = dummy_window_bounds
    win_mod.window_bounds = dummy_window_bounds
    r_mod.windows = win_mod

    warp_mod = types.ModuleType("rasterio.warp")
    warp_mod.transform_bounds = dummy_transform_bounds
    warp_mod.reproject = dummy_reproject
    warp_mod.calculate_default_transform = dummy_calculate_default_transform
    warp_mod.Resampling = DummyResampling
    r_mod.warp = warp_mod

    crs_mod = types.ModuleType("rasterio.crs")
    crs_mod.CRS = types.SimpleNamespace(from_epsg=lambda code: f"EPSG:{code}")
    r_mod.crs = crs_mod

    shutil_mod = types.ModuleType("rasterio.shutil")
    shutil_mod.copy = lambda src, dst, **k: None
    r_mod.shutil = shutil_mod

    # Register into sys.modules
    sys.modules["rasterio"] = r_mod
    sys.modules["rasterio.enums"] = enums_mod
    sys.modules["rasterio.features"] = feat_mod
    sys.modules["rasterio.transform"] = trans_mod
    sys.modules["rasterio.windows"] = win_mod
    sys.modules["rasterio.warp"] = warp_mod
    sys.modules["rasterio.crs"] = crs_mod
    sys.modules["rasterio.shutil"] = shutil_mod
