"""Six-band Sentinel-2 water specialist with explicit band and radiometry checks."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from satquery_engine.services.ingestion import _starts


MODEL_ID = "giswqs/s2-water-unetplusplus-efficientnet-b4"
MODEL_SHA256 = "fb8ae300b8b8e929fa764d72ac476a419e8eb9f121e52cce6c0c8816a48d487e"
REQUIRED = ("B02", "B03", "B04", "B08", "B11", "B12")


def _canonical_band(value: str) -> str | None:
    match = re.fullmatch(r"(?:B|BAND)0*(\d{1,2})", value.upper().replace(" ", ""))
    return f"B{int(match.group(1)):02d}" if match else None


def band_indexes(src) -> tuple[int, ...]:
    from satquery_engine.services.buildings import resolution_m

    gsd = resolution_m(src)
    if gsd is None or not 8 <= gsd <= 30:
        raise ValueError("Six-band Sentinel-2 water inference requires georeferenced 8–30 metre pixels.")
    if src.count < 6:
        raise ValueError("Six-band Sentinel-2 water inference requires B02, B03, B04, B08, B11 and B12.")
    descriptions = list(src.descriptions)
    if not any(descriptions):
        declared = src.tags().get("band_order", "")
        descriptions = [item.strip() for item in declared.split(",")]
    names = [_canonical_band(str(value or "")) for value in descriptions]
    if any(names.count(name) != 1 for name in REQUIRED):
        raise ValueError("Six-band water model needs unique named B02, B03, B04, B08, B11 and B12 bands; band order is not guessed.")
    return tuple(names.index(name) + 1 for name in REQUIRED)


@lru_cache(maxsize=1)
def load_model(checkpoint: str):
    import segmentation_models_pytorch as smp
    import torch

    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError("The six-band Sentinel-2 water checkpoint is unavailable.")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("Six-band water checkpoint identity could not be verified.")
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None,
                             in_channels=6, classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def predict(src, checkpoint: Path):
    """Return native-grid water probability, validity, and exact model provenance."""
    import torch

    indexes = band_indexes(src)
    model = load_model(str(checkpoint.resolve()))
    from satquery_engine.models.device import torch_device
    height, width = src.height, src.width
    rows = _starts(height, 512, 256)
    columns = _starts(width, 512, 256)
    # A per-window range check can normalize one dark tile as reflectance and
    # its brighter neighbour as digital numbers. Decide once for the scene.
    reflectance_input = False
    if all(np.issubdtype(np.dtype(src.dtypes[i - 1]), np.floating) for i in indexes):
        scene_max = 0.0
        for row in rows:
            for col in columns:
                raw = src.read(indexes, window=Window(col, row, min(512, width-col), min(512, height-row)), masked=True)
                finite = raw.compressed()
                if finite.size:
                    if not np.isfinite(finite).all() or finite.min() < 0 or finite.max() > 12000:
                        raise ValueError("Six-band water model requires finite nonnegative Sentinel-2 values up to 12000.")
                    scene_max = max(scene_max, float(finite.max()))
        reflectance_input = 0 < scene_max <= 1.1
    elif not all(np.issubdtype(np.dtype(src.dtypes[i - 1]), np.integer) for i in indexes):
        raise ValueError("Six-band water model requires a consistent numeric type across its bands.")
    device = torch_device()
    model.to(device)
    accum = np.zeros((height, width), dtype="float32")
    weights = np.zeros_like(accum)
    valid = np.zeros((height, width), dtype=bool)
    axis = np.maximum(np.hanning(512), .05)
    kernel = np.outer(axis, axis).astype("float32")
    tiles = 0
    # The publisher's inference normalizes stored Sentinel-2 digital numbers by
    # 255, not by 10,000. Altering that scale changed predictions drastically.
    try:
        with torch.inference_mode():
            for row in rows:
                for col in columns:
                    h, w = min(512, height - row), min(512, width - col)
                    raw = src.read(indexes, window=Window(col, row, w, h), masked=True)
                    values = raw.astype("float32").filled(np.nan)
                    good = np.all(np.isfinite(values), axis=0)
                    finite = values[:, good]
                    if finite.size:
                        maximum = float(np.max(finite))
                        minimum = float(np.min(finite))
                        if minimum < 0 or maximum > 12000:
                            raise ValueError("Six-band water model expects nonnegative Sentinel-2 digital numbers up to 12000.")
                    if reflectance_input:
                        values *= 10000.0  # scene-wide reflectance -> training digital-number scale
                    image = np.zeros((6, 512, 512), dtype="float32")
                    image[:, :h, :w] = np.where(good[None], values / 255.0, 0.0)
                    logits = model(torch.from_numpy(image)[None].to(device))
                    if logits.shape != (1, 2, 512, 512) or not torch.isfinite(logits).all():
                        raise ValueError("Six-band water model returned invalid logits.")
                    probability = torch.softmax(logits, dim=1)[0, 1, :h, :w].cpu().numpy()
                    k = kernel[:h, :w]
                    accum[row:row+h, col:col+w] += probability * k
                    weights[row:row+h, col:col+w] += k
                    valid[row:row+h, col:col+w] |= good
                    tiles += 1
    finally:
        if device == "cuda":
            model.cpu()
            torch.cuda.empty_cache()
    if not valid.any() or np.any(weights <= 0):
        raise ValueError("Six-band Sentinel-2 water inference has no valid full-scene coverage.")
    probability = np.where(valid, accum / weights, 0.0)
    return probability, valid, {
        "model_id": MODEL_ID, "threshold": .5, "min_area_pixels": 0,
        "checkpoint_sha256": MODEL_SHA256, "checkpoint_execution": "s2_water_unetplusplus",
        "device": device,
        "tile_count": tiles, "preprocessing": {"band_order": list(REQUIRED),
            "normalization": "scene-wide reflectance * 10000 / 255" if reflectance_input else "Sentinel-2 digital numbers / 255", "tile_size": 512,
            "overlap": 256},
    }
