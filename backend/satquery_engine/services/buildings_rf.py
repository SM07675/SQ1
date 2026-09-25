"""Local, pinned satellite building instance model and global tile merging."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage
from PIL import Image
from rasterio.windows import Window

from satquery_engine.services.ingestion import _starts
from satquery_engine.services.radiometry import rgb_unit_data

MODEL_ID = "merve/rf-detr-seg-satellite-buildings"
REVISION = "05b80dce9a57701724ad6a0fc052827c8b724257"
MODEL_SHA256 = "f254c1400f780f7ea72a6bc588a2150bf8b4d8845ded7269655e26a275f41ac7"
TILE = 512
STRIDE = 384
THRESHOLD = .5
MIN_INSTANCE_PIXELS = 16


@dataclass(frozen=True)
class RfBuildingSession:
    model: object
    processor: object
    checkpoint_sha256: str


@lru_cache(maxsize=1)
def load_rf_building(path: str) -> RfBuildingSession:
    from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation

    directory = Path(path)
    state = directory / "model.safetensors"
    if not state.is_file():
        raise ValueError("The satellite building instance model is unavailable.")
    with state.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Satellite building model identity could not be verified.")
    config = json.loads((directory / "config.json").read_text())
    processor_config = json.loads((directory / "preprocessor_config.json").read_text())
    if config.get("model_type") != "rf_detr" or config.get("id2label") != {"0": "building"}:
        raise ValueError("Unexpected satellite building model classes or architecture.")
    if (processor_config.get("image_processor_type") != "RfDetrImageProcessor"
            or processor_config.get("size") != {"height": 432, "width": 432}
            or not processor_config.get("do_rescale") or not processor_config.get("do_normalize")):
        raise ValueError("Unexpected satellite building preprocessing contract.")
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
    from satquery_engine.models.device import torch_device
    model = RfDetrForInstanceSegmentation.from_pretrained(directory, local_files_only=True).eval().to(torch_device())
    if model.config.id2label != {0: "building"}:
        raise ValueError("Satellite building model does not predict the building class.")
    return RfBuildingSession(model, processor, digest)


def _overlap(candidate, accepted):
    cy, cx, cmask, _ = candidate[:4]
    ay, ax, amask, _ = accepted[:4]
    top, left = max(cy, ay), max(cx, ax)
    bottom, right = min(cy + cmask.shape[0], ay + amask.shape[0]), min(cx + cmask.shape[1], ax + amask.shape[1])
    if bottom <= top or right <= left:
        return 0.0, 0.0
    a = cmask[top-cy:bottom-cy, left-cx:right-cx]
    b = amask[top-ay:bottom-ay, left-ax:right-ax]
    intersection = int(np.count_nonzero(a & b))
    if not intersection:
        return 0.0, 0.0
    a_size, b_size = int(cmask.sum()), int(amask.sum())
    return intersection / (a_size + b_size - intersection), intersection / min(a_size, b_size)


def merge_instance_tiles(candidates, shape, valid):
    """Count only global surviving instances after score-ordered duplicate removal."""
    labels = np.zeros(shape, dtype="int32")
    confidence = np.zeros(shape, dtype="float32")
    kept = []
    duplicates = 0
    # Complete interior predictions take precedence over roofs cut by an
    # internal tile edge. Confidence alone can otherwise preserve half a roof.
    for candidate in sorted(candidates, key=lambda item: (bool(item[4]) if len(item) > 4 else False, -item[3])):
        iou, containment = (0.0, 0.0)
        for previous in kept:
            current_iou, current_containment = _overlap(candidate, previous)
            iou, containment = max(iou, current_iou), max(containment, current_containment)
        if iou >= .5 or containment >= .8:
            duplicates += 1
            continue
        y, x, mask, score = candidate[:4]
        area = labels[y:y+mask.shape[0], x:x+mask.shape[1]]
        supported = mask & valid[y:y+mask.shape[0], x:x+mask.shape[1]]
        free = supported & (area == 0)
        # Several accepted roofs can jointly cover a duplicate candidate even
        # when no single pair reaches the overlap threshold. Never count the
        # small leftover strip of such a candidate as another building.
        if free.sum() < MIN_INSTANCE_PIXELS or free.sum() < .5 * supported.sum():
            duplicates += 1
            continue
        components, count = ndimage.label(free)
        sizes = np.bincount(components.ravel())
        minimum = max(MIN_INSTANCE_PIXELS, int(np.ceil(sizes[1:].max() * .02)))
        keep = sizes >= minimum
        keep[0] = False
        free = keep[components]
        if not free.any():
            duplicates += 1
            continue
        ident = len(kept) + 1
        area[free] = ident
        confidence[y:y+mask.shape[0], x:x+mask.shape[1]][free] = score
        kept.append((y, x, free.copy(), score))
    return labels, confidence, duplicates


def detect_rf_instances(path, session, scene_valid, progress=None):
    import torch

    with rasterio.open(path) as source:
        height, width = source.height, source.width
        positions = [(y, x) for y in _starts(height, TILE, STRIDE) for x in _starts(width, TILE, STRIDE)]
        candidates = []
        tiles = []
        for index, (y, x) in enumerate(positions):
            th, tw = min(TILE, height-y), min(TILE, width-x)
            rgb, good, audit = rgb_unit_data(source, window=Window(x, y, tw, th), scene_valid=scene_valid)
            image = Image.fromarray(np.moveaxis((rgb * 255).round().clip(0, 255).astype("uint8"), 0, -1))
            from satquery_engine.models.device import torch_device
            inputs = session.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(torch_device()) for key, value in inputs.items()}
            with torch.inference_mode():
                output = session.model(**inputs)
            result = session.processor.post_process_instance_segmentation(
                output, threshold=THRESHOLD, target_sizes=[(th, tw)]
            )[0]
            segmentation = result["segmentation"].cpu().numpy()
            if segmentation.shape != (th, tw):
                raise ValueError("Satellite building instance grid does not match its source tile.")
            for item in result["segments_info"]:
                if item.get("label_id") != 0 or not np.isfinite(item.get("score", np.nan)):
                    raise ValueError("Satellite model returned an invalid class or confidence.")
                score = float(item["score"])
                if not 0 <= score <= 1:
                    raise ValueError("Satellite model returned an invalid confidence.")
                mask = (segmentation == item["id"]) & good
                points = np.argwhere(mask)
                if not len(points):
                    continue
                top, left = points.min(axis=0)
                bottom, right = points.max(axis=0) + 1
                clipped = ((top == 0 and y > 0) or (left == 0 and x > 0)
                           or (bottom == th and y + th < height) or (right == tw and x + tw < width))
                candidates.append((y+int(top), x+int(left), mask[top:bottom, left:right].copy(), score, clipped))
            tiles.append({"tile_id": index, "x_offset": x, "y_offset": y,
                          "width": tw, "height": th, "valid_pixels": int(good.sum()),
                          "candidate_instances": len(result["segments_info"])})
            if progress:
                progress(f"Detecting buildings: tile {index+1} of {len(positions)}")
    labels, confidence, duplicate_count = merge_instance_tiles(candidates, (height, width), scene_valid)
    return labels, confidence, duplicate_count, tiles, audit
