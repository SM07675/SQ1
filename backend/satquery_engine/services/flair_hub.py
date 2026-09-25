"""Strict IGNF FLAIR-HUB LC-A RGB inference on the source pixel grid."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window

from satquery_engine.config import settings
from satquery_engine.models.device import torch_device
from satquery_engine.services.ingestion import _starts
from satquery_engine.services.radiometry import rgb_unit_data


MODEL_ID = "IGNF/FLAIR-HUB_LC-A_RGB_swinbase-upernet"
WEIGHT_NAME = "FLAIR-HUB_LC-A_RGB_swinbase-upernet.safetensors"
TILE = 512
STRIDE = 384


def _root() -> Path:
    manifest_file = settings.model_dir / "manifests/models.yaml"
    if not manifest_file.is_file():
        raise ValueError("FLAIR-HUB is not registered.")
    entry = (yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}).get("models", {}).get("land_flair_hub")
    if not isinstance(entry, dict) or entry.get("health_status") != "READY" or not entry.get("path"):
        raise ValueError("FLAIR-HUB has no verified READY registration.")
    configured = Path(str(entry["path"]))
    root = (configured if configured.is_absolute() else settings.model_dir / configured).resolve()
    if not root.is_relative_to(settings.model_dir.resolve()):
        raise ValueError("FLAIR-HUB path escapes SATQUERY_MODEL_DIR.")
    return root


def _contract(root: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    modality = yaml.safe_load((root / "configs_train/config_modalities.yaml").read_text(encoding="utf-8"))["modalities"]
    supervision = yaml.safe_load((root / "configs_train/config_supervision.yaml").read_text(encoding="utf-8"))["labels_configs"]["AERIAL_LABEL-COSIA"]
    classes = supervision["value_name"]
    if sorted(classes) != list(range(19)) or modality["inputs_channels"]["AERIAL_RGBI"] != [1, 2, 3]:
        raise ValueError("FLAIR-HUB class or channel map differs from the registered model contract.")
    if any(supervision["value_weights"]["default_exceptions"].get(i) != 0 for i in (15, 16, 17, 18)):
        raise ValueError("FLAIR-HUB ignored class IDs have changed.")
    norm = modality["normalization"]
    if norm["norm_type"] != "custom":
        raise ValueError("FLAIR-HUB RGB normalization has changed.")
    mean = np.asarray(norm["AERIAL_RGBI_means"], dtype="float32")[:, None, None]
    std = np.asarray(norm["AERIAL_RGBI_stds"], dtype="float32")[:, None, None]
    return tuple(classes[i] for i in range(19)), mean, std


@lru_cache(maxsize=1)
def load_flair_hub():
    root = _root()
    payload = load_flair_hub_from_root(str(root))
    proof = root / "satquery_model_health.json"
    if not proof.is_file():
        raise ValueError("FLAIR-HUB smoke-inference proof is missing.")
    import json
    attestation = json.loads(proof.read_text(encoding="utf-8"))
    if (attestation.get("checkpoint_sha256") != payload[4] or
            attestation.get("inference_verified") is not True or
            attestation.get("output_shape") != [1, 19, 512, 512]):
        raise ValueError("FLAIR-HUB health proof does not match the checkpoint and output schema.")
    return payload


@lru_cache(maxsize=1)
def load_flair_hub_from_root(root_path: str):
    import segmentation_models_pytorch as smp
    import torch
    from safetensors.torch import load_file

    if smp.__version__ != "0.4.0":
        raise ValueError("FLAIR-HUB requires segmentation-models-pytorch==0.4.0 for its saved decoder.")
    root = Path(root_path)
    classes, mean, std = _contract(root)
    config = yaml.safe_load((root / "configs_train/config_models.yaml").read_text(encoding="utf-8"))
    if config["models"]["monotemp_model"]["arch"] != "swin_base_patch4_window12_384-upernet":
        raise ValueError("FLAIR-HUB architecture differs from the saved checkpoint.")
    model = smp.create_model(
        arch="upernet", encoder_name="tu-swin_base_patch4_window12_384",
        encoder_weights=None, classes=len(classes), in_channels=3, img_size=TILE,
    )
    # Upstream constructs its task decoder with a one-channel placeholder
    # encoder, then connects it to the three-channel aerial encoder. That
    # changes one UPerNet skip-convolution shape, so reproduce it exactly.
    decoder_source = smp.create_model(
        arch="upernet", encoder_name="tu-swin_base_patch4_window12_384",
        encoder_weights=None, classes=len(classes), in_channels=1, img_size=TILE,
    )
    model.decoder = decoder_source.decoder
    model.segmentation_head = decoder_source.segmentation_head
    del decoder_source
    source = load_file(str(root / WEIGHT_NAME), device="cpu")
    state = {}
    encoder = "model.encoders.AERIAL_RGBI.seg_model."
    decoder = "model.main_decoders.AERIAL_LABEL-COSIA.seg_model."
    for key, value in source.items():
        if key.startswith(encoder):
            state["encoder." + key[len(encoder):]] = value
        elif key.startswith(decoder):
            state[key[len(decoder):]] = value
    if set(state) != set(model.state_dict()):
        raise ValueError("FLAIR-HUB checkpoint keys do not match the exact model architecture.")
    model.load_state_dict(state, strict=True)
    model.eval()
    with (root / WEIGHT_NAME).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return model, classes, mean, std, digest


@lru_cache(maxsize=1)
def predict_flair_hub(path: Path) -> dict:
    """Blend probabilities before assigning labels; retain all 19 raw channels."""
    import torch
    from scipy.special import softmax

    model, classes, mean, std, digest = load_flair_hub()
    device = torch_device()
    model.to(device)
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        if h * w > 4_000_000:
            raise ValueError("FLAIR-HUB RGB inference is limited to 4 million pixels on this host.")
        _, scene_valid, _ = rgb_unit_data(src)
        accum = np.zeros((len(classes), h, w), dtype="float32")
        weights = np.zeros((h, w), dtype="float32")
        axis = np.maximum(np.hanning(TILE), 1e-3)
        kernel = np.maximum(np.outer(axis, axis), 1e-4).astype("float32")
        tile_records = []
        for y in _starts(h, TILE, STRIDE):
            for x in _starts(w, TILE, STRIDE):
                th, tw = min(TILE, h-y), min(TILE, w-x)
                rgb, valid, radiometry = rgb_unit_data(src, window=Window(x, y, tw, th), scene_valid=scene_valid)
                patch = np.pad(rgb * 255.0, ((0, 0), (0, TILE-th), (0, TILE-tw)), mode="edge")
                normalized = (patch - mean) / std
                with torch.inference_mode():
                    logits = model(torch.from_numpy(normalized).unsqueeze(0).to(device))[0].cpu().numpy()
                if logits.shape != (len(classes), TILE, TILE) or not np.isfinite(logits).all():
                    raise ValueError("FLAIR-HUB produced an invalid output schema.")
                probability = softmax(logits, axis=0)
                k = kernel[:th, :tw]
                accum[:, y:y+th, x:x+tw] += probability[:, :th, :tw] * k
                weights[y:y+th, x:x+tw] += k
                tile_records.append({"tile_id": len(tile_records), "x_offset": x, "y_offset": y,
                                     "width": tw, "height": th, "padding": [TILE-th, TILE-tw],
                                     "valid_pixels": int(valid.sum())})
        if not scene_valid.any() or np.any(weights <= 0):
            raise ValueError("FLAIR-HUB did not cover all valid source pixels.")
        result = accum / weights
        result[:, ~scene_valid] = 0.0
        if device == "cuda":
            model.cpu()
            torch.cuda.empty_cache()
        return {"probability": result, "valid": scene_valid, "model_id": MODEL_ID,
                "checkpoint_sha256": digest, "classes": classes, "tiles": tile_records, "device": device,
                "preprocessing": {"rgb": radiometry, "normalization": "official AERIAL_RGBI custom mean/std in uint8 scale"},
                "domain_note": "FLAIR-HUB was evaluated on its own aerial test set; transfer accuracy here is unmeasured."}
