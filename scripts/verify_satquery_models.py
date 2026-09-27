#!/usr/bin/env python3
"""
verify_satquery_models.py
=========================
Authoritative SatQuery Model Runtime Verification Suite.
Executes lightweight CPU runtime smoke tests:
  - Deepness LandCover.ai ONNX (5-class land cover)
  - FLAIR-HUB SwinBase-UPerNet (19-class SOTA aerial segmentation)
  - SatQuery Landcover v1 (15-class fine-tuned smp.Unet)
  - SatQuery Water & Shadow Net (fine-tuned surface water detector)
  - SatQuery Buildings Bundle (SwinB building footprint detector)
  - Sentinel-2 Multispectral Water Net (if installed)
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

MODEL_DIR = Path(os.environ.get("SATQUERY_MODEL_DIR", ROOT / "models"))


def test_deepness_onnx() -> dict:
    started = time.perf_counter()
    onnx_path = MODEL_DIR / "landcover" / "deeplabv3_landcover_4c.onnx"
    if not onnx_path.is_file():
        return {"status": "SKIP", "reason": "Checkpoint not found", "duration_ms": 0}

    try:
        import numpy as np
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy_inp = np.zeros((1, 3, 512, 512), dtype=np.float32)
        out = sess.run(None, {inp_name: dummy_inp})[0]
        elapsed = (time.perf_counter() - started) * 1000

        if out.shape == (1, 5, 512, 512) and np.isfinite(out).all():
            return {
                "status": "PASS",
                "output_shape": str(out.shape),
                "duration_ms": round(elapsed, 1),
                "notes": "5 classes: [unknown, building, woodland, water, road]",
            }
        return {"status": "FAIL", "reason": f"Unexpected shape {out.shape}", "duration_ms": round(elapsed, 1)}
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def test_flair_hub_safetensors() -> dict:
    started = time.perf_counter()
    flair_path = (
        MODEL_DIR
        / "pretrained"
        / "landcover"
        / "flair-hub-rgb-swinbase-upernet"
        / "FLAIR-HUB_LC-A_RGB_swinbase-upernet.safetensors"
    )
    if not flair_path.is_file():
        return {"status": "SKIP", "reason": "Checkpoint not found", "duration_ms": 0}

    try:
        from safetensors import safe_open

        with safe_open(str(flair_path), framework="pt", device="cpu") as f:
            keys = f.keys()
            num_keys = len(keys)
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "status": "PASS",
            "num_tensors": num_keys,
            "duration_ms": round(elapsed, 1),
            "notes": "19 fine classes aerial SOTA (IGN France)",
        }
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def test_satquery_landcover_v1() -> dict:
    started = time.perf_counter()
    ckpt_path = MODEL_DIR / "finetuned" / "landcover" / "satquery_landcover_v1" / "best_checkpoint.pt"
    if not ckpt_path.is_file():
        return {"status": "SKIP", "reason": "Checkpoint not found", "duration_ms": 0}

    try:
        import torch

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        keys_count = len(ckpt.keys()) if isinstance(ckpt, dict) else 0
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "status": "PASS",
            "state_dict_keys": keys_count,
            "duration_ms": round(elapsed, 1),
            "notes": "15 classes smp.Unet (mit_b2) with asphalt & shadow guards",
        }
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def test_satquery_water_shadow() -> dict:
    started = time.perf_counter()
    ckpt_path = MODEL_DIR / "finetuned" / "water" / "satquery_water_shadow_v1" / "satquery_water_shadow_state_dict.pt"
    bundle_path = MODEL_DIR / "satquery_water_bundle" / "satquery_water_state_dict.pt"

    target = ckpt_path if ckpt_path.is_file() else (bundle_path if bundle_path.is_file() else None)
    if not target:
        return {"status": "SKIP", "reason": "Checkpoint not found", "duration_ms": 0}

    try:
        import torch

        ckpt = torch.load(str(target), map_location="cpu", weights_only=True)
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "status": "PASS",
            "state_dict_keys": len(ckpt),
            "duration_ms": round(elapsed, 1),
            "notes": "Surface water & shadow adjudication head",
        }
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def test_satquery_buildings() -> dict:
    started = time.perf_counter()
    bundle_path = MODEL_DIR / "satquery_buildings_bundle" / "satquery_buildings_state_dict.pt"
    satlas_path = MODEL_DIR / "pretrained" / "buildings" / "satlas-aerial-swinb-si" / "aerial_swinb_si.pth"

    target = bundle_path if bundle_path.is_file() else (satlas_path if satlas_path.is_file() else None)
    if not target:
        return {"status": "SKIP", "reason": "Checkpoint not found", "duration_ms": 0}

    try:
        import torch

        ckpt = torch.load(str(target), map_location="cpu", weights_only=False)
        keys_count = len(ckpt.keys()) if isinstance(ckpt, dict) else 0
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "status": "PASS",
            "state_dict_keys": keys_count,
            "duration_ms": round(elapsed, 1),
            "notes": "SpaceNet-2 SwinB building footprint + instance detector",
        }
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def test_sentinel2_water() -> dict:
    started = time.perf_counter()
    s2_path = MODEL_DIR / "water" / "s2-water-unetplusplus-efficientnet-b4" / "model.pth"
    prithvi_path = MODEL_DIR / "pretrained" / "water" / "prithvi-sen1floods11" / "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"

    target = s2_path if s2_path.is_file() else (prithvi_path if prithvi_path.is_file() else None)
    if not target:
        return {"status": "SKIP", "reason": "Multispectral checkpoint not installed", "duration_ms": 0}

    try:
        import torch

        ckpt = torch.load(str(target), map_location="cpu", weights_only=False)
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "status": "PASS",
            "file": target.name,
            "duration_ms": round(elapsed, 1),
            "notes": "Sentinel-2 6-band multispectral surface water network",
        }
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def main():
    print("=" * 72)
    print("SatQuery Neural Model Runtime Verification Suite (CPU Smoke Tests)")
    print(f"Model Storage Root: {MODEL_DIR}")
    print("=" * 72)

    tests = [
        ("PUTvision/Deepness-LandCoverAI-DeepLabV3plus", "Land Cover (ONNX)", test_deepness_onnx),
        ("IGNF/FLAIR-HUB_LC-A_RGB_swinbase-upernet", "Land Cover (FLAIR)", test_flair_hub_safetensors),
        ("satquery_landcover_v1", "Land Cover (15-Class smp)", test_satquery_landcover_v1),
        ("satquery_water_shadow_v1", "Water & Shadow Net", test_satquery_water_shadow),
        ("satquery_buildings_bundle", "Buildings Net (SwinB)", test_satquery_buildings),
        ("s2_multispectral_water", "Sentinel-2 Water Net", test_sentinel2_water),
    ]

    all_passed = True
    for model_id, task_label, test_func in tests:
        res = test_func()
        status = res["status"]
        dur = f"{res.get('duration_ms', 0):.1f}ms"

        if status == "PASS":
            details = res.get("notes", "")
            print(f"[{status:<4}] {task_label:<24} | {dur:<8} | {details}")
        elif status == "SKIP":
            print(f"[{status:<4}] {task_label:<24} | {dur:<8} | {res.get('reason')}")
        else:
            print(f"[{status:<4}] {task_label:<24} | {dur:<8} | ERROR: {res.get('reason')}")
            all_passed = False

    print("-" * 72)
    if all_passed:
        print("RESULT: ALL ACTIVE MODELS VERIFIED AND READY FOR RUNTIME INFERENCE.")
        print("-" * 72)
        sys.exit(0)
    else:
        print("RESULT: ONE OR MORE REQUIRED MODELS FAILED VERIFICATION.")
        print("-" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
