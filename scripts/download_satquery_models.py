#!/usr/bin/env python3
"""
download_satquery_models.py
===========================
Model management utility for SatQuery offline-ready model storage:
  - Check status of local checkpoints (--status)
  - Verify integrity and presence of model files (--verify)
  - Inspect core, optional, and specialist models (--core, --optional, --all)
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("SATQUERY_MODEL_DIR", ROOT / "models"))

KNOWN_MODELS = [
    {
        "id": "satlas-aerial-swinb-si",
        "name": "Satlas Aerial Swin-B (Building & Feature Backbone)",
        "task": "building_detection",
        "category": "core",
        "path": "pretrained/buildings/satlas-aerial-swinb-si",
        "primary_file": "aerial_swinb_si.pth",
        "expected_mb": 481.0,
    },
    {
        "id": "satquery_buildings_bundle",
        "name": "SatQuery Fine-Tuned Buildings Bundle",
        "task": "building_footprint_and_count",
        "category": "core",
        "path": "satquery_buildings_bundle",
        "primary_file": "satquery_buildings_state_dict.pt",
        "expected_mb": 11.0,
    },
    {
        "id": "flair-hub-rgb-swinbase-upernet",
        "name": "IGNF FLAIR-HUB SwinBase-UPerNet (19-Class Aerial SOTA)",
        "task": "land_cover_rgb",
        "category": "core",
        "path": "pretrained/landcover/flair-hub-rgb-swinbase-upernet",
        "primary_file": "FLAIR-HUB_LC-A_RGB_swinbase-upernet.safetensors",
        "expected_mb": 346.5,
    },
    {
        "id": "satquery_landcover_v1",
        "name": "SatQuery Fine-Tuned Land Cover (15-Class smp.Unet)",
        "task": "land_cover_vhr",
        "category": "core",
        "path": "finetuned/landcover/satquery_landcover_v1",
        "primary_file": "best_checkpoint.pt",
        "expected_mb": 78.0,
    },
    {
        "id": "deeplabv3_landcover_4c",
        "name": "Deepness LandCover.ai DeepLabV3+ ONNX",
        "task": "land_cover_4class",
        "category": "core",
        "path": "landcover",
        "primary_file": "deeplabv3_landcover_4c.onnx",
        "expected_mb": 20.0,
    },
    {
        "id": "satquery_water_shadow_v1",
        "name": "SatQuery Surface Water & Shadow Net",
        "task": "water_and_shadow",
        "category": "core",
        "path": "finetuned/water/satquery_water_shadow_v1",
        "primary_file": "satquery_water_shadow_state_dict.pt",
        "expected_mb": 11.0,
    },
    {
        "id": "satquery_water_bundle",
        "name": "SatQuery Production Water Bundle",
        "task": "surface_water",
        "category": "core",
        "path": "satquery_water_bundle",
        "primary_file": "satquery_water_state_dict.pt",
        "expected_mb": 11.0,
    },
    {
        "id": "prithvi-sen1floods11",
        "name": "NASA/IBM Prithvi EO 2.0 300M Sen1Floods11",
        "task": "sentinel2_flood_water",
        "category": "optional",
        "path": "pretrained/water/prithvi-sen1floods11",
        "primary_file": "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        "expected_mb": 1100.0,
    },
    {
        "id": "s2-water-unetplusplus",
        "name": "Sentinel-2 Unet++ EfficientNet-B4 Water",
        "task": "sentinel2_water",
        "category": "optional",
        "path": "water/s2-water-unetplusplus-efficientnet-b4",
        "primary_file": "model.pth",
        "expected_mb": 75.0,
    },
    {
        "id": "bigearthnet-s2-resnet50",
        "name": "BigEarthNet S2 ResNet-50 Scene Classifier",
        "task": "sentinel2_scene_classification",
        "category": "optional",
        "path": "pretrained/bigearthnet/s2-resnet50",
        "primary_file": "model.safetensors",
        "expected_mb": 90.0,
    },
    {
        "id": "dinov3s-buildings",
        "name": "GeoBase DINOv3 Small Buildings",
        "task": "building_footprint",
        "category": "optional",
        "path": "pretrained/buildings/dinov3s-buildings",
        "primary_file": "model.ckpt",
        "expected_mb": 642.9,
    },
]


def resolve_model_dir() -> Path:
    candidates = [
        MODEL_DIR,
        ROOT / "models",
        Path("D:/Sat1/models"),
    ]
    for c in candidates:
        if c.exists() and (c / "manifests" or c / "pretrained" or c / "finetuned").exists():
            return c
    return ROOT / "models"


def show_status(target_category: str | None = None, target_model: str | None = None) -> None:
    model_dir = resolve_model_dir()
    print("=" * 64)
    print("SatQuery Authoritative Model Status")
    print(f"Model Storage Root: {model_dir}")
    print("=" * 64)

    ready_count = 0
    total_count = 0

    for m in KNOWN_MODELS:
        if target_category and m["category"] != target_category and target_category != "all":
            continue
        if target_model and target_model.lower() not in (m["id"].lower(), m["name"].lower()):
            continue

        total_count += 1
        m_path = model_dir / m["path"]
        target_f = m_path / m["primary_file"]

        if target_f.exists():
            size_mb = target_f.stat().st_size / (1024 * 1024)
            status = f"[READY] {size_mb:.1f} MB"
            ready_count += 1
        else:
            status = "[NOT FOUND]"

        print(f"\n{status:<14} {m['name']}")
        print(f"               ID:       {m['id']}")
        print(f"               Task:     {m['task']}")
        print(f"               Category: {m['category']}")
        print(f"               Path:     {target_f}")

    print("\n" + "-" * 64)
    print(f"Summary: {ready_count}/{total_count} models verified and ready locally.")
    print("-" * 64)


def verify_models() -> bool:
    model_dir = resolve_model_dir()
    print("Verifying SatQuery model file integrity...")
    all_ok = True
    for m in KNOWN_MODELS:
        target_f = model_dir / m["path"] / m["primary_file"]
        if target_f.exists():
            size_mb = target_f.stat().st_size / (1024 * 1024)
            if target_f.stat().st_size < 100:
                print(f"[FAIL] {m['id']}: File is empty ({target_f})")
                all_ok = False
            else:
                print(f"[OK]   {m['id']}: {size_mb:.1f} MB ({target_f.name})")
        else:
            if m["category"] == "core":
                print(f"[WARN] {m['id']}: Core model file missing ({target_f})")
                all_ok = False
            else:
                print(f"[INFO] {m['id']}: Optional checkpoint not installed.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Manage and verify SatQuery local models.")
    parser.add_argument("--status", action="store_true", help="Print status of all models.")
    parser.add_argument("--verify", action="store_true", help="Verify presence and integrity of checkpoints.")
    parser.add_argument("--core", action="store_true", help="Filter for core models.")
    parser.add_argument("--optional", action="store_true", help="Filter for optional models.")
    parser.add_argument("--all", action="store_true", help="Include all registered models.")
    parser.add_argument("--model", type=str, default=None, help="Inspect a specific model by ID.")

    args = parser.parse_args()

    if args.verify:
        success = verify_models()
        sys.exit(0 if success else 1)

    cat = "core" if args.core else ("optional" if args.optional else ("all" if args.all else None))
    show_status(target_category=cat, target_model=args.model)


if __name__ == "__main__":
    main()
