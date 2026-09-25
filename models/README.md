# SatQuery Local Model Storage System

Welcome to the authoritative local model repository for SatQuery. All deep learning checkpoints, vision foundation models, and specialist neural network weights are stored and managed here for offline-ready inference.

---

## 1. Directory Structure

```
D:\Sat1\models\
├── buildings\
│   ├── dinov3s-buildings\            # Primary building footprint segmentation (DINOv3 Small ONNX)
│   └── geobase-building-detection\   # Secondary / validation building detector
├── land_cover\
│   ├── flair-rgb-15cl-deeplabv3\     # Primary high-resolution RGB 15-class land-cover segmentation
│   ├── bigearthnet-s2-resnet50\      # Sentinel-2 10-band scene-level land-cover classifier
│   ├── bigearthnet-s1-resnet50\      # Sentinel-1 SAR 2-band scene-level land-cover classifier
│   └── bigearthnet-s1s2-resnet101\   # Joint Sentinel-1 + Sentinel-2 12-band scene-level classifier
├── water\
│   └── prithvi-sen1floods11\         # Sentinel-2 6-band water/flood segmentation (Prithvi EO 2.0)
├── vlm\
│   └── earthdial-4b\                 # Remote sensing vision-language model (RGB & Multispectral)
├── remoteclip\
│   ├── rn50\                         # RemoteCLIP ResNet-50 text-to-tile retrieval
│   └── vit-b-32\                     # RemoteCLIP ViT-B-32 text-to-tile retrieval
├── fusion\
│   └── croma\                        # Optical + SAR foundation model (CROMA base / large)
├── change_detection\
│   ├── tinycd\                       # TinyCD LEVIR / WHU change detection
│   ├── bit\                          # BIT ResNet-18 change detection
│   ├── changerex\                    # ChangerEx ResNet-18 change detection
│   ├── ban\                          # BAN ViT-L14 + MIT-B0 change detection
│   └── changemamba\                  # ChangeMamba documentation & Linux runtime notes
├── grounding\
│   ├── grounding-dino\               # Optional: Grounding DINO open-vocabulary detector
│   └── sam2\                         # Optional: SAM 2 mask refinement
├── manifests\
│   ├── models.yaml                   # Declarative metadata, tasks, modalities, and band orderings
│   └── download_manifest.json        # Checksums, file sizes, and download audit trail
└── cache\
    └── huggingface\                  # Hugging Face cache root (HF_HOME)
```

---

## 2. Configuration

Configure model storage location via environment variable:

```bash
# In .env:
SATQUERY_MODEL_DIR=D:\Sat1\models
HF_HOME=D:\Sat1\models\cache\huggingface
SATQUERY_OFFLINE_MODE=false
```

- **Default path**: `D:\Sat1\models`
- **Fallback**: `<project_root>/models`
- **Offline mode**: When `SATQUERY_OFFLINE_MODE=true`, models load **strictly** from local files and all external network calls are disabled.

---

## 3. How to Download Models

Run the download orchestrator with the appropriate profile:

```powershell
# Download core models (buildings, land cover, water):
python scripts/download_satquery_models.py --core

# Download optional models (secondary building detector, Grounding DINO, SAM2):
python scripts/download_satquery_models.py --optional

# Download all models:
python scripts/download_satquery_models.py --all

# Download a specific model:
python scripts/download_satquery_models.py --model dinov3s-buildings
```

---

## 4. How to Check Model Status

Inspect all models, sizes, paths, and readiness:

```powershell
python scripts/download_satquery_models.py --status
```

Example output:
```text
SatQuery Model Status
------------------------------------------------
[READY] building_primary
        geobase/dinov3s-buildings
        Path: D:\Sat1\models\buildings\dinov3s-buildings
        Size: 852.3 MB

[READY] land_rgb
        IGNF/FLAIR-INC_rgb_15cl_resnet34-deeplabv3
        Path: D:\Sat1\models\land_cover\flair-rgb-15cl-deeplabv3
        Size: 100.3 MB
...
```

---

## 5. How to Verify Model Loadability

Run the lightweight smoke test suite:

```powershell
# File integrity and checksum verification:
python scripts/download_satquery_models.py --verify

# Lightweight runtime smoke tests (loads one model at a time on CPU):
python scripts/verify_satquery_models.py
```

---

## 6. How Backend Resolves Models

The backend utilizes the authoritative resolver:

```python
from app.services.model_registry import ModelRegistry

# Resolves to the local Path from models.yaml or default directory layout
checkpoint_path = ModelRegistry.get_path("dinov3s-buildings")
```

The resolver dynamically checks:
1. `D:\Sat1\models\manifests\models.yaml`
2. Pinned subdirectory paths under `SATQUERY_MODEL_DIR`
3. Fallback to existing legacy checkpoints
