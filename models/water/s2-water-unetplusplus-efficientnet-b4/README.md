---
license: cc-by-4.0
library_name: segmentation-models-pytorch
tags:
  - remote-sensing
  - sentinel-2
  - water-detection
  - semantic-segmentation
  - earth-observation
  - unetplusplus
  - efficientnet
datasets:
  - giswqs/s2-water-dataset
metrics:
  - iou
  - dice
model-index:
  - name: s2-water-unetplusplus-efficientnet-b4
    results:
      - task:
          type: image-segmentation
          name: Semantic Segmentation
        dataset:
          name: Earth Surface Water Dataset (Sentinel-2)
          type: giswqs/s2-water-dataset
        metrics:
          - type: iou
            value: 0.9635
            name: Validation IoU (full scenes)
          - type: dice
            value: 0.9814
            name: Validation Dice (full scenes)
          - type: iou
            value: 0.9863
            name: Best Tile-level Val IoU
---

# Surface Water Detection — EfficientNet-B4 + UNet++

A semantic segmentation model for detecting surface water from **Sentinel-2** multispectral satellite imagery, trained on the [Earth Surface Water Dataset](https://zenodo.org/records/5205674).

## Model Description

| Property | Value |
|----------|-------|
| Architecture | UNet++ |
| Encoder | EfficientNet-B4 (ImageNet pretrained) |
| Input | 6-band Sentinel-2 (B2, B3, B4, B8, B11, B12) |
| Input Size | 512 × 512 tiles |
| Output Classes | 2 (Background, Water) |
| Parameters | ~19M |
| Framework | segmentation-models-pytorch + PyTorch Lightning |

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | [Earth Surface Water Dataset](https://huggingface.co/datasets/giswqs/s2-water-dataset) |
| Training Tiles | 1,581 (from 64 scenes, 512×512, stride 128) |
| Validation Tiles | 396 |
| Epochs | 50 |
| Batch Size | 8 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Optimizer | Adam |
| Loss | Cross-Entropy |

## Performance

### Tile-level Validation (during training)

| Metric | Value |
|--------|-------|
| Best Val IoU | **0.9863** |
| Best Val Loss | 0.0112 |
| Train IoU (final) | 0.989 |

### Full-scene Validation (31 scenes, sliding window 512×512, overlap 256)

| Metric | Value |
|--------|-------|
| IoU | **0.9635** |
| Dice | **0.9814** |

## Usage

### Installation

```bash
pip install geoai-py timm segmentation-models-pytorch
```

### Inference from HuggingFace Hub

```python
import geoai

geoai.timm_segmentation_from_hub(
    input_path="sentinel2_scene.tif",
    output_path="water_mask.tif",
    repo_id="giswqs/s2-water-unetplusplus-efficientnet-b4",
    window_size=512,
    overlap=256,
    batch_size=4,
)
```

### Input Format

The model expects 6-band Sentinel-2 L2A GeoTIFF images:

| Band Index | Sentinel-2 Band | Wavelength (nm) | Resolution |
|------------|----------------|-----------------|------------|
| 1 | B2 (Blue) | 490 | 10m |
| 2 | B3 (Green) | 560 | 10m |
| 3 | B4 (Red) | 665 | 10m |
| 4 | B8 (NIR) | 842 | 10m |
| 5 | B11 (SWIR1) | 1610 | 20m |
| 6 | B12 (SWIR2) | 2190 | 20m |

## Dataset

The [Earth Surface Water Dataset](https://huggingface.co/datasets/giswqs/s2-water-dataset) contains 95 globally distributed Sentinel-2 scenes with binary water masks:
- **Training**: 64 scenes
- **Validation**: 31 scenes
- **Source**: [Zenodo](https://zenodo.org/records/5205674)

## Citation

```bibtex
@ARTICLE{Luo2021-te,
  title     = "{An applicable and automatic method for earth surface water
               mapping based on multispectral images}",
  author    = "Luo, Xin and Tong, Xiaohua and Hu, Zhongwen",
  journal   = "International Journal of Applied Earth Observation and
               Geoinformation",
  publisher = "Elsevier BV",
  volume    =  103,
  pages     =  102472,
  year      =  2021,
  url       = "http://dx.doi.org/10.1016/j.jag.2021.102472",
  doi       = "10.1016/j.jag.2021.102472",
  issn      = "1569-8432,1872-826X",
}
```

## License

This model is released under the [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/), consistent with the training dataset license.
