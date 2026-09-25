---
license: cc-by-4.0
base_model: kshitijrajsharma/dinov3
datasets:
  - hotosm/vhr-building-segmentation
tags:
  - building-segmentation
  - remote-sensing
  - dinov3
  - fair
  - hotosm
---

# kshitijrajsharma/dinov3s-buildings

Building footprint segmentation for very-high-resolution aerial and satellite imagery. A frozen
DINOv3 ViT-S/16 backbone with a trainable UperNet decoder, used in HOTOSM fAIr to extract building
polygons from RGB imagery.

## Inputs and outputs

- Input: RGB image tile.
- Output: per-pixel building probability, thresholded and vectorised to building polygons.

## Inference

- Full-resolution continuous sliding window: 256 px window, 192 px stride.
- Default probability threshold: `0.4371`.
- Polygons are vectorised from the probability mask and simplified.

## Files

- `model.onnx`: self-contained inference graph (RGB tile -> building probability).
- `model.ckpt`: Lightning checkpoint for evaluation or further training.

## Backbone

DINOv3 ViT-S/16, frozen. The UperNet decoder and segmentation head are the trained parameters.

## Training data

Trained on [`hotosm/vhr-building-segmentation`](https://huggingface.co/datasets/hotosm/vhr-building-segmentation)
(CC-BY-4.0 / ODbL).

## License

CC-BY-4.0, following the training data. The DINOv3 backbone weights are subject to the separate
DINOv3 license from their original release.
