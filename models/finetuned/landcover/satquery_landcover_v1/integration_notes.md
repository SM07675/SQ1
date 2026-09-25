# SatQuery land-cover integration

Copy this directory to:

`D:\Sat1\models\finetuned\landcover\satquery_landcover_v1`

## Contract

- Task: multiclass dense semantic segmentation; not scene classification.
- Input: RGB VHR optical/aerial imagery. Sentinel-1 SAR and raw multispectral Sentinel-2 require separate models/adapters.
- Preprocess: divide uint8 RGB by 255, then normalize with mean `[0.4535237052059657, 0.4693471165828942, 0.4352777167252762]` and std `[0.21047783363935432, 0.18425352096507464, 0.18222531962736283]`.
- Tile: 512×512. For large rasters use overlap (recommended 64 px), average overlapping logits, then softmax + argmax.
- Output train IDs and names are defined only by `class_map.json`; do not hard-code label order in the backend.
- Confidence: multiclass argmax requires no threshold. An optional minimum confidence may mark uncertain output as 255 without relabeling it.
- Load: instantiate `segmentation_models_pytorch.Unet(encoder_name='mit_b2', in_channels=3, classes=15, activation=None)`, then load `checkpoint['model']`.
- Preserve source georeferencing when writing masks/GeoJSON. This bundle does not infer map scale or CRS from a plain PNG/JPEG.

## API suggestion

Return `class_id`, `class_name`, pixel count/area (only when ground sampling distance is known), confidence summary, and artifact paths for mask/overlay/GeoJSON. Include model version `satquery_landcover_v1` in every response.
