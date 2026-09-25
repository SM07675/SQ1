# SatQuery land-cover pipeline

This bundle provides strict, geospatial inference for RGB very-high-resolution (VHR) imagery and 13-band Sentinel-2 L1C imagery. It preserves source CRS, affine transform, dimensions, and validity masks; blends overlapping tiles with a Hann window; separates buildings once on the final mosaic; and records every model and threshold in `model_registry.json`.

## Current model status

| Route | Model | Status | Honest validation status |
|---|---|---|---|
| RGB land cover | U-Net, MiT-B2 encoder, 15 FLAIR classes aggregated to 7 canonical classes | trained checkpoint included | bounded FLAIR toy run; validation mIoU 0.2638, not target-domain production validation |
| RGB buildings | SatlasPretrain `Aerial_SwinB_SI`, Swin-v2-Base + FPN, footprint/boundary heads | trained sibling bundle | held-out Khartoum instance F1@0.5 0.3166; counts are low-confidence under domain shift |
| Sentinel-2 water | SatlasPretrain `Sentinel2_SwinB_SI_MS` + six-feature spectral branch | trained sibling bundle | held-out Bolivia/Paraguay F1 0.8343; dark-land FPR 0.0060 |
| Sentinel-2 land cover | SatlasPretrain `Sentinel2_SwinB_SI_MS` + segmentation head | training workflow included, checkpoint absent | unavailable until full geographically separated training and road-label coverage are completed |

The pipeline does not substitute RGB data into a multispectral model or multispectral data into the RGB model. The missing Sentinel-2 land-cover checkpoint is reported explicitly; `--require-all-models` makes it fatal.

## Setup

Python 3.10–3.13 is recommended. From this directory:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

PyTorch installation can be selected separately from the official PyTorch installer when a particular CUDA build is needed. CPU fallback is automatic; CUDA inference uses mixed precision unless `--no-amp` is supplied.

## Input contract

RGB VHR input must be a three-band GeoTIFF in exact red, green, blue order, with values in 0–1 or 0–255. Band descriptions are checked when present. Individual-building counts are meaningful only for suitable VHR resolution; target-domain validation remains required.

Sentinel-2 input must contain all 13 L1C bands on one aligned grid in this exact order:

```text
B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
```

Values must be reflectance in 0–1 or digital numbers in approximately 0–10000. Because guessing band order is unsafe, a 13-band file must have exact band descriptions or be accompanied by `--band-order`. A cloud mask may be supplied with `--cloud-mask`; nonzero pixels are invalid and remain unknown. The mask must match the raster grid exactly.

## Inference

RGB:

```powershell
python -m satquery.cli input_rgb.tif outputs_rgb
```

Sentinel-2 with explicit band order:

```powershell
python -m satquery.cli input_s2.tif outputs_s2 --band-order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
```

Require a complete sensor route and fail if any registered model is missing:

```powershell
python -m satquery.cli input_s2.tif outputs_s2 --require-all-models --band-order B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
```

## Outputs

- `landcover_classes.tif`: canonical class IDs: 0 unknown, 1 built-up, 2 vegetation, 3 cropland, 4 bare soil, 5 water, 6 road/impervious.
- `probability_*.tif`: one GeoTIFF per canonical class.
- `landcover_confidence.tif` and `ambiguous_pixels.tif`: confidence and pixels withheld as unknown.
- `water_probability.tif` and `water_mask.tif`: specialist probability/mask for Sentinel-2; RGB uses the land-cover water channel and never invokes the multispectral model.
- `building_probability.tif`, `building_boundary_probability.tif`, and `building_instance_ids.tif`: RGB building outputs.
- `buildings.geojson`: compact, globally deduplicated building instances with scores.
- `inference_metadata.json`: source grid, model versions and hashes, thresholds, model-reported metrics, warnings, and output paths.

## Deterministic fusion

The land-cover output is the base. Invalid, nodata, cloud-masked, and ambiguous pixels stay unknown. Sentinel-2 water overrides the base only when its probability threshold and NDWI/MNDWI, NDVI, AWEI-shadow, brightness, and cirrus gates agree. Building footprints remain a separate instance layer and, when enabled, override land cover to built-up. Water never erases a confident building. All values are registry-controlled and must be retuned only on validation data.

## Evaluation and training

`satquery.evaluation` implements per-class land-cover IoU/precision/recall/F1, macro IoU/F1, balanced accuracy, confusion matrices, water specificity and shadow false-positive rate, and building semantic/instance/count metrics. Reports accept per-scene/geography groups, and `save_failure_visualizations` writes worst-case panels for shadows, clouds, dense/tiny buildings, mixed shorelines, and tile boundaries.

Open `notebooks/landcover_finetuning_colab.ipynb` in Colab. It retains Colab's compatible PyTorch, checks NumPy/SciPy binary compatibility, performs checksum-verified retrying/cached DFC2020 downloads, groups neighboring patches by geography, uses SatlasPretrain multispectral weights, applies geometric/radiometric augmentation, focal-weighted cross-entropy plus Dice, AMP, checkpoint resume, early stopping, and exports a ZIP. Set `MODE = 'full'` before final training. The notebook deliberately blocks production registration in its model card because DFC2020 has no independent road mask; add geographically separated road/impervious masks, rerun, and register the resulting checkpoint only after every required class has held-out support.

## Limitations and troubleshooting

- The existing RGB land-cover weights came from a small demonstration run. Do not infer production accuracy from successful execution.
- The current Sentinel-2 route can produce validated water output but not validated full land cover. Use `--require-all-models` when partial output is unacceptable.
- Optical models cannot see through thick cloud. Supply a cloud mask and consider Sentinel-1 fusion for all-weather flood work.
- Building performance transfers poorly across sensors, geographies, look angles, and resolutions; touching roofs, tiny structures, and shadows affect counts.
- If Colab reports a NumPy/SciPy ABI error after installing packages, restart the runtime and rerun all cells.
- If a band-order or scale error occurs, fix source metadata or pass the exact order; do not reorder by guesswork.

See `FINAL_AUDIT.md` for verification evidence and the precise remaining validation gaps.

