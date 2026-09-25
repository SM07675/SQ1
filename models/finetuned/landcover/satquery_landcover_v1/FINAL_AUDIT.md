# Final implementation audit

Date: 2026-09-22

## Preserved assets

No existing weight, metric, image, or configuration file was overwritten. Because the weights were only opened read-only, no backup copy was necessary.

Checkpoint SHA-256 values at audit time:

- RGB land cover `best_checkpoint.pt`: `20d1fe299bc28f37d9ae7b3bcd13d8995eb8d2f5fe64d9b0b03a97a12337df49`
- RGB buildings sibling checkpoint: `93e065da2f6f12644ae0543bc5ec434fdb66d53b7a41445793b71fb9568809a2`
- Sentinel-2 water sibling checkpoint: `e93903a19a09b98d9d8c2ac3f9973491f03359c679e8fc5cda3c9c8432b956d5`

## Added files

- `satquery/registry.py`: registry parsing, sensor detection, exact band/order/scale validation.
- `satquery/preprocess.py`: RGB normalization and documented Sentinel-2 backbone plus NDWI, MNDWI, NDVI, AWEI-shadow, brightness, and cirrus features.
- `satquery/models.py`: cached RGB land-cover, Satlas building, and Satlas water loaders with AMP/CPU execution.
- `satquery/tiling.py`: edge-safe tiles and Hann-weighted mosaic blending.
- `satquery/postprocess.py`: filtered global, boundary-aware watershed with compact instance IDs.
- `satquery/fusion.py`: explicit unknown/water/building priority and ambiguity handling.
- `satquery/geoio.py`: CRS/transform-preserving rasters and building GeoJSON.
- `satquery/inference.py` and `satquery/cli.py`: unified pipeline and command line.
- `satquery/evaluation.py`: global plus per-scene metric primitives and failure visualization export.
- `satquery/training.py`: geographic leakage guard, combined loss, deterministic seeding, resume checkpoints, calibration, and ZIP export helpers.
- `model_registry.json`, `pyproject.toml`, `requirements.txt`, `README.md`, `MODEL_CARD.md`, Colab notebook, and 14 regression tests.

## Models and measured metrics

These values are copied from the supplied metrics files; they were not fabricated or recomputed during this audit.

- RGB land cover: U-Net/MiT-B2, FLAIR-1 official geographic domains, toy mode (207 train and 43 validation patches). Best validation mIoU `0.263760`; macro F1 `0.372335`. No held-out test or target-domain result is present.
- Buildings: SatlasPretrain `Aerial_SwinB_SI`, SpaceNet 2 train Vegas/Shanghai, validation Paris, test Khartoum. Test semantic IoU `0.564029`, instance precision/recall/F1@0.5 `0.268991/0.384752/0.316623`, matched IoU `0.703252`, count MAE `13.6167`, relative count error `0.698804`.
- Water: SatlasPretrain `Sentinel2_SwinB_SI_MS` with explicit spectral branch, Sen1Floods11 event-exclusive test on Bolivia and Paraguay. Test IoU `0.715703`, precision `0.920903`, recall `0.762580`, F1 `0.834297`, specificity `0.993137`, dark-land/shadow FPR `0.005988`.
- Sentinel-2 land cover: no trained checkpoint and therefore no metrics. The registry marks it unavailable.

## Verification performed

- `pytest`: 15 passed. Coverage includes band/order/scale rejection, tiling and padding, seam-free Hann blending, deterministic output, global overlap deduplication, touching-building separation, nodata/all-unknown behavior, fusion priority, shadow-versus-water gating, CRS/transform/dimensions/mask preservation, resolution validation, and evaluation sanity checks.
- Python byte compilation passed for package and tests.
- Colab notebook parsed as notebook format 4 with 11 cells.
- Building checkpoint loaded with `strict=True`: 89,698,146 parameters.
- Water checkpoint loaded with `strict=True`: 89,832,290 parameters.
- The RGB land-cover checkpoint architecture is recorded as `segmentation_models_pytorch.Unet(encoder_name='mit_b2', classes=15)`; strict loading was not run in the local audit environment because that optional package is not installed there. Setup installs it before inference.

## Remaining work before an accuracy claim

1. Run the Colab notebook in `full` mode after adding geographically separated road/impervious pixel labels; DFC2020 alone cannot validate that independent class.
2. Copy the exported best Sentinel-2 checkpoint into this bundle, set its registry path and `available` flag, then run `python -m satquery.cli ... --require-all-models` on representative 13-band scenes.
3. Evaluate all three tasks on held-out target geographies and sensors, report by scene/geography, and generate the required categorized worst-failure panels.
4. Retune thresholds only on validation scenes. Do not tune on test or deployment imagery.

Until those steps are complete, the implementation is structurally verified but the overall unified pipeline is not validated for production accuracy.
