# Model integration contracts

The runnable MVP keeps heavy models in separate GPU services. This prevents the
FastAPI process from loading every checkpoint and allows lazy routing.

## Provider-neutral VLM contract

Set `SATQUERY_VLM_ENDPOINT` to any managed LLaVA-Geospatial, BLIP-2 or other
multimodal service. SatQuery sends `POST {endpoint}/infer` as multipart form
data:

- `request`: JSON containing query, validated raster metadata, tile count and
  the required response schema.
- `images`: up to 16 normalized PNG model tiles.
- `Authorization: Bearer ...`: included only when
  `SATQUERY_VLM_API_KEY` is configured.

The response is validated and must contain `answer` and a 0..1 `confidence`.
Optional `token_logprobs`, `supports_claim`, `boxes` and `metrics` are
preserved as evidence. Mean token probability contributes to the quantitative
GeoProof confidence breakdown.

## Endpoint contract

Each configured model base URL must expose:

```http
POST /infer
Content-Type: application/json
```

Successful responses use this shape:

```json
{
  "answer": "Evidence-grounded model output",
  "confidence": 0.74,
  "evidence": [],
  "metrics": {}
}
```

The API never accepts model-generated area, distance, count or coordinates as
trusted metrics. Those values must be recomputed by Rasterio/PostGIS and linked
to the evidence manifest.

## EarthDial

- Official code: https://github.com/hiyamdebary/EarthDial
- Checkpoints: `EarthDial_4B_RGB` and `EarthDial_4B_MS`
- Configure: `SATQUERY_EARTHDIAL_ENDPOINT=http://earthdial:9001`
- Inputs: `query`, `image_paths`
- Purpose: VQA, captioning, grounding and evidence explanation.

Benchmark both EarthDial checkpoints on the same VRSBench/RSVQA validation
subset. Use the better optical checkpoint for single-image tasks and the MS
checkpoint for multispectral/SAR workflows.

## CROMA

- Official code: https://github.com/antofuller/croma
- Configure: `SATQUERY_CROMA_ENDPOINT=http://croma:9002`
- Inputs: `optical_path`, `sar_path`
- Purpose: independent Sentinel-1/Sentinel-2 representation and agreement evidence.

CROMA expects correctly prepared Sentinel-1 and Sentinel-2 channels. Do not send
an arbitrary grayscale image and label it SAR evidence.

## TinyCD / Open-CD

- Open-CD: https://github.com/likyoo/open-cd
- TinyCD: https://github.com/AndreaCodegoni/Tiny_model_4_CD
- Configure: `SATQUERY_CHANGE_ENDPOINT=http://change:9003`
- Inputs: `before_path`, `after_path`

The included deterministic fallback is intentionally generic. It proves the
end-to-end mask, polygon, area, audit and abstention flow while the trained
change checkpoint is being integrated.

## RemoteCLIP

- Official code: https://github.com/ChenDelong1999/RemoteCLIP
- Configure: `SATQUERY_REMOTECLIP_ENDPOINT=http://remoteclip:9004`
- Inputs: `query`, `tile_paths`
- Purpose: semantic tile retrieval before the VLM sees a large scene.

## Promotion rule

No model becomes core merely because its published paper reports a higher
number. Promote it only after same-split evaluation on the team's frozen test
harness and after measuring latency and GPU memory on the deployment machine.
