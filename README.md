# SatQuery GeoProof v1.0 - SIH26167 Functional Prototype

SatQuery GeoProof is an evidence-first remote-sensing assistant for the
ISRO Smart India Hackathon problem statement SIH26167. It accepts GeoTIFF/TIFF
imagery and natural-language queries, selects a typed workflow, produces visual
and geospatial evidence, and blocks unsupported claims.

## Flagship working demonstrations

### 1. Built-up change with proof

Upload the generated before/after multispectral pair and ask:

```text
Has built-up area increased between these two dates?
```

The pipeline validates alignment, creates a generic change map, calculates
NDBI on both dates, maps NDBI-consistent built-up growth, polygonizes the
regions, calculates square-metre/hectare area, applies GeoProof and exports a
PDF investigation report.

### 2. Optical-SAR water cross-verification

Upload the generated optical and SAR pair and ask:

```text
Use optical and SAR evidence together to identify water-covered regions.
```

The pipeline combines optical NDWI evidence with SAR low-backscatter evidence,
creates a sensor-agreement map, calculates IoU, region area and confidence, and
keeps the deterministic heuristic clearly labelled.

### 3. Single-image water grounding

Upload the demo multispectral image and ask:

```text
Highlight the largest water body.
```

The result includes the NDWI layer, water mask, GeoJSON region, coordinates via
the raster affine transform, area, evidence record and PDF report.

## What works immediately

- Single or paired image upload with streaming size limits.
- GeoTIFF and NetCDF ingestion. NetCDF raster variables are discovered through
  GDAL and the requested/largest valid variable is converted for analysis.
- Overlapping 448x448 model-tile generation with percentile normalization,
  pixel windows, geographic bounds, CRS and a machine-readable tile manifest.
- GeoTIFF metadata, CRS, bounds, bands, resolution and NoData validation.
- Pair compatibility checks for CRS, dimensions, geographic overlap,
  resolution ratio and alignment score.
- Before/after swipe comparison and dynamic evidence layers.
- Deterministic intent routing for VQA, grounding, change and optical-SAR tasks.
- Generic bi-temporal radiometric change mask plus deterministic semantic
  spectral change using NDVI, NDWI and NDBI when named bands exist.
- Optical NDWI plus SAR low-backscatter agreement analysis for water.
- Change polygon GeoJSON, changed-pixel percentage and square-metre area.
- GeoProof evidence sufficiency, limitations and safe abstention.
- Observable execution trace and downloadable JSON evidence manifest.
- Automatically generated, visually verified GeoProof PDF report.
- Downloadable GeoJSON regions and all visual evidence layers.
- Colourful responsive React dashboard with model-status rail, KPIs, evidence
  ledger, confidence, limitations and report controls.
- Model-service adapters for EarthDial, CROMA, RemoteCLIP and Open-CD/TinyCD.
- A provider-neutral multipart VLM connector for managed LLaVA-Geospatial,
  BLIP-2 or another multimodal server.
- Persistent SQLite asset/result registry with separate upload, query and
  result-retrieval APIs.
- Quantitative confidence breakdown using input quality, alignment, evidence
  strength, ensemble agreement and optional token log probabilities.
- Pytest suite and Docker Compose stack.

## Important truth about this ZIP

Large model weights and datasets are not included. Learned open-ended VQA,
learned change detection and CROMA representation fusion require their model
checkpoints to be downloaded and exposed through the documented adapter
endpoints. The built-in spectral and sensor-agreement tools remain available,
but are labelled as deterministic proxies rather than trained-model outputs.

## Fast local setup

### Windows one-command setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

This opens the API and dashboard in separate terminal windows.

### Linux/macOS one-command setup

```bash
bash scripts/setup_unix.sh
bash scripts/run_unix.sh
```

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

API documentation: http://localhost:8000/docs

## API workflow

```text
POST /api/v1/assets
  multipart: image, optional netcdf_variable
  -> reusable asset_id + metadata + preview + tile manifest

POST /api/v1/query
  JSON: {"query":"Detect urban expansion between 2020 and 2025",
         "asset_ids":["before-id","after-id"],
         "pair_type":"bi_temporal"}
  -> complete AnalysisResponse

GET /api/v1/results/{result_id}
  -> persisted response with evidence, confidence, trace and artifacts

POST /api/v1/analyze
  -> convenience endpoint for upload and query in one request
```

### 2. Demo GeoTIFF pair

From the project root, with the backend environment active:

```bash
python scripts/make_demo_data.py
```

This creates:

- `data/demo_before_multispectral.tif`
- `data/demo_after_multispectral.tif`
- `data/demo_optical_water.tif`
- `data/demo_sar_water.tif`
- `data/DEMO_QUERIES.txt`

Every expected value is computed at runtime; no analytical metric is hard-coded.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and submit the demo pair with:

```text
What changed between these two dates, and where did the change occur?
```

For multispectral images with correctly named bands, class-specific vegetation,
water and built-up questions use explainable spectral proxies. If the required
bands are unavailable, the system abstains unless a configured domain model
provides adequate evidence.

## Run tests

```bash
cd backend
pytest
```

```bash
cd frontend
npm install
npm run build
```

## Docker setup

```bash
docker compose up --build
```

- Web application: http://localhost:8080
- API: http://localhost:8000
- PostGIS: localhost:5432

## Connect real models

Copy `.env.example` to `.env`, configure one or more endpoints, and follow
[`docs/MODEL_INTEGRATION.md`](docs/MODEL_INTEGRATION.md).

```env
SATQUERY_EARTHDIAL_ENDPOINT=http://localhost:9001
SATQUERY_CROMA_ENDPOINT=http://localhost:9002
SATQUERY_CHANGE_ENDPOINT=http://localhost:9003
SATQUERY_REMOTECLIP_ENDPOINT=http://localhost:9004
SATQUERY_VLM_ENDPOINT=http://localhost:9010
SATQUERY_VLM_API_KEY=optional-bearer-token
SATQUERY_VLM_MODEL_NAME=llava-geospatial
```

The generic VLM service must expose `POST /infer`, accept multipart fields
`request` (JSON) and repeated `images`, and return:

```json
{
  "answer": "Evidence-grounded answer",
  "confidence": 0.82,
  "token_logprobs": [-0.12, -0.08],
  "supports_claim": true,
  "boxes": [[0.1, 0.2, 0.4, 0.5]],
  "metrics": {}
}
```

The connector validates this contract, attaches tile metadata and incorporates
token probabilities into GeoProof confidence. No vendor secret is hard-coded.

## Project structure

```text
satquery-geoproof/
  backend/               FastAPI, geospatial pipeline, GeoProof and tests
  frontend/              React + TypeScript + Vite dashboard
  infra/postgres/        PostGIS schema
  scripts/               fixed demo data generator
  docs/                  architecture and model adapter contracts
  docker-compose.yml
```

## Production implementation gates (All 7 Complete)

1. [x] **Gate 1**: Validate EarthDial-4B-MS and EarthDial-4B-RGB on fixed VRSBench samples (`backend/app/services/benchmarks.py`, `/api/v1/benchmarks/evaluate`, CLI runner `scripts/evaluate_vrsbench.py`).
2. [x] **Gate 2**: TinyCD/Open-CD learned change detection witness with structural probability heatmaps, dual-witness consensus, and spatial polygonization (`backend/app/services/change_detector.py`).
3. [x] **Gate 3**: CROMA Sentinel-1 SAR linear-to-dB backscatter calibration and Sentinel-2 reflectance normalization with cross-sensor fusion (`backend/app/services/croma_pipeline.py`).
4. [x] **Gate 4**: RemoteCLIP hierarchical semantic tile retrieval for large GeoTIFFs with Spatial NMS and geographic ROI GeoJSON generation (`backend/app/services/remoteclip_retrieval.py`, `scripts/evaluate_remoteclip.py`).
5. [x] **Gate 5**: Persistent spatial database & evidence geometries in PostGIS/SQLite with spatial bounding box queries (`backend/app/repository.py`, `/api/v1/spatial/geometries`, `/api/v1/spatial/runs`).
6. [x] **Gate 6**: Multi-dataset frozen-split evaluation dashboard in React supporting VRSBench, RSVQA, and CDVQA with real-time audit tables and KPI cards (`frontend/src/App.tsx`).
7. [x] **Gate 7**: Calibrated confidence estimation engine with Platt temperature scaling, 95% confidence intervals, and Expected Calibration Error (ECE) metric (`backend/app/services/confidence.py`).


## Truth and scope

This ZIP is a strong, runnable hackathon MVP. It performs real raster, spectral,
spatial and reporting operations without model weights. EarthDial, CROMA,
RemoteCLIP and TinyCD/Open-CD remain optional external services because their
checkpoints are large and have separate setup/licensing requirements. The code
never claims those models ran when their endpoints are not configured.

## References

- FastAPI file upload: https://fastapi.tiangolo.com/tutorial/request-files/
- Rasterio windowed processing: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
- PostGIS area: https://postgis.net/docs/ST_Area.html
- Vite: https://vite.dev/guide/
- EarthDial: https://github.com/hiyamdebary/EarthDial
- CROMA: https://github.com/antofuller/croma
- Open-CD: https://github.com/likyoo/open-cd
- RemoteCLIP: https://github.com/ChenDelong1999/RemoteCLIP
