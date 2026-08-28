# Architecture

```text
React UI
  -> FastAPI upload/API
  -> SQLite asset/result registry
  -> GeoTIFF/NetCDF preparation
  -> GeoTIFF compatibility and quality gate
  -> Overlapping normalized model tiles
  -> Typed query planner
  -> Deterministic spectral and temporal tools
  -> Optional lazy model/tool registry
  -> Rasterio measurement and GeoJSON polygonization
  -> GeoProof verification and abstention
  -> PDF report + evidence manifest + PNG/GeoJSON artifacts
  -> persistent result retrieval
```

## Trust boundary

The VLM can interpret and explain. Specialist models can produce detections,
embeddings or masks. Only deterministic geospatial code can produce trusted
area, distance, count and coordinates.

## Current v0.2 MVP

- Working: upload validation, preview generation, task routing, generic
  bi-temporal mask, NDVI/NDWI/NDBI analysis, class-specific spectral change,
  optical-SAR water agreement, polygonization, square-metre area, evidence
  manifest, PDF report, quality checks, abstention and React interface.
- Adapter-ready: EarthDial, CROMA, RemoteCLIP and external Open-CD inference.
- Scaffolded: PostGIS schema and container.
- Next: STAC ingestion, tile-level retrieval, calibrated confidence, real model
  services and benchmark dashboard.

## Evidence hierarchy

1. Generic radiometric change proves that pixel values changed, not what changed.
2. NDVI, NDWI and NDBI provide explainable vegetation, water and built-up
   proxies when correctly named bands are present.
3. Optical-SAR agreement provides independent cross-sensor confirmation for
   supported deterministic workflows.
4. EarthDial, CROMA and trained change models provide learned evidence after
   their services are configured.
5. GeoProof preserves the producer and limitations of every claim instead of
   silently converting a heuristic into a model prediction.

## Working data flow

```text
Query + GeoTIFF(s)
  -> typed plan
  -> metadata/alignment gate
  -> RGB previews
  -> radiometric change (paired time series)
  -> NDVI/NDWI/NDBI class evidence (when bands exist)
  -> optical-SAR agreement (supported water workflow)
  -> optional specialist model evidence
  -> affine-transform polygonization and equal-area measurement
  -> evidence sufficiency + confidence + safe abstention
  -> dashboard layers, GeoJSON, JSON manifest and PDF report
```
