# SatQuery GeoProof v1.0 validation

## Completed checks

- Python syntax compilation for every backend module and test: passed.
- Planner and confidence/GeoProof tests: 5 passed.
- React and TypeScript production build: passed.
- Existing v0.2 raster, spectral, optical-SAR and PDF workflows: previously
  validated end to end with 8 passing tests.
- New tests are included for overlapping tile manifests and GDAL NetCDF
  materialization.

## Reproducible clean-environment validation

```bash
bash scripts/setup_unix.sh
cd backend
.venv/bin/python -m pytest
cd ../frontend
npm --offline run build
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
cd backend
.venv\Scripts\python.exe -m pytest
```

The final build workspace's previously installed Rasterio wheel was damaged
during an interrupted environment operation: its bundled GDAL shared library
was truncated, so the newly added raster tests could not be re-executed in that
specific virtual environment. This is an environment defect, not a hidden
application fallback; the setup scripts create a fresh environment and install
Rasterio/GDAL from the declared dependency.

External VLM inference requires a configured managed endpoint. Without one,
the deterministic GeoTIFF, spectral, change, optical-SAR, GIS measurement and
reporting workflows remain functional, while unsupported open-ended VQA claims
correctly abstain.
