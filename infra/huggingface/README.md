---
title: SatQuery API
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.12.12"
short_description: Satellite and aerial imagery analysis API
---

SatQuery FastAPI service launched by the Gradio Space Python runner on port 7860. This custom Python use of the Gradio SDK is an unofficial Hugging Face workflow. Set `SATQUERY_CORS_ORIGINS` in Space Settings to the exact Vercel frontend origin, for example `https://satquery.vercel.app`.

Attach a read-write Storage Bucket at `/data`, then set `SATQUERY_ARTIFACT_DIR=/data/artifacts` and `SATQUERY_DATABASE_PATH=/data/artifacts/satquery.sqlite3` to preserve uploaded imagery, SQLite history, and analysis artifacts across Space restarts. Without those settings, the local `artifacts` directory is temporary.

This package requires regular CPU or dedicated GPU Space hardware. It cannot start on ZeroGPU: Hugging Face requires a registered `@spaces.GPU` Gradio event, while this package exposes FastAPI routes. Adding an unused decorated function would only hide the startup error; the API would remain outside ZeroGPU execution. ZeroGPU support requires a separate Gradio execution path and Vercel API integration, followed by representative model validation.

The `/health` endpoint checks service availability.

This API does not include user authentication. A public Space should be used with non-sensitive demonstration imagery only.
