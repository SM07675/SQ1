---
title: SatQuery API
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.12"
short_description: Satellite and aerial imagery analysis API
---

SatQuery FastAPI service launched by the Gradio Space Python runner on port 7860. This custom Python use of the Gradio SDK is an unofficial Hugging Face workflow. Set `SATQUERY_CORS_ORIGINS` in Space Settings to the exact Vercel frontend origin, for example `https://satquery.vercel.app`.

Attach a read-write Storage Bucket at `/data`, then set `SATQUERY_ARTIFACT_DIR=/data/artifacts` and `SATQUERY_DATABASE_PATH=/data/artifacts/satquery.sqlite3` to preserve uploaded imagery, SQLite history, and analysis artifacts across Space restarts. Without those settings, the local `artifacts` directory is temporary.

ZeroGPU does not automatically accelerate this API. Its model calls are CPU-bound unless explicitly adapted to Hugging Face's ZeroGPU decorators; available CPU and memory may be insufficient for some analyses. Verify build logs and representative requests before relying on this deployment.

The `/health` endpoint checks service availability.

This API does not include user authentication. A public Space should be used with non-sensitive demonstration imagery only.
