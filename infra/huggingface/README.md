---
title: SatQuery API
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: cpu-upgrade
short_description: Satellite and aerial imagery analysis API
---

SatQuery FastAPI service. Set `SATQUERY_CORS_ORIGINS` in Space Settings to the exact Vercel frontend origin, for example `https://satquery.vercel.app`.

Attach a read-write Storage Bucket at `/data` to preserve uploaded imagery, SQLite history, and analysis artifacts across Space restarts. Without a bucket, those files are temporary.

The `/health` endpoint checks service availability.

This API does not include user authentication. A public Space should be used with non-sensitive demonstration imagery only.
