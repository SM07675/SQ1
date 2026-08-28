#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
trap 'kill 0' EXIT
(cd "$project_root/backend" && .venv/bin/python -m uvicorn app.main:app --reload --port 8000) &
(cd "$project_root/frontend" && npm run dev) &
wait
