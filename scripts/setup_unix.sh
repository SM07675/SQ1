#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -e "backend[dev]"
backend/.venv/bin/python scripts/make_demo_data.py
(cd frontend && npm install)
echo "Setup complete. Run: bash scripts/run_unix.sh"

