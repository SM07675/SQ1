"""Hugging Face Gradio SDK entry point for the existing FastAPI service.

The Gradio Space runner executes this file. SatQuery keeps its API routes so the
separate Vercel frontend can call the same endpoints without a UI rewrite.
"""

import os

import uvicorn


os.environ.setdefault("SATQUERY_ENV", "production")
os.environ.setdefault("SATQUERY_OFFLINE_MODE", "true")
os.environ.setdefault("SATQUERY_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
os.environ.setdefault("SATQUERY_ARTIFACT_DIR", "artifacts")
os.environ.setdefault("SATQUERY_DATABASE_PATH", "artifacts/satquery.sqlite3")

from app.main import app  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
