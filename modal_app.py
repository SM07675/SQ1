"""Modal serverless deployment for SatQuery GeoProof FastAPI service.

Deploy with:
    .\\backend\\.venv-integrated\\Scripts\\python.exe -m modal deploy modal_app.py
"""

import os
from pathlib import Path
import modal

CURRENT_DIR = Path(__file__).parent.resolve()

# 1. Define persistent storage for generated reports, SQLite database, and preview tiles
volume = modal.Volume.from_name("satquery-artifacts", create_if_missing=True)

# 2. Build the serverless container image with system libraries and dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1", "ffmpeg", "libsm6", "libxext6", "libgl1")
    .pip_install(
        "fastapi>=0.116,<1",
        "uvicorn[standard]>=0.35,<1",
        "python-multipart>=0.0.20,<1",
        "pydantic>=2.11,<3",
        "numpy>=2.2,<3",
        "rasterio>=1.4,<2",
        "Pillow>=11,<13",
        "shapely>=2.1,<3",
        "pyproj>=3.7,<4",
        "scipy>=1.13,<2",
        "httpx>=0.28,<1",
        "reportlab>=4.4,<5",
        "torch>=2.0",
        "torchvision>=0.15",
        "opencv-python-headless>=4.8",
        "python-dotenv>=1,<2",
        "PyYAML>=6,<7",
        "onnxruntime==1.29.0",
        "onnx>=1.17,<2",
        "scikit-image>=0.25,<1",
        "transformers==5.9.0",
        "segmentation-models-pytorch==0.4.0",
        "satlaspretrain-models==0.3.1",
        "safetensors>=0.4,<1",
    )
    .add_local_dir(
        str(CURRENT_DIR / "models"),
        remote_path="/root/models",
        copy=True,
        ignore=lambda p: ".cache" in str(p) or str(p).endswith(".metadata"),
    )
    .add_local_dir(
        str(CURRENT_DIR / "backend" / "satquery_engine"),
        remote_path="/root/satquery_engine",
        copy=True,
        ignore=lambda p: "__pycache__" in str(p) or str(p).endswith(".pyc"),
    )
    .add_local_dir(
        str(CURRENT_DIR / "backend" / "app"),
        remote_path="/root/app",
        copy=True,
        ignore=lambda p: "__pycache__" in str(p) or str(p).endswith(".pyc"),
    )
)

# 3. Create Modal App
app = modal.App("satquery-api")


@app.function(
    image=image,
    volumes={"/root/artifacts": volume},
    cpu=2.0,
    memory=8192,
    timeout=600,
    scaledown_window=300,
    max_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    import sys

    # Ensure /root is first in sys.path
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    os.environ["SATQUERY_ENV"] = "production"
    os.environ["SATQUERY_OFFLINE_MODE"] = "true"
    os.environ["SATQUERY_MODEL_DIR"] = "/root/models"
    os.environ["SATQUERY_ARTIFACT_DIR"] = "/root/artifacts"
    os.environ["SATQUERY_DATABASE_PATH"] = "/root/artifacts/satquery.sqlite3"
    os.environ["SATQUERY_DEVICE"] = "cpu"

    from app.main import app as web_app
    from starlette.middleware.cors import CORSMiddleware

    # Ensure CORS allows connections from any frontend origin (Vercel, custom domain, localhost)
    for m in web_app.user_middleware:
        if m.cls == CORSMiddleware:
            m.kwargs["allow_origin_regex"] = r".*"
            m.kwargs["allow_origins"] = []
    web_app.middleware_stack = None

    return web_app
