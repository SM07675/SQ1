from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.repository import Repository
from app.schemas import (
    AnalysisResponse,
    AnalysisRunRecord,
    AssetRecord,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    DatasetSummary,
    QueryRequest,
    SpatialFeatureCollection,
    SpatialGeometryRecord,
)
from app.services.benchmarks import list_available_datasets, run_vrsbench_suite
from app.services.ingestion import NETCDF_SUFFIXES, build_model_tiles, prepare_source
from app.services.model_registry import registry
from app.services.orchestrator import analyze
from app.services.raster import inspect_raster, render_preview
from app.services.report import write_pdf_report


ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", *NETCDF_SUFFIXES}
repository = Repository(settings.database_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    repository.init()
    yield


app = FastAPI(
    title="SatQuery GeoProof API",
    version="1.0.0",
    description="Evidence-first spectral, temporal and optical-SAR geospatial analysis for SIH26167.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/artifacts/{result_id}/GeoProof_Report.pdf")
def get_artifact_pdf(result_id: str):
    """Dynamically serves or generates GeoProof_Report.pdf on the fly."""
    output_dir = settings.artifact_dir / result_id
    pdf_path = output_dir / "GeoProof_Report.pdf"
    if not pdf_path.exists():
        payload = repository.get_result(result_id)
        if payload is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                pdf_path = write_pdf_report(output_dir, payload)
            except Exception as exc:
                raise HTTPException(500, f"Failed to generate PDF report: {exc}") from exc
    if pdf_path.exists():
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=GeoProof_Report_{result_id[:8]}.pdf"},
        )
    raise HTTPException(404, "Report not found")


@app.get("/api/v1/results/{result_id}/report")
@app.get("/api/v1/results/{result_id}/pdf")
def get_result_pdf(result_id: str):
    """Direct API endpoint for retrieving or generating the PDF audit report."""
    output_dir = settings.artifact_dir / result_id
    pdf_path = output_dir / "GeoProof_Report.pdf"
    if not pdf_path.exists():
        payload = repository.get_result(result_id)
        if payload is None:
            raise HTTPException(404, "Result not found")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            pdf_path = write_pdf_report(output_dir, payload)
        except Exception as exc:
            raise HTTPException(500, f"Failed to generate PDF report: {exc}") from exc
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=GeoProof_Report_{result_id[:8]}.pdf"},
    )


settings.artifact_dir.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=settings.artifact_dir), name="artifacts")
repository.init()


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    suffix = Path(upload.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file extension: {suffix}")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "environment": settings.environment, "models": registry.capabilities()}


@app.get("/api/v1/models")
def model_capabilities() -> list[dict[str, object]]:
    return registry.capabilities()


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_images(
    query: Annotated[str, Form(min_length=2, max_length=1000)],
    image_a: Annotated[UploadFile, File(description="Primary GeoTIFF/TIFF or benchmark image")],
    image_b: Annotated[UploadFile | None, File(description="Optional paired image")] = None,
    pair_type: Annotated[str, Form(pattern="^(auto|single|bi_temporal|optical_sar)$")] = "auto",
    netcdf_variable: Annotated[str | None, Form()] = None,
) -> AnalysisResponse:
    if pair_type in {"bi_temporal", "optical_sar"} and image_b is None:
        raise HTTPException(422, f"pair_type={pair_type} requires image_b")

    result_id = str(uuid.uuid4())
    output_dir = settings.artifact_dir / result_id
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    try:
        first = output_dir / f"input_a{Path(image_a.filename or '.tif').suffix.lower()}"
        await _save_upload(image_a, first)
        paths.append(first)
        if image_b is not None:
            second = output_dir / f"input_b{Path(image_b.filename or '.tif').suffix.lower()}"
            await _save_upload(image_b, second)
            paths.append(second)
        prepared_paths = [
            prepare_source(path, output_dir / f"prepared_{index + 1}", netcdf_variable).raster_path
            for index, path in enumerate(paths)
        ]
        response = await analyze(
            result_id=result_id,
            query=query,
            pair_type=pair_type,
            image_paths=prepared_paths,
            output_dir=output_dir,
        )
        repository.save_result(result_id, response.model_dump(mode="json"), response.generated_at, output_dir=output_dir)
        return response
    except HTTPException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(422, f"Analysis failed: {exc}") from exc


@app.post("/api/v1/assets", response_model=AssetRecord)
async def upload_asset(
    image: Annotated[UploadFile, File(description="GeoTIFF, NetCDF or raster image")],
    netcdf_variable: Annotated[str | None, Form()] = None,
) -> AssetRecord:
    asset_id = str(uuid.uuid4())
    asset_dir = settings.artifact_dir / "assets" / asset_id
    asset_dir.mkdir(parents=True, exist_ok=False)
    try:
        original = asset_dir / f"original{Path(image.filename or '.tif').suffix.lower()}"
        await _save_upload(image, original)
        prepared = prepare_source(original, asset_dir / "prepared", netcdf_variable)
        metadata = inspect_raster(prepared.raster_path).model_copy(update={"source_format": prepared.source_format})
        preview = asset_dir / "preview.png"
        render_preview(prepared.raster_path, preview)
        manifest_url = f"/artifacts/assets/{asset_id}/preprocess/tiles_manifest.json"
        preprocessing, _ = build_model_tiles(
            prepared.raster_path,
            asset_dir / "preprocess",
            public_manifest_url=manifest_url,
            source_format=prepared.source_format,
            selected_variable=prepared.selected_variable,
            available_variables=prepared.available_variables,
            tile_size=settings.tile_size,
            overlap=settings.tile_overlap,
            max_tiles=settings.max_model_tiles,
        )
        created_at = datetime.now(UTC).isoformat()
        record = AssetRecord(
            asset_id=asset_id,
            original_filename=image.filename or original.name,
            created_at=created_at,
            metadata=metadata,
            preprocessing=preprocessing,
            preview_url=f"/artifacts/assets/{asset_id}/preview.png",
        )
        repository.save_asset(asset_id, prepared.raster_path, record.model_dump(mode="json"), created_at)
        return record
    except Exception as exc:
        shutil.rmtree(asset_dir, ignore_errors=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(422, f"Asset ingestion failed: {exc}") from exc


@app.get("/api/v1/assets/{asset_id}", response_model=AssetRecord)
def get_asset(asset_id: str) -> AssetRecord:
    payload = repository.get_asset(asset_id)
    if payload is None:
        raise HTTPException(404, "Asset not found")
    payload.pop("_raster_path", None)
    return AssetRecord.model_validate(payload)


@app.post("/api/v1/query", response_model=AnalysisResponse)
async def query_assets(request: QueryRequest) -> AnalysisResponse:
    assets = []
    for asset_id in request.asset_ids:
        asset = repository.get_asset(asset_id)
        if asset is None:
            raise HTTPException(404, f"Asset not found: {asset_id}")
        assets.append(asset)
    if request.pair_type in {"bi_temporal", "optical_sar"} and len(assets) != 2:
        raise HTTPException(422, f"pair_type={request.pair_type} requires exactly two assets")
    result_id = str(uuid.uuid4())
    output_dir = settings.artifact_dir / result_id
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        response = await analyze(
            result_id=result_id,
            query=request.query,
            pair_type=request.pair_type,
            image_paths=[Path(asset["_raster_path"]) for asset in assets],
            output_dir=output_dir,
        )
        repository.save_result(result_id, response.model_dump(mode="json"), response.generated_at, output_dir=output_dir)
        return response
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(422, f"Query failed: {exc}") from exc


@app.get("/api/v1/results/{result_id}", response_model=AnalysisResponse)
def get_result(result_id: str) -> AnalysisResponse:
    payload = repository.get_result(result_id)
    if payload is None:
        raise HTTPException(404, "Result not found")
    return AnalysisResponse.model_validate(payload)


@app.get("/api/v1/benchmarks/datasets", response_model=list[DatasetSummary])
def get_benchmark_datasets() -> list[DatasetSummary]:
    return list_available_datasets()


@app.post("/api/v1/benchmarks/evaluate", response_model=BenchmarkRunResponse)
async def evaluate_benchmarks(request: BenchmarkRunRequest) -> BenchmarkRunResponse:
    try:
        return await run_vrsbench_suite(
            registry=registry,
            dataset_name=request.dataset_name,
            model_variant=request.model_variant,
            max_samples=request.max_samples,
        )
    except Exception as exc:
        raise HTTPException(500, f"Benchmark evaluation failed: {exc}") from exc


@app.get("/api/v1/spatial/geometries", response_model=list[SpatialGeometryRecord])
def get_spatial_geometries(
    result_id: str | None = None,
    evidence_kind: str | None = None,
    bbox: str | None = None,
    limit: int = 100,
) -> list[SpatialGeometryRecord]:
    parsed_bbox = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                parsed_bbox = (parts[0], parts[1], parts[2], parts[3])
        except Exception:
            raise HTTPException(400, "bbox must be in format 'min_x,min_y,max_x,max_y'")

    records = repository.query_geometries(
        result_id=result_id,
        evidence_kind=evidence_kind,
        bbox=parsed_bbox,
        limit=limit,
    )
    return [SpatialGeometryRecord.model_validate(r) for r in records]


@app.get("/api/v1/spatial/runs", response_model=list[AnalysisRunRecord])
def list_analysis_runs(
    verdict_status: str | None = None,
    task_type: str | None = None,
    limit: int = 50,
) -> list[AnalysisRunRecord]:
    runs = repository.list_analysis_runs(
        verdict_status=verdict_status,
        task_type=task_type,
        limit=limit,
    )
    return [AnalysisRunRecord.model_validate(r) for r in runs]


