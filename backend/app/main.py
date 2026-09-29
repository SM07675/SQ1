from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header
from pydantic import BaseModel, Field
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
    ChatCreateRequest,
    ChatDetailResponse,
    ChatImageResponse,
    ChatMessageResponse,
    ChatRenameRequest,
    ChatSummaryItem,
    DatasetSummary,
    QueryRequest,
    SpatialFeatureCollection,
    SpatialGeometryRecord,
)
from app.services.benchmarks import list_available_datasets, run_vrsbench_suite
from app.services.chat_service import answer_follow_up_query, generate_chat_title, get_topic_icon
from app.services.ingestion import NETCDF_SUFFIXES, build_model_tiles, prepare_source
from app.services.model_registry import registry
from app.services.integrated_analysis import analyze
from app.services.planner import is_optical_sar_query
from app.services.raster import inspect_raster, render_preview
from app.services.report import write_pdf_report
from app.services.report_storage import ReportStorage


ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", *NETCDF_SUFFIXES}
report_storage = ReportStorage(settings.report_bucket) if settings.report_bucket else None
repository = Repository(settings.database_path, report_storage=report_storage)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    repository.init()
    try:
        chats_dir = settings.artifact_dir / "chats"
        if chats_dir.exists():
            for tif in chats_dir.glob("*/*.tif"):
                prev = tif.with_name(f"{tif.stem}_preview.png")
                if not prev.exists():
                    try:
                        render_preview(tif, prev, max_size=1200)
                    except Exception:
                        pass
    except Exception:
        pass
    yield


app = FastAPI(
    title="SatQuery GeoProof API",
    version="1.0.0",
    description="Evidence-first spectral, temporal and optical-SAR geospatial analysis for SIH26167.",
    lifespan=lifespan,
)


@app.get("/api/v1/preview")
def get_raster_preview(path: str):
    """Dynamically serve or render a web-compatible PNG preview for any TIFF/GeoTIFF raster."""
    clean_path = path.lstrip("/").replace("artifacts/", "")
    resolved = settings.artifact_dir / clean_path
    if not resolved.exists():
        resolved = Path(path)
        if not resolved.exists():
            raise HTTPException(404, "Raster file not found")

    if resolved.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
        return FileResponse(resolved)

    preview_file = resolved.with_name(f"{resolved.stem}_preview.png")
    if not preview_file.exists():
        try:
            render_preview(resolved, preview_file, max_size=1200)
        except Exception as exc:
            raise HTTPException(500, f"Could not render preview: {exc}")
    return FileResponse(preview_file, media_type="image/png")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-chi-mauve-51.vercel.app",
        "https://frontend-git-main-unirohans-projects.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/artifacts/{result_id}/GeoProof_Report.pdf")
def get_artifact_pdf(result_id: str):
    """Dynamically serves or generates GeoProof_Report.pdf on the fly."""
    output_dir = settings.artifact_dir / result_id
    pdf_path = output_dir / "GeoProof_Report.pdf"
    if not pdf_path.exists():
        if report_storage is not None:
            report_storage.restore_pdf(result_id, pdf_path)
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
        if report_storage is not None:
            report_storage.restore_pdf(result_id, pdf_path)
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


class UploadSessionRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)


def _upload_blob(upload_id: str):
    if not settings.upload_bucket:
        raise HTTPException(503, "Large image uploads are not configured on this server.")
    try:
        object_id, suffix = upload_id.rsplit(".", 1)
        uuid.UUID(object_id)
    except (ValueError, AttributeError):
        raise HTTPException(422, "Invalid upload reference") from None
    if f".{suffix}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported upload format")
    from google.cloud import storage

    return storage.Client().bucket(settings.upload_bucket).blob(f"incoming/{upload_id}")


def _save_staged_upload(upload_id: str, destination: Path) -> None:
    blob = _upload_blob(upload_id)
    from google.api_core.exceptions import NotFound
    try:
        blob.reload()
    except NotFound:
        raise HTTPException(422, "Large image upload has not finished") from None
    if not blob.size or blob.size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB")
    blob.download_to_filename(str(destination))


@app.post("/api/v1/uploads/session")
def create_upload_session(payload: UploadSessionRequest, origin: str | None = Header(default=None)) -> dict[str, str]:
    if origin and origin not in {
        "https://frontend-chi-mauve-51.vercel.app",
        "https://frontend-git-main-unirohans-projects.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    }:
        raise HTTPException(403, "Upload origin is not allowed")
    suffix = Path(payload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file extension: {suffix}")
    if payload.size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB")
    upload_id = f"{uuid.uuid4()}{suffix}"
    blob = _upload_blob(upload_id)
    try:
        url = blob.create_resumable_upload_session(
            content_type="image/tiff" if suffix in {".tif", ".tiff"} else "application/octet-stream",
            size=payload.size,
            origin=origin,
            if_generation_match=0,
        )
    except Exception as exc:
        raise HTTPException(503, f"Could not start large image upload: {exc}") from exc
    return {"upload_id": upload_id, "upload_url": url}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "environment": settings.environment, "models": registry.capabilities()}


@app.get("/api/v1/models")
def model_capabilities() -> list[dict[str, object]]:
    from satquery_engine.services.model_registry import registry as integrated_registry

    current = registry.capabilities()
    surface = [
        item for item in integrated_registry.capabilities()
        if item.get("name") in {"building_satellite", "land_deepness", "land_flair_hub", "water_finetuned", "water_s2_surface"}
    ]
    return current + surface


@app.get("/api/v1/models/status")
def model_status() -> dict[str, object]:
    from satquery_engine.config import settings as integrated_settings

    return {"model_root": str(integrated_settings.model_dir), "models": model_capabilities()}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_images(
    query: Annotated[str, Form(min_length=2, max_length=1000)],
    image_a: Annotated[UploadFile | None, File(description="Primary GeoTIFF/TIFF or benchmark image")] = None,
    image_b: Annotated[UploadFile | None, File(description="Optional paired image")] = None,
    image_a_upload: Annotated[str | None, Form()] = None,
    image_b_upload: Annotated[str | None, Form()] = None,
    image_a_name: Annotated[str | None, Form()] = None,
    image_b_name: Annotated[str | None, Form()] = None,
    pair_type: Annotated[str, Form(pattern="^(auto|single|bi_temporal|optical_sar)$")] = "auto",
    netcdf_variable: Annotated[str | None, Form()] = None,
) -> AnalysisResponse:
    if (image_a is None) == (image_a_upload is None):
        raise HTTPException(422, "Provide exactly one primary image")
    if image_b is not None and image_b_upload is not None:
        raise HTTPException(422, "Provide only one second image")
    if pair_type in {"bi_temporal", "optical_sar"} and image_b is None and image_b_upload is None:
        raise HTTPException(422, f"pair_type={pair_type} requires image_b")

    result_id = str(uuid.uuid4())
    output_dir = settings.artifact_dir / result_id
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    try:
        first = output_dir / f"input_a{Path(image_a.filename or '.tif').suffix.lower() if image_a else Path(image_a_upload).suffix.lower()}"
        if image_a:
            await _save_upload(image_a, first)
        else:
            _save_staged_upload(image_a_upload, first)
        paths.append(first)
        if image_b is not None or image_b_upload is not None:
            second = output_dir / f"input_b{Path(image_b.filename or '.tif').suffix.lower() if image_b else Path(image_b_upload).suffix.lower()}"
            if image_b:
                await _save_upload(image_b, second)
            else:
                _save_staged_upload(image_b_upload, second)
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


# =========================================================================
# Persistent Conversation & Chat System Endpoints
# =========================================================================

@app.get("/api/v1/chats", response_model=list[ChatSummaryItem])
def list_chats(limit: int = 100) -> list[ChatSummaryItem]:
    chats = repository.list_chats(limit=limit)
    res = []
    for c in chats:
        icon = get_topic_icon(c["title"], c.get("last_message") or "")
        res.append(ChatSummaryItem(
            chat_id=c["chat_id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
            last_message=c.get("last_message"),
            message_count=c.get("message_count", 0),
            image_count=c.get("image_count", 0),
            icon=icon,
            metadata=c.get("metadata", {}),
        ))
    return res


@app.post("/api/v1/chats", response_model=ChatSummaryItem)
def create_chat(req: ChatCreateRequest) -> ChatSummaryItem:
    chat_id = req.chat_id or f"chat_{uuid.uuid4().hex[:12]}"
    title = req.title or "New Analysis"
    created = repository.create_chat(chat_id, title, req.metadata)
    icon = get_topic_icon(title)
    return ChatSummaryItem(
        chat_id=created["chat_id"],
        title=created["title"],
        created_at=created["created_at"],
        updated_at=created["updated_at"],
        icon=icon,
        metadata=created["metadata"],
    )


@app.get("/api/v1/chats/{chat_id}", response_model=ChatDetailResponse)
def get_chat(chat_id: str) -> ChatDetailResponse:
    chat = repository.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")

    db_images = repository.get_chat_images(chat_id)
    images = [
        ChatImageResponse(
            image_id=img["image_id"],
            chat_id=img["chat_id"],
            filename=img["filename"],
            url=img["url"],
            metadata=img.get("metadata", {}),
            created_at=img["created_at"],
        )
        for img in db_images
    ]

    db_messages = repository.get_chat_messages(chat_id)
    messages = []
    latest_result = None
    for msg in db_messages:
        res_obj = None
        if msg.get("result"):
            try:
                res_obj = AnalysisResponse.model_validate(msg["result"])
                latest_result = res_obj
            except Exception:
                pass
        messages.append(ChatMessageResponse(
            message_id=msg["message_id"],
            chat_id=msg["chat_id"],
            role=msg["role"],
            content=msg["content"],
            created_at=msg["created_at"],
            attachments=msg.get("attachments", []),
            result=res_obj,
        ))

    icon = get_topic_icon(chat["title"], messages[0].content if messages else "")
    return ChatDetailResponse(
        chat_id=chat["chat_id"],
        title=chat["title"],
        created_at=chat["created_at"],
        updated_at=chat["updated_at"],
        icon=icon,
        metadata=chat["metadata"],
        images=images,
        messages=messages,
        latest_result=latest_result,
    )


@app.patch("/api/v1/chats/{chat_id}", response_model=ChatSummaryItem)
def rename_chat(chat_id: str, req: ChatRenameRequest) -> ChatSummaryItem:
    chat = repository.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, f"Chat not found: {chat_id}")
    repository.update_chat(chat_id, title=req.title.strip())
    updated = repository.get_chat(chat_id)
    icon = get_topic_icon(updated["title"])
    return ChatSummaryItem(
        chat_id=updated["chat_id"],
        title=updated["title"],
        created_at=updated["created_at"],
        updated_at=updated["updated_at"],
        icon=icon,
        metadata=updated["metadata"],
    )


@app.delete("/api/v1/chats/{chat_id}")
def delete_chat(chat_id: str) -> dict[str, Any]:
    chat = repository.get_chat(chat_id)
    if chat is None:
        raise HTTPException(404, f"Chat not found: {chat_id}")

    chat_dir = settings.artifact_dir / "chats" / chat_id
    if chat_dir.exists():
        shutil.rmtree(chat_dir, ignore_errors=True)

    repository.delete_chat(chat_id)
    return {"status": "ok", "deleted": True, "deleted_chat_id": chat_id}


@app.post("/api/v1/chats/{chat_id}/messages", response_model=ChatMessageResponse)
async def post_chat_message(
    chat_id: str,
    query: Annotated[str, Form(min_length=1, max_length=1000)],
    image_a: Annotated[UploadFile | None, File()] = None,
    image_b: Annotated[UploadFile | None, File()] = None,
    image_a_upload: Annotated[str | None, Form()] = None,
    image_b_upload: Annotated[str | None, Form()] = None,
    image_a_name: Annotated[str | None, Form()] = None,
    image_b_name: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    pair_type: Annotated[str, Form(pattern="^(auto|single|bi_temporal|optical_sar)$")] = "auto",
    netcdf_variable: Annotated[str | None, Form()] = None,
) -> ChatMessageResponse:
    chat = repository.get_chat(chat_id)
    if chat is None:
        chat = repository.create_chat(chat_id, "New Analysis")

    if files and len(files) > 0:
        if image_a is None and len(files) >= 1:
            image_a = files[0]
        if image_b is None and len(files) >= 2:
            image_b = files[1]
    if image_a is not None and image_a_upload is not None:
        raise HTTPException(422, "Provide only one primary image")
    if image_b is not None and image_b_upload is not None:
        raise HTTPException(422, "Provide only one second image")
    if (image_b is not None or image_b_upload is not None) and image_a is None and image_a_upload is None:
        raise HTTPException(422, "Second image requires a primary image")

    chat_dir = settings.artifact_dir / "chats" / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)

    db_images = repository.get_chat_images(chat_id)
    stored_paths: list[Path] = [Path(img["storage_path"]) for img in db_images if Path(img["storage_path"]).exists()]
    user_msg_attachments: list[dict[str, Any]] = []

    # CASE 1: New image(s) provided in message
    if image_a is not None or image_a_upload is not None:
        ext_a = Path(image_a.filename or ".png").suffix.lower() if image_a else Path(image_a_upload).suffix.lower()
        img_a_path = chat_dir / f"image_1{ext_a}"
        if image_a:
            await _save_upload(image_a, img_a_path)
        else:
            _save_staged_upload(image_a_upload, img_a_path)
        img_a_url = f"/artifacts/chats/{chat_id}/image_1{ext_a}"
        preview_a_url = img_a_url
        if ext_a in (".tif", ".tiff"):
            prev_a_path = chat_dir / "image_1_preview.png"
            try:
                render_preview(img_a_path, prev_a_path, max_size=1200)
                preview_a_url = f"/artifacts/chats/{chat_id}/image_1_preview.png"
            except Exception:
                pass
        img_id_1 = str(uuid.uuid4())
        name_a = image_a.filename if image_a else image_a_name or image_a_upload
        repository.add_chat_image(img_id_1, chat_id, name_a or "image_1", img_a_path, img_a_url)
        stored_paths = [img_a_path]
        user_msg_attachments.append({"name": name_a, "url": img_a_url, "preview_url": preview_a_url, "type": "image"})

        if image_b is not None or image_b_upload is not None:
            ext_b = Path(image_b.filename or ".png").suffix.lower() if image_b else Path(image_b_upload).suffix.lower()
            img_b_path = chat_dir / f"image_2{ext_b}"
            if image_b:
                await _save_upload(image_b, img_b_path)
            else:
                _save_staged_upload(image_b_upload, img_b_path)
            img_b_url = f"/artifacts/chats/{chat_id}/image_2{ext_b}"
            preview_b_url = img_b_url
            if ext_b in (".tif", ".tiff"):
                prev_b_path = chat_dir / "image_2_preview.png"
                try:
                    render_preview(img_b_path, prev_b_path, max_size=1200)
                    preview_b_url = f"/artifacts/chats/{chat_id}/image_2_preview.png"
                except Exception:
                    pass
            img_id_2 = str(uuid.uuid4())
            name_b = image_b.filename if image_b else image_b_name or image_b_upload
            repository.add_chat_image(img_id_2, chat_id, name_b or "image_2", img_b_path, img_b_url)
            stored_paths.append(img_b_path)
            user_msg_attachments.append({"name": name_b, "url": img_b_url, "preview_url": preview_b_url, "type": "image"})

        # Record user message
        user_msg_id = str(uuid.uuid4())
        repository.add_chat_message(user_msg_id, chat_id, "user", query, attachments=user_msg_attachments)

        # Run pipeline
        result_id = str(uuid.uuid4())
        output_dir = settings.artifact_dir / result_id
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_paths = [
            prepare_source(path, output_dir / f"prepared_{index + 1}", netcdf_variable).raster_path
            for index, path in enumerate(stored_paths)
        ]

        effective_pair_type = pair_type
        if len(stored_paths) == 2 and pair_type in ("single", "auto"):
            effective_pair_type = "optical_sar" if is_optical_sar_query(query) else "bi_temporal"

        response = await analyze(
            result_id=result_id,
            query=query,
            pair_type=effective_pair_type,
            image_paths=prepared_paths,
            output_dir=output_dir,
        )
        repository.save_result(result_id, response.model_dump(mode="json"), response.generated_at, output_dir=output_dir)

        # Update chat title if still default
        if chat["title"] in ("New Analysis", "Untitled"):
            new_title = generate_chat_title(query, pair_type, response)
            repository.update_chat(chat_id, title=new_title)

        # Record assistant message
        asst_msg_id = str(uuid.uuid4())
        assistant_content = response.verdict.answer
        if response.summary and response.summary.explanation:
            assistant_content = response.summary.explanation

        asst_msg = repository.add_chat_message(
            asst_msg_id,
            chat_id,
            "assistant",
            assistant_content,
            attachments=[],
            result=response.model_dump(mode="json"),
        )
        return ChatMessageResponse(
            message_id=asst_msg["message_id"],
            chat_id=chat_id,
            role="assistant",
            content=asst_msg["content"],
            created_at=asst_msg["created_at"],
            attachments=[],
            result=response,
        )

    # CASE 2A: No image provided and conversation has no imagery yet -> Auto-bind sample satellite imagery
    if not stored_paths:
        workspace_root = Path(__file__).resolve().parent.parent.parent
        data_dir = workspace_root / "data"
        urban_sample = workspace_root / "urban_coastal_tmp.png"
        demo_optical = data_dir / "demo_optical_water.tif"
        demo_sar = data_dir / "demo_sar_water.tif"
        demo_before = data_dir / "demo_before_multispectral.tif"
        demo_after = data_dir / "demo_after_multispectral.tif"

        is_bitemporal = (
            pair_type == "bi_temporal" or
            any(k in query.lower() for k in ("compare", "change", "changed", "before", "after", "difference", "temporal"))
        )
        is_sar = (
            pair_type == "optical_sar" or
            any(k in query.lower() for k in ("sar", "radar", "fusion", "croma"))
        )

        sample_pairs: list[tuple[str, Path]] = []
        effective_pair_type = pair_type
        if is_bitemporal and demo_before.exists() and demo_after.exists():
            sample_pairs = [("demo_before.tif", demo_before), ("demo_after.tif", demo_after)]
            effective_pair_type = "bi_temporal"
        elif is_sar and demo_optical.exists() and demo_sar.exists():
            sample_pairs = [("demo_optical.tif", demo_optical), ("demo_sar.tif", demo_sar)]
            effective_pair_type = "optical_sar"
        elif urban_sample.exists():
            sample_pairs = [("coastal_urban_sample.png", urban_sample)]
            effective_pair_type = "single"
        elif demo_optical.exists():
            sample_pairs = [("demo_optical_water.tif", demo_optical)]
            effective_pair_type = "single"

        if sample_pairs:
            for idx, (filename, src_p) in enumerate(sample_pairs):
                ext = src_p.suffix.lower()
                dest_p = chat_dir / f"image_{idx + 1}{ext}"
                shutil.copyfile(src_p, dest_p)
                img_url = f"/artifacts/chats/{chat_id}/image_{idx + 1}{ext}"
                preview_url = img_url
                if ext in (".tif", ".tiff"):
                    prev_p = chat_dir / f"image_{idx + 1}_preview.png"
                    try:
                        render_preview(dest_p, prev_p, max_size=1200)
                        preview_url = f"/artifacts/chats/{chat_id}/image_{idx + 1}_preview.png"
                    except Exception:
                        pass
                img_id = str(uuid.uuid4())
                repository.add_chat_image(img_id, chat_id, filename, dest_p, img_url)
                stored_paths.append(dest_p)
                user_msg_attachments.append({"name": filename, "url": img_url, "preview_url": preview_url, "type": "image"})

            user_msg_id = str(uuid.uuid4())
            repository.add_chat_message(user_msg_id, chat_id, "user", query, attachments=user_msg_attachments)

            result_id = str(uuid.uuid4())
            output_dir = settings.artifact_dir / result_id
            output_dir.mkdir(parents=True, exist_ok=True)
            prepared_paths = [
                prepare_source(path, output_dir / f"prepared_{index + 1}", netcdf_variable).raster_path
                for index, path in enumerate(stored_paths)
            ]
            response = await analyze(
                result_id=result_id,
                query=query,
                pair_type=effective_pair_type,
                image_paths=prepared_paths,
                output_dir=output_dir,
            )
            repository.save_result(result_id, response.model_dump(mode="json"), response.generated_at, output_dir=output_dir)

            if chat["title"] in ("New Analysis", "Untitled"):
                new_title = generate_chat_title(query, effective_pair_type, response)
                repository.update_chat(chat_id, title=new_title)

            asst_msg_id = str(uuid.uuid4())
            assistant_content = response.verdict.answer
            if response.summary and response.summary.explanation:
                assistant_content = response.summary.explanation

            asst_msg = repository.add_chat_message(
                asst_msg_id,
                chat_id,
                "assistant",
                assistant_content,
                attachments=[],
                result=response.model_dump(mode="json"),
            )
            return ChatMessageResponse(
                message_id=asst_msg["message_id"],
                chat_id=chat_id,
                role="assistant",
                content=asst_msg["content"],
                created_at=asst_msg["created_at"],
                attachments=[],
                result=response,
            )

    # CASE 2B: Follow-up question on existing image(s)
    user_msg_id = str(uuid.uuid4())
    repository.add_chat_message(user_msg_id, chat_id, "user", query, attachments=[])

    # Fetch latest result from previous messages
    db_messages = repository.get_chat_messages(chat_id)
    latest_result_dict = None
    for m in reversed(db_messages):
        if m.get("result"):
            latest_result_dict = m["result"]
            break

    # Determine answer or reanalysis requirement
    image_path_strs = [str(p) for p in stored_paths]
    answer_text, needs_reanalysis = answer_follow_up_query(
        query=query,
        latest_result=latest_result_dict,
        image_paths=image_path_strs,
    )

    result_to_attach: dict[str, Any] | None = None
    parsed_response: AnalysisResponse | None = None

    if needs_reanalysis and stored_paths:
        result_id = str(uuid.uuid4())
        output_dir = settings.artifact_dir / result_id
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared_paths = [
            prepare_source(path, output_dir / f"prepared_{index + 1}", netcdf_variable).raster_path
            for index, path in enumerate(stored_paths)
        ]
        effective_pair_type = pair_type
        if len(stored_paths) == 2 and pair_type in ("single", "auto"):
            effective_pair_type = "optical_sar" if is_optical_sar_query(query) else "bi_temporal"

        new_response = await analyze(
            result_id=result_id,
            query=query,
            pair_type=effective_pair_type,
            image_paths=prepared_paths,
            output_dir=output_dir,
        )
        repository.save_result(result_id, new_response.model_dump(mode="json"), new_response.generated_at, output_dir=output_dir)
        result_to_attach = new_response.model_dump(mode="json")
        parsed_response = new_response
        answer_text = new_response.verdict.answer
        if new_response.summary and new_response.summary.explanation:
            answer_text = new_response.summary.explanation

    asst_msg_id = str(uuid.uuid4())
    asst_msg = repository.add_chat_message(
        asst_msg_id,
        chat_id,
        "assistant",
        answer_text,
        attachments=[],
        result=result_to_attach,
    )

    return ChatMessageResponse(
        message_id=asst_msg["message_id"],
        chat_id=chat_id,
        role="assistant",
        content=asst_msg["content"],
        created_at=asst_msg["created_at"],
        attachments=[],
        result=parsed_response,
    )
