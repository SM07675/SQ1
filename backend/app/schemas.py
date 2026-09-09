from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SINGLE_VQA = "single_vqa"
    GROUNDING = "grounding"
    BI_TEMPORAL_CHANGE = "bi_temporal_change"
    OPTICAL_SAR = "optical_sar"
    SCENE_DESCRIPTION = "scene_description"
    SPECTRAL_INDEX = "spectral_index"
    LAND_COVER = "land_cover"
    OBJECT_IDENTIFICATION = "object_identification"
    VEGETATION_ANALYSIS = "vegetation_analysis"
    WATER_ANALYSIS = "water_analysis"
    BUILT_UP_ANALYSIS = "built_up_analysis"
    UNSUPPORTED = "unsupported"
    UNCLEAR = "unclear"


class VerdictStatus(str, Enum):
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class TaskPlan(BaseModel):
    task: TaskType
    application: str = "single_image"  # "single_image" | "bi_temporal" | "optical_sar" | "unsupported" | "unclear"
    specific_task: str = "scene_description"
    sub_tasks: list[str] = Field(default_factory=list)
    multi_intent: bool = False
    target: str | None = None
    asks_direction: bool = False
    tools: list[str]
    reason: str
    aoi_text: str | None = None
    years: list[int] = Field(default_factory=list)


class RasterMetadata(BaseModel):
    filename: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None
    bounds: list[float]
    resolution: list[float]
    nodata: float | None
    nodata_percent: float
    band_names: list[str]
    available_indices: list[str] = Field(default_factory=list)
    source_format: str = "raster"


class QualityReport(BaseModel):
    score: float = Field(ge=0, le=1)
    compatible: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    kind: str
    producer: str
    summary: str
    confidence: float = Field(ge=0, le=1)
    artifact_url: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    supports_claim: bool | None = None
    confidence_source: str = "producer_score"


class TraceStep(BaseModel):
    step: int
    component: str
    action: str
    status: str
    duration_ms: int
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    name: str
    url: str
    mime_type: str


class ConfidenceBreakdown(BaseModel):
    input_quality: float = Field(ge=0, le=1)
    spatial_alignment: float = Field(ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=1)
    ensemble_agreement: float = Field(ge=0, le=1)
    model_probability: float | None = Field(default=None, ge=0, le=1)
    warning_penalty: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)
    is_calibrated: bool = True
    calibration_mode: str = "temperature_scaled_platt"
    confidence_interval: list[float] = Field(default_factory=list)
    expected_calibration_error: float | None = None



class PreprocessingReport(BaseModel):
    filename: str
    source_format: str
    selected_variable: str | None = None
    available_variables: list[str] = Field(default_factory=list)
    tile_size: int
    overlap: int
    total_candidate_tiles: int
    materialized_tiles: int
    complete_coverage: bool
    normalization: str
    manifest_url: str


class GeoVerdict(BaseModel):
    status: VerdictStatus
    answer: str
    confidence: float = Field(ge=0, le=1)
    confidence_kind: str
    contradictions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence_breakdown: ConfidenceBreakdown


class AnalysisResponse(BaseModel):
    result_id: str
    task_plan: TaskPlan
    mode: str
    assets: list[RasterMetadata]
    quality: QualityReport
    evidence: list[EvidenceItem]
    verdict: GeoVerdict
    trace: list[TraceStep]
    artifacts: list[ArtifactRef]
    query: str
    generated_at: str
    preprocessing: list[PreprocessingReport] = Field(default_factory=list)


class AssetRecord(BaseModel):
    asset_id: str
    original_filename: str
    created_at: str
    metadata: RasterMetadata
    preprocessing: PreprocessingReport
    preview_url: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    asset_ids: list[str] = Field(min_length=1, max_length=2)
    pair_type: str = Field(default="auto", pattern="^(auto|single|bi_temporal|optical_sar)$")


class VRSBenchCategory(str, Enum):
    OPTICAL_VQA = "optical_vqa"
    MULTISPECTRAL_VQA = "multispectral_vqa"
    VISUAL_GROUNDING = "visual_grounding"
    LAND_COVER_VERIFICATION = "land_cover_verification"


class BenchmarkItemResult(BaseModel):
    sample_id: str
    category: VRSBenchCategory
    question: str
    target_model: str
    expected_answer: str
    predicted_answer: str
    exact_match: bool
    semantic_similarity: float
    confidence: float
    token_perplexity: float | None = None
    predicted_boxes: list[list[float]] = Field(default_factory=list)
    ground_truth_boxes: list[list[float]] = Field(default_factory=list)
    iou: float | None = None
    passed: bool
    latency_ms: int


class BenchmarkSummaryMetrics(BaseModel):
    total_samples: int
    passed_samples: int
    accuracy_percent: float
    mean_semantic_similarity: float
    mean_iou: float | None
    mean_confidence: float
    mean_latency_ms: float
    optical_accuracy: float
    multispectral_accuracy: float
    grounding_mean_iou: float
    model_variants_evaluated: list[str]


class BenchmarkRunRequest(BaseModel):
    dataset_name: str = Field(default="vrsbench_sample_split", description="Dataset split to evaluate")
    model_variant: str = Field(default="auto", description="'auto', 'earthdial-4b-rgb', 'earthdial-4b-ms' or 'mock'")
    max_samples: int | None = Field(default=None, description="Limit evaluation to top N samples")


class BenchmarkRunResponse(BaseModel):
    run_id: str
    dataset_name: str
    evaluated_at: str
    summary: BenchmarkSummaryMetrics
    results: list[BenchmarkItemResult]


class DatasetSummary(BaseModel):
    name: str
    description: str
    total_samples: int
    categories: list[str]
    supported_models: list[str]


class SpatialGeometryRecord(BaseModel):
    geometry_id: int | str
    result_id: str
    producer: str
    evidence_kind: str
    area_m2: float | None = None
    crs: str | None = None
    bounds: list[float] = Field(default_factory=list)
    geometry: dict[str, Any]
    properties: dict[str, Any] = Field(default_factory=dict)


class SpatialFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict[str, Any]]
    total_count: int
    properties: dict[str, Any] = Field(default_factory=dict)


class AnalysisRunRecord(BaseModel):
    result_id: str
    task_type: str
    query_text: str
    verdict_status: str
    confidence: float
    created_at: str
    geometries_count: int


