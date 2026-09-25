from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class TaskType(str, Enum):
    BUILDING_COUNT = "building_count"
    BUILDING_CHANGE = "building_change"
    MULTI_INTENT = "multi_intent"
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
    SUPPORTED_WITH_LIMITATIONS = "supported_with_limitations"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_TASK = "unsupported_task"
    DISPUTED = "disputed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MODEL_UNAVAILABLE = "model_unavailable"
    DEGRADED_ANALYSIS = "degraded_analysis"


class PlanNode(BaseModel):
    node_id: str
    tool: str
    depends_on: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


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
    intents: list[str] = Field(default_factory=list)
    nodes: list[PlanNode] = Field(default_factory=list)
    requires_pair: bool = False
    requires_temporal_relationship: bool = False
    clarification: str | None = None


class RasterMetadata(BaseModel):
    asset_id: str | None = None
    file_hash: str | None = None
    band_map: dict[str, Any] = Field(default_factory=dict)
    band_mapping: dict[str, Any] = Field(default_factory=dict)
    wavelengths: dict[str, float] = Field(default_factory=dict)
    band_wavelengths: dict[str, float] = Field(default_factory=dict)
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    quality_score: float | None = None
    valid_fraction: float = 1.0
    filename: str
    width: int
    height: int
    bands: int
    band_count: int = 0
    dtype: str
    crs: str | None
    bounds: list[float]
    resolution: list[float]
    nodata: float | None
    nodata_percent: float
    band_names: list[str]
    available_indices: list[str] = Field(default_factory=list)
    source_format: str = "raster"
    transform: list[float] = Field(default_factory=list)
    modality: str = "unknown"
    acquisition_date: str | None = None
    acquisition_time: str | None = None
    sensor: str | None = None
    platform: str | None = None
    sun_azimuth: float | None = None
    sun_elevation: float | None = None
    cloud_information: dict[str, Any] = Field(default_factory=dict)
    polarization: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    wgs84_bounds: list[float] | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.band_count:
            self.band_count = self.bands
        if not self.band_mapping and self.band_map:
            self.band_mapping = self.band_map
        if not self.band_wavelengths and self.wavelengths:
            self.band_wavelengths = self.wavelengths
        if not self.platform and self.sensor:
            self.platform = self.sensor
        if not self.acquisition_time and self.acquisition_date:
            self.acquisition_time = self.acquisition_date


RasterAsset = RasterMetadata
RasterProfile = RasterMetadata



class QualityReport(BaseModel):
    score: float = Field(ge=0, le=1)
    compatible: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    evidence_id: str = ""
    kind: str
    producer: str
    summary: str
    confidence: float = Field(ge=0, le=1)
    artifact_url: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    supports_claim: bool | None = None
    confidence_source: str = "producer_score"
    asset_indices: list[int] = Field(default_factory=list)
    timestamp: str | None = None


class TraceStep(BaseModel):
    step: int
    component: str
    action: str
    status: str
    duration_ms: int
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    artifact_id: str | None = None
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
    is_calibrated: bool = False
    calibration_mode: str = "uncalibrated_evidence_strength_v1"
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
    input_configuration: str = "INVALID"
    findings: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, Any] = Field(default_factory=dict)
    models_used: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    clarification: str | None = None
    report_id: str | None = None

    @computed_field
    @property
    def analysis_id(self) -> str:
        return self.result_id

    @computed_field
    @property
    def input_type(self) -> str:
        return self.input_configuration

    @computed_field
    @property
    def tasks(self) -> list[str]:
        return self.task_plan.intents

    @computed_field
    @property
    def buildings(self) -> dict[str, Any] | None:
        return self.statistics.get("buildings_a")

    @computed_field
    @property
    def water(self) -> dict[str, Any] | None:
        return self.statistics.get("water_measure")

    @computed_field
    @property
    def land_cover(self) -> dict[str, Any] | None:
        return self.statistics.get("land_cover")

    @computed_field
    @property
    def shadow(self) -> dict[str, Any] | None:
        water = self.statistics.get("water_measure")
        return water.get("shadow") if isinstance(water, dict) else None

    @computed_field
    @property
    def uncertainty(self) -> dict[str, Any] | None:
        water = self.statistics.get("water_measure")
        if not isinstance(water, dict):
            return None
        return {
            "water_shadow_conflict_score": water.get("water_shadow_conflict_score"),
            "uncertainty_score": water.get("uncertainty_score"),
        }

    @computed_field
    @property
    def raster_profile(self) -> RasterMetadata | None:
        return self.assets[0] if self.assets else None

    @computed_field
    @property
    def plan(self) -> TaskPlan:
        return self.task_plan

    @computed_field
    @property
    def vegetation(self) -> dict[str, Any] | None:
        return self.statistics.get("vegetation_measure") or self.statistics.get("spectral")

    @computed_field
    @property
    def built_up(self) -> dict[str, Any] | None:
        return self.statistics.get("built_up_measure") or self.statistics.get("buildings_a")

    @computed_field
    @property
    def change(self) -> dict[str, Any] | None:
        return self.statistics.get("change") or self.statistics.get("spectral_change") or self.statistics.get("building_match")

    @computed_field
    @property
    def locations(self) -> list[dict[str, Any]]:
        loc = self.statistics.get("user_location")
        return [loc] if loc else []

    @computed_field
    @property
    def confidence(self) -> float:
        return self.verdict.confidence

    @computed_field
    @property
    def warnings(self) -> list[str]:
        return self.quality.warnings


# One canonical model, retaining the existing public API name for compatibility.
AnalysisResult = AnalysisResponse


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


