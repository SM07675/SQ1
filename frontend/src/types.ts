export type VerdictStatus = "supported" | "supported_with_limitations" | "disputed" | "insufficient_evidence" | "low_confidence" | "invalid_input" | "unsupported_task" | "model_unavailable" | "degraded_analysis";

export interface StructuredLimitation {
  type: string;
  task: string;
  required?: string[];
  available?: string[];
  impact: string;
  mitigation?: string | null;
}

export interface TaskPlan {
  task: string;
  application?: string;
  specific_task?: string;
  sub_tasks?: string[];
  multi_intent?: boolean;
  target?: string | null;
  asks_direction: boolean;
  tools: string[];
  reason: string;
  aoi_text?: string | null;
  years?: number[];
}

export interface RasterMetadata {
  filename: string;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  crs?: string | null;
  bounds: number[];
  resolution: number[];
  nodata_percent: number;
  band_names: string[];
  available_indices: string[];
  file_format?: "png" | "jpeg" | "tiff" | "geotiff" | string;
  modality?: "optical" | "sar" | "sar_like" | "unknown" | string;
  sensor_verified?: boolean;
  georeferenced?: boolean;
  metadata_available?: boolean;
  analysis_capabilities?: string[];
  limitations?: string[];
}

export interface QualityReport {
  score: number;
  compatible: boolean;
  blockers: string[];
  warnings: string[];
  checks: Record<string, unknown>;
}

export interface EvidenceItem {
  kind: string;
  producer: string;
  summary: string;
  confidence: number;
  artifact_url?: string | null;
  metrics: Record<string, unknown>;
}

export interface TraceStep {
  step: number;
  component: string;
  action: string;
  status: string;
  duration_ms: number;
  details: Record<string, unknown>;
}

export interface ArtifactRef {
  name: string;
  url: string;
  mime_type: string;
  artifact_id?: string;
  role?: string;
}

export interface SummaryMetric {
  label: string;
  value: string;
  icon?: string | null;
}

export interface AnalysisSummary {
  title: string;
  headline: string;
  metrics: SummaryMetric[];
  explanation: string;
  detected_changes?: string[];
  confidence_percent?: number | null;
  is_insufficient?: boolean;
}

export interface AnalysisResponse {
  result_id: string;
  query: string;
  generated_at: string;
  task_plan: TaskPlan;
  mode: string;
  assets: RasterMetadata[];
  quality: QualityReport;
  evidence: EvidenceItem[];
  verdict: {
    status: VerdictStatus;
    answer: string;
    confidence: number;
    confidence_kind: string;
    contradictions: string[];
    limitations: string[];
    structured_limitations?: StructuredLimitation[];
    confidence_breakdown: {
      input_quality: number;
      spatial_alignment: number;
      evidence_strength: number;
      ensemble_agreement: number;
      model_probability?: number | null;
      warning_penalty: number;
      final_score: number;
      is_calibrated?: boolean;
      calibration_mode?: string;
      confidence_interval?: number[];
      expected_calibration_error?: number | null;
    };
  };
  trace: TraceStep[];
  artifacts: ArtifactRef[];
  preprocessing: Array<{
    filename: string;
    source_format: string;
    selected_variable?: string | null;
    tile_size: number;
    overlap: number;
    total_candidate_tiles: number;
    materialized_tiles: number;
    complete_coverage: boolean;
    normalization: string;
    manifest_url: string;
  }>;
  summary?: AnalysisSummary | null;
  input_configuration?: string;
  findings?: Array<Record<string, unknown>>;
  statistics?: Record<string, unknown>;
  models_used?: string[];
  timings?: Record<string, number>;
  clarification?: string | null;
}

export interface ModelCapability {
  name: string;
  purpose: string;
  available: boolean;
  required_inputs: string[];
}

export interface DatasetSummary {
  name: string;
  description: string;
  total_samples: number;
  categories: string[];
  supported_models: string[];
}

export interface BenchmarkItemResult {
  sample_id: string;
  category: string;
  question: string;
  target_model: string;
  expected_answer: string;
  predicted_answer: string;
  exact_match: boolean;
  semantic_similarity: number;
  confidence: number;
  token_perplexity?: number | null;
  predicted_boxes: number[][];
  ground_truth_boxes: number[][];
  iou?: number | null;
  passed: boolean;
  latency_ms: number;
}

export interface BenchmarkSummaryMetrics {
  total_samples: number;
  passed_samples: number;
  accuracy_percent: number;
  mean_semantic_similarity: number;
  mean_iou?: number | null;
  mean_confidence: number;
  mean_latency_ms: number;
  optical_accuracy: number;
  multispectral_accuracy: number;
  grounding_mean_iou: number;
  model_variants_evaluated: string[];
}

export interface BenchmarkRunResponse {
  run_id: string;
  dataset_name: string;
  evaluated_at: string;
  summary: BenchmarkSummaryMetrics;
  results: BenchmarkItemResult[];
}

export interface AnalysisRunRecord {
  result_id: string;
  task_type: string;
  query_text: string;
  verdict_status: string;
  confidence: number;
  created_at: string;
  geometries_count: number;
}

export interface ChatImageRecord {
  image_id: string;
  chat_id: string;
  filename: string;
  storage_path: string;
  url: string;
  uploaded_at: string;
}

export interface ChatMessageRecord {
  message_id: string;
  chat_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  attachments?: Array<{ name: string; url: string; type: string }>;
  result?: AnalysisResponse | null;
}

export interface ChatSummary {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  icon?: string;
  metadata?: Record<string, unknown>;
  message_count?: number;
  image_count?: number;
  last_message?: string | null;
  preview_text?: string;
  latest_result?: AnalysisResponse | null;
}

export interface ChatDetail {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  icon?: string;
  metadata?: Record<string, unknown>;
  messages: ChatMessageRecord[];
  images: ChatImageRecord[];
  latest_result?: AnalysisResponse | null;
}

export type NavTab = "analyze" | "compare" | "explore" | "reports";
export type ThemeMode = "dark" | "light";
export type CompareMode = "swipe" | "side_by_side" | "overlay";

export interface SpatialGeometryRecord {
  geometry_id: string;
  result_id: string;
  evidence_kind: string;
  geometry: {
    type: string;
    coordinates: any;
  };
  properties: Record<string, unknown>;
  confidence?: number;
}
