import type {
  AnalysisResponse,
  AnalysisRunRecord,
  BenchmarkRunResponse,
  DatasetSummary,
  ModelCapability,
} from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export function artifactUrl(path?: string | null): string | undefined {
  if (!path) return undefined;
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE}${path}`;
}

export async function downloadPdfReport(resultId: string, customFilename?: string): Promise<void> {
  const filename = customFilename || `GeoProof_Report_${resultId.slice(0, 8)}.pdf`;
  const url = artifactUrl(`/artifacts/${resultId}/GeoProof_Report.pdf`);
  if (!url) return;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download report (HTTP ${response.status})`);
    }
    const blob = await response.blob();
    const pdfBlob = new Blob([blob], { type: "application/pdf" });
    const blobUrl = window.URL.createObjectURL(pdfBlob);
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = filename.endsWith(".pdf") ? filename : `${filename}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => window.URL.revokeObjectURL(blobUrl), 1000);
  } catch (err) {
    console.error("Blob download failed, opening direct link:", err);
    const directUrl = artifactUrl(`/artifacts/${resultId}/GeoProof_Report.pdf?download=true`);
    if (directUrl) {
      const fallbackLink = document.createElement("a");
      fallbackLink.href = directUrl;
      fallbackLink.download = filename;
      document.body.appendChild(fallbackLink);
      fallbackLink.click();
      document.body.removeChild(fallbackLink);
    }
  }
}

export async function analyzeImages(input: {
  query: string;
  pairType: string;
  imageA: File;
  imageB?: File;
}): Promise<AnalysisResponse> {
  const body = new FormData();
  body.set("query", input.query);
  body.set("pair_type", input.pairType);
  body.set("image_a", input.imageA);
  if (input.imageB) body.set("image_b", input.imageB);

  const response = await fetch(`${API_BASE}/api/v1/analyze`, { method: "POST", body });
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Analysis failed with HTTP ${response.status}`);
  }
  return response.json() as Promise<AnalysisResponse>;
}

export async function modelCapabilities(): Promise<ModelCapability[]> {
  const response = await fetch(`${API_BASE}/api/v1/models`);
  if (!response.ok) return [];
  return response.json() as Promise<ModelCapability[]>;
}

export async function fetchBenchmarkDatasets(): Promise<DatasetSummary[]> {
  const response = await fetch(`${API_BASE}/api/v1/benchmarks/datasets`);
  if (!response.ok) return [];
  return response.json() as Promise<DatasetSummary[]>;
}

export async function runBenchmarkEvaluation(input: {
  datasetName: string;
  modelVariant: string;
  maxSamples?: number;
}): Promise<BenchmarkRunResponse> {
  const response = await fetch(`${API_BASE}/api/v1/benchmarks/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: input.datasetName,
      model_variant: input.modelVariant,
      max_samples: input.maxSamples,
    }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Benchmark evaluation failed with HTTP ${response.status}`);
  }
  return response.json() as Promise<BenchmarkRunResponse>;
}

export async function fetchSpatialRuns(filters?: {
  verdictStatus?: string;
  taskType?: string;
  limit?: number;
}): Promise<AnalysisRunRecord[]> {
  const params = new URLSearchParams();
  if (filters?.verdictStatus) params.set("verdict_status", filters.verdictStatus);
  if (filters?.taskType) params.set("task_type", filters.taskType);
  if (filters?.limit) params.set("limit", String(filters.limit));
  const url = `${API_BASE}/api/v1/spatial/runs${params.toString() ? `?${params}` : ""}`;
  const response = await fetch(url);
  if (!response.ok) return [];
  return response.json() as Promise<AnalysisRunRecord[]>;
}

export async function fetchResultById(resultId: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/api/v1/results/${resultId}`);
  if (!response.ok) {
    throw new Error(`Failed to load analysis result ${resultId} (HTTP ${response.status})`);
  }
  return response.json() as Promise<AnalysisResponse>;
}


