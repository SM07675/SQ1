import type {
  AnalysisResponse,
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

