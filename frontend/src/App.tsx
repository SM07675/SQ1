import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  analyzeImages,
  fetchBenchmarkDatasets,
  modelCapabilities,
  runBenchmarkEvaluation,
} from "./api";
import { AnalysisStudio } from "./components/AnalysisStudio";
import { BenchmarkDashboard } from "./components/BenchmarkDashboard";
import { ModelRail } from "./components/ModelRail";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import type {
  AnalysisResponse,
  BenchmarkRunResponse,
  DatasetSummary,
  ModelCapability,
} from "./types";

const templates = [
  {
    label: "Built-up change",
    pair: "bi_temporal",
    query: "Has built-up area increased between these two dates?",
  },
  {
    label: "Water grounding",
    pair: "single",
    query: "Highlight the largest water body.",
  },
  {
    label: "Optical + SAR",
    pair: "optical_sar",
    query: "Use optical and SAR evidence together to identify water-covered regions.",
  },
  {
    label: "Vegetation loss",
    pair: "bi_temporal",
    query: "Where has vegetation decreased between these two dates?",
  },
];

const layerNames: Record<string, string> = {
  compare: "Before / After (Swipe)",
  "preview_1.png": "T1 Optical Before",
  "preview_2.png": "T2 Optical After",
  "water_grounding_mask.png": "Water Grounding (Largest)",
  "water_mask.png": "Water Extraction",
  "ndwi.png": "NDWI Water Index",
  "change_mask.png": "Change Detection (All)",
  "semantic_change_mask.png": "Semantic Change",
  "sensor_agreement.png": "Sensor Agreement (Optical + SAR)",
  "croma_sensor_agreement.png": "CROMA Radar-Optical Agreement",
  "croma_sar_db_preview.png": "Sentinel-1 SAR Backscatter (dB)",
  "croma_optical_preview.png": "Sentinel-2 Optical (RGB)",
  "tinycd_change_probability_heatmap.png": "TinyCD Change Probability",
  "tinycd_change_mask.png": "TinyCD Structural Change",
  "built-up_mask.png": "Built-up Footprint",
  "vegetation_mask.png": "Vegetation Canopy",
  "ndbi_before.png": "NDBI Before (Built-up)",
  "ndbi_after.png": "NDBI After (Built-up)",
  "ndwi_before.png": "NDWI Before (Water)",
  "ndwi_after.png": "NDWI After (Water)",
  "ndvi_before.png": "NDVI Before (Vegetation)",
  "ndvi_after.png": "NDVI After (Vegetation)",
  "ndvi.png": "NDVI Canopy",
  "ndbi.png": "NDBI Index",
  "ndwi_optical.png": "Optical NDWI",
  "remoteclip_top_tiles_mosaic.png": "RemoteCLIP Retrieval",
};

function uniqueMetrics(result: AnalysisResponse | null): [string, unknown][] {
  const values = new Map<string, unknown>();
  result?.evidence.forEach((item) => {
    Object.entries(item.metrics).forEach(([key, value]) => values.set(key, value));
  });
  const priority = [
    "target",
    "direction",
    "changed_percent",
    "coverage_percent",
    "area_m2",
    "sensor_agreement_percent",
    "region_count",
    "spectral_index",
  ];
  return [...values.entries()].sort(([a], [b]) => {
    const ai = priority.indexOf(a);
    const bi = priority.indexOf(b);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  });
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"analysis" | "benchmarks">("analysis");
  const [query, setQuery] = useState(templates[0].query);
  const [pairType, setPairType] = useState(templates[0].pair);
  const [imageA, setImageA] = useState<File | null>(null);
  const [imageB, setImageB] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [activeLayer, setActiveLayer] = useState("compare");
  const [swipe, setSwipe] = useState(50);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelCapability[]>([]);

  // Benchmark Dashboard State
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("vrsbench_sample_split");
  const [selectedVariant, setSelectedVariant] = useState("auto");
  const [benchRunning, setBenchRunning] = useState(false);
  const [benchResult, setBenchResult] = useState<BenchmarkRunResponse | null>(null);
  const [benchError, setBenchError] = useState<string | null>(null);

  useEffect(() => {
    modelCapabilities().then(setModels).catch(() => setModels([]));
    fetchBenchmarkDatasets().then(setDatasets).catch(() => setDatasets([]));
  }, []);

  const availableLayers = useMemo(() => {
    if (!result) return ["compare"];
    const named = result.artifacts
      .filter((item) => item.mime_type === "image/png" && layerNames[item.name])
      .map((item) => item.name);
    return result.artifacts.some((item) => item.name === "preview_2.png")
      ? ["compare", ...named]
      : named;
  }, [result]);

  function handleSelectTemplate(tpl: (typeof templates)[0]) {
    setQuery(tpl.query);
    setPairType(tpl.pair);
  }

  async function submitAnalysis(event: FormEvent) {
    event.preventDefault();
    if (!imageA) {
      setError("Please select or drop Image A (Primary / Before Optical file).");
      return;
    }
    if ((pairType === "bi_temporal" || pairType === "optical_sar") && !imageB) {
      setError("This analysis workflow requires Image B (After or SAR file).");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeImages({
        query,
        pairType,
        imageA,
        imageB: imageB ?? undefined,
      });
      setResult(response);

      // Auto-select the most relevant output layer
      if (response.artifacts.some((item) => item.name === "water_grounding_mask.png")) {
        setActiveLayer("water_grounding_mask.png");
      } else if (response.artifacts.some((item) => item.name === "water_mask.png")) {
        setActiveLayer("water_mask.png");
      } else if (response.artifacts.some((item) => item.name === "semantic_change_mask.png")) {
        setActiveLayer("semantic_change_mask.png");
      } else if (response.artifacts.some((item) => item.name === "sensor_agreement.png")) {
        setActiveLayer("sensor_agreement.png");
      } else if (response.artifacts.some((item) => item.name === "change_mask.png")) {
        setActiveLayer("change_mask.png");
      } else if (response.artifacts.some((item) => item.name === "preview_2.png")) {
        setActiveLayer("compare");
      } else {
        setActiveLayer("preview_1.png");
      }
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Analysis engine encountered an error. Check file compatibility."
      );
    } finally {
      setBusy(false);
    }
  }

  async function triggerBenchmark() {
    setBenchRunning(true);
    setBenchError(null);
    try {
      const data = await runBenchmarkEvaluation({
        datasetName: selectedDataset,
        modelVariant: selectedVariant,
      });
      setBenchResult(data);
    } catch (err) {
      setBenchError(
        err instanceof Error ? err.message : "Failed to execute frozen-split benchmark suite"
      );
    } finally {
      setBenchRunning(false);
    }
  }

  const calculatedMetrics = useMemo(() => uniqueMetrics(result), [result]);

  return (
    <div className="app-shell">
      {/* Navigation Sidebar */}
      <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

      {/* Main Geospatial Workspace */}
      <main className="workspace">
        {/* Top Header & Branding Bar */}
        <TopBar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          analysisMode={result?.mode}
          isEngineOnline={true}
        />

        {/* Pipeline Capability Chip Rail */}
        <ModelRail models={models} />

        {/* Tab Content */}
        {activeTab === "analysis" ? (
          <AnalysisStudio
            result={result}
            templates={templates}
            query={query}
            pairType={pairType}
            imageA={imageA}
            imageB={imageB}
            activeLayer={activeLayer}
            availableLayers={availableLayers}
            layerNames={layerNames}
            swipe={swipe}
            busy={busy}
            error={error}
            uniqueMetrics={calculatedMetrics}
            onQueryChange={setQuery}
            onPairTypeChange={setPairType}
            onImageAChange={setImageA}
            onImageBChange={setImageB}
            onSelectTemplate={handleSelectTemplate}
            onSelectLayer={setActiveLayer}
            onSwipeChange={setSwipe}
            onSubmit={submitAnalysis}
          />
        ) : (
          <BenchmarkDashboard
            datasets={datasets}
            selectedDataset={selectedDataset}
            selectedVariant={selectedVariant}
            benchRunning={benchRunning}
            benchResult={benchResult}
            benchError={benchError}
            onSelectDataset={setSelectedDataset}
            onSelectVariant={setSelectedVariant}
            onRunBenchmark={triggerBenchmark}
          />
        )}
      </main>
    </div>
  );
}
