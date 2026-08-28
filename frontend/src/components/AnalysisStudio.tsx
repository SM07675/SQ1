import React, { FormEvent } from "react";
import type { AnalysisResponse } from "../types";
import { AnalysisSetup, QueryTemplate } from "./AnalysisSetup";
import { EvidencePanel } from "./EvidencePanel";
import { ImageryViewport } from "./ImageryViewport";
import { MetricsBar } from "./MetricsBar";
import { VerdictPanel } from "./VerdictPanel";

interface AnalysisStudioProps {
  result: AnalysisResponse | null;
  templates: QueryTemplate[];
  query: string;
  pairType: string;
  imageA: File | null;
  imageB: File | null;
  activeLayer: string;
  availableLayers: string[];
  layerNames: Record<string, string>;
  swipe: number;
  busy: boolean;
  error: string | null;
  uniqueMetrics: [string, unknown][];
  onQueryChange: (query: string) => void;
  onPairTypeChange: (pairType: string) => void;
  onImageAChange: (file: File | null) => void;
  onImageBChange: (file: File | null) => void;
  onSelectTemplate: (template: QueryTemplate) => void;
  onSelectLayer: (layer: string) => void;
  onSwipeChange: (swipe: number) => void;
  onSubmit: (e: FormEvent) => void;
}

export const AnalysisStudio: React.FC<AnalysisStudioProps> = ({
  result,
  templates,
  query,
  pairType,
  imageA,
  imageB,
  activeLayer,
  availableLayers,
  layerNames,
  swipe,
  busy,
  error,
  uniqueMetrics,
  onQueryChange,
  onPairTypeChange,
  onImageAChange,
  onImageBChange,
  onSelectTemplate,
  onSelectLayer,
  onSwipeChange,
  onSubmit,
}) => {
  return (
    <div className="analysis-studio-view">
      {/* Top Metrics KPI Bar */}
      <MetricsBar result={result} />

      {/* Main 3-Column Geospatial Analysis Grid */}
      <div className="analysis-content-grid">
        {/* Left Column: Analysis Setup & Imagery Inputs */}
        <AnalysisSetup
          templates={templates}
          query={query}
          pairType={pairType}
          imageA={imageA}
          imageB={imageB}
          busy={busy}
          error={error}
          onQueryChange={onQueryChange}
          onPairTypeChange={onPairTypeChange}
          onImageAChange={onImageAChange}
          onImageBChange={onImageBChange}
          onSelectTemplate={onSelectTemplate}
          onSubmit={onSubmit}
        />

        {/* Center Dominant Column: Imagery Viewport & Evidence Telemetry */}
        <div className="center-workspace-column">
          <ImageryViewport
            result={result}
            activeLayer={activeLayer}
            availableLayers={availableLayers}
            layerNames={layerNames}
            swipe={swipe}
            busy={busy}
            onSelectLayer={onSelectLayer}
            onSwipeChange={onSwipeChange}
          />

          {result && (
            <EvidencePanel
              evidence={result.evidence}
              metrics={uniqueMetrics}
              contradictions={result.verdict.contradictions}
            />
          )}
        </div>

        {/* Right Column: GeoProof Verdict & Calibrated Telemetry */}
        <VerdictPanel result={result} />
      </div>
    </div>
  );
};
