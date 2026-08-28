import React from "react";
import type { BenchmarkRunResponse, DatasetSummary } from "../types";
import { BenchmarkControls } from "./BenchmarkControls";
import { BenchmarkKPIs } from "./BenchmarkKPIs";
import { BenchmarkTable } from "./BenchmarkTable";

interface BenchmarkDashboardProps {
  datasets: DatasetSummary[];
  selectedDataset: string;
  selectedVariant: string;
  benchRunning: boolean;
  benchResult: BenchmarkRunResponse | null;
  benchError: string | null;
  onSelectDataset: (name: string) => void;
  onSelectVariant: (variant: string) => void;
  onRunBenchmark: () => void;
}

export const BenchmarkDashboard: React.FC<BenchmarkDashboardProps> = ({
  datasets,
  selectedDataset,
  selectedVariant,
  benchRunning,
  benchResult,
  benchError,
  onSelectDataset,
  onSelectVariant,
  onRunBenchmark,
}) => {
  return (
    <div className="benchmark-dashboard-view">
      {/* Benchmark Control Bar */}
      <BenchmarkControls
        datasets={datasets}
        selectedDataset={selectedDataset}
        selectedVariant={selectedVariant}
        benchRunning={benchRunning}
        onSelectDataset={onSelectDataset}
        onSelectVariant={onSelectVariant}
        onRunBenchmark={onRunBenchmark}
      />

      {benchError && (
        <div className="error-banner bench-error-banner" role="alert">
          <span className="error-icon">⚠</span>
          <span className="error-text">{benchError}</span>
        </div>
      )}

      {benchResult ? (
        <div className="bench-results-wrapper">
          <BenchmarkKPIs summary={benchResult.summary} />
          <BenchmarkTable
            results={benchResult.results}
            datasetName={benchResult.dataset_name}
          />
        </div>
      ) : !benchRunning ? (
        <div className="bench-empty-placeholder">
          <div className="bench-empty-icon">📊</div>
          <h3>Select a Frozen-Split Dataset</h3>
          <p>
            Run Gate 6 evaluation across VRSBench, RSVQA, or CDVQA test suites with Platt calibrated
            confidence and token perplexity analysis.
          </p>
        </div>
      ) : null}
    </div>
  );
};
