import React from "react";
import type { DatasetSummary } from "../types";

interface BenchmarkControlsProps {
  datasets: DatasetSummary[];
  selectedDataset: string;
  selectedVariant: string;
  benchRunning: boolean;
  onSelectDataset: (name: string) => void;
  onSelectVariant: (variant: string) => void;
  onRunBenchmark: () => void;
}

export const BenchmarkControls: React.FC<BenchmarkControlsProps> = ({
  datasets,
  selectedDataset,
  selectedVariant,
  benchRunning,
  onSelectDataset,
  onSelectVariant,
  onRunBenchmark,
}) => {
  return (
    <section className="bench-control-bar" aria-label="Benchmark Evaluation Controls">
      <div className="bench-selects-group">
        <div className="bench-field">
          <label htmlFor="dataset-select" className="bench-label">
            Frozen-Split Dataset:
          </label>
          <select
            id="dataset-select"
            value={selectedDataset}
            onChange={(e) => onSelectDataset(e.target.value)}
            className="bench-select"
          >
            {datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.total_samples} samples)
              </option>
            ))}
          </select>
        </div>

        <div className="bench-field">
          <label htmlFor="variant-select" className="bench-label">
            Model Variant / Pipeline:
          </label>
          <select
            id="variant-select"
            value={selectedVariant}
            onChange={(e) => onSelectVariant(e.target.value)}
            className="bench-select"
          >
            <option value="auto">Auto Variant Resolver</option>
            <option value="earthdial-4b-rgb">EarthDial-4B-RGB (Optical)</option>
            <option value="earthdial-4b-ms">EarthDial-4B-MS (Multispectral)</option>
          </select>
        </div>
      </div>

      <button
        type="button"
        className="bench-run-action-btn"
        onClick={onRunBenchmark}
        disabled={benchRunning}
      >
        {benchRunning ? (
          <>
            <span className="btn-spinner" />
            <span>Evaluating Frozen Split...</span>
          </>
        ) : (
          <>
            <span>▶</span>
            <span>Run Frozen-Split Benchmark</span>
          </>
        )}
      </button>
    </section>
  );
};
