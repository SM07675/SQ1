import React from "react";
import type { BenchmarkSummaryMetrics } from "../types";

interface BenchmarkKPIsProps {
  summary: BenchmarkSummaryMetrics;
}

export const BenchmarkKPIs: React.FC<BenchmarkKPIsProps> = ({ summary }) => {
  return (
    <section className="kpi-row bench-kpi-row" aria-label="Benchmark Summary KPIs">
      <article className="kpi-card">
        <span className="kpi-label">OVERALL ACCURACY</span>
        <strong className="kpi-value">{summary.accuracy_percent.toFixed(1)}%</strong>
        <small className="kpi-meta">
          {summary.passed_samples} / {summary.total_samples} samples passed
        </small>
      </article>

      <article className="kpi-card">
        <span className="kpi-label">SEMANTIC SIMILARITY</span>
        <strong className="kpi-value">
          {(summary.mean_semantic_similarity * 100).toFixed(1)}%
        </strong>
        <small className="kpi-meta">Word-embedding cosine alignment</small>
      </article>

      <article className="kpi-card">
        <span className="kpi-label">GROUNDING MEAN IOU</span>
        <strong className="kpi-value">
          {summary.grounding_mean_iou > 0
            ? `${(summary.grounding_mean_iou * 100).toFixed(1)}%`
            : "N/A"}
        </strong>
        <small className="kpi-meta">Spatial bounding box overlap</small>
      </article>

      <article className="kpi-card accent-kpi">
        <span className="kpi-label">CALIBRATED CONFIDENCE</span>
        <strong className="kpi-value">
          {(summary.mean_confidence * 100).toFixed(1)}%
        </strong>
        <small className="kpi-meta">Platt Temperature Scaled (T=1.32)</small>
      </article>
    </section>
  );
};
