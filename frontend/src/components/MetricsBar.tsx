import React from "react";
import type { AnalysisResponse } from "../types";

interface MetricsBarProps {
  result: AnalysisResponse | null;
}

export const MetricsBar: React.FC<MetricsBarProps> = ({ result }) => {
  const qualityScore = result ? Math.round(result.quality.score * 100) : null;
  const alignment = result?.quality.checks.alignment_score;
  const alignmentPercent =
    typeof alignment === "number" ? Math.round(alignment * 100) : null;
  const evidenceCount = result?.evidence.length ?? null;
  const verdictStatus = result?.verdict.status;

  return (
    <section className="kpi-row" aria-label="Analysis Metrics Summary">
      <article className="kpi-card">
        <span className="kpi-label">INPUT QUALITY</span>
        <strong className="kpi-value">
          {qualityScore !== null ? `${qualityScore}%` : "—"}
        </strong>
        <small className="kpi-meta">Metadata + CRS consistency</small>
      </article>

      <article className="kpi-card">
        <span className="kpi-label">GRID ALIGNMENT</span>
        <strong className="kpi-value">
          {alignmentPercent !== null ? `${alignmentPercent}%` : "—"}
        </strong>
        <small className="kpi-meta">Sub-pixel spatial overlap</small>
      </article>

      <article className="kpi-card">
        <span className="kpi-label">EVIDENCE SOURCES</span>
        <strong className="kpi-value">
          {evidenceCount !== null ? evidenceCount : "—"}
        </strong>
        <small className="kpi-meta">Independent multi-sensor proofs</small>
      </article>

      <article className={`kpi-card accent-kpi ${verdictStatus ? `verdict-${verdictStatus}` : ""}`}>
        <span className="kpi-label">GEOPROOF VERDICT</span>
        <strong className="kpi-value">
          {verdictStatus ? verdictStatus.replace(/_/g, " ").toUpperCase() : "AWAITING RUN"}
        </strong>
        <small className="kpi-meta">Safe abstention & calibrated bounds</small>
      </article>
    </section>
  );
};
