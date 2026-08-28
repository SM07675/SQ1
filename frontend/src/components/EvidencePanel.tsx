import React, { useState } from "react";
import type { EvidenceItem } from "../types";

interface EvidencePanelProps {
  evidence: EvidenceItem[];
  metrics: [string, unknown][];
  contradictions?: string[];
}

function formatMetricValue(value: unknown): string {
  if (typeof value !== "number") return String(value ?? "n/a");
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} km²`;
  if (Math.abs(value) >= 10_000) return `${(value / 10_000).toFixed(2)} ha`;
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  evidence,
  metrics,
  contradictions,
}) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const toggleExpand = (idx: number) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  return (
    <section className="evidence-panel-component">
      {/* Evidence Items Ledger */}
      <div className="evidence-section-block">
        <div className="section-title-row">
          <span className="block-title">MULTI-SENSOR EVIDENCE PROOFS</span>
          <span className="count-badge">{evidence.length} sources</span>
        </div>

        {evidence.length === 0 ? (
          <p className="empty-subtext">No evidence produced yet. Run an analysis to generate telemetry.</p>
        ) : (
          <div className="evidence-cards-list">
            {evidence.map((item, idx) => {
              const isExpanded = expandedIndex === idx;
              const confPct = Math.round(item.confidence * 100);

              return (
                <article
                  key={`${item.producer}-${idx}`}
                  className={`evidence-item-card ${isExpanded ? "expanded" : ""}`}
                  onClick={() => toggleExpand(idx)}
                >
                  <div className="item-header-row">
                    <div className="item-kind-badge">
                      <span className="kind-icon">✦</span>
                      <span className="kind-text">{item.kind.replace(/_/g, " ")}</span>
                    </div>
                    <span className="item-producer-tag">{item.producer}</span>
                    <span className="item-conf-score">{confPct}% conf</span>
                  </div>

                  <p className="item-summary-text">{item.summary}</p>

                  {/* Metrics preview row */}
                  {Object.keys(item.metrics).length > 0 && (
                    <div className="item-metrics-pills">
                      {Object.entries(item.metrics).map(([k, v]) => (
                        <span key={k} className="metric-tag">
                          <b>{k.replace(/_/g, " ")}:</b> {formatMetricValue(v)}
                        </span>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>

      {/* Global Consolidated Metrics Grid */}
      {metrics.length > 0 && (
        <div className="evidence-section-block">
          <span className="block-title">MEASURED QUANTITIES & METRICS</span>
          <div className="metrics-summary-grid">
            {metrics.map(([key, value]) => (
              <div key={key} className="metric-box">
                <span className="metric-box-label">{key.replace(/_/g, " ")}</span>
                <strong className="metric-box-val">{formatMetricValue(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Contradictions & Warnings */}
      {contradictions && contradictions.length > 0 && (
        <div className="contradictions-block">
          <span className="contradictions-title">⚠ Contradictions Detected</span>
          <ul className="contradictions-list">
            {contradictions.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
};
