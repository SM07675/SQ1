import React from "react";
import { artifactUrl } from "../api";
import type { AnalysisResponse } from "../types";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { StatusPill } from "./shared/StatusPill";

interface VerdictPanelProps {
  result: AnalysisResponse | null;
}

export const VerdictPanel: React.FC<VerdictPanelProps> = ({ result }) => {
  if (!result) {
    return (
      <aside className="panel verdict-panel empty-verdict" aria-label="Verdict & Telemetry">
        <div className="panel-header">
          <div className="panel-header-icon">⚖</div>
          <div className="panel-header-titles">
            <h3>GeoProof Verdict</h3>
            <small>Awaiting analysis execution</small>
          </div>
        </div>
        <div className="verdict-placeholder">
          <div className="placeholder-icon">📋</div>
          <p>Submit your geospatial query to inspect verified reasoning and calibrated confidence.</p>
        </div>
      </aside>
    );
  }

  const { verdict, artifacts } = result;
  const report = artifacts.find((item) => item.name === "GeoProof_Report.pdf");
  const manifest = artifacts.find((item) => item.name === "analysis_manifest.json");
  const geojson = artifacts.find((item) => item.mime_type === "application/geo+json");
  const breakdown = verdict.confidence_breakdown;

  return (
    <aside className={`panel verdict-panel status-${verdict.status}`} aria-label="Verdict & Telemetry">
      <div className="panel-header">
        <div className="panel-header-icon">⚖</div>
        <div className="panel-header-titles">
          <h3>GeoProof™ Verdict</h3>
          <small>SIH26167 Verified Intelligence</small>
        </div>
      </div>

      <div className="verdict-content">
        {/* Status banner */}
        <div className="verdict-status-banner">
          <StatusPill variant={verdict.status} />
          <span className="verdict-timestamp">
            {new Date(result.generated_at).toLocaleTimeString()}
          </span>
        </div>

        {/* Confidence Gauge + Breakdown */}
        <div className="verdict-gauge-section">
          <ConfidenceGauge
            confidence={breakdown?.final_score ?? verdict.confidence}
            isCalibrated={breakdown?.is_calibrated}
            confidenceInterval={breakdown?.confidence_interval}
            ece={breakdown?.expected_calibration_error}
          />
        </div>

        {/* Natural Language Answer Block */}
        <div className="verdict-answer-box">
          <span className="box-eyebrow">GROUNDED CONCLUSION</span>
          <p className="answer-text">{verdict.answer}</p>
        </div>

        {/* Score Breakdown factors */}
        {breakdown && (
          <div className="score-factors-grid">
            <div className="factor-pill">
              <span>Input Quality</span>
              <strong>{Math.round(breakdown.input_quality * 100)}%</strong>
            </div>
            <div className="factor-pill">
              <span>Grid Alignment</span>
              <strong>{Math.round(breakdown.spatial_alignment * 100)}%</strong>
            </div>
            <div className="factor-pill">
              <span>Evidence Strength</span>
              <strong>{Math.round(breakdown.evidence_strength * 100)}%</strong>
            </div>
            <div className="factor-pill">
              <span>Ensemble Agreement</span>
              <strong>{Math.round(breakdown.ensemble_agreement * 100)}%</strong>
            </div>
          </div>
        )}

        {/* Limitations & Safe Abstention Warnings */}
        {verdict.limitations && verdict.limitations.length > 0 && (
          <div className="limitations-box">
            <span className="limitations-title">⚠ Analysis Limitations</span>
            <ul className="limitations-list">
              {verdict.limitations.map((lim, i) => (
                <li key={i}>{lim}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Download Audit Reports & Artifacts */}
        <div className="download-actions">
          {report && (
            <a
              href={artifactUrl(report.url)}
              download="GeoProof_Report.pdf"
              className="download-btn primary-download"
            >
              <span>📄</span>
              <span>Download Audit Report (PDF)</span>
            </a>
          )}
          <div className="secondary-downloads-row">
            {geojson && (
              <a
                href={artifactUrl(geojson.url)}
                download="evidence_geometries.geojson"
                className="download-btn secondary-download"
                title="Export GeoJSON Boundaries"
              >
                <span>🌐 GeoJSON</span>
              </a>
            )}
            {manifest && (
              <a
                href={artifactUrl(manifest.url)}
                download="analysis_manifest.json"
                className="download-btn secondary-download"
                title="Export Reproducible Manifest"
              >
                <span>📦 Manifest</span>
              </a>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
};
