import React, { useState } from "react";
import {
  X,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Layers,
  MapPin,
  Cpu,
  Clock,
  FileCode,
  Shield,
  HelpCircle,
} from "lucide-react";
import type { AnalysisResponse } from "../../types";
import { artifactUrl } from "../../api";

interface AnalysisInspectorProps {
  isOpen: boolean;
  onClose: () => void;
  result: AnalysisResponse | null;
  onOpenImage?: (url: string, title: string) => void;
}

export const AnalysisInspector: React.FC<AnalysisInspectorProps> = ({
  isOpen,
  onClose,
  result,
  onOpenImage,
}) => {
  const [activeTab, setActiveTab] = useState<"overview" | "evidence" | "geospatial" | "technical">("overview");

  if (!isOpen || !result) return null;

  const { verdict, quality, assets, evidence, trace, artifacts, models_used, timings } = result;

  return (
    <>
      <div className="drawer-backdrop visible" onClick={onClose} />
      <aside className="inspector-drawer open" aria-label="Analysis Inspector">
        <div className="inspector-header">
          <div className="inspector-header-left">
            <Sliders size={16} className="inspector-icon" />
            <h3 className="inspector-title">Analysis Inspector</h3>
          </div>
          <button
            type="button"
            className="drawer-close-btn"
            onClick={onClose}
            title="Close inspector (Esc)"
          >
            <X size={16} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="inspector-tabs-nav">
          <button
            type="button"
            className={`inspector-tab-btn ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            Overview
          </button>
          <button
            type="button"
            className={`inspector-tab-btn ${activeTab === "evidence" ? "active" : ""}`}
            onClick={() => setActiveTab("evidence")}
          >
            Evidence
          </button>
          <button
            type="button"
            className={`inspector-tab-btn ${activeTab === "geospatial" ? "active" : ""}`}
            onClick={() => setActiveTab("geospatial")}
          >
            Geospatial
          </button>
          <button
            type="button"
            className={`inspector-tab-btn ${activeTab === "technical" ? "active" : ""}`}
            onClick={() => setActiveTab("technical")}
          >
            Technical
          </button>
        </div>

        {/* Content Body */}
        <div className="inspector-body">
          {/* TAB 1: OVERVIEW */}
          {activeTab === "overview" && (
            <div className="inspector-tab-content">
              <div className="inspector-section">
                <span className="section-label">QUERY INTENT</span>
                <p className="query-quote">"{result.query}"</p>
                <div className="meta-pill-row">
                  <span className="meta-pill">Mode: {result.mode}</span>
                  <span className="meta-pill">
                    Planner Task: {result.task_plan?.task || "Analysis"}
                  </span>
                </div>
              </div>

              <div className="inspector-section">
                <span className="section-label">GEOVERDICT STATUS</span>
                <div className="verdict-status-box">
                  <div className="status-name-row">
                    <span className="status-text">{verdict.status.replace(/_/g, " ").toUpperCase()}</span>
                    <span className="status-score">{Math.round(verdict.confidence * 100)}% Confidence</span>
                  </div>
                  <p className="status-desc">{verdict.answer}</p>
                </div>
              </div>

              <div className="inspector-section">
                <span className="section-label">CONFIDENCE BREAKDOWN</span>
                <div className="breakdown-list">
                  <div className="breakdown-item">
                    <span>Input Raster Quality</span>
                    <strong>{Math.round(verdict.confidence_breakdown.input_quality * 100)}%</strong>
                  </div>
                  <div className="breakdown-item">
                    <span>Spatial Alignment</span>
                    <strong>{Math.round(verdict.confidence_breakdown.spatial_alignment * 100)}%</strong>
                  </div>
                  <div className="breakdown-item">
                    <span>Evidence Strength</span>
                    <strong>{Math.round(verdict.confidence_breakdown.evidence_strength * 100)}%</strong>
                  </div>
                  <div className="breakdown-item">
                    <span>Ensemble Agreement</span>
                    <strong>{Math.round(verdict.confidence_breakdown.ensemble_agreement * 100)}%</strong>
                  </div>
                </div>
              </div>

              {verdict.limitations?.length > 0 && (
                <div className="inspector-section">
                  <span className="section-label">DOCUMENTED LIMITATIONS</span>
                  <ul className="limitations-list">
                    {verdict.limitations.map((lim, idx) => (
                      <li key={idx}>
                        <AlertTriangle size={12} className="lim-bullet" />
                        <span>{lim}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: EVIDENCE */}
          {activeTab === "evidence" && (
            <div className="inspector-tab-content">
              <div className="inspector-section">
                <span className="section-label">MULTI-WITNESS EVIDENCE ARTIFACTS</span>
                <div className="evidence-grid-inspector">
                  {evidence.map((item, idx) => {
                    const fullUrl = item.artifact_url ? artifactUrl(item.artifact_url) : null;
                    return (
                      <div key={idx} className="evidence-inspector-card">
                        <div className="evidence-card-top">
                          <span className="evidence-kind-badge">
                            {item.kind.replace(/_/g, " ")}
                          </span>
                          <span className="evidence-score">
                            {Math.round(item.confidence * 100)}%
                          </span>
                        </div>
                        <p className="evidence-summary-text">{item.summary}</p>
                        {fullUrl && (
                          <div
                            className="evidence-thumb-container"
                            onClick={() => onOpenImage && onOpenImage(fullUrl, item.kind)}
                          >
                            <img src={fullUrl} alt={item.kind} loading="lazy" />
                            <span className="thumb-hover-label">Click to zoom</span>
                          </div>
                        )}
                        {item.metrics && Object.keys(item.metrics).length > 0 && (
                          <div className="evidence-metrics-tags">
                            {Object.entries(item.metrics).slice(0, 4).map(([k, v]) => (
                              <span key={k} className="metric-tag">
                                {k}: {String(v)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: GEOSPATIAL */}
          {activeTab === "geospatial" && (
            <div className="inspector-tab-content">
              {assets.map((asset, idx) => (
                <div key={idx} className="inspector-section">
                  <span className="section-label">RASTER {idx + 1}: {asset.filename}</span>
                  <div className="geo-metadata-table">
                    <div className="geo-row">
                      <span className="geo-key">Coordinate Reference (CRS)</span>
                      <code className="geo-val">{asset.crs || "Unprojected Local Grid"}</code>
                    </div>
                    <div className="geo-row">
                      <span className="geo-key">Raster Dimensions</span>
                      <span className="geo-val">{asset.width} × {asset.height} px</span>
                    </div>
                    <div className="geo-row">
                      <span className="geo-key">Spectral Bands</span>
                      <span className="geo-val">{asset.bands} ({asset.band_names?.join(", ") || asset.dtype})</span>
                    </div>
                    <div className="geo-row">
                      <span className="geo-key">Ground Resolution</span>
                      <span className="geo-val">
                        {asset.resolution && asset.resolution.length >= 2
                          ? `${asset.resolution[0].toFixed(2)}m × ${asset.resolution[1].toFixed(2)}m / px`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="geo-row">
                      <span className="geo-key">Georeferenced</span>
                      <span className="geo-val">
                        {asset.crs ? "✓ Projected Coordinates" : "Local Pixel Grid"}
                      </span>
                    </div>
                    {asset.bounds && asset.bounds.length === 4 && (
                      <div className="geo-row">
                        <span className="geo-key">Bounding Box</span>
                        <code className="geo-val bounds">
                          [{asset.bounds.map((b) => b.toFixed(1)).join(", ")}]
                        </code>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* TAB 4: TECHNICAL */}
          {activeTab === "technical" && (
            <div className="inspector-tab-content">
              <div className="inspector-section">
                <span className="section-label">MODELS EXECUTED</span>
                <div className="models-tags-wrap">
                  {models_used && models_used.length > 0 ? (
                    models_used.map((m) => (
                      <span key={m} className="model-chip">
                        <Cpu size={12} />
                        <span>{m}</span>
                      </span>
                    ))
                  ) : (
                    <span className="no-models-text">Deterministic Spectral Analysis Engine</span>
                  )}
                </div>
              </div>

              {timings && (
                <div className="inspector-section">
                  <span className="section-label">EXECUTION LATENCY</span>
                  <div className="latency-row">
                    <Clock size={13} />
                    <span>Analysis Pipeline: {Math.round(timings.analysis_ms || 0)} ms</span>
                  </div>
                </div>
              )}

              <div className="inspector-section">
                <span className="section-label">PIPELINE EXECUTION TRACE</span>
                <div className="trace-steps-list">
                  {trace.map((step) => (
                    <div key={step.step} className="trace-step-row">
                      <span className="step-num">#{step.step}</span>
                      <div className="step-info">
                        <div className="step-action-name">{step.action}</div>
                        <div className="step-component">{step.component}</div>
                      </div>
                      <span className="step-duration">{step.duration_ms}ms</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="inspector-section">
                <span className="section-label">GENERATED ARTIFACTS</span>
                <ul className="artifacts-list">
                  {artifacts.map((a, idx) => (
                    <li key={idx} className="artifact-item-row">
                      <FileCode size={13} />
                      <span className="artifact-name">{a.name}</span>
                      <span className="artifact-type">{a.mime_type}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
};
