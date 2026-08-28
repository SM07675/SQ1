import React, { FormEvent } from "react";
import { FileDropZone } from "./shared/FileDropZone";

export interface QueryTemplate {
  label: string;
  pair: string;
  query: string;
}

interface AnalysisSetupProps {
  templates: QueryTemplate[];
  query: string;
  pairType: string;
  imageA: File | null;
  imageB: File | null;
  busy: boolean;
  error: string | null;
  onQueryChange: (query: string) => void;
  onPairTypeChange: (pairType: string) => void;
  onImageAChange: (file: File | null) => void;
  onImageBChange: (file: File | null) => void;
  onSelectTemplate: (template: QueryTemplate) => void;
  onSubmit: (e: FormEvent) => void;
}

export const AnalysisSetup: React.FC<AnalysisSetupProps> = ({
  templates,
  query,
  pairType,
  imageA,
  imageB,
  busy,
  error,
  onQueryChange,
  onPairTypeChange,
  onImageAChange,
  onImageBChange,
  onSelectTemplate,
  onSubmit,
}) => {
  const needsImageB = pairType === "bi_temporal" || pairType === "optical_sar";

  return (
    <aside className="panel setup-panel" aria-label="Analysis Setup Controls">
      <div className="panel-header">
        <div className="panel-header-icon">⚙</div>
        <div className="panel-header-titles">
          <h3>Analysis Configuration</h3>
          <small>Geospatial question & multi-sensor inputs</small>
        </div>
      </div>

      <form onSubmit={onSubmit} className="setup-form">
        {/* Quick query template chips */}
        <div className="form-section">
          <label className="section-label">Quick Templates</label>
          <div className="template-pills">
            {templates.map((tpl) => (
              <button
                type="button"
                key={tpl.label}
                onClick={() => onSelectTemplate(tpl)}
                className={`template-pill ${query === tpl.query ? "active" : ""}`}
                title={tpl.query}
              >
                {tpl.label}
              </button>
            ))}
          </div>
        </div>

        {/* Workflow Pair Type */}
        <div className="form-section">
          <label className="section-label" htmlFor="pair-type-select">
            Workflow Mode
          </label>
          <div className="custom-select-wrapper">
            <select
              id="pair-type-select"
              value={pairType}
              onChange={(e) => onPairTypeChange(e.target.value)}
              className="gis-select"
            >
              <option value="single">Single Image / Visual Grounding (Optical or SAR)</option>
              <option value="bi_temporal">Bi-temporal Change Detection (Optical / Optical)</option>
              <option value="optical_sar">Multi-Modal Fusion (CROMA Optical + SAR)</option>
            </select>
          </div>
        </div>

        {/* Natural Language Query Input */}
        <div className="form-section">
          <label className="section-label" htmlFor="geo-query-input">
            Geospatial Query
          </label>
          <textarea
            id="geo-query-input"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Ask about urban expansion, water bodies, canopy index or land cover change..."
            rows={3}
            className="gis-textarea"
            required
          />
        </div>

        {/* File Drop Zones */}
        <div className="form-section file-drop-grid">
          <FileDropZone
            label="Image A (Primary / Before)"
            sublabel="Optical GeoTIFF / PNG"
            badgeText="T1 / OPTICAL"
            file={imageA}
            onFileSelect={onImageAChange}
            required
          />

          <FileDropZone
            label={pairType === "optical_sar" ? "Image B (SAR Sensor)" : "Image B (After)"}
            sublabel={pairType === "optical_sar" ? "Sentinel-1 GRD / SAR" : "Optical T2 GeoTIFF"}
            badgeText={pairType === "optical_sar" ? "T1 / SAR" : "T2 / OPTICAL"}
            file={imageB}
            onFileSelect={onImageBChange}
            required={needsImageB}
          />
        </div>

        {/* Error readout */}
        {error && (
          <div className="error-banner" role="alert">
            <span className="error-icon">⚠</span>
            <span className="error-text">{error}</span>
          </div>
        )}

        {/* Submit Execution Button */}
        <button
          type="submit"
          className="run-analysis-btn"
          disabled={busy || !imageA || (needsImageB && !imageB)}
        >
          {busy ? (
            <>
              <span className="btn-spinner" />
              <span>Analyzing Imagery...</span>
            </>
          ) : (
            <>
              <span className="btn-icon">▶</span>
              <span>Execute GeoProof Analysis</span>
            </>
          )}
        </button>
      </form>
    </aside>
  );
};
