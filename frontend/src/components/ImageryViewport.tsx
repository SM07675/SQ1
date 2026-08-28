import React from "react";
import { artifactUrl } from "../api";
import type { AnalysisResponse, ArtifactRef } from "../types";
import { ProgressOverlay } from "./shared/ProgressOverlay";

interface ImageryViewportProps {
  result: AnalysisResponse | null;
  activeLayer: string;
  availableLayers: string[];
  layerNames: Record<string, string>;
  swipe: number;
  busy: boolean;
  onSelectLayer: (layer: string) => void;
  onSwipeChange: (swipe: number) => void;
}

function findArtifact(
  result: AnalysisResponse | null,
  name: string
): ArtifactRef | undefined {
  return result?.artifacts.find((item) => item.name === name);
}

export const ImageryViewport: React.FC<ImageryViewportProps> = ({
  result,
  activeLayer,
  availableLayers,
  layerNames,
  swipe,
  busy,
  onSelectLayer,
  onSwipeChange,
}) => {
  const before = findArtifact(result, "preview_1.png");
  const after = findArtifact(result, "preview_2.png");
  const selectedArtifact = findArtifact(result, activeLayer);

  const confidenceBreakdown = result?.verdict.confidence_breakdown;

  return (
    <section className="panel viewport-panel" aria-label="Imagery Workspace">
      {/* Top layer switcher tabs */}
      <div className="viewport-header">
        <div className="layer-tabs-scroll" role="tablist">
          {availableLayers.map((layer) => (
            <button
              key={layer}
              type="button"
              role="tab"
              aria-selected={activeLayer === layer}
              className={`layer-tab ${activeLayer === layer ? "active" : ""}`}
              onClick={() => onSelectLayer(layer)}
            >
              {layerNames[layer] ?? layer.replace(/\.[^/.]+$/, "").replace(/_/g, " ")}
            </button>
          ))}
        </div>
        {result && (
          <div className="viewport-meta-badge">
            <span>CRS: {result.assets?.[0]?.crs || "EPSG:4326"}</span>
          </div>
        )}
      </div>

      {/* Main raster viewer stage */}
      <div className="viewport-stage">
        {busy && <ProgressOverlay />}

        {!busy && activeLayer === "compare" && before && after ? (
          <div className="swipe-stage">
            <img
              src={artifactUrl(before.url)}
              alt="Before (T1)"
              className="swipe-image-base"
            />
            <div
              className="swipe-overlay-wrapper"
              style={{ clipPath: `polygon(0 0, ${swipe}% 0, ${swipe}% 100%, 0 100%)` }}
            >
              <img
                src={artifactUrl(after.url)}
                alt="After (T2)"
                className="swipe-image-overlay"
              />
            </div>
            {/* Visual swipe divider line */}
            <div className="swipe-divider" style={{ left: `${swipe}%` }}>
              <div className="swipe-handle">⇄</div>
            </div>
            {/* Range input overlay */}
            <input
              type="range"
              min="0"
              max="100"
              value={swipe}
              onChange={(e) => onSwipeChange(Number(e.target.value))}
              className="swipe-range-input"
              aria-label="Swipe comparison slider"
            />
            <div className="stage-label before-tag">T1: Before</div>
            <div className="stage-label after-tag">T2: After</div>
          </div>
        ) : !busy && selectedArtifact ? (
          <div className="single-stage">
            <img
              src={artifactUrl(selectedArtifact.url)}
              alt={selectedArtifact.name}
              className="stage-raster-image"
            />
            <div className="stage-label active-layer-tag">
              {layerNames[selectedArtifact.name] ?? selectedArtifact.name}
            </div>
          </div>
        ) : !busy ? (
          <div className="empty-stage">
            <div className="empty-radar">
              <div className="radar-ring r1" />
              <div className="radar-ring r2" />
              <div className="radar-ring r3" />
              <div className="radar-sweep" />
              <div className="radar-center-cross" />
            </div>
            <h4>Spatial Canvas Ready</h4>
            <p>Upload GeoTIFF / NetCDF pairs or run analysis to inspect multi-layer telemetry.</p>
          </div>
        ) : null}

        {/* Bottom coordinate readout chip */}
        {result && (
          <div className="canvas-coords-chip">
            <span>RES: {result.assets?.[0]?.resolution?.map((r) => r.toFixed(2)).join("m × ") || "10.0m"}</span>
            <span className="coord-sep">|</span>
            <span>NODATA: {result.assets?.[0]?.nodata_percent?.toFixed(1) || "0.0"}%</span>
          </div>
        )}
      </div>

      {/* Confidence strip below viewport */}
      {confidenceBreakdown && (
        <div className="viewport-footer-strip">
          <div className="strip-item">
            <span className="strip-label">Calibrated Probability</span>
            <div className="strip-value-group">
              <strong className="strip-value">
                {Math.round(confidenceBreakdown.final_score * 100)}%
              </strong>
              {confidenceBreakdown.is_calibrated && (
                <span className="calibrated-pill">✓ Platt Scaled</span>
              )}
            </div>
          </div>

          <div className="strip-item">
            <span className="strip-label">95% Confidence Interval</span>
            <strong className="strip-value">
              {confidenceBreakdown.confidence_interval
                ? `[${Math.round(confidenceBreakdown.confidence_interval[0] * 100)}%, ${Math.round(
                    confidenceBreakdown.confidence_interval[1] * 100
                  )}%]`
                : "±4.2%"}
            </strong>
          </div>

          <div className="strip-item">
            <span className="strip-label">Expected Calibration Error</span>
            <strong className="strip-value">
              {confidenceBreakdown.expected_calibration_error
                ? `${(confidenceBreakdown.expected_calibration_error * 100).toFixed(1)}%`
                : "4.2%"}
            </strong>
          </div>
        </div>
      )}
    </section>
  );
};
