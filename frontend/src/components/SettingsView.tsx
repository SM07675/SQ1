import React from "react";

export const SettingsView: React.FC = () => {
  return (
    <div className="settings-view-container">
      <div className="panel settings-panel">
        <div className="panel-header">
          <span className="panel-step-badge">SYSTEM CONFIG</span>
          <h3>SatQuery GeoProof™ System Settings</h3>
        </div>

        <div className="settings-content-grid">
          <div className="setting-card">
            <h4>🛰 Problem Statement & Challenge</h4>
            <p>
              <strong>SIH Code:</strong> SIH26167
            </p>
            <p>
              <strong>Agency:</strong> Indian Space Research Organisation (ISRO)
            </p>
            <p>
              <strong>Objective:</strong> Autonomous multimodal remote-sensing inquiry with verifiable, multi-witness GeoProof arbitration.
            </p>
          </div>

          <div className="setting-card">
            <h4>⚙ Model Orchestration Settings</h4>
            <div className="settings-rows">
              <div className="setting-row">
                <span>Tile Window Dimensions:</span>
                <strong>448 × 448 px (50% Overlap)</strong>
              </div>
              <div className="setting-row">
                <span>Normalization Strategy:</span>
                <strong>Percentile (2% - 98%) Per-Band</strong>
              </div>
              <div className="setting-row">
                <span>Confidence Calibration Mode:</span>
                <strong>Temperature-Scaled Platt Calibration</strong>
              </div>
              <div className="setting-row">
                <span>Multi-Witness Protocol:</span>
                <strong>Ensemble Radiometric + Spectral Agreement</strong>
              </div>
            </div>
          </div>

          <div className="setting-card">
            <h4>📁 Persistence & Telemetry Registry</h4>
            <div className="settings-rows">
              <div className="setting-row">
                <span>Database Engine:</span>
                <strong>SQLite with Spatial Geometry Records</strong>
              </div>
              <div className="setting-row">
                <span>Artifact Storage:</span>
                <strong>artifacts/ (GeoJSON, Masks, PNGs, PDF Reports)</strong>
              </div>
              <div className="setting-row">
                <span>Audit Report Generator:</span>
                <strong>ReportLab High-Res Vector PDF</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
