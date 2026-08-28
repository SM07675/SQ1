import React from "react";

interface ProgressOverlayProps {
  statusText?: string;
  subText?: string;
}

export const ProgressOverlay: React.FC<ProgressOverlayProps> = ({
  statusText = "Analyzing Satellite Imagery...",
  subText = "Running multi-modal verification & confidence calibration",
}) => {
  return (
    <div className="progress-overlay" role="status" aria-live="polite">
      <div className="radar-scanner">
        <div className="radar-grid" />
        <div className="radar-sweep" />
        <div className="radar-blip blip-1" />
        <div className="radar-blip blip-2" />
        <div className="radar-center-cross" />
      </div>
      <div className="progress-details">
        <div className="progress-spinner-ring" />
        <h4>{statusText}</h4>
        <p>{subText}</p>
        <div className="progress-stepper">
          <span>Alignment</span>
          <span className="step-sep">→</span>
          <span>Spectral / CD</span>
          <span className="step-sep">→</span>
          <span>Fusion</span>
          <span className="step-sep">→</span>
          <span>Platt Calibration</span>
        </div>
      </div>
    </div>
  );
};
