import React, { useState, useEffect } from "react";
import { Check, Loader2 } from "lucide-react";

interface AnalysisProgressProps {
  busy: boolean;
  hasAttachedImages: boolean;
}

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
  busy,
  hasAttachedImages,
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = hasAttachedImages
    ? [
        "Referencing spatial raster context",
        "Evaluating multi-witness evidence layers",
        "Formulating verified answer",
      ]
    : [
        "Raster format & CRS validated",
        "Selecting optimal spatial models",
        "Analyzing spectral & geometric patterns",
        "Synthesizing GeoProof findings",
      ];

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (busy) {
      setCurrentStep(0);
      interval = setInterval(() => {
        setCurrentStep((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
      }, 1200);
    }
    return () => clearInterval(interval);
  }, [busy, steps.length]);

  if (!busy) return null;

  return (
    <div className="analysis-progress-card">
      <div className="progress-header">
        <Loader2 size={15} className="progress-spinner-icon" />
        <span className="progress-title">Analyzing satellite imagery…</span>
      </div>

      <div className="progress-steps-list">
        {steps.map((step, idx) => {
          const isDone = idx < currentStep;
          const isCurrent = idx === currentStep;

          return (
            <div
              key={step}
              className={`progress-step-item ${isDone ? "done" : isCurrent ? "current" : "pending"}`}
            >
              <div className="step-marker">
                {isDone ? (
                  <Check size={11} className="step-check-icon" />
                ) : isCurrent ? (
                  <div className="step-pulse-dot" />
                ) : (
                  <div className="step-empty-dot" />
                )}
              </div>
              <span className="step-label">{step}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
