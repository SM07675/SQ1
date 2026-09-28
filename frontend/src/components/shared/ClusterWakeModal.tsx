import React, { useState, useEffect } from "react";
import { Server, CheckCircle2, ShieldCheck, ArrowRight, X, Cpu, HardDrive, Radio, Sparkles } from "lucide-react";
import { API_BASE } from "../../api";

interface ClusterWakeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onClusterReady: () => void;
  onExploreDemo: () => void;
  autoWakeOnMount?: boolean;
}

export const ClusterWakeModal: React.FC<ClusterWakeModalProps> = ({
  isOpen,
  onClose,
  onClusterReady,
  onExploreDemo,
}) => {
  const [stage, setStage] = useState<"checking" | "waking" | "ready" | "error">("checking");
  const [statusMessage, setStatusMessage] = useState("Checking Google Cloud Run...");
  const [currentStep, setCurrentStep] = useState<number>(1);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    let completed = false;
    setStage("waking");
    setStatusMessage("Waiting for Google Cloud Run to respond...");
    setCurrentStep(1);
    // Cloud Run starts on demand. The health endpoint is the source of truth.
    const checkHealth = async () => {
      try {
        const healthUrl = API_BASE ? `${API_BASE}/health` : "/health";
        const res = await fetch(healthUrl, {
          signal: AbortSignal.timeout(10000),
        });
        if (res.ok) {
          const data = await res.json().catch(() => null);
          if (!completed && data && (data.status === "ok" || data.environment || data.models)) {
            if (isMounted) {
              completed = true;
              setCurrentStep(4);
              setStage("ready");
              setStatusMessage("The analysis service is responding.");
              setTimeout(() => {
                if (isMounted) {
                  onClusterReady();
                  onClose();
                }
              }, 1200);
            }
          }
        }
      } catch {
        if (isMounted) setStatusMessage("The analysis service is still starting or temporarily unavailable.");
      }
    };
    void checkHealth();
    const pollInterval = setInterval(checkHealth, 5000);

    return () => {
      isMounted = false;
      clearInterval(pollInterval);
    };
  }, [isOpen, onClusterReady, onClose]);

  if (!isOpen) return null;

  const progressPercent = stage === "ready" ? 100 : 20;

  return (
    <div
      className="cluster-wake-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="cluster-wake-heading"
    >
      <div className="cluster-wake-card">
        {/* Glow ambient background effects */}
        <div className="cluster-wake-ambient-glow" />
        <div className="cluster-wake-ambient-glow-bottom" />

        {/* Top-right close button */}
        <button
          type="button"
          className="cluster-wake-close-btn"
          onClick={onClose}
          aria-label="Close modal"
          title="Dismiss (Esc)"
        >
          <X size={16} />
        </button>

        {/* Header */}
        <div className="cluster-wake-header">
          <div className={`cluster-wake-icon-box ${stage === "ready" ? "ready" : ""}`}>
            {stage === "ready" ? (
              <CheckCircle2 size={26} />
            ) : (
              <Radio size={24} className="cluster-pulse-icon" />
            )}
          </div>
          <div className="cluster-wake-meta">
            <div className="cluster-wake-title-row">
              <h3 id="cluster-wake-heading" className="cluster-wake-title">
                {stage === "ready"
                  ? "SatQuery AI System Ready"
                  : "Checking analysis service..."}
              </h3>
              <span className={`cluster-wake-scale-tag ${stage === "ready" ? "ready" : ""}`}>
                {stage === "ready" ? "Online" : "Checking"}
              </span>
            </div>
            <p className="cluster-wake-subtitle">
              {stage === "ready"
                ? "Cloud Run health check passed · Ready for your query"
                : "Connecting to Google Cloud Run"}
            </p>
          </div>
        </div>

        {/* Stepper overview */}
        <div className="cluster-wake-steps-grid">
          <div className={`cluster-step-item ${currentStep >= 1 ? "active" : ""} ${currentStep > 1 || stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {currentStep > 1 || stage === "ready" ? <CheckCircle2 size={12} /> : "1"}
            </div>
            <div className="cluster-step-label">
              <span>Connect</span>
              <small>Cloud Run</small>
            </div>
          </div>

          <div className={`cluster-step-item ${currentStep >= 2 ? "active" : ""} ${currentStep > 2 || stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {currentStep > 2 || stage === "ready" ? <CheckCircle2 size={12} /> : "2"}
            </div>
            <div className="cluster-step-label">
              <span>Health Check</span>
              <small>Service response</small>
            </div>
          </div>

          <div className={`cluster-step-item ${currentStep >= 3 ? "active" : ""} ${currentStep > 3 || stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {currentStep > 3 || stage === "ready" ? <CheckCircle2 size={12} /> : "3"}
            </div>
            <div className="cluster-step-label">
              <span>Analysis API</span>
              <small>Availability</small>
            </div>
          </div>

          <div className={`cluster-step-item ${stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {stage === "ready" ? <CheckCircle2 size={12} /> : "4"}
            </div>
            <div className="cluster-step-label">
              <span>Ready</span>
              <small>Continue analysis</small>
            </div>
          </div>
        </div>

        {/* Informative message box */}
        <div className="cluster-wake-info-box">
          <div className="cluster-wake-info-badge">
            <ShieldCheck size={14} />
            <span>Automatic Cloud Scaling</span>
          </div>
          {stage === "ready" ? (
            <span style={{ color: "#34d399", fontWeight: 550 }}>
              The AI compute engine is live and verified. You can now immediately enter your natural-language satellite queries.
            </span>
          ) : (
            <span>
              Cloud Run starts instances when needed. This check verifies that the analysis API is responding.
            </span>
          )}
        </div>

        {/* Live Status Indicator */}
        <div className="cluster-wake-status-block">
          <div className="cluster-wake-status-row">
            <span className="cluster-wake-status-text">
              {stage === "ready" ? (
                <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", color: "#34d399", fontWeight: 650 }}>
                  <Sparkles size={14} />
                  System Ready! Enter your query below.
                </span>
              ) : (
                statusMessage
              )}
            </span>
          </div>

          {/* Progress bar */}
          <div className="cluster-wake-progress-track">
            <div
              className={`cluster-wake-progress-bar ${stage === "ready" ? "ready" : ""}`}
              style={{
                width: `${progressPercent}%`,
              }}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="cluster-wake-actions">
          <button
            type="button"
            onClick={() => {
              onExploreDemo();
              onClose();
            }}
            className="cluster-wake-btn-primary"
          >
            <span>Browse Interactive Demo Instantly</span>
            <ArrowRight size={14} />
          </button>

          <button
            type="button"
            onClick={onClose}
            className="cluster-wake-btn-dismiss"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
};

export default ClusterWakeModal;
