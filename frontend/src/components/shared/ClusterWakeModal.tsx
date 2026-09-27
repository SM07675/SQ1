import React, { useState, useEffect } from "react";
import { Server, CheckCircle2, ShieldCheck, ArrowRight, X, Cpu, HardDrive, Radio, Sparkles } from "lucide-react";

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
  const [countdown, setCountdown] = useState(35);
  const [statusMessage, setStatusMessage] = useState("Checking cloud node status...");
  const [nodeCloudStatus, setNodeCloudStatus] = useState<string>("Starting");
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
    setStage("waking");
    setStatusMessage("Sending wake signal to E2E 16 GB Node...");
    setCountdown(35);
    setCurrentStep(1);

    // 1. Dispatch Wake call to Vercel Serverless Function / E2E API
    const triggerWake = async () => {
      try {
        const res = await fetch("/e2e/wake", { method: "POST" }).catch(() =>
          fetch("/api/e2e?action=wake", { method: "POST" })
        );
        if (res && res.ok) {
          const data = await res.json().catch(() => null);
          if (isMounted) {
            setCurrentStep(2);
            if (data?.message?.includes("already running")) {
              setStatusMessage("Node is already active! Booting models into RAM...");
              setNodeCloudStatus("Running");
            } else {
              setStatusMessage("Power-on signal received by E2E Cloud. Booting Linux instance...");
              setNodeCloudStatus("Starting");
            }
          }
        }
      } catch (err) {
        console.warn("Auto-wake dispatch error:", err);
      }
    };

    triggerWake();

    // 2. Poll /e2e/status to track cloud VM state
    const statusPollInterval = setInterval(async () => {
      try {
        const res = await fetch("/e2e/status").catch(() =>
          fetch("/api/e2e?action=status")
        );
        if (res && res.ok) {
          const data = await res.json().catch(() => null);
          if (isMounted && data?.status) {
            setNodeCloudStatus(data.status);
            if (data.status === "Running" && currentStep < 3) {
              setCurrentStep(3);
              setStatusMessage("Compute node is live! Initializing Docker & loading 11 models into RAM...");
            }
          }
        }
      } catch {
        // ignore
      }
    }, 4000);

    // 3. Poll /health every 2.5 seconds
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch("/health", {
          signal: AbortSignal.timeout(2800),
        });
        if (res.ok) {
          const data = await res.json().catch(() => null);
          if (data && (data.status === "ok" || data.environment || data.models)) {
            clearInterval(pollInterval);
            clearInterval(statusPollInterval);
            if (isMounted) {
              setCurrentStep(4);
              setStage("ready");
              setCountdown(0);
              setStatusMessage("All 11 Geospatial Vision Models loaded into RAM! System Ready.");
              setTimeout(() => {
                onClusterReady();
                onClose();
              }, 1200);
            }
          }
        }
      } catch {
        // Still booting, keep polling
      }
    }, 2500);

    // 4. Countdown timer & step advancement
    const timerInterval = setInterval(() => {
      setCountdown((prev) => {
        const next = prev > 0 ? prev - 1 : 0;
        if (isMounted) {
          if (next <= 22 && next > 10 && currentStep < 3) {
            setCurrentStep(3);
            setStatusMessage("Starting Docker container & mounting neural model registry...");
          } else if (next <= 10 && next > 0 && currentStep < 4) {
            setCurrentStep(3);
            setStatusMessage("Loading PyTorch runtime & 11 Geospatial Vision Models into RAM...");
          }
        }
        return next;
      });
    }, 1000);

    return () => {
      isMounted = false;
      clearInterval(pollInterval);
      clearInterval(statusPollInterval);
      clearInterval(timerInterval);
    };
  }, [isOpen, onClusterReady, onClose]);

  if (!isOpen) return null;

  const progressPercent =
    stage === "ready"
      ? 100
      : Math.min(95, Math.max(12, Math.round(((35 - countdown) / 35) * 100)));

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
                  : "Waiting for system to start..."}
              </h3>
              <span className={`cluster-wake-scale-tag ${stage === "ready" ? "ready" : ""}`}>
                {stage === "ready" ? "Online" : "Starting Up"}
              </span>
            </div>
            <p className="cluster-wake-subtitle">
              {stage === "ready"
                ? "All 11 Models Loaded · Ready for satellite query"
                : "Waking E2E Networks Node C3-16GB-578 (Chennai, India)"}
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
              <span>Wake Command</span>
              <small>E2E Cloud API</small>
            </div>
          </div>

          <div className={`cluster-step-item ${currentStep >= 2 ? "active" : ""} ${currentStep > 2 || stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {currentStep > 2 || stage === "ready" ? <CheckCircle2 size={12} /> : "2"}
            </div>
            <div className="cluster-step-label">
              <span>Linux Instance</span>
              <small>{nodeCloudStatus === "Running" ? "Running" : "Booting"}</small>
            </div>
          </div>

          <div className={`cluster-step-item ${currentStep >= 3 ? "active" : ""} ${currentStep > 3 || stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {currentStep > 3 || stage === "ready" ? <CheckCircle2 size={12} /> : "3"}
            </div>
            <div className="cluster-step-label">
              <span>11 Models in RAM</span>
              <small>PyTorch & GDAL</small>
            </div>
          </div>

          <div className={`cluster-step-item ${stage === "ready" ? "completed" : ""}`}>
            <div className="cluster-step-bullet">
              {stage === "ready" ? <CheckCircle2 size={12} /> : "4"}
            </div>
            <div className="cluster-step-label">
              <span>Inference Ready</span>
              <small>FastAPI /health</small>
            </div>
          </div>
        </div>

        {/* Informative message box */}
        <div className="cluster-wake-info-box">
          <div className="cluster-wake-info-badge">
            <ShieldCheck size={14} />
            <span>Energy-Saving Architecture Active</span>
          </div>
          {stage === "ready" ? (
            <span style={{ color: "#34d399", fontWeight: 550 }}>
              The AI compute engine is live and verified. You can now immediately enter your natural-language satellite queries.
            </span>
          ) : (
            <span>
              This node automatically shuts down when idle to conserve compute resources. We are spinning it up now with all 11 geospatial vision models ready in memory.
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
            {stage !== "ready" && (
              <span className="cluster-wake-countdown">~{countdown}s remaining</span>
            )}
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
