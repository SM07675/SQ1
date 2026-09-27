import React, { useState, useEffect } from "react";
import { Server, CheckCircle2, ShieldCheck, ArrowRight, X } from "lucide-react";

interface ClusterWakeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onClusterReady: () => void;
  onExploreDemo: () => void;
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

    // 1. Dispatch Wake call to Vercel Serverless Function
    fetch("/e2e/wake")
      .then((r) => r.json())
      .then(() => {
        if (!isMounted) return;
        setStatusMessage("Node power-on initiated. Starting Docker container...");
      })
      .catch((err) => {
        console.warn("Auto-wake dispatch error:", err);
      });

    // 2. Poll /health every 3.5 seconds
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch("/health", {
          signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.status === "ok" || data.environment) {
            clearInterval(pollInterval);
            if (isMounted) {
              setStage("ready");
              setStatusMessage("Dedicated AI Node is online & all 11 models loaded!");
              setTimeout(() => {
                onClusterReady();
                onClose();
              }, 1800);
            }
          }
        }
      } catch {
        // Still booting, keep polling
      }
    }, 3500);

    // 3. Countdown timer
    const timerInterval = setInterval(() => {
      setCountdown((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);

    return () => {
      isMounted = false;
      clearInterval(pollInterval);
      clearInterval(timerInterval);
    };
  }, [isOpen, onClusterReady, onClose]);

  if (!isOpen) return null;

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
              <CheckCircle2 size={24} />
            ) : (
              <Server size={24} />
            )}
          </div>
          <div className="cluster-wake-meta">
            <div className="cluster-wake-title-row">
              <h3 id="cluster-wake-heading" className="cluster-wake-title">
                {stage === "ready" ? "AI Cluster Online" : "Waking SatQuery AI Cluster"}
              </h3>
              <span className="cluster-wake-scale-tag">
                Scale-to-Zero
              </span>
            </div>
            <p className="cluster-wake-subtitle">E2E Networks Node C3-16GB-578 (Chennai, India)</p>
          </div>
        </div>

        {/* Informative message box */}
        <div className="cluster-wake-info-box">
          <div className="cluster-wake-info-badge">
            <ShieldCheck size={14} />
            <span>Energy-Saving Architecture Active</span>
          </div>
          This node shuts down automatically when idle to conserve compute resources. Waking up launches PyTorch, GDAL, and all 11 geospatial models into RAM.
        </div>

        {/* Live Status Indicator */}
        <div className="cluster-wake-status-block">
          <div className="cluster-wake-status-row">
            <span className="cluster-wake-status-text">{statusMessage}</span>
            {stage !== "ready" && (
              <span className="cluster-wake-countdown">~{countdown}s remaining</span>
            )}
          </div>

          {/* Progress bar */}
          <div className="cluster-wake-progress-track">
            <div
              className={`cluster-wake-progress-bar ${stage === "ready" ? "ready" : ""}`}
              style={{
                width: stage === "ready" ? "100%" : `${Math.min(100, Math.max(10, ((35 - countdown) / 35) * 100))}%`,
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
