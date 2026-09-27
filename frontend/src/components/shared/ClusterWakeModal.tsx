import React, { useState, useEffect } from "react";
import { Server, Zap, Moon, CheckCircle2, AlertTriangle, ArrowRight, ShieldCheck } from "lucide-react";

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

  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    setStage("waking");
    setStatusMessage("Sending wake signal to E2E 16 GB Node...");
    setCountdown(35);

    // 1. Dispatch Wake call to Vercel Serverless Function
    fetch("/e2e/wake")
      .then((r) => r.json())
      .then((data) => {
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
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-fade-in">
      <div className="relative w-full max-w-lg p-6 overflow-hidden rounded-2xl bg-[#0f172a] border border-cyan-500/30 shadow-2xl text-white">
        {/* Glow ambient background */}
        <div className="absolute -top-24 -left-24 w-48 h-48 bg-cyan-500/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -right-24 w-48 h-48 bg-emerald-500/20 rounded-full blur-3xl pointer-events-none" />

        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            {stage === "ready" ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-400 animate-bounce" />
            ) : (
              <Server className="w-6 h-6 text-cyan-400 animate-pulse" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-slate-100">
                {stage === "ready" ? "AI Cluster Online" : "Waking SatQuery AI Cluster"}
              </h3>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                Scale-to-Zero
              </span>
            </div>
            <p className="text-xs text-slate-400">E2E Networks Node C3-16GB-578 (Chennai, India)</p>
          </div>
        </div>

        {/* Informative message */}
        <div className="p-3 mb-5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300 leading-relaxed">
          <div className="flex items-center gap-1.5 text-cyan-300 font-medium mb-1">
            <ShieldCheck size={14} />
            <span>Energy-Saving Architecture Active</span>
          </div>
          This node shuts down automatically when idle to conserve compute resources. Waking up launches PyTorch, GDAL, and all 11 geospatial models into RAM.
        </div>

        {/* Live Status Indicator */}
        <div className="mb-6 space-y-2">
          <div className="flex justify-between text-xs text-slate-300 font-medium">
            <span>{statusMessage}</span>
            {stage !== "ready" && (
              <span className="text-cyan-400 font-mono">~{countdown}s remaining</span>
            )}
          </div>

          {/* Progress bar */}
          <div className="w-full h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full transition-all duration-1000 ease-out ${
                stage === "ready"
                  ? "w-full bg-emerald-400"
                  : "bg-gradient-to-r from-cyan-500 via-sky-400 to-indigo-500"
              }`}
              style={{
                width: stage === "ready" ? "100%" : `${Math.min(100, Math.max(10, ((35 - countdown) / 35) * 100))}%`,
              }}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-2.5 pt-2">
          <button
            type="button"
            onClick={() => {
              onExploreDemo();
              onClose();
            }}
            className="flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition border border-slate-700"
          >
            <span>Browse Interactive Demo Instantly</span>
            <ArrowRight size={14} />
          </button>

          <button
            type="button"
            onClick={onClose}
            className="py-2.5 px-4 rounded-xl bg-slate-900/60 hover:bg-slate-800 text-slate-400 text-xs transition border border-slate-800"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
};
