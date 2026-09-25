import React, { useState, useEffect } from "react";
import {
  X,
  Settings,
  Sun,
  Moon,
  Monitor,
  Cpu,
  Layers,
  CheckCircle2,
  Sliders,
  Check,
} from "lucide-react";
import type { ThemeMode, ModelCapability } from "../../types";
import { fetchModelStatus } from "../../api";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  theme,
  onToggleTheme,
}) => {
  const [activeSection, setActiveSection] = useState<"appearance" | "analysis" | "advanced">("appearance");
  const [models, setModels] = useState<ModelCapability[]>([]);
  const [modelRoot, setModelRoot] = useState<string>("");

  useEffect(() => {
    if (isOpen) {
      fetchModelStatus()
        .then((res) => {
          setModels(res.models || []);
          setModelRoot(res.model_root || "");
        })
        .catch(() => {});
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="settings-dialog-box" onClick={(e) => e.stopPropagation()}>
        <div className="settings-dialog-header">
          <div className="settings-header-left">
            <Settings size={18} className="settings-header-icon" />
            <h3 className="settings-dialog-title">Settings</h3>
          </div>
          <button
            type="button"
            className="drawer-close-btn"
            onClick={onClose}
            title="Close Settings (Esc)"
          >
            <X size={16} />
          </button>
        </div>

        <div className="settings-dialog-body">
          {/* Settings Sidebar Nav */}
          <div className="settings-dialog-nav">
            <button
              type="button"
              className={`settings-nav-btn ${activeSection === "appearance" ? "active" : ""}`}
              onClick={() => setActiveSection("appearance")}
            >
              <Sun size={15} />
              <span>Appearance</span>
            </button>

            <button
              type="button"
              className={`settings-nav-btn ${activeSection === "analysis" ? "active" : ""}`}
              onClick={() => setActiveSection("analysis")}
            >
              <Sliders size={15} />
              <span>Analysis Display</span>
            </button>

            <button
              type="button"
              className={`settings-nav-btn ${activeSection === "advanced" ? "active" : ""}`}
              onClick={() => setActiveSection("advanced")}
            >
              <Cpu size={15} />
              <span>Advanced & Models</span>
            </button>
          </div>

          {/* Settings Content Area */}
          <div className="settings-dialog-content">
            {activeSection === "appearance" && (
              <div className="settings-pane">
                <h4 className="pane-title">Theme Preference</h4>
                <p className="pane-desc">Choose between SatQuery Dark Orbit and Clean Light Mode.</p>

                <div className="theme-options-grid">
                  <div
                    className={`theme-card ${theme === "dark" ? "active" : ""}`}
                    onClick={() => theme !== "dark" && onToggleTheme()}
                  >
                    <div className="theme-preview dark">
                      <div className="theme-preview-nav" />
                      <div className="theme-preview-body" />
                    </div>
                    <div className="theme-card-label">
                      <Moon size={14} />
                      <span>Dark Orbit</span>
                      {theme === "dark" && <Check size={14} className="theme-check-icon" />}
                    </div>
                  </div>

                  <div
                    className={`theme-card ${theme === "light" ? "active" : ""}`}
                    onClick={() => theme !== "light" && onToggleTheme()}
                  >
                    <div className="theme-preview light">
                      <div className="theme-preview-nav" />
                      <div className="theme-preview-body" />
                    </div>
                    <div className="theme-card-label">
                      <Sun size={14} />
                      <span>Light Clean</span>
                      {theme === "light" && <Check size={14} className="theme-check-icon" />}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeSection === "analysis" && (
              <div className="settings-pane">
                <h4 className="pane-title">Visualization & Confidence</h4>
                <p className="pane-desc">Configure default satellite imagery presentation options.</p>

                <div className="setting-toggle-row">
                  <div>
                    <strong>Calibrated Platt Confidence</strong>
                    <p className="setting-subtext">Display temperature-scaled confidence intervals</p>
                  </div>
                  <span className="setting-badge-active">Enabled</span>
                </div>

                <div className="setting-toggle-row">
                  <div>
                    <strong>Aspect Ratio Preservation</strong>
                    <p className="setting-subtext">Render satellite rasters at native sensor aspect ratio</p>
                  </div>
                  <span className="setting-badge-active">Natural</span>
                </div>

                <div className="setting-toggle-row">
                  <div>
                    <strong>Dual-Witness Arbitration</strong>
                    <p className="setting-subtext">Cross-verify deep learning findings with spectral indices</p>
                  </div>
                  <span className="setting-badge-active">Active</span>
                </div>
              </div>
            )}

            {activeSection === "advanced" && (
              <div className="settings-pane">
                <h4 className="pane-title">Model Registry Status</h4>
                <p className="pane-desc">Available neural weights, vision backbones, and spectral witnesses.</p>

                {modelRoot && (
                  <div className="model-root-badge">
                    <span>Model Root:</span> <code>{modelRoot}</code>
                  </div>
                )}

                <div className="models-status-list">
                  {models.length === 0 ? (
                    <p className="empty-models-text">Loading model capabilities...</p>
                  ) : (
                    models.map((m) => (
                      <div key={m.name} className="model-status-item">
                        <div className="model-item-top">
                          <span className="model-name">{m.name}</span>
                          <span className={`model-avail-badge ${m.available ? "online" : "offline"}`}>
                            {m.available ? "Available" : "Standby"}
                          </span>
                        </div>
                        <p className="model-purpose">{m.purpose}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
