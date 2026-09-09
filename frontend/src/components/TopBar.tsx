import React from "react";
import type { NavigationTab } from "./Sidebar";

interface TopBarProps {
  activeTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
  analysisMode?: string;
  isEngineOnline?: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  activeTab,
  onSelectTab,
  analysisMode,
  isEngineOnline = true,
}) => {
  return (
    <header className="hero-bar">
      <div className="brand-group">
        <p className="eyebrow">EVIDENCE-FIRST EARTH INTELLIGENCE</p>
        <h1>
          SatQuery <span>GeoProof</span><sup>™</sup>
        </h1>
        <p className="tagline">Every satellite-AI answer must prove itself with grounded telemetry.</p>
      </div>

      <div className="tab-switcher" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "global"}
          className={`tab-btn ${activeTab === "global" ? "active" : ""}`}
          onClick={() => onSelectTab("global")}
        >
          <span className="tab-dot" />
          Global Analysis
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "single" || activeTab === "bi_temporal" || activeTab === "optical_sar"}
          className={`tab-btn ${activeTab === "single" || activeTab === "bi_temporal" || activeTab === "optical_sar" ? "active" : ""}`}
          onClick={() => onSelectTab("single")}
        >
          <span className="tab-dot" />
          Specialized Workflows
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "models"}
          className={`tab-btn ${activeTab === "models" ? "active" : ""}`}
          onClick={() => onSelectTab("models")}
        >
          <span className="tab-dot" />
          Model Registry
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "archive"}
          className={`tab-btn ${activeTab === "archive" ? "active" : ""}`}
          onClick={() => onSelectTab("archive")}
        >
          <span className="tab-dot" />
          History
        </button>
      </div>

      <div className="system-state">
        <span className={`pulse ${isEngineOnline ? "online" : "offline"}`} />
        <div className="system-state-text">
          <strong className="engine-status">
            {isEngineOnline ? "ANALYSIS ENGINE ONLINE" : "ENGINE OFFLINE"}
          </strong>
          <small className="engine-mode">
            {analysisMode ? analysisMode.replace(/_/g, " ").toUpperCase() : activeTab.replace(/_/g, " ").toUpperCase()}
          </small>
        </div>
      </div>
    </header>
  );
};
