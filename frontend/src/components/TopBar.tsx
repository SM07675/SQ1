import React from "react";

interface TopBarProps {
  activeTab: "analysis" | "benchmarks";
  onSelectTab: (tab: "analysis" | "benchmarks") => void;
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
          aria-selected={activeTab === "analysis"}
          className={`tab-btn ${activeTab === "analysis" ? "active" : ""}`}
          onClick={() => onSelectTab("analysis")}
        >
          <span className="tab-dot" />
          Analysis Studio
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "benchmarks"}
          className={`tab-btn ${activeTab === "benchmarks" ? "active" : ""}`}
          onClick={() => onSelectTab("benchmarks")}
        >
          <span className="tab-dot" />
          Benchmark Dashboard (Gates 6 & 7)
        </button>
      </div>

      <div className="system-state">
        <span className={`pulse ${isEngineOnline ? "online" : "offline"}`} />
        <div className="system-state-text">
          <strong className="engine-status">
            {isEngineOnline ? "ANALYSIS ENGINE ONLINE" : "ENGINE OFFLINE"}
          </strong>
          <small className="engine-mode">
            {analysisMode ? analysisMode.replace(/_/g, " ").toUpperCase() : "Awaiting inputs"}
          </small>
        </div>
      </div>
    </header>
  );
};
