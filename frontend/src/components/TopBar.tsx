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
          aria-selected={activeTab === "analysis"}
          className={`tab-btn ${activeTab === "analysis" ? "active" : ""}`}
          onClick={() => onSelectTab("analysis")}
        >
          <span className="tab-dot" />
          Studio
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "benchmarks"}
          className={`tab-btn ${activeTab === "benchmarks" ? "active" : ""}`}
          onClick={() => onSelectTab("benchmarks")}
        >
          <span className="tab-dot" />
          Benchmarks
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "archive"}
          className={`tab-btn ${activeTab === "archive" ? "active" : ""}`}
          onClick={() => onSelectTab("archive")}
        >
          <span className="tab-dot" />
          Archive
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === "reports"}
          className={`tab-btn ${activeTab === "reports" ? "active" : ""}`}
          onClick={() => onSelectTab("reports")}
        >
          <span className="tab-dot" />
          Audit Reports
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
