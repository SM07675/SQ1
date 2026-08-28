import React from "react";

interface SidebarProps {
  activeTab: "analysis" | "benchmarks";
  onSelectTab: (tab: "analysis" | "benchmarks") => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab }) => {
  return (
    <aside className="sidebar">
      <div className="brand-orbit" title="SatQuery GeoProof™">
        <span>SQ</span>
      </div>

      <nav className="nav-menu">
        <button
          type="button"
          className={`nav-item ${activeTab === "analysis" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("analysis")}
          title="Analysis Studio"
        >
          <span className="nav-icon">⌁</span>
          <span className="nav-label">Analyze</span>
        </button>

        <button
          type="button"
          className={`nav-item ${activeTab === "benchmarks" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("benchmarks")}
          title="Benchmark Evaluation & Calibration"
        >
          <span className="nav-icon">◫</span>
          <span className="nav-label">Benchmarks</span>
        </button>

        <button
          type="button"
          className="nav-item"
          onClick={() => onSelectTab("analysis")}
          title="Spatial Evidence Archive"
        >
          <span className="nav-icon">⌖</span>
          <span className="nav-label">Archive</span>
        </button>

        <button
          type="button"
          className="nav-item"
          onClick={() => onSelectTab("analysis")}
          title="Audit Reports"
        >
          <span className="nav-icon">⇩</span>
          <span className="nav-label">Reports</span>
        </button>
      </nav>

      <div className="mission-tag">
        <span className="sih-code">SIH26167</span>
        <strong className="agency-badge">ISRO</strong>
      </div>
    </aside>
  );
};
