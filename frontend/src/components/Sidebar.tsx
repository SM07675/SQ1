import React from "react";

export type NavigationTab =
  | "global"
  | "single"
  | "bi_temporal"
  | "optical_sar"
  | "archive"
  | "models"
  | "settings"
  | "reports";

interface SidebarProps {
  activeTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab }) => {
  return (
    <aside className="sidebar">
      <div className="brand-orbit" title="SatQuery GeoProof™ SIH26167">
        <span>SQ</span>
      </div>

      <nav className="nav-menu">
        <div className="nav-group-label">ENTRY POINT</div>
        <button
          type="button"
          className={`nav-item ${activeTab === "global" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("global")}
          title="Global Query-Driven Analysis"
        >
          <span className="nav-icon">✧</span>
          <span className="nav-label">Global Analysis</span>
        </button>

        <div className="nav-group-label">SPECIALIZED</div>
        <button
          type="button"
          className={`nav-item ${activeTab === "single" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("single")}
          title="Single Image Intelligence & Grounding"
        >
          <span className="nav-icon">⌖</span>
          <span className="nav-label">Single Image</span>
        </button>

        <button
          type="button"
          className={`nav-item ${activeTab === "bi_temporal" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("bi_temporal")}
          title="Bi-Temporal Change Analysis"
        >
          <span className="nav-icon">⧖</span>
          <span className="nav-label">Change Analysis</span>
        </button>

        <button
          type="button"
          className={`nav-item ${activeTab === "optical_sar" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("optical_sar")}
          title="Optical + SAR Cross-Modal Fusion"
        >
          <span className="nav-icon">◫</span>
          <span className="nav-label">Optical + SAR</span>
        </button>

        <div className="nav-group-label">PLATFORM</div>
        <button
          type="button"
          className={`nav-item ${activeTab === "archive" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("archive")}
          title="Analysis History & Geometries"
        >
          <span className="nav-icon">◷</span>
          <span className="nav-label">Analysis History</span>
        </button>

        <button
          type="button"
          className={`nav-item ${activeTab === "models" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("models")}
          title="Model Registry & Benchmark Suite"
        >
          <span className="nav-icon">⚡</span>
          <span className="nav-label">Model Registry</span>
        </button>

        <button
          type="button"
          className={`nav-item ${activeTab === "settings" ? "nav-active" : ""}`}
          onClick={() => onSelectTab("settings")}
          title="System Settings & Telemetry"
        >
          <span className="nav-icon">⚙</span>
          <span className="nav-label">Settings</span>
        </button>
      </nav>

      <div className="mission-tag">
        <span className="sih-code">SIH26167</span>
        <strong className="agency-badge">ISRO</strong>
      </div>
    </aside>
  );
};
