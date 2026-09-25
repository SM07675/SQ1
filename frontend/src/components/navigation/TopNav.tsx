import React from "react";
import {
  Sparkles,
  Layers,
  GitCompare,
  Compass,
  FileText,
  Search,
  Sun,
  Moon,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Command,
} from "lucide-react";
import type { NavTab, ThemeMode } from "../../types";

interface TopNavProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  isHistoryOpen: boolean;
  onToggleHistory: () => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onOpenCommandPalette: () => void;
  onOpenSettings: () => void;
}

export const TopNav: React.FC<TopNavProps> = ({
  activeTab,
  onSelectTab,
  isHistoryOpen,
  onToggleHistory,
  theme,
  onToggleTheme,
  onOpenCommandPalette,
  onOpenSettings,
}) => {
  return (
    <header className="satquery-topnav">
      <div className="topnav-left">
        <button
          type="button"
          className="topnav-btn history-toggle-btn"
          onClick={onToggleHistory}
          title={isHistoryOpen ? "Close conversation history" : "Open conversation history"}
          aria-label={isHistoryOpen ? "Close history" : "Open history"}
          id="history-drawer-toggle"
        >
          {isHistoryOpen ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
        </button>

        <button
          type="button"
          className="brand-link"
          onClick={() => onSelectTab("analyze")}
          title="Return to SatQuery home"
          aria-label="SatQuery Home"
        >
          <div className="brand-orbit-icon">
            <div className="satellite-dot"></div>
            <Sparkles size={15} className="brand-sparkle" />
          </div>
          <div className="brand-name-group">
            <span className="brand-title">SatQuery</span>
            <span className="brand-edition">ORBIT</span>
          </div>
        </button>
      </div>

      <nav className="topnav-center" aria-label="Primary Navigation">
        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "analyze" ? "active" : ""}`}
          onClick={() => onSelectTab("analyze")}
          id="nav-tab-analyze"
        >
          <Sparkles size={15} />
          <span>Analyze</span>
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "compare" ? "active" : ""}`}
          onClick={() => onSelectTab("compare")}
          id="nav-tab-compare"
        >
          <GitCompare size={15} />
          <span>Compare</span>
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "explore" ? "active" : ""}`}
          onClick={() => onSelectTab("explore")}
          id="nav-tab-explore"
        >
          <Compass size={15} />
          <span>Explore</span>
        </button>

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "reports" ? "active" : ""}`}
          onClick={() => onSelectTab("reports")}
          id="nav-tab-reports"
        >
          <FileText size={15} />
          <span>Reports</span>
        </button>
      </nav>

      <div className="topnav-right">
        <button
          type="button"
          className="search-palette-trigger"
          onClick={onOpenCommandPalette}
          title="Search chats & commands (Ctrl+K)"
          aria-label="Open command palette"
        >
          <Search size={14} className="search-icon" />
          <span className="search-placeholder">Search...</span>
          <kbd className="cmd-shortcut">
            <Command size={10} />K
          </kbd>
        </button>

        <button
          type="button"
          className="topnav-btn theme-toggle-btn"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          aria-label="Toggle theme"
          id="theme-toggle-button"
        >
          {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
        </button>

        <button
          type="button"
          className="topnav-btn settings-btn"
          onClick={onOpenSettings}
          title="System & Analysis Settings"
          aria-label="Settings"
          id="settings-modal-toggle"
        >
          <Settings size={17} />
        </button>

        <div className="user-profile-badge" title="Active Session: Verified">
          <span className="user-avatar-initials">SQ</span>
          <span className="status-indicator-dot" title="AI Backend Ready"></span>
        </div>
      </div>
    </header>
  );
};
