import React from "react";
import {
  Sparkles,
  Compass,
  FileText,
  Search,
  Sun,
  Moon,
  Settings,
  PanelLeft,
  SquarePen,
} from "lucide-react";
import type { NavTab, ThemeMode } from "../../types";

interface TopNavProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  onNewChat: () => void;
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
  onNewChat,
  isHistoryOpen,
  onToggleHistory,
  theme,
  onToggleTheme,
  onOpenCommandPalette,
  onOpenSettings,
}) => {
  return (
    <header className={`satquery-topnav ${isHistoryOpen ? "sidebar-open" : ""}`} role="banner">
      {/* Brand logo pill (Clean branding, no 3-line hamburger menu icon) */}
      <div className={`topnav-brand ${isHistoryOpen ? "sidebar-open" : ""}`}>
        <button
          type="button"
          className="brand-logo-btn"
          onClick={() => {
            onSelectTab("analyze");
            if (isHistoryOpen) onToggleHistory();
          }}
          title="SatQuery AI Home"
          aria-label="SatQuery Home"
        >
          <span className="brand-logo-name">SatQuery</span>
        </button>
      </div>

      {/* Centered floating glass nav pill */}
      <nav
        className={`topnav-center ${isHistoryOpen ? "sidebar-open" : ""}`}
        aria-label="Primary Navigation"
      >
        {/* Prominent Sidebar Toggle Button */}
        <button
          type="button"
          className={`nav-sidebar-btn ${isHistoryOpen ? "sidebar-active" : ""}`}
          onClick={onToggleHistory}
          id="nav-tab-sidebar-toggle"
          title={isHistoryOpen ? "Close sidebar" : "Open sidebar"}
          aria-label={isHistoryOpen ? "Close sidebar" : "Open sidebar"}
        >
          <PanelLeft size={16} strokeWidth={2.2} />
        </button>

        <span className="topnav-tab-divider" />

        {/* New Chat Action Button - Exact same button style as other navigation tabs */}
        <button
          type="button"
          className="nav-tab-btn"
          onClick={onNewChat}
          id="nav-new-chat-button"
          title="Start a new analysis conversation"
          aria-label="New Chat"
        >
          <SquarePen size={15} strokeWidth={2.2} />
          <span>New Chat</span>
        </button>

        <span className="topnav-tab-divider" />

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "analyze" ? "active" : ""}`}
          onClick={() => onSelectTab("analyze")}
          id="nav-tab-analyze"
          aria-current={activeTab === "analyze" ? "page" : undefined}
        >
          <Sparkles size={15} strokeWidth={2.2} />
          <span>Analyze</span>
        </button>

        <span className="topnav-tab-divider" />

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "explore" ? "active" : ""}`}
          onClick={() => onSelectTab("explore")}
          id="nav-tab-explore"
          aria-current={activeTab === "explore" ? "page" : undefined}
        >
          <Compass size={15} strokeWidth={2.2} />
          <span>Explore</span>
        </button>

        <span className="topnav-tab-divider" />

        <button
          type="button"
          className={`nav-tab-btn ${activeTab === "reports" ? "active" : ""}`}
          onClick={() => onSelectTab("reports")}
          id="nav-tab-reports"
          aria-current={activeTab === "reports" ? "page" : undefined}
        >
          <FileText size={15} strokeWidth={2.2} />
          <span>Reports</span>
        </button>
      </nav>

      {/* Right utility nav pill */}
      <div className={`topnav-right ${isHistoryOpen ? "sidebar-open" : ""}`}>
        <button
          type="button"
          className="search-palette-trigger"
          onClick={onOpenCommandPalette}
          title="Search chats & commands (Ctrl+K)"
          aria-label="Open search"
          id="search-trigger"
        >
          <Search size={16} className="search-icon" />
          <span className="search-placeholder">Search...</span>
          <kbd className="cmd-shortcut">K</kbd>
        </button>

        <div className="topnav-separator" />

        <button
          type="button"
          className="topnav-btn theme-toggle-btn"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          aria-label="Toggle theme"
          id="theme-toggle-button"
        >
          {theme === "dark" ? <Moon size={18} strokeWidth={2} /> : <Sun size={18} strokeWidth={2} />}
        </button>

        <button
          type="button"
          className="topnav-btn settings-btn"
          onClick={onOpenSettings}
          title="Settings"
          aria-label="Settings"
          id="settings-modal-toggle"
        >
          <Settings size={18} strokeWidth={2} />
        </button>

        <button
          type="button"
          className="topnav-btn history-toggle-btn"
          onClick={onToggleHistory}
          title={isHistoryOpen ? "Close history" : "Open history"}
          aria-label={isHistoryOpen ? "Close history" : "Open history"}
          id="history-drawer-toggle"
          style={{ display: "none" }} // History accessed via drawer icon
        >
          <PanelLeft size={18} />
        </button>

        <div
          className="user-profile-badge"
          title="Active Session"
          onClick={onToggleHistory}
          role="button"
          tabIndex={0}
          aria-label="Open conversation history"
        >
          <span className="user-avatar-initials">HS</span>
          <span className="status-indicator-dot" title="Ready" />
        </div>
      </div>
    </header>
  );
};
