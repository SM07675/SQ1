import React, { useState, useRef, useEffect } from "react";
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
  LogOut,
  History,
  Shield,
} from "lucide-react";
import type { NavTab, ThemeMode } from "../../types";
import type { AuthUser } from "../views/LoginPage";

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
  clusterOnline?: boolean;
  clusterState?: "online" | "starting" | "standby";
  onToggleClusterPower?: () => void;
  authUser?: AuthUser | null;
  onLogout?: () => void;
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
  clusterOnline = true,
  clusterState = "online",
  onToggleClusterPower,
  authUser,
  onLogout,
}) => {
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  // Close profile dropdown on click outside
  useEffect(() => {
    if (!profileMenuOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(e.target as Node)) {
        setProfileMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [profileMenuOpen]);

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

        {/* E2E Cluster Power Controller with dedicated Apple Glass styling */}
        {onToggleClusterPower && (
          <button
            type="button"
            onClick={onToggleClusterPower}
            className={`cluster-power-pill ${clusterState || (clusterOnline ? "online" : "standby")}`}
            title={
              clusterState === "starting"
                ? "AI Node is Starting... (Click to view live progress)"
                : clusterOnline
                ? "Cluster is Online (Click to sleep & save credits)"
                : "Cluster is in Standby (Click to wake up)"
            }
            aria-label={
              clusterState === "starting"
                ? "Cluster is Starting"
                : clusterOnline
                ? "Cluster is Online"
                : "Cluster is in Standby"
            }
          >
            <span className="cluster-power-dot" />
            <span>
              {clusterState === "starting"
                ? "Starting..."
                : clusterOnline
                ? "16GB Online"
                : "Standby"}
            </span>
          </button>
        )}

        {/* User profile badge with interactive profile popover menu */}
        <div className="topnav-profile-wrap" ref={profileMenuRef}>
          <div
            className="user-profile-badge"
            title={`${authUser?.name || "Active Session"} · ${authUser?.role || "Analyst"}`}
            onClick={() => setProfileMenuOpen((prev) => !prev)}
            role="button"
            tabIndex={0}
            aria-label="Open account menu"
            aria-expanded={profileMenuOpen}
          >
            <span className="user-avatar-initials">{authUser?.initials || "HS"}</span>
            <span className="status-indicator-dot" title="Ready" />
          </div>

          {profileMenuOpen && (
            <div className="topnav-profile-menu" role="menu">
              <div className="profile-menu-header">
                <div className="profile-menu-avatar">
                  <span>{authUser?.initials || "HS"}</span>
                </div>
                <div className="profile-menu-user">
                  <span className="profile-menu-name">{authUser?.name || "Harshit S."}</span>
                  <span className="profile-menu-email">{authUser?.email || "demo@satquery.ai"}</span>
                </div>
              </div>

              <div className="profile-menu-badge">
                <Shield size={12} />
                <span>{authUser?.organization || "ISRO / SIH 2026"}</span>
              </div>

              <div className="profile-menu-actions">
                <button
                  type="button"
                  className="profile-menu-item"
                  onClick={() => {
                    setProfileMenuOpen(false);
                    onToggleHistory();
                  }}
                >
                  <History size={14} />
                  <span>Conversation History</span>
                </button>

                <button
                  type="button"
                  className="profile-menu-item"
                  onClick={() => {
                    setProfileMenuOpen(false);
                    onOpenSettings();
                  }}
                >
                  <Settings size={14} />
                  <span>Settings & Preferences</span>
                </button>

                {onLogout && (
                  <button
                    type="button"
                    className="profile-menu-item logout"
                    onClick={() => {
                      setProfileMenuOpen(false);
                      onLogout();
                    }}
                  >
                    <LogOut size={14} />
                    <span>Sign Out / Switch Account</span>
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default TopNav;
