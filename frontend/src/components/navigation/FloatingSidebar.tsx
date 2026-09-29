import React, { useState, useMemo } from "react";
import {
  MessageSquare,
  Compass,
  FileText,
  ChevronRight,
  ChevronUp,
  PanelLeftClose,
  SquarePen,
  Sun,
  Moon,
  Settings,
  LogOut,
  Sparkles,
} from "lucide-react";
import type { NavTab, ChatSummary, ThemeMode } from "../../types";
import type { AuthUser } from "../views/LoginPage";

interface FloatingSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  onNewChat: () => void;
  chats: ChatSummary[];
  currentChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onOpenFullHistory: () => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  authUser?: AuthUser | null;
  onLogout?: () => void;
}

export const FloatingSidebar: React.FC<FloatingSidebarProps> = ({
  isOpen,
  onClose,
  activeTab,
  onSelectTab,
  onNewChat,
  chats,
  currentChatId,
  onSelectChat,
  onOpenFullHistory,
  theme,
  onToggleTheme,
  onOpenSettings,
  authUser,
  onLogout,
}) => {
  const [showAll, setShowAll] = useState(false);

  // Sort real chats by latest timestamp descending
  const sortedRecentChats = useMemo(() => {
    return [...chats].sort((a, b) => {
      const timeA = new Date(a.updated_at || a.created_at).getTime();
      const timeB = new Date(b.updated_at || b.created_at).getTime();
      return timeB - timeA;
    });
  }, [chats]);

  // Display top 6 most recent or all if expanded
  const displayChats = useMemo(() => {
    if (showAll) {
      return sortedRecentChats;
    }
    return sortedRecentChats.slice(0, 6);
  }, [sortedRecentChats, showAll]);

  const handleViewAllClick = () => {
    if (sortedRecentChats.length > 6 && !showAll) {
      setShowAll(true);
    } else {
      onOpenFullHistory();
    }
  };

  return (
    <>
      {/* Backdrop for closing on mobile/tablet */}
      {isOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`floating-sidebar-container app-compact-sidebar ${isOpen ? "open" : ""}`}
        aria-label="Side Navigation Menu"
        aria-hidden={!isOpen}
      >
        {/* 1. Header: Brand + Collapse Button */}
        <div className="sidebar-header-section">
          <button
            type="button"
            className="sidebar-brand-btn"
            onClick={() => {
              onSelectTab("analyze");
              onClose();
            }}
            title="SatQuery Home"
          >
            <div className="sidebar-brand-icon">
              <Sparkles size={15} />
            </div>
            <span className="sidebar-brand-name">SatQuery</span>
          </button>

          <button
            type="button"
            className="sidebar-close-btn"
            onClick={onClose}
            title="Close sidebar navigation"
            aria-label="Close sidebar navigation"
          >
            <PanelLeftClose size={15} strokeWidth={2.2} />
          </button>
        </div>

        {/* 2. New Chat Button */}
        <div className="sidebar-action-section">
          <button
            type="button"
            className="sidebar-new-chat-btn"
            onClick={() => {
              onNewChat();
              onClose();
            }}
            title="Start a new satellite analysis conversation"
          >
            <SquarePen size={14} strokeWidth={2.2} />
            <span>New Chat</span>
          </button>
        </div>

        {/* 3. Main Navigation Links (Chat, Explore, Reports) */}
        <nav className="sidebar-nav-section" aria-label="Sidebar Navigation">
          <button
            type="button"
            className={`sidebar-nav-link ${activeTab === "analyze" ? "active" : ""}`}
            onClick={() => {
              onSelectTab("analyze");
            }}
          >
            <MessageSquare size={15} strokeWidth={2} className="sidebar-nav-icon" />
            <span>Chat</span>
          </button>

          <button
            type="button"
            className={`sidebar-nav-link ${activeTab === "explore" ? "active" : ""}`}
            onClick={() => {
              onSelectTab("explore");
            }}
          >
            <Compass size={15} strokeWidth={2} className="sidebar-nav-icon" />
            <span>Explore</span>
          </button>

          <button
            type="button"
            className={`sidebar-nav-link ${activeTab === "reports" ? "active" : ""}`}
            onClick={() => {
              onSelectTab("reports");
            }}
          >
            <FileText size={15} strokeWidth={2} className="sidebar-nav-icon" />
            <span>Reports</span>
          </button>
        </nav>

        {/* Crisp section divider */}
        <div className="sidebar-section-divider" />

        {/* 4. Recent Chats Section (Scrollable list, clean and easy to scan) */}
        <div className="sidebar-recent-section">
          <div className="sidebar-recent-header">
            <span className="sidebar-section-title">Recent Chats</span>
            {sortedRecentChats.length > 6 && (
              <button
                type="button"
                className="sidebar-expand-toggle-btn"
                onClick={() => setShowAll((prev) => !prev)}
                title={showAll ? "Show fewer recent chats" : "Show more recent chats"}
              >
                <ChevronUp
                  size={12}
                  style={{
                    transform: showAll ? "rotate(0deg)" : "rotate(180deg)",
                    transition: "transform 0.2s ease",
                  }}
                />
              </button>
            )}
          </div>

          <div className="sidebar-recent-list">
            {displayChats.length === 0 ? (
              <div className="sidebar-empty-recent">
                <span>No recent conversations</span>
              </div>
            ) : (
              displayChats.map((chat) => {
                const isSelected = chat.chat_id === currentChatId;
                return (
                  <button
                    key={chat.chat_id}
                    type="button"
                    className={`sidebar-chat-item ${isSelected ? "selected" : ""}`}
                    onClick={() => {
                      onSelectChat(chat.chat_id);
                      onSelectTab("analyze");
                    }}
                    title={chat.title || "Untitled Analysis"}
                  >
                    <MessageSquare size={13} className="sidebar-chat-icon" />
                    <span className="sidebar-chat-title">{chat.title || "Untitled Analysis"}</span>
                  </button>
                );
              })
            )}
          </div>

          {/* View All Button */}
          <button
            type="button"
            className="sidebar-view-all-link"
            onClick={handleViewAllClick}
            title={showAll ? "Open full history drawer" : "View all recent chats"}
          >
            <span>{showAll ? "Full History Drawer" : "View all chats"}</span>
            <ChevronRight size={13} strokeWidth={2} />
          </button>
        </div>

        {/* 5. User Profile, Theme & Settings (Compact footer docked at bottom) */}
        <div className="sidebar-footer-section" aria-label="Account, Theme and Settings">
          <div
            className="sidebar-user-compact"
            onClick={onOpenFullHistory}
            role="button"
            tabIndex={0}
            title="Active Session - Open History"
          >
            <div className="sidebar-user-avatar">
              <span className="sidebar-avatar-initials">{authUser ? authUser.initials : "HS"}</span>
              <span className="sidebar-user-status-dot" title="Ready" />
            </div>
            <div className="sidebar-user-meta">
              <span className="sidebar-user-name">{authUser ? authUser.name : "Harshit S."}</span>
              <span className="sidebar-user-sub">{authUser ? authUser.role : "Active Analyst"}</span>
            </div>
          </div>

          <div className="sidebar-footer-actions">
            <button
              type="button"
              className="sidebar-action-btn"
              onClick={onToggleTheme}
              title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Moon size={14} strokeWidth={2} /> : <Sun size={14} strokeWidth={2} />}
            </button>

            <button
              type="button"
              className="sidebar-action-btn"
              onClick={onOpenSettings}
              title="Open Settings"
              aria-label="Settings"
            >
              <Settings size={14} strokeWidth={2} />
            </button>

            {onLogout && (
              <button
                type="button"
                className="sidebar-action-btn sidebar-logout-btn"
                onClick={onLogout}
                title="Sign Out"
                aria-label="Sign Out"
              >
                <LogOut size={14} strokeWidth={2} />
              </button>
            )}
          </div>
        </div>
      </aside>
    </>
  );
};

export default FloatingSidebar;
