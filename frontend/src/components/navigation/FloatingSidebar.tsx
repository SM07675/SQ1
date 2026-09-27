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

  // Sort real chats by latest timestamp descending (real chats only, no fake presets)
  const sortedRecentChats = useMemo(() => {
    return [...chats].sort((a, b) => {
      const timeA = new Date(a.updated_at || a.created_at).getTime();
      const timeB = new Date(b.updated_at || b.created_at).getTime();
      return timeB - timeA;
    });
  }, [chats]);

  // Display top 5 most recent or all if expanded
  const displayChats = useMemo(() => {
    if (showAll) {
      return sortedRecentChats;
    }
    return sortedRecentChats.slice(0, 5);
  }, [sortedRecentChats, showAll]);

  const handleViewAllClick = () => {
    if (sortedRecentChats.length > 5 && !showAll) {
      setShowAll(true);
    } else {
      onOpenFullHistory();
    }
  };

  return (
    <aside
      className={`floating-sidebar-container ${isOpen ? "open" : ""}`}
      aria-label="Side Navigation Menu"
      aria-hidden={!isOpen}
    >
      {/* 1. Brand Logo Header Pill (No three-line hamburger icon) */}
      <div className="floating-sidebar-brand-pill">
        <button
          type="button"
          className="sidebar-brand-title-btn"
          onClick={() => {
            onSelectTab("analyze");
            onClose();
          }}
          title="SatQuery Home"
        >
          <span className="sidebar-brand-title">SatQuery</span>
        </button>

        <button
          type="button"
          className="sidebar-close-toggle-btn"
          onClick={onClose}
          title="Close sidebar navigation"
          aria-label="Close sidebar navigation"
        >
          <PanelLeftClose size={16} strokeWidth={2.2} />
        </button>
      </div>

      {/* 2. Navigation Glass Card (New Chat + Transferred Navigation: Chat, Explore, Reports) */}
      <nav className="floating-sidebar-nav-card" aria-label="Sidebar Navigation">
        <button
          type="button"
          className="sidebar-new-chat-btn"
          onClick={() => {
            onNewChat();
            onClose();
          }}
          title="Start a new satellite analysis conversation"
        >
          <SquarePen size={15} strokeWidth={2.2} />
          <span>New Chat</span>
        </button>

        <div className="sidebar-nav-divider" />

        <button
          type="button"
          className={`sidebar-nav-item ${activeTab === "analyze" ? "active" : ""}`}
          onClick={() => {
            onSelectTab("analyze");
          }}
        >
          <MessageSquare size={16} strokeWidth={2} className="sidebar-nav-icon" />
          <span>Chat</span>
        </button>

        <button
          type="button"
          className={`sidebar-nav-item ${activeTab === "explore" ? "active" : ""}`}
          onClick={() => {
            onSelectTab("explore");
          }}
        >
          <Compass size={16} strokeWidth={2} className="sidebar-nav-icon" />
          <span>Explore</span>
        </button>

        <button
          type="button"
          className={`sidebar-nav-item ${activeTab === "reports" ? "active" : ""}`}
          onClick={() => {
            onSelectTab("reports");
          }}
        >
          <FileText size={17} strokeWidth={2} className="sidebar-nav-icon" />
          <span>Reports</span>
        </button>
      </nav>

      {/* 3. Recent Chats Glass Card (Real chats only) */}
      <div className="floating-sidebar-recent-card">
        <div className="sidebar-recent-header">
          <span>Recent Chats</span>
          {showAll && sortedRecentChats.length > 5 && (
            <button
              type="button"
              className="sidebar-collapse-btn"
              onClick={() => setShowAll(false)}
              title="Show fewer recent chats"
            >
              <ChevronUp size={13} />
            </button>
          )}
        </div>

        <div className={`sidebar-recent-list ${showAll ? "expanded" : ""}`}>
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
                  <MessageSquare size={14} className="sidebar-chat-icon" />
                  <span className="sidebar-chat-title">{chat.title || "Untitled Analysis"}</span>
                </button>
              );
            })
          )}
        </div>

        {/* View All Button */}
        <button
          type="button"
          className="sidebar-view-all-btn"
          onClick={handleViewAllClick}
          title={showAll ? "Open full history drawer" : "View all recent chats"}
        >
          <span>{showAll ? "Full History Drawer" : "View all"}</span>
          <ChevronRight size={14} strokeWidth={2} />
        </button>
      </div>

      {/* 4. User Profile, Theme & Settings Card (Transferred from top-right to sidebar) */}
      <div className="floating-sidebar-user-card" aria-label="Account, Theme and Settings">
        <div
          className="sidebar-user-info-wrap"
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

        <div className="sidebar-user-actions">
          <button
            type="button"
            className="sidebar-action-pill-btn"
            onClick={onToggleTheme}
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Moon size={15} strokeWidth={2} /> : <Sun size={15} strokeWidth={2} />}
          </button>

          <button
            type="button"
            className="sidebar-action-pill-btn"
            onClick={onOpenSettings}
            title="Open Settings"
            aria-label="Settings"
          >
            <Settings size={15} strokeWidth={2} />
          </button>

          {onLogout && (
            <button
              type="button"
              className="sidebar-action-pill-btn sidebar-logout-btn"
              onClick={onLogout}
              title="Sign Out"
              aria-label="Sign Out"
            >
              <LogOut size={15} strokeWidth={2} />
            </button>
          )}
        </div>
      </div>
    </aside>
  );
};

export default FloatingSidebar;
