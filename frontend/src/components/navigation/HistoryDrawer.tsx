import React, { useState, useMemo, useRef, useEffect } from "react";
import {
  Plus,
  Search,
  Pin,
  PinOff,
  Edit2,
  Trash2,
  Archive,
  MoreVertical,
  Droplets,
  Building2,
  GitCompare,
  Layers,
  Sparkles,
  Check,
  X,
} from "lucide-react";
import type { ChatSummary } from "../../types";

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  chats: ChatSummary[];
  currentChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  onRenameChat: (chatId: string, title: string) => Promise<void>;
  onDeleteChat: (chatId: string) => Promise<void>;
  pinnedChatIds: string[];
  onTogglePin: (chatId: string) => void;
  archivedChatIds: string[];
  onToggleArchive: (chatId: string) => void;
}

function getChatIcon(title: string, lastMessage?: string | null) {
  const text = `${title} ${lastMessage || ""}`.toLowerCase();
  if (text.includes("water") || text.includes("flood") || text.includes("river") || text.includes("lake")) {
    return <Droplets size={14} className="chat-row-icon water" />;
  }
  if (text.includes("build") || text.includes("footprint") || text.includes("structure") || text.includes("urban")) {
    return <Building2 size={14} className="chat-row-icon building" />;
  }
  if (text.includes("change") || text.includes("temporal") || text.includes("diff")) {
    return <GitCompare size={14} className="chat-row-icon change" />;
  }
  if (text.includes("land") || text.includes("vegetation") || text.includes("canopy") || text.includes("cover")) {
    return <Layers size={14} className="chat-row-icon land" />;
  }
  return <Sparkles size={14} className="chat-row-icon default" />;
}

function formatRelativeTime(dateString: string): string {
  try {
    const d = new Date(dateString);
    const now = new Date();
    const diffMin = Math.floor((now.getTime() - d.getTime()) / 60000);
    if (diffMin < 1) return "Just now";
    if (diffMin < 60) return `${diffMin}m`;
    const diffHours = Math.floor(diffMin / 60);
    if (diffHours < 24) return `${diffHours}h`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 7) return `${diffDays}d`;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  chats,
  currentChatId,
  onSelectChat,
  onNewChat,
  onRenameChat,
  onDeleteChat,
  pinnedChatIds,
  onTogglePin,
  archivedChatIds,
  onToggleArchive,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [menuOpenChatId, setMenuOpenChatId] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<ChatSummary | null>(null);

  const menuRef = useRef<HTMLDivElement>(null);

  // Close overflow menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenChatId(null);
      }
    };
    if (menuOpenChatId) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [menuOpenChatId]);

  // Filter chats by search and archive state
  const visibleChats = useMemo(() => {
    return chats.filter((chat) => {
      const isArchived = archivedChatIds.includes(chat.chat_id);
      if (showArchived) {
        if (!isArchived) return false;
      } else {
        if (isArchived) return false;
      }

      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      return (
        chat.title.toLowerCase().includes(q) ||
        (chat.preview_text && chat.preview_text.toLowerCase().includes(q))
      );
    });
  }, [chats, searchQuery, archivedChatIds, showArchived]);

  // Group chats by date and pin status
  const groupedSections = useMemo(() => {
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const oneDayMs = 86400000;
    const startOfYesterday = startOfToday - oneDayMs;
    const startOf7Days = startOfToday - 6 * oneDayMs;
    const startOf30Days = startOfToday - 29 * oneDayMs;

    const pinned: ChatSummary[] = [];
    const today: ChatSummary[] = [];
    const yesterday: ChatSummary[] = [];
    const prev7Days: ChatSummary[] = [];
    const prev30Days: ChatSummary[] = [];
    const older: ChatSummary[] = [];

    visibleChats.forEach((chat) => {
      if (pinnedChatIds.includes(chat.chat_id) && !showArchived) {
        pinned.push(chat);
        return;
      }
      const time = new Date(chat.updated_at || chat.created_at).getTime();
      if (time >= startOfToday) {
        today.push(chat);
      } else if (time >= startOfYesterday) {
        yesterday.push(chat);
      } else if (time >= startOf7Days) {
        prev7Days.push(chat);
      } else if (time >= startOf30Days) {
        prev30Days.push(chat);
      } else {
        older.push(chat);
      }
    });

    const sections: { title: string; items: ChatSummary[] }[] = [];
    if (pinned.length > 0) sections.push({ title: "Pinned", items: pinned });
    if (today.length > 0) sections.push({ title: "Today", items: today });
    if (yesterday.length > 0) sections.push({ title: "Yesterday", items: yesterday });
    if (prev7Days.length > 0) sections.push({ title: "Previous 7 Days", items: prev7Days });
    if (prev30Days.length > 0) sections.push({ title: "Previous 30 Days", items: prev30Days });
    if (older.length > 0) sections.push({ title: "Older", items: older });

    return sections;
  }, [visibleChats, pinnedChatIds, showArchived]);

  const handleStartRename = (chat: ChatSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingChatId(chat.chat_id);
    setEditingTitle(chat.title);
    setMenuOpenChatId(null);
  };

  const handleSaveRename = async (chatId: string) => {
    if (editingTitle.trim()) {
      await onRenameChat(chatId, editingTitle.trim());
    }
    setEditingChatId(null);
  };

  const handleCancelRename = () => {
    setEditingChatId(null);
  };

  const handleConfirmDelete = async () => {
    if (chatToDelete) {
      await onDeleteChat(chatToDelete.chat_id);
      setChatToDelete(null);
    }
  };

  return (
    <>
      {/* Backdrop overlay on mobile or when drawer is active */}
      <div
        className={`drawer-backdrop ${isOpen ? "visible" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`satquery-history-drawer ${isOpen ? "open" : ""}`} aria-label="Conversation History">
        <div className="drawer-header">
          <button
            type="button"
            className="new-analysis-btn"
            onClick={() => {
              onNewChat();
              if (window.innerWidth < 768) onClose();
            }}
            id="history-new-analysis-btn"
          >
            <Plus size={16} />
            <span>New Analysis</span>
          </button>

          <button
            type="button"
            className="drawer-close-btn"
            onClick={onClose}
            title="Close drawer (Esc)"
            aria-label="Close drawer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Search & Filter Bar */}
        <div className="drawer-search-bar">
          <div className="search-input-wrapper">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="drawer-search-input"
              aria-label="Search conversation history"
            />
            {searchQuery && (
              <button
                type="button"
                className="clear-search-btn"
                onClick={() => setSearchQuery("")}
                title="Clear search"
              >
                <X size={12} />
              </button>
            )}
          </div>

          <div className="archive-toggle-row">
            <button
              type="button"
              className={`archive-pill-toggle ${!showArchived ? "active" : ""}`}
              onClick={() => setShowArchived(false)}
            >
              Active ({chats.length - archivedChatIds.length})
            </button>
            <button
              type="button"
              className={`archive-pill-toggle ${showArchived ? "active" : ""}`}
              onClick={() => setShowArchived(true)}
            >
              Archived ({archivedChatIds.length})
            </button>
          </div>
        </div>

        {/* Conversation List */}
        <div className="drawer-chat-list">
          {visibleChats.length === 0 ? (
            <div className="empty-history-state">
              <Sparkles size={24} className="empty-history-icon" />
              <p className="empty-history-title">
                {searchQuery
                  ? "No matching conversations"
                  : showArchived
                  ? "No archived conversations"
                  : "Your satellite analyses will appear here."}
              </p>
              {!searchQuery && !showArchived && (
                <button
                  type="button"
                  className="empty-start-btn"
                  onClick={() => {
                    onNewChat();
                    if (window.innerWidth < 768) onClose();
                  }}
                >
                  Start analysis
                </button>
              )}
            </div>
          ) : (
            groupedSections.map((sec) => (
              <div key={sec.title} className="history-group">
                <div className="history-group-title">{sec.title}</div>
                <div className="history-group-items">
                  {sec.items.map((chat) => {
                    const isSelected = chat.chat_id === currentChatId;
                    const isPinned = pinnedChatIds.includes(chat.chat_id);
                    const isEditing = editingChatId === chat.chat_id;
                    const isMenuOpen = menuOpenChatId === chat.chat_id;

                    return (
                      <div
                        key={chat.chat_id}
                        className={`history-chat-row ${isSelected ? "selected" : ""} ${
                          isMenuOpen ? "menu-open" : ""
                        }`}
                        onClick={() => {
                          if (!isEditing) {
                            onSelectChat(chat.chat_id);
                            if (window.innerWidth < 768) onClose();
                          }
                        }}
                      >
                        <div className="chat-row-left">
                          {getChatIcon(chat.title, chat.preview_text)}
                        </div>

                        <div className="chat-row-center">
                          {isEditing ? (
                            <form
                              onSubmit={(e) => {
                                e.preventDefault();
                                handleSaveRename(chat.chat_id);
                              }}
                              className="rename-inline-form"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <input
                                type="text"
                                value={editingTitle}
                                onChange={(e) => setEditingTitle(e.target.value)}
                                autoFocus
                                className="rename-inline-input"
                              />
                              <button
                                type="submit"
                                className="rename-action-btn check"
                                title="Save title"
                              >
                                <Check size={12} />
                              </button>
                              <button
                                type="button"
                                className="rename-action-btn cancel"
                                onClick={handleCancelRename}
                                title="Cancel"
                              >
                                <X size={12} />
                              </button>
                            </form>
                          ) : (
                            <>
                              <span className="chat-row-title" title={chat.title}>
                                {chat.title}
                              </span>
                              <span className="chat-row-time">
                                {formatRelativeTime(chat.updated_at || chat.created_at)}
                              </span>
                            </>
                          )}
                        </div>

                        {/* Pin Indicator */}
                        {isPinned && !isEditing && (
                          <span title="Pinned">
                            <Pin size={11} className="pin-indicator-icon" />
                          </span>
                        )}

                        {/* Hover Action Menu */}
                        {!isEditing && (
                          <div className="chat-row-actions" onClick={(e) => e.stopPropagation()}>
                            <button
                              type="button"
                              className="row-menu-btn"
                              onClick={() =>
                                setMenuOpenChatId(isMenuOpen ? null : chat.chat_id)
                              }
                              title="Conversation actions"
                              aria-label="Actions"
                            >
                              <MoreVertical size={14} />
                            </button>

                            {isMenuOpen && (
                              <div className="row-overflow-popover" ref={menuRef}>
                                <button
                                  type="button"
                                  className="popover-item"
                                  onClick={(e) => {
                                    onTogglePin(chat.chat_id);
                                    setMenuOpenChatId(null);
                                  }}
                                >
                                  {isPinned ? <PinOff size={13} /> : <Pin size={13} />}
                                  <span>{isPinned ? "Unpin" : "Pin"}</span>
                                </button>

                                <button
                                  type="button"
                                  className="popover-item"
                                  onClick={(e) => handleStartRename(chat, e)}
                                >
                                  <Edit2 size={13} />
                                  <span>Rename</span>
                                </button>

                                <button
                                  type="button"
                                  className="popover-item"
                                  onClick={() => {
                                    onToggleArchive(chat.chat_id);
                                    setMenuOpenChatId(null);
                                  }}
                                >
                                  <Archive size={13} />
                                  <span>
                                    {archivedChatIds.includes(chat.chat_id)
                                      ? "Unarchive"
                                      : "Archive"}
                                  </span>
                                </button>

                                <div className="popover-divider" />

                                <button
                                  type="button"
                                  className="popover-item danger"
                                  onClick={() => {
                                    setMenuOpenChatId(null);
                                    setChatToDelete(chat);
                                  }}
                                >
                                  <Trash2 size={13} />
                                  <span>Delete</span>
                                </button>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Delete Confirmation Modal */}
      {chatToDelete && (
        <div className="modal-backdrop" onClick={() => setChatToDelete(null)}>
          <div className="confirm-modal-box" onClick={(e) => e.stopPropagation()}>
            <h3 className="modal-title">Delete conversation?</h3>
            <p className="modal-desc">
              Are you sure you want to delete <strong>"{chatToDelete.title}"</strong>?
              This will remove its conversation messages and cached raster previews.
            </p>
            <div className="modal-actions">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setChatToDelete(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-danger"
                onClick={handleConfirmDelete}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
