import React, { useState, useEffect, useRef } from "react";
import {
  Search,
  Sparkles,
  GitCompare,
  Compass,
  FileText,
  Sun,
  Moon,
  Settings,
  Plus,
  ArrowRight,
} from "lucide-react";
import type { ChatSummary, NavTab, ThemeMode } from "../../types";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTab: (tab: NavTab) => void;
  onNewChat: () => void;
  chats: ChatSummary[];
  onSelectChat: (chatId: string) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectTab,
  onNewChat,
  chats,
  onSelectChat,
  theme,
  onToggleTheme,
  onOpenSettings,
}) => {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Global keydown handler for Ctrl/Cmd + K and Esc
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (isOpen) onClose();
        else {
          // If closed, caller controls isOpen, but this helps when mounted
        }
      } else if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const baseCommands = [
    {
      id: "cmd-new",
      title: "New Analysis",
      subtitle: "Start a fresh satellite query",
      icon: <Plus size={16} />,
      action: () => {
        onNewChat();
        onClose();
      },
    },
    {
      id: "cmd-analyze",
      title: "Analyze Home",
      subtitle: "Conversational satellite workspace",
      icon: <Sparkles size={16} />,
      action: () => {
        onSelectTab("analyze");
        onClose();
      },
    },
    {
      id: "cmd-compare",
      title: "Compare Imagery",
      subtitle: "Bi-temporal change & Optical + SAR fusion",
      icon: <GitCompare size={16} />,
      action: () => {
        onSelectTab("compare");
        onClose();
      },
    },
    {
      id: "cmd-explore",
      title: "Explore Gallery",
      subtitle: "Browse historical satellite analyses",
      icon: <Compass size={16} />,
      action: () => {
        onSelectTab("explore");
        onClose();
      },
    },
    {
      id: "cmd-reports",
      title: "Reports Library",
      subtitle: "Generated GeoProof audit reports",
      icon: <FileText size={16} />,
      action: () => {
        onSelectTab("reports");
        onClose();
      },
    },
    {
      id: "cmd-theme",
      title: theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode",
      subtitle: "Toggle visual interface theme",
      icon: theme === "dark" ? <Sun size={16} /> : <Moon size={16} />,
      action: () => {
        onToggleTheme();
        onClose();
      },
    },
    {
      id: "cmd-settings",
      title: "Settings",
      subtitle: "Appearance, displays & model telemetry",
      icon: <Settings size={16} />,
      action: () => {
        onOpenSettings();
        onClose();
      },
    },
  ];

  const matchedChats = chats
    .filter((c) =>
      query.trim()
        ? c.title.toLowerCase().includes(query.toLowerCase()) ||
          (c.preview_text && c.preview_text.toLowerCase().includes(query.toLowerCase()))
        : false
    )
    .slice(0, 5)
    .map((c) => ({
      id: `chat-${c.chat_id}`,
      title: c.title,
      subtitle: `Analysis • ${new Date(c.updated_at || c.created_at).toLocaleDateString()}`,
      icon: <Sparkles size={16} />,
      action: () => {
        onSelectChat(c.chat_id);
        onClose();
      },
    }));

  const filteredCommands = baseCommands.filter(
    (cmd) =>
      !query.trim() ||
      cmd.title.toLowerCase().includes(query.toLowerCase()) ||
      cmd.subtitle.toLowerCase().includes(query.toLowerCase())
  );

  const allItems = [...filteredCommands, ...matchedChats];

  const handleKeyDownList = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev < allItems.length - 1 ? prev + 1 : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : allItems.length - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (allItems[selectedIndex]) {
        allItems[selectedIndex].action();
      }
    }
  };

  return (
    <div className="command-palette-backdrop" onClick={onClose}>
      <div
        className="command-palette-modal"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDownList}
      >
        <div className="palette-input-row">
          <Search size={18} className="palette-search-icon" />
          <input
            ref={inputRef}
            type="text"
            className="palette-input"
            placeholder="Type a command or search analyses..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
          />
          <kbd className="palette-esc-badge" onClick={onClose}>
            ESC
          </kbd>
        </div>

        <div className="palette-results-list">
          {allItems.length === 0 ? (
            <div className="palette-empty">No matching commands or analyses found.</div>
          ) : (
            allItems.map((item, idx) => (
              <div
                key={item.id}
                className={`palette-item ${idx === selectedIndex ? "selected" : ""}`}
                onClick={item.action}
                onMouseEnter={() => setSelectedIndex(idx)}
              >
                <div className="palette-item-icon">{item.icon}</div>
                <div className="palette-item-text">
                  <span className="palette-item-title">{item.title}</span>
                  <span className="palette-item-subtitle">{item.subtitle}</span>
                </div>
                <ArrowRight size={14} className="palette-item-arrow" />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
