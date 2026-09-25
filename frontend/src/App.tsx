import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  fetchChats,
  createChat,
  fetchChatDetail,
  renameChat,
  deleteChat,
  sendChatMessage,
  fetchSpatialRuns,
  fetchResultById,
} from "./api";
import type {
  ChatSummary,
  ChatDetail,
  NavTab,
  ThemeMode,
  AnalysisRunRecord,
  AnalysisResponse,
} from "./types";
import { TopNav } from "./components/navigation/TopNav";
import { HistoryDrawer } from "./components/navigation/HistoryDrawer";
import { CommandPalette } from "./components/navigation/CommandPalette";
import { HomeZeroState } from "./components/views/HomeZeroState";
import { CompareView } from "./components/views/CompareView";
import { ExploreGallery } from "./components/views/ExploreGallery";
import { ReportsLibrary } from "./components/views/ReportsLibrary";
import { SettingsModal } from "./components/views/SettingsModal";
import { ConversationWorkspace } from "./components/chat/ConversationWorkspace";
import { Composer } from "./components/composer/Composer";
import { ImageLightbox } from "./components/inspectors/ImageLightbox";
import { ReportViewerModal } from "./components/inspectors/ReportViewerModal";
import { AnalysisInspector } from "./components/inspectors/AnalysisInspector";
import { EvidenceDrawer } from "./components/inspectors/EvidenceDrawer";
import { MapPanel } from "./components/inspectors/MapPanel";
import { Toast, type ToastMessage } from "./components/shared/Toast";

const PINNED_STORAGE_KEY = "satquery_pinned_chats";
const ARCHIVED_STORAGE_KEY = "satquery_archived_chats";
const THEME_STORAGE_KEY = "satquery_theme";
const DRAFT_QUERY_KEY = "satquery_draft_query";

export default function App() {
  // Navigation & View Routing
  const [activeTab, setActiveTab] = useState<NavTab>("analyze");

  // Chats & Conversation State
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentChatDetail, setCurrentChatDetail] = useState<ChatDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // History Drawer & Organization State
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [pinnedChatIds, setPinnedChatIds] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(PINNED_STORAGE_KEY) || "[]");
    } catch {
      return [];
    }
  });
  const [archivedChatIds, setArchivedChatIds] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(ARCHIVED_STORAGE_KEY) || "[]");
    } catch {
      return [];
    }
  });

  // Theme Management
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
    return saved === "light" ? "light" : "dark";
  });

  // Historical Analysis Runs (for Explore & Reports)
  const [runs, setRuns] = useState<AnalysisRunRecord[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);

  // Modals, Drawers & Lightboxes
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [lightboxImage, setLightboxImage] = useState<{ url: string; title: string } | null>(null);
  const [reportModalResultId, setReportModalResultId] = useState<string | null>(null);
  const [inspectorResult, setInspectorResult] = useState<AnalysisResponse | null>(null);
  const [evidenceDrawerResult, setEvidenceDrawerResult] = useState<AnalysisResponse | null>(null);
  const [mapPanelResult, setMapPanelResult] = useState<AnalysisResponse | null>(null);

  // Toasts
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  // Draft text persistence
  const [draftQuery, setDraftQuery] = useState(() => localStorage.getItem(DRAFT_QUERY_KEY) || "");

  const initialLoadRef = useRef(false);

  // Apply theme to document element
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  // Persist pinned & archived
  useEffect(() => {
    localStorage.setItem(PINNED_STORAGE_KEY, JSON.stringify(pinnedChatIds));
  }, [pinnedChatIds]);

  useEffect(() => {
    localStorage.setItem(ARCHIVED_STORAGE_KEY, JSON.stringify(archivedChatIds));
  }, [archivedChatIds]);

  const addToast = (type: "success" | "error" | "info", message: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  };

  const handleDismissToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    try {
      const data = await fetchSpatialRuns({ limit: 100 });
      setRuns(data);
    } catch {
      // Degraded fallback
    } finally {
      setRunsLoading(false);
    }
  }, []);

  const refreshChatsList = useCallback(async () => {
    try {
      const list = await fetchChats();
      setChats(list);
      return list;
    } catch {
      return [];
    }
  }, []);

  const selectChat = useCallback(async (chatId: string) => {
    setCurrentChatId(chatId);
    setActiveTab("analyze");
    window.location.hash = `#/chat/${chatId}`;
    try {
      const detail = await fetchChatDetail(chatId);
      setCurrentChatDetail(detail);
      setError(null);
    } catch (err) {
      console.error("Failed to load chat detail:", err);
      setError(err instanceof Error ? err.message : "Failed to load conversation");
    }
  }, []);

  const handleNewChat = useCallback(() => {
    // Reset active chat to zero-state without deleting history
    setCurrentChatId(null);
    setCurrentChatDetail(null);
    setError(null);
    setActiveTab("analyze");
    window.location.hash = "#/analyze";
  }, []);

  // Handle URL Hash routing & initial setup
  useEffect(() => {
    refreshRuns();

    const handleHash = async () => {
      const hash = window.location.hash;
      const chatMatch = hash.match(/^#\/chat\/([a-zA-Z0-9_-]+)/);
      const existingChats = await refreshChatsList();

      if (chatMatch && chatMatch[1]) {
        await selectChat(chatMatch[1]);
      } else if (hash === "#/compare") {
        setActiveTab("compare");
      } else if (hash === "#/explore") {
        setActiveTab("explore");
      } else if (hash === "#/reports") {
        setActiveTab("reports");
      } else {
        // Default to analyze home
        setActiveTab("analyze");
        if (existingChats.length > 0 && !initialLoadRef.current) {
          // Keep at zero state or select latest if preferred
        }
      }
      initialLoadRef.current = true;
    };

    handleHash();

    const onHashChange = () => {
      const hash = window.location.hash;
      const chatMatch = hash.match(/^#\/chat\/([a-zA-Z0-9_-]+)/);
      if (chatMatch && chatMatch[1]) {
        selectChat(chatMatch[1]);
      } else if (hash === "#/compare") setActiveTab("compare");
      else if (hash === "#/explore") setActiveTab("explore");
      else if (hash === "#/reports") setActiveTab("reports");
      else if (hash === "#/analyze" || !hash) {
        setActiveTab("analyze");
      }
    };

    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [refreshChatsList, selectChat, refreshRuns]);

  // Tab change handler
  const handleSelectTab = (tab: NavTab) => {
    setActiveTab(tab);
    window.location.hash = `#/${tab}`;
  };

  // Chat management actions
  const handleRenameChat = async (chatId: string, title: string) => {
    try {
      await renameChat(chatId, title);
      await refreshChatsList();
      if (currentChatDetail && currentChatDetail.chat_id === chatId) {
        setCurrentChatDetail((prev) => (prev ? { ...prev, title } : prev));
      }
      addToast("success", "Conversation renamed");
    } catch (err) {
      addToast("error", err instanceof Error ? err.message : "Failed to rename chat");
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      await deleteChat(chatId);
      const remaining = await refreshChatsList();
      if (currentChatId === chatId) {
        handleNewChat();
      }
      addToast("info", "Conversation deleted");
    } catch (err) {
      addToast("error", err instanceof Error ? err.message : "Failed to delete chat");
    }
  };

  const handleTogglePin = (chatId: string) => {
    setPinnedChatIds((prev) =>
      prev.includes(chatId) ? prev.filter((id) => id !== chatId) : [...prev, chatId]
    );
  };

  const handleToggleArchive = (chatId: string) => {
    setArchivedChatIds((prev) =>
      prev.includes(chatId) ? prev.filter((id) => id !== chatId) : [...prev, chatId]
    );
  };

  // Send message from Composer
  const handleSendMessage = async (query: string, pairType: string, files: File[]) => {
    let chatId = currentChatId;

    // If starting a fresh chat from home zero-state, create it first
    if (!chatId) {
      try {
        const created = await createChat("New Analysis");
        chatId = created.chat_id;
        setCurrentChatId(chatId);
        window.location.hash = `#/chat/${chatId}`;
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not initialize conversation");
        return;
      }
    }

    setBusy(true);
    setError(null);
    setActiveTab("analyze");

    // Optimistically append user message to UI
    const tempUserMsgId = `temp-${Date.now()}`;
    setCurrentChatDetail((prev) => {
      const baseDetail: ChatDetail = prev || {
        chat_id: chatId!,
        title: "New Analysis",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        messages: [],
        images: [],
      };
      return {
        ...baseDetail,
        messages: [
          ...baseDetail.messages,
          {
            message_id: tempUserMsgId,
            chat_id: chatId!,
            role: "user",
            content: query,
            created_at: new Date().toISOString(),
            attachments: files.map((f) => ({ name: f.name, url: "", type: "image" })),
          },
        ],
      };
    });

    try {
      await sendChatMessage({
        chatId,
        query,
        pairType,
        imageA: files.length > 0 ? files[0] : undefined,
        imageB: files.length > 1 ? files[1] : undefined,
      });

      // Synchronize exact chat detail from backend
      const updatedDetail = await fetchChatDetail(chatId);
      setCurrentChatDetail(updatedDetail);
      await refreshChatsList();
      refreshRuns();
      setDraftQuery("");
      localStorage.removeItem(DRAFT_QUERY_KEY);
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err instanceof Error ? err.message : "Analysis failed with raster processing error.");
      addToast("error", "Analysis failed to process image");
    } finally {
      setBusy(false);
    }
  };

  // Reopen old analysis result
  const handleLoadRunResult = async (resultId: string) => {
    setBusy(true);
    setError(null);
    try {
      const loaded = await fetchResultById(resultId);
      const newChat = await createChat(`Analysis: ${loaded.query.slice(0, 28)}`);
      await refreshChatsList();
      await selectChat(newChat.chat_id);
      addToast("info", "Loaded historical analysis into workspace");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load historical result");
    } finally {
      setBusy(false);
    }
  };

  // Determine active context image name from existing chat
  const activeContextImageName =
    currentChatDetail?.images && currentChatDetail.images.length > 0
      ? currentChatDetail.images[0].filename
      : null;

  const messages = currentChatDetail?.messages || [];
  const isChatEmpty = messages.length === 0;

  return (
    <div className="satquery-app" data-theme={theme}>
      {/* 1. Floating Top Navigation (Apple / ChatGPT style) */}
      <TopNav
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        isHistoryOpen={isHistoryOpen}
        onToggleHistory={() => setIsHistoryOpen((prev) => !prev)}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* 2. Slide-out History Drawer (Replaces permanent sidebar) */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        chats={chats}
        currentChatId={currentChatId}
        onSelectChat={selectChat}
        onNewChat={handleNewChat}
        onRenameChat={handleRenameChat}
        onDeleteChat={handleDeleteChat}
        pinnedChatIds={pinnedChatIds}
        onTogglePin={handleTogglePin}
        archivedChatIds={archivedChatIds}
        onToggleArchive={handleToggleArchive}
      />

      {/* 3. Main Stage */}
      <main className="main-stage">
        {/* TAB 1: ANALYZE / CHAT */}
        {activeTab === "analyze" && (
          <>
            {isChatEmpty && !busy ? (
              <HomeZeroState
                onSendMessage={handleSendMessage}
                busy={busy}
                recentChats={chats}
                onSelectChat={selectChat}
              />
            ) : (
              <>
                <ConversationWorkspace
                  messages={messages}
                  busy={busy}
                  error={error}
                  onRetry={() => {
                    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
                    if (lastUserMsg) handleSendMessage(lastUserMsg.content, "auto", []);
                  }}
                  onOpenLightbox={(url, title) => setLightboxImage({ url, title })}
                  onOpenMap={(result) => setMapPanelResult(result)}
                  onOpenEvidenceDrawer={(result) => setEvidenceDrawerResult(result)}
                  onOpenInspector={(result) => setInspectorResult(result)}
                  onPreviewReport={(resultId) => setReportModalResultId(resultId)}
                  hasAttachedImages={Boolean(activeContextImageName)}
                />

                {/* Floating Bottom Composer */}
                <Composer
                  onSendMessage={handleSendMessage}
                  busy={busy}
                  activeContextImageName={activeContextImageName}
                />
              </>
            )}
          </>
        )}

        {/* TAB 2: COMPARE */}
        {activeTab === "compare" && (
          <CompareView onSendMessage={handleSendMessage} busy={busy} />
        )}

        {/* TAB 3: EXPLORE */}
        {activeTab === "explore" && (
          <ExploreGallery
            runs={runs}
            chats={chats}
            onSelectChat={selectChat}
            onLoadRunResult={handleLoadRunResult}
          />
        )}

        {/* TAB 4: REPORTS */}
        {activeTab === "reports" && (
          <ReportsLibrary
            runs={runs}
            loading={runsLoading}
            onRefresh={refreshRuns}
            onPreviewReport={(resultId) => setReportModalResultId(resultId)}
            onGoToAnalyze={() => handleSelectTab("analyze")}
          />
        )}
      </main>

      {/* 4. Optional Side Panels / Inspectors (Opened only on demand) */}
      <ImageLightbox
        isOpen={Boolean(lightboxImage)}
        imageUrl={lightboxImage?.url || ""}
        imageTitle={lightboxImage?.title || ""}
        onClose={() => setLightboxImage(null)}
      />

      <ReportViewerModal
        isOpen={Boolean(reportModalResultId)}
        resultId={reportModalResultId}
        onClose={() => setReportModalResultId(null)}
      />

      <AnalysisInspector
        isOpen={Boolean(inspectorResult)}
        onClose={() => setInspectorResult(null)}
        result={inspectorResult}
        onOpenImage={(url, title) => setLightboxImage({ url, title })}
      />

      <EvidenceDrawer
        isOpen={Boolean(evidenceDrawerResult)}
        onClose={() => setEvidenceDrawerResult(null)}
        result={evidenceDrawerResult}
        onOpenLightbox={(url, title) => setLightboxImage({ url, title })}
      />

      <MapPanel
        isOpen={Boolean(mapPanelResult)}
        onClose={() => setMapPanelResult(null)}
        result={mapPanelResult}
      />

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onSelectTab={handleSelectTab}
        onNewChat={handleNewChat}
        chats={chats}
        onSelectChat={selectChat}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* 5. Toast Notifications */}
      <Toast toasts={toasts} onDismiss={handleDismissToast} />
    </div>
  );
}
