import React, { useState, useMemo } from "react";
import {
  Compass,
  Search,
  Filter,
  Sparkles,
  Droplets,
  Building2,
  GitCompare,
  Layers,
  ArrowRight,
  Clock,
  Calendar,
} from "lucide-react";
import type { AnalysisRunRecord, ChatSummary } from "../../types";
import { artifactUrl } from "../../api";

interface ExploreGalleryProps {
  runs: AnalysisRunRecord[];
  chats: ChatSummary[];
  onSelectChat: (chatId: string) => void;
  onLoadRunResult: (resultId: string) => void;
}

export const ExploreGallery: React.FC<ExploreGalleryProps> = ({
  runs,
  chats,
  onSelectChat,
  onLoadRunResult,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");

  const categories = [
    { id: "all", label: "All Analyses" },
    { id: "water", label: "Water & Flood", icon: <Droplets size={13} /> },
    { id: "building", label: "Buildings & Urban", icon: <Building2 size={13} /> },
    { id: "change", label: "Temporal Change", icon: <GitCompare size={13} /> },
    { id: "land", label: "Land Cover", icon: <Layers size={13} /> },
  ];

  // Match runs with chats
  const items = useMemo(() => {
    return runs.map((run) => {
      // Find matching chat if possible
      const matchedChat = chats.find(
        (c) =>
          c.title.toLowerCase().includes(run.query_text.slice(0, 15).toLowerCase()) ||
          (c.preview_text && c.preview_text.includes(run.result_id.slice(0, 8)))
      );
      return {
        run,
        chatId: matchedChat?.chat_id,
        thumbnailUrl: artifactUrl(`/artifacts/${run.result_id}/preview_1.png`),
        overlayUrl: artifactUrl(`/artifacts/${run.result_id}/surface_context/water_0/water_overlay.png`),
      };
    });
  }, [runs, chats]);

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      if (selectedCategory !== "all") {
        const t = (item.run.task_type || "").toLowerCase();
        const q = (item.run.query_text || "").toLowerCase();
        if (selectedCategory === "water" && !t.includes("water") && !q.includes("water")) return false;
        if (selectedCategory === "building" && !t.includes("build") && !q.includes("build")) return false;
        if (selectedCategory === "change" && !t.includes("change") && !q.includes("change")) return false;
        if (selectedCategory === "land" && !t.includes("land") && !q.includes("land")) return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          item.run.query_text.toLowerCase().includes(q) ||
          item.run.task_type.toLowerCase().includes(q) ||
          item.run.result_id.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [items, selectedCategory, searchQuery]);

  return (
    <div className="explore-gallery-page">
      <div className="explore-header-row">
        <div className="explore-header-left">
          <div className="explore-badge">
            <Compass size={14} />
            <span>Visual Analysis Memory</span>
          </div>
          <h2 className="explore-title">Explore Previous Analyses</h2>
          <p className="explore-subtitle">
            Browse verified satellite intelligence results, detection layers, and spatial findings across your workspace.
          </p>
        </div>

        {/* Search Input */}
        <div className="explore-search-wrap">
          <Search size={14} className="search-icon" />
          <input
            type="text"
            placeholder="Search imagery or queries..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="explore-search-input"
          />
        </div>
      </div>

      {/* Category Filter Chips */}
      <div className="explore-filter-pills">
        {categories.map((cat) => (
          <button
            key={cat.id}
            type="button"
            className={`filter-pill-btn ${selectedCategory === cat.id ? "active" : ""}`}
            onClick={() => setSelectedCategory(cat.id)}
          >
            {cat.icon}
            <span>{cat.label}</span>
          </button>
        ))}
      </div>

      {/* Gallery Grid */}
      {filteredItems.length === 0 ? (
        <div className="explore-empty-state">
          <Compass size={32} className="explore-empty-icon" />
          <p className="empty-title">No analyses match your search</p>
          <span className="empty-sub">Run an analysis in the chat to populate this gallery.</span>
        </div>
      ) : (
        <div className="explore-grid">
          {filteredItems.map(({ run, chatId, thumbnailUrl }) => {
            const confPct = Math.round((run.confidence || 0) * 100);
            return (
              <div
                key={run.result_id}
                className="explore-card"
                onClick={() => {
                  if (chatId) onSelectChat(chatId);
                  else onLoadRunResult(run.result_id);
                }}
              >
                <div className="explore-card-thumb-wrap">
                  <img
                    src={thumbnailUrl}
                    alt={run.query_text}
                    className="explore-card-thumb"
                    loading="lazy"
                    onError={(e) => {
                      // Fallback placeholder gradient if raster preview missing
                      (e.target as HTMLElement).style.display = "none";
                    }}
                  />
                  <div className="explore-thumb-gradient" />
                  <span className="explore-task-badge">
                    {run.task_type.replace(/_/g, " ")}
                  </span>
                </div>

                <div className="explore-card-body">
                  <div className="explore-date-row">
                    <Clock size={11} />
                    <span>{new Date(run.created_at).toLocaleDateString()}</span>
                    <span className="bullet">•</span>
                    <span className="explore-confidence">{confPct}% Confidence</span>
                  </div>

                  <h4 className="explore-card-query" title={run.query_text}>
                    "{run.query_text}"
                  </h4>

                  <div className="explore-card-footer">
                    <span className="explore-status-pill">{run.verdict_status.replace(/_/g, " ")}</span>
                    <div className="explore-open-link">
                      <span>Inspect</span>
                      <ArrowRight size={13} />
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
