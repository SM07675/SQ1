import React, { useState, useMemo } from "react";
import {
  FileText,
  Search,
  Download,
  Eye,
  LayoutGrid,
  List,
  Calendar,
  Droplets,
  Building2,
  GitCompare,
  Layers,
  Sparkles,
  RefreshCw,
  X,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";
import type { AnalysisRunRecord } from "../../types";
import { downloadPdfReport } from "../../api";

interface ReportsLibraryProps {
  runs: AnalysisRunRecord[];
  loading: boolean;
  onRefresh: () => void;
  onPreviewReport: (resultId: string) => void;
  onGoToAnalyze: () => void;
}

function formatTaskType(taskType: string = ""): string {
  const clean = taskType.replace(/_/g, " ").trim();
  if (!clean) return "Satellite Analysis";
  return clean.replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatVerdict(verdict: string = ""): string {
  switch (verdict.toLowerCase()) {
    case "supported":
      return "Verified";
    case "supported_with_limitations":
      return "Supported (Limited)";
    case "disputed":
      return "Disputed";
    case "insufficient_evidence":
      return "Inconclusive";
    default:
      return verdict ? verdict.replace(/_/g, " ") : "Analyzed";
  }
}

function getVerdictClass(verdict: string = ""): string {
  const v = verdict.toLowerCase();
  if (v.includes("supported") && !v.includes("limitations")) return "verdict-supported";
  if (v.includes("limitations")) return "verdict-limited";
  if (v.includes("disputed")) return "verdict-disputed";
  return "verdict-neutral";
}

function getConfidenceClass(conf: number = 0): string {
  if (conf >= 0.75) return "conf-high";
  if (conf >= 0.40) return "conf-medium";
  return "conf-low";
}

function getCategoryIcon(taskType: string = "") {
  const t = taskType.toLowerCase();
  if (t.includes("water") || t.includes("flood")) return <Droplets size={13} />;
  if (t.includes("build") || t.includes("urban")) return <Building2 size={13} />;
  if (t.includes("change") || t.includes("diff")) return <GitCompare size={13} />;
  return <Layers size={13} />;
}

export const ReportsLibrary: React.FC<ReportsLibraryProps> = ({
  runs,
  loading,
  onRefresh,
  onPreviewReport,
  onGoToAnalyze,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  // Calculate filter counts
  const counts = useMemo(() => {
    let water = 0, building = 0, change = 0, land = 0;
    runs.forEach((r) => {
      const t = (r.task_type || "").toLowerCase();
      const q = (r.query_text || "").toLowerCase();
      if (t.includes("water") || q.includes("water")) water++;
      if (t.includes("build") || q.includes("build")) building++;
      if (t.includes("change") || q.includes("change")) change++;
      if (t.includes("land") || q.includes("land")) land++;
    });
    return { all: runs.length, water, building, change, land };
  }, [runs]);

  const filters = [
    { id: "all", label: "All Reports", count: counts.all, icon: <FileText size={13} /> },
    { id: "water", label: "Water & Flood", count: counts.water, icon: <Droplets size={13} /> },
    { id: "building", label: "Buildings", count: counts.building, icon: <Building2 size={13} /> },
    { id: "change", label: "Temporal Change", count: counts.change, icon: <GitCompare size={13} /> },
    { id: "land", label: "Land Cover", count: counts.land, icon: <Layers size={13} /> },
  ];

  const filteredRuns = useMemo(() => {
    return runs.filter((run) => {
      if (selectedFilter !== "all") {
        const t = (run.task_type || "").toLowerCase();
        const q = (run.query_text || "").toLowerCase();
        if (selectedFilter === "water" && !t.includes("water") && !q.includes("water")) return false;
        if (selectedFilter === "building" && !t.includes("build") && !q.includes("build")) return false;
        if (selectedFilter === "change" && !t.includes("change") && !q.includes("change")) return false;
        if (selectedFilter === "land" && !t.includes("land") && !q.includes("land")) return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          run.query_text?.toLowerCase().includes(q) ||
          run.task_type?.toLowerCase().includes(q) ||
          run.result_id?.toLowerCase().includes(q) ||
          run.verdict_status?.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [runs, selectedFilter, searchQuery]);

  // Group reports by date
  const groupedReports = useMemo(() => {
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const oneDayMs = 86400000;
    const startOfThisWeek = startOfToday - 6 * oneDayMs;
    const startOfThisMonth = startOfToday - 29 * oneDayMs;

    const today: AnalysisRunRecord[] = [];
    const thisWeek: AnalysisRunRecord[] = [];
    const thisMonth: AnalysisRunRecord[] = [];
    const older: AnalysisRunRecord[] = [];

    // Sort filtered runs newest first
    const sorted = [...filteredRuns].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );

    sorted.forEach((run) => {
      const time = new Date(run.created_at).getTime();
      if (time >= startOfToday) today.push(run);
      else if (time >= startOfThisWeek) thisWeek.push(run);
      else if (time >= startOfThisMonth) thisMonth.push(run);
      else older.push(run);
    });

    const groups: { label: string; count: number; items: AnalysisRunRecord[] }[] = [];
    if (today.length > 0) groups.push({ label: "Today", count: today.length, items: today });
    if (thisWeek.length > 0) groups.push({ label: "This Week", count: thisWeek.length, items: thisWeek });
    if (thisMonth.length > 0) groups.push({ label: "This Month", count: thisMonth.length, items: thisMonth });
    if (older.length > 0) groups.push({ label: "Older", count: older.length, items: older });

    return groups;
  }, [filteredRuns]);

  return (
    <div className="reports-library-page" role="main">
      {/* 1. Hero Header */}
      <header className="reports-hero-header">
        <div className="reports-badge-pill">
          <FileText size={13} className="reports-badge-icon" />
          <span>GeoProof Report Library</span>
        </div>
        <h1 className="reports-hero-title">Analysis Reports & Audit Records</h1>
        <p className="reports-hero-subtitle">
          Verifiable, publication-ready PDF audit records generated from multi-spectral satellite inquiries.
        </p>

        {/* 2. Interactive Control Bar */}
        <div className="reports-control-panel">
          <div className="reports-search-container">
            <Search size={15} className="reports-search-icon" />
            <input
              type="text"
              placeholder="Search reports by query, category, or ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="reports-search-input"
              aria-label="Search reports"
            />
            {searchQuery && (
              <button
                type="button"
                className="reports-clear-btn"
                onClick={() => setSearchQuery("")}
                title="Clear search"
                aria-label="Clear search"
              >
                <X size={13} />
              </button>
            )}
          </div>

          <div className="reports-controls-right">
            {/* Filter Chips */}
            <div className="reports-filters-track">
              {filters.map((f) => (
                <button
                  key={f.id}
                  type="button"
                  className={`reports-filter-chip ${selectedFilter === f.id ? "active" : ""}`}
                  onClick={() => setSelectedFilter(f.id)}
                >
                  <span className="chip-icon">{f.icon}</span>
                  <span className="chip-label">{f.label}</span>
                  <span className="chip-count">{f.count}</span>
                </button>
              ))}
            </div>

            {/* View Mode & Refresh */}
            <div className="reports-view-actions">
              <div className="reports-view-toggle">
                <button
                  type="button"
                  className={`reports-view-btn ${viewMode === "grid" ? "active" : ""}`}
                  onClick={() => setViewMode("grid")}
                  title="Grid View"
                  aria-label="Grid View"
                >
                  <LayoutGrid size={15} />
                </button>
                <button
                  type="button"
                  className={`reports-view-btn ${viewMode === "list" ? "active" : ""}`}
                  onClick={() => setViewMode("list")}
                  title="List View"
                  aria-label="List View"
                >
                  <List size={15} />
                </button>
              </div>

              <button
                type="button"
                className="reports-refresh-action-btn"
                onClick={onRefresh}
                title="Refresh reports library"
                aria-label="Refresh reports"
                disabled={loading}
              >
                <RefreshCw size={14} className={loading ? "spin" : ""} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* 3. Reports Stage */}
      {filteredRuns.length === 0 ? (
        <div className="reports-empty-card">
          <div className="reports-empty-icon-wrap">
            <FileText size={32} />
          </div>
          <h3 className="reports-empty-title">
            {searchQuery ? "No matching reports found" : "No analysis reports yet"}
          </h3>
          <p className="reports-empty-subtitle">
            {searchQuery
              ? `No audit reports matched "${searchQuery}". Try a different keyword or clear your filter.`
              : "Generate publication-grade PDF audit trails by running satellite inquiries in the Chat tab."}
          </p>
          {searchQuery ? (
            <button
              type="button"
              className="reports-empty-btn"
              onClick={() => {
                setSearchQuery("");
                setSelectedFilter("all");
              }}
            >
              Reset filters
            </button>
          ) : (
            <button type="button" className="reports-empty-btn" onClick={onGoToAnalyze}>
              <Sparkles size={14} />
              <span>Start an Analysis</span>
            </button>
          )}
        </div>
      ) : (
        <div className="reports-groups-stack">
          {groupedReports.map((group) => (
            <section key={group.label} className="reports-date-section" aria-label={group.label}>
              <div className="reports-group-header">
                <span className="reports-group-title">{group.label}</span>
                <span className="reports-group-badge">{group.count} {group.count === 1 ? "report" : "reports"}</span>
                <div className="reports-group-line" />
              </div>

              {viewMode === "grid" ? (
                <div className="reports-cards-grid">
                  {group.items.map((run) => (
                    <article key={run.result_id} className="report-doc-card">
                      {/* Top Row: PDF Badge + Confidence + Verdict */}
                      <div className="report-doc-top">
                        <div className="report-pdf-pill">
                          <FileText size={16} className="pdf-icon" />
                          <span className="pdf-label">PDF</span>
                        </div>

                        <div className="report-badges-group">
                          <span className={`report-conf-pill ${getConfidenceClass(run.confidence)}`}>
                            <span className="conf-pulse-dot" />
                            <span>{Math.round((run.confidence || 0) * 100)}%</span>
                          </span>

                          <span className={`report-verdict-pill ${getVerdictClass(run.verdict_status)}`}>
                            {formatVerdict(run.verdict_status)}
                          </span>
                        </div>
                      </div>

                      {/* Main Info */}
                      <div className="report-doc-content">
                        <h4 className="report-doc-title" title={run.query_text}>
                          {run.query_text ? `Analysis: ${run.query_text}` : "SatQuery GeoProof Report"}
                        </h4>

                        <div className="report-meta-tags-row">
                          <span className="report-meta-tag task">
                            {getCategoryIcon(run.task_type)}
                            <span>{formatTaskType(run.task_type)}</span>
                          </span>

                          <span className="report-meta-tag date">
                            <Calendar size={12} />
                            <span>
                              {new Date(run.created_at).toLocaleDateString(undefined, {
                                month: "short",
                                day: "numeric",
                                year: "numeric",
                              })}
                            </span>
                          </span>
                        </div>
                      </div>

                      {/* Actions Footer */}
                      <div className="report-card-actions">
                        <button
                          type="button"
                          className="report-action-preview-btn"
                          onClick={() => onPreviewReport(run.result_id)}
                          title="Preview PDF audit report"
                        >
                          <Eye size={13} />
                          <span>Preview</span>
                        </button>

                        <button
                          type="button"
                          className="report-action-download-btn"
                          onClick={() => downloadPdfReport(run.result_id)}
                          title="Download PDF audit report"
                        >
                          <Download size={13} />
                          <span>Download</span>
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="reports-table-card">
                  <table className="reports-list-table">
                    <thead>
                      <tr>
                        <th>Document</th>
                        <th>Analysis Category</th>
                        <th>Confidence</th>
                        <th>Status</th>
                        <th>Created Date</th>
                        <th className="text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.items.map((run) => (
                        <tr key={run.result_id} className="reports-table-row">
                          <td className="table-doc-cell">
                            <div className="table-pdf-tag">
                              <FileText size={15} />
                            </div>
                            <div className="table-doc-info">
                              <span className="table-doc-name" title={run.query_text}>
                                {run.query_text || `Report ${run.result_id.slice(0, 8)}`}
                              </span>
                              <span className="table-doc-id">ID: {run.result_id.slice(0, 8)}</span>
                            </div>
                          </td>

                          <td>
                            <span className="table-tag task">
                              {getCategoryIcon(run.task_type)}
                              <span>{formatTaskType(run.task_type)}</span>
                            </span>
                          </td>

                          <td>
                            <span className={`table-conf-tag ${getConfidenceClass(run.confidence)}`}>
                              {Math.round((run.confidence || 0) * 100)}%
                            </span>
                          </td>

                          <td>
                            <span className={`table-verdict-tag ${getVerdictClass(run.verdict_status)}`}>
                              {formatVerdict(run.verdict_status)}
                            </span>
                          </td>

                          <td className="table-date-cell">
                            {new Date(run.created_at).toLocaleDateString(undefined, {
                              month: "short",
                              day: "numeric",
                              year: "numeric",
                            })}
                          </td>

                          <td className="text-right">
                            <div className="table-actions-group">
                              <button
                                type="button"
                                className="table-action-icon-btn"
                                onClick={() => onPreviewReport(run.result_id)}
                                title="Preview Report"
                                aria-label="Preview"
                              >
                                <Eye size={14} />
                              </button>
                              <button
                                type="button"
                                className="table-action-icon-btn primary"
                                onClick={() => downloadPdfReport(run.result_id)}
                                title="Download PDF"
                                aria-label="Download"
                              >
                                <Download size={14} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ))}
        </div>
      )}
    </div>
  );
};

export default ReportsLibrary;
