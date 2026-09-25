import React, { useState, useMemo } from "react";
import {
  FileText,
  Search,
  Filter,
  Download,
  Eye,
  LayoutGrid,
  List,
  Calendar,
  CheckCircle2,
  Droplets,
  Building2,
  GitCompare,
  Layers,
  ArrowUpRight,
} from "lucide-react";
import type { AnalysisRunRecord } from "../../types";
import { downloadPdfReport, artifactUrl } from "../../api";

interface ReportsLibraryProps {
  runs: AnalysisRunRecord[];
  loading: boolean;
  onRefresh: () => void;
  onPreviewReport: (resultId: string) => void;
  onGoToAnalyze: () => void;
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

  const filters = [
    { id: "all", label: "All Reports" },
    { id: "water", label: "Water", icon: <Droplets size={13} /> },
    { id: "building", label: "Buildings", icon: <Building2 size={13} /> },
    { id: "change", label: "Change", icon: <GitCompare size={13} /> },
    { id: "land", label: "Land Cover", icon: <Layers size={13} /> },
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
          run.query_text.toLowerCase().includes(q) ||
          run.task_type.toLowerCase().includes(q) ||
          run.result_id.toLowerCase().includes(q)
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

    filteredRuns.forEach((run) => {
      const time = new Date(run.created_at).getTime();
      if (time >= startOfToday) today.push(run);
      else if (time >= startOfThisWeek) thisWeek.push(run);
      else if (time >= startOfThisMonth) thisMonth.push(run);
      else older.push(run);
    });

    const groups: { label: string; items: AnalysisRunRecord[] }[] = [];
    if (today.length > 0) groups.push({ label: "Today", items: today });
    if (thisWeek.length > 0) groups.push({ label: "This Week", items: thisWeek });
    if (thisMonth.length > 0) groups.push({ label: "This Month", items: thisMonth });
    if (older.length > 0) groups.push({ label: "Older", items: older });

    return groups;
  }, [filteredRuns]);

  return (
    <div className="reports-library-page">
      <div className="reports-header-row">
        <div className="reports-header-left">
          <div className="reports-badge">
            <FileText size={14} />
            <span>Document Library</span>
          </div>
          <h2 className="reports-title">GeoProof Analysis Reports</h2>
          <p className="reports-subtitle">
            Formal, publication-ready PDF audit records generated from verifiable multi-witness satellite inquiries.
          </p>
        </div>

        {/* View mode toggle & search */}
        <div className="reports-header-actions">
          <div className="reports-search-wrap">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              placeholder="Search reports..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="reports-search-input"
            />
          </div>

          <div className="view-mode-toggle-group">
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "grid" ? "active" : ""}`}
              onClick={() => setViewMode("grid")}
              title="Grid View"
            >
              <LayoutGrid size={15} />
            </button>
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "list" ? "active" : ""}`}
              onClick={() => setViewMode("list")}
              title="List View"
            >
              <List size={15} />
            </button>
          </div>
        </div>
      </div>

      {/* Filter Chips */}
      <div className="reports-filter-pills">
        {filters.map((f) => (
          <button
            key={f.id}
            type="button"
            className={`filter-pill-btn ${selectedFilter === f.id ? "active" : ""}`}
            onClick={() => setSelectedFilter(f.id)}
          >
            {f.icon}
            <span>{f.label}</span>
          </button>
        ))}
      </div>

      {/* Reports Content */}
      {filteredRuns.length === 0 ? (
        <div className="reports-empty-state">
          <FileText size={36} className="reports-empty-icon" />
          <h3 className="empty-title">No reports yet</h3>
          <p className="empty-desc">
            Reports are automatically generated whenever you complete a satellite imagery analysis.
          </p>
          <button type="button" className="empty-analyze-btn" onClick={onGoToAnalyze}>
            Analyze imagery
          </button>
        </div>
      ) : (
        <div className="reports-groups-container">
          {groupedReports.map((group) => (
            <div key={group.label} className="reports-date-group">
              <h3 className="reports-group-heading">{group.label}</h3>

              {viewMode === "grid" ? (
                <div className="reports-cards-grid">
                  {group.items.map((run) => (
                    <div key={run.result_id} className="report-library-card">
                      <div className="report-card-top-icon">
                        <FileText size={22} className="pdf-doc-icon" />
                        <span className="report-conf-badge">
                          {Math.round((run.confidence || 0) * 100)}% Conf
                        </span>
                      </div>

                      <h4 className="report-doc-title">
                        {run.query_text ? `Analysis: ${run.query_text}` : "SatQuery GeoProof Report"}
                      </h4>

                      <div className="report-doc-meta">
                        <span className="doc-task-tag">{run.task_type.replace(/_/g, " ")}</span>
                        <span className="doc-date-tag">
                          {new Date(run.created_at).toLocaleDateString()}
                        </span>
                      </div>

                      <div className="report-card-actions-row">
                        <button
                          type="button"
                          className="report-preview-action-btn"
                          onClick={() => onPreviewReport(run.result_id)}
                        >
                          <Eye size={13} />
                          <span>Preview</span>
                        </button>

                        <button
                          type="button"
                          className="report-download-action-btn"
                          onClick={() => downloadPdfReport(run.result_id)}
                          title="Download PDF"
                        >
                          <Download size={13} />
                          <span>Download</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="reports-table-wrap">
                  <table className="reports-list-table">
                    <thead>
                      <tr>
                        <th>Report Name</th>
                        <th>Analysis Type</th>
                        <th>Date</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {group.items.map((run) => (
                        <tr key={run.result_id} className="report-table-row">
                          <td className="report-name-cell">
                            <FileText size={16} className="table-pdf-icon" />
                            <span className="table-report-name">
                              {run.query_text || `Report ${run.result_id.slice(0, 8)}`}
                            </span>
                          </td>
                          <td>
                            <span className="table-task-pill">
                              {run.task_type.replace(/_/g, " ")}
                            </span>
                          </td>
                          <td className="table-date-cell">
                            {new Date(run.created_at).toLocaleDateString()}
                          </td>
                          <td>
                            <span className="table-verdict-pill">
                              {run.verdict_status.replace(/_/g, " ")}
                            </span>
                          </td>
                          <td className="table-actions-cell">
                            <button
                              type="button"
                              className="table-action-btn"
                              onClick={() => onPreviewReport(run.result_id)}
                              title="Preview"
                            >
                              <Eye size={14} />
                            </button>
                            <button
                              type="button"
                              className="table-action-btn"
                              onClick={() => downloadPdfReport(run.result_id)}
                              title="Download"
                            >
                              <Download size={14} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
