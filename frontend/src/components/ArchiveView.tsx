import React, { useState, useMemo } from "react";
import type { AnalysisRunRecord } from "../types";
import { artifactUrl } from "../api";

interface ArchiveViewProps {
  runs: AnalysisRunRecord[];
  loading: boolean;
  onRefresh: () => void;
  onLoadResult: (resultId: string) => void;
}

function formatRelativeTime(dateString: string): string {
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (diffSec < 60) return "just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return dateString;
  }
}

function formatTaskBadge(task: string): { label: string; icon: string; className: string } {
  const t = task.toLowerCase();
  if (t.includes("optical_sar") || t.includes("croma")) {
    return { label: "Optical + SAR Fusion", icon: "⌖", className: "badge-croma" };
  }
  if (t.includes("change") || t.includes("bi_temporal")) {
    return { label: "Bi-Temporal Change", icon: "⇄", className: "badge-change" };
  }
  if (t.includes("water")) {
    return { label: "Water Grounding", icon: "💧", className: "badge-water" };
  }
  if (t.includes("built") || t.includes("urban")) {
    return { label: "Built-up Footprint", icon: "🏢", className: "badge-built" };
  }
  if (t.includes("veg") || t.includes("canopy")) {
    return { label: "Vegetation Canopy", icon: "🌿", className: "badge-veg" };
  }
  return { label: task.replace(/_/g, " "), icon: "◈", className: "badge-default" };
}

function formatVerdictBadge(status: string): { label: string; className: string } {
  const s = status.toLowerCase();
  if (s === "supported") return { label: "Verified & Supported", className: "verdict-pill-supported" };
  if (s === "refuted") return { label: "Refuted / Inconsistent", className: "verdict-pill-refuted" };
  if (s === "inconclusive") return { label: "Inconclusive Evidence", className: "verdict-pill-inconclusive" };
  return { label: "Insufficient Evidence", className: "verdict-pill-insufficient" };
}

export const ArchiveView: React.FC<ArchiveViewProps> = ({
  runs,
  loading,
  onRefresh,
  onLoadResult,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [loadingResultId, setLoadingResultId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      if (statusFilter !== "all" && r.verdict_status.toLowerCase() !== statusFilter.toLowerCase()) {
        return false;
      }
      if (taskFilter !== "all" && !r.task_type.toLowerCase().includes(taskFilter.toLowerCase())) {
        return false;
      }
      if (searchTerm.trim()) {
        const term = searchTerm.toLowerCase();
        return (
          r.query_text.toLowerCase().includes(term) ||
          r.result_id.toLowerCase().includes(term) ||
          r.task_type.toLowerCase().includes(term)
        );
      }
      return true;
    });
  }, [runs, searchTerm, statusFilter, taskFilter]);

  const stats = useMemo(() => {
    const total = runs.length;
    const supported = runs.filter((r) => r.verdict_status.toLowerCase() === "supported").length;
    const supportRate = total > 0 ? Math.round((supported / total) * 100) : 0;
    const avgConf =
      total > 0
        ? Math.round(
            (runs.reduce((acc, curr) => acc + (curr.confidence || 0), 0) / total) * 100
          )
        : 0;
    const totalGeom = runs.reduce((acc, curr) => acc + (curr.geometries_count || 0), 0);
    return { total, supportRate, avgConf, totalGeom };
  }, [runs]);

  async function handleLoad(resultId: string) {
    setLoadingResultId(resultId);
    try {
      await onLoadResult(resultId);
    } finally {
      setLoadingResultId(null);
    }
  }

  function handleCopy(id: string) {
    navigator.clipboard?.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }

  return (
    <div className="archive-view-container">
      {/* View Header & KPIs */}
      <div className="archive-header-banner">
        <div className="archive-header-titles">
          <div className="view-tag">
            <span className="live-dot" />
            <span>GeoProof™ Spatial Evidence Archive</span>
          </div>
          <h1 className="view-title">Historical Spatial Query & Audit Ledger</h1>
          <p className="view-subtitle">
            Cryptographically timestamped, persistent evidence ledger recording remote-sensing multi-sensor runs,
            Platt-calibrated verifications, and vector geometry extractions.
          </p>
        </div>

        <button
          type="button"
          className="refresh-archive-btn"
          onClick={onRefresh}
          disabled={loading}
          title="Refresh ledger from database"
        >
          <span className={`refresh-icon ${loading ? "spinning" : ""}`}>↻</span>
          <span>{loading ? "Syncing..." : "Sync Archive"}</span>
        </button>
      </div>

      {/* KPI Stats Bar */}
      <div className="archive-kpi-grid">
        <div className="archive-kpi-card">
          <span className="kpi-label">TOTAL HISTORICAL PROOFS</span>
          <div className="kpi-value-row">
            <span className="kpi-number">{stats.total}</span>
            <span className="kpi-pill">Indexed</span>
          </div>
          <span className="kpi-hint">Recorded in SQLite repository</span>
        </div>

        <div className="archive-kpi-card">
          <span className="kpi-label">VALIDATED PROOF RATE</span>
          <div className="kpi-value-row">
            <span className="kpi-number text-emerald">{stats.supportRate}%</span>
            <span className="kpi-pill pill-emerald">Supported</span>
          </div>
          <span className="kpi-hint">Passed multi-witness criteria</span>
        </div>

        <div className="archive-kpi-card">
          <span className="kpi-label">MEAN PLATT CONFIDENCE</span>
          <div className="kpi-value-row">
            <span className="kpi-number text-cyan">{stats.avgConf}%</span>
            <span className="kpi-pill pill-cyan">Calibrated</span>
          </div>
          <span className="kpi-hint">Temperature scaled (ECE &lt; 0.05)</span>
        </div>

        <div className="archive-kpi-card">
          <span className="kpi-label">INDEXED VECTOR GEOMETRIES</span>
          <div className="kpi-value-row">
            <span className="kpi-number text-amber">{stats.totalGeom.toLocaleString()}</span>
            <span className="kpi-pill pill-amber">Polygons</span>
          </div>
          <span className="kpi-hint">Decomposed GeoJSON features</span>
        </div>
      </div>

      {/* Filters & Search Toolbar */}
      <div className="archive-toolbar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search queries, UUIDs, or tasks..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="archive-search-input"
          />
          {searchTerm && (
            <button
              type="button"
              className="clear-search-btn"
              onClick={() => setSearchTerm("")}
            >
              ✕
            </button>
          )}
        </div>

        <div className="filter-chips-group">
          <span className="filter-label">Status:</span>
          {["all", "supported", "refuted", "inconclusive"].map((s) => (
            <button
              key={s}
              type="button"
              className={`filter-chip ${statusFilter === s ? "active" : ""}`}
              onClick={() => setStatusFilter(s)}
            >
              {s.toUpperCase()}
            </button>
          ))}
        </div>

        <div className="filter-chips-group">
          <span className="filter-label">Task:</span>
          {[
            { id: "all", label: "ALL" },
            { id: "optical_sar", label: "OPTICAL+SAR" },
            { id: "change", label: "CHANGE" },
            { id: "water", label: "WATER" },
          ].map((t) => (
            <button
              key={t.id}
              type="button"
              className={`filter-chip ${taskFilter === t.id ? "active" : ""}`}
              onClick={() => setTaskFilter(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Runs List Table / Cards */}
      <div className="archive-cards-list">
        {filteredRuns.length === 0 ? (
          <div className="archive-empty-state">
            <div className="empty-icon">🗂</div>
            <h3>No matching evidence records</h3>
            <p>
              {runs.length === 0
                ? "No spatial analyses have been run yet. Run an analysis from the Studio to record evidence."
                : "Try clearing search keywords or switching filters to see historical runs."}
            </p>
          </div>
        ) : (
          filteredRuns.map((run) => {
            const taskInfo = formatTaskBadge(run.task_type);
            const verdictInfo = formatVerdictBadge(run.verdict_status);
            const confPct = Math.round(run.confidence * 100);
            const reportUrl = artifactUrl(`/artifacts/${run.result_id}/GeoProof_Report.pdf`);

            return (
              <article key={run.result_id} className="archive-run-card">
                {/* Top Row: Meta info & Task Badge */}
                <div className="run-card-top">
                  <div className="run-id-time">
                    <span className="run-time-tag" title={run.created_at}>
                      🕒 {formatRelativeTime(run.created_at)}
                    </span>
                    <button
                      type="button"
                      className="run-uuid-badge"
                      onClick={() => handleCopy(run.result_id)}
                      title="Click to copy full UUID"
                    >
                      <span>{run.result_id.slice(0, 8)}...</span>
                      <span className="copy-icon">{copiedId === run.result_id ? "✓" : "📋"}</span>
                    </button>
                  </div>

                  <span className={`task-badge ${taskInfo.className}`}>
                    <span className="task-icon">{taskInfo.icon}</span>
                    <span>{taskInfo.label}</span>
                  </span>
                </div>

                {/* Middle Row: Query Text & Verdict */}
                <div className="run-card-body">
                  <h3 className="run-query-text">"{run.query_text}"</h3>

                  <div className="run-verdict-metrics">
                    <span className={`verdict-pill ${verdictInfo.className}`}>
                      {verdictInfo.label}
                    </span>

                    <div className="run-confidence-bar" title={`Calibrated Platt Confidence: ${confPct}%`}>
                      <div className="conf-track">
                        <div
                          className="conf-fill"
                          style={{
                            width: `${confPct}%`,
                            backgroundColor:
                              confPct >= 70
                                ? "var(--accent-emerald)"
                                : confPct >= 40
                                ? "var(--accent-cyan)"
                                : "var(--accent-amber)",
                          }}
                        />
                      </div>
                      <span className="conf-value">{confPct}% Conf.</span>
                    </div>

                    <div className="run-geom-badge" title="Indexed GeoJSON Polygon Geometries">
                      <span className="geom-icon">▱</span>
                      <span>{run.geometries_count.toLocaleString()} Polygons</span>
                    </div>
                  </div>
                </div>

                {/* Bottom Row: Actions */}
                <div className="run-card-actions">
                  <button
                    type="button"
                    className="action-btn-studio"
                    onClick={() => handleLoad(run.result_id)}
                    disabled={loadingResultId === run.result_id}
                    title="Load complete raster viewport, layers, and evidence into Analysis Studio"
                  >
                    <span>{loadingResultId === run.result_id ? "Loading..." : "⇄ Load in Studio"}</span>
                  </button>

                  {reportUrl && (
                    <a
                      href={reportUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="action-btn-report"
                      title="Open publication-grade GeoProof audit report in a new tab"
                    >
                      <span>📄 View PDF</span>
                    </a>
                  )}

                  {reportUrl && (
                    <a
                      href={reportUrl}
                      download={`GeoProof_Report_${run.result_id.slice(0, 8)}.pdf`}
                      className="action-btn-download"
                      title="Download PDF audit report"
                    >
                      <span>⇩ Download</span>
                    </a>
                  )}
                </div>
              </article>
            );
          })
        )}
      </div>
    </div>
  );
};
