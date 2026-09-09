import React, { useState, useMemo } from "react";
import type { AnalysisRunRecord } from "../types";
import { artifactUrl, downloadPdfReport } from "../api";

interface ReportsViewProps {
  runs: AnalysisRunRecord[];
  loading: boolean;
  onRefresh: () => void;
  onLoadResult: (resultId: string) => void;
}

function formatDate(dateString: string): string {
  try {
    const d = new Date(dateString);
    return d.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateString;
  }
}

export const ReportsView: React.FC<ReportsViewProps> = ({
  runs,
  loading,
  onRefresh,
  onLoadResult,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPdfUrl, setSelectedPdfUrl] = useState<string | null>(null);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);

  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
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
  }, [runs, searchTerm]);

  const kpis = useMemo(() => {
    const total = runs.length;
    const verified = runs.filter((r) => r.verdict_status.toLowerCase() === "supported").length;
    const avgConf =
      total > 0
        ? Math.round(
            (runs.reduce((acc, curr) => acc + (curr.confidence || 0), 0) / total) * 100
          )
        : 0;
    const totalGeom = runs.reduce((acc, curr) => acc + (curr.geometries_count || 0), 0);
    return { total, verified, avgConf, totalGeom };
  }, [runs]);

  const latestRun = runs[0];
  const latestReportUrl = latestRun ? artifactUrl(`/artifacts/${latestRun.result_id}/GeoProof_Report.pdf`) : null;

  return (
    <div className="reports-view-container">
      {/* Top Banner: Defense-grade Audit Certificate */}
      <div className="reports-hero-banner">
        <div className="hero-content">
          <div className="hero-badge">
            <span className="shield-icon">🛡</span>
            <span>GeoProof™ Formal Spatial Audit Ledger</span>
          </div>
          <h1 className="hero-title">Automated Geospatial Verification Reports</h1>
          <p className="hero-description">
            Defense-grade, publication-ready PDF audit certificates generated automatically for every remote-sensing
            analysis run. Each report incorporates dual-sensor evidence tables, Platt calibration statistics,
            spectral index maps, and cryptographic execution hashes.
          </p>

          {latestReportUrl && (
            <div className="hero-cta-row">
              <a
                href={latestReportUrl}
                target="_blank"
                rel="noreferrer"
                className="hero-primary-btn"
                title="Open the latest generated audit report"
              >
                <span>📄 View Latest Audit Report</span>
                <span className="btn-arrow">↗</span>
              </a>
              <button
                type="button"
                onClick={() => downloadPdfReport(latestRun.result_id, `GeoProof_Report_${latestRun.result_id.slice(0, 8)}.pdf`)}
                className="hero-secondary-btn"
                title="Download publication-grade GeoProof PDF report"
              >
                <span>⇩ Download Latest PDF</span>
              </button>
            </div>
          )}
        </div>

        <div className="hero-specs-card">
          <div className="spec-header">
            <span className="spec-icon">⚖</span>
            <strong>Audit Standards Compliance</strong>
          </div>
          <ul className="spec-list">
            <li>
              <span className="check-bullet">✓</span>
              <span><strong>ISRO SIH26167:</strong> Evidence-grounded spatial claim verifier</span>
            </li>
            <li>
              <span className="check-bullet">✓</span>
              <span><strong>Calibration:</strong> Temperature-scaled Platt calibration (ECE &lt; 0.05)</span>
            </li>
            <li>
              <span className="check-bullet">✓</span>
              <span><strong>Dual Witnesses:</strong> Deterministic spectral + deep foundation models</span>
            </li>
            <li>
              <span className="check-bullet">✓</span>
              <span><strong>Format:</strong> High-resolution Vector PDF via ReportLab</span>
            </li>
          </ul>
        </div>
      </div>

      {/* KPI Metrics */}
      <div className="reports-kpi-row">
        <div className="report-kpi">
          <span className="kpi-title">TOTAL AUDIT REPORTS</span>
          <span className="kpi-number">{kpis.total}</span>
          <span className="kpi-subtitle">Generated & archived</span>
        </div>
        <div className="report-kpi">
          <span className="kpi-title">SUPPORTED PROOFS</span>
          <span className="kpi-number text-emerald">{kpis.verified}</span>
          <span className="kpi-subtitle">Passed multi-sensor cross-check</span>
        </div>
        <div className="report-kpi">
          <span className="kpi-title">MEAN CONFIDENCE</span>
          <span className="kpi-number text-cyan">{kpis.avgConf}%</span>
          <span className="kpi-subtitle">Calibrated reliability</span>
        </div>
        <div className="report-kpi">
          <span className="kpi-title">POLYGON BOUNDARIES</span>
          <span className="kpi-number text-amber">{kpis.totalGeom.toLocaleString()}</span>
          <span className="kpi-subtitle">Vector proof coordinates</span>
        </div>
      </div>

      {/* Report Explorer Toolbar */}
      <div className="reports-toolbar">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search reports by query or result UUID..."
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

        <button
          type="button"
          className="refresh-archive-btn"
          onClick={onRefresh}
          disabled={loading}
          title="Refresh reports ledger"
        >
          <span className={`refresh-icon ${loading ? "spinning" : ""}`}>↻</span>
          <span>{loading ? "Syncing..." : "Sync Reports"}</span>
        </button>
      </div>

      {/* Reports Table Grid */}
      <div className="reports-table-wrapper">
        {filteredRuns.length === 0 ? (
          <div className="archive-empty-state">
            <div className="empty-icon">📑</div>
            <h3>No reports found</h3>
            <p>Execute an analysis in the Studio to generate formal PDF audit reports.</p>
          </div>
        ) : (
          <table className="reports-table">
            <thead>
              <tr>
                <th>Report Document</th>
                <th>Query / Claim Analyzed</th>
                <th>Task Mode</th>
                <th>Verdict Status</th>
                <th>Platt Conf.</th>
                <th>Generated At</th>
                <th className="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((run) => {
                const pdfUrl = artifactUrl(`/artifacts/${run.result_id}/GeoProof_Report.pdf`);
                const isSupported = run.verdict_status.toLowerCase() === "supported";

                return (
                  <tr key={run.result_id} className="report-table-row">
                    {/* Document Name */}
                    <td className="doc-name-cell">
                      <div className="doc-icon-wrap">
                        <span className="doc-icon">📄</span>
                        <div>
                          <strong className="doc-title">GeoProof_Report_{run.result_id.slice(0, 8)}.pdf</strong>
                          <span className="doc-uuid">{run.result_id.slice(0, 18)}...</span>
                        </div>
                      </div>
                    </td>

                    {/* Query */}
                    <td className="query-cell" title={run.query_text}>
                      <span className="query-truncated">"{run.query_text}"</span>
                    </td>

                    {/* Task Mode */}
                    <td>
                      <span className="task-pill-mini">
                        {run.task_type.replace(/_/g, " ")}
                      </span>
                    </td>

                    {/* Verdict */}
                    <td>
                      <span
                        className={`verdict-pill-mini ${
                          isSupported ? "pill-emerald" : "pill-amber"
                        }`}
                      >
                        {run.verdict_status.toUpperCase()}
                      </span>
                    </td>

                    {/* Confidence */}
                    <td>
                      <span className="conf-badge-mini">
                        {Math.round(run.confidence * 100)}%
                      </span>
                    </td>

                    {/* Date */}
                    <td className="date-cell">
                      <span>{formatDate(run.created_at)}</span>
                    </td>

                    {/* Actions */}
                    <td className="actions-cell">
                      <div className="action-buttons-group">
                        <button
                          type="button"
                          className="btn-preview-report"
                          onClick={() => {
                            setSelectedPdfUrl(pdfUrl || null);
                            setSelectedResultId(run.result_id);
                          }}
                          title="Preview PDF inside modal viewer"
                        >
                          👁 Preview
                        </button>

                        {pdfUrl && (
                          <a
                            href={pdfUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="btn-open-pdf"
                            title="Open PDF in new tab"
                          >
                            ↗ Tab
                          </a>
                        )}

                        {pdfUrl && (
                          <button
                            type="button"
                            onClick={() => downloadPdfReport(run.result_id, `GeoProof_Report_${run.result_id.slice(0, 8)}.pdf`)}
                            className="btn-download-pdf"
                            title="Download publication-grade GeoProof PDF report"
                          >
                            ⇩
                          </button>
                        )}

                        <button
                          type="button"
                          className="btn-inspect-studio"
                          onClick={() => onLoadResult(run.result_id)}
                          title="Load this result in Analysis Studio"
                        >
                          ⇄ Studio
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Embedded PDF Preview Modal */}
      {selectedPdfUrl && (
        <div className="pdf-modal-backdrop" onClick={() => setSelectedPdfUrl(null)}>
          <div className="pdf-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="pdf-modal-header">
              <div className="modal-title-row">
                <span className="doc-icon">📄</span>
                <div>
                  <h3>GeoProof™ Audit Report Preview</h3>
                  <span className="modal-uuid">Result ID: {selectedResultId}</span>
                </div>
              </div>

              <div className="modal-controls">
                <a
                  href={selectedPdfUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="modal-open-tab-btn"
                >
                  ↗ Open in Full Tab
                </a>
                <button
                  type="button"
                  onClick={() => selectedResultId && downloadPdfReport(selectedResultId, `GeoProof_Report_${selectedResultId.slice(0, 8)}.pdf`)}
                  className="modal-download-btn"
                  title="Download publication-grade GeoProof PDF report"
                >
                  ⇩ Download PDF
                </button>
                <button
                  type="button"
                  className="modal-close-btn"
                  onClick={() => setSelectedPdfUrl(null)}
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="pdf-modal-body">
              <iframe
                src={selectedPdfUrl}
                title="GeoProof Audit Report PDF"
                className="pdf-iframe-viewer"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
