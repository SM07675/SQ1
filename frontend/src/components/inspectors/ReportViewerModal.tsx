import React, { useEffect } from "react";
import { Download, ExternalLink, X, FileText } from "lucide-react";
import { artifactUrl, downloadPdfReport } from "../../api";

interface ReportViewerModalProps {
  isOpen: boolean;
  resultId: string | null;
  onClose: () => void;
}

export const ReportViewerModal: React.FC<ReportViewerModalProps> = ({
  isOpen,
  resultId,
  onClose,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    if (isOpen) window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !resultId) return null;

  const pdfUrl = artifactUrl(`/artifacts/${resultId}/GeoProof_Report.pdf`);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="report-modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="report-modal-header">
          <div className="report-header-left">
            <FileText size={18} className="report-header-icon" />
            <div className="report-header-text">
              <h3 className="report-modal-title">SatQuery GeoProof Analysis Report</h3>
              <span className="report-id-code">ID: {resultId}</span>
            </div>
          </div>

          <div className="report-header-actions">
            <button
              type="button"
              className="report-action-btn"
              onClick={() => downloadPdfReport(resultId)}
              title="Download PDF file"
            >
              <Download size={15} />
              <span>Download PDF</span>
            </button>

            {pdfUrl && (
              <a
                href={pdfUrl}
                target="_blank"
                rel="noreferrer"
                className="report-action-btn"
                title="Open PDF in new browser tab"
              >
                <ExternalLink size={15} />
                <span>Open in Tab</span>
              </a>
            )}

            <button
              type="button"
              className="report-action-btn close"
              onClick={onClose}
              title="Close modal (Esc)"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="report-modal-body">
          {pdfUrl ? (
            <iframe
              src={`${pdfUrl}#toolbar=0`}
              title="SatQuery GeoProof Report Preview"
              className="report-pdf-frame"
            />
          ) : (
            <div className="report-loading-placeholder">
              <p>Preparing PDF report document...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
