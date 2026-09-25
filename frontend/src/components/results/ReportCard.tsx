import React from "react";
import { FileText, Download, Eye, ExternalLink } from "lucide-react";
import { downloadPdfReport, artifactUrl } from "../../api";

interface ReportCardProps {
  resultId: string;
  reportName?: string;
  onPreview: (resultId: string) => void;
}

export const ReportCard: React.FC<ReportCardProps> = ({
  resultId,
  reportName = "SatQuery GeoProof Analysis Report",
  onPreview,
}) => {
  return (
    <div className="report-attachment-card">
      <div className="report-card-icon-box">
        <FileText size={20} className="report-pdf-icon" />
      </div>

      <div className="report-card-info">
        <div className="report-card-title">{reportName}</div>
        <div className="report-card-sub">
          <span className="file-type-tag">PDF Document</span>
          <span className="bullet-sep">•</span>
          <span className="file-status-tag">GeoProof Verified</span>
        </div>
      </div>

      <div className="report-card-actions">
        <button
          type="button"
          className="report-btn preview-btn"
          onClick={() => onPreview(resultId)}
          title="Preview report inline"
        >
          <Eye size={14} />
          <span>Preview</span>
        </button>

        <button
          type="button"
          className="report-btn download-btn"
          onClick={() => downloadPdfReport(resultId)}
          title="Download PDF report"
        >
          <Download size={14} />
          <span>Download</span>
        </button>
      </div>
    </div>
  );
};
