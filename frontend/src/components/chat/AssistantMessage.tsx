import React, { useState } from "react";
import {
  Sparkles,
  Copy,
  Check,
  Sliders,
  Layers,
  FileText,
  Maximize2,
  ExternalLink,
} from "lucide-react";
import type { AnalysisResponse } from "../../types";
import { artifactUrl } from "../../api";
import { ResultSummary } from "../results/ResultSummary";
import { VisualResultCard } from "../results/VisualResultCard";
import { ReportCard } from "../results/ReportCard";

interface AssistantMessageProps {
  content: string;
  result?: AnalysisResponse | null;
  onOpenLightbox: (imageUrl: string, title: string) => void;
  onOpenMap: (result: AnalysisResponse) => void;
  onOpenEvidenceDrawer: (result: AnalysisResponse) => void;
  onOpenInspector: (result: AnalysisResponse) => void;
  onPreviewReport: (resultId: string) => void;
}

export const AssistantMessage: React.FC<AssistantMessageProps> = ({
  content,
  result,
  onOpenLightbox,
  onOpenMap,
  onOpenEvidenceDrawer,
  onOpenInspector,
  onPreviewReport,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Determine evidence thumbnails to display (2-4 items max)
  const evidenceArtifacts = result?.artifacts.filter(
    (a) =>
      a.mime_type.startsWith("image/") &&
      !a.name.endsWith(".tif") &&
      !a.name.endsWith(".tiff")
  ) || [];

  const previewEvidence = evidenceArtifacts.slice(0, 4);

  // Determine if report exists
  const hasReport = result?.artifacts.some((a) => a.name.endsWith(".pdf"));

  return (
    <div className="message-row assistant-row">
      <div className="assistant-avatar">
        <Sparkles size={16} className="assistant-sparkle" />
      </div>

      <div className="assistant-message-body">
        {/* Natural Language Explanation (ChatGPT style, mostly unboxed) */}
        <div className="assistant-conversational-text">
          {content.split("\n\n").map((para, i) => (
            <p key={i}>{para}</p>
          ))}
        </div>

        {/* Structured Result Components (Only present if this turn produced an analysis) */}
        {result && (
          <div className="assistant-structured-results">
            {/* 1. Primary Metrics Summary */}
            <ResultSummary result={result} />

            {/* 2. Visual Satellite Result Viewport */}
            <VisualResultCard
              result={result}
              onOpenLightbox={onOpenLightbox}
              onOpenMap={onOpenMap}
            />

            {/* 3. Evidence Gallery (2-4 items + View all evidence) */}
            {previewEvidence.length > 0 && (
              <div className="mini-evidence-section">
                <div className="mini-evidence-header">
                  <span className="section-title">Visual Evidence</span>
                  {evidenceArtifacts.length > 1 && (
                    <button
                      type="button"
                      className="view-all-evidence-btn"
                      onClick={() => onOpenEvidenceDrawer(result)}
                    >
                      <Layers size={13} />
                      <span>View all evidence ({evidenceArtifacts.length})</span>
                    </button>
                  )}
                </div>

                <div className="mini-evidence-gallery">
                  {previewEvidence.map((art, idx) => {
                    const fullUrl = artifactUrl(art.url) || "";
                    const cleanName = art.name.replace(/_/g, " ").replace(/\.[^/.]+$/, "");
                    return (
                      <div
                        key={idx}
                        className="mini-evidence-card"
                        onClick={() => onOpenLightbox(fullUrl, cleanName)}
                        title="Click to expand evidence"
                      >
                        <img src={fullUrl} alt={cleanName} loading="lazy" />
                        <span className="mini-evidence-label">{cleanName}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 4. PDF Report Card Attachment */}
            {hasReport && (
              <ReportCard
                resultId={result.result_id}
                onPreview={onPreviewReport}
              />
            )}
          </div>
        )}

        {/* Bottom Message Actions Toolbar */}
        <div className="assistant-msg-actions">
          <button
            type="button"
            className="asst-action-btn"
            onClick={handleCopy}
            title="Copy answer"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
            <span>{copied ? "Copied" : "Copy"}</span>
          </button>

          {result && (
            <button
              type="button"
              className="asst-action-btn inspector"
              onClick={() => onOpenInspector(result)}
              title="Inspect technical evidence & geospatial telemetry"
            >
              <Sliders size={13} />
              <span>Analysis Inspector</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
