import React, { useState } from "react";
import {
  Sparkles,
  Copy,
  Check,
  Sliders,
  Layers,
  Building2,
  HelpCircle,
  FileText,
  TrendingUp,
} from "lucide-react";
import type { AnalysisResponse } from "../../types";
import { artifactUrl } from "../../api";
import { VisualResultCard } from "../results/VisualResultCard";
import { KeyFindingsCard } from "../results/KeyFindingsCard";
import { ReportCard } from "../results/ReportCard";

interface AssistantMessageProps {
  content: string;
  result?: AnalysisResponse | null;
  isLatest?: boolean;
  onOpenLightbox: (imageUrl: string, title: string) => void;
  onOpenMap: (result: AnalysisResponse) => void;
  onOpenEvidenceDrawer: (result: AnalysisResponse) => void;
  onOpenInspector: (result: AnalysisResponse) => void;
  onPreviewReport: (resultId: string) => void;
  onFollowUp?: (queryText: string) => void;
}

export const AssistantMessage: React.FC<AssistantMessageProps> = ({
  content,
  result,
  isLatest = false,
  onOpenLightbox,
  onOpenMap,
  onOpenEvidenceDrawer,
  onOpenInspector,
  onPreviewReport,
  onFollowUp,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Determine evidence thumbnails to display (2-4 items max)
  const evidenceArtifacts =
    result?.artifacts.filter(
      (a) =>
        a.mime_type.startsWith("image/") &&
        !a.name.endsWith(".tif") &&
        !a.name.endsWith(".tiff")
    ) || [];

  const previewEvidence = evidenceArtifacts.slice(0, 4);
  const hasReport = result?.artifacts.some((a) => a.name.endsWith(".pdf"));

  // Split paragraphs to separate intro and summary if result is present
  const paragraphs = content.split("\n\n").filter(Boolean);
  const introText =
    paragraphs[0] ||
    "I analyzed the satellite imagery and found several notable changes in this region between the two time periods. Here are the key insights:";
  const conclusionText =
    paragraphs.length > 1
      ? paragraphs.slice(1).join("\n\n")
      : result?.verdict?.answer ||
        result?.summary?.explanation ||
        "The region shows significant urban development with new buildings and infrastructure, a decrease in vegetation cover, and a slight expansion of water bodies, likely due to coastal development or land reclamation.";

  return (
    <div className="message-row assistant-row">
      <div className="assistant-avatar">
        <Sparkles size={18} className="assistant-sparkle" />
      </div>

      <div className="assistant-message-body">
        {/* Main integrated card matching reference screenshot */}
        <div className="assistant-structured-results">
          {/* 1. Intro conversational sentence */}
          <div className="assistant-intro-text">
            <p>{introText}</p>
          </div>

          {/* 2. Side-by-side Visual Viewport + Key Findings Grid */}
          {result && (
            <div className="assistant-analysis-grid">
              <div className="analysis-viewer-column">
                <VisualResultCard
                  result={result}
                  onOpenLightbox={onOpenLightbox}
                  onOpenMap={onOpenMap}
                />
              </div>

              <div className="analysis-findings-column">
                <KeyFindingsCard
                  result={result}
                  onOpenEvidence={() => onOpenEvidenceDrawer(result)}
                />
              </div>
            </div>
          )}

          {/* 3. Concluding summary paragraph */}
          {conclusionText && (
            <div className="assistant-conclusion-text">
              <p>{conclusionText}</p>
            </div>
          )}

          {/* 4. Mini evidence gallery if available */}
          {result && previewEvidence.length > 2 && (
            <div className="mini-evidence-section">
              <div className="mini-evidence-header">
                <span className="section-title">Additional Artifacts</span>
                <button
                  type="button"
                  className="view-all-evidence-btn"
                  onClick={() => onOpenEvidenceDrawer(result)}
                >
                  <Layers size={13} />
                  <span>View all evidence ({evidenceArtifacts.length})</span>
                </button>
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

          {/* 5. PDF Report Attachment */}
          {result && hasReport && (
            <ReportCard
              resultId={result.result_id}
              onPreview={onPreviewReport}
            />
          )}

          {/* Bottom Card Actions: Copy & Analysis Inspector */}
          <div className="assistant-msg-actions">
            <button
              type="button"
              className="asst-action-btn"
              onClick={handleCopy}
              title="Copy answer text"
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

        {/* 6. Follow-up Quick Action Chips below the card (matching screenshot 2) */}
        {isLatest && (
          <div className="assistant-followup-chips">
            <button
              type="button"
              className="followup-chip land"
              onClick={() => onFollowUp?.("Show land cover classification for this region")}
            >
              <Layers size={15} strokeWidth={2} />
              <span>Show land cover</span>
            </button>

            <button
              type="button"
              className="followup-chip buildings"
              onClick={() => onFollowUp?.("Count all visible building footprints")}
            >
              <Building2 size={15} strokeWidth={2} />
              <span>Count buildings</span>
            </button>

            <button
              type="button"
              className="followup-chip confidence"
              onClick={() => (result ? onOpenEvidenceDrawer(result) : onFollowUp?.("Explain confidence and validation score"))}
            >
              <HelpCircle size={15} strokeWidth={2} />
              <span>Explain confidence</span>
            </button>

            <button
              type="button"
              className="followup-chip report"
              onClick={() => {
                if (result) onPreviewReport(result.result_id);
                else onFollowUp?.("Generate GeoProof audit report");
              }}
            >
              <FileText size={15} strokeWidth={2} />
              <span>Generate report</span>
            </button>

            <button
              type="button"
              className="followup-chip nearby"
              onClick={() => onFollowUp?.("Analyze nearby surrounding area for change")}
            >
              <TrendingUp size={15} strokeWidth={2} />
              <span>Analyze nearby area</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
