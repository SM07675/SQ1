import React from "react";
import { createPortal } from "react-dom";
import { X, Layers, Maximize2, Download } from "lucide-react";
import type { AnalysisResponse } from "../../types";
import { artifactUrl } from "../../api";
import { getPreviewUrl } from "../../utils/tiffViewer";

function evidenceLabel(name: string): string {
  const stem = name.replace(/\.[^.]+$/, "").toLowerCase();
  if (/^(preview|prepared)[_ -]?1$/.test(stem)) return "Original image";
  if (/^(preview|prepared)[_ -]?2$/.test(stem)) return "Later image";
  if (stem.includes("water_overlay")) return "Water overlay";
  if (stem.includes("land_cover_overlay")) return "Land cover overlay";
  if (stem.includes("land_only_overlay")) return "Land overlay";
  if (stem.includes("building") && stem.includes("overlay")) return "Building overlay";
  if (stem.includes("change") && stem.includes("mask")) return "Change mask";
  return stem.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

interface EvidenceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  result: AnalysisResponse | null;
  onOpenLightbox: (imageUrl: string, title: string) => void;
}

export const EvidenceDrawer: React.FC<EvidenceDrawerProps> = ({
  isOpen,
  onClose,
  result,
  onOpenLightbox,
}) => {
  if (!isOpen || !result) return null;

  const imageArtifacts = result.artifacts.filter(
    (a) =>
      a.mime_type.startsWith("image/") &&
      !a.name.endsWith(".tif") &&
      !a.name.endsWith(".tiff")
  );

  return createPortal(
    <>
      <div className="evidence-overlay" onClick={onClose} aria-hidden="true" />
      <aside className="evidence-drawer-panel" aria-label="Visual evidence" role="dialog" aria-modal="true">
        <div className="inspector-header">
          <div className="inspector-header-left">
            <Layers size={16} className="inspector-icon" />
            <div>
              <h3 className="inspector-title">Visual evidence ({imageArtifacts.length})</h3>
              <p className="evidence-drawer-subtitle">Select an image to inspect it at full size.</p>
            </div>
          </div>
          <button
            type="button"
            className="drawer-close-btn"
            onClick={onClose}
            title="Close drawer (Esc)"
            aria-label="Close visual evidence"
          >
            <X size={16} />
          </button>
        </div>

        <div className="evidence-drawer-body">
          <div className="evidence-drawer-grid">
            {imageArtifacts.map((art, idx) => {
              const fullUrl = artifactUrl(art.url) || "";
              const displayThumb = getPreviewUrl(fullUrl) || fullUrl;
              const cleanName = evidenceLabel(art.name);

              return (
                <article key={`${art.url}-${idx}`} className="evidence-grid-card">
                  <button
                    type="button"
                    className="evidence-card-img-wrap"
                    onClick={() => onOpenLightbox(fullUrl, cleanName)}
                    aria-label={`View ${cleanName} at full size`}
                  >
                    <img src={displayThumb} alt={cleanName} loading="lazy" />
                    <div className="evidence-card-overlay">
                      <Maximize2 size={16} />
                      <span>Maximize</span>
                    </div>
                  </button>

                  <div className="evidence-card-footer">
                    <span className="evidence-art-name" title={cleanName}>
                      {cleanName}
                    </span>
                    <a
                      href={fullUrl}
                      download={art.name}
                      className="evidence-download-icon"
                      title={`Download ${cleanName}`}
                      aria-label={`Download ${cleanName}`}
                    >
                      <Download size={13} />
                    </a>
                  </div>
                </article>
              );
            })}
            {imageArtifacts.length === 0 && <p className="evidence-empty">No visual evidence was generated for this analysis.</p>}
          </div>
        </div>
      </aside>
    </>,
    document.body
  );
};
