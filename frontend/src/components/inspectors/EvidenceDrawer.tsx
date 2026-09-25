import React from "react";
import { X, Layers, Maximize2, Download } from "lucide-react";
import type { AnalysisResponse } from "../../types";
import { artifactUrl } from "../../api";

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

  return (
    <>
      <div className="drawer-backdrop visible" onClick={onClose} />
      <aside className="evidence-drawer open" aria-label="Evidence Gallery Drawer">
        <div className="inspector-header">
          <div className="inspector-header-left">
            <Layers size={16} className="inspector-icon" />
            <h3 className="inspector-title">All Visual Evidence ({imageArtifacts.length})</h3>
          </div>
          <button
            type="button"
            className="drawer-close-btn"
            onClick={onClose}
            title="Close drawer (Esc)"
          >
            <X size={16} />
          </button>
        </div>

        <div className="evidence-drawer-body">
          <div className="evidence-drawer-grid">
            {imageArtifacts.map((art, idx) => {
              const fullUrl = artifactUrl(art.url) || "";
              const cleanName = art.name.replace(/_/g, " ").replace(/\.[^/.]+$/, "");

              return (
                <div key={idx} className="evidence-grid-card">
                  <div
                    className="evidence-card-img-wrap"
                    onClick={() => onOpenLightbox(fullUrl, cleanName)}
                  >
                    <img src={fullUrl} alt={cleanName} loading="lazy" />
                    <div className="evidence-card-overlay">
                      <Maximize2 size={16} />
                      <span>Maximize</span>
                    </div>
                  </div>

                  <div className="evidence-card-footer">
                    <span className="evidence-art-name" title={cleanName}>
                      {cleanName}
                    </span>
                    <a
                      href={fullUrl}
                      download={`${art.name}.png`}
                      className="evidence-download-icon"
                      title="Download image"
                    >
                      <Download size={13} />
                    </a>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </aside>
    </>
  );
};
