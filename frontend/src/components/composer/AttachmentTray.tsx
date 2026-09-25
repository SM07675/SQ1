import React from "react";
import { X, ArrowLeftRight, Image as ImageIcon, MapPin, FileCheck } from "lucide-react";

export interface AttachedFileItem {
  file: File;
  previewUrl?: string;
  isGeotiff?: boolean;
  dimensions?: { width: number; height: number };
  role?: "earlier" | "later" | "optical" | "sar" | "single";
}

interface AttachmentTrayProps {
  files: AttachedFileItem[];
  onRemove: (index: number) => void;
  onSwap?: () => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const AttachmentTray: React.FC<AttachmentTrayProps> = ({
  files,
  onRemove,
  onSwap,
}) => {
  if (files.length === 0) return null;

  const isPair = files.length === 2;

  return (
    <div className="attachment-tray">
      <div className="attachment-cards-scroll">
        {files.map((item, index) => {
          const ext = item.file.name.split(".").pop()?.toUpperCase() || "IMG";
          const isTiff = ext === "TIF" || ext === "TIFF";

          let roleLabel = "";
          if (isPair) {
            roleLabel = index === 0 ? "Earlier / Image A" : "Later / Image B";
            if (item.role === "optical") roleLabel = "Optical";
            if (item.role === "sar") roleLabel = "SAR";
          }

          return (
            <div key={`${item.file.name}-${index}`} className="attachment-chip">
              <div className="chip-thumbnail-wrap">
                {item.previewUrl ? (
                  <img src={item.previewUrl} alt={item.file.name} className="chip-thumbnail" />
                ) : (
                  <div className="chip-fallback-icon">
                    <ImageIcon size={16} />
                  </div>
                )}
                {isTiff && <span className="tiff-badge">GeoTIFF</span>}
              </div>

              <div className="chip-info">
                <div className="chip-name-row">
                  <span className="chip-filename" title={item.file.name}>
                    {item.file.name}
                  </span>
                  {roleLabel && <span className="chip-role-tag">{roleLabel}</span>}
                </div>

                <div className="chip-meta-row">
                  <span className="chip-size">{formatFileSize(item.file.size)}</span>
                  {item.dimensions && (
                    <span className="chip-dims">
                      {item.dimensions.width}×{item.dimensions.height}
                    </span>
                  )}
                  {isTiff && (
                    <span className="chip-geo-indicator" title="Georeferenced raster metadata">
                      <MapPin size={10} /> Georeferenced
                    </span>
                  )}
                </div>
              </div>

              <button
                type="button"
                className="chip-remove-btn"
                onClick={() => onRemove(index)}
                title="Remove attachment"
                aria-label="Remove attachment"
              >
                <X size={13} />
              </button>
            </div>
          );
        })}
      </div>

      {isPair && onSwap && (
        <button
          type="button"
          className="swap-pair-btn"
          onClick={onSwap}
          title="Swap Earlier / Later order"
        >
          <ArrowLeftRight size={13} />
          <span>Swap Pair</span>
        </button>
      )}
    </div>
  );
};
