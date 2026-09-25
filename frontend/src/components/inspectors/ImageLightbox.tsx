import React, { useState, useEffect, useRef } from "react";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  ExternalLink,
  Download,
  X,
} from "lucide-react";

interface ImageLightboxProps {
  isOpen: boolean;
  imageUrl: string;
  imageTitle: string;
  onClose: () => void;
}

export const ImageLightbox: React.FC<ImageLightboxProps> = ({
  isOpen,
  imageUrl,
  imageTitle,
  onClose,
}) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (isOpen) {
      setScale(1);
      setPosition({ x: 0, y: 0 });
    }
  }, [isOpen, imageUrl]);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
    }
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !imageUrl) return null;

  const handleZoomIn = () => setScale((s) => Math.min(s + 0.25, 4));
  const handleZoomOut = () => setScale((s) => Math.max(s - 0.25, 0.5));
  const handleResetZoom = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };
  const handleFit = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setScale((s) => Math.min(Math.max(s + delta, 0.4), 4.5));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (scale > 1) {
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX - position.x, y: e.clientY - position.y };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && scale > 1) {
      setPosition({
        x: e.clientX - dragStartRef.current.x,
        y: e.clientY - dragStartRef.current.y,
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const cleanTitle = imageTitle.replace(/_/g, " ").replace(/\.[^/.]+$/, "");

  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox-content-box" onClick={(e) => e.stopPropagation()}>
        {/* Top Floating Control Bar */}
        <div className="lightbox-navbar">
          <div className="lightbox-title-group">
            <span className="lightbox-title">{cleanTitle}</span>
            <span className="lightbox-tag">Full Resolution</span>
          </div>

          <div className="lightbox-controls-group">
            <button
              type="button"
              className="lightbox-tool-btn"
              onClick={handleZoomOut}
              title="Zoom out (-)"
            >
              <ZoomOut size={16} />
            </button>

            <button
              type="button"
              className="lightbox-tool-btn zoom-level-btn"
              onClick={handleResetZoom}
              title="Reset Zoom (100%)"
            >
              <span>{Math.round(scale * 100)}%</span>
            </button>

            <button
              type="button"
              className="lightbox-tool-btn"
              onClick={handleZoomIn}
              title="Zoom in (+)"
            >
              <ZoomIn size={16} />
            </button>

            <button
              type="button"
              className="lightbox-tool-btn"
              onClick={handleFit}
              title="Fit to Screen"
            >
              <Maximize2 size={16} />
            </button>

            <div className="lightbox-divider" />

            <a
              href={imageUrl}
              target="_blank"
              rel="noreferrer"
              className="lightbox-tool-btn"
              title="Open full resolution in new tab"
            >
              <ExternalLink size={16} />
            </a>

            <a
              href={imageUrl}
              download={`${imageTitle || "satellite_imagery"}.png`}
              className="lightbox-tool-btn"
              title="Download image"
            >
              <Download size={16} />
            </a>

            <button
              type="button"
              className="lightbox-tool-btn close-btn"
              onClick={onClose}
              title="Close (Esc)"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Viewport with wheel zoom and pan */}
        <div
          className="lightbox-viewport"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default" }}
        >
          <img
            src={imageUrl}
            alt={cleanTitle}
            className="lightbox-target-img"
            draggable={false}
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transition: isDragging ? "none" : "transform 0.15s ease-out",
            }}
          />
        </div>

        <div className="lightbox-footer-hint">
          <span>Scroll to zoom • Drag to pan when zoomed • Press Esc to close</span>
        </div>
      </div>
    </div>
  );
};
