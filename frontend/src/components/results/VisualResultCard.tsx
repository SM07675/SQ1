import React, { useState, useRef, useEffect } from "react";
import {
  Maximize2,
  Map,
  Download,
  SlidersHorizontal,
  ChevronDown,
  Layers,
  Sparkles,
} from "lucide-react";
import type { AnalysisResponse, ArtifactRef } from "../../types";
import { artifactUrl, downloadPdfReport } from "../../api";

interface VisualResultCardProps {
  result: AnalysisResponse;
  onOpenLightbox: (imageUrl: string, title: string) => void;
  onOpenMap: (result: AnalysisResponse) => void;
}

export const VisualResultCard: React.FC<VisualResultCardProps> = ({
  result,
  onOpenLightbox,
  onOpenMap,
}) => {
  const [activeLayer, setActiveLayer] = useState<string>("result");
  const [compareMode, setCompareMode] = useState<"slider" | "off">("off");
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState<boolean>(false);
  const [showDownloadMenu, setShowDownloadMenu] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const downloadMenuRef = useRef<HTMLDivElement>(null);

  // Close download menu on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (downloadMenuRef.current && !downloadMenuRef.current.contains(e.target as Node)) {
        setShowDownloadMenu(false);
      }
    };
    if (showDownloadMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showDownloadMenu]);

  // Identify original preview artifact
  const originalArtifact = result.artifacts.find(
    (a) =>
      a.name.includes("preview_1") ||
      a.name.includes("original") ||
      a.name.includes("input_a")
  );

  // Identify secondary image if bi-temporal or optical-sar
  const secondaryArtifact = result.artifacts.find(
    (a) => a.name.includes("preview_2") || a.name.includes("input_b")
  );

  const queryLower = (result.query || "").toLowerCase();
  const taskName = (result.task_plan?.task || "").toLowerCase();
  const useQueryForLayer = !taskName || taskName === "grounding";
  const isLandResult = taskName === "land_cover" || (useQueryForLayer && /\b(land|soil|cover|grass)\b/.test(queryLower));
  const isBuildingResult = ["building_count", "building_detection", "buildings"].includes(taskName) || (useQueryForLayer && /\b(buildings?|footprints?)\b/.test(queryLower));
  const isWaterResult = ["water_analysis", "water_detection"].includes(taskName) || (useQueryForLayer && /\b(water|lake|river|flood)\b/.test(queryLower));
  const preferredOverlayNames = isLandResult
    ? ["land_cover_overlay.png", "land_only_overlay.png"]
    : isBuildingResult
      ? ["buildings_overlay.png", "building_overlay.png"]
      : isWaterResult
        ? ["water_overlay.png"]
        : [];
  const overlayArtifact = preferredOverlayNames
    .map((name) => result.artifacts.find((a) => a.name === name))
    .find(Boolean) || result.artifacts.find(
      (a) => a.name.includes("overlay") || a.name.includes("change_mask") || a.name.includes("detection_mask")
    );

  // List all available image layers that actually exist
  const imageArtifacts = result.artifacts.filter(
    (a) =>
      a.mime_type.startsWith("image/") &&
      !a.name.endsWith(".tif") &&
      !a.name.endsWith(".tiff")
  );

  // Determine current image URL to render based on active layer
  let currentImageUrl = overlayArtifact?.url || originalArtifact?.url || imageArtifacts[0]?.url;

  if (activeLayer === "original" && originalArtifact) {
    currentImageUrl = originalArtifact.url;
  } else if (activeLayer === "secondary" && secondaryArtifact) {
    currentImageUrl = secondaryArtifact.url;
  } else if (activeLayer === "result" && overlayArtifact) {
    currentImageUrl = overlayArtifact.url;
  } else {
    const customMatch = imageArtifacts.find((a) => a.name === activeLayer);
    if (customMatch) currentImageUrl = customMatch.url;
  }

  const resolvedImageUrl = artifactUrl(currentImageUrl) || "";
  const resolvedOriginalUrl = artifactUrl(originalArtifact?.url) || resolvedImageUrl;

  // Detect analysis category for contextual color legend
  const hasLandCover = isLandResult && !!result.artifacts.find((a) => a.name === "land_cover_overlay.png");

  const hasWater =
    !hasLandCover &&
    isWaterResult;

  const hasBuildings =
    isBuildingResult;

  const hasChange =
    queryLower.includes("change") ||
    taskName === "bi_temporal_change" ||
    result.artifacts.some((a) => a.name.includes("change"));

  const landBreakdown = (result.statistics?.land_cover as
    | { breakdown?: Record<string, { pixels?: number }> }
    | undefined)?.breakdown;
  const showLandClass = (name: string) => !landBreakdown || (landBreakdown[name]?.pixels ?? 0) > 0;

  // Handle draggable swipe slider for Before / After comparison
  const handleSliderMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSliderPos(percent);
  };

  const handleMouseDown = () => setIsDraggingSlider(true);
  const handleMouseUp = () => setIsDraggingSlider(false);

  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isDraggingSlider) {
        handleSliderMove(e.clientX);
      }
    };
    const handleGlobalTouchMove = (e: TouchEvent) => {
      if (isDraggingSlider && e.touches.length > 0) {
        handleSliderMove(e.touches[0].clientX);
      }
    };
    const handleGlobalMouseUp = () => setIsDraggingSlider(false);

    if (isDraggingSlider) {
      window.addEventListener("mousemove", handleGlobalMouseMove);
      window.addEventListener("touchmove", handleGlobalTouchMove);
      window.addEventListener("mouseup", handleGlobalMouseUp);
      window.addEventListener("touchend", handleGlobalMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleGlobalMouseMove);
      window.removeEventListener("touchmove", handleGlobalTouchMove);
      window.removeEventListener("mouseup", handleGlobalMouseUp);
      window.removeEventListener("touchend", handleGlobalMouseUp);
    };
  }, [isDraggingSlider]);

  const geojsonArtifact = result.artifacts.find((a) => a.mime_type.includes("json") && a.name.endsWith(".geojson"));
  const reportArtifact = result.artifacts.find((a) => a.name.endsWith(".pdf"));

  return (
    <div className="visual-result-card">
      {/* Top Toolbar */}
      <div className="visual-card-toolbar">
        <div className="layer-switcher-group">
          {originalArtifact && (
            <button
              type="button"
              className={`layer-switch-btn ${
                activeLayer === "original" && compareMode === "off" ? "active" : ""
              }`}
              onClick={() => {
                setActiveLayer("original");
                setCompareMode("off");
              }}
            >
              Original
            </button>
          )}

          {overlayArtifact && (
            <button
              type="button"
              className={`layer-switch-btn ${
                activeLayer === "result" && compareMode === "off" ? "active" : ""
              }`}
              onClick={() => {
                setActiveLayer("result");
                setCompareMode("off");
              }}
            >
              Result Overlay
            </button>
          )}

          {/* Swipe Comparison Mode if both original & overlay or pair exists */}
          {originalArtifact && (overlayArtifact || secondaryArtifact) && (
            <button
              type="button"
              className={`layer-switch-btn compare ${compareMode === "slider" ? "active" : ""}`}
              onClick={() => {
                setCompareMode((prev) => (prev === "slider" ? "off" : "slider"));
                if (compareMode === "off") setActiveLayer("result");
              }}
            >
              <SlidersHorizontal size={13} />
              <span>Compare</span>
            </button>
          )}

          {/* Dynamic Extra Layers Dropdown if multiple exist */}
          {imageArtifacts.length > 2 && (
            <div className="extra-layers-select-wrap">
              <select
                className="extra-layers-select"
                value={activeLayer}
                onChange={(e) => {
                  setActiveLayer(e.target.value);
                  setCompareMode("off");
                }}
              >
                <option value="result">Layers...</option>
                {imageArtifacts.map((art) => (
                  <option key={art.name} value={art.name}>
                    {art.name.replace(/_/g, " ").replace(/\.[^/.]+$/, "")}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Action Controls: Lightbox, Map, Download */}
        <div className="visual-card-actions">
          <button
            type="button"
            className="action-icon-btn"
            onClick={() => onOpenLightbox(resolvedImageUrl, result.query || "Satellite Imagery")}
            title="Expand to Fullscreen Lightbox"
            aria-label="Expand image"
          >
            <Maximize2 size={15} />
            <span className="btn-label-desktop">Expand</span>
          </button>

          <button
            type="button"
            className="action-icon-btn"
            onClick={() => onOpenMap(result)}
            title="Inspect on Interactive Map"
            aria-label="View on map"
          >
            <Map size={15} />
            <span className="btn-label-desktop">Map</span>
          </button>

          {/* Unified Download Menu */}
          <div className="download-dropdown-wrap" ref={downloadMenuRef}>
            <button
              type="button"
              className="action-icon-btn download"
              onClick={() => setShowDownloadMenu((prev) => !prev)}
              title="Download analysis artifacts"
              aria-label="Download menu"
            >
              <Download size={15} />
              <ChevronDown size={12} />
            </button>

            {showDownloadMenu && (
              <div className="download-popover-menu">
                <a
                  href={resolvedImageUrl}
                  download={`SatQuery_Result_${result.result_id.slice(0, 8)}.png`}
                  className="download-menu-item"
                  onClick={() => setShowDownloadMenu(false)}
                >
                  <span>Result Imagery (PNG)</span>
                </a>

                {reportArtifact && (
                  <button
                    type="button"
                    className="download-menu-item"
                    onClick={() => {
                      downloadPdfReport(result.result_id);
                      setShowDownloadMenu(false);
                    }}
                  >
                    <span>GeoProof Audit Report (PDF)</span>
                  </button>
                )}

                {geojsonArtifact && (
                  <a
                    href={artifactUrl(geojsonArtifact.url)}
                    download={`SatQuery_Detection_${result.result_id.slice(0, 8)}.geojson`}
                    className="download-menu-item"
                    onClick={() => setShowDownloadMenu(false)}
                  >
                    <span>Spatial Vector Polygons (GeoJSON)</span>
                  </a>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Image Viewport with Draggable Comparison */}
      <div
        className="visual-viewport-container"
        ref={containerRef}
        onClick={() => {
          if (compareMode === "off") {
            onOpenLightbox(resolvedImageUrl, result.query || "Satellite Analysis");
          }
        }}
      >
        {compareMode === "slider" ? (
          <div className="swipe-compare-viewport" onMouseDown={handleMouseDown}>
            {/* Base Image (Underneath) */}
            <img
              src={resolvedOriginalUrl}
              alt="Original satellite imagery baseline"
              className="compare-img base-img"
              draggable={false}
            />

            {/* Overlaid Image (Clipped by slider position) */}
            <div
              className="compare-overlay-wrap"
              style={{ width: `${sliderPos}%` }}
            >
              <img
                src={resolvedImageUrl}
                alt="Analyzed satellite overlay"
                className="compare-img overlay-img"
                draggable={false}
                style={{
                  width: containerRef.current ? `${containerRef.current.clientWidth}px` : "100%",
                }}
              />
            </div>

            {/* Draggable Divider Handle */}
            <div
              className="swipe-divider"
              style={{ left: `${sliderPos}%` }}
              onMouseDown={handleMouseDown}
              onTouchStart={handleMouseDown}
            >
              <div className="divider-handle">
                <SlidersHorizontal size={14} />
              </div>
            </div>

            {/* Labels */}
            <div className="compare-label left">Overlay</div>
            <div className="compare-label right">Original</div>
          </div>
        ) : (
          <div className="single-viewport">
            <img
              src={resolvedImageUrl}
              alt="High-resolution satellite analysis result"
              className="visual-main-image"
              loading="lazy"
            />
            <div className="image-hover-hint">
              <Maximize2 size={16} />
              <span>Click to maximize</span>
            </div>
          </div>
        )}
      </div>

      {/* Contextual Visual Color Legend */}
      {hasLandCover && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Land Palette:</span>
          <div className="visual-legend-items">
            {showLandClass("water") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#143c8c" }} />
              <span>Water (Dark Blue)</span>
            </span>}
            {showLandClass("vegetation") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#22c55e" }} />
              <span>Grass (Green)</span>
            </span>}
            {showLandClass("woodland") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#166534" }} />
              <span>Forest (Dark Green)</span>
            </span>}
            {showLandClass("unknown") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#c29b38" }} />
              <span>Unclassified Surface (Amber)</span>
            </span>}
            {showLandClass("bare_pervious") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#a68250" }} />
              <span>Bare Soil / Pervious (Tan)</span>
            </span>}
            {showLandClass("road") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#f5c81e" }} />
              <span>Roads / Paved (Yellow)</span>
            </span>}
            {showLandClass("built_up") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#dc3545" }} />
              <span>Built-up (Red)</span>
            </span>}
            {showLandClass("agriculture") && <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#7cb342" }} />
              <span>Cropland (Olive)</span>
            </span>}
          </div>
        </div>
      )}

      {hasWater && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Water Palette:</span>
          <div className="visual-legend-items">
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#143c8c" }} />
              <span>Water Surface (Dark Blue)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#0f2d78" }} />
              <span>Deep Water / Core (Navy)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#c29b38" }} />
              <span>Land / Shoreline (Dark Yellow)</span>
            </span>
          </div>
        </div>
      )}

      {hasBuildings && !hasLandCover && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Building Palette:</span>
          <div className="visual-legend-items">
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#dc3545" }} />
              <span>Building Footprints (Warm Red)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#ffffff", border: "1px solid #999" }} />
              <span>Instance Boundaries (White)</span>
            </span>
          </div>
        </div>
      )}

      {hasChange && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Change Palette:</span>
          <div className="visual-legend-items">
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#ef4444" }} />
              <span>Surface Change (Red)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#10b981" }} />
              <span>Vegetation Shift (Green)</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
