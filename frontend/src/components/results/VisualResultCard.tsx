import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import {
  Maximize2,
  Map,
  Download,
  SlidersHorizontal,
  ChevronDown,
  ChevronsLeftRight,
} from "lucide-react";
import type { AnalysisResponse } from "../../types";
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
  const containerRef = useRef<HTMLDivElement>(null);
  const downloadMenuRef = useRef<HTMLDivElement>(null);

  const [activeLayer, setActiveLayer] = useState<string>("result");
  const [compareMode, setCompareMode] = useState<"slider" | "off">("slider");
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState<boolean>(false);
  const [showDownloadMenu, setShowDownloadMenu] = useState<boolean>(false);

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

  // List all available displayable image layers (excluding raw GeoTIFFs)
  const imageArtifacts = useMemo(
    () =>
      result.artifacts.filter(
        (a) =>
          a.mime_type.startsWith("image/") &&
          !a.name.endsWith(".tif") &&
          !a.name.endsWith(".tiff")
      ),
    [result.artifacts]
  );

  const queryLower = (result.query || "").toLowerCase();
  const taskName = (result.task_plan?.task || "").toLowerCase();
  const useQueryForLayer = !taskName || taskName === "grounding";

  const isLandResult =
    taskName === "land_cover" ||
    (useQueryForLayer && /\b(land|soil|cover|grass|vegetation|terrain)\b/.test(queryLower));

  const isBuildingResult =
    ["building_count", "building_detection", "buildings"].includes(taskName) ||
    (useQueryForLayer && /\b(buildings?|footprints?|houses?|structures?)\b/.test(queryLower));

  const isWaterResult =
    ["water_analysis", "water_detection"].includes(taskName) ||
    (useQueryForLayer && /\b(water|lake|river|flood|ocean|sea|pond|reservoir)\b/.test(queryLower));

  const isBiTemporal =
    taskName === "bi_temporal_change" ||
    queryLower.includes("change") ||
    queryLower.includes("between") ||
    queryLower.includes("compare") ||
    (result.task_plan?.years && result.task_plan.years.length >= 2);

  // 1. Previous / Baseline / Before / Original Image (ALWAYS Left side of slider)
  const originalArtifact = useMemo(() => {
    return (
      imageArtifacts.find(
        (a) =>
          a.name.includes("preview_1") ||
          a.name.includes("prepared_1") ||
          a.name.includes("before") ||
          a.name.includes("original") ||
          a.name.includes("input_a") ||
          a.role === "earlier"
      ) || imageArtifacts[0]
    );
  }, [imageArtifacts]);

  // 2. Later / After / Secondary Image (for bi-temporal comparison)
  const secondaryArtifact = useMemo(() => {
    return (
      imageArtifacts.find(
        (a) =>
          a.name.includes("preview_2") ||
          a.name.includes("prepared_2") ||
          a.name.includes("after") ||
          a.name.includes("input_b") ||
          a.role === "later"
      ) ||
      (imageArtifacts.length > 1 &&
      imageArtifacts[1] !== originalArtifact &&
      !imageArtifacts[1].name.includes("overlay") &&
      !imageArtifacts[1].name.includes("mask")
        ? imageArtifacts[1]
        : undefined)
    );
  }, [imageArtifacts, originalArtifact]);

  // 3. AI Analysis Result Overlay Artifact
  const overlayArtifact = useMemo(() => {
    const prefersLandOnly =
      isLandResult &&
      !/\b(water|lake|river|flood|ocean|sea|pond|waterbody)\b/.test(queryLower);

    const preferredOverlayNames = prefersLandOnly
      ? ["land_only_overlay.png", "land_cover_overlay.png"]
      : isLandResult
        ? ["land_cover_overlay.png", "land_only_overlay.png"]
        : isBuildingResult
          ? ["buildings_overlay.png", "building_change_overlay.png", "building_overlay.png"]
          : isWaterResult
            ? ["water_overlay.png"]
            : [];

    return (
      preferredOverlayNames
        .map((name) => imageArtifacts.find((a) => a.name === name))
        .find(Boolean) ||
      imageArtifacts.find(
        (a) =>
          a.name.includes("overlay") ||
          a.name.includes("change_mask") ||
          a.name.includes("tinycd") ||
          a.name.includes("difference_map") ||
          a.name.includes("detection_mask")
      )
    );
  }, [imageArtifacts, isLandResult, isBuildingResult, isWaterResult, queryLower]);

  // Comparison target in bi-temporal mode: "after" (Image 2) or "overlay" (Change Mask)
  const [compareTarget, setCompareTarget] = useState<"after" | "overlay">("after");

  const canCompare = !!originalArtifact && (!!secondaryArtifact || !!overlayArtifact);

  // Determine images for the slider
  // Left image: ALWAYS the Previous / Original Image
  const leftImageUrl = artifactUrl(originalArtifact?.url) || "";

  // Right image: Either After image or AI Overlay
  const rightArtifact =
    isBiTemporal && secondaryArtifact && compareTarget === "after"
      ? secondaryArtifact
      : overlayArtifact || secondaryArtifact || originalArtifact;
  const rightImageUrl = artifactUrl(rightArtifact?.url) || leftImageUrl;

  // Single layer image URL (when compareMode === "off")
  let currentSingleImageUrl = overlayArtifact?.url || originalArtifact?.url || imageArtifacts[0]?.url;
  if (activeLayer === "original" && originalArtifact) {
    currentSingleImageUrl = originalArtifact.url;
  } else if (activeLayer === "secondary" && secondaryArtifact) {
    currentSingleImageUrl = secondaryArtifact.url;
  } else if (activeLayer === "result" && overlayArtifact) {
    currentSingleImageUrl = overlayArtifact.url;
  } else {
    const customMatch = imageArtifacts.find((a) => a.name === activeLayer);
    if (customMatch) currentSingleImageUrl = customMatch.url;
  }
  const resolvedSingleImageUrl = artifactUrl(currentSingleImageUrl) || leftImageUrl;

  // Accurate labels for bi-temporal or baseline/overlay comparison
  const years = result.task_plan?.years || [];
  let leftLabel = "Previous";
  let rightLabel = "Current";

  if (isBiTemporal && (secondaryArtifact || years.length >= 2)) {
    if (years.length >= 2) {
      leftLabel = `Mar ${years[0]}`;
      rightLabel = compareTarget === "overlay" && overlayArtifact ? "Change Overlay" : `Mar ${years[1]}`;
    } else {
      leftLabel = "Previous (Before)";
      rightLabel = compareTarget === "overlay" && overlayArtifact ? "Change Overlay" : "Current (After)";
    }
  } else {
    leftLabel = "Original Imagery";
    rightLabel = overlayArtifact?.name?.includes("land")
      ? "Land Cover Overlay"
      : overlayArtifact?.name?.includes("building")
        ? "Building Footprints"
        : overlayArtifact?.name?.includes("water")
          ? "Water Overlay"
          : "AI Analysis Overlay";
  }

  // Detect analysis category for contextual color legend
  const hasLandCover = isLandResult && !!result.artifacts.find((a) => a.name.includes("land_cover") || a.name.includes("land_only"));
  const hasWater = !hasLandCover && isWaterResult;
  const hasBuildings = isBuildingResult;
  const hasChange =
    queryLower.includes("change") ||
    taskName === "bi_temporal_change" ||
    result.artifacts.some((a) => a.name.includes("change"));

  const landBreakdown = (result.statistics?.land_cover as
    | { breakdown?: Record<string, { pixels?: number }> }
    | undefined)?.breakdown;
  const showLandClass = (name: string) => !landBreakdown || (landBreakdown[name]?.pixels ?? 0) > 0;

  // Slider drag physics
  const handleSliderMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.round((x / rect.width) * 1000) / 10;
    setSliderPos(percent);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDraggingSlider(true);
    handleSliderMove(e.clientX);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      setIsDraggingSlider(true);
      handleSliderMove(e.touches[0].clientX);
    }
  };

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
      window.addEventListener("touchmove", handleGlobalTouchMove, { passive: true });
      window.addEventListener("mouseup", handleGlobalMouseUp);
      window.addEventListener("touchend", handleGlobalMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleGlobalMouseMove);
      window.removeEventListener("touchmove", handleGlobalTouchMove);
      window.removeEventListener("mouseup", handleGlobalMouseUp);
      window.removeEventListener("touchend", handleGlobalMouseUp);
    };
  }, [isDraggingSlider, handleSliderMove]);

  const geojsonArtifact = result.artifacts.find((a) => a.mime_type.includes("json") && a.name.endsWith(".geojson"));
  const reportArtifact = result.artifacts.find((a) => a.name.endsWith(".pdf"));

  return (
    <div className="visual-result-card">
      {/* Top Toolbar */}
      <div className="visual-card-toolbar">
        <div className="layer-switcher-group">
          {/* Comparison Mode Toggles */}
          {canCompare && isBiTemporal && secondaryArtifact && (
            <>
              <button
                type="button"
                className={`layer-switch-btn compare ${compareMode === "slider" && compareTarget === "after" ? "active" : ""}`}
                onClick={() => {
                  setCompareMode("slider");
                  setCompareTarget("after");
                }}
                title="Swipe compare: Previous Image vs Later Image"
              >
                <SlidersHorizontal size={13} />
                <span>Compare (Before / After)</span>
              </button>

              {overlayArtifact && (
                <button
                  type="button"
                  className={`layer-switch-btn compare ${compareMode === "slider" && compareTarget === "overlay" ? "active" : ""}`}
                  onClick={() => {
                    setCompareMode("slider");
                    setCompareTarget("overlay");
                  }}
                  title="Swipe compare: Previous Image vs Change Overlay"
                >
                  <SlidersHorizontal size={13} />
                  <span>Compare (Before / Overlay)</span>
                </button>
              )}
            </>
          )}

          {canCompare && (!isBiTemporal || !secondaryArtifact) && (
            <button
              type="button"
              className={`layer-switch-btn compare ${compareMode === "slider" ? "active" : ""}`}
              onClick={() => {
                setCompareMode((prev) => (prev === "slider" ? "off" : "slider"));
                if (compareMode === "off") setActiveLayer("result");
              }}
              title="Swipe compare: Original vs AI Analysis"
            >
              <SlidersHorizontal size={13} />
              <span>Compare (Slider)</span>
            </button>
          )}

          {/* Single Layer Views */}
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
              {isBiTemporal && secondaryArtifact ? "Before Image" : "Original"}
            </button>
          )}

          {secondaryArtifact && (
            <button
              type="button"
              className={`layer-switch-btn ${
                activeLayer === "secondary" && compareMode === "off" ? "active" : ""
              }`}
              onClick={() => {
                setActiveLayer("secondary");
                setCompareMode("off");
              }}
            >
              After Image
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

          {/* Dynamic Extra Layers Dropdown if multiple exist */}
          {imageArtifacts.length > 2 && (
            <div className="extra-layers-select-wrap">
              <select
                className="extra-layers-select"
                value={compareMode === "off" ? activeLayer : "compare"}
                onChange={(e) => {
                  if (e.target.value === "compare") {
                    setCompareMode("slider");
                  } else {
                    setActiveLayer(e.target.value);
                    setCompareMode("off");
                  }
                }}
              >
                <option value="compare">Layer options...</option>
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
            onClick={() => onOpenLightbox(compareMode === "slider" ? rightImageUrl : resolvedSingleImageUrl, result.query || "Satellite Imagery")}
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
                  href={resolvedSingleImageUrl}
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
            onOpenLightbox(resolvedSingleImageUrl, result.query || "Satellite Analysis");
          }
        }}
      >
        {compareMode === "slider" && canCompare ? (
          <div
            className="swipe-compare-viewport"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            role="slider"
            aria-label="Comparison slider"
            aria-valuenow={Math.round(sliderPos)}
            aria-valuemin={0}
            aria-valuemax={100}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft") setSliderPos((p) => Math.max(0, p - 2));
              else if (e.key === "ArrowRight") setSliderPos((p) => Math.min(100, p + 2));
              else if (e.key === "Home") setSliderPos(0);
              else if (e.key === "End") setSliderPos(100);
            }}
          >
            {/* Layer 1: Left / Previous Image (Base underneath, 100% width, defines aspect ratio) */}
            <img
              src={leftImageUrl}
              alt={leftLabel}
              className="compare-img base-layer"
              draggable={false}
            />

            {/* Layer 2: Right / Current or Overlay Image (Clipped from left using inset) */}
            <div
              className="compare-clipped-wrap"
              style={{
                clipPath: `inset(0 0 0 ${sliderPos}%)`,
                WebkitClipPath: `inset(0 0 0 ${sliderPos}%)`,
              }}
            >
              <img
                src={rightImageUrl}
                alt={rightLabel}
                className="compare-img top-layer"
                draggable={false}
              />
            </div>

            {/* Vertical Divider Line with Grab Handle */}
            <div
              className="swipe-divider-line"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="swipe-handle-knob" title="Drag left or right to compare">
                <ChevronsLeftRight size={15} strokeWidth={2.5} />
              </div>
            </div>

          </div>
        ) : (
          <div className="single-viewport">
            <img
              src={resolvedSingleImageUrl}
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

      {/* Contextual Visual Color Legend: Differentiating land, building, and other classes */}
      {hasLandCover && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Land Cover Palette:</span>
          <div className="visual-legend-items">
            {showLandClass("built_up") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#dc2626" }} />
                <span>Buildings / Built-up (Red)</span>
              </span>
            )}
            {showLandClass("bare_pervious") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#d97706" }} />
                <span>Land / Bare Soil (Amber)</span>
              </span>
            )}
            {showLandClass("road") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#facc15" }} />
                <span>Roads & Paved (Yellow)</span>
              </span>
            )}
            {showLandClass("vegetation") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#22c55e" }} />
                <span>Grass & Canopy (Green)</span>
              </span>
            )}
            {showLandClass("woodland") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#166534" }} />
                <span>Forest (Dark Green)</span>
              </span>
            )}
            {showLandClass("water") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#1d4ed8" }} />
                <span>Water (Deep Blue)</span>
              </span>
            )}
            {showLandClass("swimming_pool") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#06b6d4" }} />
                <span>Swimming Pool (Cyan)</span>
              </span>
            )}
            {showLandClass("agriculture") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#7cb342" }} />
                <span>Cropland (Olive)</span>
              </span>
            )}
            {showLandClass("unknown") && (
              <span className="visual-legend-item">
                <span className="legend-swatch" style={{ background: "#9ca3af" }} />
                <span>Unclassified (Gray)</span>
              </span>
            )}
          </div>
        </div>
      )}

      {hasWater && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Water Palette:</span>
          <div className="visual-legend-items">
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#1d4ed8" }} />
              <span>Water Surface (Deep Blue)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#0f2d78" }} />
              <span>Deep Water / Core (Navy)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#d97706" }} />
              <span>Land / Shoreline (Amber Tan)</span>
            </span>
          </div>
        </div>
      )}

      {hasBuildings && !hasLandCover && (
        <div className="visual-card-legend">
          <span className="visual-legend-title">Building Palette:</span>
          <div className="visual-legend-items">
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#dc2626" }} />
              <span>Building Footprints (Crimson Red)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#ffffff", border: "1px solid #999" }} />
              <span>Instance Boundaries (White)</span>
            </span>
            <span className="visual-legend-item">
              <span className="legend-swatch" style={{ background: "#d97706" }} />
              <span>Surrounding Ground (Amber)</span>
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
