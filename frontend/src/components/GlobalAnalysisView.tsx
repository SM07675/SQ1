import React, { useState, useMemo, useEffect, useRef } from "react";
import type { AnalysisResponse } from "../types";
import { ImageryViewport } from "./ImageryViewport";
import { EvidencePanel } from "./EvidencePanel";
import { VerdictPanel } from "./VerdictPanel";
import { MetricsBar } from "./MetricsBar";

interface ImageCardData {
  id: string;
  file: File;
  previewUrl: string;
  format: string;
  dimensions: string;
  modality: "Optical / Multispectral" | "SAR (Radar)" | "Standard Raster";
}

interface SelectedTool {
  step: string;
  name: string;
  purpose: string;
  category: "registration" | "detector" | "spectral" | "fusion" | "arbiter" | "calibration";
}

interface GlobalAnalysisViewProps {
  result: AnalysisResponse | null;
  busy: boolean;
  error: string | null;
  activeLayer: string;
  availableLayers: string[];
  layerNames: Record<string, string>;
  swipe: number;
  uniqueMetrics: [string, unknown][];
  onSelectLayer: (layer: string) => void;
  onSwipeChange: (swipe: number) => void;
  onExecuteGlobalAnalysis: (query: string, images: File[]) => void;
  onLoadHistoricalResult?: (resultId: string) => void;
}

const SUGGESTION_CHIPS = [
  "Describe this image.",
  "What land cover is visible in this scene?",
  "Highlight the largest water body.",
  "Where is vegetation concentrated?",
  "Identify built-up areas.",
  "How many buildings are visible?",
  "What changed between these two images?",
  "Has vegetation decreased?",
  "Has built-up area increased?",
  "Has the water body expanded?",
  "Use optical and SAR together to identify water.",
  "Use both sensors to identify urban areas.",
  "What information does SAR add?",
];

export const GlobalAnalysisView: React.FC<GlobalAnalysisViewProps> = ({
  result,
  busy,
  error,
  activeLayer,
  availableLayers,
  layerNames,
  swipe,
  uniqueMetrics,
  onSelectLayer,
  onSwipeChange,
  onExecuteGlobalAnalysis,
}) => {
  const [query, setQuery] = useState("");
  const [images, setImages] = useState<ImageCardData[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up object URLs on unmount
  useEffect(() => {
    return () => {
      images.forEach((img) => URL.revokeObjectURL(img.previewUrl));
    };
  }, [images]);

  // Helper to detect modality from filename and properties
  const detectModality = (file: File): ImageCardData["modality"] => {
    const name = file.name.toLowerCase();
    if (name.includes("sar") || name.includes("radar") || name.includes("grd") || name.includes("s1")) {
      return "SAR (Radar)";
    }
    if (name.includes("multi") || name.includes("ms") || name.includes("optical") || name.includes("s2") || name.includes("sentinel")) {
      return "Optical / Multispectral";
    }
    return "Optical / Multispectral";
  };

  const handleAddFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const newItems: ImageCardData[] = [];
    const maxAllowed = 2;
    const currentCount = images.length;

    for (let i = 0; i < fileList.length && currentCount + newItems.length < maxAllowed; i++) {
      const file = fileList[i];
      const previewUrl = URL.createObjectURL(file);
      const ext = file.name.split(".").pop()?.toUpperCase() || "TIF";
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);

      newItems.push({
        id: `${file.name}-${Date.now()}-${i}`,
        file,
        previewUrl,
        format: ext,
        dimensions: `${sizeMb} MB`,
        modality: detectModality(file),
      });
    }

    setImages((prev) => [...prev, ...newItems]);
  };

  const handleRemoveImage = (id: string) => {
    setImages((prev) => {
      const target = prev.find((img) => img.id === id);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((img) => img.id !== id);
    });
  };

  // 1. TWO-LEVEL INTENT CLASSIFIER (APPLICATION + SPECIFIC TASK)
  const classification = useMemo(() => {
    const q = query.toLowerCase().trim();
    if (!q) {
      return {
        app: "EMPTY",
        title: "Awaiting Query",
        task: "Awaiting Input",
        subTasks: [] as string[],
        multiIntent: false,
        category: null,
        target: null,
        reason: "Enter a natural-language geospatial request to initiate two-level intent routing.",
      };
    }

    // A. Unsupported / Off-Topic Check
    const unsupportedPatterns = ["capital of", "who is", "who was", "python code", "write a code", "how to cook", "weather in", "translate", "calculate"];
    if (unsupportedPatterns.some((p) => q.includes(p))) {
      return {
        app: "UNSUPPORTED",
        title: "Non-Geospatial Request",
        task: "Unsupported Domain",
        subTasks: ["unsupported"],
        multiIntent: false,
        category: "Off-Topic General Knowledge",
        target: null,
        reason: "Request does not pertain to remote sensing, satellite data, or Earth observation analysis.",
      };
    }

    // B. Unclear Check
    if (q.length < 3 || ["hello", "hi", "hey", "test", "run", "do it", "analyze", "check", "what"].includes(q)) {
      return {
        app: "UNCLEAR",
        title: "Ambiguous Query",
        task: "Clarification Needed",
        subTasks: ["unclear"],
        multiIntent: false,
        category: "Underspecified Input",
        target: null,
        reason: "Query does not specify analytical parameters or target features.",
      };
    }

    // C. Optical + SAR keywords
    const optSarWords = [
      "optical + sar", "optical and sar", "radar", "sar", "multispectral and radar",
      "both sensors", "cross-modal", "combine optical and sar", "radar evidence", "optical evidence",
      "what information does sar add", "surface characteristics"
    ];
    if (optSarWords.some((w) => q.includes(w))) {
      let specificTask = "Cross-Modal Comparative Analysis";
      let target = "general";
      if (q.includes("water") || q.includes("flood")) {
        specificTask = "Cross-Modal Water Analysis";
        target = "water";
      } else if (q.includes("built-up") || q.includes("urban") || q.includes("building") || q.includes("city")) {
        specificTask = "Cross-Modal Built-up Analysis";
        target = "built-up";
      }
      return {
        app: "OPTICAL_SAR",
        title: "Optical + SAR Cross-Modal",
        task: specificTask,
        subTasks: [specificTask],
        multiIntent: false,
        category: "Cross-Sensor Radar-Optical Fusion",
        target,
        reason: "Detected request for joint multi-modal optical and synthetic aperture radar verification.",
      };
    }

    // D. Bi-temporal Change keywords
    const changeWords = [
      "change", "changed", "before/after", "before and after", "increase", "decrease",
      "expansion", "reduction", "growth", "loss", "difference", "between dates",
      "between these", "between two", "over time", "temporal comparison",
      "urban expansion", "vegetation loss", "water expansion", "evolve", "compared"
    ];
    if (changeWords.some((w) => q.includes(w))) {
      let specificTask = "General Change Analysis";
      let target = "general";
      if (q.includes("built-up") || q.includes("urban") || q.includes("building") || q.includes("construction") || q.includes("development")) {
        specificTask = "Built-up Expansion Change";
        target = "built-up";
      } else if (q.includes("vegetation") || q.includes("forest") || q.includes("canopy") || q.includes("green") || q.includes("tree")) {
        specificTask = "Vegetation Loss & Canopy Dynamics";
        target = "vegetation";
      } else if (q.includes("water") || q.includes("flood") || q.includes("lake") || q.includes("river")) {
        specificTask = "Water Dynamic & Extent Change";
        target = "water";
      }
      return {
        app: "BI_TEMPORAL",
        title: "Bi-Temporal Change Analysis",
        task: specificTask,
        subTasks: [specificTask],
        multiIntent: false,
        category: "Time-Series Change Detection",
        target,
        reason: "Detected request for temporal comparison and land dynamics analysis over time.",
      };
    }

    // E. Single Image Intelligence (Multi-Task & Multi-Intent)
    const hasLandCover = ["land cover", "landcover", "land-cover", "terrain", "types of land", "types of terrain", "composition"].some((k) => q.includes(k));
    const hasWater = ["water", "lake", "river", "flood", "pond", "reservoir", "ocean"].some((k) => q.includes(k));
    const hasVeg = ["vegetation", "canopy", "forest", "crop", "greenery", "tree", "ndvi"].some((k) => q.includes(k));
    const hasBuilt = ["built-up", "built up", "building", "buildings", "urban", "city", "settlement", "construction"].some((k) => q.includes(k));
    const hasScene = ["describe", "overview", "what is visible", "summarize", "caption"].some((k) => q.includes(k));
    const hasCounting = ["how many", "count", "number of"].some((k) => q.includes(k));

    const subTasks: string[] = [];
    if (hasLandCover) subTasks.push("Land Cover Classification");
    if (hasWater) subTasks.push("Water Grounding / Analysis");
    if (hasVeg) subTasks.push("Vegetation Canopy Analysis");
    if (hasBuilt) subTasks.push("Built-up Footprint Analysis");
    if (hasScene && !hasLandCover) subTasks.push("Scene Description");

    const multiIntent = subTasks.length > 1;

    let primaryTask = "Scene Understanding";
    let target = "general";

    if (hasLandCover) {
      primaryTask = "Land Cover Understanding";
      target = "land_cover";
    } else if (hasWater && (q.includes("highlight") || q.includes("largest") || q.includes("locate") || q.includes("find") || q.includes("water"))) {
      primaryTask = "Water Body Grounding & Delineation";
      target = "water";
    } else if (hasVeg && (q.includes("concentrated") || q.includes("where is") || q.includes("canopy") || q.includes("vegetation"))) {
      primaryTask = "Vegetation Canopy Analysis";
      target = "vegetation";
    } else if (hasBuilt && (q.includes("built-up") || q.includes("urban") || q.includes("buildings") || q.includes("where are"))) {
      primaryTask = "Built-up & Urban Footprint";
      target = "built-up";
    } else if (hasCounting || q.includes("how many")) {
      primaryTask = "Visual Question Answering (Counting & Verification)";
      target = "vqa";
    } else if (hasScene) {
      primaryTask = "Scene Description & Overview";
      target = "scene_description";
    } else if (["highlight", "locate", "where is", "where are", "find the", "identify objects"].some((k) => q.includes(k))) {
      primaryTask = "Object Identification & Grounding";
      target = "grounding";
    }

    return {
      app: "SINGLE_IMAGE",
      title: "Single Image Intelligence",
      task: multiIntent ? `Multi-Intent (${subTasks.join(" + ")})` : primaryTask,
      subTasks: subTasks.length > 0 ? subTasks : [primaryTask],
      multiIntent,
      category: "Scene Understanding & Grounding",
      target,
      reason: "Detected single-observation inquiry for visual grounding, land cover identification, or VQA.",
    };
  }, [query]);

  // 2. INPUT VALIDATION LOGIC
  const validation = useMemo(() => {
    if (classification.app === "EMPTY") {
      return { status: "IDLE", message: "Enter a query and upload image(s).", allowExecute: false };
    }

    if (classification.app === "UNSUPPORTED" || classification.app === "UNCLEAR") {
      return {
        status: "VALID",
        message: "Query classified. Click analyze to receive system domain guidance.",
        actionText: "SatQuery will provide guidance on remote-sensing scope.",
        allowExecute: true,
      };
    }

    const count = images.length;
    const hasSar = images.some((img) => img.modality === "SAR (Radar)" || img.file.name.toLowerCase().includes("sar"));

    if (classification.app === "SINGLE_IMAGE") {
      if (count === 0) {
        return {
          status: "INCOMPLETE",
          message: "Single Image Intelligence requires 1 satellite image.",
          actionText: "Please upload an optical or multispectral image to proceed.",
          promptUploadFirst: true,
          allowExecute: false,
        };
      }
      if (count === 1) {
        return {
          status: "VALID",
          message: "Input requirements satisfied: 1 valid satellite image detected.",
          actionText: "Ready for automated model execution.",
          allowExecute: true,
        };
      }
      return {
        status: "VALID",
        message: "2 images provided; primary image will be analyzed under Single Image Intelligence.",
        actionText: "Ready for execution.",
        allowExecute: true,
      };
    }

    if (classification.app === "BI_TEMPORAL") {
      if (count === 0) {
        return {
          status: "INCOMPLETE",
          message: "Two images are required for bi-temporal change analysis.",
          actionText: "Please upload a BEFORE image (T1) and an AFTER image (T2) to continue.",
          promptUploadFirst: true,
          allowExecute: false,
        };
      }
      if (count === 1) {
        return {
          status: "INCOMPLETE",
          message: "Two images are required for bi-temporal change analysis.",
          actionText: "Please upload an AFTER image (T2) to continue without producing unverified results.",
          promptUploadSecond: true,
          allowExecute: false,
        };
      }
      return {
        status: "VALID",
        message: "2 spatially corresponding temporal images detected (T1 Before & T2 After).",
        actionText: "Ready for automated registration and change detection.",
        allowExecute: true,
      };
    }

    if (classification.app === "OPTICAL_SAR") {
      if (count === 0) {
        return {
          status: "INCOMPLETE",
          message: "Optical + SAR cross-modal analysis requires two complementary sensor images.",
          actionText: "Please upload both an optical/multispectral image and a SAR image.",
          promptUploadFirst: true,
          allowExecute: false,
        };
      }
      if (count === 1) {
        return {
          status: "INCOMPLETE",
          message: "Optical + SAR cross-modal analysis requires two complementary sensor images.",
          actionText: hasSar
            ? "Please upload the complementary Optical/Multispectral image."
            : "Please upload the complementary SAR (Radar) image.",
          promptUploadSecond: true,
          allowExecute: false,
        };
      }
      if (!hasSar) {
        return {
          status: "INVALID",
          message: "Two images were provided, but Optical + SAR requires one optical/multispectral image and one SAR (Radar) image.",
          actionText: "Please replace one image with a SAR (Radar) GeoTIFF.",
          allowExecute: false,
        };
      }
      return {
        status: "VALID",
        message: "Cross-modal pair verified: Optical surface reflectance + SAR backscatter detected.",
        actionText: "Ready for CROMA radar-optical fusion.",
        allowExecute: true,
      };
    }

    return { status: "IDLE", message: "", allowExecute: false };
  }, [classification, images]);

  // 3. DYNAMIC SELECTED ANALYSIS TOOLS
  const selectedTools: SelectedTool[] = useMemo(() => {
    if (classification.app === "EMPTY" || classification.app === "UNSUPPORTED" || classification.app === "UNCLEAR") return [];

    if (classification.app === "BI_TEMPORAL") {
      const tools: SelectedTool[] = [
        {
          step: "01",
          name: "Phase Correlation + ECC Registration",
          purpose: "Sub-pixel spatial co-registration & alignment across temporal epochs",
          category: "registration",
        },
        {
          step: "02",
          name: "SSIM & Structural Change Detector",
          purpose: "Pixel-level radiometric difference and structural anomaly filtering",
          category: "detector",
        },
      ];

      if (classification.target === "built-up") {
        tools.push({
          step: "03",
          name: "NDBI Spectral Engine",
          purpose: "Normalized Difference Built-up Index delta mapping",
          category: "spectral",
        });
      } else if (classification.target === "vegetation") {
        tools.push({
          step: "03",
          name: "NDVI Spectral Canopy Engine",
          purpose: "Normalized Difference Vegetation Index delta & canopy loss quantification",
          category: "spectral",
        });
      } else if (classification.target === "water") {
        tools.push({
          step: "03",
          name: "NDWI Spectral Engine",
          purpose: "Surface water boundary expansion/recession mapping",
          category: "spectral",
        });
      } else {
        tools.push({
          step: "03",
          name: "TinyCD / OpenCD Siamese Witness",
          purpose: "Deep learning learned structural change verification",
          category: "detector",
        });
      }

      tools.push(
        {
          step: "04",
          name: "GeoProof Multi-Witness Arbiter",
          purpose: "Cross-validate radiometric change with polygon geometry and affine transform",
          category: "arbiter",
        },
        {
          step: "05",
          name: "Platt Scaling Calibrator",
          purpose: "Temperature-scaled uncertainty estimation & confidence interval bounding",
          category: "calibration",
        }
      );
      return tools;
    }

    if (classification.app === "OPTICAL_SAR") {
      return [
        {
          step: "01",
          name: "CROMA Radar-Optical Transformer",
          purpose: "Deep cross-modal representation alignment between Sentinel-1 & Sentinel-2",
          category: "fusion",
        },
        {
          step: "02",
          name: "Optical Spectral Evidence Engine",
          purpose: "NDWI water reflectance & multispectral cloud-penetrating verification",
          category: "spectral",
        },
        {
          step: "03",
          name: "SAR Low-Backscatter Specular Detector",
          purpose: "Radar specular backscatter signature mapping (dB thresholding)",
          category: "detector",
        },
        {
          step: "04",
          name: "GeoProof Multi-Witness Spatial Arbiter",
          purpose: "Inter-sensor agreement matrix, IoU computation and spatial arbitration",
          category: "arbiter",
        },
        {
          step: "05",
          name: "Platt Scaling Calibrator",
          purpose: "Calibrate cross-modal confidence against sensor agreement score",
          category: "calibration",
        },
      ];
    }

    // SINGLE IMAGE
    const tools: SelectedTool[] = [
      {
        step: "01",
        name: "Raster & CRS Validator",
        purpose: "Validate metadata, affine transformation matrix, resolution and NoData bands",
        category: "registration",
      },
    ];

    if (classification.task.includes("Land Cover")) {
      tools.push(
        {
          step: "02",
          name: "Multispectral Land Cover Engine",
          purpose: "Multi-class surface classification (Water, Vegetation, Forest, Agriculture, Built-up, Bare Soil)",
          category: "spectral",
        },
        {
          step: "03",
          name: "RemoteCLIP Zero-Shot Extractor",
          purpose: "Geospatial semantic tile retrieval and land cover classification",
          category: "detector",
        }
      );
    } else if (classification.target === "water") {
      tools.push(
        {
          step: "02",
          name: "Optical Water Grounding Engine v2",
          purpose: "Dominant water body segmentation & morphological polygon bounding",
          category: "detector",
        },
        {
          step: "03",
          name: "NDWI Spectral Engine",
          purpose: "Spectral verification using Green & NIR band differential",
          category: "spectral",
        }
      );
    } else if (classification.target === "vegetation") {
      tools.push(
        {
          step: "02",
          name: "NDVI Canopy Grounding Engine",
          purpose: "Vegetation density and canopy cover segmentation",
          category: "spectral",
        },
        {
          step: "03",
          name: "EarthDial-4B-MS",
          purpose: "Multispectral vision-language scene comprehension",
          category: "fusion",
        }
      );
    } else if (classification.target === "built-up") {
      tools.push(
        {
          step: "02",
          name: "NDBI Spectral Engine",
          purpose: "Built-up index and urban structure extraction",
          category: "spectral",
        },
        {
          step: "03",
          name: "RemoteCLIP Tile Retriever",
          purpose: "Urban footprint semantic clustering",
          category: "detector",
        }
      );
    } else {
      tools.push(
        {
          step: "02",
          name: "EarthDial-4B-RGB",
          purpose: "Zero-shot visual question answering & scene description",
          category: "fusion",
        },
        {
          step: "03",
          name: "RemoteCLIP Zero-Shot Extractor",
          purpose: "Geospatial semantic tile retrieval and land cover classification",
          category: "detector",
        }
      );
    }

    tools.push({
      step: "04",
      name: "GeoProof Evidence Arbiter",
      purpose: "Synthesize polygon geometries, calculate geographic area, and enforce guardrails",
      category: "arbiter",
    });

    return tools;
  }, [classification]);

  // Extract Land Cover Breakdown from evidence if available
  const landCoverEvidence = useMemo(() => {
    if (!result) return null;
    const lcItem = result.evidence.find(
      (e) => e.kind === "land_cover_classification" || e.metrics?.breakdown
    );
    if (!lcItem || !lcItem.metrics?.breakdown) return null;
    return lcItem.metrics.breakdown as Record<
      string,
      { percent: number; status: string; color: string }
    >;
  }, [result]);

  const handleStartAnalysis = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validation.allowExecute || (images.length === 0 && classification.app !== "UNSUPPORTED" && classification.app !== "UNCLEAR")) return;
    onExecuteGlobalAnalysis(
      query,
      images.map((img) => img.file)
    );
  };

  return (
    <div className="global-analysis-container">
      {/* Top Banner & Header */}
      <div className="global-header-panel">
        <div className="global-title-row">
          <div className="global-badge">TWO-LEVEL INTENT SYSTEM</div>
          <h2>GLOBAL ANALYSIS</h2>
          <p className="global-subtitle">
            Describe your remote-sensing task in natural language. SatQuery autonomously identifies the high-level application, decomposes fine-grained analytical tasks, validates inputs, and executes specialized model pipelines.
          </p>
        </div>

        {/* Suggestion Chips */}
        <div className="suggestion-chips-row">
          <span className="chips-label">SUGGESTED QUERIES:</span>
          <div className="chips-list">
            {SUGGESTION_CHIPS.map((chip, idx) => (
              <button
                key={idx}
                type="button"
                className="query-chip"
                onClick={() => setQuery(chip)}
                title="Click to populate query"
              >
                {chip}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Analysis Input & Routing Grid */}
      <div className="global-input-grid">
        {/* Left Column: Natural Language Query & Upload Area */}
        <div className="global-query-col">
          <div className="panel query-box-panel">
            <div className="panel-header">
              <span className="panel-step-badge">INPUT 01</span>
              <h3>NATURAL LANGUAGE QUERY</h3>
            </div>

            <textarea
              className="global-query-textarea"
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Example: What land cover is visible in this scene? / Has built-up area increased between these two images?"
            />

            {/* Flexible Multi-Image Upload Area */}
            <div className="images-upload-section">
              <div className="images-upload-header">
                <span className="panel-step-badge">INPUT 02</span>
                <h4>IMAGE INPUTS ({images.length}/2)</h4>
              </div>

              {/* Upload Drop Zone / Add Image Button */}
              {images.length < 2 && (
                <div
                  className={`global-dropzone ${dragOver ? "drag-active" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    handleAddFiles(e.dataTransfer.files);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    multiple
                    style={{ display: "none" }}
                    accept=".tif,.tiff,.png,.jpg,.jpeg,.nc"
                    onChange={(e) => handleAddFiles(e.target.files)}
                  />
                  <div className="dropzone-content">
                    <span className="dropzone-icon">+</span>
                    <strong>Add Image (GeoTIFF / PNG / SAR)</strong>
                    <small>Click to browse or drop raster files here (Max 2 images)</small>
                  </div>
                </div>
              )}

              {/* Uploaded Images Card Deck */}
              {images.length > 0 && (
                <div className="image-cards-deck">
                  {images.map((img, idx) => (
                    <div key={img.id} className="image-card">
                      <div className="image-card-thumb">
                        <img src={img.previewUrl} alt={img.file.name} />
                        <span className="slot-badge">IMAGE {idx + 1}</span>
                      </div>
                      <div className="image-card-meta">
                        <strong className="image-card-name" title={img.file.name}>
                          {img.file.name}
                        </strong>
                        <div className="image-card-tags">
                          <span className="meta-tag format">{img.format}</span>
                          <span className="meta-tag size">{img.dimensions}</span>
                          <span className={`meta-tag modality ${img.modality.includes("SAR") ? "sar" : "optical"}`}>
                            {img.modality}
                          </span>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="image-card-remove"
                        onClick={() => handleRemoveImage(img.id)}
                        title="Remove this image"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Incomplete / Guidance Action Banners */}
            {validation.status === "INCOMPLETE" && (
              <div className="validation-banner incomplete">
                <div className="banner-icon">⚠</div>
                <div className="banner-body">
                  <strong>{validation.message}</strong>
                  <p>{validation.actionText}</p>
                </div>
                {validation.promptUploadSecond && images.length === 1 && (
                  <button
                    type="button"
                    className="banner-action-btn"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    + Upload Second Image
                  </button>
                )}
                {validation.promptUploadFirst && images.length === 0 && (
                  <button
                    type="button"
                    className="banner-action-btn"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    + Upload Image
                  </button>
                )}
              </div>
            )}

            {validation.status === "INVALID" && (
              <div className="validation-banner invalid">
                <div className="banner-icon">✕</div>
                <div className="banner-body">
                  <strong>{validation.message}</strong>
                  <p>{validation.actionText}</p>
                </div>
              </div>
            )}

            {/* Analyze Request Button */}
            <button
              type="button"
              className="global-analyze-btn"
              disabled={busy || !validation.allowExecute}
              onClick={handleStartAnalysis}
            >
              {busy ? (
                <>
                  <span className="btn-spinner" />
                  <span>EXECUTING QUERY-DRIVEN PIPELINE...</span>
                </>
              ) : (
                <>
                  <span className="btn-icon">⚡</span>
                  <span>ANALYZE REQUEST</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: 8-Stage Observable Execution Timeline & Tool Selection */}
        <div className="global-pipeline-col">
          {/* Observable Execution Stages Card */}
          <div className="panel pipeline-panel">
            <div className="panel-header">
              <span className="panel-step-badge">8-STAGE PIPELINE</span>
              <h3>OBSERVABLE EXECUTION TIMELINE</h3>
            </div>

            <div className="execution-stages-list">
              {/* Stage 01: Query Understanding */}
              <div className={`stage-row ${query.trim() ? "active" : "pending"}`}>
                <div className="stage-num">01</div>
                <div className="stage-info">
                  <div className="stage-title">QUERY UNDERSTANDING</div>
                  <div className="stage-desc">
                    {query.trim() ? `"${query}" parsed` : "Awaiting natural-language input"}
                  </div>
                </div>
                <span className={`stage-status ${query.trim() ? "ok" : ""}`}>
                  {query.trim() ? "✓ READY" : "WAITING"}
                </span>
              </div>

              {/* Stage 02: Application Detection */}
              <div className={`stage-row ${classification.app !== "EMPTY" ? "active" : "pending"}`}>
                <div className="stage-num">02</div>
                <div className="stage-info">
                  <div className="stage-title">APPLICATION DETECTION</div>
                  <div className="stage-desc">
                    <strong>{classification.title}</strong>
                    {classification.category && <span className="cat-pill">{classification.category}</span>}
                  </div>
                </div>
                <span className={`stage-status ${classification.app !== "EMPTY" ? "ok" : ""}`}>
                  {classification.app !== "EMPTY" ? "✓ DETECTED" : "WAITING"}
                </span>
              </div>

              {/* Stage 03: Task Detection */}
              <div className={`stage-row ${classification.app !== "EMPTY" ? "active" : "pending"}`}>
                <div className="stage-num">03</div>
                <div className="stage-info">
                  <div className="stage-title">TASK DETECTION</div>
                  <div className="stage-desc">
                    <strong>{classification.task}</strong>
                    {classification.multiIntent && (
                      <span className="cat-pill multi">MULTI-INTENT ({classification.subTasks.length})</span>
                    )}
                  </div>
                </div>
                <span className={`stage-status ${classification.app !== "EMPTY" ? "ok" : ""}`}>
                  {classification.app !== "EMPTY" ? "✓ CLASSIFIED" : "WAITING"}
                </span>
              </div>

              {/* Stage 04: Input Validation */}
              <div className={`stage-row ${validation.status === "VALID" ? "active" : validation.status === "INCOMPLETE" ? "warning" : "pending"}`}>
                <div className="stage-num">04</div>
                <div className="stage-info">
                  <div className="stage-title">INPUT VALIDATION</div>
                  <div className="stage-desc">
                    {validation.status === "VALID" && `✓ Requirements satisfied: ${images.length} image(s) verified`}
                    {validation.status === "INCOMPLETE" && `⚠ ${validation.message}`}
                    {validation.status === "INVALID" && `✕ ${validation.message}`}
                    {validation.status === "IDLE" && "Awaiting input validation"}
                  </div>
                </div>
                <span className={`stage-status ${validation.status === "VALID" ? "ok" : validation.status === "INCOMPLETE" ? "warn" : ""}`}>
                  {validation.status === "VALID" ? "✓ VALID" : validation.status === "INCOMPLETE" ? "⚠ REQUIRED" : "WAITING"}
                </span>
              </div>

              {/* Stage 05: Model Selection */}
              <div className={`stage-row ${selectedTools.length > 0 && validation.status === "VALID" ? "active" : "pending"}`}>
                <div className="stage-num">05</div>
                <div className="stage-info">
                  <div className="stage-title">MODEL / METHOD SELECTION</div>
                  <div className="stage-desc">
                    {selectedTools.length > 0
                      ? `${selectedTools.length} specialized analytical engines coordinated`
                      : "Pending input validation"}
                  </div>
                </div>
                <span className={`stage-status ${selectedTools.length > 0 && validation.status === "VALID" ? "ok" : ""}`}>
                  {selectedTools.length > 0 && validation.status === "VALID" ? "✓ SELECTED" : "WAITING"}
                </span>
              </div>

              {/* Stage 06: Output Generation */}
              <div className={`stage-row ${result ? "active" : "pending"}`}>
                <div className="stage-num">06</div>
                <div className="stage-info">
                  <div className="stage-title">OUTPUT GENERATION</div>
                  <div className="stage-desc">
                    {result ? "Raster layers & spatial masks materialized" : "Awaiting execution"}
                  </div>
                </div>
                <span className={`stage-status ${result ? "ok" : ""}`}>
                  {result ? "✓ COMPLETED" : "WAITING"}
                </span>
              </div>

              {/* Stage 07: Evidence Compilation */}
              <div className={`stage-row ${result ? "active" : "pending"}`}>
                <div className="stage-num">07</div>
                <div className="stage-info">
                  <div className="stage-title">EVIDENCE COMPILATION</div>
                  <div className="stage-desc">
                    {result ? `${result.evidence.length} evidence items compiled with GeoJSON bounds` : "Awaiting execution"}
                  </div>
                </div>
                <span className={`stage-status ${result ? "ok" : ""}`}>
                  {result ? "✓ COMPILED" : "WAITING"}
                </span>
              </div>

              {/* Stage 08: Calibrated Confidence */}
              <div className={`stage-row ${result ? "active" : "pending"}`}>
                <div className="stage-num">08</div>
                <div className="stage-info">
                  <div className="stage-title">CALIBRATED CONFIDENCE</div>
                  <div className="stage-desc">
                    {result
                      ? `Score: ${(result.verdict.confidence * 100).toFixed(1)}% (${result.verdict.status.toUpperCase()})`
                      : "Awaiting execution"}
                  </div>
                </div>
                <span className={`stage-status ${result ? "ok" : ""}`}>
                  {result ? "✓ CALIBRATED" : "WAITING"}
                </span>
              </div>
            </div>
          </div>

          {/* Dynamic Tool Selection Card */}
          <div className="panel tools-selection-panel">
            <div className="panel-header">
              <span className="panel-step-badge">ORCHESTRATION</span>
              <h3>SELECTED ANALYSIS TOOLS</h3>
            </div>

            {selectedTools.length === 0 ? (
              <div className="tools-empty-state">
                <p>Type a natural-language query to see dynamically coordinated models and spectral engines.</p>
              </div>
            ) : (
              <div className="selected-tools-grid">
                {selectedTools.map((tool) => (
                  <div key={tool.name} className={`tool-item-card ${tool.category}`}>
                    <div className="tool-card-head">
                      <span className="tool-step-idx">{tool.step}</span>
                      <strong className="tool-name">{tool.name}</strong>
                    </div>
                    <p className="tool-purpose">
                      <strong>Purpose:</strong> {tool.purpose}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Error readout */}
      {error && (
        <div className="global-error-banner" role="alert">
          <span className="error-icon">⚠</span>
          <span className="error-text">{error}</span>
        </div>
      )}

      {/* Global Analysis Results Workspace */}
      {result && (
        <div className="global-results-section">
          {/* Top KPI Metrics Bar */}
          <MetricsBar result={result} />

          {/* Two-Level Intent Result Header Card */}
          <div className="panel global-result-summary-card">
            <div className="summary-header-row">
              <div>
                <div className="intent-tags-row">
                  <span className="summary-app-tag">
                    APPLICATION: {classification.title.toUpperCase()}
                  </span>
                  <span className="summary-task-tag">
                    TASK: {classification.task.toUpperCase()}
                  </span>
                  {classification.multiIntent && (
                    <span className="summary-multi-tag">MULTI-INTENT</span>
                  )}
                </div>
                <h3 className="summary-query-title">"{result.query}"</h3>
              </div>
              <div className="summary-verdict-tag">
                VERDICT: <strong className={result.verdict.status}>{result.verdict.status.toUpperCase()}</strong>
                <span className="confidence-pill">
                  {Math.round(result.verdict.confidence * 100)}% CONFIDENCE
                </span>
              </div>
            </div>
            <div className="summary-answer-box">
              <p>{result.verdict.answer}</p>
            </div>
          </div>

          {/* TASK-ADAPTIVE CARD: Land Cover Breakdown (if present) */}
          {landCoverEvidence && (
            <div className="panel land-cover-breakdown-card">
              <div className="panel-header">
                <span className="panel-step-badge">LAND COVER</span>
                <h3>SURFACE COMPOSITION & CLASS DISTRIBUTION</h3>
              </div>
              <div className="land-cover-grid">
                {Object.entries(landCoverEvidence).map(([clsKey, clsData]) => (
                  <div key={clsKey} className="lc-class-card" style={{ borderLeftColor: clsData.color }}>
                    <div className="lc-class-head">
                      <strong className="lc-class-name">{clsKey.replace("_", " ").toUpperCase()}</strong>
                      <span className={`lc-status-pill ${clsData.status.toLowerCase()}`}>
                        {clsData.status}
                      </span>
                    </div>
                    <div className="lc-percent-val">{clsData.percent}%</div>
                    <div className="lc-progress-bar">
                      <div
                        className="lc-progress-fill"
                        style={{ width: `${Math.min(100, clsData.percent)}%`, backgroundColor: clsData.color }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Visual Workspace & Telemetry Grid */}
          <div className="analysis-content-grid">
            {/* Center Viewport */}
            <div className="center-workspace-column" style={{ gridColumn: "span 2" }}>
              <ImageryViewport
                result={result}
                activeLayer={activeLayer}
                availableLayers={availableLayers}
                layerNames={layerNames}
                swipe={swipe}
                busy={busy}
                onSelectLayer={onSelectLayer}
                onSwipeChange={onSwipeChange}
              />

              <EvidencePanel
                evidence={result.evidence}
                metrics={uniqueMetrics}
                contradictions={result.verdict.contradictions}
              />
            </div>

            {/* Right Column: Verdict & Trace */}
            <VerdictPanel result={result} />
          </div>
        </div>
      )}
    </div>
  );
};
