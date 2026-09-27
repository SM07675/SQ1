import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Layers,
  Droplets,
  Building2,
  GitCompareArrows,
  TrendingUp,
  ArrowUp,
  ImagePlus,
} from "lucide-react";
import type { ChatSummary } from "../../types";
import { AttachmentTray, type AttachedFileItem } from "../composer/AttachmentTray";

interface HomeZeroStateProps {
  onSendMessage: (query: string, pairType: string, files: File[]) => Promise<void>;
  busy: boolean;
  recentChats: ChatSummary[];
  onSelectChat: (chatId: string) => void;
}

export const HomeZeroState: React.FC<HomeZeroStateProps> = ({
  onSendMessage,
  busy,
}) => {
  const [query, setQuery] = useState("");
  const [attachedFiles, setAttachedFiles] = useState<AttachedFileItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const newHeight = Math.max(26, Math.min(el.scrollHeight, 160));
    el.style.height = `${newHeight}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [query, adjustHeight]);

  const processFiles = (fileList: File[] | FileList) => {
    const incomingFiles = Array.from(fileList);
    if (incomingFiles.length === 0) return;

    setAttachedFiles((prev) => {
      const combinedFiles = [...prev.map((p) => p.file), ...incomingFiles].slice(0, 2);
      return combinedFiles.map((file, idx) => {
        const isGeotiff =
          file.name.toLowerCase().endsWith(".tif") || file.name.toLowerCase().endsWith(".tiff");
        const existing = prev.find((p) => p.file.name === file.name && p.file.size === file.size);
        let previewUrl = existing?.previewUrl;
        if (!previewUrl && file.type.startsWith("image/") && !isGeotiff) {
          previewUrl = URL.createObjectURL(file);
        }
        return {
          file,
          isGeotiff,
          previewUrl,
          role: combinedFiles.length === 2 ? (idx === 0 ? "earlier" : "later") : "single",
        };
      });
    });
  };

  const loadDemoScene = async (type: "single" | "bitemporal" = "single") => {
    try {
      if (type === "bitemporal") {
        const resA = await fetch("/demo_before.tif");
        const blobA = await resA.blob();
        const fileA = new File([blobA], "demo_before.tif", { type: "image/tiff" });

        const resB = await fetch("/demo_after.tif");
        const blobB = await resB.blob();
        const fileB = new File([blobB], "demo_after.tif", { type: "image/tiff" });

        setAttachedFiles([
          { file: fileA, isGeotiff: true, role: "earlier" },
          { file: fileB, isGeotiff: true, role: "later" },
        ]);
        setQuery("Compare these two satellite images and find differences.");
      } else {
        const res = await fetch("/sample_satellite.png");
        const blob = await res.blob();
        const file = new File([blob], "coastal_urban_sample.png", { type: "image/png" });
        const previewUrl = URL.createObjectURL(file);
        setAttachedFiles([{ file, isGeotiff: false, previewUrl, role: "single" }]);
        if (!query.trim()) {
          setQuery("Find all water bodies and calculate coverage");
        }
      }
      setTimeout(() => textareaRef.current?.focus(), 50);
    } catch (err) {
      console.error("Failed to load demo scene:", err);
    }
  };

  const handleSwapPair = () => {
    setAttachedFiles((prev) => {
      if (prev.length !== 2) return prev;
      return [
        { ...prev[1], role: "earlier" },
        { ...prev[0], role: "later" },
      ];
    });
  };

  const handleRemoveFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    const trimmed = query.trim();
    if (!trimmed && attachedFiles.length === 0) return;

    let pairType = "auto";
    if (attachedFiles.length === 2) {
      const isOpticalSar =
        trimmed.toLowerCase().includes("sar") ||
        trimmed.toLowerCase().includes("fusion") ||
        attachedFiles.some((f) => f.file.name.toLowerCase().includes("sar"));
      pairType = isOpticalSar ? "optical_sar" : "bi_temporal";
    } else if (attachedFiles.length === 1) {
      pairType = "single";
    }

    const defaultText =
      attachedFiles.length === 2
        ? "Compare these two satellite images and find differences."
        : "Analyze this satellite imagery.";
    const text = trimmed || defaultText;
    const rawFiles = attachedFiles.map((a) => a.file);
    setQuery("");
    setAttachedFiles([]);
    await onSendMessage(text, pairType, rawFiles);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
      e.target.value = "";
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length) {
      processFiles(e.dataTransfer.files);
    }
  };

  const quickActions = [
    {
      label: "Find land",
      icon: <Layers size={14} strokeWidth={2} />,
      iconClass: "chip-icon-land",
      text: "Classify dominant land cover types",
      type: "single" as const,
    },
    {
      label: "Detect water",
      icon: <Droplets size={14} strokeWidth={2} />,
      iconClass: "chip-icon-water",
      text: "Find all water bodies and calculate coverage",
      type: "single" as const,
    },
    {
      label: "Count buildings",
      icon: <Building2 size={14} strokeWidth={2} />,
      iconClass: "chip-icon-building",
      text: "Count all visible building footprints",
      type: "single" as const,
    },
    {
      label: "Compare images",
      icon: <GitCompareArrows size={14} strokeWidth={2} />,
      iconClass: "chip-icon-compare",
      text: "Compare these images to detect changes",
      type: "bitemporal" as const,
    },
    {
      label: "Analyze change",
      icon: <TrendingUp size={14} strokeWidth={2} />,
      iconClass: "chip-icon-change",
      text: "Detect and quantify surface changes over time",
      type: "bitemporal" as const,
    },
  ];

  return (
    <div className="home-zero-state">
      <div className="home-hero-content">
        {/* Hero headline */}
        <h1 className="hero-headline">
          Good to see <span className="hero-gradient-text">you</span>
        </h1>

        {/* Main composer */}
        <div className="hero-composer-wrapper">
          <div className="composer-container">
            <div
              className={`floating-composer ${isDragging ? "dragging" : ""} ${busy ? "busy" : ""} ${
                attachedFiles.length > 0 ? "has-attachments" : ""
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
              onDrop={handleDrop}
            >
              {/* Attached files indicator */}
              {attachedFiles.length > 0 && (
                <AttachmentTray
                  files={attachedFiles}
                  onRemove={handleRemoveFile}
                  onSwap={attachedFiles.length === 2 ? handleSwapPair : undefined}
                />
              )}

              <div className="composer-input-row">
                <input
                  type="file"
                  ref={fileInputRef}
                  multiple
                  style={{ display: "none" }}
                  accept=".tif,.tiff,.png,.jpg,.jpeg,.nc"
                  onChange={handleFileChange}
                />

                <button
                  type="button"
                  className="composer-action-btn attach-btn"
                  onClick={() => fileInputRef.current?.click()}
                  title="Attach satellite imagery (GeoTIFF, PNG, JPEG)"
                  aria-label="Attach imagery"
                >
                  <ImagePlus size={19} strokeWidth={1.8} />
                </button>


                <textarea
                  ref={textareaRef}
                  className="composer-textarea"
                  placeholder="Ask anything about satellite imagery (or select a quick prompt below)..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={busy}
                  aria-label="Satellite query input"
                />

                <button
                  type="button"
                  className={`composer-send-btn ${
                    (query.trim() || attachedFiles.length > 0) && !busy ? "ready" : "disabled"
                  }`}
                  onClick={handleSubmit}
                  disabled={busy || (!query.trim() && attachedFiles.length === 0)}
                  title="Send query (Enter)"
                  aria-label="Send query"
                >
                  <ArrowUp size={18} strokeWidth={2.5} />
                </button>
              </div>
            </div>

            {/* Quick action chips */}
            <div className="composer-suggestions-row">
              {quickActions.map((action) => (
                <button
                  key={action.label}
                  type="button"
                  className="suggestion-chip-btn"
                  onClick={() => {
                    setQuery(action.text);
                    if (attachedFiles.length === 0) {
                      loadDemoScene(action.type);
                    }
                    setTimeout(() => {
                      adjustHeight();
                      textareaRef.current?.focus();
                    }, 60);
                  }}
                  title={`Click to load prompt & scene: ${action.text}`}
                >
                  <span className={action.iconClass}>{action.icon}</span>
                  <span>{action.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
