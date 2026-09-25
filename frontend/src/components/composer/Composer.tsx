import React, { useState, useRef, useEffect, useCallback } from "react";
import { Plus, ArrowUp, Paperclip, X, Image as ImageIcon, Sparkles, CheckCircle2 } from "lucide-react";
import { AttachmentTray, type AttachedFileItem } from "./AttachmentTray";

interface ComposerProps {
  onSendMessage: (query: string, pairType: string, files: File[]) => Promise<void>;
  busy: boolean;
  activeContextImageName?: string | null;
  onClearContext?: () => void;
  showSuggestions?: boolean;
  onSelectSuggestion?: (text: string) => void;
  placeholder?: string;
  initialQuery?: string;
  initialFiles?: File[];
}

export const Composer: React.FC<ComposerProps> = ({
  onSendMessage,
  busy,
  activeContextImageName,
  onClearContext,
  showSuggestions = false,
  onSelectSuggestion,
  placeholder: customPlaceholder,
  initialQuery = "",
  initialFiles = [],
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFileItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync initial query if changed
  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery);
      setTimeout(() => adjustHeight(), 10);
    }
  }, [initialQuery]);

  // Sync initial files if changed
  useEffect(() => {
    if (initialFiles && initialFiles.length > 0) {
      processFiles(initialFiles);
    }
  }, [initialFiles]);

  // Auto-resize textarea
  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const nextHeight = Math.min(el.scrollHeight, 180);
    el.style.height = `${Math.max(nextHeight, 28)}px`;
  }, []);

  useEffect(() => {
    adjustHeight();
  }, [query, adjustHeight]);

  const processFiles = (fileList: File[] | FileList) => {
    const filesArray = Array.from(fileList).slice(0, 2);
    setUploadStatus("Processing raster attachments...");

    const items: AttachedFileItem[] = [];
    let count = 0;

    filesArray.forEach((file, idx) => {
      const isGeotiff = file.name.toLowerCase().endsWith(".tif") || file.name.toLowerCase().endsWith(".tiff");
      const item: AttachedFileItem = {
        file,
        isGeotiff,
        role: filesArray.length === 2 ? (idx === 0 ? "earlier" : "later") : "single",
      };

      // Create preview thumbnail if browser can render it
      if (file.type.startsWith("image/") && !isGeotiff) {
        const reader = new FileReader();
        reader.onload = (e) => {
          item.previewUrl = e.target?.result as string;
          count++;
          if (count === filesArray.length) {
            setAttachedFiles((prev) => [...prev, ...items].slice(0, 2));
            setUploadStatus(null);
          }
        };
        reader.readAsDataURL(file);
      } else {
        count++;
        if (count === filesArray.length) {
          setAttachedFiles((prev) => [...prev, ...items].slice(0, 2));
          setUploadStatus(null);
        }
      }
      items.push(item);
    });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  const handleRemoveFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
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

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  // Paste handler
  const handlePaste = (e: React.ClipboardEvent) => {
    if (e.clipboardData.files && e.clipboardData.files.length > 0) {
      e.preventDefault();
      processFiles(e.clipboardData.files);
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
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

    const rawFiles = attachedFiles.map((a) => a.file);
    const queryText = trimmed || "Analyze this satellite imagery.";

    setQuery("");
    setAttachedFiles([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = "28px";
    }

    await onSendMessage(queryText, pairType, rawFiles);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const getDynamicPlaceholder = () => {
    if (customPlaceholder) return customPlaceholder;
    if (attachedFiles.length === 2) return "Ask what changed between these satellite images…";
    if (attachedFiles.length === 1) return "Ask about this satellite imagery…";
    if (activeContextImageName) return `Ask about ${activeContextImageName}…`;
    return "Ask anything about your satellite imagery or upload files…";
  };

  const suggestionChips = [
    { label: "Find water", text: "Find all water bodies and calculate coverage" },
    { label: "Count buildings", text: "Count all visible building footprints" },
    { label: "Show land cover", text: "Classify dominant land cover types" },
    { label: "Compare images", text: "Compare these images to detect changes" },
  ];

  return (
    <div className="composer-container">
      {/* Context indicator if following up on previous result */}
      {activeContextImageName && attachedFiles.length === 0 && (
        <div className="composer-context-tray">
          <span className="context-chip">
            <span className="context-label">Using:</span>
            <span className="context-filename">{activeContextImageName}</span>
            {onClearContext && (
              <button
                type="button"
                className="context-clear-btn"
                onClick={onClearContext}
                title="Detach context for new image"
              >
                <X size={11} />
              </button>
            )}
          </span>
        </div>
      )}

      {/* Main Composer Box */}
      <div
        className={`floating-composer ${isDragging ? "dragging" : ""} ${
          busy ? "busy" : ""
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Upload status banner if active */}
        {uploadStatus && (
          <div className="composer-upload-status">
            <span className="upload-pulse-dot"></span>
            <span>{uploadStatus}</span>
          </div>
        )}

        {/* Attachment chips */}
        <AttachmentTray
          files={attachedFiles}
          onRemove={handleRemoveFile}
          onSwap={attachedFiles.length === 2 ? handleSwapPair : undefined}
        />

        {/* Text Input Row */}
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
            title="Attach GeoTIFF or image (PNG, JPEG)"
            aria-label="Attach satellite imagery"
          >
            <Plus size={18} />
          </button>

          <textarea
            ref={textareaRef}
            className="composer-textarea"
            placeholder={getDynamicPlaceholder()}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            rows={1}
            disabled={busy}
            aria-label="Satellite query composer"
          />

          <button
            type="button"
            className={`composer-send-btn ${
              (query.trim() || attachedFiles.length > 0) && !busy ? "ready" : "disabled"
            }`}
            onClick={() => handleSubmit()}
            disabled={busy || (!query.trim() && attachedFiles.length === 0)}
            title="Send query (Enter)"
            aria-label="Send query"
          >
            <ArrowUp size={16} />
          </button>
        </div>
      </div>

      {/* Suggestion Chips */}
      {showSuggestions && (
        <div className="composer-suggestions-row">
          {suggestionChips.map((chip) => (
            <button
              key={chip.label}
              type="button"
              className="suggestion-chip-btn"
              onClick={() => {
                if (onSelectSuggestion) onSelectSuggestion(chip.text);
                else {
                  setQuery(chip.text);
                  setTimeout(() => adjustHeight(), 10);
                }
              }}
            >
              <span>{chip.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
