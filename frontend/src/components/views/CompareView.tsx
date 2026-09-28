import React, { useState, useRef, useEffect } from "react";
import { GitCompare, Upload, ArrowLeftRight, X, Image as ImageIcon, Loader2 } from "lucide-react";
import { Composer } from "../composer/Composer";
import { convertTiffToDataUrl, isTiffPath } from "../../utils/tiffViewer";

function useFilePreview(file: File | null): { url: string | null; loading: boolean; error: boolean } {
  const [preview, setPreview] = useState<{ url: string | null; loading: boolean; error: boolean }>({ url: null, loading: false, error: false });

  useEffect(() => {
    if (!file) {
      setPreview({ url: null, loading: false, error: false });
      return;
    }
    let cancelled = false;
    if (isTiffPath(file.name)) {
      setPreview({ url: null, loading: true, error: false });
      convertTiffToDataUrl(file, 1200)
        .then((url) => { if (!cancelled) setPreview({ url, loading: false, error: false }); })
        .catch(() => { if (!cancelled) setPreview({ url: null, loading: false, error: true }); });
      return () => { cancelled = true; };
    }
    if (file.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(file.name)) {
      const url = URL.createObjectURL(file);
      setPreview({ url, loading: false, error: false });
      return () => { URL.revokeObjectURL(url); };
    }
    setPreview({ url: null, loading: false, error: true });
    return () => { cancelled = true; };
  }, [file]);

  return preview;
}

interface CompareViewProps {
  onSendMessage: (query: string, pairType: string, files: File[]) => Promise<void>;
  busy: boolean;
}

export const CompareView: React.FC<CompareViewProps> = ({
  onSendMessage,
  busy,
}) => {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const previewA = useFilePreview(fileA);
  const previewB = useFilePreview(fileB);
  const [mode, setMode] = useState<"bi_temporal" | "optical_sar">("bi_temporal");

  const inputARef = useRef<HTMLInputElement>(null);
  const inputBRef = useRef<HTMLInputElement>(null);
  const dualDropRef = useRef<HTMLInputElement>(null);

  const handleDualFiles = (files: FileList | null) => {
    if (!files) return;
    if (files.length >= 1) handleFileA(files[0]);
    if (files.length >= 2) handleFileB(files[1]);
  };

  const handleFileA = (file: File) => {
    setFileA(file);
  };

  const handleFileB = (file: File) => {
    setFileB(file);
  };

  const handleSwap = () => {
    setFileA(fileB);
    setFileB(fileA);
  };

  const handleSendFromCompare = async (query: string, pairType: string, extraFiles: File[]) => {
    const filesToSend: File[] = [];
    if (fileA) filesToSend.push(fileA);
    if (fileB) filesToSend.push(fileB);
    const finalFiles = filesToSend.length > 0 ? filesToSend : extraFiles;
    await onSendMessage(query || "Compare these two satellite images to detect changes.", mode, finalFiles);
  };

  return (
    <div className="compare-view-page">
      <div className="compare-header">
        <div className="compare-badge">
          <GitCompare size={14} />
          <span>Bi-Temporal & Dual-Modal Comparison</span>
        </div>
        <h2 className="compare-title">Compare Satellite Imagery</h2>
        <p className="compare-subtitle">
          Upload two temporal observation scenes or paired Optical + SAR images to detect surface changes, flood extent, or structural shifts.
        </p>

        {/* Mode Selector Pill */}
        <div className="compare-mode-toggle">
          <button
            type="button"
            className={`compare-toggle-btn ${mode === "bi_temporal" ? "active" : ""}`}
            onClick={() => setMode("bi_temporal")}
          >
            Temporal Change (Before / After)
          </button>
          <button
            type="button"
            className={`compare-toggle-btn ${mode === "optical_sar" ? "active" : ""}`}
            onClick={() => setMode("optical_sar")}
          >
            Optical + SAR Fusion
          </button>
        </div>
      </div>

      {/* Dual Upload Zone */}
      <div className="dual-upload-zone">
        <input
          type="file"
          ref={dualDropRef}
          multiple
          style={{ display: "none" }}
          accept=".tif,.tiff,.png,.jpg,.jpeg,.nc"
          onChange={(e) => handleDualFiles(e.target.files)}
        />

        {/* Image A Slot */}
        <div
          className={`image-slot ${fileA ? "filled" : "empty"}`}
          onClick={() => !fileA && inputARef.current?.click()}
        >
          <input
            type="file"
            ref={inputARef}
            style={{ display: "none" }}
            accept=".tif,.tiff,.png,.jpg,.jpeg,.nc"
            onChange={(e) => e.target.files?.[0] && handleFileA(e.target.files[0])}
          />
          <span className="slot-badge">{mode === "bi_temporal" ? "Earlier Scene (Image A)" : "Optical Satellite"}</span>

          {fileA ? (
            <div className="slot-preview-content">
              {previewA.url ? (
                <img src={previewA.url} alt={fileA.name} className="slot-img-preview" />
              ) : (
                <div className="slot-fallback-icon">
                  {previewA.loading ? <Loader2 size={32} className="animate-spin" /> : <ImageIcon size={32} />}
                  <span>{previewA.loading ? "Rendering raster…" : previewA.error ? "Preview unavailable" : "Image preview"}</span>
                </div>
              )}
              <div className="slot-file-meta">
                <span className="slot-filename">{fileA.name}</span>
                <span className="slot-filesize">{(fileA.size / (1024 * 1024)).toFixed(1)} MB</span>
              </div>
              <button
                type="button"
                className="slot-remove-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setFileA(null);
                }}
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <div className="slot-empty-prompt">
              <Upload size={24} className="slot-upload-icon" />
              <span className="slot-prompt-text">Choose or drop Image A</span>
              <span className="slot-prompt-sub">GeoTIFF, TIFF, PNG, JPEG</span>
            </div>
          )}
        </div>

        {/* Swap Button */}
        {fileA && fileB && (
          <button
            type="button"
            className="compare-swap-btn"
            onClick={handleSwap}
            title="Swap Image A and B"
          >
            <ArrowLeftRight size={16} />
          </button>
        )}

        {/* Image B Slot */}
        <div
          className={`image-slot ${fileB ? "filled" : "empty"}`}
          onClick={() => !fileB && inputBRef.current?.click()}
        >
          <input
            type="file"
            ref={inputBRef}
            style={{ display: "none" }}
            accept=".tif,.tiff,.png,.jpg,.jpeg,.nc"
            onChange={(e) => e.target.files?.[0] && handleFileB(e.target.files[0])}
          />
          <span className="slot-badge">{mode === "bi_temporal" ? "Later Scene (Image B)" : "SAR Sensor (Sentinel-1)"}</span>

          {fileB ? (
            <div className="slot-preview-content">
              {previewB.url ? (
                <img src={previewB.url} alt={fileB.name} className="slot-img-preview" />
              ) : (
                <div className="slot-fallback-icon">
                  {previewB.loading ? <Loader2 size={32} className="animate-spin" /> : <ImageIcon size={32} />}
                  <span>{previewB.loading ? "Rendering raster…" : previewB.error ? "Preview unavailable" : "Image preview"}</span>
                </div>
              )}
              <div className="slot-file-meta">
                <span className="slot-filename">{fileB.name}</span>
                <span className="slot-filesize">{(fileB.size / (1024 * 1024)).toFixed(1)} MB</span>
              </div>
              <button
                type="button"
                className="slot-remove-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setFileB(null);
                }}
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <div className="slot-empty-prompt">
              <Upload size={24} className="slot-upload-icon" />
              <span className="slot-prompt-text">Choose or drop Image B</span>
              <span className="slot-prompt-sub">GeoTIFF, TIFF, PNG, JPEG</span>
            </div>
          )}
        </div>
      </div>

      {/* Composer */}
      <div className="compare-composer-wrap">
        <Composer
          onSendMessage={handleSendFromCompare}
          busy={busy}
          placeholder={
            fileA && fileB
              ? "Ask what changed between these satellite images…"
              : "Upload two images above or ask a comparison query…"
          }
        />
      </div>
    </div>
  );
};
