import React, { DragEvent, useRef, useState } from "react";

interface FileDropZoneProps {
  label: string;
  sublabel?: string;
  badgeText: string;
  file: File | null;
  onFileSelect: (file: File | null) => void;
  accept?: string;
  required?: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export const FileDropZone: React.FC<FileDropZoneProps> = ({
  label,
  sublabel,
  badgeText,
  file,
  onFileSelect,
  accept = ".tif,.tiff,.png,.jpg,.jpeg,.nc",
  required = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      onFileSelect(droppedFile);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileSelect(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      className={`file-drop-zone ${isDragOver ? "drag-over" : ""} ${file ? "has-file" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <div className="drop-zone-header">
        <span className="drop-badge">{badgeText}</span>
        <div className="drop-labels">
          <span className="drop-title">
            {label} {required && <span className="req-star">*</span>}
          </span>
          {sublabel && <small className="drop-sub">{sublabel}</small>}
        </div>
      </div>

      {file ? (
        <div className="drop-file-info">
          <div className="file-icon">🛰️</div>
          <div className="file-meta">
            <span className="file-name" title={file.name}>
              {file.name}
            </span>
            <span className="file-size">{formatBytes(file.size)}</span>
          </div>
          <button
            type="button"
            className="clear-file-btn"
            onClick={clearFile}
            title="Remove file"
          >
            ✕
          </button>
        </div>
      ) : (
        <div className="drop-placeholder">
          <span className="upload-arrow">⇪</span>
          <p>
            Drop GeoTIFF / PNG / NetCDF or <u>browse</u>
          </p>
        </div>
      )}
    </div>
  );
};
