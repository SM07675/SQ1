import React, { useState } from "react";
import { Copy, Check } from "lucide-react";
import { artifactUrl } from "../../api";
import { TiffPreviewImage } from "../shared/TiffPreviewImage";
import { isTiffPath, getPreviewUrl } from "../../utils/tiffViewer";

interface UserMessageProps {
  content: string;
  attachments?: Array<{
    name: string;
    url: string;
    preview_url?: string;
    previewUrl?: string;
    type: string;
  }>;
  onOpenImage?: (url: string, title: string) => void;
}

export const UserMessage: React.FC<UserMessageProps> = ({
  content,
  attachments,
  onOpenImage,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="message-row user-row">
      {/* User avatar */}
      <div className="user-avatar">
        <span className="user-avatar-text">HS</span>
      </div>

      {/* User message card */}
      <div className="user-message-bubble">
        <div className="user-text-content">{content}</div>

        {/* Attached image previews */}
        {attachments && attachments.length > 0 && (
          <div className={`user-attachments-row ${attachments.length > 1 ? "is-pair" : "is-single"}`}>
            {attachments.map((att, idx) => {
              const preview = att.preview_url || att.previewUrl;
              const isTiff = isTiffPath(att.name) || isTiffPath(att.url);
              const isImage =
                att.type === "image" ||
                isTiff ||
                Boolean(att.name.match(/\.(png|jpe?g|webp|gif|svg)$/i));

              if (isImage) {
                // Determine best URL for full resolution lightbox
                return (
                  <TiffPreviewImage
                    key={idx}
                    src={att.url}
                    alt={att.name}
                    previewUrl={preview}
                    onClick={(displayUrl) =>
                      onOpenImage && onOpenImage(displayUrl || preview || getPreviewUrl(att.url) || att.url, att.name)
                    }
                  />
                );
              }

              const fullUrl = artifactUrl(att.url) || att.url;
              return (
                <div
                  key={idx}
                  className="user-att-chip"
                  onClick={() => fullUrl && onOpenImage && onOpenImage(fullUrl, att.name)}
                  title={fullUrl ? "Click to view" : undefined}
                >
                  <span className="att-chip-name">{att.name}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Copy button */}
        <button
          type="button"
          className="msg-action-btn copy-btn"
          onClick={handleCopy}
          title="Copy message"
          aria-label="Copy query text"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
        </button>
      </div>
    </div>
  );
};
