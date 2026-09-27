import React, { useState } from "react";
import { Copy, Check, Maximize2 } from "lucide-react";
import { artifactUrl } from "../../api";

interface UserMessageProps {
  content: string;
  attachments?: Array<{ name: string; url: string; type: string }>;
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
          <div className="user-attachments-row">
            {attachments.map((att, idx) => {
              const fullUrl = artifactUrl(att.url) || att.url;
              const isImage = att.type === "image" || att.name.match(/\.(png|jpg|jpeg)$/i);

              if (isImage && fullUrl) {
                return (
                  <div
                    key={idx}
                    className="user-attached-image-preview"
                    onClick={() => onOpenImage && onOpenImage(fullUrl, att.name)}
                    title="Click to view full image"
                  >
                    <img src={fullUrl} alt={att.name} loading="lazy" />
                    <div className="maximize-overlay">
                      <Maximize2 size={14} />
                    </div>
                  </div>
                );
              }

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
