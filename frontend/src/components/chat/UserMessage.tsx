import React, { useState } from "react";
import { Copy, Check, Image as ImageIcon } from "lucide-react";
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
      <div className="user-message-bubble">
        {/* Attached image preview chips if any */}
        {attachments && attachments.length > 0 && (
          <div className="user-attachments-row">
            {attachments.map((att, idx) => {
              const fullUrl = artifactUrl(att.url) || att.url;
              return (
                <div
                  key={idx}
                  className="user-att-chip"
                  onClick={() => fullUrl && onOpenImage && onOpenImage(fullUrl, att.name)}
                  title={fullUrl ? "Click to view full image" : undefined}
                >
                  <ImageIcon size={13} className="att-chip-icon" />
                  <span className="att-chip-name">{att.name}</span>
                </div>
              );
            })}
          </div>
        )}

        <div className="user-text-content">{content}</div>

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
