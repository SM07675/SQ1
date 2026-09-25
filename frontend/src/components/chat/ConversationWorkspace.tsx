import React, { useRef, useEffect } from "react";
import { AlertCircle, RotateCcw } from "lucide-react";
import type { ChatMessageRecord, AnalysisResponse } from "../../types";
import { UserMessage } from "./UserMessage";
import { AssistantMessage } from "./AssistantMessage";
import { AnalysisProgress } from "./AnalysisProgress";

interface ConversationWorkspaceProps {
  messages: ChatMessageRecord[];
  busy: boolean;
  error: string | null;
  onRetry?: () => void;
  onOpenLightbox: (imageUrl: string, title: string) => void;
  onOpenMap: (result: AnalysisResponse) => void;
  onOpenEvidenceDrawer: (result: AnalysisResponse) => void;
  onOpenInspector: (result: AnalysisResponse) => void;
  onPreviewReport: (resultId: string) => void;
  hasAttachedImages: boolean;
}

export const ConversationWorkspace: React.FC<ConversationWorkspaceProps> = ({
  messages,
  busy,
  error,
  onRetry,
  onOpenLightbox,
  onOpenMap,
  onOpenEvidenceDrawer,
  onOpenInspector,
  onPreviewReport,
  hasAttachedImages,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy, error]);

  return (
    <div className="conversation-workspace">
      <div className="conversation-message-stream">
        {messages.map((msg) => {
          if (msg.role === "user") {
            return (
              <UserMessage
                key={msg.message_id}
                content={msg.content}
                attachments={msg.attachments}
                onOpenImage={onOpenLightbox}
              />
            );
          }

          if (msg.role === "assistant") {
            return (
              <AssistantMessage
                key={msg.message_id}
                content={msg.content}
                result={msg.result}
                onOpenLightbox={onOpenLightbox}
                onOpenMap={onOpenMap}
                onOpenEvidenceDrawer={onOpenEvidenceDrawer}
                onOpenInspector={onOpenInspector}
                onPreviewReport={onPreviewReport}
              />
            );
          }

          return null;
        })}

        {/* Real-time AI Progress State */}
        {busy && <AnalysisProgress busy={busy} hasAttachedImages={hasAttachedImages} />}

        {/* Friendly Error State */}
        {error && (
          <div className="chat-error-banner">
            <div className="error-icon-box">
              <AlertCircle size={18} />
            </div>
            <div className="error-text-content">
              <div className="error-heading">Analysis could not be completed</div>
              <div className="error-message">{error}</div>
            </div>
            {onRetry && (
              <button type="button" className="retry-action-btn" onClick={onRetry}>
                <RotateCcw size={14} />
                <span>Retry</span>
              </button>
            )}
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} className="conversation-bottom-anchor" />
      </div>
    </div>
  );
};
