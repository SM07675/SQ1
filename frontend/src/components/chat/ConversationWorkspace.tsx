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
  onSendMessage?: (query: string, pairType: string, files: File[]) => Promise<void>;
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
  onSendMessage,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef(false);

  // Auto-scroll only if user hasn't manually scrolled up
  useEffect(() => {
    if (!userScrolledRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, busy, error]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    userScrolledRef.current = scrollHeight - scrollTop - clientHeight > 100;
  };

  return (
    <div className="conversation-workspace" ref={containerRef} onScroll={handleScroll}>
      {/* Chat header (compact when messages exist to bring results into immediate view) */}
      <div className={`chat-page-header ${messages.length > 0 ? "has-messages" : ""}`}>
        <h2 className="chat-page-title">
          Ask <span className="gradient-text">SatQuery</span>
        </h2>
        {messages.length === 0 && (
          <p className="chat-page-subtitle">
            Ask questions about satellite imagery and receive clear, actionable insights.
          </p>
        )}
      </div>

      <div className="conversation-message-stream">
        {messages.map((msg, idx) => {
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
            const isLatest = idx === messages.length - 1;
            return (
              <AssistantMessage
                key={msg.message_id}
                content={msg.content}
                result={msg.result}
                isLatest={isLatest}
                onOpenLightbox={onOpenLightbox}
                onOpenMap={onOpenMap}
                onOpenEvidenceDrawer={onOpenEvidenceDrawer}
                onOpenInspector={onOpenInspector}
                onPreviewReport={onPreviewReport}
                onFollowUp={(queryText) => {
                  if (onSendMessage) {
                    onSendMessage(queryText, "auto", []);
                  }
                }}
              />
            );
          }

          return null;
        })}

        {/* Real-time AI Progress State */}
        {busy && <AnalysisProgress busy={busy} hasAttachedImages={hasAttachedImages} />}

        {/* Error State */}
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
