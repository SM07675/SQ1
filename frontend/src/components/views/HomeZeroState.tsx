import React from "react";
import {
  Sparkles,
  Droplets,
  Building2,
  GitCompare,
  Layers,
  ArrowRight,
  Clock,
} from "lucide-react";
import type { ChatSummary } from "../../types";
import { Composer } from "../composer/Composer";

interface HomeZeroStateProps {
  onSendMessage: (query: string, pairType: string, files: File[]) => Promise<void>;
  busy: boolean;
  recentChats: ChatSummary[];
  onSelectChat: (chatId: string) => void;
}

export const HomeZeroState: React.FC<HomeZeroStateProps> = ({
  onSendMessage,
  busy,
  recentChats,
  onSelectChat,
}) => {
  return (
    <div className="home-zero-state">
      {/* Background Satellite Earth Curve Visual (Subtle, authentic, non-distracting) */}
      <div className="earth-orbit-visual" aria-hidden="true">
        <div className="earth-atmosphere-glow" />
        <div className="earth-curvature" />
      </div>

      <div className="home-hero-content">
        <div className="hero-badge-pill">
          <Sparkles size={13} className="hero-badge-sparkle" />
          <span>Conversational Geospatial Intelligence</span>
        </div>

        <h1 className="hero-headline">Ask the Earth anything.</h1>

        <p className="hero-subtitle">
          Analyze satellite imagery with evidence you can inspect.
        </p>

        {/* Hero Composer with suggestion chips */}
        <div className="hero-composer-wrapper">
          <Composer
            onSendMessage={onSendMessage}
            busy={busy}
            showSuggestions={true}
            placeholder="Upload imagery or ask a question about satellite data…"
          />
        </div>
      </div>

      {/* Recent Analyses Below the Fold */}
      {recentChats.length > 0 && (
        <div className="home-recent-section">
          <div className="recent-section-header">
            <span className="recent-title">Recent Analyses</span>
            <span className="recent-count">{recentChats.length} conversations</span>
          </div>

          <div className="recent-grid">
            {recentChats.slice(0, 4).map((chat) => (
              <div
                key={chat.chat_id}
                className="recent-chat-card"
                onClick={() => onSelectChat(chat.chat_id)}
              >
                <div className="recent-card-top">
                  <div className="recent-icon-wrap">
                    <Sparkles size={14} />
                  </div>
                  <span className="recent-date">
                    <Clock size={11} />
                    <span>{new Date(chat.updated_at || chat.created_at).toLocaleDateString()}</span>
                  </span>
                </div>

                <h4 className="recent-chat-title">{chat.title}</h4>

                {chat.last_message && (
                  <p className="recent-chat-preview">{chat.last_message}</p>
                )}

                <div className="recent-card-footer">
                  <span>Resume analysis</span>
                  <ArrowRight size={13} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
