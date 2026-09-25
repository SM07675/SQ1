import React from "react";

export type StatusVariant =
  | "supported"
  | "supported_with_limitations"
  | "disputed"
  | "insufficient_evidence"
  | "passed"
  | "failed"
  | "pending"
  | "online"
  | "offline";

interface StatusPillProps {
  variant: StatusVariant | string;
  label?: string;
  className?: string;
}

export const StatusPill: React.FC<StatusPillProps> = ({
  variant,
  label,
  className = "",
}) => {
  const normVariant = variant.toLowerCase().replace(/\s+/g, "_");
  const displayLabel =
    label ??
    normVariant
      .replace(/_/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());

  return (
    <span className={`status-pill pill-${normVariant} ${className}`}>
      <span className="status-indicator-dot" />
      {displayLabel}
    </span>
  );
};
