import React from "react";

interface ConfidenceGaugeProps {
  confidence: number; // 0 to 1.0
  isCalibrated?: boolean;
  confidenceInterval?: number[];
  ece?: number | null;
  size?: number;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  isCalibrated = true,
  confidenceInterval,
  ece,
  size = 110,
}) => {
  const percent = Math.min(100, Math.max(0, Math.round(confidence * 100)));
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  let strokeColor = "#10b981"; // green
  if (percent < 50) {
    strokeColor = "#ef4444"; // red
  } else if (percent < 75) {
    strokeColor = "#f59e0b"; // amber
  }

  return (
    <div className="confidence-gauge-container">
      <div className="svg-gauge-wrapper" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth={strokeWidth}
            fill="none"
          />
          {/* Active progress arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
            style={{ transition: "stroke-dashoffset 0.6s ease-in-out, stroke 0.4s" }}
          />
        </svg>

        <div className="gauge-center-content">
          <span className="gauge-percent">{percent}%</span>
          <span className="gauge-sublabel">Confidence</span>
        </div>
      </div>

      <div className="gauge-meta-stats">
        <div className="gauge-status-row">
          <span className={`calibration-tag ${isCalibrated ? "calibrated" : "heuristic"}`}>
            {isCalibrated ? "✓ Platt Scaled" : "Heuristic"}
          </span>
        </div>

        {confidenceInterval && confidenceInterval.length >= 2 && (
          <div className="gauge-interval-row" title="95% Confidence Interval">
            <span className="interval-label">95% CI:</span>
            <span className="interval-val">
              [{Math.round(confidenceInterval[0] * 100)}%, {Math.round(confidenceInterval[1] * 100)}%]
            </span>
          </div>
        )}

        {typeof ece === "number" && (
          <div className="gauge-ece-row" title="Expected Calibration Error (ECE)">
            <span className="ece-label">ECE:</span>
            <span className="ece-val">{(ece * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>
    </div>
  );
};
