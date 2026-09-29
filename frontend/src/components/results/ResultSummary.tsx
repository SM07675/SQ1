import React from "react";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  XCircle,
  Droplets,
  Building2,
  GitCompare,
  Layers,
  MapPin,
  TrendingUp,
  Radio,
} from "lucide-react";
import type { AnalysisResponse, VerdictStatus } from "../../types";

interface ResultSummaryProps {
  result: AnalysisResponse;
}

function formatArea(areaM2?: number | null): string | null {
  if (areaM2 == null || areaM2 <= 0) return null;
  if (areaM2 >= 1_000_000) {
    const km2 = areaM2 / 1_000_000;
    return `${km2 >= 10 ? km2.toFixed(1) : km2.toFixed(2)} km²`;
  }
  if (areaM2 >= 10_000) {
    const ha = areaM2 / 10_000;
    return `${ha >= 10 ? ha.toFixed(1) : ha.toFixed(2)} ha`;
  }
  return `${Math.round(areaM2)} m²`;
}

function getStatusBadge(status: VerdictStatus) {
  switch (status) {
    case "supported":
      return {
        label: "Supported",
        className: "status-pill-supported",
        icon: <CheckCircle2 size={13} />,
      };
    case "supported_with_limitations":
      return {
        label: "Supported with limitations",
        className: "status-pill-limited",
        icon: <AlertTriangle size={13} />,
      };
    case "disputed":
      return {
        label: "Evidence Disputed",
        className: "status-pill-disputed",
        icon: <XCircle size={13} />,
      };
    case "low_confidence":
      return {
        label: "Limited evidence",
        className: "status-pill-inconclusive",
        icon: <HelpCircle size={13} />,
      };
    case "insufficient_evidence":
    default:
      return {
        label: "Insufficient evidence",
        className: "status-pill-inconclusive",
        icon: <HelpCircle size={13} />,
      };
  }
}

export const ResultSummary: React.FC<ResultSummaryProps> = ({ result }) => {
  const { verdict, evidence, assets, statistics, summary } = result;

  // Extract primary metrics based on analysis type
  const metrics: { label: string; value: string; icon?: React.ReactNode }[] = [];

  // Check backend precomputed summary metrics
  if (summary?.metrics && summary.metrics.length > 0) {
    summary.metrics.slice(0, 4).forEach((m) => {
      metrics.push({
        label: m.label,
        value: m.value,
      });
    });
  } else {
    // Water Grounding / Detection metrics
    const waterEv = evidence.find((e) => e.kind === "water_grounding_evidence");
    const waterStats = statistics?.water_measure as Record<string, any> | undefined;

    if (waterEv || waterStats) {
      const m = (waterEv?.metrics || waterStats || {}) as Record<string, any>;
      const covPct = Math.round(Number(m.total_coverage_percent ?? m.coverage_percent ?? 0));
      const areaM2 = (m.total_area_m2 ?? m.area_m2) as number | null;
      const count = Number(m.region_count ?? 1);

      metrics.push({
        label: "Water Coverage",
        value: `${covPct}%`,
        icon: <Droplets size={14} />,
      });
      if (areaM2) {
        metrics.push({
          label: "Detected Area",
          value: formatArea(areaM2) || `${covPct}%`,
          icon: <MapPin size={14} />,
        });
      }
      if (count > 0) {
        metrics.push({
          label: "Water Bodies",
          value: String(count),
          icon: <Droplets size={14} />,
        });
      }
    }

    // Change Detection metrics
    const changeEv = evidence.find((e) =>
      ["baseline_change_detection", "learned_change_witness"].includes(e.kind)
    );
    if (changeEv) {
      const m = (changeEv.metrics || {}) as Record<string, any>;
      const changedPct = Math.round(Number(m.changed_percent ?? 0));
      const areaM2 = m.area_m2 as number | null;
      const count = Number(m.region_count ?? 0);

      metrics.push({
        label: "Changed Area",
        value: `${changedPct}%`,
        icon: <GitCompare size={14} />,
      });
      if (areaM2) {
        metrics.push({
          label: "Area Changed",
          value: formatArea(areaM2) || `${changedPct}%`,
          icon: <MapPin size={14} />,
        });
      }
      if (count > 0) {
        metrics.push({
          label: "Change Regions",
          value: String(count),
          icon: <TrendingUp size={14} />,
        });
      }
    }

    // Building footprint metrics
    const buildingEv = evidence.find((e) =>
      e.kind.includes("building") || e.kind.includes("footprint")
    );
    if (buildingEv) {
      const m = (buildingEv.metrics || {}) as Record<string, any>;
      const count = Number(m.count ?? m.building_count ?? 0);
      if (count > 0) {
        metrics.push({
          label: "Buildings Detected",
          value: String(count),
          icon: <Building2 size={14} />,
        });
      }
    }

    // Land Cover metrics
    const landEv = evidence.find((e) =>
      e.kind.includes("land_cover") || e.kind.includes("surface_context") || e.kind.includes("landcover")
    );
    if (landEv) {
      const m = (landEv.metrics || {}) as Record<string, any>;
      if (m.dominant_class && metrics.length < 4) {
        metrics.push({
          label: "Dominant Land",
          value: String(m.dominant_class).replace(/_/g, " "),
          icon: <Layers size={14} />,
        });
      }
      const landCov = m.classified_land_percent ?? m.land_coverage_percent ?? m.coverage_percent;
      if (landCov != null && metrics.length < 4) {
        metrics.push({
          label: "Land Coverage",
          value: `${Math.round(Number(landCov))}%`,
          icon: <Layers size={14} />,
        });
      }
      if (m.vegetation_percent != null && metrics.length < 4) {
        metrics.push({
          label: "Grass / Veg",
          value: `${Math.round(Number(m.vegetation_percent))}%`,
          icon: <Layers size={14} />,
        });
      }
      if (m.road_percent != null && metrics.length < 4 && Number(m.road_percent) > 0) {
        metrics.push({
          label: "Roads",
          value: `${Math.round(Number(m.road_percent))}%`,
          icon: <Layers size={14} />,
        });
      }
    }

    // SAR Radar metrics
    const sarEv = evidence.find((e) =>
      e.kind.includes("sar") || e.kind.includes("radar") || e.kind.includes("backscatter")
    );
    if (sarEv && metrics.length < 4) {
      const m = (sarEv.metrics || {}) as Record<string, any>;
      if (m.polarization || m.polarizations) {
        metrics.push({
          label: "SAR Polarization",
          value: String(m.polarization ?? m.polarizations ?? "VV"),
          icon: <Radio size={14} />,
        });
      }
    }

    // Add confidence if we have room
    if (metrics.length < 4 && verdict.confidence != null) {
      metrics.push({
        label: verdict.confidence_kind === "uncalibrated_evidence_strength" ? "Evidence strength" : "Confidence score",
        value: verdict.confidence_kind === "uncalibrated_evidence_strength" ? verdict.confidence.toFixed(2) : `${Math.round(verdict.confidence * 100)}%`,
        icon: <CheckCircle2 size={14} />,
      });
    }
  }

  const badge = getStatusBadge(verdict.status);

  return (
    <div className="result-summary-card">
      <div className="summary-header-row">
        <div className="summary-title-wrap">
          <span className="summary-section-label">GeoProof Findings</span>
          <span className={`summary-status-pill ${badge.className}`}>
            {badge.icon}
            <span>{badge.label}</span>
          </span>
        </div>
      </div>

      {/* Metrics Row: 3-4 primary clear numbers */}
      <div className="summary-metrics-grid">
        {metrics.map((m, idx) => (
          <div key={idx} className="metric-cell">
            <div className="metric-cell-label">
              {m.icon}
              <span>{m.label}</span>
            </div>
            <div className="metric-cell-value">{m.value}</div>
          </div>
        ))}
      </div>

      {/* Limitations note if applicable */}
      {verdict.status === "supported_with_limitations" && verdict.limitations?.length > 0 && (
        <div className="summary-limitation-banner">
          <AlertTriangle size={14} className="limitation-icon" />
          <span>{verdict.limitations[0]}</span>
        </div>
      )}
    </div>
  );
};
