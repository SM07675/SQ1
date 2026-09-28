import React from "react";
import { Clock, Leaf, Building2, Droplets, TrendingUp, Layers, MapPin, Info } from "lucide-react";
import type { AnalysisResponse } from "../../types";

interface KeyFindingsCardProps {
  result: AnalysisResponse;
  onOpenEvidence?: () => void;
}

interface FindingRow {
  icon: React.ReactNode;
  iconClass: string;
  label: string;
  value: string;
  confidence: string;
}

const metric = (value: unknown): number | null => {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

export const KeyFindingsCard: React.FC<KeyFindingsCardProps> = ({ result, onOpenEvidence }) => {
  const stats = result.statistics || {};
  const evidence = result.evidence || [];
  const fromEvidence = (...kinds: string[]) =>
    (evidence.find((item) => kinds.includes(item.kind))?.metrics || {}) as Record<string, unknown>;
  const change = (stats.change || fromEvidence("change", "baseline_change_detection")) as Record<string, unknown>;
  const buildingChange = (stats.building_match || fromEvidence("building_match")) as Record<string, unknown>;
  const land = (stats.land_cover || fromEvidence("land_cover", "land_cover_classification")) as Record<string, unknown>;
  const water = (stats.water_measure || fromEvidence("spectral", "water_grounding_evidence")) as Record<string, unknown>;
  const building = (stats.buildings_a || stats.building_detection || fromEvidence("buildings", "building_detection_evidence")) as Record<string, unknown>;
  const rows: FindingRow[] = [];
  const add = (icon: React.ReactNode, iconClass: string, label: string, value: string, confidence = "Estimated") => {
    rows.push({ icon, iconClass, label, value, confidence });
  };

  const changedPercent = metric(change.changed_percent);
  const changedPixels = metric(change.changed_pixels);
  if (changedPercent !== null) {
    add(<TrendingUp size={16} />, "finding-icon-trend", "Image difference", `${changedPercent.toFixed(2)}%`, "Measured");
    if (changedPixels !== null) add(<Layers size={16} />, "finding-icon-leaf", "Changed pixels", changedPixels.toLocaleString(), "Measured");
  }
  const beforeCount = metric(buildingChange.before_count);
  const afterCount = metric(buildingChange.after_count);
  if (beforeCount !== null && afterCount !== null) {
    add(<Building2 size={16} />, "finding-icon-building", "Building footprints", `${beforeCount} → ${afterCount}`);
    const newCount = metric(buildingChange.possible_new_count);
    if (newCount !== null) add(<MapPin size={16} />, "finding-icon-trend", "Possibly new", String(newCount));
    const percent = metric(buildingChange.net_count_change_percent);
    if (percent !== null) add(<TrendingUp size={16} />, "finding-icon-trend", "Count change", `${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%`);
  } else {
    const count = metric(building.count ?? building.building_count);
    if (count !== null) {
      add(<Building2 size={16} />, "finding-icon-building", "Building footprints", String(count));
      const percent = metric(building.coverage_percent);
      if (percent !== null) add(<Layers size={16} />, "finding-icon-leaf", "Footprint coverage", `${percent.toFixed(2)}%`);
    }
  }

  const landPercent = metric(land.land_percent ?? land.land_coverage_percent);
  if (landPercent !== null) add(<Layers size={16} />, "finding-icon-trend", "Land coverage", `${landPercent.toFixed(2)}%`);
  const breakdown = land.breakdown as Record<string, { percent?: number }> | undefined;
  if (breakdown) {
    const known = Object.entries(breakdown).filter(([name, part]) => name !== "unknown" && metric(part?.percent) !== null);
    known.sort((a, b) => Number(b[1].percent) - Number(a[1].percent));
    for (const [name, part] of known.slice(0, Math.max(0, 4 - rows.length))) {
      add(<Leaf size={16} />, "finding-icon-leaf", name.replace(/_/g, " "), `${Number(part.percent).toFixed(2)}%`);
    }
  }
  const waterPercent = metric(water.coverage_percent ?? water.total_coverage_percent);
  if (waterPercent !== null && rows.length < 4) {
    add(<Droplets size={16} />, "finding-icon-water", "Water coverage", `${waterPercent.toFixed(2)}%`);
    const pixels = metric(water.selected_pixels ?? water.water_pixels);
    if (pixels !== null) add(<MapPin size={16} />, "finding-icon-trend", "Water pixels", pixels.toLocaleString(), "Measured");
  }
  if (!rows.length) add(<Info size={16} />, "finding-icon-trend", "Analysis status", result.verdict.status.replace(/_/g, " "), "See evidence");

  const elapsed = metric(result.timings?.analysis_ms);
  return (
    <div className="key-findings-card" onClick={onOpenEvidence} title="Click to view full evidence details">
      <div className="key-findings-header">
        <h4 className="key-findings-title">Key Findings</h4>
        {elapsed !== null && <div className="key-findings-timing"><Clock size={13} strokeWidth={2} /><span>Analyzed in {(elapsed / 1000).toFixed(1)}s</span></div>}
      </div>
      <div className="key-findings-list">
        {rows.slice(0, 4).map((row, idx) => (
          <div key={idx} className="key-finding-row">
            <div className="finding-label-col"><span className={`finding-icon-wrap ${row.iconClass}`}>{row.icon}</span><span className="finding-label-text">{row.label}</span></div>
            <div className="finding-value-col"><span className="finding-value-text">{row.value}</span></div>
            <div className="finding-badge-col"><span className="finding-conf-badge badge-teal">{row.confidence}</span></div>
          </div>
        ))}
      </div>
    </div>
  );
};
