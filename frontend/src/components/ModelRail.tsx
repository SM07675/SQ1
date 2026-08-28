import React from "react";
import type { ModelCapability } from "../types";

interface ModelRailProps {
  models: ModelCapability[];
}

export const ModelRail: React.FC<ModelRailProps> = ({ models }) => {
  return (
    <section className="model-rail" aria-label="Pipeline Capabilities">
      <div className="model-chip always-on" title="Native GeoTIFF, NetCDF, GDAL & PostGIS Engine">
        <i className="chip-indicator" />
        <span className="chip-name">GeoTIFF + GIS Engine</span>
        <span className="chip-status">Ready</span>
      </div>

      <div className="model-chip always-on" title="Spectral Indices (NDVI, NDWI, NDBI) Processor">
        <i className="chip-indicator" />
        <span className="chip-name">Spectral Math (NDVI/WI/BI)</span>
        <span className="chip-status">Ready</span>
      </div>

      {models.map((model) => (
        <div
          key={model.name}
          className={`model-chip ${model.available ? "online" : "optional"}`}
          title={`${model.name}: ${model.purpose}`}
        >
          <i className="chip-indicator" />
          <span className="chip-name">{model.name}</span>
          <span className="chip-status">{model.available ? "Connected" : "Optional"}</span>
        </div>
      ))}
    </section>
  );
};
