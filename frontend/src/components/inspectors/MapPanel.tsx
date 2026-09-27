import React, { useEffect, useMemo, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { X, Map as MapIcon, Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw, Layers, MapPin } from "lucide-react";
import type { AnalysisResponse, ArtifactRef } from "../../types";
import { artifactUrl } from "../../api";

interface MapPanelProps {
  isOpen: boolean;
  onClose: () => void;
  result: AnalysisResponse | null;
}

type Bounds = [number, number, number, number];
type MapCollection = GeoJSON.FeatureCollection<GeoJSON.Geometry, GeoJSON.GeoJsonProperties>;

function geographicBounds(result: AnalysisResponse): Bounds | null {
  const asset = result.assets?.[0];
  const value = asset?.bounds;
  if (!asset || asset.crs?.toUpperCase() !== "EPSG:4326" || !Array.isArray(value) || value.length !== 4) return null;
  const [west, south, east, north] = value;
  if (![west, south, east, north].every(Number.isFinite) || west < -180 || east > 180 || south < -90 || north > 90 || west >= east || south >= north) return null;
  return [west, south, east, north];
}

function matchingArtifacts(result: AnalysisResponse): { original?: ArtifactRef; overlay?: ArtifactRef; vector?: ArtifactRef } {
  const task = (result.task_plan?.task || "").toLowerCase();
  const query = (result.query || "").toLowerCase();
  const inferred = !task || task === "grounding";
  const group = task === "land_cover" || (inferred && /\b(land|soil|cover|grass)\b/.test(query)) ? "land"
    : ["building_count", "building_detection", "buildings"].includes(task) || (inferred && /\b(buildings?|footprints?)\b/.test(query)) ? "building"
      : ["water_analysis", "water_detection"].includes(task) || (inferred && /\b(water|lake|river|flood)\b/.test(query)) ? "water" : "other";
  const prefersLandOnly = group === "land" && !/\b(water|lake|river|flood|ocean|sea|pond|waterbody)\b/.test(query);
  const overlayNames = prefersLandOnly
    ? ["land_only_overlay.png", "land_cover_overlay.png"]
    : group === "land"
      ? ["land_cover_overlay.png", "land_only_overlay.png"]
      : group === "building" ? ["buildings_overlay.png", "building_overlay.png"]
      : group === "water" ? ["water_overlay.png"] : [];
  const vectorNames = group === "land" ? ["land_cover.geojson"]
    : group === "building" ? ["buildings.geojson"]
      : group === "water" ? ["water.geojson"] : [];
  const find = (name: string) => result.artifacts.find((artifact) => artifact.name === name);
  return {
    original: find("preview_1.png") || result.artifacts.find((a) => a.name.includes("input_a") || a.name.includes("original")) || result.artifacts.find((a) => a.mime_type.startsWith("image/")),
    overlay: overlayNames.map(find).find(Boolean) || result.artifacts.find((a) => a.name.includes("overlay") || a.name.includes("change_mask")),
    vector: vectorNames.map(find).find(Boolean) || result.artifacts.find((a) => a.name.endsWith(".geojson")),
  };
}

function validCollection(data: unknown): data is MapCollection {
  return !!data && typeof data === "object" && "type" in data && data.type === "FeatureCollection" &&
    "features" in data && Array.isArray(data.features);
}

const COLORS: Record<string, string> = {
  water: "#2475cf", grass: "#29bd6c", forest: "#14704c", woodland: "#14704c", land: "#b49155",
  soil: "#b49155", bare_pervious: "#b49155", roads: "#ffd44e", road: "#ffd44e", built_up: "#ef5963",
  building: "#ef5963", cropland: "#90bc54", agriculture: "#90bc54", vegetation: "#29bd6c",
  swimming_pool: "#06b6d4", snow: "#f0f8ff",
};

function featureColor(feature: GeoJSON.Feature): string {
  const properties = feature.properties || {};
  const named = properties.class || properties.kind;
  if (typeof named === "string" && COLORS[named.toLowerCase()]) return COLORS[named.toLowerCase()];
  return typeof properties.color === "string" && /^#[\da-f]{6}$/i.test(properties.color) ? properties.color : "#41b9f4";
}

function pixelPath(geometry: unknown): string | null {
  if (!geometry || typeof geometry !== "object" || !("type" in geometry) || !("coordinates" in geometry)) return null;
  const value = geometry as { type: string; coordinates: unknown };
  const polygons = value.type === "Polygon" ? [value.coordinates] : value.type === "MultiPolygon" ? value.coordinates : null;
  if (!Array.isArray(polygons)) return null;
  const parts: string[] = [];
  for (const polygon of polygons) {
    if (!Array.isArray(polygon)) continue;
    for (const ring of polygon) {
      if (!Array.isArray(ring) || ring.length < 4) continue;
      const points = ring.map((point: unknown) =>
        Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1])
          ? `${point[0]} ${point[1]}` : null);
      if (points.some((point) => point === null)) continue;
      parts.push(`M ${points.join(" L ")} Z`);
    }
  }
  return parts.length ? parts.join(" ") : null;
}

export const MapPanel: React.FC<MapPanelProps> = ({ isOpen, onClose, result }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [collection, setCollection] = useState<MapCollection | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [localZoom, setLocalZoom] = useState(1);
  const [localOffset, setLocalOffset] = useState({ x: 0, y: 0 });
  const [selectedLocal, setSelectedLocal] = useState<string | null>(null);
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<maplibregl.Map | null>(null);
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);

  const bounds = useMemo(() => result ? geographicBounds(result) : null, [result]);
  const artifacts = useMemo(() => result ? matchingArtifacts(result) : null, [result]);
  const originalUrl = artifactUrl(artifacts?.original?.url) || "";
  const overlayUrl = artifactUrl(artifacts?.overlay?.url) || "";
  const isGeographic = bounds !== null && originalUrl !== "";
  const localShapes = useMemo(() => {
    if (!collection) return [];
    const metadata = collection as MapCollection & { properties?: { coordinate_space?: string } };
    return collection.features.flatMap((feature, index) => {
      const properties = feature.properties || {};
      const path = pixelPath(properties.pixel_geometry || (metadata.properties?.coordinate_space === "pixel" ? feature.geometry : null));
      if (!path) return [];
      const name = String(properties.class || properties.kind || `Feature ${index + 1}`).replaceAll("_", " ");
      const area = typeof properties.area_m2 === "number" && Number.isFinite(properties.area_m2)
        ? ` · ${properties.area_m2.toLocaleString(undefined, { maximumFractionDigits: 1 })} m²` : "";
      return [{ id: index, path, color: featureColor(feature), label: name + area }];
    });
  }, [collection]);

  useEffect(() => {
    setCollection(null);
    setMapError(null);
    setLocalZoom(1);
    setLocalOffset({ x: 0, y: 0 });
    setSelectedLocal(null);
    if (!isOpen || !artifacts?.vector) return;
    const controller = new AbortController();
    const url = artifactUrl(artifacts.vector.url);
    if (!url) return;
    fetch(url, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error("Vector layer could not be loaded."); return response.json(); })
      .then((data: unknown) => { if (validCollection(data)) setCollection(data); })
      .catch((error: unknown) => { if (!controller.signal.aborted) setMapError(error instanceof Error ? error.message : "Vector layer could not be loaded."); });
    return () => controller.abort();
  }, [isOpen, result?.result_id, artifacts?.vector?.url]);

  useEffect(() => {
    if (!isOpen || !isGeographic || !bounds || !mapContainer.current) return;
    const [west, south, east, north] = bounds;
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: { osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, attribution: "© OpenStreetMap contributors" } },
          layers: [{ id: "osm", type: "raster", source: "osm" }],
        },
        center: [(west + east) / 2, (south + north) / 2],
        zoom: 14,
        attributionControl: false,
      });
    } catch {
      setMapError("The interactive map could not start on this device.");
      return;
    }
    mapInstance.current = map;
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    map.on("load", () => {
      const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [[west, north], [east, north], [east, south], [west, south]];
      map.addSource("satquery-original", { type: "image", url: originalUrl, coordinates });
      map.addLayer({ id: "satquery-original", type: "raster", source: "satquery-original" });
      if (overlayUrl) {
        map.addSource("satquery-overlay", { type: "image", url: overlayUrl, coordinates });
        map.addLayer({ id: "satquery-overlay", type: "raster", source: "satquery-overlay", paint: { "raster-opacity": showOverlay ? 0.8 : 0 } });
      }
      map.fitBounds([[west, south], [east, north]], { padding: 35, duration: 0, maxZoom: 19 });
    });
    const observer = new ResizeObserver(() => map.resize());
    observer.observe(mapContainer.current);
    return () => { observer.disconnect(); map.remove(); if (mapInstance.current === map) mapInstance.current = null; };
    // Visibility changes are handled separately, without rebuilding the map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, result?.result_id, isGeographic, originalUrl, overlayUrl]);

  useEffect(() => {
    const map = mapInstance.current;
    if (map?.getLayer("satquery-overlay")) map.setPaintProperty("satquery-overlay", "raster-opacity", showOverlay ? 0.8 : 0);
  }, [showOverlay, isOpen, collection]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !collection) return;
    const metadata = collection as MapCollection & { properties?: { coordinate_space?: string; crs?: string } };
    if (metadata.properties?.coordinate_space !== "geographic" || metadata.properties?.crs?.toUpperCase() !== "EPSG:4326") return;
    const add = () => {
      if (!map.getSource("satquery-vectors")) {
        const polygonFeatures = collection.features.filter((feature) => feature.geometry?.type === "Polygon" || feature.geometry?.type === "MultiPolygon");
        const colored: MapCollection = { type: "FeatureCollection", features: polygonFeatures.map((feature) => ({ ...feature, properties: { ...feature.properties, _mapColor: featureColor(feature) } })) };
        map.addSource("satquery-vectors", { type: "geojson", data: colored });
        map.addLayer({ id: "satquery-vector-fill", type: "fill", source: "satquery-vectors", paint: { "fill-color": ["get", "_mapColor"], "fill-opacity": 0.14 } });
        map.addLayer({ id: "satquery-vector-line", type: "line", source: "satquery-vectors", paint: { "line-color": ["get", "_mapColor"], "line-width": 1.5, "line-opacity": 0.9 } });
        map.on("click", "satquery-vector-fill", (event) => {
          const feature = event.features?.[0];
          if (!feature) return;
          const props = feature.properties || {};
          const name = String(props.class || props.kind || "Spatial feature").replaceAll("_", " ");
          const area = typeof props.area_m2 === "number" && Number.isFinite(props.area_m2)
            ? ` · ${props.area_m2.toLocaleString(undefined, { maximumFractionDigits: 1 })} m²` : "";
          new maplibregl.Popup().setLngLat(event.lngLat).setText(`${name}${area} · ${event.lngLat.lat.toFixed(6)}°, ${event.lngLat.lng.toFixed(6)}°`).addTo(map);
        });
        map.on("mouseenter", "satquery-vector-fill", () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", "satquery-vector-fill", () => { map.getCanvas().style.cursor = ""; });
      }
      map.setLayoutProperty("satquery-vector-fill", "visibility", showVectors ? "visible" : "none");
      map.setLayoutProperty("satquery-vector-line", "visibility", showVectors ? "visible" : "none");
    };
    if (map.isStyleLoaded()) add(); else map.once("load", add);
    return () => { map.off("load", add); };
  }, [collection, showVectors, isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [isOpen, onClose]);

  if (!isOpen || !result) return null;
  const zoom = (direction: number) => {
    if (isGeographic && mapInstance.current) mapInstance.current.zoomTo(mapInstance.current.getZoom() + direction);
    else setLocalZoom((current) => Math.max(0.5, Math.min(5, current + direction * 0.25)));
  };
  const reset = () => {
    if (isGeographic && mapInstance.current && bounds) mapInstance.current.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], { padding: 35, duration: 300, maxZoom: 19 });
    else { setLocalZoom(1); setLocalOffset({ x: 0, y: 0 }); }
  };
  return <>
    <div className="drawer-backdrop visible" onClick={onClose} />
    <aside className={`satquery-map-panel ${isFullscreen ? "fullscreen" : "split"}`} aria-label="Geospatial Map Workspace">
      <div className="map-panel-header">
        <div className="map-header-left"><MapIcon size={16} className="map-panel-icon" /><div className="map-header-title-wrap"><h3 className="map-panel-title">Geospatial Map Viewer</h3><span className="map-crs-badge">{isGeographic ? "EPSG:4326" : "Image coordinates"}</span></div></div>
        <div className="map-header-controls">
          <button type="button" className="map-tool-btn" onClick={() => setIsFullscreen((value) => !value)} title={isFullscreen ? "Exit fullscreen" : "Fullscreen map"}>{isFullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}</button>
          <button type="button" className="map-tool-btn close" onClick={onClose} title="Close map"><X size={16} /></button>
        </div>
      </div>
      <div className="map-layers-strip">
        <div className="map-layer-options">
          {overlayUrl && <button type="button" className={`map-layer-pill ${showOverlay ? "active" : ""}`} onClick={() => setShowOverlay((value) => !value)} aria-pressed={showOverlay}><Layers size={12} /> Detection overlay</button>}
          {collection && (isGeographic || localShapes.length > 0) && <button type="button" className={`map-layer-pill ${showVectors ? "active" : ""}`} onClick={() => setShowVectors((value) => !value)} aria-pressed={showVectors}><MapPin size={12} /> Footprints ({collection.features.length})</button>}
        </div>
        <div className="map-zoom-tools"><button type="button" className="map-zoom-btn" onClick={() => zoom(-1)} title="Zoom out"><ZoomOut size={13} /></button><button type="button" className="map-zoom-btn" onClick={() => zoom(1)} title="Zoom in"><ZoomIn size={13} /></button><button type="button" className="map-zoom-btn" onClick={reset} title="Reset view"><RotateCcw size={13} /></button></div>
      </div>
      <div className="map-panel-viewport">
        {isGeographic ? <div ref={mapContainer} className="maplibre-viewport" aria-label="Interactive geographic map" /> :
          <div className="map-local-viewport"
            onMouseDown={(event) => { dragOrigin.current = { x: event.clientX - localOffset.x, y: event.clientY - localOffset.y }; }}
            onMouseMove={(event) => { if (dragOrigin.current) setLocalOffset({ x: event.clientX - dragOrigin.current.x, y: event.clientY - dragOrigin.current.y }); }}
            onMouseUp={() => { dragOrigin.current = null; }}
            onMouseLeave={() => { dragOrigin.current = null; }}
            onWheel={(event) => { setLocalZoom((current) => Math.max(0.5, Math.min(5, current + (event.deltaY < 0 ? 0.15 : -0.15)))); }}
          ><div className="map-local-image" style={{ transform: `translate(${localOffset.x}px, ${localOffset.y}px) scale(${localZoom})` }}>
            {originalUrl ? <img src={originalUrl} alt="Original satellite image" draggable={false} /> : <p>No image is available for this result.</p>}
            {showOverlay && overlayUrl && <img className="map-local-overlay" src={overlayUrl} alt="Analysis overlay" draggable={false} />}
            {showVectors && localShapes.length > 0 && result.assets?.[0] && <svg className="map-local-vectors" viewBox={`0 0 ${result.assets[0].width} ${result.assets[0].height}`} aria-label="Detected footprint outlines">
              {localShapes.map((shape) => <path key={shape.id} d={shape.path} fill={shape.color} fillOpacity={0.12} stroke={shape.color} strokeWidth={2} fillRule="evenodd"
                onMouseDown={(event) => event.stopPropagation()} onClick={() => setSelectedLocal(shape.label)} />)}
            </svg>}
          </div><div className="map-local-label">Location metadata is unavailable; showing image coordinates only.</div>
            {selectedLocal && <div className="map-local-feature">{selectedLocal}<button type="button" onClick={() => setSelectedLocal(null)} aria-label="Close feature details"><X size={12} /></button></div>}
          </div>}
        {mapError && <div className="map-error" role="status">{mapError}</div>}
        {bounds && <div className="map-location-badge">{((bounds[1] + bounds[3]) / 2).toFixed(5)}°, {((bounds[0] + bounds[2]) / 2).toFixed(5)}°</div>}
      </div>
    </aside>
  </>;
};
