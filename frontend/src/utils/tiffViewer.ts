import { fromBlob, fromArrayBuffer } from "geotiff";
import { artifactUrl } from "../api";

const previewCache = new Map<string, string>();

/**
 * Checks if a filename or URL points to a TIFF / GeoTIFF image.
 */
export function isTiffPath(path?: string | null): boolean {
  if (!path) return false;
  return /\.(tif|tiff)($|\?)/i.test(path);
}

/**
 * Returns a web-displayable preview URL for any image path.
 * If path is a .tif/.tiff artifact, maps to the corresponding _preview.png,
 * or fallback to /api/v1/preview?path=...
 */
export function getPreviewUrl(path?: string | null): string {
  if (!path) return "";
  if (path.startsWith("data:") || path.startsWith("blob:")) return path;

  // If already a standard browser-supported image
  if (/\.(png|jpe?g|webp|svg)($|\?)/i.test(path)) {
    return artifactUrl(path) || path;
  }

  // If .tif or .tiff, look for _preview.png or cached data URL
  if (isTiffPath(path)) {
    if (previewCache.has(path)) {
      return previewCache.get(path)!;
    }
    const cleanUrl = path.split("?")[0];
    const previewPath = cleanUrl.replace(/\.(tif|tiff)$/i, "_preview.png");
    return artifactUrl(previewPath) || path;
  }

  return artifactUrl(path) || path;
}

/**
 * Converts a TIFF/GeoTIFF File, Blob, ArrayBuffer, or remote URL into a high-quality PNG data URL
 * via client-side canvas rendering.
 */
export async function convertTiffToDataUrl(
  source: File | Blob | ArrayBuffer | string,
  maxDimension = 1024
): Promise<string> {
  const cacheKey =
    typeof source === "string"
      ? source
      : source instanceof File
      ? `${source.name}-${source.size}-${source.lastModified}`
      : null;

  if (cacheKey && previewCache.has(cacheKey)) {
    return previewCache.get(cacheKey)!;
  }

  try {
    let tiff;
    if (typeof source === "string") {
      const response = await fetch(source);
      if (!response.ok) throw new Error(`HTTP ${response.status} fetching TIFF`);
      const buffer = await response.arrayBuffer();
      tiff = await fromArrayBuffer(buffer);
    } else if (source instanceof Blob || source instanceof File) {
      tiff = await fromBlob(source);
    } else {
      tiff = await fromArrayBuffer(source);
    }

    const image = await tiff.getImage();
    const origWidth = image.getWidth();
    const origHeight = image.getHeight();

    // Scale dimensions while preserving aspect ratio
    const scale = Math.min(1.0, maxDimension / Math.max(origWidth, origHeight));
    const targetWidth = Math.max(1, Math.round(origWidth * scale));
    const targetHeight = Math.max(1, Math.round(origHeight * scale));

    const rasters = await image.readRasters({
      width: targetWidth,
      height: targetHeight,
      resampleMethod: "bilinear",
    });

    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not acquire 2D canvas context");

    const imgData = ctx.createImageData(targetWidth, targetHeight);
    const rgba = imgData.data;
    const totalPixels = targetWidth * targetHeight;

    const numBands = rasters.length;
    const rBand = rasters[0] as ArrayLike<number>;
    const gBand = numBands >= 2 ? (rasters[1] as ArrayLike<number>) : rBand;
    const bBand = numBands >= 3 ? (rasters[2] as ArrayLike<number>) : (numBands >= 2 ? gBand : rBand);

    // Helper for per-band normalization to 0-255
    const normalize = (band: ArrayLike<number>) => {
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < totalPixels; i++) {
        const val = band[i];
        if (Number.isFinite(val)) {
          if (val < min) min = val;
          if (val > max) max = val;
        }
      }
      if (max <= min) return () => 0;
      if (min >= 0 && max <= 255 && band instanceof Uint8Array) {
        return (val: number) => val;
      }
      const range = max - min;
      return (val: number) => {
        if (!Number.isFinite(val)) return 0;
        return Math.min(255, Math.max(0, Math.round(((val - min) / range) * 255)));
      };
    };

    const normR = normalize(rBand);
    const normG = normalize(gBand);
    const normB = normalize(bBand);

    for (let i = 0; i < totalPixels; i++) {
      const offset = i * 4;
      rgba[offset + 0] = normR(rBand[i]);
      rgba[offset + 1] = normG(gBand[i]);
      rgba[offset + 2] = normB(bBand[i]);
      rgba[offset + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);
    const dataUrl = canvas.toDataURL("image/png");

    if (cacheKey) {
      previewCache.set(cacheKey, dataUrl);
    }
    return dataUrl;
  } catch (err) {
    console.warn("TIFF decode failed:", err);
    throw err;
  }
}
