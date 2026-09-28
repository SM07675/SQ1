import React, { useState, useEffect } from "react";
import { Maximize2, Satellite, Layers, Loader2 } from "lucide-react";
import { artifactUrl } from "../../api";
import { isTiffPath, getPreviewUrl, convertTiffToDataUrl } from "../../utils/tiffViewer";

interface TiffPreviewImageProps {
  src: string;
  alt: string;
  previewUrl?: string;
  className?: string;
  onClick?: (displayUrl: string) => void;
  showBadge?: boolean;
}

export const TiffPreviewImage: React.FC<TiffPreviewImageProps> = ({
  src,
  alt,
  previewUrl,
  className = "",
  onClick,
  showBadge = true,
}) => {
  const isTiff = isTiffPath(src) || isTiffPath(alt);
  const [displaySrc, setDisplaySrc] = useState<string>(() => {
    if (previewUrl) return artifactUrl(previewUrl) || previewUrl;
    if (src.startsWith("data:") || src.startsWith("blob:")) {
      return src;
    }
    return getPreviewUrl(src);
  });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [hasError, setHasError] = useState<boolean>(false);

  // If initial src is a raw TIFF blob without previewUrl, decode it client-side
  useEffect(() => {
    let isCancelled = false;

    if (previewUrl) {
      setDisplaySrc(artifactUrl(previewUrl) || previewUrl);
      setHasError(false);
      return;
    }

    if (src.startsWith("data:")) {
      setDisplaySrc(src);
      setHasError(false);
      return;
    }

    if (src.startsWith("blob:") && isTiff) {
      setIsLoading(true);
      convertTiffToDataUrl(src)
        .then((dataUrl) => {
          if (!isCancelled) {
            setDisplaySrc(dataUrl);
            setIsLoading(false);
            setHasError(false);
          }
        })
        .catch(() => {
          if (!isCancelled) {
            setIsLoading(false);
          }
        });
      return () => {
        isCancelled = true;
      };
    }

    // Otherwise use backend preview URL
    const targetUrl = getPreviewUrl(src);
    setDisplaySrc(targetUrl);
    setHasError(false);

    return () => {
      isCancelled = true;
    };
  }, [src, previewUrl, isTiff]);

  const handleImageError = async () => {
    // If backend preview failed or raw TIFF failed, attempt client-side GeoTIFF decoding
    if (isTiff && !displaySrc.startsWith("data:")) {
      try {
        setIsLoading(true);
        // Try resolving the raw image artifact or original URL
        const originalUrl = artifactUrl(src) || src;
        const dataUrl = await convertTiffToDataUrl(originalUrl);
        setDisplaySrc(dataUrl);
        setHasError(false);
        setIsLoading(false);
        return;
      } catch (err) {
        console.warn("Client fallback TIFF render failed:", err);
      }
    }
    setIsLoading(false);
    setHasError(true);
  };

  if (hasError) {
    return (
      <div
        className={`tiff-preview-fallback ${className}`}
        onClick={() => onClick?.(displaySrc)}
        title={`${alt} (Click to open)`}
      >
        <div className="fallback-inner">
          <Satellite size={18} className="fallback-icon" />
          <span className="fallback-name">{alt}</span>
          <span className="fallback-type">GeoTIFF Raster</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`user-attached-image-preview ${className}`}
      onClick={() => onClick?.(displaySrc)}
      title={`Click to inspect ${alt}`}
    >
      {isLoading && (
        <div className="tiff-loading-overlay">
          <Loader2 size={16} className="animate-spin text-cyan-400" />
          <span className="text-[10px] text-cyan-200 mt-1">Rendering GeoTIFF…</span>
        </div>
      )}

      <img
        src={displaySrc}
        alt={alt}
        loading="lazy"
        onLoad={() => setIsLoading(false)}
        onError={handleImageError}
        style={{ opacity: isLoading ? 0.3 : 1, transition: "opacity 0.2s ease" }}
      />

      {showBadge && isTiff && (
        <span className="tiff-corner-badge" title="Georeferenced Multispectral Raster">
          <Layers size={9} />
          <span>GeoTIFF</span>
        </span>
      )}

      <div className="maximize-overlay">
        <Maximize2 size={14} />
      </div>
    </div>
  );
};
