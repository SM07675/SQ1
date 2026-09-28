/**
 * Client-side image preprocessor and optimizer.
 * Prevents payload rejection by Google Cloud Run (32MB limit)
 * and Vercel edge proxy (4.5MB limit).
 */

export const MAX_SAFE_TIFF_BYTES = 30 * 1024 * 1024; // 30 MB ceiling for Cloud Run direct upload

export function isTiff(file: File): boolean {
  return (
    file.name.toLowerCase().endsWith(".tif") ||
    file.name.toLowerCase().endsWith(".tiff") ||
    file.type === "image/tiff"
  );
}

/**
 * Optimizes standard web images (PNG, JPEG, WebP) if they exceed 10 MB.
 * GeoTIFF files (.tif, .tiff) are preserved as raw binary to preserve geospatial metadata.
 */
export async function optimizeImageIfNeeded(file: File, maxDimension = 3000, quality = 0.92): Promise<File> {
  // Never convert or downsample GeoTIFFs via HTML Canvas (strips georeferencing tags and multispectral bands)
  if (isTiff(file)) {
    if (file.size > MAX_SAFE_TIFF_BYTES) {
      throw new Error(
        `GeoTIFF file is ${(file.size / (1024 * 1024)).toFixed(1)} MB. Google Cloud Run's upload limit is 32 MB. Please provide a cropped tile under 30 MB.`
      );
    }
    return file;
  }

  // If already under 10 MB, keep as-is
  if (file.size <= 10 * 1024 * 1024) {
    return file;
  }

  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      let { width, height } = img;

      if (width > maxDimension || height > maxDimension) {
        if (width > height) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        } else {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(file);
        return;
      }

      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob(
        (blob) => {
          if (!blob || blob.size >= file.size) {
            resolve(file);
            return;
          }
          const optimizedFile = new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), {
            type: "image/jpeg",
            lastModified: Date.now(),
          });
          resolve(optimizedFile);
        },
        "image/jpeg",
        quality
      );
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(file);
    };

    img.src = url;
  });
}
