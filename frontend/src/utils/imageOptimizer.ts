/**
 * Client-side image preprocessor and optimizer.
 * Keeps GeoTIFF bytes intact; the API client uploads large files directly
 * to private storage rather than sending them through the request proxy.
 */

export const MAX_DIRECT_UPLOAD_BYTES = 30 * 1024 * 1024;

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
