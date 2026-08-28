from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


@dataclass(frozen=True)
class RegistrationReport:
    attempted: bool
    method: str
    alignment_score: float
    status: str
    shift_x: float
    shift_y: float
    original_dims_a: tuple[int, int]
    original_dims_b: tuple[int, int]
    normalized_dims: tuple[int, int]
    message: str


def _normalize_luminance(arr: np.ndarray) -> np.ndarray:
    """Normalize 2D array to [0.0, 1.0] ignoring extreme outliers."""
    arr = arr.astype("float32")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype="float32")
    p2, p98 = np.percentile(finite, [2, 98])
    if p98 <= p2:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)


def _compute_phase_correlation_shift(
    im1: np.ndarray, im2: np.ndarray, max_shift_ratio: float = 0.2
) -> tuple[float, float, float]:
    """Compute translation shift (dy, dx) between two 2D grayscale arrays using FFT phase correlation.

    Returns:
        (shift_y, shift_x, peak_correlation)
    """
    h, w = im1.shape
    if (h, w) != im2.shape or h < 8 or w < 8:
        return 0.0, 0.0, 0.0

    # Apply Hann window to avoid edge artifacts in FFT
    window_y = np.hanning(h)[:, None]
    window_x = np.hanning(w)[None, :]
    window = (window_y * window_x).astype("float32")

    w1 = (im1 - im1.mean()) * window
    w2 = (im2 - im2.mean()) * window

    f1 = np.fft.fft2(w1)
    f2 = np.fft.fft2(w2)

    cross_power = f1 * np.conj(f2)
    norm = np.abs(cross_power)
    norm[norm < 1e-9] = 1e-9
    cross_power /= norm

    corr = np.fft.ifft2(cross_power)
    corr = np.abs(np.fft.fftshift(corr))

    cy, cx = h // 2, w // 2
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    shift_y = float(max_idx[0] - cy)
    shift_x = float(max_idx[1] - cx)
    peak_val = float(corr[max_idx])

    # Reject unreasonable large jumps (beyond max_shift_ratio)
    if abs(shift_y) > h * max_shift_ratio or abs(shift_x) > w * max_shift_ratio:
        return 0.0, 0.0, 0.0

    return shift_y, shift_x, peak_val


def _apply_shift(image_arr: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Shift array (2D or 3D) by integer or rounded displacement and edge pad."""
    sy = int(round(shift_y))
    sx = int(round(shift_x))
    if sy == 0 and sx == 0:
        return image_arr

    shifted = np.zeros_like(image_arr)
    h, w = image_arr.shape[:2]

    # Source slices
    src_y1 = max(0, -sy)
    src_y2 = min(h, h - sy)
    src_x1 = max(0, -sx)
    src_x2 = min(w, w - sx)

    # Destination slices
    dst_y1 = max(0, sy)
    dst_y2 = min(h, h + sy)
    dst_x1 = max(0, sx)
    dst_x2 = min(w, w + sx)

    if dst_y2 > dst_y1 and dst_x2 > dst_x1 and src_y2 > src_y1 and src_x2 > src_x1:
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = image_arr[src_y1:src_y2, src_x1:src_x2]

    return shifted


def compute_alignment_quality(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute alignment score in [0.0, 1.0] using structural and correlation metrics."""
    if im1.shape != im2.shape or im1.size == 0:
        return 0.0

    # Ensure grayscale 2D float [0, 1]
    g1 = _normalize_luminance(im1)
    g2 = _normalize_luminance(im2)

    # 1. Pearson Correlation Coefficient
    mean1, mean2 = float(g1.mean()), float(g2.mean())
    std1, std2 = float(g1.std()), float(g2.std())
    if std1 < 1e-6 or std2 < 1e-6:
        # Flat image
        diff = np.abs(g1 - g2).mean()
        return float(np.clip(1.0 - diff, 0.0, 1.0))

    cov = float(((g1 - mean1) * (g2 - mean2)).mean())
    pearson = cov / (std1 * std2)

    # 2. Mean Absolute Error similarity
    mae = float(np.abs(g1 - g2).mean())
    mae_sim = max(0.0, 1.0 - (mae * 2.0))

    # 3. Structural Gradient overlap
    gy1, gx1 = np.gradient(g1)
    gy2, gx2 = np.gradient(g2)
    grad1 = np.sqrt(gy1**2 + gx1**2)
    grad2 = np.sqrt(gy2**2 + gx2**2)
    g_std1, g_std2 = float(grad1.std()), float(grad2.std())
    if g_std1 > 1e-6 and g_std2 > 1e-6:
        g_cov = float(((grad1 - grad1.mean()) * (grad2 - grad2.mean())).mean())
        grad_sim = max(0.0, g_cov / (g_std1 * g_std2))
    else:
        grad_sim = mae_sim

    # Composite alignment score
    score = 0.45 * max(0.0, pearson) + 0.35 * mae_sim + 0.20 * grad_sim
    return float(np.clip(round(score, 4), 0.0, 1.0))


def normalize_and_register_pair(
    path_a: Path,
    path_b: Path,
    output_dir: Path,
    max_size: int = 1024,
) -> tuple[Path, Path, RegistrationReport]:
    """Inspects, normalizes, and registers a before/after image pair.

    Never modifies original uploaded files. Creates working copies in output_dir.

    Returns:
        (path_working_a, path_working_b, report)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    working_a_path = output_dir / "prepared_1.png"
    working_b_path = output_dir / "prepared_2.png"

    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        dims_a = (src_a.width, src_a.height)
        dims_b = (src_b.width, src_b.height)
        has_crs_a = src_a.crs is not None
        has_crs_b = src_b.crs is not None

        # Determine target normalized dimension based on Primary Image A
        ratio_a = min(1.0, max_size / max(src_a.width, src_a.height))
        target_w = max(16, round(src_a.width * ratio_a))
        target_h = max(16, round(src_a.height * ratio_a))
        normalized_dims = (target_w, target_h)

        # Mode 2: Both have valid CRS -> Use GIS geospatial warping / reproject
        if has_crs_a and has_crs_b and src_a.crs == src_b.crs:
            data_a = src_a.read(
                out_shape=(src_a.count, target_h, target_w),
                resampling=Resampling.bilinear,
            )
            # Reproject B directly into A's coordinate window
            data_b = np.zeros_like(data_a)
            reproject(
                source=rasterio.band(src_b, list(range(1, src_b.count + 1))),
                destination=data_b,
                src_transform=src_b.transform,
                src_crs=src_b.crs,
                dst_transform=src_a.transform * Affine.scale(src_a.width / target_w, src_a.height / target_h),
                dst_crs=src_a.crs,
                resampling=Resampling.bilinear,
            )
            method = "geospatial_crs_reproject"
            shift_x, shift_y = 0.0, 0.0
            gray_a = _normalize_luminance(data_a[0])
            gray_b = _normalize_luminance(data_b[0])
        else:
            # Mode 1: Standard Pixel-Space (PNG/JPG or mixed CRS)
            # Read and resize Image A
            indexes_a = [1, 2, 3] if src_a.count >= 3 else [1]
            data_a = src_a.read(
                indexes_a,
                out_shape=(len(indexes_a), target_h, target_w),
                resampling=Resampling.bilinear,
            )

            # Read and resize Image B to identical target grid
            indexes_b = [1, 2, 3] if src_b.count >= 3 else [1]
            data_b = src_b.read(
                indexes_b,
                out_shape=(len(indexes_b), target_h, target_w),
                resampling=Resampling.bilinear,
            )

            # Perform Phase-Correlation Registration on normalized grayscale
            gray_a = _normalize_luminance(data_a[0])
            gray_b = _normalize_luminance(data_b[0])

            shift_y, shift_x, peak_corr = _compute_phase_correlation_shift(gray_a, gray_b)

            if abs(shift_x) > 0.5 or abs(shift_y) > 0.5:
                # Apply registration shift across all channels of Image B
                for ch in range(data_b.shape[0]):
                    data_b[ch] = _apply_shift(data_b[ch], shift_y, shift_x)
                method = f"phase_correlation_shift (dx={shift_x:.1f}, dy={shift_y:.1f})"
                gray_b = _normalize_luminance(data_b[0])
            else:
                method = "direct_pixel_alignment"

    # Compute genuine alignment quality using already normalized arrays
    alignment_score = compute_alignment_quality(gray_a, gray_b)

    # Determine status
    if alignment_score >= 0.70:
        status = "optimal"
        msg = f"Images successfully normalized and registered ({method}). Alignment score: {alignment_score:.2f}."
    elif alignment_score >= 0.35:
        status = "acceptable"
        msg = f"Images normalized with minor spatial difference ({method}). Alignment score: {alignment_score:.2f}."
    else:
        status = "low_confidence"
        msg = f"Significant scene/perspective divergence detected. Alignment score: {alignment_score:.2f}."

    # Save normalized working image files (RGB PNG)
    def _to_rgb_image(arr: np.ndarray) -> Image.Image:
        if arr.shape[0] >= 3:
            r = (_normalize_luminance(arr[0]) * 255).astype("uint8")
            g = (_normalize_luminance(arr[1]) * 255).astype("uint8")
            b = (_normalize_luminance(arr[2]) * 255).astype("uint8")
            return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
        gray = (_normalize_luminance(arr[0]) * 255).astype("uint8")
        return Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")

    _to_rgb_image(data_a).save(working_a_path, format="PNG")
    _to_rgb_image(data_b).save(working_b_path, format="PNG")

    report = RegistrationReport(
        attempted=True,
        method=method,
        alignment_score=alignment_score,
        status=status,
        shift_x=shift_x,
        shift_y=shift_y,
        original_dims_a=dims_a,
        original_dims_b=dims_b,
        normalized_dims=normalized_dims,
        message=msg,
    )

    # Write registration manifest for auditability
    report_dict = {
        "attempted": report.attempted,
        "method": report.method,
        "alignment_score": report.alignment_score,
        "status": report.status,
        "shift_x": report.shift_x,
        "shift_y": report.shift_y,
        "original_dims_a": list(report.original_dims_a),
        "original_dims_b": list(report.original_dims_b),
        "normalized_dims": list(report.normalized_dims),
        "message": report.message,
    }
    (output_dir / "registration_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    return working_a_path, working_b_path, report
