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


import cv2


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
    transform: list[list[float]] | None = None
    matched_keypoints: int = 0
    inlier_ratio: float = 0.0
    fallback_reason: str | None = None


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


def _warp_channels(data: np.ndarray, warp_matrix: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Warp all 2D channels in a (C, H, W) array using an affine transform."""
    out = np.zeros_like(data)
    for ch in range(data.shape[0]):
        out[ch] = cv2.warpAffine(
            data[ch].astype("float32"),
            warp_matrix.astype("float32"),
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return out


def _refine_ecc(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    init_shift_x: float,
    init_shift_y: float,
) -> tuple[np.ndarray | None, bool]:
    """Refine spatial alignment using Enhanced Correlation Coefficient (ECC) maximization."""
    try:
        im1_u8 = (gray_a * 255.0).astype(np.uint8)
        im2_u8 = (gray_b * 255.0).astype(np.uint8)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0, 2] = float(init_shift_x)
        warp_matrix[1, 2] = float(init_shift_y)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-3)
        _, refined = cv2.findTransformECC(im1_u8, im2_u8, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        return refined, True
    except Exception:
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[0, 2] = float(init_shift_x)
            warp_matrix[1, 2] = float(init_shift_y)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-3)
            _, refined = cv2.findTransformECC(im1_u8, im2_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
            return refined, True
        except Exception:
            return None, False


def _register_feature_matching(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
) -> tuple[np.ndarray | None, int, float, str]:
    """Automatic fallback for rotational/affine distortion using SIFT/ORB + RANSAC."""
    im1_u8 = (gray_a * 255.0).astype(np.uint8)
    im2_u8 = (gray_b * 255.0).astype(np.uint8)

    detector_name = "SIFT"
    try:
        sift = cv2.SIFT_create(nfeatures=1500)
        kp1, des1 = sift.detectAndCompute(im1_u8, None)
        kp2, des2 = sift.detectAndCompute(im2_u8, None)
        is_float_desc = True
    except Exception:
        kp1, des1, kp2, des2 = [], None, [], None
        is_float_desc = False

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        detector_name = "ORB"
        try:
            orb = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = orb.detectAndCompute(im1_u8, None)
            kp2, des2 = orb.detectAndCompute(im2_u8, None)
            is_float_desc = False
        except Exception:
            return None, 0, 0.0, "none"

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0, 0.0, detector_name

    try:
        matcher = cv2.BFMatcher(cv2.NORM_L2 if is_float_desc else cv2.NORM_HAMMING)
        matches = matcher.knnMatch(des2, des1, k=2)
        good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
    except Exception:
        return None, 0, 0.0, detector_name

    if len(good) < 4:
        return None, len(good), 0.0, detector_name

    pts_b = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_a = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    try:
        affine_mat, inliers = cv2.estimateAffinePartial2D(
            pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=3.5, maxIters=2000
        )
        if affine_mat is not None and inliers is not None:
            inlier_count = int(inliers.sum())
            inlier_ratio = inlier_count / len(good)
            if inlier_count >= 4 and inlier_ratio >= 0.20:
                return affine_mat, len(good), inlier_ratio, detector_name
    except Exception:
        pass

    return None, len(good), 0.0, detector_name


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
        transform_matrix: list[list[float]] | None = None
        matched_keypoints: int = 0
        inlier_ratio: float = 0.0
        fallback_reason: str | None = None
        shift_x, shift_y = 0.0, 0.0

        # Mode 2: Both have valid CRS -> Use GIS geospatial warping / reproject
        if has_crs_a and has_crs_b and src_a.crs == src_b.crs:
            data_a = src_a.read(
                out_shape=(src_a.count, target_h, target_w),
                resampling=Resampling.bilinear,
            )
            # Reproject B directly into A's coordinate window
            data_b = np.zeros((src_b.count, target_h, target_w), dtype="float32")
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
            alignment_score = compute_alignment_quality(gray_a, gray_b)
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
            data_b_orig = data_b.copy()

            shift_y, shift_x, peak_corr = _compute_phase_correlation_shift(gray_a, gray_b)

            if abs(shift_x) > 0.5 or abs(shift_y) > 0.5:
                for ch in range(data_b.shape[0]):
                    data_b[ch] = _apply_shift(data_b[ch], shift_y, shift_x)
                method = f"phase_correlation_shift (dx={shift_x:.1f}, dy={shift_y:.1f})"
                gray_b = _normalize_luminance(data_b[0])

                # Optional ECC Refinement
                ecc_mat, ecc_ok = _refine_ecc(gray_a, gray_b, 0.0, 0.0)
                if ecc_ok and ecc_mat is not None:
                    refined_b = _warp_channels(data_b, ecc_mat, target_w, target_h)
                    refined_gray_b = _normalize_luminance(refined_b[0])
                    refined_score = compute_alignment_quality(gray_a, refined_gray_b)
                    base_score = compute_alignment_quality(gray_a, gray_b)
                    if refined_score >= base_score:
                        data_b = refined_b
                        gray_b = refined_gray_b
                        method = f"phase_correlation_ecc_refined (dx={shift_x:.1f}, dy={shift_y:.1f})"
                        transform_matrix = ecc_mat.tolist()
            else:
                method = "direct_pixel_alignment"

            # Check alignment quality
            alignment_score = compute_alignment_quality(gray_a, gray_b)

            # If alignment is poor (< 0.65), escalate to SIFT/ORB feature matching + RANSAC
            if alignment_score < 0.65:
                affine_mat, matched_kps, inliers_rat, det_name = _register_feature_matching(gray_a, _normalize_luminance(data_b_orig[0]))
                if affine_mat is not None and matched_kps >= 4:
                    candidate_b = _warp_channels(data_b_orig, affine_mat, target_w, target_h)
                    candidate_gray_b = _normalize_luminance(candidate_b[0])
                    candidate_score = compute_alignment_quality(gray_a, candidate_gray_b)

                    if candidate_score > alignment_score:
                        data_b = candidate_b
                        gray_b = candidate_gray_b
                        method = f"feature_matching_{det_name.lower()}_ransac (matches={matched_kps}, inlier_ratio={inliers_rat:.2f})"
                        fallback_reason = (
                            f"Initial phase correlation yielded low alignment ({alignment_score:.2f}); "
                            f"escalated to {det_name}+RANSAC which achieved alignment score {candidate_score:.2f}."
                        )
                        alignment_score = candidate_score
                        transform_matrix = affine_mat.tolist()
                        matched_keypoints = matched_kps
                        inlier_ratio = inliers_rat
                    else:
                        fallback_reason = (
                            f"Feature matching ({det_name}) found {matched_kps} keypoint matches, "
                            f"but candidate score ({candidate_score:.2f}) did not exceed phase correlation ({alignment_score:.2f})."
                        )
                else:
                    fallback_reason = f"Feature matching ({det_name}) yielded insufficient reliable inliers; preserved phase correlation."

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
        transform=transform_matrix,
        matched_keypoints=matched_keypoints,
        inlier_ratio=round(inlier_ratio, 3),
        fallback_reason=fallback_reason,
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
        "transform": report.transform,
        "matched_keypoints": report.matched_keypoints,
        "inlier_ratio": report.inlier_ratio,
        "fallback_reason": report.fallback_reason,
    }
    (output_dir / "registration_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    return working_a_path, working_b_path, report
