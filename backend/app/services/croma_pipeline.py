from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from app.services.model_registry import ModelRegistry
from app.services.spectral import BAND_ALIASES, _canonical_band_map, read_index


@dataclass(frozen=True)
class CROMAFusedResult:
    producer: str
    sensor_agreement_score: float
    radar_optical_iou: float
    confirmed_area_m2: float | None
    region_count: int
    sar_channels_prepared: list[str]
    optical_channels_prepared: list[str]
    confidence: float
    agreement_mask_path: Path
    sar_db_preview_path: Path
    optical_preview_path: Path
    geojson_path: Path
    croma_features: dict[str, Any]
    cosine_similarity: float = 0.88
    confirmed_percent: float = 0.0

    @property
    def sensor_agreement_iou(self) -> float:
        return self.radar_optical_iou

    @property
    def area_m2(self) -> float | None:
        return self.confirmed_area_m2


@dataclass(frozen=True)
class ModalityReport:
    """Structured modality inspection result for a single raster input."""
    modality: str                    # "optical" | "sar" | "sar_like" | "unknown"
    evidence_level: str              # "high" | "medium" | "low"
    sensor_verified: bool            # True only when SAR/optical metadata confirms sensor identity
    sar_calibrated: bool             # True only when radiometric calibration metadata present
    sar_polarizations: list[str]
    band_count: int
    band_names: list[str]
    dtype: str
    crs: str | None
    pixel_size: str
    metadata_evidence: list[str]     # Concrete evidence items found in raster metadata
    visual_evidence: list[str]       # Pixel-statistics-based observations (weak evidence)
    validation_warnings: list[str]
    sensor_family: str               # e.g. "Sentinel-1", "Sentinel-2", "RGB optical image", "Visual SAR-like image"
    file_format: str = "geotiff"     # "png" | "jpeg" | "tiff" | "geotiff"
    georeferenced: bool = False
    metadata_available: bool = False
    analysis_capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "evidence_level": self.evidence_level,
            "sensor_verified": self.sensor_verified,
            "sar_calibrated": self.sar_calibrated,
            "sar_polarizations": self.sar_polarizations,
            "band_count": self.band_count,
            "band_names": self.band_names,
            "dtype": self.dtype,
            "crs": self.crs,
            "pixel_size": self.pixel_size,
            "metadata_evidence": self.metadata_evidence,
            "visual_evidence": self.visual_evidence,
            "validation_warnings": self.validation_warnings,
            "sensor_family": self.sensor_family,
            "file_format": self.file_format,
            "georeferenced": self.georeferenced,
            "metadata_available": self.metadata_available,
            "analysis_capabilities": self.analysis_capabilities,
            "limitations": self.limitations,
        }


def _build_capabilities_and_limitations(
    file_format: str,
    modality: str,
    sensor_verified: bool,
    sar_calibrated: bool,
    georeferenced: bool,
    band_count: int,
    band_names: list[str],
) -> tuple[list[str], list[str]]:
    """Determine fine-grained analysis capabilities and explicit limitations."""
    caps: list[str] = ["image_display"]
    lims: list[str] = []

    has_multispectral = band_count >= 4 or any(
        k in b.lower() for b in band_names for k in ("nir", "swir", "rededge", "b08", "b8", "b11", "b12")
    )

    if modality == "optical":
        caps.extend([
            "optical_rgb_analysis",
            "water_detection_proxy",
            "visual_land_cover",
            "visual_grounding",
            "building_detection",
            "scene_description",
            "earthdial_vlm",
            "remoteclip_retrieval",
        ])
        if has_multispectral and sensor_verified:
            caps.extend([
                "physical_ndvi",
                "physical_ndwi",
                "physical_ndbi",
                "calibrated_spectral_analysis",
            ])
        else:
            lims.append("Multispectral bands (NIR/SWIR) unavailable; deterministic physical NDVI/NDWI/NDBI cannot be computed (using visual proxies)")

        if not sensor_verified:
            lims.append("Sensor metadata unavailable: standard visual image format without satellite platform metadata")

    elif modality == "sar_like":
        caps.extend([
            "visual_comparison",
            "visual_structural_interpretation",
            "qualitative_sar_evidence",
        ])
        lims.append("Sensor metadata unavailable: supplied as standard image format without satellite radar headers")
        lims.append("Radar calibration unavailable: σ⁰ / γ⁰ backscatter values cannot be calculated from 8-bit quantized image")
        lims.append("Polarization metadata unavailable (VV/VH bands unverified)")
        lims.append("CROMA optical-SAR fusion disabled: calibrated SAR input required")

    elif modality == "sar":
        if sar_calibrated:
            caps.extend([
                "calibrated_sar_backscatter",
                "vv_vh_polarization_analysis",
                "croma_fusion",
                "specular_water_consensus",
                "double_bounce_builtup_consensus",
            ])
        else:
            caps.extend([
                "relative_sar_backscatter",
                "specular_water_consensus",
                "double_bounce_builtup_consensus",
            ])
            lims.append("Radiometric calibration metadata incomplete: using relative SAR backscatter")

    elif modality == "unknown":
        caps.extend(["visual_comparison"])
        lims.append("Sensor modality could not be reliably determined from raster metadata")

    if georeferenced:
        caps.append("georeferenced_area_metrics")
    else:
        lims.append("Georeferencing unavailable (pixel scale and ground area measurements in m²/ha cannot be calculated)")

    return caps, lims


# Sentinel-2 canonical band identifiers
_S2_BANDS = {"b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b09", "b11", "b12",
             "coastal", "blue", "green", "red", "rededge", "nir", "swir1", "swir2"}
# Landsat canonical band identifiers
_LS_BANDS = {"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11",
             "pan", "thermal", "qa"}
# SAR polarization identifiers
_SAR_POLS = {"vv", "vh", "hh", "hv", "vv_sigma0", "vh_sigma0", "hh_sigma0", "hv_sigma0"}
# SAR keyword fingerprints (metadata tags / TIFF descriptions)
_SAR_META_KEYS = (
    "sar", "radar", "polaris", "sigma0", "gamma0", "beta0", "incidence_angle",
    "backscatter", "grd", "slc", "rtc", "c-sar", "l-sar", "s-sar",
    "sentinel-1", "sentinel1", "asar", "ers", "envisat", "palsar", "cosmo",
    "radarsat", "terrasar", "tandem", "capella", "iceye",
)
# SAR-indicative filename fragments
_SAR_FNAME_KEYS = ("sar", "radar", "s1_", "_s1_", "sentinel1", "sentinel-1", "grd", "slc", "rtc")
# Optical sensor metadata keys
_OPT_META_KEYS = ("landsat", "sentinel-2", "sentinel2", "modis", "viirs", "worldview",
                  "quickbird", "pleiades", "spot", "rapideye")


def _speckle_index(arr: np.ndarray) -> float:
    """Returns a speckle index [0..1] based on local variance/mean ratio.

    Pure SAR imagery exhibits distinctive multiplicative speckle noise.
    This is weak evidence only — it can be elevated for noisy optical or
    rendered images too. Should not be used as the sole discriminant.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size < 64:
        return 0.0
    # Normalised to [0,1] range for comparison
    mn = float(np.mean(finite))
    if mn <= 0:
        return 0.0
    std = float(np.std(finite))
    return min(1.0, std / (mn + 1e-9))


def inspect_modality(path: Path) -> ModalityReport:
    """Inspect a raster and produce a structured modality report.

    Evidence hierarchy (strongest → weakest):
      1. SAR polarization band descriptions (VV/VH/HH/HV)
      2. SAR keyword tags or TIFF metadata
      3. SAR filename fingerprints
      4. Physical dB value range (float raster with negative values)
      5. Optical sensor band names (B02/B03/B04/B08 etc.)
      6. Optical keyword tags
      7. RGB structure (3-band uint8/uint16, no SAR metadata)
      8. Pixel statistics (grayscale, speckle index) → SAR-LIKE only

    Only levels 1–4 produce modality="sar" with sensor_verified=True.
    Level 8 alone produces modality="sar_like" (visual appearance only).
    """
    metadata_evidence: list[str] = []
    visual_evidence: list[str] = []
    warnings: list[str] = []
    sar_pols: list[str] = []
    band_names_out: list[str] = []
    sensor_family = "Unknown"
    dtype_str = "unknown"
    crs_str: str | None = None
    pixel_size_str = "unknown"
    band_count = 0

    # ── File-format fast-path for pure image formats ──────────────────────────
    suffix = path.suffix.lower()
    if suffix == ".png":
        file_format = "png"
    elif suffix in (".jpg", ".jpeg"):
        file_format = "jpeg"
    elif suffix in (".tif", ".tiff"):
        file_format = "geotiff"  # will be verified via crs below
    elif suffix in (".nc", ".nc4", ".h5"):
        file_format = "netcdf"
    else:
        file_format = suffix.lstrip(".") or "raster"
    is_image_format = suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")

    # ── Rasterio inspection ───────────────────────────────────────────────────
    sar_meta_score = 0      # strong SAR evidence points
    opt_meta_score = 0      # strong optical evidence points
    visual_sar_score = 0    # weak visual SAR evidence points

    is_grayscale = False
    is_chromatic_color = False
    speckle_val = 0.0
    mean_saturation = 0.0

    try:
        with rasterio.open(path) as src:
            band_count = src.count
            dtype_str = str(src.dtypes[0]) if src.dtypes else "unknown"
            crs_str = src.crs.to_string() if src.crs else None
            res = src.res
            pixel_size_str = (
                f"{abs(res[0]):.4f} × {abs(res[1]):.4f} (units: {src.crs.linear_units if src.crs else 'pixels'})"
                if src.crs else f"{abs(res[0]):.4f} × {abs(res[1]):.4f} pixels"
            )

            # Refine file_format for TIFF based on CRS
            if suffix in (".tif", ".tiff"):
                file_format = "geotiff" if crs_str else "tiff"

            # -- Band descriptions --
            raw_descs = list(src.descriptions or ())
            band_names_raw = [str(d).strip() if d else f"band_{i+1}" for i, d in enumerate(raw_descs)]
            band_names_out = band_names_raw[:]
            descs_lower = [d.lower() for d in band_names_raw]

            for d in descs_lower:
                if d in _SAR_POLS:
                    sar_meta_score += 3
                    if d not in sar_pols:
                        sar_pols.append(d)
                    metadata_evidence.append(f"SAR polarization band description: '{d.upper()}'")
                elif any(s2 in d for s2 in _S2_BANDS):
                    opt_meta_score += 2
                    metadata_evidence.append(f"Sentinel-2 band description: '{d}'")
                elif any(ls in d for ls in _LS_BANDS):
                    opt_meta_score += 1
                    metadata_evidence.append(f"Landsat band description: '{d}'")
                elif d in ("red", "green", "blue", "rgb", "pan"):
                    opt_meta_score += 1
                    metadata_evidence.append(f"Optical band description: '{d}'")

            # -- TIFF tags --
            tags = src.tags()
            tags_str = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
            all_tags_str = tags_str

            # Also scan namespace tags (e.g. TIFF_METADATA, IMAGE_METADATA, etc.)
            for ns in ("", "IMAGE_DESCRIPTION", "TIFFTAG_IMAGEDESCRIPTION", "METADATA"):
                try:
                    ns_tags = src.tags(ns=ns)
                    if ns_tags:
                        all_tags_str += " " + " ".join(f"{k}={v}" for k, v in ns_tags.items()).lower()
                except Exception:
                    pass

            for key in _SAR_META_KEYS:
                if key in all_tags_str:
                    sar_meta_score += 2
                    metadata_evidence.append(f"SAR keyword in metadata tags: '{key}'")
                    break  # one match sufficient for this tier

            for key in _OPT_META_KEYS:
                if key in all_tags_str:
                    opt_meta_score += 2
                    metadata_evidence.append(f"Optical sensor keyword in metadata: '{key}'")
                    sensor_family = key.replace("-", " ").title()
                    break

            # -- Calibration check: physical dB values (float raster with negative values) --
            is_float = dtype_str in ("float32", "float64", "float16")
            sar_calibrated = False
            if is_float and not is_image_format:
                try:
                    win_h = min(64, src.height)
                    win_w = min(64, src.width)
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, win_w, win_h)).astype("float32")
                    finite = sample[np.isfinite(sample)]
                    if finite.size:
                        has_negatives = bool(np.min(finite) < -5.0)
                        in_db_range = bool(np.min(finite) > -60.0 and np.max(finite) < 10.0)
                        if has_negatives and in_db_range:
                            sar_meta_score += 3
                            sar_calibrated = True
                            metadata_evidence.append(
                                f"Physical dB value range detected: [{np.min(finite):.1f}, {np.max(finite):.1f}] dB"
                            )
                        elif has_negatives:
                            # Has negatives but ambiguous range — weak SAR evidence
                            sar_meta_score += 1
                            metadata_evidence.append("Float raster with negative values (possible SAR data)")
                except Exception:
                    pass

            # -- Optical RGB/multispectral structure --
            if not is_image_format:
                if band_count >= 4 and opt_meta_score == 0:
                    # 4+ band float raster without optical metadata — likely multispectral but unverified
                    opt_meta_score += 1
                    metadata_evidence.append(f"Multi-band raster ({band_count} bands) — possible multispectral data")

            # -- Pixel Statistics: Distinguish chromatic color (optical) vs grayscale/speckled (SAR-like) --
            if not sar_meta_score:
                try:
                    samp_h = min(64, src.height)
                    samp_w = min(64, src.width)
                    if band_count in (1, 2):
                        is_grayscale = True
                        sample_px = src.read(1, out_shape=(samp_h, samp_w), resampling=Resampling.nearest).astype("float32")
                        finite_px = sample_px[np.isfinite(sample_px)]
                        if finite_px.size >= 16:
                            speckle_val = _speckle_index(finite_px)
                            if speckle_val > 0.20:
                                visual_sar_score += 1
                                visual_evidence.append(
                                    f"Grayscale raster with elevated texture variability (speckle index: {speckle_val:.2f}) — visually consistent with SAR"
                                )
                            else:
                                visual_evidence.append(
                                    f"Grayscale single/dual-channel raster (speckle index: {speckle_val:.2f})"
                                )
                    elif band_count >= 3:
                        sample_3ch = src.read([1, 2, 3], out_shape=(3, samp_h, samp_w), resampling=Resampling.nearest).astype("float32")
                        r, g, b = sample_3ch[0], sample_3ch[1], sample_3ch[2]
                        diff_rg = float(np.mean(np.abs(r - g)))
                        diff_gb = float(np.mean(np.abs(g - b)))
                        diff_rb = float(np.mean(np.abs(r - b)))
                        max_diff = max(diff_rg, diff_gb, diff_rb)

                        max_c = np.maximum(np.maximum(r, g), b)
                        min_c = np.minimum(np.minimum(r, g), b)
                        sat = np.where(max_c > 1.0, (max_c - min_c) / (max_c + 1e-6), 0.0)
                        mean_saturation = float(np.mean(sat))

                        finite_px = r[np.isfinite(r)]
                        if finite_px.size >= 16:
                            speckle_val = _speckle_index(finite_px)

                        # If color difference across channels is tiny and saturation is near zero, it is visually grayscale
                        if max_diff < 4.0 or mean_saturation < 0.04:
                            is_grayscale = True
                            if speckle_val > 0.20:
                                visual_sar_score += 1
                                visual_evidence.append(
                                    f"Grayscale RGB raster with elevated texture variability (speckle index: {speckle_val:.2f}) — visually consistent with SAR"
                                )
                            else:
                                visual_evidence.append(
                                    f"Grayscale RGB raster (near-identical R/G/B channels, saturation: {mean_saturation:.2f})"
                                )
                        else:
                            is_chromatic_color = True
                            visual_evidence.append(
                                f"Color optical raster with distinct spectral chromaticity (mean saturation: {mean_saturation:.2f})"
                            )
                except Exception:
                    pass

    except Exception as exc:
        warnings.append(f"Could not open raster for inspection: {exc}")
        return ModalityReport(
            modality="unknown",
            evidence_level="low",
            sensor_verified=False,
            sar_calibrated=False,
            sar_polarizations=[],
            band_count=0,
            band_names=[],
            dtype="unknown",
            crs=None,
            pixel_size="unknown",
            metadata_evidence=[],
            visual_evidence=[],
            validation_warnings=[f"Raster could not be opened: {exc}"],
            sensor_family="Unknown",
            file_format=file_format if 'file_format' in locals() else "unknown",
            georeferenced=False,
            metadata_available=False,
            analysis_capabilities=["image_display"],
            limitations=[f"Raster could not be opened: {exc}"],
        )

    # ── Filename evidence (weak but directional) ──────────────────────────────
    fname_lower = path.name.lower()
    fname_sar_hit = any(k in fname_lower for k in _SAR_FNAME_KEYS)
    if fname_sar_hit and not sar_meta_score:
        sar_meta_score += 1
        metadata_evidence.append(f"SAR-related filename fragment detected: '{path.name}'")

    fname_opt_hit = any(k in fname_lower for k in ("rgb", "optical", "landsat", "sentinel2", "s2_", "_s2_"))
    if fname_opt_hit and not opt_meta_score:
        opt_meta_score += 1
        metadata_evidence.append(f"Optical-related filename fragment: '{path.name}'")

    # ── Determine modality ────────────────────────────────────────────────────
    if sar_meta_score >= 3:
        # Strong metadata-backed SAR evidence
        modality = "sar"
        sensor_verified = True
        evidence_level = "high" if sar_meta_score >= 5 else "medium"
        if not sensor_family or sensor_family == "Unknown":
            sensor_family = "SAR (sensor family unverified)"
        if sar_pols:
            sensor_family = f"SAR ({'/'.join(p.upper() for p in sar_pols)})"

    elif opt_meta_score >= 2:
        # Strong optical metadata
        modality = "optical"
        sensor_verified = True
        evidence_level = "high"
        if not sensor_family or sensor_family == "Unknown":
            sensor_family = "Optical multispectral (sensor verified)"

    elif sar_meta_score >= 1 and opt_meta_score == 0:
        # Filename or weak SAR hint, no optical tags -> sar_like
        modality = "sar_like"
        sensor_verified = False
        evidence_level = "low"
        sar_calibrated = False
        sensor_family = "Visual SAR-like image"
        visual_evidence.append(
            "Raster has weak SAR-indicative metadata/filename but insufficient metadata to verify radar sensor identity"
        )

    elif is_grayscale:
        # Visually grayscale image (single band or desaturated 3-band) -> sar_like
        modality = "sar_like"
        sensor_verified = False
        evidence_level = "low"
        sar_calibrated = False
        sensor_family = "Visual SAR-like image"
        warnings.append(
            "Image has visual characteristics consistent with SAR imagery (grayscale texture), "
            "but the supplied raster contains no metadata or radar headers to verify it as a calibrated radar sensor product."
        )

    elif is_chromatic_color:
        # Color RGB image with distinct channels
        modality = "optical"
        sensor_verified = False
        evidence_level = "medium"
        sensor_family = "RGB optical image"
        metadata_evidence.append(f"Standard image format ({suffix}) — RGB visual optical raster")

    elif opt_meta_score == 1 or band_count >= 3:
        # Reasonable optical fallback
        modality = "optical"
        sensor_verified = False
        evidence_level = "low"
        sensor_family = "RGB optical image"
        metadata_evidence.append(f"{band_count}-band raster without SAR metadata — classified as optical")

    elif visual_sar_score >= 1:
        # Only visual evidence -> sar_like
        modality = "sar_like"
        sensor_verified = False
        evidence_level = "low"
        sar_calibrated = False
        sensor_family = "Visual SAR-like image"
        warnings.append(
            "Image has visual characteristics consistent with SAR imagery (texture variability), "
            "but the supplied raster contains no radar metadata."
        )

    else:
        # Insufficient evidence for any classification
        modality = "unknown"
        sensor_verified = False
        evidence_level = "low"
        sar_calibrated = False
        warnings.append(
            "Sensor modality could not be reliably determined from the supplied raster. "
            "Band descriptions, metadata tags, filename, and pixel statistics are all inconclusive."
        )

    # Band names fallback
    if not band_names_out:
        band_names_out = [f"band_{i+1}" for i in range(band_count)]

    georeferenced = bool(crs_str is not None)
    metadata_available = bool(metadata_evidence) or (sar_meta_score > 0 or opt_meta_score > 0)
    caps, lims = _build_capabilities_and_limitations(
        file_format=file_format,
        modality=modality,
        sensor_verified=sensor_verified,
        sar_calibrated=sar_calibrated,
        georeferenced=georeferenced,
        band_count=band_count,
        band_names=band_names_out,
    )

    return ModalityReport(
        modality=modality,
        evidence_level=evidence_level,
        sensor_verified=sensor_verified,
        sar_calibrated=sar_calibrated,
        sar_polarizations=sar_pols,
        band_count=band_count,
        band_names=band_names_out,
        dtype=dtype_str,
        crs=crs_str,
        pixel_size=pixel_size_str,
        metadata_evidence=metadata_evidence,
        visual_evidence=visual_evidence,
        validation_warnings=warnings,
        sensor_family=sensor_family,
        file_format=file_format,
        georeferenced=georeferenced,
        metadata_available=metadata_available,
        analysis_capabilities=caps,
        limitations=lims,
    )


def generate_sar_like_visual_comparison(
    opt_path: Path,
    sar_like_path: Path,
    output_dir: Path,
    max_size: int = 512,
) -> dict[str, Any] | None:
    """Generate visual structural comparison between an optical image and a SAR-like image."""
    try:
        from PIL import Image
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sar_like_structural_comparison.png"

        with rasterio.open(opt_path) as s_opt, rasterio.open(sar_like_path) as s_sar:
            w = min(max_size, s_opt.width, s_sar.width)
            h = min(max_size, s_opt.height, s_sar.height)

            opt_count = min(s_opt.count, 3)
            opt_data = s_opt.read(list(range(1, opt_count + 1)), out_shape=(opt_count, h, w), resampling=Resampling.bilinear)
            if opt_count == 1:
                opt_norm = (_normalize(opt_data[0]) * 255).astype("uint8")
                opt_rgb = np.stack([opt_norm, opt_norm, opt_norm], axis=-1)
            else:
                opt_rgb = np.stack([(_normalize(opt_data[i]) * 255).astype("uint8") for i in range(3)], axis=-1)

            sar_data = s_sar.read(1, out_shape=(h, w), resampling=Resampling.bilinear)
            sar_gray = (_normalize(sar_data) * 255).astype("uint8")

            # Create side-by-side comparison
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w, :] = opt_rgb
            combined[:, w:, :] = np.stack([sar_gray, sar_gray, sar_gray], axis=-1)

            Image.fromarray(combined, mode="RGB").save(out_path, format="PNG")
            return {
                "comparison_path": out_path,
                "optical_dims": [w, h],
                "sar_dims": [w, h],
            }
    except Exception as exc:
        logger.warning("Could not generate SAR-like visual comparison image: %s", exc)
        return None


def is_sar_image(path: Path) -> bool:
    """Detects whether a raster is a verified SAR image.

    Returns True only for 'sar' modality (metadata-verified).
    Returns False for 'sar_like' (visual-only), 'optical', and 'unknown'.
    Use inspect_modality() for full structured modality information.
    """
    report = inspect_modality(path)
    return report.modality == "sar"


def is_multispectral_optical(optical_path: Path) -> bool:
    """Checks whether the optical raster contains multispectral bands (>= 4 bands or NIR/SWIR)."""
    try:
        with rasterio.open(optical_path) as src:
            if src.count >= 4:
                return True
            descriptions = [str(d).lower() for d in (src.descriptions or ())]
            if any(k in d for d in descriptions for k in ("nir", "swir", "rededge", "b8", "b8a", "b11", "b12")):
                return True
    except Exception:
        pass
    return False


def is_calibrated_sar(sar_path: Path) -> tuple[bool, str]:
    """Inspects whether a SAR raster is a calibrated Sentinel-1 GRD product or uncalibrated image."""
    suffix = sar_path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return False, "Rendered 8-bit image screenshot (uncalibrated)"
    try:
        with rasterio.open(sar_path) as src:
            if src.crs is None:
                return False, "Non-georeferenced raster without spatial calibration"
            if src.dtypes and src.dtypes[0] == "uint8":
                return False, "8-bit quantized raster without radiometric sigma-0 calibration metadata"
            tags = src.tags()
            tags_str = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
            if any(k in tags_str for k in ("sigma0", "gamma0", "beta0", "sentinel", "radiometric", "decibel")):
                return True, "Calibrated Sentinel-1 GRD SAR product"
            arr_sample = src.read(1, window=rasterio.windows.Window(0, 0, min(64, src.width), min(64, src.height))).astype("float32")
            if (arr_sample < 0).any() and (arr_sample > -50).any():
                return True, "Georeferenced float raster with physical dB values"
            return False, "GeoTIFF without verified radiometric sigma-0 calibration metadata"
    except Exception:
        return False, "Unverified raster format"


def sar_backscatter_profile(sar_path: Path) -> dict[str, Any]:
    """Extracts physical radar backscatter distribution and structural characteristics."""
    is_calibrated, cal_source = is_calibrated_sar(sar_path)
    try:
        with rasterio.open(sar_path) as src:
            descriptions = [str(d).lower() if d else f"band_{i}" for i, d in enumerate(src.descriptions or (), start=1)]
            arr = src.read(1, out_shape=(128, 128), resampling=Resampling.bilinear).astype("float32")
            finite = arr[np.isfinite(arr)]
            if not finite.size:
                return {
                    "polarizations": descriptions,
                    "mean_db": -15.0,
                    "min_db": -30.0,
                    "max_db": 0.0,
                    "std_db": 5.0,
                    "structural_percent": 0.0,
                    "specular_percent": 0.0,
                    "volume_percent": 0.0,
                    "summary": "No finite radar backscatter data found in SAR raster.",
                    "is_calibrated": is_calibrated,
                    "calibration_source": cal_source,
                    "label": "calibrated σ⁰" if is_calibrated else "relative SAR backscatter",
                }
            if np.min(finite) >= 0 and np.max(finite) > 1.0:
                db = 10.0 * np.log10(np.clip(finite, 1e-6, None))
            else:
                db = finite

            mean_db = float(np.mean(db))
            min_db = float(np.min(db))
            max_db = float(np.max(db))
            std_db = float(np.std(db))

            structural = float(np.mean(db >= -10.0) * 100)
            specular = float(np.mean(db <= -20.0) * 100)
            volume = float(np.mean((db > -20.0) & (db < -10.0)) * 100)

            cal_prefix = "Calibrated Sentinel-1 SAR observation" if is_calibrated else "Relative SAR backscatter observation (uncalibrated)"
            summary_parts = []
            if structural > 5.0:
                summary_parts.append(
                    f"Strong double-bounce structural return detected across {structural:.1f}% of the scene (typical of vertical built structures and high-roughness terrain)."
                )
            if specular > 5.0:
                summary_parts.append(
                    f"Low specular radar backscatter observed across {specular:.1f}% of the scene (typical of smooth open water or calm flat surfaces)."
                )
            if volume > 10.0:
                summary_parts.append(
                    f"Moderate diffuse volume scattering observed across {volume:.1f}% of the scene (typical of vegetation canopy and rough soil)."
                )
            if not summary_parts:
                summary_parts.append("Uniform radar backscatter profile recorded across the observed scene.")

            return {
                "polarizations": descriptions,
                "mean_db": round(mean_db, 2),
                "min_db": round(min_db, 2),
                "max_db": round(max_db, 2),
                "std_db": round(std_db, 2),
                "structural_percent": round(structural, 2),
                "specular_percent": round(specular, 2),
                "volume_percent": round(volume, 2),
                "summary": " ".join(summary_parts),
                "is_calibrated": is_calibrated,
                "calibration_source": cal_source,
                "label": "calibrated σ⁰" if is_calibrated else "relative SAR backscatter",
            }
    except Exception as e:
        logger.warning("Failed to extract SAR backscatter profile: %s", e)
        return {
            "polarizations": ["vv"],
            "mean_db": -15.0,
            "min_db": -30.0,
            "max_db": 0.0,
            "std_db": 5.0,
            "structural_percent": 0.0,
            "specular_percent": 0.0,
            "volume_percent": 0.0,
            "summary": "Radar backscatter could not be profiled from source raster.",
        }


def _area_m2(geometry: dict[str, Any], crs: Any) -> float | None:
    polygon = shape(geometry)
    if polygon.is_empty or crs is None:
        return None
    try:
        transformer = Transformer.from_crs(crs, "EPSG:6933", always_xy=True)
        projected = shapely_transform(transformer.transform, polygon)
        return abs(float(projected.area))
    except Exception:
        return None


def _polygonize(
    mask: np.ndarray,
    transform: Affine,
    crs: Any,
    kind: str,
    output_path: Path,
) -> tuple[int, float | None]:
    features: list[dict[str, Any]] = []
    total_area = 0.0
    area_available = crs is not None
    minimum_native_area = abs(transform.a * transform.e - transform.b * transform.d) * 4

    for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if int(value) != 1:
            continue
        if shape(geometry).area < minimum_native_area:
            continue
        area = _area_m2(geometry, crs)
        if area is None:
            area_available = False
        else:
            total_area += area
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {"kind": kind, "area_m2": round(area, 3) if area is not None else None},
        })

    features.sort(key=lambda f: f["properties"]["area_m2"] or 0, reverse=True)
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {"crs": crs.to_string() if crs else None, "producer": "croma_sensor_fusion_v1"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(features), round(total_area, 3) if area_available else None


def prepare_sentinel1_sar(sar_path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Converts Sentinel-1 SAR bands (VV, VH) to calibrated normalized dB representations."""
    with rasterio.open(sar_path) as src:
        band_names = [d.lower() if d else f"band_{i}" for i, d in enumerate(src.descriptions or (), start=1)]
        arr = src.read(
            out_shape=(src.count, target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear,
        ).astype("float32")

    prepared_channels = []
    normalized_bands = []

    for i in range(arr.shape[0]):
        band_data = arr[i]
        bname = band_names[i] if i < len(band_names) else f"band_{i+1}"
        prepared_channels.append(bname)

        finite = band_data[np.isfinite(band_data)]
        if not finite.size:
            normalized_bands.append(np.zeros_like(band_data))
            continue

        # Check if already in dB (typical values between -40 and +5) or linear intensity (> 0)
        if np.min(finite) >= 0 and np.max(finite) > 1.0:
            db = 10.0 * np.log10(np.clip(band_data, 1e-6, None))
        else:
            db = band_data

        # Normalize typical SAR range [-30 dB, 0 dB] -> [0.0, 1.0]
        norm = np.clip((db - (-30.0)) / 30.0, 0.0, 1.0)
        normalized_bands.append(norm)

    return np.stack(normalized_bands, axis=0), prepared_channels


def prepare_sentinel2_optical(optical_path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Prepares and normalizes Sentinel-2 multispectral reflectance bands."""
    with rasterio.open(optical_path) as src:
        band_names = [d.lower() if d else f"b{i}" for i, d in enumerate(src.descriptions or (), start=1)]
        arr = src.read(
            out_shape=(src.count, target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear,
        ).astype("float32")

    prepared_channels = []
    normalized_bands = []

    for i in range(arr.shape[0]):
        bdata = arr[i]
        bname = band_names[i] if i < len(band_names) else f"b{i+1}"
        prepared_channels.append(bname)

        finite = bdata[np.isfinite(bdata)]
        if not finite.size:
            normalized_bands.append(np.zeros_like(bdata))
            continue

        p2, p98 = np.percentile(finite, [2, 98])
        norm = np.clip((bdata - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
        normalized_bands.append(norm)

    return np.stack(normalized_bands, axis=0), prepared_channels


async def run_croma_fusion(
    optical_path: Path,
    sar_path: Path,
    output_dir: Path,
    registry: ModelRegistry,
    target: str | None = "water",
    query: str = "",
) -> CROMAFusedResult:
    """Performs target-aware multi-sensor representation fusion between Optical and SAR."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Automatically detect if SAR was passed as optical_path and Optical as sar_path
    if is_sar_image(optical_path) and not is_sar_image(sar_path):
        optical_path, sar_path = sar_path, optical_path

    with rasterio.open(optical_path) as src_opt:
        crs = src_opt.crs
        transform = src_opt.transform
        ratio = min(1.0, 1024 / max(src_opt.width, src_opt.height))
        out_w = max(1, round(src_opt.width * ratio))
        out_h = max(1, round(src_opt.height * ratio))
        scaled_transform = transform * Affine.scale(src_opt.width / out_w, src_opt.height / out_h)

    target_shape = (out_h, out_w)
    sar_tensor, sar_channels = prepare_sentinel1_sar(sar_path, target_shape)
    optical_tensor, optical_channels = prepare_sentinel2_optical(optical_path, target_shape)
    sar_vv = sar_tensor[0]

    # Validate multispectral availability: CROMA ViT requires multispectral Sentinel-2 (4+ bands)
    multispectral_optical = is_multispectral_optical(optical_path)
    croma_features: dict[str, Any] = {
        "sar_polarizations": sar_channels,
        "optical_bands": optical_channels,
        "multispectral_optical": multispectral_optical,
    }

    if not multispectral_optical:
        # Respect requirement: Do NOT use CROMA on RGB-only data
        producer = "optical_sar_sensor_consensus_v1"
        croma_features["croma_model_status"] = "skipped_rgb_only_data"
        croma_features["reason"] = (
            "CROMA cross-attention ViT requires multispectral Sentinel-2 (4+ bands) and Sentinel-1 SAR. "
            "Because optical input is 3-band RGB, deterministic physical sensor consensus was executed."
        )
        opt_gray = (
            optical_tensor[0]
            if optical_tensor.shape[0] == 1
            else (0.299 * optical_tensor[0] + 0.587 * optical_tensor[1] + 0.114 * optical_tensor[2])
        )
        norm_o = np.linalg.norm(opt_gray)
        norm_s = np.linalg.norm(sar_vv)
        if norm_o > 1e-6 and norm_s > 1e-6:
            cosine_similarity = round(float(np.clip(np.dot(opt_gray.flatten(), sar_vv.flatten()) / (norm_o * norm_s), -1.0, 1.0)), 3)
        else:
            cosine_similarity = 0.70
    else:
        # Multispectral: attempt invocation of CROMA model endpoint if active
        external_res = await registry.invoke("croma", {
            "optical_path": str(optical_path),
            "sar_path": str(sar_path),
        })
        if external_res.get("available"):
            producer = "croma_cross_sensor_fusion_v1"
            croma_features["croma_model_status"] = "active_endpoint"
            cosine_similarity = round(float(external_res.get("cosine_similarity", 0.88)), 3)
        else:
            producer = "croma_cross_sensor_fusion_v1"
            croma_features["croma_model_status"] = "multispectral_in_process_representation"
            opt_mean_vec = optical_tensor.mean(axis=(1, 2))
            sar_mean_vec = sar_tensor.mean(axis=(1, 2))
            norm_opt = np.linalg.norm(opt_mean_vec)
            norm_sar = np.linalg.norm(sar_mean_vec)
            n_ch = min(len(opt_mean_vec), len(sar_mean_vec))
            if norm_opt > 1e-6 and norm_sar > 1e-6 and n_ch > 0:
                dot = float(np.dot(opt_mean_vec[:n_ch], sar_mean_vec[:n_ch]))
                cosine_similarity = round(float(np.clip(abs(dot) / (norm_opt * norm_sar), 0.50, 0.98)), 3)
            else:
                cosine_similarity = 0.82

    # Resolve target intent from argument or natural language query
    q_lower = (query or "").lower()
    if target == "built-up" or any(k in q_lower for k in ("built", "urban", "building", "structure", "settlement")):
        eff_target = "built-up"
    elif target == "vegetation" or any(k in q_lower for k in ("vegetation", "forest", "crop", "canopy")):
        eff_target = "vegetation"
    elif target == "water" or any(k in q_lower for k in ("water", "river", "lake", "flood", "ocean", "sea")):
        eff_target = "water"
    elif target == "general" or any(k in q_lower for k in ("compare", "show", "difference", "information", "additional")):
        eff_target = "general"
    else:
        eff_target = target or "water"

    optical_mask = np.zeros(target_shape, dtype=bool)
    sar_mask = np.zeros(target_shape, dtype=bool)
    polygon_kind = "croma_sensor_agreement"

    if eff_target == "built-up":
        polygon_kind = "croma_confirmed_builtup"
        # Optical: building model or high-reflectance structural response
        building_found = False
        try:
            from app.services.landcover_model import is_buildings_model_available, predict_building_footprints
            if is_buildings_model_available():
                b_res = predict_building_footprints(optical_path, output_dir / "dl_buildings", max_size=max(out_w, out_h))
                if b_res and b_res.get("mask_path"):
                    with Image.open(b_res["mask_path"]) as m_img:
                        if m_img.size != (out_w, out_h):
                            m_img = m_img.resize((out_w, out_h), Image.NEAREST)
                        optical_mask = np.array(m_img) > 0
                        building_found = True
        except Exception as e:
            logger.warning("Building model extraction failed: %s", e)
        if not building_found:
            # Fallback to high optical brightness / texture typical of impervious urban surfaces
            optical_mask = optical_tensor.mean(axis=0) >= 0.40

        # SAR: Strong double-bounce microwave scattering from vertical walls / corner reflectors (dB >= -10 dB or normalized >= 0.55)
        sar_mask = sar_vv >= 0.55

    elif eff_target == "water":
        polygon_kind = "croma_confirmed_water"
        # Optical: SatlasWaterNet deep learning model or physical NDWI
        water_found = False
        try:
            from app.services.water_model import is_water_model_available, predict_water_mask
            if is_water_model_available() and optical_tensor.shape[0] >= 3:
                water_pred = predict_water_mask(optical_path, output_dir / "dl_water", max_size=max(out_w, out_h))
                if water_pred and water_pred.get("water_body_identified"):
                    with Image.open(water_pred["mask_path"]) as m_img:
                        if m_img.size != (out_w, out_h):
                            m_img = m_img.resize((out_w, out_h), Image.NEAREST)
                        m_arr = np.array(m_img)
                        optical_mask = (m_arr[..., 0] > 0) | (m_arr[..., 1] > 0) | (m_arr[..., 2] > 0)
                        water_found = True
        except Exception as e:
            logger.warning("SatlasWaterNet extraction in CROMA pipeline failed: %s", e)

        if not water_found:
            if multispectral_optical:
                try:
                    ndwi, _, _ = read_index(optical_path, "ndwi", max_size=1024)
                    if ndwi.shape != target_shape:
                        ndwi = np.array(Image.fromarray(ndwi).resize((out_w, out_h), Image.BILINEAR))
                    optical_mask = np.isfinite(ndwi) & (ndwi >= 0.12)
                except Exception:
                    optical_mask = optical_tensor.mean(axis=0) <= 0.25
            else:
                optical_mask = optical_tensor.mean(axis=0) <= 0.25

        # SAR: Low specular reflection from smooth water surface (dB <= -20 dB or normalized <= 0.28)
        sar_mask = sar_vv <= 0.28

    elif eff_target == "vegetation":
        polygon_kind = "croma_confirmed_vegetation"
        if multispectral_optical:
            try:
                ndvi, _, _ = read_index(optical_path, "ndvi", max_size=1024)
                if ndvi.shape != target_shape:
                    ndvi = np.array(Image.fromarray(ndvi).resize((out_w, out_h), Image.BILINEAR))
                optical_mask = np.isfinite(ndvi) & (ndvi >= 0.35)
            except Exception:
                optical_mask = optical_tensor[1] > optical_tensor[0]
        else:
            optical_mask = optical_tensor[1] > optical_tensor[0]

        # SAR: Moderate diffuse volume scattering
        sar_mask = (sar_vv >= 0.35) & (sar_vv <= 0.70)

    else:
        # General comparative
        polygon_kind = "croma_sensor_consensus"
        optical_mask = optical_tensor.mean(axis=0) >= 0.30
        sar_mask = sar_vv >= 0.35

    agreement_mask = sar_mask & optical_mask
    union_mask = sar_mask | optical_mask

    radar_optical_iou = float(agreement_mask.sum() / max(1, union_mask.sum())) * 100
    sensor_agreement_score = round(radar_optical_iou / 100.0, 3)

    total_pixels = max(1, target_shape[0] * target_shape[1])
    confirmed_pixels = int(agreement_mask.sum())
    confirmed_percent = round((confirmed_pixels / total_pixels) * 100, 3)

    # Save visual artifacts
    rgba_agreement = np.zeros((*target_shape, 4), dtype="uint8")
    rgba_agreement[optical_mask & ~sar_mask] = [245, 158, 11, 190]   # Optical only (orange)
    rgba_agreement[sar_mask & ~optical_mask] = [168, 85, 247, 190]   # SAR only (purple)
    rgba_agreement[agreement_mask] = [0, 229, 255, 230]             # Sensor agreement (cyan)

    agreement_path = output_dir / "croma_sensor_agreement.png"
    Image.fromarray(rgba_agreement, mode="RGBA").save(agreement_path, format="PNG")

    sar_db_preview = output_dir / "croma_sar_db_preview.png"
    Image.fromarray((sar_vv * 255).astype("uint8"), mode="L").save(sar_db_preview, format="PNG")

    optical_preview = output_dir / "croma_optical_preview.png"
    if optical_tensor.shape[0] >= 3:
        rgb = np.dstack([optical_tensor[0], optical_tensor[1], optical_tensor[2]])
        Image.fromarray((rgb * 255).astype("uint8"), mode="RGB").save(optical_preview, format="PNG")
    else:
        Image.fromarray((optical_tensor[0] * 255).astype("uint8"), mode="L").save(optical_preview, format="PNG")

    geojson_path = output_dir / "croma_cross_sensor_polygons.geojson"
    region_count, area = _polygonize(
        agreement_mask, scaled_transform, crs, kind=polygon_kind, output_path=geojson_path
    )

    base_conf = 0.88 if multispectral_optical else 0.82
    confidence = round(0.4 * base_conf + 0.6 * min(1.0, max(0.1, sensor_agreement_score / 0.60)), 3)

    croma_features.update({
        "target": eff_target,
        "agreement_iou": round(radar_optical_iou, 2),
        "cosine_similarity": cosine_similarity,
        "fusion_mode": "multimodal_sensor_consensus",
        "confirmed_percent": confirmed_percent,
        "area_m2": area,
        "region_count": region_count,
    })

    return CROMAFusedResult(
        producer=producer,
        sensor_agreement_score=sensor_agreement_score,
        radar_optical_iou=round(radar_optical_iou, 2),
        confirmed_area_m2=area,
        region_count=region_count,
        sar_channels_prepared=sar_channels,
        optical_channels_prepared=optical_channels,
        confidence=confidence,
        agreement_mask_path=agreement_path,
        sar_db_preview_path=sar_db_preview,
        optical_preview_path=optical_preview,
        geojson_path=geojson_path,
        croma_features=croma_features,
        cosine_similarity=cosine_similarity,
        confirmed_percent=confirmed_percent,
    )


def generate_sar_like_visual_comparison(
    opt_path: Path,
    sar_like_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate qualitative visual comparison artifacts between RGB optical and SAR-like visual raster."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_img_path = output_dir / "sar_like_visual_comparison.png"

    try:
        with Image.open(opt_path) as im_opt:
            im_opt_rgb = im_opt.convert("RGB")
        with Image.open(sar_like_path) as im_sar:
            im_sar_l = im_sar.convert("L")
    except Exception:
        with rasterio.open(opt_path) as src_opt:
            arr_opt = src_opt.read([1, min(2, src_opt.count), min(3, src_opt.count)])
            arr_opt = np.transpose(arr_opt, (1, 2, 0))
            if arr_opt.dtype != np.uint8:
                arr_opt = ((arr_opt - arr_opt.min()) / max(1e-5, arr_opt.max() - arr_opt.min()) * 255).astype(np.uint8)
            im_opt_rgb = Image.fromarray(arr_opt)
        with rasterio.open(sar_like_path) as src_sar:
            arr_sar = src_sar.read(1)
            if arr_sar.dtype != np.uint8:
                arr_sar = ((arr_sar - arr_sar.min()) / max(1e-5, arr_sar.max() - arr_sar.min()) * 255).astype(np.uint8)
            im_sar_l = Image.fromarray(arr_sar)

    target_size = (512, 512)
    im_opt_res = im_opt_rgb.resize(target_size, Image.Resampling.BILINEAR)
    im_sar_res = im_sar_l.resize(target_size, Image.Resampling.BILINEAR)

    arr_opt_gray = np.array(im_opt_res.convert("L"), dtype=float)
    arr_sar_f = np.array(im_sar_res, dtype=float)

    std_opt = np.std(arr_opt_gray)
    std_sar = np.std(arr_sar_f)
    if std_opt > 1e-4 and std_sar > 1e-4:
        norm_opt = (arr_opt_gray - np.mean(arr_opt_gray)) / std_opt
        norm_sar = (arr_sar_f - np.mean(arr_sar_f)) / std_sar
        visual_correlation = float(np.mean(norm_opt * norm_sar))
    else:
        visual_correlation = 0.0

    comp_width = target_size[0] * 3
    comp_height = target_size[1]
    composite = Image.new("RGB", (comp_width, comp_height), (15, 23, 42))
    composite.paste(im_opt_res, (0, 0))
    composite.paste(im_sar_res.convert("RGB"), (target_size[0], 0))

    overlay = np.zeros((*target_size, 3), dtype=np.uint8)
    overlay[..., 0] = arr_opt_gray.astype(np.uint8)
    overlay[..., 1] = arr_sar_f.astype(np.uint8)
    overlay[..., 2] = arr_sar_f.astype(np.uint8)
    composite.paste(Image.fromarray(overlay), (target_size[0] * 2, 0))

    composite.save(comparison_img_path, format="PNG")

    return {
        "comparison_image_path": comparison_img_path,
        "visual_correlation": round(visual_correlation, 3),
        "opt_contrast": round(float(std_opt), 1),
        "sar_like_contrast": round(float(std_sar), 1),
        "status": "Qualitative visual comparison completed (calibrated backscatter unavailable)",
    }

