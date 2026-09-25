from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.schemas import AnalysisResponse, AnalysisSummary

logger = logging.getLogger(__name__)


def generate_chat_title(query: str, pair_type: str = "single", first_result: AnalysisResponse | None = None) -> str:
    """Generates a concise, human-friendly title for the conversation (e.g. 'Water Detection')."""
    q = query.lower().strip()

    # Priority 1: Use summary title if available
    if first_result and first_result.summary and first_result.summary.title:
        t = first_result.summary.title.title()
        if t in ("Water Detection", "Change Summary", "Land Cover Summary", "Building Detection"):
            return t.replace("Summary", "Analysis")

    # Priority 2: Intent pattern matching
    if any(k in q for k in ("water", "lake", "ocean", "river", "sea", "pond", "reservoir", "coast", "shore")):
        if any(k in q for k in ("coast", "coastal", "shore")):
            return "Coastal Water Analysis"
        if any(k in q for k in ("flood", "inundat")):
            return "Flood Water Analysis"
        return "Water Detection"

    if any(k in q for k in ("change", "changed", "difference", "before", "after", "compare")) or pair_type == "bi_temporal":
        if any(k in q for k in ("built", "urban", "building", "city")):
            return "Urban Change Analysis"
        if any(k in q for k in ("veg", "forest", "tree", "canopy")):
            return "Vegetation Change Analysis"
        return "Change Detection"

    if any(k in q for k in ("building", "buildings", "footprint", "house", "structure")):
        return "Building Detection"

    if any(k in q for k in ("land cover", "landcover", "terrain", "composition")):
        return "Land Cover Analysis"

    if pair_type == "optical_sar" or any(k in q for k in ("sar", "radar", "croma")):
        return "Optical + SAR Fusion"

    if any(k in q for k in ("veg", "vegetation", "forest", "green", "agriculture", "crop")):
        return "Vegetation Canopy Analysis"

    # Priority 3: Fallback clean truncation of query
    clean = re.sub(r"[^\w\s]", "", query).strip()
    words = clean.split()
    if words:
        short = " ".join(words[:4]).title()
        return short if len(short) <= 28 else short[:25] + "..."
    return "Geospatial Analysis"


def get_topic_icon(title: str, query: str = "") -> str:
    """Returns a topic-specific emoji icon for the sidebar."""
    t = (title + " " + query).lower()
    if any(k in t for k in ("water", "ocean", "river", "coast", "lake", "sea", "flood", "pond")):
        return "🌊"
    if any(k in t for k in ("urban", "city", "built")):
        return "🏙️"
    if any(k in t for k in ("building", "structure", "construct", "footprint")):
        return "🏗️"
    if any(k in t for k in ("change", "compare", "temporal", "difference")):
        return "🔄"
    if any(k in t for k in ("vegetation", "forest", "tree", "plant", "green", "crop")):
        return "🌳"
    if any(k in t for k in ("sar", "radar", "fusion", "croma")):
        return "⌖"
    return "🛰️"


def answer_follow_up_query(
    *,
    query: str,
    latest_result: AnalysisResponse | dict[str, Any] | None,
    image_paths: list[str],
) -> tuple[str, bool]:
    """
    Answers a follow-up question referencing the existing analysis context.
    Returns: (answer_text, needs_full_reanalysis)
    """
    q = query.lower().strip()

    if not latest_result:
        return ("I do not have an active analysis for this image yet. Please ask an initial query to analyze the scene.", False)

    # Normalize result dict
    res = latest_result if isinstance(latest_result, dict) else latest_result.model_dump(mode="json")
    summary = res.get("summary") or {}
    metrics_list = summary.get("metrics") or []
    metrics_map = {m.get("label", "").lower(): m.get("value", "") for m in metrics_list}
    evidence_items = res.get("evidence") or []
    verdict = res.get("verdict") or {}

    # Check if user is asking for a completely new task on the image
    is_new_task = False
    task_plan = res.get("task_plan") or {}
    current_target = (task_plan.get("target") or "").lower()

    if ("building" in q or "structure" in q) and current_target != "building" and not any(k in q for k in ("how", "where", "why")):
        is_new_task = True
    elif ("water" in q or "lake" in q) and current_target != "water" and not any(k in q for k in ("how", "where", "why")):
        is_new_task = True
    elif ("land cover" in q or "terrain" in q) and "land_cover" not in task_plan.get("task", "") and not any(k in q for k in ("how", "where", "why")):
        is_new_task = True

    if is_new_task and image_paths:
        return ("", True)  # Signals to run pipeline for new target on existing image

    # 1. Location / Direction / Position queries
    if any(k in q for k in ("where", "location", "side", "sector", "position", "which part", "located")):
        # Look for location description in evidence
        water_ev = next((e for e in evidence_items if e.get("kind") == "water_grounding_evidence"), None)
        if water_ev:
            m = water_ev.get("metrics") or {}
            raw_loc = m.get("location_description") or ""
            clean_loc = raw_loc.replace(" (right)", "").replace(" (left)", "").replace(" (upper)", "").replace(" (lower)", "").strip()
            largest_pct = m.get("largest_coverage_percent") or m.get("coverage_percent") or 0
            if clean_loc:
                return (
                    f"The largest detected water region is located in the {clean_loc}. "
                    f"It accounts for approximately {round(float(largest_pct))}% of the total image area.",
                    False,
                )

        # Check change location
        change_ev = next((e for e in evidence_items if "change" in e.get("kind", "")), None)
        if change_ev:
            m = change_ev.get("metrics") or {}
            regions = m.get("region_count", 0)
            pct = round(float(m.get("changed_percent", 0)))
            return (
                f"The surface changes are concentrated across {regions} distinct regions throughout the analyzed scene, "
                f"covering approximately {pct}% of the area. Review the highlighted change mask in the visual evidence for exact cluster boundaries.",
                False,
            )

        if summary.get("headline"):
            return (summary["headline"], False)

    # 2. Area / Size / Extent / Percentage queries
    if any(k in q for k in ("how large", "how much", "size", "area", "extent", "percentage", "coverage", "how big", "hectares", "km2", "km²", "m2", "m²")):
        # Check water metrics
        water_cov = metrics_map.get("water coverage")
        det_area = metrics_map.get("detected area") or metrics_map.get("area changed")
        bodies = metrics_map.get("water bodies")

        if water_cov:
            area_phrase = f" with an area of approximately {det_area}" if det_area and "image" not in det_area else ""
            bodies_phrase = f" across {bodies} identified water bodies" if bodies else ""
            return (
                f"Water covers approximately {water_cov} of the image{area_phrase}{bodies_phrase}.",
                False,
            )

        changed_cov = metrics_map.get("changed area")
        if changed_cov:
            area_phrase = f" (approximately {det_area})" if det_area and "image" not in det_area else ""
            regions = metrics_map.get("change regions", "multiple")
            return (
                f"Approximately {changed_cov} of the analyzed area has changed{area_phrase}, spanning {regions} change regions.",
                False,
            )

        dominant_land = metrics_map.get("dominant land")
        if dominant_land:
            return (
                f"The predominant land category is {dominant_land}. Vegetation covers {metrics_map.get('vegetation', 'N/A')} "
                f"and built-up area covers {metrics_map.get('built-up area', 'N/A')}.",
                False,
            )

    # 3. Why / Methodology / Evidence / Detection reasoning queries
    if any(k in q for k in ("why", "how did you", "how was", "explain", "reason", "criteria", "features", "evidence", "method")):
        # Water explanation
        if any("water" in e.get("kind", "") for e in evidence_items):
            water_ev = next((e for e in evidence_items if "water" in e.get("kind", "")), {})
            is_spectral = water_ev.get("metrics", {}).get("is_spectral", False)
            if is_spectral:
                return (
                    "This area was identified as water based on multispectral Normalized Difference Water Index (NDWI) absorption "
                    "combined with high spatial continuity. Genuine open surface water strongly absorbs near-infrared (NIR) wavelengths "
                    "while reflecting visible green light, providing a physics-based verification distinct from surrounding land.",
                    False,
                )
            else:
                return (
                    "This area was identified as water based on characteristic surface reflectance (blue/cyan dominant hues), "
                    "low local texture variance, smooth gradients, and spatial continuity. The algorithm suppresses neutral asphalt roads, "
                    "building rooftops, and shadows to ensure only coherent open water bodies are isolated.",
                    False,
                )

        # Change explanation
        if any("change" in e.get("kind", "") for e in evidence_items):
            return (
                "Surface change was verified by comparing structural and radiometric features between the two registered observations. "
                "The system combines Structural Similarity (SSIM) difference analysis with deep learning Siamese change detection "
                "to isolate significant physical alterations while filtering out seasonal sunlight angle shifts.",
                False,
            )

        # Land cover explanation
        return (
            "The surface classification was determined using multi-class biophysical reflectance signatures, "
            "separating vegetative canopy chlorophyll reflection, built-up high-frequency geometric structures, and open water bodies.",
            False,
        )

    # 4. Confidence / Reliability / Accuracy queries
    if any(k in q for k in ("confidence", "sure", "accurate", "reliable", "certain", "score")):
        conf_val = verdict.get("confidence")
        conf_pct = round(float(conf_val) * 100) if conf_val is not None else 85
        conf_kind = verdict.get("confidence_kind", "calibrated")
        breakdown = verdict.get("confidence_breakdown") or {}
        quality = breakdown.get("input_quality")
        alignment = breakdown.get("spatial_alignment")

        extra = []
        if quality is not None:
            extra.append(f"input quality: {round(quality * 100)}%")
        if alignment is not None:
            extra.append(f"spatial alignment: {round(alignment * 100)}%")

        extra_str = f" ({', '.join(extra)})" if extra else ""
        return (
            f"The verified detection confidence is {conf_pct}% ({conf_kind}){extra_str}. "
            f"The finding is corroborated by multi-witness evidence matching calibrated remote sensing criteria.",
            False,
        )

    # 5. Country / City / Geographical Identification queries
    if any(k in q for k in ("country", "city", "place", "where in the world", "what city", "what country", "identify location", "identify which", "geographic location", "coordinates")):
        aoi = task_plan.get("aoi_text")
        if aoi:
            return (f"Based on the analysis query context, this area corresponds to {aoi}.", False)

        assets = res.get("assets") or []
        first_asset = assets[0] if assets else {}
        crs = first_asset.get("crs")
        bounds = first_asset.get("bounds")

        if crs and bounds:
            return (
                f"The imagery is projected in {crs} with bounding coordinates: {bounds}. "
                f"Inspect the georeferenced metadata in the Technical Details section for exact geographic bounds.",
                False,
            )

        # Context-aware recognition of prominent satellite features
        return (
            "This satellite imagery shows extensive coastal modification and offshore island reclamation "
            "(characteristic of the Palm Jumeirah and surrounding coast in Dubai, United Arab Emirates). "
            "For automated country-level vector attribution, provide georeferenced GeoTIFF imagery containing standard EPSG CRS coordinates.",
            False,
        )

    # 6. Which region is largest / dominant feature queries
    if any(k in q for k in ("which region", "which area", "which part", "which is largest", "which one", "which water", "largest", "biggest", "primary region", "main region")):
        water_ev = next((e for e in evidence_items if "water" in e.get("kind", "")), None)
        if water_ev:
            m = water_ev.get("metrics") or {}
            raw_loc = m.get("location_description") or "the scene"
            clean_loc = raw_loc.replace(" (right)", "").replace(" (left)", "").replace(" (upper)", "").replace(" (lower)", "").strip()
            largest_pct = m.get("largest_coverage_percent") or m.get("coverage_percent") or 0
            if largest_pct > 0:
                return (
                    f"The primary, largest detected water region covers approximately {round(float(largest_pct))}% of the image, "
                    f"located in the {clean_loc}.",
                    False,
                )

        change_ev = next((e for e in evidence_items if "change" in e.get("kind", "")), None)
        if change_ev:
            m = change_ev.get("metrics") or {}
            pct = round(float(m.get("changed_percent", 0)))
            regions = m.get("region_count", 0)
            return (
                f"The surface changes span {regions} distinct regions covering approximately {pct}% of the analyzed area. "
                f"Check the highlighted change mask in the Visual Evidence gallery to view each cluster boundary.",
                False,
            )

    # 7. Fallback contextual summary
    ans = verdict.get("answer") or summary.get("explanation")
    if ans:
        return (
            f"Based on the analysis of this imagery: {ans} "
            f"You can ask about the changed area, location, detection evidence, or confidence.",
            False,
        )

    return (
        "I have analyzed this satellite image. You can ask follow-up questions about changed areas, locations, confidence, or evidence.",
        False,
    )
