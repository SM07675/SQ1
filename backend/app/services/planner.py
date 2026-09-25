from __future__ import annotations

import re
from typing import Any

from app.schemas import TaskPlan, TaskType


TARGETS = {
    "built-up": (
        "built-up", "built up", "construction", "building", "buildings", "urban", "structures",
        "commercial", "residential", "development", "urban expansion", "settlement", "settlements",
        "houses", "city", "town", "infrastructure", "airport", "runway"
    ),
    "water": (
        "water", "river", "rivers", "lake", "lakes", "flood", "ocean", "sea", "reservoir", "pond", "ponds",
        "water body", "water bodies", "waterbody", "canal", "estuary", "bay", "coastal", "aquatic",
        "wetland", "water expansion"
    ),
    "vegetation": (
        "vegetation", "forest", "forests", "crop", "crops", "green", "greenery", "ndvi", "canopy",
        "tree", "trees", "agriculture", "agricultural", "farmland", "woodland", "vegetation loss",
        "deforestation", "pasture", "dense canopy"
    ),
    "road": ("road", "roads", "highway", "highways", "bridge", "bridges", "street", "streets", "transportation"),
}

CHANGE_KEYWORDS = (
    "change", "changed", "before/after", "before and after", "before", "after", "increase", "increased",
    "decrease", "decreased", "expansion", "expanded", "expand", "reduction", "reduced", "growth", "grown",
    "loss", "lost", "difference", "between dates", "between these", "between two", "over time",
    "temporal comparison", "urban expansion", "vegetation loss", "water expansion", "evolve", "compared",
    "dynamics", "shrunk", "shrinking"
)

OPTICAL_SAR_KEYWORDS = (
    "optical + sar", "optical and sar", "radar", "sar", "multispectral and radar",
    "both sensors", "cross-modal", "cross modal", "combine optical and sar",
    "use both images", "radar evidence", "optical evidence", "sentinel-1 and sentinel-2",
    "optical-sar", "sar-optical", "what information does sar add", "surface characteristics",
    "what does optical show", "what does sar show", "what additional information does sar provide",
    "identify built-up areas using both", "identify water using optical and sar", "compare optical and sar",
    "using both", "both images", "optical and radar", "radar and optical", "complementary analysis",
    "sar show", "optical show", "radar show", "sar add"
)

LAND_COVER_KEYWORDS = (
    "land cover", "landcover", "land-cover", "terrain", "types of terrain", "types of land",
    "what types of land", "composed of", "composition of land", "agricultural land",
    "is this area mostly agricultural", "classify the visible land cover", "what can i see here",
    "what land cover", "what land is", "what land", "land-cover types", "land use", "land classification",
    "find land", "find the land", "locate land", "locate land cover", "identify land",
    "detect land", "detect land cover", "map land", "map land cover", "land cover map",
    "segment land", "segment land cover", "land segmentation", "land classes", "land categories",
    "classify land", "classify land cover", "surface composition"
)

SCENE_DESCRIPTION_KEYWORDS = (
    "describe this image", "describe the image", "describe this scene", "describe the scene",
    "describe", "what is visible in this scene", "what is visible in this image", "give me an overview",
    "what does this image contain", "overview of this satellite image", "overview", "summarize this image",
    "summarize", "caption this", "caption", "tell me about this scene", "what is the scene",
    "what is visible", "what's visible"
)

OBJECT_IDENTIFICATION_KEYWORDS = (
    "where are the buildings", "identify roads", "find the river", "locate agricultural fields",
    "identify major objects", "what objects are visible", "what objects are present", "objects present",
    "locate structures", "where are the settlements", "locate the airport", "locate", "identify",
    "find the", "detect objects", "where is", "where are"
)

BUILDING_DETECTION_KEYWORDS = (
    "how many buildings", "count buildings", "count the buildings", "number of buildings",
    "building footprint", "building footprints", "detect buildings", "building detection",
    "identify buildings", "map buildings", "building count", "buildings in this image",
    "structures detected", "footprint detection", "delineate buildings",
)


UNSUPPORTED_PATTERNS = (
    r"\bcapital of\b",
    r"\bwho is\b",
    r"\bwho was\b",
    r"\bpython code\b",
    r"\bwrite a code\b",
    r"\bwrite a script\b",
    r"\bhow to cook\b",
    r"\bweather in\b",
    r"\btranslate\b",
    r"\bcalculate \d+\b",
    r"\bwhat is \d+ \+ \d+\b",
    r"\bpresident of\b",
    r"\bmeaning of life\b",
)


def _target(query: str) -> str | None:
    for canonical, words in TARGETS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", query) for word in words):
            return canonical
    return None


def is_optical_sar_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in OPTICAL_SAR_KEYWORDS)


def is_change_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in CHANGE_KEYWORDS)


def is_unsupported_query(query: str) -> bool:
    q = query.lower().strip()
    return any(re.search(pat, q) for pat in UNSUPPORTED_PATTERNS)


def is_unclear_query(query: str) -> bool:
    q = query.lower().strip()
    if len(q) < 3:
        return True
    if q in ("hello", "hi", "hey", "test", "run", "do it", "analyze", "check", "what", "ok", "please"):
        return True
    return False


def plan_query(query: str, image_count: int = 1, pair_type: str = "auto") -> TaskPlan:
    q = query.lower().strip()
    if not q:
        raise ValueError("Query cannot be empty")

    # 1. Off-Topic / Unsupported Check
    if is_unsupported_query(q):
        return TaskPlan(
            task=TaskType.UNSUPPORTED,
            application="unsupported",
            specific_task="unsupported",
            tools=[],
            reason="Query is not related to remote-sensing satellite imagery analysis.",
        )

    # 2. Unclear / Ambiguous Check
    if is_unclear_query(q):
        return TaskPlan(
            task=TaskType.UNCLEAR,
            application="unclear",
            specific_task="unclear",
            tools=[],
            reason="Query does not contain sufficient detail to determine analytical intent.",
        )

    target = _target(q)
    asks_direction = any(word in q for word in ("increase", "increased", "decrease", "decreased", "grew", "reduced", "growth", "loss", "direction", "expanded", "expansion"))
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", q)]
    aoi_match = re.search(r"\b(?:in|within|around|near)\s+([a-z][a-z0-9 ._-]{1,60}?)(?:\s+between|\s+from|\s+in\s+\d{4}|[?.!,]|$)", q)
    aoi_text = aoi_match.group(1).strip() if aoi_match else None

    # Auto-resolve multi-image pair type: 2 images default to bi_temporal (or optical_sar if specified)
    effective_pair = pair_type
    if image_count >= 2:
        if pair_type == "optical_sar" or is_optical_sar_query(q):
            effective_pair = "optical_sar"
        else:
            effective_pair = "bi_temporal"
    elif is_change_query(q):
        effective_pair = "bi_temporal"

    # =========================================================================
    # APPLICATION 3: Optical + SAR Cross-Modal Analysis
    # =========================================================================
    if effective_pair == "optical_sar" or is_optical_sar_query(q):
        if any(k in q for k in ("what does optical show", "optical show", "only optical", "visible in optical")):
            specific_task = "optical_focus_analysis"
            tools = ["raster_validator", "optical_evidence_engine", "earthdial_4b_ms", "geoproof_arbiter"]
        elif any(k in q for k in ("what does sar show", "sar show", "what does radar show", "radar show", "only sar")):
            specific_task = "sar_focus_analysis"
            tools = ["raster_validator", "sar_backscatter_detector", "earthdial_4b_ms", "geoproof_arbiter"]
        elif any(k in q for k in ("what additional information does sar provide", "what information does sar add", "what does sar add", "what additional information")):
            specific_task = "sar_complementary_analysis"
            tools = ["raster_validator", "sar_backscatter_detector", "croma", "earthdial_4b_ms", "geoproof_arbiter"]
        elif target == "water":
            specific_task = "cross_modal_water"
            tools = ["raster_validator", "optical_water_grounding_engine", "sar_backscatter_detector", "croma", "geoproof_arbiter"]
        elif target == "built-up":
            specific_task = "cross_modal_builtup"
            tools = ["raster_validator", "satquery_buildings_dl", "sar_backscatter_detector", "croma", "geoproof_arbiter"]
        elif any(k in q for k in ("compare optical and sar", "compare", "comparative")):
            specific_task = "cross_modal_comparative"
            tools = ["raster_validator", "croma", "sar_backscatter_detector", "earthdial_4b_ms", "geoproof_arbiter"]
        else:
            specific_task = "cross_modal_comparative"
            tools = ["raster_validator", "croma", "optical_evidence_engine", "sar_backscatter_detector", "earthdial_4b_ms", "geoproof_arbiter"]

        return TaskPlan(
            task=TaskType.OPTICAL_SAR,
            application="optical_sar",
            specific_task=specific_task,
            target=target,
            asks_direction=asks_direction,
            tools=tools,
            reason="Natural language request identifies cross-sensor Optical + SAR joint verification.",
            aoi_text=aoi_text,
            years=years,
        )

    # =========================================================================
    # APPLICATION 2: Bi-Temporal Change Analysis
    # =========================================================================
    if effective_pair == "bi_temporal" or is_change_query(q) or image_count >= 2:
        tools = ["raster_validator", "phase_correlation_ecc_registration", "change_detector"]
        specific_task = "general_change"
        if target == "built-up":
            specific_task = "built_up_change"
            tools.append("ndbi_spectral_engine")
        elif target == "vegetation":
            specific_task = "vegetation_change"
            tools.append("ndvi_spectral_engine")
        elif target == "water":
            specific_task = "water_change"
            tools.append("ndwi_spectral_engine")
        else:
            tools.append("ssim_spectral_change_engine")
        tools.extend(["tinycd_opencd", "polygonizer", "gis_metrics", "geoproof_arbiter", "platt_scaling"])
        return TaskPlan(
            task=TaskType.BI_TEMPORAL_CHANGE,
            application="bi_temporal",
            specific_task=specific_task,
            target=target,
            asks_direction=asks_direction,
            tools=tools,
            reason="Query expresses temporal comparison or land cover dynamic change over time.",
            aoi_text=aoi_text,
            years=years,
        )

    # =========================================================================
    # APPLICATION 1: Single Image Intelligence (Multi-Task & Multi-Intent)
    # =========================================================================

    # Check for Multi-Intent query (e.g. "describe land cover AND identify water bodies")
    has_land_cover_intent = any(k in q for k in LAND_COVER_KEYWORDS)
    has_water_intent = any(k in q for k in ("water", "lake", "river", "ocean", "flood", "pond", "reservoir", "wetland"))
    has_veg_intent = any(k in q for k in ("vegetation", "greenery", "forest", "crop", "tree", "canopy", "ndvi"))
    has_built_intent = any(k in q for k in ("built-up", "built up", "building", "buildings", "urban", "construction", "city", "settlement"))
    has_building_detection_intent = any(k in q for k in BUILDING_DETECTION_KEYWORDS)
    has_scene_intent = any(k in q for k in SCENE_DESCRIPTION_KEYWORDS)

    sub_tasks: list[str] = []
    if has_land_cover_intent:
        sub_tasks.append("land_cover")
    if has_water_intent and ("water" in q or "river" in q or "lake" in q or "flood" in q):
        sub_tasks.append("water_analysis")
    if has_veg_intent and ("vegetation" in q or "greenery" in q or "forest" in q or "canopy" in q):
        sub_tasks.append("vegetation_analysis")
    if has_built_intent and ("built-up" in q or "urban" in q or "building" in q or "city" in q):
        sub_tasks.append("built_up_analysis")
    if has_building_detection_intent:
        sub_tasks.append("building_detection")
    if has_scene_intent and not has_land_cover_intent:
        sub_tasks.append("scene_description")

    is_multi_intent = len(sub_tasks) > 1

    # Task Category 1: Land Cover Understanding
    if has_land_cover_intent:
        tools = ["raster_validator", "satquery_landcover_dl", "land_cover_engine", "ndvi_spectral", "ndwi_spectral", "ndbi_spectral", "remoteclip", "earthdial_4b_rgb", "geoproof_arbiter"]
        return TaskPlan(
            task=TaskType.LAND_COVER,
            application="single_image",
            specific_task="land_cover",
            sub_tasks=sub_tasks if is_multi_intent else ["land_cover"],
            multi_intent=is_multi_intent,
            target=target or "land_cover",
            tools=tools,
            reason="Natural language request identifies land-cover classification and terrain understanding.",
            aoi_text=aoi_text,
            years=years,
        )

    # Task Category 2: Water Grounding / Analysis
    if (has_water_intent and any(k in q for k in ("highlight", "largest", "find water", "where are the water", "water bodies", "where is the river", "show the lake", "locate water", "how much water"))) or (target == "water" and not has_scene_intent):
        tools = ["raster_validator", "optical_water_grounding_engine", "ndwi_spectral", "remoteclip", "gis_metrics", "geoproof_arbiter"]
        task_type = TaskType.GROUNDING if any(k in q for k in ("highlight", "locate", "where is", "find", "show")) else TaskType.WATER_ANALYSIS
        return TaskPlan(
            task=task_type,
            application="single_image",
            specific_task="water_grounding" if task_type == TaskType.GROUNDING else "water_analysis",
            sub_tasks=sub_tasks if is_multi_intent else ["water_analysis"],
            multi_intent=is_multi_intent,
            target="water",
            tools=tools,
            reason="Natural language request asks to locate, highlight or analyze water bodies.",
            aoi_text=aoi_text,
            years=years,
        )

    # Task Category 3: Vegetation Analysis
    if (has_veg_intent and any(k in q for k in ("concentrated", "greenery", "analyze vegetation", "where is vegetation", "is vegetation present", "dense", "canopy", "forest"))) or (target == "vegetation" and not has_scene_intent):
        tools = ["raster_validator", "ndvi_spectral_engine", "canopy_grounding_engine", "earthdial_4b_rgb", "geoproof_arbiter"]
        return TaskPlan(
            task=TaskType.VEGETATION_ANALYSIS,
            application="single_image",
            specific_task="vegetation_analysis",
            sub_tasks=sub_tasks if is_multi_intent else ["vegetation_analysis"],
            multi_intent=is_multi_intent,
            target="vegetation",
            tools=tools,
            reason="Natural language request asks for vegetation distribution and canopy analysis.",
            aoi_text=aoi_text,
            years=years,
        )

    is_vqa_question = any(k in q for k in ("how many", "count", "is there", "are there", "what is the number", "number of", "can you see", "is this area mostly"))

    # Task Category 4a: Building Detection (DL footprint + instance)
    if has_building_detection_intent or (is_vqa_question and any(k in q for k in ("building", "buildings", "structure", "structures", "footprint"))):
        tools = ["raster_validator", "satquery_buildings_dl", "building_footprint_engine", "remoteclip", "geoproof_arbiter"]
        return TaskPlan(
            task=TaskType.BUILDINGS,
            application="single_image",
            specific_task="building_detection",
            sub_tasks=sub_tasks if is_multi_intent else ["building_detection"],
            multi_intent=is_multi_intent,
            target="built-up",
            tools=tools,
            reason="Natural language request asks to detect, count, or delineate building footprints.",
            aoi_text=aoi_text,
            years=years,
        )

    # Task Category 4b: Built-up / Urban Analysis
    if not is_vqa_question and ((has_built_intent and any(k in q for k in ("built-up", "built up", "urban", "urbanized", "where are the buildings", "locate buildings", "city", "settlement", "identify built-up"))) or (target == "built-up" and not has_scene_intent)):
        tools = ["raster_validator", "ndbi_spectral_engine", "builtup_grounding_engine", "remoteclip", "earthdial_4b_rgb", "geoproof_arbiter"]
        return TaskPlan(
            task=TaskType.BUILT_UP_ANALYSIS,
            application="single_image",
            specific_task="built_up_analysis",
            sub_tasks=sub_tasks if is_multi_intent else ["built_up_analysis"],
            multi_intent=is_multi_intent,
            target="built-up",
            tools=tools,
            reason="Natural language request asks for built-up footprint and urban structure identification.",
            aoi_text=aoi_text,
            years=years,
        )

    # Task Category 5: Object / Feature Identification & Grounding
    if any(k in q for k in OBJECT_IDENTIFICATION_KEYWORDS) or ("identify" in q and ("object" in q or "structure" in q or "feature" in q)):
        tools = ["raster_validator", "remoteclip_tile_retriever", "earthdial_grounding", "geoproof_arbiter"]
        return TaskPlan(
            task=TaskType.OBJECT_IDENTIFICATION,
            application="single_image",
            specific_task="object_identification",
            sub_tasks=sub_tasks if is_multi_intent else ["object_identification"],
            multi_intent=is_multi_intent,
            target=target,
            tools=tools,
            reason="Natural language request indicates visual object localization and semantic grounding.",
            aoi_text=aoi_text,
            years=years,
        )

    # Task Category 6: Scene Description
    if has_scene_intent:
        tools = ["raster_validator", "remoteclip", "earthdial_4b_rgb", "land_cover_engine", "geoproof_arbiter"]
        return TaskPlan(
            task=TaskType.SCENE_DESCRIPTION,
            application="single_image",
            specific_task="scene_description",
            sub_tasks=sub_tasks if is_multi_intent else ["scene_description"],
            multi_intent=is_multi_intent,
            target=target,
            tools=tools,
            reason="Natural language request asks for comprehensive single-image scene description.",
            aoi_text=aoi_text,
            years=years,
        )

    # Task Category 7: Visual Question Answering (Default Single Image)
    tools = ["raster_validator", "earthdial_4b_rgb", "remoteclip", "geoproof_arbiter"]
    if target == "water":
        tools.append("ndwi_spectral")
    elif target == "vegetation":
        tools.append("ndvi_spectral")
    elif target == "built-up":
        tools.append("ndbi_spectral")

    return TaskPlan(
        task=TaskType.SINGLE_VQA,
        application="single_image",
        specific_task="vqa",
        sub_tasks=sub_tasks if is_multi_intent else ["vqa"],
        multi_intent=is_multi_intent,
        target=target,
        tools=tools,
        reason="Query processed under single-image remote sensing visual question answering.",
        aoi_text=aoi_text,
        years=years,
    )
