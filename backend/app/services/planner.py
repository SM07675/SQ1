from __future__ import annotations

import re

from app.schemas import TaskPlan, TaskType


TARGETS = {
    "built-up": ("built-up", "construction", "building", "urban", "structures", "commercial", "residential"),
    "water": (
        "water", "river", "lake", "flood", "ocean", "sea", "reservoir", "pond",
        "water body", "waterbody", "canal", "estuary", "bay", "coastal", "aquatic", "wetland"
    ),
    "vegetation": ("vegetation", "forest", "crop", "green", "ndvi", "canopy", "tree", "agriculture"),
    "road": ("road", "highway", "bridge", "street", "transportation"),
}


def _target(query: str) -> str | None:
    for canonical, words in TARGETS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", query) for word in words):
            return canonical
    return None


def plan_query(query: str, image_count: int, pair_type: str = "auto") -> TaskPlan:
    q = query.lower().strip()
    if not q:
        raise ValueError("Query cannot be empty")

    target = _target(q)
    asks_direction = any(word in q for word in ("increase", "decrease", "grew", "reduced", "direction"))
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", q)]
    aoi_match = re.search(r"\b(?:in|within|around|near)\s+([a-z][a-z0-9 ._-]{1,60}?)(?:\s+between|\s+from|\s+in\s+\d{4}|[?.!,]|$)", q)
    aoi_text = aoi_match.group(1).strip() if aoi_match else None

    if pair_type == "optical_sar" or (image_count > 1 and "sar" in q):
        return TaskPlan(
            task=TaskType.OPTICAL_SAR,
            target=target,
            asks_direction=asks_direction,
            tools=["raster_validator", "croma", "earthdial_ms", "geoproof"],
            reason="The input/query requests joint optical-SAR analysis.",
            aoi_text=aoi_text,
            years=years,
        )

    change_words = ("change", "between", "before", "after", "date", "increase", "decrease")
    if pair_type == "bi_temporal" or (image_count > 1 and any(word in q for word in change_words)):
        return TaskPlan(
            task=TaskType.BI_TEMPORAL_CHANGE,
            target=target,
            asks_direction=asks_direction,
            tools=["raster_validator", "change_detector", "polygonizer", "gis_metrics", "earthdial", "geoproof"],
            reason="Two images and change-oriented language require a bi-temporal workflow.",
            aoi_text=aoi_text,
            years=years,
        )

    if any(word in q for word in ("highlight", "locate", "where", "show me", "mark")):
        return TaskPlan(
            task=TaskType.GROUNDING,
            target=target,
            tools=["raster_validator", "earthdial_grounding", "gis_metrics", "geoproof"],
            reason="The query requests spatial localization or grounding.",
            aoi_text=aoi_text,
            years=years,
        )

    if any(word in q for word in ("describe", "caption", "summarize", "scene")):
        return TaskPlan(
            task=TaskType.SCENE_DESCRIPTION,
            target=target,
            tools=["raster_validator", "earthdial", "geoproof"],
            reason="The query asks for a scene-level description.",
            aoi_text=aoi_text,
            years=years,
        )

    return TaskPlan(
        task=TaskType.SINGLE_VQA,
        target=target,
        tools=["raster_validator", "earthdial", "geoproof"],
        reason="The query is handled as single-image remote-sensing VQA.",
        aoi_text=aoi_text,
        years=years,
    )
