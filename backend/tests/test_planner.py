from app.schemas import TaskType
from app.services.planner import plan_query


def test_routes_change_query() -> None:
    plan = plan_query("Has built-up area increased between these dates?", 2, "auto")
    assert plan.task == TaskType.BI_TEMPORAL_CHANGE
    assert plan.application == "bi_temporal"
    assert plan.target == "built-up"
    assert plan.asks_direction is True


def test_routes_optical_sar_pair() -> None:
    plan = plan_query("Use optical and SAR to identify water", 2, "optical_sar")
    assert plan.task == TaskType.OPTICAL_SAR
    assert plan.application == "optical_sar"
    assert "croma" in plan.tools


def test_routes_grounding() -> None:
    plan = plan_query("Highlight the largest water body", 1)
    assert plan.task in (TaskType.GROUNDING, TaskType.WATER_ANALYSIS)
    assert plan.application == "single_image"


def test_16_canonical_queries() -> None:
    # 1. Scene Description
    p1 = plan_query("Describe this image.", 1)
    assert p1.application == "single_image"
    assert p1.task == TaskType.SCENE_DESCRIPTION

    # 2. Land Cover Analysis
    p2 = plan_query("What land cover is visible in this scene?", 1)
    assert p2.application == "single_image"
    assert p2.task == TaskType.LAND_COVER

    # 3. Object Identification
    p3 = plan_query("Identify the major objects.", 1)
    assert p3.application == "single_image"
    assert p3.task == TaskType.OBJECT_IDENTIFICATION

    # 4. Water Grounding
    p4 = plan_query("Highlight the largest water body.", 1)
    assert p4.application == "single_image"
    assert p4.task in (TaskType.GROUNDING, TaskType.WATER_ANALYSIS)

    # 5. Vegetation Analysis
    p5 = plan_query("Where is vegetation concentrated?", 1)
    assert p5.application == "single_image"
    assert p5.task == TaskType.VEGETATION_ANALYSIS

    # 6. Built-up Analysis
    p6 = plan_query("Identify built-up areas.", 1)
    assert p6.application == "single_image"
    assert p6.task == TaskType.BUILT_UP_ANALYSIS

    # 7. Single VQA
    p7 = plan_query("How many buildings are visible?", 1)
    assert p7.application == "single_image"
    assert p7.task == TaskType.SINGLE_VQA

    # 8. General Change Analysis
    p8 = plan_query("What changed between these two images?", 2)
    assert p8.application == "bi_temporal"
    assert p8.task == TaskType.BI_TEMPORAL_CHANGE

    # 9. Vegetation Loss Change
    p9 = plan_query("Has vegetation decreased?", 2)
    assert p9.application == "bi_temporal"
    assert p9.task == TaskType.BI_TEMPORAL_CHANGE
    assert p9.target == "vegetation"

    # 10. Built-up Expansion Change
    p10 = plan_query("Has built-up area increased?", 2)
    assert p10.application == "bi_temporal"
    assert p10.task == TaskType.BI_TEMPORAL_CHANGE
    assert p10.target == "built-up"

    # 11. Water Dynamic Change
    p11 = plan_query("Has the water body expanded?", 2)
    assert p11.application == "bi_temporal"
    assert p11.task == TaskType.BI_TEMPORAL_CHANGE
    assert p11.target == "water"

    # 12. Optical + SAR Water
    p12 = plan_query("Use optical and SAR together to identify water.", 2)
    assert p12.application == "optical_sar"
    assert p12.task == TaskType.OPTICAL_SAR

    # 13. Optical + SAR Built-up
    p13 = plan_query("Use both sensors to identify urban areas.", 2)
    assert p13.application == "optical_sar"
    assert p13.task == TaskType.OPTICAL_SAR

    # 14. Optical + SAR Comparative
    p14 = plan_query("What information does SAR add?", 2)
    assert p14.application == "optical_sar"
    assert p14.task == TaskType.OPTICAL_SAR

    # 15. Unsupported query
    p15 = plan_query("What is the capital of France?", 1)
    assert p15.application == "unsupported"
    assert p15.task == TaskType.UNSUPPORTED

    # 16. Unclear query
    p16 = plan_query("hello", 1)
    assert p16.application == "unclear"
    assert p16.task == TaskType.UNCLEAR
