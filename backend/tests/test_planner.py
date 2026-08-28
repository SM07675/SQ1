from app.schemas import TaskType
from app.services.planner import plan_query


def test_routes_change_query() -> None:
    plan = plan_query("Has built-up area increased between these dates?", 2, "auto")
    assert plan.task == TaskType.BI_TEMPORAL_CHANGE
    assert plan.target == "built-up"
    assert plan.asks_direction is True


def test_routes_optical_sar_pair() -> None:
    plan = plan_query("Use optical and SAR to identify water", 2, "optical_sar")
    assert plan.task == TaskType.OPTICAL_SAR
    assert "croma" in plan.tools


def test_routes_grounding() -> None:
    plan = plan_query("Highlight the largest water body", 1)
    assert plan.task == TaskType.GROUNDING

