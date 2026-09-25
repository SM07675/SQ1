"""Use the imported surface specialists only for single-image surface tasks."""

from pathlib import Path

from app.schemas import AnalysisResponse, AnalysisSummary, SummaryMetric
from app.schemas import TaskType
from app.services.planner import plan_query


def uses_surface_pipeline(query: str, pair_type: str, image_count: int, modality: str = "optical") -> bool:
    plan = plan_query(query, image_count, pair_type)
    surface_tasks = {TaskType.WATER_ANALYSIS, TaskType.LAND_COVER, TaskType.BUILDINGS}
    surface_subtasks = {"water_analysis", "land_cover", "built_up_analysis", "building_detection"}
    return (
        image_count == 1
        and modality in {"optical", "multispectral"}
        and pair_type not in {"bi_temporal", "optical_sar"}
        and (
            plan.task in surface_tasks
            or (plan.task == TaskType.GROUNDING and plan.target in {"water", "land_cover", "built-up"})
        )
        and set(plan.sub_tasks).issubset(surface_subtasks)
    )


async def analyze(
    *,
    result_id: str,
    query: str,
    pair_type: str,
    image_paths: list[Path],
    output_dir: Path,
) -> AnalysisResponse:
    modality = "unknown"
    if len(image_paths) == 1:
        from app.services.croma_pipeline import inspect_modality

        modality = inspect_modality(image_paths[0]).modality
    if not uses_surface_pipeline(query, pair_type, len(image_paths), modality):
        # Keep the current project's SAR, change, vegetation, VQA, and other
        # analysis routes and models unchanged.
        from app.services.orchestrator import analyze as analyze_current

        return await analyze_current(
            result_id=result_id,
            query=query,
            pair_type=pair_type,
            image_paths=image_paths,
            output_dir=output_dir,
        )

    from satquery_engine.services.execution import analyze as run_engine
    from satquery_engine.services.report import write_manifest

    result = await run_engine(
        result_id=result_id,
        query=query,
        pair_type=pair_type,
        image_paths=image_paths,
        output_dir=output_dir,
    )
    payload = result.model_dump(mode="json")
    # The current dashboard expects this summary field. Use the canonical answer
    # verbatim so that counts, areas, and limitations cannot drift from the report.
    verdict = result.verdict
    payload["summary"] = AnalysisSummary(
        title="ANALYSIS SUMMARY",
        headline=verdict.answer,
        explanation=verdict.answer,
        metrics=[SummaryMetric(label="Status", value=verdict.status.value.replace("_", " ").title())],
        is_insufficient=verdict.status.value in {
            "insufficient_evidence", "invalid_input", "unsupported_task", "model_unavailable"
        },
    ).model_dump(mode="json")
    response = AnalysisResponse.model_validate(payload)
    write_manifest(output_dir, response.model_dump(mode="json"))
    return response
