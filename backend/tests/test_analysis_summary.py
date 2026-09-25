from pathlib import Path
from unittest.mock import MagicMock

from app.schemas import EvidenceItem, TaskType, VerdictStatus
from app.services.orchestrator import build_analysis_summary, format_human_area


def test_format_human_area():
    assert format_human_area(None) is None
    assert format_human_area(-10) is None
    assert format_human_area(0) is None
    assert format_human_area(500) == "500 m²"
    assert format_human_area(25_000) == "2.5 ha"
    assert format_human_area(2_400_000) == "2.4 km²"
    assert format_human_area(15_000_000) == "15.0 km²"


def test_build_analysis_summary_water():
    plan = MagicMock()
    plan.task = TaskType.WATER_ANALYSIS

    verdict = MagicMock()
    verdict.status = VerdictStatus.SUPPORTED
    verdict.confidence = 0.92

    water_ev = EvidenceItem(
        kind="water_grounding_evidence",
        producer="test_water_engine",
        summary="Water detected",
        confidence=0.92,
        metrics={
            "water_body_identified": True,
            "coverage_percent": 30.0,
            "largest_coverage_percent": 30.0,
            "area_m2": 2_400_000.0,
            "region_count": 3,
            "location_description": "eastern (right) portion of the image",
        },
    )

    metadata = [MagicMock(crs="EPSG:4326")]

    summary, answer = build_analysis_summary(
        task_plan=plan,
        verdict=verdict,
        evidence=[water_ev],
        metadata=metadata,
        query="Highlight the largest water body",
    )

    assert summary.title == "WATER DETECTION"
    assert "30%" in summary.headline
    assert "eastern portion of the image" in summary.headline
    assert summary.confidence_percent == 92
    assert not summary.is_insufficient

    metric_labels = {m.label: m.value for m in summary.metrics}
    assert metric_labels["Water Coverage"] == "30%"
    assert metric_labels["Detected Area"] == "2.4 km²"
    assert metric_labels["Water Bodies"] == "3"
    assert metric_labels["Confidence"] == "92%"


def test_build_analysis_summary_change_detection():
    plan = MagicMock()
    plan.task = TaskType.BI_TEMPORAL_CHANGE

    verdict = MagicMock()
    verdict.status = VerdictStatus.SUPPORTED
    verdict.confidence = 0.89

    change_ev = EvidenceItem(
        kind="baseline_change_detection",
        producer="test_detector",
        summary="Changed area",
        confidence=0.89,
        metrics={
            "changed_percent": 30.0,
            "area_m2": 1_800_000.0,
            "region_count": 3,
        },
    )

    metadata = [MagicMock(crs="EPSG:4326"), MagicMock(crs="EPSG:4326")]
    detected_changes = [
        "Built-up area increased by 8%",
        "Vegetation decreased by 12%",
        "Water area increased by 15%",
    ]

    summary, answer = build_analysis_summary(
        task_plan=plan,
        verdict=verdict,
        evidence=[change_ev],
        metadata=metadata,
        query="Has this area changed?",
        bi_temporal_changes=detected_changes,
    )

    assert summary.title == "CHANGE SUMMARY"
    assert "30%" in summary.headline
    assert summary.confidence_percent == 89
    assert len(summary.detected_changes) == 3

    metric_labels = {m.label: m.value for m in summary.metrics}
    assert metric_labels["Changed Area"] == "30%"
    assert metric_labels["Area Changed"] == "1.8 km²"
    assert metric_labels["Change Regions"] == "3"
    assert metric_labels["Confidence"] == "89%"


def test_build_analysis_summary_insufficient_evidence():
    plan = MagicMock()
    plan.task = TaskType.BI_TEMPORAL_CHANGE

    verdict = MagicMock()
    verdict.status = VerdictStatus.INSUFFICIENT_EVIDENCE
    verdict.confidence = 0.25

    metadata = [MagicMock(crs=None), MagicMock(crs=None)]

    summary, answer = build_analysis_summary(
        task_plan=plan,
        verdict=verdict,
        evidence=[],
        metadata=metadata,
        query="Has this area changed?",
    )

    assert summary.is_insufficient is True
    assert summary.confidence_percent is None
    assert "could not determine the result reliably" in summary.headline
