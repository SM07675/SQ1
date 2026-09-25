"""The imported specialists may only replace the three requested tasks."""

import pytest

from app.services.integrated_analysis import uses_surface_pipeline


@pytest.mark.parametrize(
    ("query", "pair_type", "image_count", "modality", "expected"),
    [
        ("How many buildings are visible?", "auto", 1, "optical", True),
        ("Highlight the largest water body", "auto", 1, "optical", True),
        ("What land cover is visible?", "auto", 1, "optical", True),
        ("Count buildings, measure water area, and map land cover", "auto", 1, "optical", True),
        ("Highlight the largest water body", "auto", 1, "sar", False),
        ("Highlight the largest water body", "auto", 1, "sar_like", False),
        ("Highlight the largest water body", "auto", 1, "unknown", False),
        ("Use optical and SAR together to identify water", "optical_sar", 2, "optical", False),
        ("What changed between these two images?", "bi_temporal", 2, "optical", False),
        ("Describe the scene", "auto", 1, "optical", False),
        ("Find vegetation", "auto", 1, "optical", False),
        ("Identify built-up areas", "auto", 1, "optical", False),
    ],
)
def test_only_requested_single_image_tasks_use_surface_pipeline(query, pair_type, image_count, modality, expected):
    assert uses_surface_pipeline(query, pair_type, image_count, modality) is expected
