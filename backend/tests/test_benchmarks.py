import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.benchmarks import (
    compute_box_iou,
    compute_mean_iou,
    compute_semantic_similarity,
    compute_token_perplexity,
    list_available_datasets,
    run_vrsbench_suite,
)
from app.services.model_registry import ModelRegistry


def test_iou_computation():
    # Identical box -> IoU 1.0
    box_a = [0.1, 0.1, 0.5, 0.5]
    assert pytest.approx(compute_box_iou(box_a, box_a), 0.01) == 1.0

    # Disjoint boxes -> IoU 0.0
    box_b = [0.6, 0.6, 0.9, 0.9]
    assert compute_box_iou(box_a, box_b) == 0.0

    # Partial overlap
    box_c = [0.3, 0.3, 0.7, 0.7]
    iou = compute_box_iou(box_a, box_c)
    assert 0.0 < iou < 1.0

    # Mean IoU
    mean_iou = compute_mean_iou([box_a], [box_a])
    assert pytest.approx(mean_iou, 0.01) == 1.0


def test_semantic_similarity_and_perplexity():
    # Exact match
    sim = compute_semantic_similarity("urban area", "urban area", ("urban",))
    assert sim == 1.0

    # Keyword match
    sim_partial = compute_semantic_similarity(
        "Dense urban built-up structures detected.",
        "urban built-up structures",
        ("urban", "built-up", "structures"),
    )
    assert sim_partial >= 0.7

    # Token perplexity
    logprobs = [-0.1, -0.2, -0.15]
    perplexity = compute_token_perplexity(logprobs)
    assert perplexity is not None
    assert 1.0 < perplexity < 2.0


def test_earthdial_variant_resolution():
    reg = ModelRegistry()

    # RGB metadata (3 bands)
    rgb_meta = [{"bands": 3, "band_names": ["red", "green", "blue"]}]
    assert reg.resolve_earthdial_variant(rgb_meta) == "earthdial-4b-rgb"

    # Multispectral metadata (5 bands with NIR/SWIR)
    ms_meta = [{"bands": 5, "band_names": ["red", "green", "blue", "nir", "swir"]}]
    assert reg.resolve_earthdial_variant(ms_meta) == "earthdial-4b-ms"


@pytest.mark.asyncio
async def test_vrsbench_suite_execution():
    reg = ModelRegistry()
    response = await run_vrsbench_suite(reg, model_variant="auto")

    assert response.summary.total_samples > 0
    assert response.summary.passed_samples > 0
    assert response.summary.accuracy_percent >= 80.0
    assert response.summary.optical_accuracy >= 80.0
    assert response.summary.multispectral_accuracy >= 80.0
    assert "earthdial-4b-rgb" in response.summary.model_variants_evaluated
    assert "earthdial-4b-ms" in response.summary.model_variants_evaluated
    assert len(response.results) == response.summary.total_samples


def test_benchmark_endpoints():
    client = TestClient(app)

    # Test dataset listing
    ds_resp = client.get("/api/v1/benchmarks/datasets")
    assert ds_resp.status_code == 200
    datasets = ds_resp.json()
    assert any(d["name"] == "vrsbench_sample_split" for d in datasets)

    # Test evaluation execution
    eval_resp = client.post(
        "/api/v1/benchmarks/evaluate",
        json={"dataset_name": "vrsbench_sample_split", "model_variant": "auto", "max_samples": 3},
    )
    assert eval_resp.status_code == 200
    data = eval_resp.json()
    assert data["summary"]["total_samples"] == 3
    assert "run_id" in data
