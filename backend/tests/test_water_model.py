from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.config import settings
from app.services.model_registry import ModelRegistry
from app.services.spectral import extract_water_grounding
from app.services.water_model import (
    get_water_config,
    get_water_metrics,
    get_water_model,
    is_water_model_available,
    predict_water_mask,
    run_windowed_inference,
)


def test_water_model_availability_and_weights_loading() -> None:
    assert is_water_model_available() is True, "Water model checkpoint should be found and enabled"
    model = get_water_model()
    assert model is not None, "get_water_model() returned None"

    # Test forward pass with 9 multispectral channels and 6 spectral indices
    x9 = torch.randn(1, 9, 512, 512)
    x_spec = torch.randn(1, 6, 512, 512)
    with torch.no_grad():
        logits = model(x9, x_spec)

    assert logits.shape == (1, 2, 512, 512), f"Unexpected logits shape: {logits.shape}"


def test_water_model_availability_and_config():
    assert is_water_model_available() is True
    cfg = get_water_config()
    assert cfg.get("architecture") == "SatlasWaterNet"
    assert round(cfg.get("postprocess", {}).get("threshold", 0.0), 2) == 0.60

    metrics = get_water_metrics()
    assert metrics["precision"] > 0.90, f"Expected test precision > 0.90, got {metrics['precision']}"
    assert metrics["f1"] > 0.80, f"Expected test F1 > 0.80, got {metrics['f1']}"
    assert metrics["dark_land_fpr"] < 0.01, f"Expected dark-land FPR < 0.01, got {metrics['dark_land_fpr']}"


def test_windowed_inference_blending() -> None:
    model = get_water_model()
    assert model is not None

    h, w = 600, 700
    bb_cube = np.random.uniform(0.0, 1.0, (9, h, w)).astype("float32")
    sp_cube = np.random.uniform(-1.0, 1.0, (6, h, w)).astype("float32")

    prob_map = run_windowed_inference(model, bb_cube, sp_cube, tile_size=512, overlap=128)
    assert prob_map.shape == (h, w)
    assert np.all(prob_map >= 0.0) and np.all(prob_map <= 1.0)


def test_predict_water_mask_on_demo_image(tmp_path: Path) -> None:
    demo_path = Path(__file__).resolve().parent.parent.parent / "data" / "demo_optical_water.tif"
    assert demo_path.exists(), f"Demo optical file not found at {demo_path}"

    out_dir = tmp_path / "water_dl_test"
    res = predict_water_mask(demo_path, out_dir, query="Highlight the largest water body.", max_size=512)
    assert res is not None
    assert res["target"] == "water"
    assert res["producer"] == "satquery_waternet_swinv2"
    assert res["is_deep_learning"] is True
    assert (out_dir / "water_probability.png").exists()
    assert (out_dir / "water_mask.png").exists()
    assert (out_dir / "water_grounding_mask.png").exists()
    assert (out_dir / "water_regions.geojson").exists()

    fc = json.loads((out_dir / "water_regions.geojson").read_text(encoding="utf-8"))
    assert fc.get("type") == "FeatureCollection"


def test_extract_water_grounding_uses_deep_learning(tmp_path: Path) -> None:
    demo_path = Path(__file__).resolve().parent.parent.parent / "data" / "demo_optical_water.tif"
    out_dir = tmp_path / "extract_grounding_test"
    res = extract_water_grounding(demo_path, out_dir, query="Identify water bodies.")
    assert res is not None
    assert res["producer"] == "satquery_waternet_swinv2"
    assert res["confidence"] >= 0.70


@pytest.mark.asyncio
async def test_model_registry_satquery_waternet(tmp_path: Path) -> None:
    demo_path = Path(__file__).resolve().parent.parent.parent / "data" / "demo_optical_water.tif"
    registry = ModelRegistry()

    assert "satquery-waternet" in registry.specs
    spec = registry.specs["satquery-waternet"]
    assert spec.available is True

    caps = registry.capabilities()
    water_cap = next((c for c in caps if c["name"] == "satquery-waternet"), None)
    assert water_cap is not None
    assert water_cap["available"] is True

    invoke_res = await registry.invoke(
        "satquery-waternet",
        {
            "image_path": str(demo_path),
            "output_dir": str(tmp_path / "registry_water_out"),
            "query": "Delineate water bodies.",
        },
    )
    assert invoke_res.get("available") is True
    assert invoke_res.get("model") == "satquery-waternet"
    assert "answer" in invoke_res
