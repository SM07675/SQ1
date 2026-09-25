import numpy as np
from pathlib import Path
from PIL import Image
from satquery_engine.services.spectral import extract_water_grounding


def test_urban_false_positive_rejection(tmp_path: Path):
    """
    Test that an urban coastal scene with dark rooftops, roads, and high-frequency
    building shadows correctly isolates the continuous ocean body on the left
    and completely suppresses the urban false positives on the right.
    """
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Left side (x: 0 to 95): Large contiguous water body (dark blue/cyan tone, smooth)
    # Slight subtle water texture
    water_base = np.array([20, 65, 88], dtype=np.float32)
    noise = np.random.normal(0, 1.5, (h, 96, 3))
    water_pixels = np.clip(water_base + noise, 0, 255).astype(np.uint8)
    img[:, :96] = water_pixels

    # 2. Right side (x: 96 to 255): Dense urban scene
    # Sand/concrete ground
    img[:, 96:] = [180, 175, 165]

    # Add dark asphalt roads
    img[60:70, 96:] = [35, 38, 42]
    img[180:190, 96:] = [30, 32, 36]
    img[:, 170:180] = [28, 30, 35]

    # Add dense buildings with dark rooftops and sharp edges
    for r in range(10, 240, 30):
        for c in range(105, 245, 35):
            # Building roof (dark grey/blue-grey tiles that naive detectors confuse with water)
            roof_color = [30, 45, 55] if (r + c) % 2 == 0 else [45, 45, 50]
            img[r:r+20, c:c+22] = roof_color
            # Building shadow (very dark strip)
            img[r+20:r+24, c:c+22] = [12, 14, 18]

    # Save synthetic scene
    scene_path = tmp_path / "urban_coastal.png"
    Image.fromarray(img).save(scene_path)

    # Run water grounding
    out_dir = tmp_path / "water_out"
    res = extract_water_grounding(scene_path, out_dir, query="Highlight the largest water body.")

    assert res is not None
    assert res["water_body_identified"] is True
    assert res["is_spectral"] is False
    assert res["primary_region"] is not None
    
    # Primary region must be the large ocean on the left
    assert res["largest_coverage_percent"] > 25.0
    assert "left" in res["location_description"].lower() or "west" in res["location_description"].lower()
    
    # Check that mask does NOT mark the urban right side as water
    mask = np.array(Image.open(res["mask_path"]))
    left_water_detected = np.sum(mask[:, :96] > 0)
    right_urban_detected = np.sum(mask[:, 96:] > 0)
    
    # Left water should be largely detected
    assert left_water_detected > (h * 96 * 0.70)
    # Right urban false positives should be heavily suppressed (< 1% of urban area)
    assert right_urban_detected < (h * 160 * 0.01)


def test_insufficient_water_scene_abstains(tmp_path: Path):
    """
    Test that an image containing only desert and buildings without any water
    safely abstains (water_body_identified = False).
    """
    h, w = 128, 128
    # Desert / dry land with a few dark shadows
    img = np.full((h, w, 3), [210, 185, 140], dtype=np.uint8)
    # Small shadow
    img[40:45, 40:48] = [20, 20, 25]
    
    scene_path = tmp_path / "desert_dry.png"
    Image.fromarray(img).save(scene_path)
    
    out_dir = tmp_path / "water_out_dry"
    res = extract_water_grounding(scene_path, out_dir, query="Highlight the largest water body.")
    
    assert res is not None
    assert res["water_body_identified"] is False
    assert "No coherent, continuous water body" in res["findings"]


def test_case_a_coastal_river_regression(tmp_path: Path):
    """Regression test on Case A (coastal + winding river).

    Expectations:
    - Sea detected
    - Sinuous river detected
    - Land/canopy largely excluded
    """
    case_a_path = Path(__file__).resolve().parents[1] / "artifacts/assets/1db704bc-114a-4b81-a117-188f0ccc9171/original.tif"
    if not case_a_path.is_file():
        return  # Skip if artifact asset is not present locally

    from satquery_engine.services.water_engine import execute_water_pipeline
    out = tmp_path / "water_case_a"
    res = execute_water_pipeline(case_a_path, out)

    assert res is not None
    assert res["water_body_identified"] is True
    # Sea is a major water body (> 10% of scene)
    assert res["coverage_percent"] > 10.0
    # Must identify primary region with high confidence
    assert res["evidence_strength"] >= 0.70
    assert (out / "water_mask.tif").is_file()
    assert (out / "water.geojson").is_file()


def test_case_b_city_river_regression(tmp_path: Path):
    """Regression test on Case B (city + large river + reservoir).

    Expectations:
    - Continuous main river detected
    - Reservoirs/lakes detected
    - Urban dark false positives suppressed
    """
    case_b_path = Path(__file__).resolve().parents[1] / "artifacts/assets/d8148181-b0eb-44bb-8dcb-e54af122f923/original.jpeg"
    if not case_b_path.is_file():
        return  # Skip if artifact asset is not present locally

    from satquery_engine.services.water_engine import execute_water_pipeline
    out = tmp_path / "water_case_b"
    res = execute_water_pipeline(case_b_path, out)

    assert res is not None
    assert res["water_body_identified"] is True
    # River + reservoir detected, but urban land is NOT falsely classified as water (< 5% total water)
    assert res["selected_pixels"] > 500
    assert res["coverage_percent"] < 5.0
    assert (out / "water_mask.tif").is_file()
    assert (out / "water.geojson").is_file()

