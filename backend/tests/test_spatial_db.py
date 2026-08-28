import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repository import Repository


def test_spatial_geometry_indexing_and_bbox_query(tmp_path: Path):
    db_path = tmp_path / "test_spatial.sqlite3"
    repo = Repository(db_path)
    repo.init()

    # Create dummy result and output dir with GeoJSON
    result_id = "test-run-123"
    out_dir = tmp_path / result_id
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_data = {
        "type": "FeatureCollection",
        "properties": {"crs": "EPSG:32643", "producer": "spectral_toolkit_v2"},
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[500000, 2200000], [500500, 2200000], [500500, 2200500], [500000, 2200500], [500000, 2200000]]],
                },
                "properties": {"kind": "water_body", "area_m2": 250000.0},
            }
        ],
    }
    (out_dir / "water_regions.geojson").write_text(json.dumps(geojson_data), encoding="utf-8")

    payload = {
        "result_id": result_id,
        "query": "Detect water body",
        "task_plan": {"task": "single_vqa"},
        "verdict": {"status": "supported", "confidence": 0.92},
    }
    repo.save_result(result_id, payload, "2026-08-27T12:00:00Z", output_dir=out_dir)

    # Query all geometries for result
    geoms = repo.query_geometries(result_id=result_id)
    assert len(geoms) == 1
    assert geoms[0]["evidence_kind"] == "water_body"
    assert geoms[0]["area_m2"] == 250000.0

    # Query with matching BBOX
    match_bbox = repo.query_geometries(bbox=(499900, 2199900, 500600, 2200600))
    assert len(match_bbox) == 1

    # Query with non-matching BBOX
    no_match_bbox = repo.query_geometries(bbox=(600000, 3000000, 601000, 3001000))
    assert len(no_match_bbox) == 0

    # Query analysis runs
    runs = repo.list_analysis_runs()
    assert len(runs) >= 1
    assert runs[0]["result_id"] == result_id
    assert runs[0]["geometries_count"] == 1


def test_spatial_api_endpoints():
    client = TestClient(app)

    # Query runs
    runs_res = client.get("/api/v1/spatial/runs")
    assert runs_res.status_code == 200
    assert isinstance(runs_res.json(), list)

    # Query geometries
    geom_res = client.get("/api/v1/spatial/geometries?limit=10")
    assert geom_res.status_code == 200
    assert isinstance(geom_res.json(), list)
