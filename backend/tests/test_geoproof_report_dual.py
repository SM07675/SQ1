from __future__ import annotations

import tempfile
from pathlib import Path
from PIL import Image
from app.services.report import write_pdf_report, write_manifest


def _create_dummy_image(path: Path) -> Path:
    img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    img.save(path)
    return path


def test_write_pdf_report_single_image_vegetation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        _create_dummy_image(out_dir / "preview_1.png")
        _create_dummy_image(out_dir / "water_grounding_mask.png")

        payload = {
            "result_id": "00000000-0000-0000-0000-000000000001",
            "generated_at": "2026-09-05T22:00:00.000Z",
            "query": "Where is vegetation concentrated?",
            "mode": "standard_image_analysis",
            "task_plan": {"task": "grounding", "target": "vegetation"},
            "assets": [
                {
                    "filename": "forest_scene.png",
                    "width": 1024,
                    "height": 1024,
                    "bands": 3,
                    "dtype": "uint8",
                    "crs": None,
                    "nodata_percent": 0.0,
                }
            ],
            "quality": {"score": 0.94, "passed": True},
            "verdict": {
                "status": "SUPPORTED",
                "confidence": 0.82,
                "answer": "Vegetation is concentrated mainly across the western and central portions of the observed area.",
                "limitations": ["No CRS present in source raster."],
                "confidence_breakdown": {
                    "calibration_mode": "temperature_scaled_platt",
                    "expected_calibration_error": 0.038,
                    "ensemble_agreement": 0.92,
                },
            },
            "evidence": [
                {
                    "producer": "optical_water_grounding_engine_v2",
                    "kind": "mask",
                    "confidence": 0.85,
                    "raw_score": 0.88,
                    "metrics": {"largest_coverage_percent": 34.2, "region_count": 3},
                },
                {
                    "producer": "remoteclip_hierarchical_retriever_v1",
                    "kind": "patch_scores",
                    "confidence": 0.80,
                    "raw_score": 0.82,
                    "metrics": {},
                },
            ],
            "trace": [
                {"step": 1, "component": "typed_planner", "action": "Decompose natural language query", "status": "ok", "duration_ms": 0.8},
                {"step": 2, "component": "raster_validator", "action": "Verify pixel bounds and bands", "status": "ok", "duration_ms": 115.0},
                {"step": 3, "component": "remoteclip_retriever", "action": "Semantic patch embeddings", "status": "ok", "duration_ms": 4120.0},
                {"step": 4, "component": "geoproof", "action": "Cross-witness consensus arbitration", "status": "supported", "duration_ms": 0.9},
            ],
        }

        manifest = write_manifest(out_dir, payload)
        assert manifest.exists()

        pdf_path = write_pdf_report(out_dir, payload)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 10000


def test_write_pdf_report_bi_temporal_change():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        _create_dummy_image(out_dir / "preview_1.png")
        _create_dummy_image(out_dir / "preview_2.png")
        _create_dummy_image(out_dir / "semantic_change_mask.png")

        payload = {
            "result_id": "00000000-0000-0000-0000-000000000002",
            "generated_at": "2026-09-05T22:05:00.000Z",
            "query": "Has urban built-up area expanded between the two dates?",
            "mode": "bi_temporal_change",
            "task_plan": {"task": "bi_temporal_change", "target": "built-up"},
            "assets": [
                {
                    "filename": "t1_baseline.tif",
                    "width": 1024,
                    "height": 1024,
                    "bands": 3,
                    "dtype": "uint8",
                    "crs": 32643,
                    "nodata_percent": 0.0,
                },
                {
                    "filename": "t2_followup.tif",
                    "width": 1024,
                    "height": 1024,
                    "bands": 3,
                    "dtype": "uint8",
                    "crs": 32643,
                    "nodata_percent": 0.0,
                },
            ],
            "quality": {"score": 0.98, "passed": True},
            "verdict": {
                "status": "SUPPORTED",
                "confidence": 0.89,
                "answer": "Surface modifications were detected across the observation interval, with new development in the highlighted zones.",
                "limitations": [],
                "confidence_breakdown": {
                    "calibration_mode": "temperature_scaled_platt",
                    "expected_calibration_error": 0.025,
                    "ensemble_agreement": 0.95,
                },
            },
            "evidence": [
                {
                    "producer": "ssim_color_difference_detector_v1",
                    "kind": "mask",
                    "confidence": 0.88,
                    "raw_score": 0.91,
                    "metrics": {"changed_percent": 18.7, "region_count": 5},
                },
                {
                    "producer": "tinycd_siamese_change_witness_v1",
                    "kind": "mask",
                    "confidence": 0.90,
                    "raw_score": 0.92,
                    "metrics": {"changed_percent": 17.9},
                },
            ],
            "trace": [
                {"step": 1, "component": "typed_planner", "action": "Plan bi-temporal task", "status": "ok", "duration_ms": 1.2},
                {"step": 2, "component": "raster_validator", "action": "Check CRS and resolution alignment", "status": "ok", "duration_ms": 85.0},
                {"step": 3, "component": "baseline_change_detector", "action": "Calculate structural difference map", "status": "ok", "duration_ms": 320.0},
                {"step": 4, "component": "geoproof", "action": "Fuse change evidence", "status": "supported", "duration_ms": 0.5},
            ],
        }

        pdf_path = write_pdf_report(out_dir, payload)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 10000


def test_write_pdf_report_optical_sar_fusion():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        _create_dummy_image(out_dir / "preview_1.png")
        _create_dummy_image(out_dir / "sar_db_preview.png")
        _create_dummy_image(out_dir / "sensor_agreement.png")

        payload = {
            "result_id": "00000000-0000-0000-0000-000000000003",
            "generated_at": "2026-09-05T22:10:00.000Z",
            "query": "Identify high-confidence urban clusters corroborated by optical and SAR radar imagery.",
            "mode": "multimodal_sar_fusion",
            "task_plan": {"task": "optical_sar", "target": "built-up"},
            "assets": [
                {
                    "filename": "opt_scene.png",
                    "width": 512,
                    "height": 512,
                    "bands": 3,
                    "dtype": "uint8",
                    "crs": None,
                    "nodata_percent": 0.0,
                },
                {
                    "filename": "sar_scene.png",
                    "width": 512,
                    "height": 512,
                    "bands": 1,
                    "dtype": "uint8",
                    "crs": None,
                    "nodata_percent": 0.0,
                },
            ],
            "quality": {"score": 0.95, "passed": True},
            "verdict": {
                "status": "SUPPORTED",
                "confidence": 0.93,
                "answer": "Both optical reflectance and SAR radar observations corroborate the presence of the identified feature.",
                "limitations": ["No CRS metadata provided."],
                "confidence_breakdown": {
                    "calibration_mode": "temperature_scaled_platt",
                    "expected_calibration_error": 0.021,
                    "ensemble_agreement": 0.97,
                },
            },
            "evidence": [
                {
                    "producer": "croma_cross_attention_fusion",
                    "kind": "multimodal_mask",
                    "confidence": 0.94,
                    "raw_score": 0.96,
                    "metrics": {"confirmed_percent": 42.1, "region_count": 2},
                }
            ],
            "trace": [
                {"step": 1, "component": "typed_planner", "action": "Plan optical-SAR fusion", "status": "ok", "duration_ms": 0.9},
                {"step": 2, "component": "croma_fusion_pipeline", "action": "Cross-attention joint feature fusion", "status": "ok", "duration_ms": 1450.0},
                {"step": 3, "component": "geoproof", "action": "Arbitrate sensor consensus", "status": "supported", "duration_ms": 0.7},
            ],
        }

        pdf_path = write_pdf_report(out_dir, payload)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 10000
