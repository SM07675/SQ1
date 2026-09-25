import math
import pytest

from app.schemas import EvidenceItem, QualityReport
from app.services.confidence import (
    calculate_expected_calibration_error,
    confidence_breakdown,
    estimate_confidence_intervals,
    probability_from_logprobs,
)


def test_probability_from_logprobs_with_temperature():
    logprobs = [-0.1, -0.2, -0.15]
    prob = probability_from_logprobs(logprobs, temperature=1.35)
    assert prob is not None
    assert 0.0 < prob <= 1.0


def test_calculate_expected_calibration_error():
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    correctness = [True, True, True, False, False]
    ece = calculate_expected_calibration_error(confidences, correctness, n_bins=3)
    assert 0.0 <= ece <= 1.0


def test_estimate_confidence_intervals():
    score = 0.85
    intervals = estimate_confidence_intervals(score, n_evidence=3, quality_score=0.90)
    assert len(intervals) == 2
    low, high = intervals
    assert low < score < high
    assert 0.0 <= low and high <= 1.0


def test_confidence_breakdown_calibrated_output():
    quality = QualityReport(
        score=0.95,
        compatible=True,
        blockers=[],
        warnings=[],
        checks={"alignment_score": 0.98},
    )
    evidence = [
        EvidenceItem(
            kind="spectral",
            producer="spectral_toolkit",
            summary="NDWI indicates water body",
            confidence=0.88,
            metrics={"token_probability": 0.92},
        ),
        EvidenceItem(
            kind="vlm",
            producer="earthdial",
            summary="Confirmed water body",
            confidence=0.85,
            metrics={"token_logprobs": [-0.1, -0.08]},
        ),
    ]

    breakdown = confidence_breakdown(quality, evidence)
    assert breakdown.is_calibrated is True
    assert breakdown.calibration_mode == "temperature_scaled_platt"
    assert len(breakdown.confidence_interval) == 2
    assert breakdown.expected_calibration_error is not None
    assert breakdown.final_score >= 0.70


def test_confidence_breakdown_unlabelled_scene():
    quality = QualityReport(
        score=0.90,
        compatible=True,
        blockers=[],
        warnings=[],
        checks={"alignment_score": 0.95},
    )
    evidence = [
        EvidenceItem(
            kind="land_cover_classification",
            producer="satquery_landcover_v1",
            summary="Land cover segmentation",
            confidence=0.82,
            metrics={"mean_pixel_confidence": 0.78},
        ),
    ]

    breakdown = confidence_breakdown(quality, evidence, is_target_domain_labeled=False)
    assert breakdown.is_calibrated is False
    assert breakdown.calibration_mode == "uncalibrated_evidence_strength"
    assert breakdown.expected_calibration_error is None

