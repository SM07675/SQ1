from __future__ import annotations

import math
from typing import Any

from satquery_engine.schemas import ConfidenceBreakdown, EvidenceItem, QualityReport


# No fitted calibration data is installed.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_BENCHMARK_ECE = None


def probability_from_logprobs(logprobs: list[float], temperature: float = DEFAULT_TEMPERATURE) -> float | None:
    if not logprobs:
        return None
    clipped = [max(-20.0, min(0.0, float(value))) for value in logprobs]
    raw_prob = math.exp(sum(clipped) / len(clipped))
    # Platt temperature scaling calibration: p_calibrated = 1 / (1 + exp(-logit / T))
    if raw_prob <= 0.0 or raw_prob >= 1.0:
        return raw_prob
    logit = math.log(raw_prob / (1.0 - raw_prob))
    calibrated = 1.0 / (1.0 + math.exp(-logit / temperature))
    return max(0.0, min(1.0, calibrated))


def calculate_expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 5,
) -> float:
    """Calculates Expected Calibration Error (ECE) across prediction probability bins."""
    if not confidences or len(confidences) != len(correctness):
        return DEFAULT_BENCHMARK_ECE

    n = len(confidences)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        bin_indices = [
            i for i, c in enumerate(confidences)
            if (low <= c < high) or (b == n_bins - 1 and low <= c <= high)
        ]
        if not bin_indices:
            continue
        bin_size = len(bin_indices)
        bin_acc = sum(1 for i in bin_indices if correctness[i]) / bin_size
        bin_conf = sum(confidences[i] for i in bin_indices) / bin_size
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return round(ece, 4)


def estimate_confidence_intervals(
    score: float,
    n_evidence: int,
    quality_score: float,
    alpha: float = 0.05,
) -> list[float]:
    """Computes Wilson/Wald binomial confidence interval bounds [low, high]."""
    effective_n = max(3, n_evidence * 4 + int(quality_score * 8))
    # Standard error estimate
    se = math.sqrt(max(1e-6, score * (1.0 - score) / effective_n))
    z = 1.96  # 95% confidence interval

    low = max(0.0, score - z * se)
    high = min(1.0, score + z * se)
    return [round(low, 3), round(high, 3)]


def confidence_breakdown(quality: QualityReport, evidence: list[EvidenceItem]) -> ConfidenceBreakdown:
    # A transparent evidence-strength heuristic, not a calibrated probability.
    scores=[]
    for item in evidence:
        value=item.confidence
        if item.kind=="vlm": value=0.0
        if item.kind=="buildings" and (item.metrics.get("benchmark_f1") is None or item.metrics["benchmark_f1"]<.6):
            value=min(value,.35)
        if "proxy" in str(item.metrics.get("method","")).lower(): value=min(value,.25)
        scores.append(value)
    strength = min(scores) if scores else 0.0
    alignment = float(quality.checks.get("alignment_score", 1.0))
    penalty = min(.5, .04 * len(quality.warnings))
    score = strength * quality.score * alignment * (1-penalty) if quality.compatible else 0.0
    return ConfidenceBreakdown(input_quality=quality.score, spatial_alignment=alignment,
        evidence_strength=strength, ensemble_agreement=0.0, warning_penalty=penalty,
        final_score=round(max(0.,min(1.,score)),4), is_calibrated=False,
        calibration_mode="weakest_evidence_with_domain_caps_times_quality_alignment_warning_v2",
        confidence_interval=[], expected_calibration_error=None)
