from __future__ import annotations

import logging
from typing import Any

from app.schemas import EvidenceItem, GeoVerdict, QualityReport, StructuredLimitation, VerdictStatus
from app.services.confidence import confidence_breakdown

logger = logging.getLogger(__name__)


def _detect_inter_witness_agreement(evidence: list[EvidenceItem]) -> tuple[list[str], list[str], float]:
    """
    Detect agreement/disagreement between independent witnesses on the same target.

    Returns:
        agreements:    list of agreement descriptions
        disagreements: list of disagreement descriptions
        agreement_bonus: confidence adjustment (positive = agreement, negative = disagreement)
    """
    agreements: list[str] = []
    disagreements: list[str] = []
    bonus = 0.0

    # Group evidence by target type
    water_evidence = [ev for ev in evidence if ev.kind == "water_grounding_evidence"]
    land_evidence = [ev for ev in evidence if ev.kind == "land_cover_classification"]

    # Check water grounding: DL consensus flag
    for wev in water_evidence:
        m = wev.metrics
        if m.get("dl_consensus"):
            agreements.append(
                "Spectral water index and deep learning segmentation independently agree on water locations."
            )
            bonus += 0.03

        if m.get("is_deep_learning") and m.get("water_body_identified"):
            agreements.append(
                "SatlasWaterNet deep learning model confirms water body presence."
            )
            bonus += 0.02

    # Cross-validate water between water grounding and land cover engines
    if water_evidence and land_evidence:
        for wev in water_evidence:
            for lev in land_evidence:
                w_cov = wev.metrics.get("total_coverage_percent", 0.0)
                lc_water = lev.metrics.get("water_percent", 0.0)

                if w_cov > 0 and lc_water > 0:
                    ratio = min(w_cov, lc_water) / max(w_cov, lc_water) if max(w_cov, lc_water) > 0 else 0
                    if ratio > 0.5:
                        agreements.append(
                            f"Water grounding engine ({w_cov:.1f}%) and land cover classifier ({lc_water:.1f}%) "
                            f"show strong agreement on water coverage."
                        )
                        bonus += 0.02
                    elif ratio < 0.2 and abs(w_cov - lc_water) > 10:
                        disagreements.append(
                            f"Water grounding engine ({w_cov:.1f}%) and land cover classifier ({lc_water:.1f}%) "
                            f"disagree substantially on water extent — treat result with caution."
                        )
                        bonus -= 0.03

    # Cross-validate Optical vs SAR radar witnesses
    sar_evidence = [
        ev for ev in evidence
        if ev.kind in ("sar_structural_evidence", "sar_specular_evidence", "croma_cross_attention_fusion", "cross_sensor_agreement")
    ]
    opt_evidence = [
        ev for ev in evidence
        if ev.kind in ("building_detection_evidence", "water_grounding_evidence", "land_cover_classification", "spectral_water_analysis", "optical_scene_evidence")
    ]
    if sar_evidence and opt_evidence:
        sar_supports = any(s.supports_claim is True for s in sar_evidence)
        opt_supports = any(o.supports_claim is True for o in opt_evidence)
        sar_refutes = any(s.supports_claim is False for s in sar_evidence)
        opt_refutes = any(o.supports_claim is False for o in opt_evidence)

        if sar_supports and opt_supports:
            agreements.append(
                "Optical surface reflectance and SAR radar backscatter independently corroborate the observation."
            )
            bonus += 0.04
        elif (sar_supports and opt_refutes) or (opt_supports and sar_refutes):
            disagreements.append(
                "Optical reflectance and SAR radar backscatter exhibit conflicting signatures over candidate regions — claim is disputed."
            )
            bonus -= 0.05

    # Check for multiple supporting witnesses
    supporting = [ev for ev in evidence if ev.supports_claim is True]
    if len(supporting) >= 3:
        agreements.append(
            f"{len(supporting)} independent witnesses all support the analysis claim."
        )
        bonus += 0.02

    return agreements, disagreements, round(max(-0.10, min(0.10, bonus)), 3)


def verify(
    *,
    quality: QualityReport,
    evidence: list[EvidenceItem],
    requires_semantic_model: bool,
    semantic_model_available: bool,
    answer: str,
    limitations: list[str] | None = None,
    contradictions: list[str] | None = None,
    claim_status: VerdictStatus | None = None,
    structured_limitations: list[StructuredLimitation] | None = None,
) -> GeoVerdict:
    final_contradictions: list[str] = list(contradictions or [])
    final_limitations: list[str] = list(limitations or [])
    final_struct_limits: list[StructuredLimitation] = list(structured_limitations or [])
    breakdown = confidence_breakdown(quality, evidence)

    # 1. Fatal Blockers (True unrecoverable failures: corrupted files, 0 readable bands)
    if quality.blockers:
        final_contradictions.extend(quality.blockers)
        return GeoVerdict(
            status=VerdictStatus.INSUFFICIENT_EVIDENCE,
            answer=f"Analysis stopped because hard input constraints failed: {'; '.join(quality.blockers)}.",
            confidence=0.0,
            confidence_kind="hard_input_rejection",
            contradictions=final_contradictions,
            limitations=final_limitations,
            structured_limitations=final_struct_limits,
            confidence_breakdown=breakdown,
        )

    # 2. Critical spatial misalignment (alignment_score < 0.15)
    alignment_score = quality.checks.get("alignment_score")
    if isinstance(alignment_score, (int, float)) and alignment_score < 0.15:
        final_limitations.append("Severe spatial misalignment between images prevents reliable change deduction.")
        return GeoVerdict(
            status=VerdictStatus.INSUFFICIENT_EVIDENCE,
            answer="Analysis cannot produce reliable change evidence because the input images cannot be spatially aligned.",
            confidence=round(float(alignment_score) * 0.2, 3),
            confidence_kind="severe_misalignment_abstention",
            contradictions=final_contradictions,
            limitations=final_limitations,
            structured_limitations=final_struct_limits,
            confidence_breakdown=breakdown,
        )

    # 3. No Evidence Items Generated
    if not evidence:
        return GeoVerdict(
            status=VerdictStatus.INSUFFICIENT_EVIDENCE,
            answer=answer or "The analysis completed but generated no actionable evidence.",
            confidence=0.0,
            confidence_kind="empty_evidence_abstention",
            contradictions=final_contradictions,
            limitations=final_limitations + ["No evidence item was produced."],
            structured_limitations=final_struct_limits,
            confidence_breakdown=breakdown,
        )

    # 4. Inter-witness agreement detection
    agreements, disagreements, agreement_bonus = _detect_inter_witness_agreement(evidence)
    if agreements:
        logger.info("GeoProof inter-witness agreements: %s", agreements)
    if disagreements:
        final_limitations.extend(disagreements)
        logger.warning("GeoProof inter-witness disagreements: %s", disagreements)

    # 5. Check for Disputed Claim (Evidence explicitly refutes directional/presence claim)
    has_refuting_evidence = any(item.supports_claim is False for item in evidence)
    has_supporting_evidence = any(item.supports_claim is True for item in evidence)

    if claim_status == VerdictStatus.DISPUTED or (has_refuting_evidence and not has_supporting_evidence):
        if quality.warnings:
            final_limitations.extend(quality.warnings)
        adjusted_score = round(max(0.0, min(1.0, breakdown.final_score + agreement_bonus)), 3)
        return GeoVerdict(
            status=VerdictStatus.DISPUTED,
            answer=answer,
            confidence=adjusted_score,
            confidence_kind="evidence_weighted_spatial_ensemble_v2",
            contradictions=final_contradictions,
            limitations=final_limitations,
            structured_limitations=final_struct_limits,
            confidence_breakdown=breakdown,
        )

    # 6. Missing required specialist domain model when semantic proof requested
    if requires_semantic_model and not semantic_model_available and claim_status != VerdictStatus.SUPPORTED_WITH_LIMITATIONS:
        final_limitations.append(
            "A domain-adapted semantic/VLM endpoint is required for class identity or directional claims."
        )
        return GeoVerdict(
            status=VerdictStatus.INSUFFICIENT_EVIDENCE,
            answer=answer,
            confidence=round(min(0.49, quality.score * 0.45), 3),
            confidence_kind="semantic_model_unavailable_abstention",
            contradictions=final_contradictions,
            limitations=final_limitations,
            structured_limitations=final_struct_limits,
            confidence_breakdown=breakdown,
        )

    # 7. Supported Claim / General Verified Analysis
    if quality.warnings:
        final_limitations.extend(quality.warnings)

    adjusted_score = round(max(0.0, min(1.0, breakdown.final_score + agreement_bonus)), 3)

    if claim_status == VerdictStatus.SUPPORTED_WITH_LIMITATIONS:
        final_status = VerdictStatus.SUPPORTED_WITH_LIMITATIONS
    elif final_struct_limits or any(
        k in lim.lower()
        for lim in final_limitations
        for k in (
            "spectral index change cannot be computed",
            "physical ndvi calculation is not available",
            "physical ndbi calculation is not available",
            "physical ndwi calculation is not available",
            "required spectral bands are unavailable",
            "croma skipped because required input channels",
            "relative sar backscatter",
            "uncalibrated",
        )
    ):
        final_status = VerdictStatus.SUPPORTED_WITH_LIMITATIONS
    else:
        final_status = VerdictStatus.SUPPORTED

    return GeoVerdict(
        status=final_status,
        answer=answer,
        confidence=adjusted_score,
        confidence_kind="evidence_weighted_spatial_ensemble_v2",
        contradictions=final_contradictions,
        limitations=final_limitations,
        structured_limitations=final_struct_limits,
        confidence_breakdown=breakdown,
    )

