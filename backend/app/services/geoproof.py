from __future__ import annotations

from app.schemas import EvidenceItem, GeoVerdict, QualityReport, VerdictStatus
from app.services.confidence import confidence_breakdown


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
) -> GeoVerdict:
    final_contradictions: list[str] = list(contradictions or [])
    final_limitations: list[str] = list(limitations or [])
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
            confidence_breakdown=breakdown,
        )

    # 4. Check for Disputed Claim (Evidence explicitly refutes directional/presence claim)
    has_refuting_evidence = any(item.supports_claim is False for item in evidence)
    has_supporting_evidence = any(item.supports_claim is True for item in evidence)

    if claim_status == VerdictStatus.DISPUTED or (has_refuting_evidence and not has_supporting_evidence):
        if quality.warnings:
            final_limitations.extend(quality.warnings)
        return GeoVerdict(
            status=VerdictStatus.DISPUTED,
            answer=answer,
            confidence=breakdown.final_score,
            confidence_kind="evidence_weighted_spatial_ensemble_v1",
            contradictions=final_contradictions,
            limitations=final_limitations,
            confidence_breakdown=breakdown,
        )

    # 5. Missing required specialist domain model when semantic proof requested
    if requires_semantic_model and not semantic_model_available:
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
            confidence_breakdown=breakdown,
        )

    # 6. Supported Claim / General Verified Analysis
    if quality.warnings:
        final_limitations.extend(quality.warnings)

    return GeoVerdict(
        status=VerdictStatus.SUPPORTED,
        answer=answer,
        confidence=breakdown.final_score,
        confidence_kind="evidence_weighted_spatial_ensemble_v1",
        contradictions=final_contradictions,
        limitations=final_limitations,
        confidence_breakdown=breakdown,
    )
