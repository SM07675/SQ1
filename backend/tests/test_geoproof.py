from app.schemas import EvidenceItem, QualityReport, VerdictStatus
from app.services.geoproof import verify


def test_abstains_without_semantic_model() -> None:
    verdict = verify(
        quality=QualityReport(score=1, compatible=True),
        evidence=[EvidenceItem(kind="mask", producer="fallback", summary="generic change", confidence=0.7)],
        requires_semantic_model=True,
        semantic_model_available=False,
        answer="Generic change exists, semantic direction unknown.",
    )
    assert verdict.status == VerdictStatus.INSUFFICIENT_EVIDENCE
    assert verdict.confidence < 0.5


def test_blocks_incompatible_inputs() -> None:
    verdict = verify(
        quality=QualityReport(score=0.2, compatible=False, blockers=["CRS mismatch"]),
        evidence=[],
        requires_semantic_model=False,
        semantic_model_available=False,
        answer="",
    )
    assert verdict.status == VerdictStatus.INSUFFICIENT_EVIDENCE
    assert "CRS mismatch" in verdict.contradictions

