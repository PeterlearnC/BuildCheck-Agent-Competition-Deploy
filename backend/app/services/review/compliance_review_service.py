"""Deterministic ACCEPT-only evidence gate for compliance review preparation."""

from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.review_unit import ReviewUnit
from app.schemas.standards import ArticleType
from app.schemas.standards_retrieval import (
    RetrievalDecision,
    StandardSearchRequest,
)
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.review.review_unit_service import ReviewUnitService
from app.services.standards.scope_validation_service import ScopeValidationService


class ComplianceEvidenceGateError(RuntimeError):
    pass


class ComplianceReviewService:
    def __init__(
        self,
        search_service: StandardsSearchService | None = None,
        scope_validator: ScopeValidationService | None = None,
        review_unit_service: ReviewUnitService | None = None,
    ) -> None:
        self.search_service = search_service or StandardsSearchService()
        self.scope_validator = scope_validator or ScopeValidationService()
        self.review_unit_service = review_unit_service or ReviewUnitService()

    def review(self, review_unit: ReviewUnit, *, top_k: int = 5) -> ComplianceReviewResult:
        review_unit = self.review_unit_service.verify(review_unit)
        validated_scope = self.scope_validator.validate_scope(
            review_unit.standard_scope
        )
        response = self.search_service.search(
            StandardSearchRequest(
                query=review_unit.retrieval_query,
                top_k=top_k,
                standard_ids=validated_scope.scope.standard_ids,
                article_types=[ArticleType.NORMATIVE],
            )
        )

        if response.retrieval_decision != RetrievalDecision.ACCEPT:
            return ComplianceReviewResult(
                review_unit=review_unit,
                status=ComplianceReviewStatus.INSUFFICIENT_EVIDENCE,
                retrieval_decision=response.retrieval_decision,
                evidence_bindings=[],
                reason=response.acceptance_reason,
                scope_warnings=list(validated_scope.warnings),
            )
        if not response.hits:
            raise ComplianceEvidenceGateError(
                "ACCEPT retrieval returned no authoritative evidence hits."
            )

        bindings: list[ReviewEvidenceBinding] = []
        for hit in response.hits:
            evidence = hit.evidence
            if evidence.source_page_start is None or evidence.source_page_end is None:
                raise ComplianceEvidenceGateError(
                    "ACCEPT evidence is missing a standard source-page interval."
                )
            bindings.append(
                ReviewEvidenceBinding(
                    review_unit_id=review_unit.review_unit_id,
                    document_id=review_unit.document_id,
                    plan_page_number=review_unit.page_number,
                    plan_char_start=review_unit.char_start,
                    plan_char_end=review_unit.char_end,
                    plan_source_text=review_unit.source_text,
                    plan_source_text_sha256=review_unit.source_text_sha256,
                    retrieval_decision=RetrievalDecision.ACCEPT,
                    standard_evidence_id=evidence.id,
                    standard_id=evidence.standard_id,
                    standard_article_number=evidence.article_number,
                    standard_page_start=evidence.source_page_start,
                    standard_page_end=evidence.source_page_end,
                    evidence=evidence,
                )
            )
        return ComplianceReviewResult(
            review_unit=review_unit,
            status=ComplianceReviewStatus.NEEDS_COMPARISON,
            retrieval_decision=RetrievalDecision.ACCEPT,
            evidence_bindings=bindings,
            reason=response.acceptance_reason,
            scope_warnings=list(validated_scope.warnings),
        )
