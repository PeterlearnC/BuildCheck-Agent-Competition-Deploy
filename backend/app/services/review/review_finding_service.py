"""Construct deterministic C.3 findings from server-rebuilt C.2 authority."""

import hashlib
import re
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.compliance_comparison import ComplianceComparisonRequest
from app.schemas.review_finding import (
    FINDING_VERSION,
    PlanCitation,
    PlanFactCitation,
    ReviewFinding,
    ReviewFindingResponse,
    StandardCitation,
    deterministic_finding_id,
    deterministic_finding_summary,
)
from app.schemas.standards_scope import StandardScope
from app.services.review.compliance_comparison_service import ComplianceComparisonService
from app.services.review.compliance_review_service import ComplianceReviewService
from app.services.review.review_unit_service import (
    ReviewDocumentNotFoundError,
    ReviewUnitService,
)


class ReviewFindingError(ValueError):
    pass


class ReviewFindingAuthorityUnavailableError(ReviewFindingError):
    pass


class ReviewFindingSourceChangedError(ReviewFindingError):
    pass


class ReviewFindingAuthorityMismatchError(RuntimeError):
    pass


class ReviewFindingService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        review_unit_service: ReviewUnitService | None = None,
        compliance_review_service: ComplianceReviewService | None = None,
        comparison_service: ComplianceComparisonService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.review_unit_service = review_unit_service or ReviewUnitService(
            settings=self.settings
        )
        self.compliance_review_service = (
            compliance_review_service or ComplianceReviewService()
        )
        self.comparison_service = comparison_service or ComplianceComparisonService()

    def create(
        self, document_id: str, request: ComplianceComparisonRequest
    ) -> ReviewFindingResponse:
        pdf_path = self._require_pdf(document_id)
        pre_sha = self._sha256(pdf_path)

        review_unit = self.review_unit_service.create(
            document_id=document_id,
            page_number=request.page_number,
            source_text=request.source_text,
            char_start=request.char_start,
            retrieval_query=request.retrieval_query,
            standard_scope=StandardScope(standard_ids=request.standard_ids),
        )
        preparation = self.compliance_review_service.review(
            review_unit, top_k=request.top_k
        )
        authority = self.comparison_service.compare(
            preparation,
            evidence_id=request.evidence_id,
            requirement_selector=request.requirement,
            plan_fact_selectors=request.plan_facts,
        )

        post_sha = self._sha256(pdf_path)
        if pre_sha != post_sha:
            raise ReviewFindingSourceChangedError(
                "Stored plan PDF changed during finding reconstruction."
            )
        if authority.comparison is None:
            raise ReviewFindingAuthorityUnavailableError(
                "C.3 requires an authoritative C.2 ComparisonResult."
            )
        if authority.evidence_binding is None or authority.requirement is None:
            raise ReviewFindingAuthorityMismatchError(
                "C.2 returned an incomplete authority chain."
            )

        result = authority.comparison
        unit = authority.preparation.review_unit
        evidence = authority.evidence_binding.evidence
        requirement = authority.requirement
        plan_citation = PlanCitation(
            document_id=unit.document_id,
            document_sha256=post_sha,
            page_number=unit.page_number,
            char_start=unit.char_start,
            char_end=unit.char_end,
            source_text=unit.source_text,
            source_text_sha256=unit.source_text_sha256,
            review_unit_id=unit.review_unit_id,
            plan_facts=[
                PlanFactCitation(
                    plan_fact_id=fact.plan_fact_id,
                    char_start=fact.char_start,
                    char_end=fact.char_end,
                    source_text=fact.source_text,
                    source_text_sha256=fact.source_text_sha256,
                    normalized_value=fact.normalized_value,
                    unit=fact.unit,
                )
                for fact in authority.plan_facts
            ],
        )
        if evidence.source_page_start is None or evidence.source_page_end is None:
            raise ReviewFindingAuthorityMismatchError(
                "C.3 standard citations require a grounded physical page interval."
            )
        standard_citation = StandardCitation(
            standard_id=evidence.standard_id,
            standard_code=evidence.standard_code,
            canonical_standard_code=evidence.canonical_standard_code,
            article_id=evidence.article_id,
            article_number=evidence.article_number,
            page_start=evidence.source_page_start,
            page_end=evidence.source_page_end,
            evidence_id=evidence.id,
            source_checksum=evidence.source_checksum,
            requirement_id=requirement.requirement_id,
            requirement_char_start=requirement.requirement_char_start,
            requirement_char_end=requirement.requirement_char_end,
            requirement_text=requirement.requirement_text,
            requirement_text_sha256=requirement.requirement_text_sha256,
        )
        finding = ReviewFinding(
            finding_id=deterministic_finding_id(
                comparison_id=result.comparison_id,
                decision=result.decision,
                reason_code=result.reason_code,
                decision_scope=result.decision_scope,
                document_sha256=post_sha,
            ),
            finding_version=FINDING_VERSION,
            document_id=unit.document_id,
            document_sha256=post_sha,
            comparison_id=result.comparison_id,
            decision=result.decision,
            decision_scope=result.decision_scope,
            reason_code=result.reason_code,
            review_unit_id=unit.review_unit_id,
            evidence_id=evidence.id,
            requirement_id=requirement.requirement_id,
            plan_fact_ids=result.plan_facts_used,
            plan_citation=plan_citation,
            standard_citation=standard_citation,
            summary=deterministic_finding_summary(authority),
            missing_information=result.missing_information,
            authoritative_comparison_response=authority,
        )
        return ReviewFindingResponse(finding=finding)

    def _require_pdf(self, document_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", document_id):
            raise ReviewDocumentNotFoundError("Stored plan document was not found.")
        path = self.settings.upload_dir / f"{document_id}.pdf"
        if not path.is_file():
            raise ReviewDocumentNotFoundError("Stored plan document was not found.")
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ReviewFindingSourceChangedError(
                "Stored plan PDF could not be read consistently."
            ) from exc
        return digest.hexdigest()
