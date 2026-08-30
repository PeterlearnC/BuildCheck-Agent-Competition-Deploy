"""Dedicated source-grounded compliance-review preparation endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.compliance_comparison import (
    ComplianceComparisonRequest,
    ComplianceComparisonResponse,
)
from app.schemas.compliance_review import ComplianceReviewRequest, ComplianceReviewResult
from app.schemas.review_finding import ReviewFindingResponse
from app.schemas.standards_scope import StandardScope
from app.services.pdf_service import (
    InvalidPDFError,
    PDFNoExtractableTextError,
    PDFTextExtractionError,
)
from app.services.retrieval.standards_search_service import UnknownStandardsError
from app.services.review.compliance_review_service import (
    ComplianceEvidenceGateError,
    ComplianceReviewService,
)
from app.services.review.compliance_comparison_service import (
    ComplianceComparisonError,
    ComplianceComparisonService,
)
from app.services.review.plan_fact_service import PlanFactError
from app.services.review.review_finding_service import (
    ReviewFindingAuthorityMismatchError,
    ReviewFindingAuthorityUnavailableError,
    ReviewFindingService,
    ReviewFindingSourceChangedError,
)
from app.services.review.requirement_service import RequirementSelectionError
from app.services.review.review_unit_service import (
    ReviewDocumentNotFoundError,
    ReviewPageNotFoundError,
    ReviewSourceSpanError,
    ReviewSourceTextAmbiguousError,
    ReviewSourceTextNotFoundError,
    ReviewUnitService,
)
from app.services.standards.scope_validation_service import ScopeValidationError


router = APIRouter(prefix="/documents", tags=["compliance review"])


def get_review_unit_service() -> ReviewUnitService:
    return ReviewUnitService()


def get_compliance_review_service() -> ComplianceReviewService:
    return ComplianceReviewService()


def get_compliance_comparison_service() -> ComplianceComparisonService:
    return ComplianceComparisonService()


def get_review_finding_service() -> ReviewFindingService:
    return ReviewFindingService()


@router.post(
    "/{document_id}/review/compliance",
    response_model=ComplianceReviewResult,
)
def prepare_compliance_review(
    document_id: str,
    request: ComplianceReviewRequest,
    review_units: ReviewUnitService = Depends(get_review_unit_service),
    reviews: ComplianceReviewService = Depends(get_compliance_review_service),
) -> ComplianceReviewResult:
    try:
        review_unit = review_units.create(
            document_id=document_id,
            page_number=request.page_number,
            source_text=request.source_text,
            char_start=request.char_start,
            retrieval_query=request.retrieval_query,
            standard_scope=StandardScope(standard_ids=request.standard_ids),
        )
        return reviews.review(review_unit, top_k=request.top_k)
    except ReviewDocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ReviewSourceTextAmbiguousError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (
        ReviewPageNotFoundError,
        ReviewSourceTextNotFoundError,
        ReviewSourceSpanError,
        PDFNoExtractableTextError,
        PDFTextExtractionError,
        ScopeValidationError,
        UnknownStandardsError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ComplianceEvidenceGateError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post(
    "/{document_id}/review/compliance/compare",
    response_model=ComplianceComparisonResponse,
)
def compare_compliance_review(
    document_id: str,
    request: ComplianceComparisonRequest,
    review_units: ReviewUnitService = Depends(get_review_unit_service),
    reviews: ComplianceReviewService = Depends(get_compliance_review_service),
    comparisons: ComplianceComparisonService = Depends(
        get_compliance_comparison_service
    ),
) -> ComplianceComparisonResponse:
    """Reconstruct C.1 authority, then perform only supported local comparison."""
    try:
        review_unit = review_units.create(
            document_id=document_id,
            page_number=request.page_number,
            source_text=request.source_text,
            char_start=request.char_start,
            retrieval_query=request.retrieval_query,
            standard_scope=StandardScope(standard_ids=request.standard_ids),
        )
        preparation = reviews.review(review_unit, top_k=request.top_k)
        return comparisons.compare(
            preparation,
            evidence_id=request.evidence_id,
            requirement_selector=request.requirement,
            plan_fact_selectors=request.plan_facts,
        )
    except ReviewDocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ReviewSourceTextAmbiguousError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (
        ReviewPageNotFoundError,
        ReviewSourceTextNotFoundError,
        ReviewSourceSpanError,
        PDFNoExtractableTextError,
        PDFTextExtractionError,
        ScopeValidationError,
        UnknownStandardsError,
        ComplianceComparisonError,
        RequirementSelectionError,
        PlanFactError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ComplianceEvidenceGateError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post(
    "/{document_id}/review/compliance/findings",
    response_model=ReviewFindingResponse,
)
def create_compliance_finding(
    document_id: str,
    request: ComplianceComparisonRequest,
    findings: ReviewFindingService = Depends(get_review_finding_service),
) -> ReviewFindingResponse:
    """Rebuild C.1/C.2 authority, then project one immutable local finding."""
    try:
        return findings.create(document_id, request)
    except ReviewDocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ReviewSourceTextAmbiguousError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (
        ReviewPageNotFoundError,
        ReviewSourceTextNotFoundError,
        ReviewSourceSpanError,
        PDFNoExtractableTextError,
        PDFTextExtractionError,
        ScopeValidationError,
        UnknownStandardsError,
        ComplianceComparisonError,
        RequirementSelectionError,
        PlanFactError,
        ReviewFindingAuthorityUnavailableError,
        ReviewFindingSourceChangedError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (
        ComplianceEvidenceGateError,
        ReviewFindingAuthorityMismatchError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
