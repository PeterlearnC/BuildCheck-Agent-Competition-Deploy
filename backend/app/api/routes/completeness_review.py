"""Completeness review endpoint over an existing V0.2 document analysis."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.completeness_review import CompletenessReviewResponse
from app.services.completeness_review_cache_service import (
    CompletenessReviewCacheError,
    CompletenessReviewCacheService,
)
from app.services.completeness_review_service import CompletenessReviewService
from app.services.document_analysis_service import AnalysisCacheError, DocumentAnalysisService
from app.services.llm_service import (
    LLMCallError,
    LLMConfigurationError,
    LLMResponseError,
    LLMService,
    get_llm_service,
)
from app.services.pdf_service import (
    InvalidPDFError,
    PDFNoExtractableTextError,
    PDFTextExtractionError,
)


router = APIRouter(prefix="/documents", tags=["completeness review"])


@router.post(
    "/{document_id}/review/completeness",
    response_model=CompletenessReviewResponse,
)
def review_document_completeness(
    document_id: str,
    force: bool = Query(default=False),
    llm_service: LLMService = Depends(get_llm_service),
) -> CompletenessReviewResponse:
    analysis_service = DocumentAnalysisService(llm_service)
    pdf_path = analysis_service.find_pdf(document_id)
    if pdf_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    try:
        # Review force never propagates to document analysis. The existing
        # analysis cache is always preferred; analysis runs only when absent.
        analysis = analysis_service.load_cache(document_id)
        if analysis is None:
            analysis = analysis_service.analyze(pdf_path)
            analysis_service.save_cache(document_id, analysis)

        cache_service = CompletenessReviewCacheService(analysis_service.settings)
        if not force:
            cached_review = cache_service.load(document_id, analysis)
            if cached_review is not None:
                return CompletenessReviewResponse(
                    document_id=document_id,
                    cached=True,
                    review=cached_review,
                )

        review = CompletenessReviewService().review(analysis)
        cache_service.save(document_id, analysis, review)
        return CompletenessReviewResponse(
            document_id=document_id,
            cached=False,
            review=review,
        )
    except (PDFNoExtractableTextError, PDFTextExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except (LLMCallError, LLMResponseError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc
    except (AnalysisCacheError, CompletenessReviewCacheError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
