"""Document analysis endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.analysis import DocumentAnalysisResponse
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


router = APIRouter(prefix="/documents", tags=["document analysis"])


@router.post("/{document_id}/analyze", response_model=DocumentAnalysisResponse)
def analyze_document(
    document_id: str,
    force: bool = Query(default=False),
    llm_service: LLMService = Depends(get_llm_service),
) -> DocumentAnalysisResponse:
    service = DocumentAnalysisService(llm_service)
    pdf_path = service.find_pdf(document_id)
    if pdf_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    try:
        if not force:
            cached = service.load_cache(document_id)
            if cached is not None:
                return DocumentAnalysisResponse(
                    document_id=document_id, cached=True, analysis=cached
                )

        analysis = service.analyze(pdf_path)
        service.save_cache(document_id, analysis)
        return DocumentAnalysisResponse(document_id=document_id, analysis=analysis)
    except (PDFNoExtractableTextError, PDFTextExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except (LLMCallError, LLMResponseError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except AnalysisCacheError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
