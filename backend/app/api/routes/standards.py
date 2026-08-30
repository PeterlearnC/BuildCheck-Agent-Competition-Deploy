"""HTTP endpoints for deterministic standards ingestion and parsing."""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.core.config import get_settings
from app.schemas.standards import (
    StandardArticlesResponse,
    StandardDetailResponse,
    StandardListResponse,
    StandardParseResponse,
    StandardParseSummary,
    StandardRegistryResponse,
    StandardUploadResponse,
)
from app.schemas.standards_retrieval import StandardSearchRequest, StandardSearchResponse
from app.services.pdf_service import (
    InvalidPDFError,
    PDFNoExtractableTextError,
    PDFTextExtractionError,
)
from app.services.standards.standard_document_service import (
    StandardDocumentService,
    StandardNotFoundError,
    StandardParseFailure,
)
from app.services.retrieval.standards_search_service import (
    StandardsSearchService,
    UnknownStandardsError,
)
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepositoryError


router = APIRouter(prefix="/standards", tags=["standards"])


def get_standard_document_service() -> StandardDocumentService:
    return StandardDocumentService()


def _is_pdf(filename: str, content_type: str | None) -> bool:
    return filename.lower().endswith(".pdf") or (content_type or "").lower() == "application/pdf"


def _not_found(exc: StandardNotFoundError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Standard not found.")


@router.post("/search", response_model=StandardSearchResponse)
def search_standards(request: StandardSearchRequest) -> StandardSearchResponse:
    try:
        return StandardsSearchService().search(request)
    except UnknownStandardsError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/registry", response_model=StandardRegistryResponse)
def list_standard_registry() -> StandardRegistryResponse:
    return StandardRegistryResponse(entries=StandardRegistryService().list_entries())


@router.post("/upload", response_model=StandardUploadResponse)
async def upload_standard(file: UploadFile = File(...)) -> StandardUploadResponse:
    filename = StandardDocumentService._display_filename(file.filename)
    if not _is_pdf(filename, file.content_type):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF standards are accepted.",
        )
    settings = get_settings()
    content = await file.read(settings.max_upload_size + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded PDF is empty.")
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF file exceeds the upload limit.",
        )
    try:
        document = get_standard_document_service().upload(filename, content)
    except InvalidPDFError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (PDFNoExtractableTextError, PDFTextExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="TEXT_NOT_AVAILABLE" if isinstance(exc, PDFNoExtractableTextError) else str(exc),
        ) from exc
    except StandardRepositoryError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return StandardUploadResponse(
        standard_id=document.standard_id,
        source_filename=document.source_filename,
        page_count=document.page_count,
        parse_status=document.parse_status,
    )


@router.post("/{standard_id}/parse", response_model=StandardParseResponse)
def parse_standard(standard_id: str, force: bool = False) -> StandardParseResponse:
    try:
        result, cached = get_standard_document_service().parse(standard_id, force=force)
    except StandardNotFoundError as exc:
        raise _not_found(exc) from exc
    except StandardParseFailure as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return StandardParseResponse(
        standard_id=standard_id,
        cached=cached,
        metadata=result.document,
        summary=StandardParseSummary(chapters=len(result.chapters), articles=len(result.articles)),
    )


@router.get("/{standard_id}", response_model=StandardDetailResponse)
def get_standard(standard_id: str) -> StandardDetailResponse:
    try:
        return get_standard_document_service().get_detail(standard_id)
    except StandardNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{standard_id}/articles", response_model=StandardArticlesResponse)
def get_standard_articles(
    standard_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    chapter_number: str | None = None,
) -> StandardArticlesResponse:
    try:
        total, articles = get_standard_document_service().get_articles(
            standard_id, limit=limit, offset=offset, chapter_number=chapter_number
        )
    except StandardNotFoundError as exc:
        raise _not_found(exc) from exc
    return StandardArticlesResponse(
        standard_id=standard_id,
        total=total,
        limit=limit,
        offset=offset,
        articles=articles,
    )


@router.get("", response_model=StandardListResponse)
def list_standards() -> StandardListResponse:
    return StandardListResponse(standards=get_standard_document_service().list_details())
