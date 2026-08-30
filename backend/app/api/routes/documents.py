"""Document upload endpoint."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.document import DocumentUploadResponse, PageText
from app.services.pdf_service import (
    InvalidPDFError,
    PDFNoExtractableTextError,
    PDFService,
    PDFTextExtractionError,
)


router = APIRouter(prefix="/documents", tags=["documents"])
pdf_service = PDFService()


def _display_filename(filename: str | None) -> str:
    """Keep only the display name, even if a client submits a path."""
    cleaned = (filename or "document.pdf").replace("\\", "/")
    return cleaned.rsplit("/", maxsplit=1)[-1] or "document.pdf"


def _is_pdf(filename: str, content_type: str | None) -> bool:
    return filename.lower().endswith(".pdf") or (content_type or "").lower() == "application/pdf"


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    settings = get_settings()
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(settings.upload_chunk_size):
                size += len(chunk)
                if size > settings.max_upload_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="PDF file exceeds the 30 MB upload limit.",
                    )
                output.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save the uploaded PDF.",
        ) from exc
    finally:
        await upload.close()

    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty.",
        )
    return size


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    settings = get_settings()
    original_filename = _display_filename(file.filename)
    if not _is_pdf(original_filename, file.content_type):
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted (application/pdf or .pdf).",
        )

    document_id = str(uuid4())
    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to prepare the PDF upload directory.",
        ) from exc
    saved_path = settings.upload_dir / f"{document_id}.pdf"
    file_size = await _save_upload(file, saved_path)

    try:
        result = pdf_service.extract_text(saved_path)
    except InvalidPDFError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PDFNoExtractableTextError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except PDFTextExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return DocumentUploadResponse(
        document_id=document_id,
        filename=original_filename,
        content_type="application/pdf",
        file_size=file_size,
        page_count=result.page_count,
        char_count=result.char_count,
        text_preview=result.preview(settings.text_preview_length),
        pages=[PageText(page_number=page.page_number, text=page.text) for page in result.pages],
    )
