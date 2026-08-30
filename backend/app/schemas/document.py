"""Request-independent schemas returned by the document API."""

from pydantic import BaseModel


class PageText(BaseModel):
    page_number: int
    text: str


class DocumentUploadResponse(BaseModel):
    success: bool = True
    document_id: str
    filename: str
    content_type: str
    file_size: int
    page_count: int
    char_count: int
    text_preview: str
    pages: list[PageText]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
