"""Lifecycle service for deterministic standard-document ingestion and parsing."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from app.schemas.standards import (
    StandardArticle,
    StandardDetailResponse,
    StandardDocument,
    StandardPage,
    StandardParseResult,
    StandardParseStatus,
)
from app.schemas.ocr import SourceKind
from app.services.pdf_service import PDFService
from app.services.standards.standard_parser_service import StandardParserService
from app.services.standards.stable_identity_service import (
    PARSER_SEMANTICS_VERSION,
    PARSER_VERSION,
)
from app.services.standards.standard_repository import StandardRepository


class StandardNotFoundError(Exception):
    pass


class StandardParseFailure(Exception):
    pass


class StandardDocumentService:
    def __init__(
        self,
        repository: StandardRepository | None = None,
        pdf_service: PDFService | None = None,
        parser: StandardParserService | None = None,
    ) -> None:
        self.repository = repository or StandardRepository()
        self.pdf_service = pdf_service or PDFService()
        self.parser = parser or StandardParserService()

    def upload(self, filename: str, content: bytes) -> StandardDocument:
        standard_id = str(uuid4())
        source_path = self.repository.save_source(standard_id, content)
        try:
            extraction = self.pdf_service.extract_text(source_path, require_text=False)
        except Exception:
            source_path.unlink(missing_ok=True)
            raise

        now = datetime.now(timezone.utc)
        document = StandardDocument(
            standard_id=standard_id,
            source_filename=self._display_filename(filename),
            source_checksum=hashlib.sha256(content).hexdigest(),
            page_count=extraction.page_count,
            parse_status=StandardParseStatus.UPLOADED,
            created_at=now,
            updated_at=now,
        )
        pages = [
            StandardPage(
                page_number=page.page_number,
                text=page.text,
                image_count=page.image_count,
                image_coverage_ratio=page.image_coverage_ratio,
            )
            for page in extraction.pages
        ]
        self.repository.save_document(document)
        self.repository.save_pages(standard_id, pages)
        return document

    def parse(self, standard_id: str, force: bool = False) -> tuple[StandardParseResult, bool]:
        document = self.require_document(standard_id)
        if document.source_kind == SourceKind.OCR_TEXT:
            raise StandardParseFailure("OCR_CONTROLLED_BOUNDARY_REQUIRED")
        if not force and self._cache_is_compatible(document):
            return StandardParseResult(
                document=document,
                pages=self.repository.load_pages(standard_id),
                chapters=self.repository.load_chapters(standard_id),
                articles=self.repository.load_articles(standard_id),
            ), True

        pages = self.repository.load_pages(standard_id)
        if not pages:
            raise StandardParseFailure("TEXT_NOT_AVAILABLE")
        result = self.parser.parse(document, pages)
        self.repository.save_document(result.document)
        self.repository.save_chapters(standard_id, result.chapters)
        self.repository.save_articles(standard_id, result.articles)
        if result.document.parse_status in {
            StandardParseStatus.PARSE_FAILED,
            StandardParseStatus.OCR_REQUIRED,
        }:
            raise StandardParseFailure(result.document.parse_error or "STANDARD_PARSE_FAILED")
        return result, False

    def _cache_is_compatible(self, document: StandardDocument) -> bool:
        return (
            document.parse_status == StandardParseStatus.PARSED
            and document.parser_version == PARSER_VERSION
            and document.corpus_semantics_version == PARSER_SEMANTICS_VERSION
            and self.repository.articles_have_fields(
                document.standard_id,
                {"article_id", "article_type", "region_type", "source_text"},
            )
        )

    def require_document(self, standard_id: str) -> StandardDocument:
        try:
            UUID(standard_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise StandardNotFoundError(standard_id) from exc
        document = self.repository.load_document(standard_id)
        if document is None:
            raise StandardNotFoundError(standard_id)
        return document

    def get_detail(self, standard_id: str) -> StandardDetailResponse:
        document = self.require_document(standard_id)
        return StandardDetailResponse(
            standard=document,
            article_count=len(self.repository.load_articles(standard_id)),
        )

    def list_details(self) -> list[StandardDetailResponse]:
        return [
            StandardDetailResponse(
                standard=document,
                article_count=len(self.repository.load_articles(document.standard_id)),
            )
            for document in self.repository.list_documents()
        ]

    def get_articles(
        self,
        standard_id: str,
        *,
        limit: int,
        offset: int,
        chapter_number: str | None = None,
    ) -> tuple[int, list[StandardArticle]]:
        self.require_document(standard_id)
        articles = self.repository.load_articles(standard_id)
        if chapter_number is not None:
            articles = [item for item in articles if item.chapter_number == chapter_number]
        return len(articles), articles[offset: offset + limit]

    @staticmethod
    def _display_filename(filename: str | None) -> str:
        cleaned = (filename or "standard.pdf").replace("\\", "/")
        return cleaned.rsplit("/", maxsplit=1)[-1] or "standard.pdf"
