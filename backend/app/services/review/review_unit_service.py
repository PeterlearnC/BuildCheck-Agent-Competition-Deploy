"""Create source-grounded review units from stored plan PDFs."""

import hashlib
import json
import re
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.review_unit import ReviewUnit
from app.schemas.standards_scope import StandardScope
from app.services.pdf_service import PDFService


REVIEW_UNIT_ID_VERSION = "v0.1-c.1-f1-review-unit"


class ReviewUnitError(ValueError):
    pass


class ReviewDocumentNotFoundError(ReviewUnitError):
    pass


class ReviewPageNotFoundError(ReviewUnitError):
    pass


class ReviewSourceTextNotFoundError(ReviewUnitError):
    pass


class ReviewSourceTextAmbiguousError(ReviewUnitError):
    pass


class ReviewSourceSpanError(ReviewUnitError):
    pass


class ReviewUnitService:
    def __init__(
        self,
        settings: Settings | None = None,
        pdf_service: PDFService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pdf_service = pdf_service or PDFService()

    def create(
        self,
        *,
        document_id: str,
        page_number: int,
        source_text: str,
        retrieval_query: str,
        standard_scope: StandardScope,
        char_start: int | None = None,
    ) -> ReviewUnit:
        pdf_path = self._require_pdf(document_id)
        extracted = self.pdf_service.extract_text(pdf_path)
        page = next(
            (item for item in extracted.pages if item.page_number == page_number),
            None,
        )
        if page is None:
            raise ReviewPageNotFoundError(
                f"Physical page {page_number} does not exist in the stored plan."
            )

        start = self._resolve_span(page.text, source_text, char_start)
        end = start + len(source_text)
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        review_unit_id = self._review_unit_id(
            document_id=document_id,
            page_number=page_number,
            char_start=start,
            char_end=end,
            source_text_sha256=source_hash,
        )
        return ReviewUnit(
            review_unit_id=review_unit_id,
            document_id=document_id,
            page_number=page_number,
            source_text=source_text,
            source_text_sha256=source_hash,
            char_start=start,
            char_end=end,
            review_text=source_text,
            retrieval_query=retrieval_query,
            standard_scope=standard_scope,
        )

    def verify(self, review_unit: ReviewUnit) -> ReviewUnit:
        """Reconfirm that a typed unit still matches its stored plan source."""
        pdf_path = self._require_pdf(review_unit.document_id)
        extracted = self.pdf_service.extract_text(pdf_path)
        page = next(
            (
                item
                for item in extracted.pages
                if item.page_number == review_unit.page_number
            ),
            None,
        )
        if page is None:
            raise ReviewPageNotFoundError(
                f"Physical page {review_unit.page_number} does not exist in the stored plan."
            )
        if not page.text.startswith(review_unit.source_text, review_unit.char_start):
            raise ReviewSourceSpanError(
                "ReviewUnit no longer matches the stored plan source span."
            )
        expected_id = self._review_unit_id(
            document_id=review_unit.document_id,
            page_number=review_unit.page_number,
            char_start=review_unit.char_start,
            char_end=review_unit.char_end,
            source_text_sha256=review_unit.source_text_sha256,
        )
        if review_unit.review_unit_id != expected_id:
            raise ReviewSourceSpanError(
                "ReviewUnit identity does not match the stored plan source span."
            )
        return review_unit

    def _require_pdf(self, document_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", document_id):
            raise ReviewDocumentNotFoundError("Stored plan document was not found.")
        path = self.settings.upload_dir / f"{document_id}.pdf"
        if not path.is_file():
            raise ReviewDocumentNotFoundError("Stored plan document was not found.")
        return path

    @staticmethod
    def _resolve_span(
        page_text: str, source_text: str, requested_start: int | None
    ) -> int:
        if not source_text:
            raise ReviewSourceTextNotFoundError("Selected source text is empty.")
        if requested_start is not None:
            if not page_text.startswith(source_text, requested_start):
                raise ReviewSourceSpanError(
                    "Selected source text does not match the requested page span."
                )
            return requested_start

        occurrences: list[int] = []
        position = page_text.find(source_text)
        while position >= 0:
            occurrences.append(position)
            position = page_text.find(source_text, position + 1)
        if not occurrences:
            raise ReviewSourceTextNotFoundError(
                "Selected source text does not exist on the requested physical page."
            )
        if len(occurrences) > 1:
            raise ReviewSourceTextAmbiguousError(
                "Selected source text occurs more than once; char_start is required."
            )
        return occurrences[0]

    @staticmethod
    def _review_unit_id(
        *,
        document_id: str,
        page_number: int,
        char_start: int,
        char_end: int,
        source_text_sha256: str,
    ) -> str:
        payload = {
            "identity_version": REVIEW_UNIT_ID_VERSION,
            "document_id": document_id,
            "page_number": page_number,
            "char_start": char_start,
            "char_end": char_end,
            "source_text_sha256": source_text_sha256,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "reviewunit_" + hashlib.sha256(serialized).hexdigest()
