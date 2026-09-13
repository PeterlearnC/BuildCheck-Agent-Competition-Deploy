"""Inspector-derived authorization for controlled OCR execution."""

from pathlib import Path

from app.schemas.ocr import OCRCapabilityDecision, OCRCapabilityState
from app.schemas.standards import DocumentCapabilityStatus, StandardPage
from app.services.ocr.provider import OCRProvider
from app.services.pdf_service import PDFService, PDFServiceError
from app.services.standards.document_inspector_service import DocumentInspectorService


class OCRCapabilityError(RuntimeError):
    """Raised when OCR execution is not authorized by source facts."""


class OCRCapabilityGate:
    def __init__(
        self,
        *,
        pdf_service: PDFService | None = None,
        inspector: DocumentInspectorService | None = None,
    ) -> None:
        self.pdf_service = pdf_service or PDFService()
        self.inspector = inspector or DocumentInspectorService()

    def decide(
        self,
        *,
        pdf_path: Path,
        requested_page_numbers: list[int],
        provider: OCRProvider,
    ) -> OCRCapabilityDecision:
        if not requested_page_numbers or any(number < 1 for number in requested_page_numbers):
            return self._blocked("Controlled OCR requires explicit one-based pages.")
        if len(set(requested_page_numbers)) != len(requested_page_numbers):
            return self._blocked("Controlled OCR page requests must be unique.")

        try:
            extraction = self.pdf_service.extract_text(pdf_path, require_text=False)
        except PDFServiceError:
            return self._blocked("PDF inspection failed or the input is unsupported.")

        pages = [
            StandardPage(
                page_number=page.page_number,
                text=page.text,
                image_count=page.image_count,
                image_coverage_ratio=page.image_coverage_ratio,
            )
            for page in extraction.pages
        ]
        document = self.inspector.inspect(pages)
        if document.status == DocumentCapabilityStatus.TEXT_READY:
            return OCRCapabilityDecision(
                state=OCRCapabilityState.NOT_REQUIRED,
                document_status=document.status.value,
                native_text_page_numbers=tuple(page.page_number for page in pages),
                reason="DocumentInspectorService found a usable native text layer.",
            )
        if document.status == DocumentCapabilityStatus.EMPTY_DOCUMENT:
            return self._blocked(
                "DocumentInspectorService found no OCR-authorizable document content.",
                document_status=document.status.value,
            )

        availability = provider.availability()
        if not availability.available:
            return OCRCapabilityDecision(
                state=OCRCapabilityState.UNAVAILABLE,
                document_status=document.status.value,
                reason=availability.reason,
            )

        page_statuses = {
            page.page_number: self.inspector.inspect([page]).status for page in pages
        }
        authorized = tuple(
            number
            for number, status in page_statuses.items()
            if status == DocumentCapabilityStatus.OCR_REQUIRED
        )
        native = tuple(
            number
            for number, status in page_statuses.items()
            if status == DocumentCapabilityStatus.TEXT_READY
        )
        blocked = tuple(
            number
            for number, status in page_statuses.items()
            if status not in {
                DocumentCapabilityStatus.OCR_REQUIRED,
                DocumentCapabilityStatus.TEXT_READY,
            }
        )
        requested = set(requested_page_numbers)
        if not authorized:
            return self._blocked(
                "Inspector facts did not identify an OCR-required page.",
                document_status=document.status.value,
                native=native,
                blocked=blocked,
            )
        if requested - set(authorized):
            return self._blocked(
                "OCR request includes a page not classified as OCR_REQUIRED.",
                document_status=document.status.value,
                native=native,
                blocked=blocked,
            )
        state = (
            OCRCapabilityState.MIXED
            if native or blocked
            else OCRCapabilityState.REQUIRED
        )
        return OCRCapabilityDecision(
            state=state,
            document_status=document.status.value,
            authorized_page_numbers=authorized,
            native_text_page_numbers=native,
            blocked_page_numbers=blocked,
            reason=(
                "Only inspector-classified OCR_REQUIRED pages are authorized."
                if state == OCRCapabilityState.MIXED
                else "DocumentInspectorService requires OCR for the selected pages."
            ),
        )

    @staticmethod
    def _blocked(
        reason: str,
        *,
        document_status: str = "BLOCKED",
        native: tuple[int, ...] = (),
        blocked: tuple[int, ...] = (),
    ) -> OCRCapabilityDecision:
        return OCRCapabilityDecision(
            state=OCRCapabilityState.BLOCKED,
            document_status=document_status,
            native_text_page_numbers=native,
            blocked_page_numbers=blocked,
            reason=reason,
        )
