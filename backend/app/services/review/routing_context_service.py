"""Build bounded routing context from already-parsed physical-page text."""

import hashlib
import re
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.review_candidate import ReviewCandidate
from app.schemas.routing_context import (
    ROUTING_CONTEXT_BOUNDARY_VERSION,
    ROUTING_CONTEXT_IDENTITY_VERSION,
    RoutingContext,
    RoutingContextBoundaryMethod,
    RoutingContextSourceBindingMethod,
    deterministic_routing_context_id,
)
from app.services.pdf_service import PDFPageText, PDFService


MAX_ROUTING_CONTEXT_CHARS = 280

# Markers must begin a physical extracted line. Decimal controls such as 4.5m
# are excluded because a one-level dot marker requires following whitespace.
_OWNER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<prefix>\d+(?:\.\d+)+(?=[ \t])|\d+[、．]|\d+\.(?=[ \t]))[ \t]*"
)


class RoutingContextError(ValueError):
    pass


class RoutingContextDocumentBindingError(RoutingContextError):
    pass


class RoutingContextService:
    """Construct and verify non-normative context without changing D.1 identity."""

    def __init__(
        self,
        settings: Settings | None = None,
        pdf_service: PDFService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pdf_service = pdf_service or PDFService()

    def build_for_candidate(self, *, candidate: ReviewCandidate) -> RoutingContext:
        """Resolve and verify the stored source document before building context."""

        if not isinstance(candidate, ReviewCandidate):
            raise TypeError("Routing context requires a qualified ReviewCandidate.")
        pdf_path = self._require_stored_pdf(candidate.document_id)
        stored_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if stored_sha != candidate.document_sha256:
            raise RoutingContextDocumentBindingError(
                "Stored document SHA does not match candidate authority."
            )
        extracted = self.pdf_service.extract_text(pdf_path)
        span = candidate.source_spans[0]
        page = next(
            (item for item in extracted.pages if item.page_number == span.page_number),
            None,
        )
        if page is None:
            raise RoutingContextDocumentBindingError(
                "Candidate physical page does not exist in the verified stored document."
            )
        return self._build_from_verified_page(candidate=candidate, page=page)

    def _build_from_verified_page(
        self, *, candidate: ReviewCandidate, page: PDFPageText
    ) -> RoutingContext:
        """Pure constructor for a page whose stored-document binding is already proven."""

        if not isinstance(candidate, ReviewCandidate):
            raise TypeError("Routing context requires a qualified ReviewCandidate.")
        if not isinstance(page, PDFPageText):
            raise TypeError("Routing context requires parsed PDFPageText.")
        span = candidate.source_spans[0]
        if page.page_number != span.page_number:
            raise RoutingContextError("Candidate and page physical numbers do not match.")
        if span.char_end > len(page.text) or page.text[span.char_start : span.char_end] != (
            candidate.source_text
        ):
            raise RoutingContextError("Candidate source span does not match parsed page text.")
        source_hash = hashlib.sha256(candidate.source_text.encode("utf-8")).hexdigest()
        if source_hash != candidate.source_text_sha256:
            raise RoutingContextError("Candidate source SHA does not match its exact text.")

        start, end, method = self._context_span(page.text, span.char_start, span.char_end)
        context_text = page.text[start:end]
        relative_start = span.char_start - start
        relative_end = span.char_end - start
        return self._make_context(
            candidate=candidate,
            page_number=page.page_number,
            context_start=start,
            context_end=end,
            context_text=context_text,
            candidate_relative_start=relative_start,
            candidate_relative_end=relative_end,
            boundary_method=method,
            source_page_text_sha256=hashlib.sha256(
                page.text.encode("utf-8")
            ).hexdigest(),
        )

    def _require_stored_pdf(self, document_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", document_id):
            raise RoutingContextDocumentBindingError(
                "Stored plan document was not found."
            )
        path = self.settings.upload_dir / f"{document_id}.pdf"
        if not path.is_file():
            raise RoutingContextDocumentBindingError(
                "Stored plan document was not found."
            )
        return path

    def verify(self, *, candidate: ReviewCandidate, context: RoutingContext) -> RoutingContext:
        if not isinstance(candidate, ReviewCandidate) or not isinstance(
            context, RoutingContext
        ):
            raise TypeError("Context verification requires typed D.1 and P2 records.")
        span = candidate.source_spans[0]
        if (
            context.candidate_id != candidate.candidate_id
            or context.document_id != candidate.document_id
            or context.document_sha256 != candidate.document_sha256
            or context.page_number != span.page_number
            or context.candidate_source_text_sha256 != candidate.source_text_sha256
            or context.context_start + context.candidate_relative_start != span.char_start
            or context.context_start + context.candidate_relative_end != span.char_end
            or context.context_text[
                context.candidate_relative_start : context.candidate_relative_end
            ]
            != candidate.source_text
        ):
            raise RoutingContextError("RoutingContext does not match candidate authority.")
        return context

    @classmethod
    def _context_span(
        cls, page_text: str, candidate_start: int, candidate_end: int
    ) -> tuple[int, int, RoutingContextBoundaryMethod]:
        markers = list(_OWNER_MARKER.finditer(page_text))
        if any(candidate_start < marker.start() < candidate_end for marker in markers):
            return candidate_start, candidate_end, RoutingContextBoundaryMethod.EXACT_ONLY
        preceding = [marker for marker in markers if marker.start() <= candidate_start]
        if not preceding:
            return candidate_start, candidate_end, RoutingContextBoundaryMethod.EXACT_ONLY
        owner = preceding[-1]
        owner_level = cls._marker_level(owner.group("prefix"))
        end = len(page_text)
        for following in markers:
            if following.start() <= candidate_start:
                continue
            if cls._marker_level(following.group("prefix")) <= owner_level:
                end = following.start()
                break
        start = owner.start()
        while end > start and page_text[end - 1] in "\r\n":
            end -= 1
        if not (start <= candidate_start < candidate_end <= end):
            return candidate_start, candidate_end, RoutingContextBoundaryMethod.EXACT_ONLY
        if end - start > MAX_ROUTING_CONTEXT_CHARS:
            return candidate_start, candidate_end, RoutingContextBoundaryMethod.EXACT_ONLY
        return start, end, RoutingContextBoundaryMethod.NUMBERED_ITEM

    @staticmethod
    def _marker_level(prefix: str) -> int:
        normalized = prefix.rstrip("、．.")
        return normalized.count(".") + 1

    @staticmethod
    def _make_context(
        *,
        candidate: ReviewCandidate,
        page_number: int,
        context_start: int,
        context_end: int,
        context_text: str,
        candidate_relative_start: int,
        candidate_relative_end: int,
        boundary_method: RoutingContextBoundaryMethod,
        source_page_text_sha256: str,
    ) -> RoutingContext:
        context_hash = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
        context_id = deterministic_routing_context_id(
            candidate_id=candidate.candidate_id,
            document_sha256=candidate.document_sha256,
            page_number=page_number,
            context_start=context_start,
            context_end=context_end,
            context_text_sha256=context_hash,
            candidate_relative_start=candidate_relative_start,
            candidate_relative_end=candidate_relative_end,
            candidate_source_text_sha256=candidate.source_text_sha256,
            source_page_text_sha256=source_page_text_sha256,
            source_binding_method=(
                RoutingContextSourceBindingMethod.STORED_DOCUMENT_SHA_VERIFIED
            ),
            boundary_method=boundary_method,
        )
        return RoutingContext(
            context_identity_version=ROUTING_CONTEXT_IDENTITY_VERSION,
            routing_context_id=context_id,
            candidate_id=candidate.candidate_id,
            document_id=candidate.document_id,
            document_sha256=candidate.document_sha256,
            page_number=page_number,
            context_start=context_start,
            context_end=context_end,
            context_text=context_text,
            context_text_sha256=context_hash,
            candidate_relative_start=candidate_relative_start,
            candidate_relative_end=candidate_relative_end,
            candidate_source_text_sha256=candidate.source_text_sha256,
            source_page_text_sha256=source_page_text_sha256,
            source_binding_method=(
                RoutingContextSourceBindingMethod.STORED_DOCUMENT_SHA_VERIFIED
            ),
            boundary_method=boundary_method,
            boundary_version=ROUTING_CONTEXT_BOUNDARY_VERSION,
        )
