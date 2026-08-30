"""PDF text extraction implemented with PyMuPDF."""

from dataclasses import dataclass
from pathlib import Path

import fitz


class PDFServiceError(Exception):
    """Base exception for failures that can be shown safely to API clients."""


class InvalidPDFError(PDFServiceError):
    """Raised when a file cannot be opened as a valid PDF."""


class PDFTextExtractionError(PDFServiceError):
    """Raised when text extraction fails after the document is opened."""


class PDFNoExtractableTextError(PDFServiceError):
    """Raised when a PDF has no meaningful extractable text layer."""


@dataclass(frozen=True)
class PDFPageText:
    page_number: int
    text: str
    image_count: int = 0
    image_coverage_ratio: float | None = None


@dataclass(frozen=True)
class PDFExtractionResult:
    page_count: int
    char_count: int
    text: str
    pages: list[PDFPageText]

    def preview(self, max_length: int) -> str:
        if max_length <= 0:
            return ""
        return self.text[:max_length]


class PDFService:
    """Open a PDF and return its complete per-page and combined text."""

    MIN_EXTRACTABLE_CHARACTERS = 10

    def extract_text(
        self, pdf_path: Path, *, require_text: bool = True
    ) -> PDFExtractionResult:
        try:
            document = fitz.open(pdf_path)
        except (fitz.FileDataError, fitz.FileNotFoundError, RuntimeError, ValueError) as exc:
            raise InvalidPDFError("PDF is damaged or cannot be opened.") from exc

        try:
            if document.needs_pass:
                raise InvalidPDFError(
                    "PDF is password-protected and cannot be opened without a password."
                )

            pages: list[PDFPageText] = []
            try:
                for index, page in enumerate(document):
                    images = page.get_images(full=True)
                    pages.append(
                        PDFPageText(
                            page_number=index + 1,
                            text=page.get_text("text"),
                            image_count=len(images),
                            image_coverage_ratio=self._image_coverage(page, images),
                        )
                    )
            except Exception as exc:
                raise PDFTextExtractionError("PDF text extraction failed.") from exc

            full_text = "\n".join(page.text for page in pages)
            extractable_text = "".join(full_text.split())
            if require_text and len(extractable_text) < self.MIN_EXTRACTABLE_CHARACTERS:
                raise PDFNoExtractableTextError(
                    "PDF contains little or no extractable text. "
                    "OCR support is not enabled in V0.1."
                )

            return PDFExtractionResult(
                page_count=document.page_count,
                char_count=len(full_text),
                text=full_text,
                pages=pages,
            )
        finally:
            document.close()

    @staticmethod
    def _image_coverage(page, images: list[tuple]) -> float:
        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        covered_area = 0.0
        seen_xrefs: set[int] = set()
        for image in images:
            xref = int(image[0])
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                rects = page.get_image_rects(xref)
            except (RuntimeError, ValueError):
                continue
            covered_area += sum(max(float(rect.width * rect.height), 0.0) for rect in rects)
        return min(covered_area / page_area, 1.0)
