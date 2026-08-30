"""Deterministic, one-page-at-a-time PDF rendering for OCR workers."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import fitz

from app.schemas.ocr import OCRRenderConfig, OCRRenderMetadata


OCR_RENDER_CONFIG_VERSION = "v0.4-b.3b.1b-render"


@dataclass(frozen=True)
class RenderedOCRPage:
    png_bytes: bytes
    metadata: OCRRenderMetadata


class OCRPageRenderer:
    def __init__(self, config: OCRRenderConfig | None = None) -> None:
        self.config = config or OCRRenderConfig(
            renderer="PyMuPDF",
            renderer_version=fitz.VersionBind,
            dpi=300,
            colorspace="GRAY",
            alpha=False,
            config_version=OCR_RENDER_CONFIG_VERSION,
        )

    def render(self, pdf_path: Path, page_number: int) -> RenderedOCRPage:
        if page_number < 1:
            raise ValueError("OCR page numbers are 1-based.")
        with fitz.open(pdf_path) as document:
            if page_number > document.page_count:
                raise ValueError("OCR page number exceeds the PDF page count.")
            page = document.load_page(page_number - 1)
            colorspace = (
                fitz.csGRAY if self.config.colorspace.upper() == "GRAY" else fitz.csRGB
            )
            pixmap = page.get_pixmap(
                dpi=self.config.dpi,
                colorspace=colorspace,
                alpha=self.config.alpha,
            )
            png_bytes = pixmap.tobytes("png")
        metadata = OCRRenderMetadata(
            **self.config.model_dump(),
            page_number=page_number,
            pixel_width=pixmap.width,
            pixel_height=pixmap.height,
            image_sha256=hashlib.sha256(png_bytes).hexdigest(),
        )
        return RenderedOCRPage(png_bytes=png_bytes, metadata=metadata)