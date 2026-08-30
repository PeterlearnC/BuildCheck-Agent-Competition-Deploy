"""Controlled OCR orchestration; no public API and no default corpus insertion."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from app.schemas.ocr import (
    OCRAcceptedPage,
    OCRPageResult,
    OCRQualityState,
    OCRRun,
    OCRRunQualityAssessment,
    OfficialSourceBinding,
    SourceKind,
)
from app.schemas.standards import (
    MetadataConfidence,
    StandardDocument,
    StandardIdentityStatus,
    StandardParseResult,
)
from app.services.ocr.accepted_boundary import OCRAcceptedBoundaryService
from app.services.ocr.ocr_repository import OCRRepository
from app.services.ocr.page_renderer import OCRPageRenderer
from app.services.ocr.provider import OCRProvider
from app.services.ocr.provider import OCRProviderError
from app.services.ocr.quality_gate import OCRQualityGate
from app.services.ocr.run_identity import (
    OCR_ADAPTER_VERSION,
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
    execution_id,
    semantic_ocr_run_id,
)
from app.services.standards.standard_parser_service import StandardParserService


@dataclass(frozen=True)
class ControlledOCRResult:
    run: OCRRun
    assessment: OCRRunQualityAssessment
    accepted_pages: list[OCRAcceptedPage]
    parse_result: StandardParseResult | None


class ControlledOCRPilotService:
    def __init__(
        self,
        *,
        renderer: OCRPageRenderer,
        provider: OCRProvider,
        repository: OCRRepository,
        quality_gate: OCRQualityGate | None = None,
        boundary: OCRAcceptedBoundaryService | None = None,
        parser: StandardParserService | None = None,
    ) -> None:
        self.renderer = renderer
        self.provider = provider
        self.repository = repository
        self.quality_gate = quality_gate or OCRQualityGate()
        self.boundary = boundary or OCRAcceptedBoundaryService()
        self.parser = parser or StandardParserService()

    def run(
        self,
        *,
        document: StandardDocument,
        pdf_path: Path,
        page_numbers: list[int],
        binding: OfficialSourceBinding,
    ) -> ControlledOCRResult:
        checksum = self._checksum(pdf_path)
        if checksum.lower() != document.source_checksum.lower():
            raise ValueError("Official PDF checksum does not match StandardDocument.")
        if binding.source_checksum.lower() != checksum.lower() or not binding.confirmed:
            raise ValueError("A confirmed curated binding for this exact source is required.")
        if not page_numbers:
            raise ValueError("Controlled OCR requires at least one explicit page.")

        raw_pages = []
        for page_number in page_numbers:
            rendered = self.renderer.render(pdf_path, page_number)
            try:
                raw_pages.append(self.provider.recognize_page(rendered))
            except OCRProviderError as exc:
                raw_pages.append(
                    OCRPageResult(
                        page_number=page_number,
                        lines=[],
                        render=rendered.metadata,
                        provider=self.provider.identity,
                        error=str(exc),
                    )
                )
        run_id = semantic_ocr_run_id(
            official_source_checksum=checksum,
            provider=self.provider.identity,
            render_config=self.renderer.config,
            render_metadata=[page.render for page in raw_pages],
        )
        run = OCRRun(
            ocr_run_id=run_id,
            execution_id=execution_id(raw_pages),
            official_source_checksum=checksum,
            provider=self.provider.identity,
            render_config=self.renderer.config,
            adapter_version=OCR_ADAPTER_VERSION,
            quality_gate_version=OCR_QUALITY_GATE_VERSION,
            corpus_semantics_version=OCR_CORPUS_SEMANTICS_VERSION,
            pages=raw_pages,
        )
        self.repository.save_raw_run(document.standard_id, run)
        assessment = self.quality_gate.assess(run, binding)
        self.repository.save_quality(document.standard_id, run, assessment)
        parser_boundary = self.boundary.parser_boundary(run, assessment)
        accepted = parser_boundary.accepted_pages
        self.repository.save_accepted_pages(document.standard_id, run, accepted)

        parse_result = None
        standard_pages = parser_boundary.standard_pages
        if standard_pages:
            controlled_document = document.model_copy(
                update={
                    "source_kind": SourceKind.OCR_TEXT,
                    "ocr_run_id": run.ocr_run_id,
                    "ocr_execution_id": run.execution_id,
                    "ocr_quality_state": assessment.state,
                    "ocr_provider": run.provider,
                    "official_source_binding": binding,
                    "ocr_corpus_semantics_version": OCR_CORPUS_SEMANTICS_VERSION,
                }
            )
            parse_result = self.parser.parse(
                controlled_document,
                standard_pages,
                structural_evidence=parser_boundary.structural_evidence,
            )
        return ControlledOCRResult(run, assessment, accepted, parse_result)

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
