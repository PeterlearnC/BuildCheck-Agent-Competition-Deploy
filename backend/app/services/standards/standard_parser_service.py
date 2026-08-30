"""Orchestrate deterministic standards metadata, chapter, and article parsing."""

from datetime import datetime, timezone

from app.schemas.standards import (
    StandardDocument,
    StandardPage,
    StandardParseResult,
    StandardParseStatus,
    DocumentCapabilityStatus,
    MetadataConfidence,
    StandardIdentityStatus,
)
from app.schemas.ocr import SourceKind
from app.schemas.structural import ValidatedStructuralEvidence
from app.services.standards.article_merge_service import ArticleMergeService
from app.services.standards.document_region_service import DocumentRegionService
from app.services.standards.document_inspector_service import DocumentInspectorService
from app.services.standards.standard_chapter_service import StandardChapterService
from app.services.standards.standard_metadata_service import StandardMetadataService
from app.services.standards.standard_text_preprocessing_service import (
    StandardTextPreprocessingService,
)
from app.services.standards.stable_identity_service import (
    PARSER_SEMANTICS_VERSION,
    PARSER_VERSION,
)


class StandardParserService:
    def __init__(self) -> None:
        self.preprocessing = StandardTextPreprocessingService()
        self.metadata = StandardMetadataService()
        self.chapters = StandardChapterService()
        self.articles = ArticleMergeService()
        self.regions = DocumentRegionService()
        self.inspector = DocumentInspectorService()

    def parse(
        self,
        document: StandardDocument,
        pages: list[StandardPage],
        *,
        structural_evidence: list[ValidatedStructuralEvidence] | None = None,
    ) -> StandardParseResult:
        capability = self.inspector.inspect(pages)
        if capability.status != DocumentCapabilityStatus.TEXT_READY:
            parse_status = (
                StandardParseStatus.OCR_REQUIRED
                if capability.status == DocumentCapabilityStatus.OCR_REQUIRED
                else StandardParseStatus.PARSE_FAILED
            )
            updated = document.model_copy(
                update={
                    "parse_status": parse_status,
                    "parse_error": capability.status.value,
                    "capability_status": capability.status,
                    "capability_reason": capability.reason,
                    "parser_version": PARSER_VERSION,
                    "corpus_semantics_version": PARSER_SEMANTICS_VERSION,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            return StandardParseResult(
                document=updated, pages=pages, chapters=[], articles=[]
            )
        processed = self.preprocessing.preprocess(pages)
        metadata = self.metadata.extract(processed)
        binding = document.official_source_binding
        binding_is_valid = (
            binding is not None
            and binding.confirmed
            and binding.source_checksum.lower() == document.source_checksum.lower()
        )
        segmentation = self.regions.segment(processed)
        if structural_evidence is None:
            resolved_structure = self.chapters.resolve_pdf_evidence(
                source_checksum=document.source_checksum,
                pages=processed,
                region_by_line=segmentation.line_regions,
            )
        else:
            if any(
                item.source_checksum.lower() != document.source_checksum.lower()
                for item in structural_evidence
            ):
                raise ValueError(
                    "Structural evidence does not belong to the parsed document."
                )
            resolved_structure = list(structural_evidence)
        chapters, positions = self.chapters.detect(
            document.standard_id,
            processed,
            standard_checksum=document.source_checksum,
            structural_evidence=resolved_structure,
            region_by_line=segmentation.line_regions,
        )
        normative_pages = self._without_structural_text(processed, resolved_structure)
        articles = self.articles.merge(
            document.standard_id,
            normative_pages,
            positions,
            standard_source_checksum=document.source_checksum,
            standard_code=(
                binding.canonical_standard_code
                if binding_is_valid
                else metadata.canonical_standard_code
                or document.canonical_standard_code
                or metadata.standard_code
                or document.standard_code
            ),
            region_by_line=segmentation.line_regions,
            raw_pages=pages,
        )
        status = (
            StandardParseStatus.PARSED
            if articles
            else StandardParseStatus.PARSE_FAILED
        )
        updated = document.model_copy(
            update={
                "standard_code": (
                    binding.display_standard_code if binding_is_valid else metadata.standard_code
                ),
                "standard_name": binding.standard_name if binding_is_valid else metadata.standard_name,
                "canonical_standard_code": (
                    binding.canonical_standard_code
                    if binding_is_valid
                    else metadata.canonical_standard_code
                ),
                "standard_display_code": (
                    binding.display_standard_code
                    if binding_is_valid
                    else metadata.display_standard_code
                ),
                "identity_status": (
                    StandardIdentityStatus.CONFIRMED
                    if binding_is_valid
                    else metadata.identity_status
                ),
                "identity_confidence": (
                    MetadataConfidence.HIGH
                    if binding_is_valid
                    else metadata.identity_confidence
                ),
                "identity_reason": (
                    f"Curated official-source binding: {binding.binding_reason}"
                    if binding_is_valid
                    else metadata.identity_reason
                ),
                "edition": metadata.edition,
                "publish_date": metadata.publish_date,
                "effective_date": metadata.effective_date,
                "metadata_confidence": metadata.confidence,
                "parse_status": status,
                "parse_error": None if articles else "NO_ARTICLES_DETECTED",
                "parser_version": PARSER_VERSION,
                "corpus_semantics_version": PARSER_SEMANTICS_VERSION,
                "capability_status": capability.status,
                "capability_reason": capability.reason,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        return StandardParseResult(
            document=updated,
            pages=normative_pages,
            chapters=chapters,
            articles=articles,
        )

    @staticmethod
    def _without_structural_text(
        pages: list[StandardPage],
        evidence: list[ValidatedStructuralEvidence],
    ) -> list[StandardPage]:
        structural_indexes: dict[int, set[int]] = {}
        for item in evidence:
            for reference in item.source_references:
                index = (
                    reference.parser_line_index
                    if reference.parser_line_index is not None
                    else reference.source_order_index
                )
                structural_indexes.setdefault(reference.page_number, set()).add(index)

        result: list[StandardPage] = []
        for page in pages:
            excluded = structural_indexes.get(page.page_number, set())
            lines = page.text.splitlines()
            normative_lines = [
                "" if index in excluded else line
                for index, line in enumerate(lines)
            ]
            offsets: dict[int, tuple[int, int]] = {}
            position = 0
            for index, line in enumerate(normative_lines):
                offsets[index] = (position, position + len(line))
                position += len(line) + 1
            mappings = []
            for mapping in page.source_mappings:
                if mapping.parser_line_index in excluded:
                    continue
                start, end = offsets[mapping.parser_line_index]
                mappings.append(
                    mapping.model_copy(
                        update={"normalized_start": start, "normalized_end": end}
                    )
                )
            text = "\n".join(normative_lines)
            if page.source_kind == SourceKind.OCR_TEXT and not mappings:
                continue
            result.append(
                page.model_copy(update={"text": text, "source_mappings": mappings})
            )
        return result
