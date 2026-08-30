"""The only boundary that converts assessed OCR into parser input pages."""

from dataclasses import dataclass
import re
import unicodedata

from app.schemas.ocr import (
    OCRAcceptedPage,
    OCRAcceptedSpan,
    OCRIssueCode,
    OCRQualityState,
    OCRRun,
    OCRRunQualityAssessment,
    OCRSourceMapping,
    SourceKind,
)
from app.schemas.standards import StandardPage
from app.schemas.structural import ValidatedStructuralEvidence
from app.services.ocr.article_structure import OCRArticleStructureService
from app.services.ocr.run_identity import validate_ocr_run_identity
from app.services.standards.document_region_service import (
    DocumentRegionService,
    RegionSourceLine,
)
from app.services.standards.structural_evidence_resolver import (
    StructuralEvidenceResolverService,
)


class OCRBoundaryError(ValueError):
    pass


@dataclass(frozen=True)
class OCRParserBoundaryResult:
    accepted_pages: list[OCRAcceptedPage]
    standard_pages: list[StandardPage]
    structural_evidence: list[ValidatedStructuralEvidence]


class OCRAcceptedBoundaryService:
    def __init__(
        self,
        article_structure: OCRArticleStructureService | None = None,
        regions: DocumentRegionService | None = None,
        resolver: StructuralEvidenceResolverService | None = None,
    ) -> None:
        self.article_structure = article_structure or OCRArticleStructureService()
        self.regions = regions or DocumentRegionService()
        self.resolver = resolver or StructuralEvidenceResolverService()

    def accepted_pages(
        self, run: OCRRun, assessment: OCRRunQualityAssessment
    ) -> list[OCRAcceptedPage]:
        return self._accepted_pages(
            run, assessment, allow_structural_transition=False
        )

    def _accepted_pages(
        self,
        run: OCRRun,
        assessment: OCRRunQualityAssessment,
        *,
        allow_structural_transition: bool,
    ) -> list[OCRAcceptedPage]:
        validate_ocr_run_identity(run)
        if (run.ocr_run_id, run.execution_id) != (
            assessment.ocr_run_id,
            assessment.execution_id,
        ):
            raise OCRBoundaryError("OCR quality assessment does not belong to the raw run.")
        if assessment.quality_gate_version != run.quality_gate_version:
            raise OCRBoundaryError("OCR quality assessment version does not match the raw run.")
        assessment_by_page = {page.page_number: page for page in assessment.pages}
        if len(assessment_by_page) != len(assessment.pages):
            raise OCRBoundaryError("OCR quality assessment contains duplicate pages.")
        if set(assessment_by_page) != {page.page_number for page in run.pages}:
            raise OCRBoundaryError("OCR quality assessment does not cover the exact raw page set.")
        all_raw_ids = {
            line.line_id for page in run.pages for line in page.lines
        }
        all_issues = [
            issue for page in assessment.pages for issue in page.issues
        ]
        affected_ids = {
            line_id for issue in all_issues for line_id in issue.affected_line_ids
        }
        triggering_ids = {
            line_id for issue in all_issues for line_id in issue.triggering_line_ids
        }
        if not affected_ids.issubset(all_raw_ids):
            raise OCRBoundaryError("OCR assessment affects an unknown raw line.")
        if not triggering_ids.issubset(all_raw_ids):
            raise OCRBoundaryError("OCR assessment references an unknown trigger line.")

        intervals = self.article_structure.intervals(run)
        transitions = self.article_structure.transitions(run)
        interval_by_heading = {
            interval.heading_line_id: interval for interval in intervals
        }
        article_structural_codes = {
            OCRIssueCode.ARTICLE_NUMBER_AMBIGUITY,
            OCRIssueCode.ARTICLE_SEQUENCE_GAP,
            OCRIssueCode.STRUCTURAL_SEQUENCE_GAP,
            OCRIssueCode.READING_ORDER_ANOMALY,
        }
        for issue in all_issues:
            interval = (
                interval_by_heading.get(issue.affected_article_heading_id)
                if issue.affected_article_heading_id
                else next(
                    (
                        candidate
                        for candidate in intervals
                        if any(
                            line_id in candidate.line_ids
                            for line_id in issue.affected_line_ids
                        )
                    ),
                    None,
                )
            )
            if interval is not None and issue.code in article_structural_codes:
                if not set(interval.line_ids).issubset(affected_ids):
                    raise OCRBoundaryError(
                        "Structurally uncertain OCR article interval is only partially excluded."
                    )

        all_accepted_ids = {
            line_id
            for page in assessment.pages
            for line_id in page.accepted_line_ids
        }
        transition_ids = {
            line_id for transition in transitions for line_id in transition.line_ids
        }
        if transition_ids & all_accepted_ids and not allow_structural_transition:
            raise OCRBoundaryError(
                "Structural transition text cannot cross the OCR parser boundary."
            )
        if any(set(interval.line_ids) & transition_ids for interval in intervals):
            raise OCRBoundaryError(
                "OCR article ownership crosses a structural terminator."
            )
        for interval in intervals:
            accepted_interval_ids = set(interval.line_ids) & all_accepted_ids
            if accepted_interval_ids and interval.heading_line_id not in all_accepted_ids:
                raise OCRBoundaryError(
                    "OCR body line cannot cross the parser boundary without its structural heading."
                )
        result = []
        for page in run.pages:
            page_assessment = assessment_by_page.get(page.page_number)
            if page_assessment is None:
                raise OCRBoundaryError("OCR page has no quality assessment.")
            accepted_ids = set(page_assessment.accepted_line_ids)
            raw_ids = {line.line_id for line in page.lines}
            if not accepted_ids.issubset(raw_ids):
                raise OCRBoundaryError("OCR assessment accepts an unknown raw line.")
            if accepted_ids & affected_ids:
                raise OCRBoundaryError("OCR assessment accepts a line affected by a quality issue.")
            if (
                page_assessment.state == OCRQualityState.OCR_REJECTED
                and accepted_ids
            ):
                raise OCRBoundaryError("Rejected OCR pages cannot contain accepted lines.")
            spans = []
            ordered_lines = sorted(
                (line for line in page.lines if line.line_id in accepted_ids),
                key=lambda line: (min(point[1] for point in line.polygon), min(point[0] for point in line.polygon)),
            )
            for parser_line_index, line in enumerate(ordered_lines):
                normalized = self._normalize(line.raw_text)
                if not normalized:
                    continue
                spans.append(
                    OCRAcceptedSpan(
                        page_number=page.page_number,
                        raw_text=line.raw_text,
                        normalized_text=normalized,
                        source_mappings=[
                            OCRSourceMapping(
                                page_number=page.page_number,
                                parser_line_index=parser_line_index,
                                normalized_start=0,
                                normalized_end=len(normalized),
                                raw_line_id=line.line_id,
                                raw_text=line.raw_text,
                                raw_character_start=0,
                                raw_character_end=len(line.raw_text),
                                raw_polygon=line.polygon,
                            )
                        ],
                    )
                )
            if spans:
                result.append(
                    OCRAcceptedPage(
                        page_number=page.page_number,
                        ocr_run_id=run.ocr_run_id,
                        execution_id=run.execution_id,
                        page_quality_state=page_assessment.state,
                        spans=spans,
                        provider=page.provider,
                        render=page.render,
                    )
                )
        return result

    def parser_boundary(
        self, run: OCRRun, assessment: OCRRunQualityAssessment
    ) -> OCRParserBoundaryResult:
        """Split quality-accepted normative text from structural-only evidence."""

        accepted_pages = self._accepted_pages(
            run, assessment, allow_structural_transition=True
        )
        region_lines: list[RegionSourceLine] = []
        line_id_by_position: dict[tuple[int, int], str] = {}
        ordered_by_page = {}
        for page in run.pages:
            ordered = sorted(
                page.lines,
                key=lambda line: (
                    min(point[1] for point in line.polygon),
                    min(point[0] for point in line.polygon),
                ),
            )
            ordered_by_page[page.page_number] = ordered
            for index, line in enumerate(ordered):
                region_lines.append(
                    RegionSourceLine(
                        page_number=page.page_number,
                        line_index=index,
                        text=self._normalize(line.raw_text),
                    )
                )
                line_id_by_position[(page.page_number, index)] = line.line_id
        segmentation = self.regions.segment_source_lines(region_lines)
        region_by_line_id = {
            line_id_by_position[position]: region
            for position, region in segmentation.line_regions.items()
        }
        candidates = self.article_structure.structural_candidates(
            run,
            assessment,
            region_by_line_id=region_by_line_id,
        )
        evidence = self.resolver.resolve_many(candidates)
        structural_line_ids = {
            reference.raw_line_id
            for item in evidence
            for reference in item.source_references
            if reference.raw_line_id is not None
        }
        affected_line_ids = {
            line_id
            for page in assessment.pages
            for issue in page.issues
            for line_id in issue.affected_line_ids
        }
        accepted_line_ids = {
            line_id
            for page in assessment.pages
            for line_id in page.accepted_line_ids
        }
        normative_line_ids = accepted_line_ids - structural_line_ids
        if normative_line_ids & affected_line_ids:
            raise OCRBoundaryError(
                "Quality-affected OCR cannot enter the normative parser channel."
            )
        if normative_line_ids & structural_line_ids:
            raise OCRBoundaryError(
                "Structural-only OCR cannot enter the normative parser channel."
            )
        standard_pages = self._slot_standard_pages(
            run,
            assessment,
            ordered_by_page=ordered_by_page,
            normative_line_ids=normative_line_ids,
        )
        return OCRParserBoundaryResult(
            accepted_pages=accepted_pages,
            standard_pages=standard_pages,
            structural_evidence=evidence,
        )

    def to_standard_pages(
        self, run: OCRRun, assessment: OCRRunQualityAssessment
    ) -> list[StandardPage]:
        accepted_pages = self.accepted_pages(run, assessment)
        result = []
        for page in accepted_pages:
            lines = [span.normalized_text for span in page.spans]
            mappings = []
            position = 0
            for index, span in enumerate(page.spans):
                start = position
                end = start + len(span.normalized_text)
                for mapping in span.source_mappings:
                    mappings.append(
                        mapping.model_copy(
                            update={
                                "parser_line_index": index,
                                "normalized_start": start,
                                "normalized_end": end,
                            }
                        )
                    )
                position = end + 1
            result.append(
                StandardPage(
                    page_number=page.page_number,
                    text="\n".join(lines),
                    source_kind=SourceKind.OCR_TEXT,
                    ocr_run_id=page.ocr_run_id,
                    ocr_execution_id=page.execution_id,
                    ocr_quality_state=OCRQualityState.OCR_ACCEPTED,
                    ocr_provider=page.provider,
                    ocr_render=page.render,
                    source_mappings=mappings,
                )
            )
        return result

    def _slot_standard_pages(
        self,
        run: OCRRun,
        assessment: OCRRunQualityAssessment,
        *,
        ordered_by_page,
        normative_line_ids: set[str],
    ) -> list[StandardPage]:
        assessment_by_page = {page.page_number: page for page in assessment.pages}
        result = []
        for page in run.pages:
            ordered = ordered_by_page[page.page_number]
            lines = [
                self._normalize(line.raw_text)
                if line.line_id in normative_line_ids
                else ""
                for line in ordered
            ]
            mappings = []
            position = 0
            for index, (line, normalized) in enumerate(zip(ordered, lines)):
                start = position
                end = start + len(normalized)
                if normalized:
                    mappings.append(
                        OCRSourceMapping(
                            page_number=page.page_number,
                            parser_line_index=index,
                            normalized_start=start,
                            normalized_end=end,
                            raw_line_id=line.line_id,
                            raw_text=line.raw_text,
                            raw_character_start=0,
                            raw_character_end=len(line.raw_text),
                            raw_polygon=line.polygon,
                        )
                    )
                position = end + 1
            if not mappings:
                continue
            result.append(
                StandardPage(
                    page_number=page.page_number,
                    text="\n".join(lines),
                    source_kind=SourceKind.OCR_TEXT,
                    ocr_run_id=run.ocr_run_id,
                    ocr_execution_id=run.execution_id,
                    ocr_quality_state=OCRQualityState.OCR_ACCEPTED,
                    ocr_provider=page.provider,
                    ocr_render=page.render,
                    source_mappings=mappings,
                )
            )
        return result

    @staticmethod
    def _normalize(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value or "")
        return re.sub(r"[ \t]+", " ", normalized).strip()
