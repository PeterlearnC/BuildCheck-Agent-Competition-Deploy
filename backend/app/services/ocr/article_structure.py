"""Raw-OCR article and structural-transition ownership."""

from dataclasses import dataclass
import re

from app.schemas.ocr import OCRLineResult, OCRRun, OCRRunQualityAssessment, SourceKind
from app.schemas.standards import DocumentRegionType
from app.schemas.structural import (
    StructuralEvidenceProducer,
    StructuralSourceReference,
    StructuralTransitionProvenance,
    StructuralValidationBasis,
)
from app.services.standards.article_detection_service import ArticleDetectionService
from app.services.standards.standard_chapter_service import StandardChapterService
from app.services.standards.structural_evidence_resolver import (
    StructuralEvidenceCandidate,
)


@dataclass(frozen=True)
class OCRArticleInterval:
    article_number: str
    heading_line_id: str
    heading_page_number: int
    line_ids: tuple[str, ...]


@dataclass(frozen=True)
class OCRStructuralTransition:
    """Raw lines between independently owned article intervals."""

    page_number: int
    line_ids: tuple[str, ...]
    structural_heading_line_ids: tuple[str, ...]
    structural_numbers: tuple[str, ...]
    previous_article_heading_id: str | None
    next_article_heading_id: str | None
    next_article_number: str | None
    ambiguous: bool = False


class OCRArticleStructureService:
    """Derive structural ownership without repairing absent OCR headings."""

    SIMPLE_NUMBER = re.compile(r"^\d+(?:\.\d+){2,3}$")
    COMPACT_STRUCTURE = re.compile(
        r"^(?P<number>\d+(?:\.\d+)?)(?P<title>[\u3400-\u9fff][\u3400-\u9fff\s]{1,30})$"
    )

    def __init__(
        self,
        detector: ArticleDetectionService | None = None,
        chapters: StandardChapterService | None = None,
    ) -> None:
        self.detector = detector or ArticleDetectionService()
        self.chapters = chapters or StandardChapterService()

    def intervals(self, run: OCRRun) -> list[OCRArticleInterval]:
        return self._structure(run)[0]

    def transitions(self, run: OCRRun) -> list[OCRStructuralTransition]:
        return self._structure(run)[1]

    def structural_candidates(
        self,
        run: OCRRun,
        assessment: OCRRunQualityAssessment,
        *,
        region_by_line_id: dict[str, DocumentRegionType],
    ) -> list[StructuralEvidenceCandidate]:
        """Adapt raw transition headings without granting structural authority."""

        ordered = self.ordered_lines(run)
        lines_by_id = {line.line_id: line for line in ordered}
        source_order: dict[str, int] = {}
        page_offsets: dict[int, int] = {}
        for line in ordered:
            index = page_offsets.get(line.page_number, 0)
            source_order[line.line_id] = index
            page_offsets[line.page_number] = index + 1
        intervals = self.intervals(run)
        article_by_heading = {
            interval.heading_line_id: interval.article_number for interval in intervals
        }
        issues = [issue for page in assessment.pages for issue in page.issues]
        result: list[StructuralEvidenceCandidate] = []
        for transition in self.transitions(run):
            previous_number = article_by_heading.get(
                transition.previous_article_heading_id
            )
            for line_id in transition.structural_heading_line_ids:
                line = lines_by_id[line_id]
                parsed = self._parse_structural_heading(line.raw_text)
                if parsed is None:
                    continue
                number, title, level = parsed
                kind = StandardChapterService.evidence_kind(number, level)
                issue_codes = list(
                    dict.fromkeys(
                        issue.code.value
                        for issue in issues
                        if line_id in issue.affected_line_ids
                        or line_id in issue.triggering_line_ids
                    )
                )
                result.append(
                    StructuralEvidenceCandidate(
                        source_checksum=run.official_source_checksum,
                        source_kind=SourceKind.OCR_TEXT,
                        producer=StructuralEvidenceProducer.OCR_TRANSITION,
                        structure_kind=kind,
                        structure_number=number,
                        structure_title=title,
                        region_role=region_by_line_id.get(
                            line_id, DocumentRegionType.OTHER
                        ),
                        source_fragments=(line.raw_text,),
                        source_references=(
                            StructuralSourceReference(
                                page_number=line.page_number,
                                source_order_index=source_order[line_id],
                                raw_line_id=line.line_id,
                                polygon=line.polygon,
                            ),
                        ),
                        previous_article_number=previous_number,
                        next_article_number=transition.next_article_number,
                        inside_article_interval=False,
                        transition=StructuralTransitionProvenance(
                            issue_codes=issue_codes,
                            previous_article_heading_id=(
                                transition.previous_article_heading_id
                            ),
                            previous_article_number=previous_number,
                            next_article_heading_id=transition.next_article_heading_id,
                            next_article_number=transition.next_article_number,
                            ambiguous=transition.ambiguous,
                        ),
                        validation_basis=(
                            StructuralValidationBasis.OCR_TRANSITION,
                            StructuralValidationBasis.EXPLICIT_HEADING_SYNTAX,
                        ),
                    )
                )
        return result

    def _structure(
        self, run: OCRRun
    ) -> tuple[list[OCRArticleInterval], list[OCRStructuralTransition]]:
        ordered = self.ordered_lines(run)
        starts: list[tuple[int, str]] = []
        for index, line in enumerate(ordered):
            start = self.detector.detect_start(line.raw_text)
            if start is not None:
                starts.append((index, start.article_number))

        intervals: list[OCRArticleInterval] = []
        transitions: list[OCRStructuralTransition] = []
        if starts:
            first_index, first_number = starts[0]
            candidates = self._heading_candidates(
                ordered, range(0, first_index), first_number
            )
            if candidates:
                cut = self._expand_heading_band(
                    ordered, candidates[0][0], lower_bound=0
                )
                transitions.append(
                    self._transition(
                        ordered,
                        cut,
                        first_index,
                        candidates,
                        previous_heading_id=None,
                        next_heading_id=ordered[first_index].line_id,
                        next_article_number=first_number,
                    )
                )

        for position, (start_index, article_number) in enumerate(starts):
            next_start = starts[position + 1] if position + 1 < len(starts) else None
            natural_end = next_start[0] if next_start is not None else len(ordered)
            next_number = next_start[1] if next_start is not None else None
            crosses_prefix = (
                next_start is not None
                and self._prefix(article_number) != self._prefix(next_number)
            )
            if crosses_prefix:
                candidates = self._heading_candidates(
                    ordered, range(start_index + 1, natural_end), next_number
                )
            elif next_start is None:
                candidates = self._terminal_heading_candidates(
                    ordered,
                    range(start_index + 1, natural_end),
                    article_number,
                )
            else:
                candidates = []
            cut = (
                self._expand_heading_band(
                    ordered, candidates[0][0], lower_bound=start_index + 1
                )
                if candidates
                else natural_end
            )
            ambiguous = False

            if (
                not candidates
                and next_start is not None
                and crosses_prefix
            ):
                next_page = ordered[next_start[0]].page_number
                next_page_leading = [
                    index
                    for index in range(start_index + 1, natural_end)
                    if ordered[index].page_number == next_page
                ]
                if next_page_leading:
                    cut = next_page_leading[0]
                    ambiguous = True
                elif ordered[start_index].page_number == next_page:
                    cut = start_index
                    ambiguous = True

            heading = ordered[start_index]
            intervals.append(
                OCRArticleInterval(
                    article_number=article_number,
                    heading_line_id=heading.line_id,
                    heading_page_number=heading.page_number,
                    line_ids=tuple(
                        line.line_id for line in ordered[start_index:cut]
                    ),
                )
            )
            if cut < natural_end:
                transitions.append(
                    self._transition(
                        ordered,
                        cut,
                        natural_end,
                        candidates,
                        previous_heading_id=heading.line_id,
                        next_heading_id=(
                            ordered[next_start[0]].line_id
                            if next_start is not None
                            else None
                        ),
                        next_article_number=next_number,
                        ambiguous=ambiguous,
                    )
                )
        return intervals, transitions

    def sequence_gaps(
        self, intervals: list[OCRArticleInterval]
    ) -> list[tuple[OCRArticleInterval, OCRArticleInterval]]:
        gaps = []
        for current, following in zip(intervals, intervals[1:]):
            left = self._parts(current.article_number)
            right = self._parts(following.article_number)
            if (
                left is not None
                and right is not None
                and len(left) == len(right)
                and left[:-1] == right[:-1]
                and right[-1] > left[-1] + 1
            ):
                gaps.append((current, following))
        return gaps

    def section_start_gaps(
        self, transitions: list[OCRStructuralTransition]
    ) -> list[OCRStructuralTransition]:
        result = []
        for transition in transitions:
            parts = self._parts_any(transition.next_article_number)
            if parts is None or parts[-1] <= 1:
                continue
            if any(
                self._is_prefix(self._parts_any(number), parts)
                for number in transition.structural_numbers
            ):
                result.append(transition)
        return result

    def _heading_candidates(self, ordered, indices, next_article_number):
        next_parts = self._parts_any(next_article_number)
        if next_parts is None:
            return []
        result = []
        for index in indices:
            parsed = self._parse_structural_heading(ordered[index].raw_text)
            if parsed is None:
                continue
            number, _title, _level = parsed
            if self._is_prefix(self._parts_any(number), next_parts):
                result.append((index, number))
        return result

    def _terminal_heading_candidates(
        self, ordered, indices, current_article_number
    ):
        current_parts = self._parts_any(current_article_number)
        if current_parts is None:
            return []
        result = []
        for index in indices:
            parsed = self._parse_structural_heading(ordered[index].raw_text)
            if parsed is None:
                continue
            number, _title, _level = parsed
            structural_parts = self._parts_any(number)
            if self._is_structural_successor(structural_parts, current_parts):
                result.append((index, number))
        return result

    def _parse_structural_heading(self, value: str):
        parsed = self.chapters.parse_heading(value)
        if parsed is not None:
            return parsed
        compact = self.COMPACT_STRUCTURE.fullmatch(
            re.sub(r"\s+", "", value.strip())
        )
        if compact is None:
            return None
        return self.chapters.parse_heading(
            f"{compact.group('number')} {compact.group('title')}"
        )

    @classmethod
    def _expand_heading_band(cls, ordered, candidate_index, *, lower_bound):
        """Include preceding OCR fragments occupying the heading's visual band."""
        candidate = ordered[candidate_index]
        candidate_top, candidate_bottom = cls._vertical_bounds(candidate.polygon)
        cut = candidate_index
        for index in range(candidate_index - 1, lower_bound - 1, -1):
            line = ordered[index]
            if line.page_number != candidate.page_number:
                break
            top, bottom = cls._vertical_bounds(line.polygon)
            overlap = min(bottom, candidate_bottom) - max(top, candidate_top)
            if overlap <= 0:
                break
            minimum_height = min(bottom - top, candidate_bottom - candidate_top)
            if overlap < minimum_height * 0.30:
                break
            cut = index
        return cut

    @staticmethod
    def _transition(
        ordered,
        start_index,
        end_index,
        heading_candidates,
        *,
        previous_heading_id,
        next_heading_id,
        next_article_number,
        ambiguous=False,
    ) -> OCRStructuralTransition:
        lines = ordered[start_index:end_index]
        return OCRStructuralTransition(
            page_number=lines[0].page_number,
            line_ids=tuple(line.line_id for line in lines),
            structural_heading_line_ids=tuple(
                ordered[index].line_id
                for index, _number in heading_candidates
                if start_index <= index < end_index
            ),
            structural_numbers=tuple(
                number
                for index, number in heading_candidates
                if start_index <= index < end_index
            ),
            previous_article_heading_id=previous_heading_id,
            next_article_heading_id=next_heading_id,
            next_article_number=next_article_number,
            ambiguous=ambiguous,
        )

    @staticmethod
    def ordered_lines(run: OCRRun) -> list[OCRLineResult]:
        return sorted(
            (line for page in run.pages for line in page.lines),
            key=lambda line: (
                line.page_number,
                min(point[1] for point in line.polygon),
                min(point[0] for point in line.polygon),
            ),
        )

    @staticmethod
    def _vertical_bounds(polygon):
        values = [point[1] for point in polygon]
        return min(values), max(values)

    @classmethod
    def _parts(cls, value: str) -> tuple[int, ...] | None:
        if cls.SIMPLE_NUMBER.fullmatch(value) is None:
            return None
        return tuple(int(part) for part in value.split("."))

    @staticmethod
    def _parts_any(value: str | None) -> tuple[int, ...] | None:
        if not value or re.fullmatch(r"\d+(?:\.\d+){0,3}", value) is None:
            return None
        return tuple(int(part) for part in value.split("."))

    @classmethod
    def _prefix(cls, value: str | None) -> tuple[int, ...] | None:
        parts = cls._parts_any(value)
        return parts[:-1] if parts else None

    @staticmethod
    def _is_prefix(
        prefix: tuple[int, ...] | None, value: tuple[int, ...]
    ) -> bool:
        return (
            bool(prefix)
            and len(prefix) < len(value)
            and value[: len(prefix)] == prefix
        )

    @staticmethod
    def _is_structural_successor(
        structural: tuple[int, ...] | None,
        article: tuple[int, ...],
    ) -> bool:
        if not structural or len(structural) >= len(article):
            return False
        if len(structural) == 1:
            return structural[0] > article[0]
        return (
            len(structural) == 2
            and len(article) >= 3
            and structural[0] == article[0]
            and structural[1] > article[1]
        )
