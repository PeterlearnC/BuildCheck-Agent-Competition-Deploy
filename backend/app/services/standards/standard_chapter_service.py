"""Materialize chapters only from validated structural evidence."""

from dataclasses import dataclass
import re

from app.schemas.ocr import SourceKind
from app.schemas.standards import DocumentRegionType, StandardChapter, StandardPage
from app.schemas.structural import (
    StructuralEvidenceKind,
    StructuralEvidenceProducer,
    StructuralSourceReference,
    StructuralTransitionProvenance,
    StructuralValidationBasis,
    ValidatedStructuralEvidence,
)
from app.services.standards.article_detection_service import ArticleDetectionService
from app.services.standards.document_region_service import DocumentRegionService
from app.services.standards.stable_identity_service import StableIdentityService
from app.services.standards.structural_evidence_resolver import (
    StructuralEvidenceCandidate,
    StructuralEvidenceResolverService,
)


@dataclass(frozen=True)
class ChapterPosition:
    page_number: int
    line_index: int
    chapter: StandardChapter


@dataclass(frozen=True)
class _ParserLine:
    page_number: int
    line_index: int
    text: str


class StandardChapterService:
    """Keep heading syntax detection separate from structural authority."""

    CHAPTER_CN = re.compile(r"^第\s*([一二三四五六七八九十百\d]+)\s*章\s+(.{1,60})$")
    APPENDIX = re.compile(r"^附录\s*([A-Z])(?:\s+(.{1,60}))?$", re.IGNORECASE)
    APPENDIX_SECTION = re.compile(r"^([A-Z]\.\d+)\s+(.{1,60})$", re.IGNORECASE)
    LEVEL_TWO = re.compile(r"^(\d+\.\d+)\s+(.{1,60})$")
    LEVEL_ONE = re.compile(r"^(\d+)\s+(.{1,60})$")
    SENTENCE_PREDICATE = re.compile(
        r"(?:应当|必须|不得|严禁|禁止|应予|应按|应符合|应设置|应进行|应采用|"
        r"应准确|应牢固|宜采用|可采用|shall\b|must\b|should\b)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        resolver: StructuralEvidenceResolverService | None = None,
        detector: ArticleDetectionService | None = None,
        regions: DocumentRegionService | None = None,
    ) -> None:
        self.resolver = resolver or StructuralEvidenceResolverService()
        self.detector = detector or ArticleDetectionService()
        self.regions = regions or DocumentRegionService()

    def detect(
        self,
        standard_id: str,
        pages: list[StandardPage],
        *,
        standard_checksum: str | None = None,
        structural_evidence: list[ValidatedStructuralEvidence] | None = None,
        region_by_line: dict[tuple[int, int], DocumentRegionType] | None = None,
    ) -> tuple[list[StandardChapter], list[ChapterPosition]]:
        """Compatibility entry point backed exclusively by validated evidence."""

        evidence = structural_evidence
        if evidence is None:
            if standard_checksum is None:
                evidence = []
            else:
                evidence = self.resolve_pdf_evidence(
                    source_checksum=standard_checksum,
                    pages=pages,
                    region_by_line=region_by_line,
                )
        return self.materialize(standard_id, pages, evidence)

    def resolve_pdf_evidence(
        self,
        *,
        source_checksum: str,
        pages: list[StandardPage],
        region_by_line: dict[tuple[int, int], DocumentRegionType] | None = None,
    ) -> list[ValidatedStructuralEvidence]:
        """Adapt PDF-text headings into the shared fail-closed resolver."""

        if region_by_line is None:
            region_by_line = self.regions.segment(pages).line_regions
        lines = self._flatten(pages)
        article_positions = [
            (index, start.article_number)
            for index, item in enumerate(lines)
            if (start := self.detector.detect_start(item.text)) is not None
        ]
        candidates: list[StructuralEvidenceCandidate] = []
        for position, item in enumerate(lines):
            parsed = self.parse_heading(item.text)
            if parsed is None:
                continue
            number, title, level = parsed
            previous = next(
                (number for index, number in reversed(article_positions) if index < position),
                None,
            )
            following = next(
                (number for index, number in article_positions if index > position),
                None,
            )
            kind = self.evidence_kind(number, level)
            candidates.append(
                StructuralEvidenceCandidate(
                    source_checksum=source_checksum,
                    source_kind=SourceKind.PDF_TEXT,
                    producer=StructuralEvidenceProducer.PDF_TEXT,
                    structure_kind=kind,
                    structure_number=number,
                    structure_title=title,
                    region_role=region_by_line.get(
                        (item.page_number, item.line_index),
                        DocumentRegionType.OTHER,
                    ),
                    source_fragments=(item.text,),
                    source_references=(
                        StructuralSourceReference(
                            page_number=item.page_number,
                            source_order_index=item.line_index,
                            parser_line_index=item.line_index,
                        ),
                    ),
                    previous_article_number=previous,
                    next_article_number=following,
                    inside_article_interval=(
                        previous is not None
                        and self.resolver.is_compatible(number, previous)
                    ),
                    transition=StructuralTransitionProvenance(
                        previous_article_number=previous,
                        next_article_number=following,
                    ),
                    validation_basis=(
                        StructuralValidationBasis.EXPLICIT_HEADING_SYNTAX,
                    ),
                )
            )
        return self.resolver.resolve_many(candidates)

    def materialize(
        self,
        standard_id: str,
        pages: list[StandardPage],
        evidence: list[ValidatedStructuralEvidence],
    ) -> tuple[list[StandardChapter], list[ChapterPosition]]:
        positions: list[ChapterPosition] = []
        stack: list[StandardChapter] = []
        ordered = sorted(
            evidence,
            key=lambda item: (
                item.source_references[0].page_number,
                item.source_references[0].source_order_index,
            ),
        )
        for item in ordered:
            reference = item.source_references[0]
            line_index = (
                reference.parser_line_index
                if reference.parser_line_index is not None
                else reference.source_order_index
            )
            while stack and stack[-1].level >= item.structure_level:
                stack.pop()
            parent_id = stack[-1].chapter_id if stack else None
            chapter = StandardChapter(
                chapter_id=StableIdentityService().chapter_id(
                    standard_checksum=item.source_checksum,
                    chapter_number=item.structure_number,
                    chapter_title=item.structure_title,
                ),
                standard_id=standard_id,
                chapter_number=item.structure_number,
                chapter_title=item.structure_title,
                level=item.structure_level,
                source_page_start=reference.page_number,
                source_page_end=reference.page_number,
                source_text=item.source_text,
                parent_chapter_id=parent_id,
                structural_evidence=item,
            )
            positions.append(ChapterPosition(reference.page_number, line_index, chapter))
            stack.append(chapter)

        chapters = [position.chapter for position in positions]
        last_page = max(
            [page.page_number for page in pages]
            + [position.page_number for position in positions],
            default=1,
        )
        for index, chapter in enumerate(chapters):
            next_boundary = next(
                (
                    item
                    for item in chapters[index + 1 :]
                    if item.level <= chapter.level
                ),
                None,
            )
            chapter.source_page_end = (
                next_boundary.source_page_start if next_boundary is not None else last_page
            )
        return chapters, positions

    def parse_heading(self, line: str) -> tuple[str, str, int] | None:
        """Parse candidate syntax only; this method grants no authority."""

        value = re.sub(r"\s+", " ", line.strip())
        if not value or len(value) > 80 or value[-1:] in "。；;，,":
            return None
        match = self.CHAPTER_CN.fullmatch(value)
        if match:
            return self._chapter_number(match.group(1)), match.group(2).strip(), 1
        match = self.APPENDIX.fullmatch(value)
        if match:
            return match.group(1).upper(), (match.group(2) or "附录").strip(), 1
        match = self.APPENDIX_SECTION.fullmatch(value)
        if match:
            return match.group(1).upper(), match.group(2).strip(), 2
        match = self.LEVEL_TWO.fullmatch(value)
        if match and not self.SENTENCE_PREDICATE.search(match.group(2)):
            return match.group(1), match.group(2).strip(), 2
        match = self.LEVEL_ONE.fullmatch(value)
        if match and not self.SENTENCE_PREDICATE.search(match.group(2)):
            return match.group(1), match.group(2).strip(), 1
        return None

    @staticmethod
    def evidence_kind(number: str, level: int) -> StructuralEvidenceKind:
        if number[:1].isalpha():
            return (
                StructuralEvidenceKind.APPENDIX
                if level == 1
                else StructuralEvidenceKind.APPENDIX_SECTION
            )
        return (
            StructuralEvidenceKind.CHAPTER
            if level == 1
            else StructuralEvidenceKind.SECTION
        )

    @staticmethod
    def _flatten(pages: list[StandardPage]) -> list[_ParserLine]:
        return [
            _ParserLine(page.page_number, line_index, raw_line.strip())
            for page in pages
            for line_index, raw_line in enumerate(page.text.splitlines())
        ]

    @staticmethod
    def _chapter_number(value: str) -> str:
        if value.isdigit():
            return value
        digits = {
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if value == "十":
            return "10"
        if "十" in value:
            left, _, right = value.partition("十")
            return str((digits.get(left, 1) * 10) + digits.get(right, 0))
        return str(digits.get(value, value))
