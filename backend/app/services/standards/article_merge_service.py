"""Recover article bodies across lines and pages until structural boundaries."""

from dataclasses import dataclass, field
import re

from app.schemas.standards import (
    ArticleType,
    ArticleParseConfidence,
    DocumentRegionType,
    StandardArticle,
    StandardPage,
)
from app.schemas.ocr import (
    OCRProviderIdentity,
    OCRQualityState,
    OCRRenderMetadata,
    OCRSourceMapping,
    SourceKind,
)
from app.services.standards.article_detection_service import (
    ArticleDetectionService,
    ArticleStart,
)
from app.services.standards.standard_chapter_service import ChapterPosition
from app.services.standards.stable_identity_service import StableIdentityService
from app.services.standards.structural_evidence_resolver import (
    StructuralEvidenceResolverService,
)


@dataclass
class _ActiveArticle:
    start: ArticleStart
    start_page: int
    chapter_position: ChapterPosition | None
    region_type: DocumentRegionType
    detected_region_type: DocumentRegionType
    content_lines: list[tuple[int, str]] = field(default_factory=list)
    source_lines: list[str] = field(default_factory=list)
    source_mappings: list[OCRSourceMapping] = field(default_factory=list)
    source_kind: SourceKind = SourceKind.PDF_TEXT
    ocr_run_id: str | None = None
    ocr_execution_id: str | None = None
    ocr_provider: OCRProviderIdentity | None = None
    ocr_render_metadata: list[OCRRenderMetadata] = field(default_factory=list)
    source_ambiguity: str | None = None


class ArticleMergeService:
    LIST_ITEM = re.compile(r"^(?:\d+[、.)）]|[（(]\d+[)）])")
    NUMBER_ONLY = re.compile(r"^(?:\d+\.){2,3}\d+(?:-\d+)?$")
    SPLIT_DIGIT_LINE = re.compile(r"^\d\s+\S+")

    def __init__(
        self,
        detector: ArticleDetectionService | None = None,
        resolver: StructuralEvidenceResolverService | None = None,
    ) -> None:
        self.detector = detector or ArticleDetectionService()
        self.resolver = resolver or StructuralEvidenceResolverService()

    def merge(
        self,
        standard_id: str,
        pages: list[StandardPage],
        chapter_positions: list[ChapterPosition],
        *,
        standard_source_checksum: str | None = None,
        standard_code: str | None = None,
        region_by_line: dict[tuple[int, int], DocumentRegionType] | None = None,
        raw_pages: list[StandardPage] | None = None,
    ) -> list[StandardArticle]:
        chapter_at = {
            (position.page_number, position.line_index): position
            for position in chapter_positions
        }
        structure_stack: list[ChapterPosition] = []
        active: _ActiveArticle | None = None
        articles: list[StandardArticle] = []
        seen_numbers: set[str] = set()
        raw_lines_by_page = {
            page.page_number: page.text.splitlines() for page in (raw_pages or pages)
        }
        pages_by_number = {page.page_number: page for page in pages}

        def flush() -> None:
            nonlocal active
            article = self._build_article(
                standard_id,
                active,
                len(articles) + 1,
                standard_source_checksum=standard_source_checksum or standard_id,
                standard_code=standard_code,
            )
            if article is not None:
                articles.append(article)
            active = None

        positions_by_page: dict[int, list[int]] = {}
        for position in chapter_positions:
            positions_by_page.setdefault(position.page_number, []).append(
                position.line_index
            )
        page_numbers = sorted(set(pages_by_number) | set(positions_by_page))

        for page_number in page_numbers:
            page = pages_by_number.get(page_number)
            page_lines = page.text.splitlines() if page is not None else []
            maximum_index = max(
                [len(page_lines) - 1, *positions_by_page.get(page_number, [])],
                default=-1,
            )
            for line_index in range(maximum_index + 1):
                raw_line = page_lines[line_index] if line_index < len(page_lines) else ""
                line = raw_line.strip()
                chapter = chapter_at.get((page_number, line_index))
                region_type = (region_by_line or {}).get(
                    (page_number, line_index), DocumentRegionType.OTHER
                )
                if chapter is not None:
                    evidence = chapter.chapter.structural_evidence
                    if evidence is not None and active is not None and self.resolver.is_compatible(
                        evidence.structure_number, active.start.article_number
                    ):
                        continue
                    if evidence is not None and active is not None:
                        flush()
                    while evidence is not None and (
                        structure_stack
                        and structure_stack[-1].chapter.level >= chapter.chapter.level
                    ):
                        structure_stack.pop()
                    if evidence is not None:
                        structure_stack.append(chapter)
                        continue
                if not line:
                    continue
                if active is not None and active.detected_region_type != region_type:
                    flush()
                start = self.detector.detect_start(line)
                if start is not None:
                    flush()
                    source_lines = raw_lines_by_page.get(page_number, [])
                    source_line = (
                        source_lines[line_index].strip()
                        if line_index < len(source_lines)
                        else raw_line.strip()
                    )
                    page_source = pages_by_number[page_number]
                    line_mappings = [
                        mapping
                        for mapping in page_source.source_mappings
                        if mapping.parser_line_index == line_index
                    ]
                    if page_source.source_kind == SourceKind.OCR_TEXT and line_mappings:
                        source_line = "".join(mapping.raw_text for mapping in line_mappings)
                    split_number_ambiguity = (
                        self.NUMBER_ONLY.fullmatch(line) is not None
                        and start.article_number in seen_numbers
                        and line_index + 1 < len(page_lines)
                        and self.SPLIT_DIGIT_LINE.match(page_lines[line_index + 1].strip())
                        is not None
                    )
                    effective_region = (
                        DocumentRegionType.OTHER
                        if split_number_ambiguity
                        else region_type
                    )
                    active = _ActiveArticle(
                        start=start,
                        start_page=page_number,
                        chapter_position=next(
                            (
                                position
                                for position in reversed(structure_stack)
                                if position.chapter.chapter_number is not None
                                and self.resolver.is_compatible(
                                    position.chapter.chapter_number,
                                    start.article_number,
                                )
                            ),
                            None,
                        ),
                        region_type=effective_region,
                        detected_region_type=region_type,
                        source_lines=[source_line],
                        source_mappings=line_mappings,
                        source_kind=page_source.source_kind,
                        ocr_run_id=page_source.ocr_run_id,
                        ocr_execution_id=page_source.ocr_execution_id,
                        ocr_provider=page_source.ocr_provider,
                        ocr_render_metadata=(
                            [page_source.ocr_render] if page_source.ocr_render else []
                        ),
                        source_ambiguity=(
                            "AMBIGUOUS_SPLIT_ARTICLE_NUMBER"
                            if split_number_ambiguity
                            else None
                        ),
                    )
                    seen_numbers.add(start.article_number)
                    if start.inline_content:
                        active.content_lines.append((page_number, start.inline_content))
                    continue
                if active is not None:
                    active.content_lines.append((page_number, line))
                    source_lines = raw_lines_by_page.get(page_number, [])
                    page_source = pages_by_number[page_number]
                    line_mappings = [
                        mapping
                        for mapping in page_source.source_mappings
                        if mapping.parser_line_index == line_index
                    ]
                    if active.source_kind == SourceKind.OCR_TEXT and line_mappings:
                        active.source_lines.append(
                            "".join(mapping.raw_text for mapping in line_mappings)
                        )
                        active.source_mappings.extend(line_mappings)
                        if (
                            page_source.ocr_render
                            and page_source.ocr_render.page_number
                            not in {item.page_number for item in active.ocr_render_metadata}
                        ):
                            active.ocr_render_metadata.append(page_source.ocr_render)
                    else:
                        active.source_lines.append(
                            source_lines[line_index].strip()
                            if line_index < len(source_lines)
                            else raw_line.strip()
                        )
        flush()
        return articles

    def _build_article(
        self,
        standard_id: str,
        active: _ActiveArticle | None,
        sequence: int,
        *,
        standard_source_checksum: str,
        standard_code: str | None,
    ) -> StandardArticle | None:
        if active is None or not active.content_lines:
            return None
        content = self._join_content([line for _, line in active.content_lines])
        if not content:
            return None
        end_page = active.content_lines[-1][0]
        chapter = active.chapter_position.chapter if active.chapter_position else None
        parent = (
            active.start.article_number.rsplit("-", 1)[0]
            if "-" in active.start.article_number
            else None
        )
        article_id = StableIdentityService().article_id(
            standard_source_checksum=standard_source_checksum,
            standard_code=standard_code,
            chapter_number=chapter.chapter_number if chapter else None,
            article_number=active.start.article_number,
            article_text=content,
            source_page_start=active.start_page,
            source_page_end=end_page,
            region_type=active.region_type.value,
            article_type=self._article_type(active.region_type).value,
            source_kind=active.source_kind.value,
            accepted_source_line_ids=[
                mapping.raw_line_id for mapping in active.source_mappings
            ],
        )
        return StandardArticle(
            article_id=article_id,
            standard_id=standard_id,
            chapter_id=chapter.chapter_id if chapter else None,
            chapter_number=chapter.chapter_number if chapter else None,
            chapter_title=chapter.chapter_title if chapter else None,
            article_number=active.start.article_number,
            content=content,
            source_page_start=active.start_page,
            source_page_end=end_page,
            source_text="\n".join(active.source_lines),
            parent_article_number=parent,
            sequence=sequence,
            is_mandatory=None,
            parse_confidence=(
                ArticleParseConfidence.LOW
                if active.source_ambiguity
                else (
                    ArticleParseConfidence.HIGH
                    if active.start_page == end_page
                    else ArticleParseConfidence.MEDIUM
                )
            ),
            article_type=self._article_type(active.region_type),
            region_type=active.region_type,
            source_kind=active.source_kind,
            ocr_run_id=active.ocr_run_id,
            ocr_execution_id=active.ocr_execution_id,
            ocr_quality_state=(
                OCRQualityState.OCR_ACCEPTED
                if active.source_kind == SourceKind.OCR_TEXT
                else None
            ),
            ocr_provider=active.ocr_provider,
            ocr_render_metadata=active.ocr_render_metadata,
            source_mappings=active.source_mappings,
            metadata=(
                {"source_integrity_status": active.source_ambiguity}
                if active.source_ambiguity
                else None
            ),
        )

    @staticmethod
    def _article_type(region_type: DocumentRegionType) -> ArticleType:
        return {
            DocumentRegionType.NORMATIVE_BODY: ArticleType.NORMATIVE,
            DocumentRegionType.EXPLANATION: ArticleType.EXPLANATION,
            DocumentRegionType.APPENDIX: ArticleType.APPENDIX,
            DocumentRegionType.REPEALED_LIST: ArticleType.REPEALED,
        }.get(region_type, ArticleType.OTHER)

    def _join_content(self, lines: list[str]) -> str:
        result = ""
        for line in lines:
            value = re.sub(r"\s+", " ", line.strip())
            if not value:
                continue
            if not result:
                result = value
            elif self.LIST_ITEM.match(value):
                result += "\n" + value
            elif self._ascii_boundary(result[-1], value[0]):
                result += " " + value
            else:
                result += value
        return result

    @staticmethod
    def _ascii_boundary(left: str, right: str) -> bool:
        return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()
