"""Structurally validated semantic-region segmentation for standards."""

from dataclasses import dataclass
import re

from app.schemas.standards import DocumentRegion, DocumentRegionType, StandardPage


@dataclass(frozen=True)
class RegionSegmentationResult:
    regions: list[DocumentRegion]
    line_regions: dict[tuple[int, int], DocumentRegionType]


@dataclass(frozen=True)
class RegionSourceLine:
    """A source-positioned line used only for region classification."""

    page_number: int
    line_index: int
    text: str


class DocumentRegionService:
    """Use heading shape and neighbouring structure, never substring switches."""

    TOC_HEADING = re.compile(r"^(?:目\s*录|目\s*次|contents)$", re.IGNORECASE)
    TOC_ENTRY = re.compile(
        r"(?:\.{2,}|…{2,}|·{2,}|-{3,}|—{3,})\s*\d+\s*$|"
        r"\s{2,}\d+\s*$"
    )
    EXPLANATION = re.compile(
        r"^(?:[一二三四五六七八九十]+[、.．]\s*)?(?:本(?:规范|标准))?条文说明$"
    )
    OTHER_HEADING = re.compile(r"^(?:附\s*[:：]?\s*)?起草说明$")
    NORMATIVE_MARKER = re.compile(r"^(?:规范|标准)?正文(?:部分)?$")
    REPEALED_HEADING = re.compile(
        r"^(?:废止的?条文|废止条款(?:清单|目录)?|"
        r"现行工程建设标准相关强制性条文同时废止)$"
    )
    REPEALED_INTRO = re.compile(r"同时废止下列.*(?:强制性)?条文[:：]?$")
    REPEALED_ITEM = re.compile(r"^[一二三四五六七八九十]+[、.．]")
    APPENDIX = re.compile(r"^附录\s*[A-Z](?:\s+[^.·…]{1,40})?$", re.IGNORECASE)
    ROOT_HEADING = re.compile(r"^(?:第\s*[一1]\s*章(?:\s+.+)?|1\s+总\s*则)$")
    CHAPTER_HEADING = re.compile(r"^(?:第\s*[一二三四五六七八九十百\d]+\s*章|\d+\s+\S+)")
    ARTICLE = re.compile(
        r"^(?:第\s*)?(?:\d+\.){2,3}\d+(?:-\d+)?(?:\s|条|$)"
    )
    APPENDIX_ARTICLE = re.compile(r"^[A-Z]\.(?:\d+\.)+\d+(?:-\d+)?(?:\s|条|$)", re.I)

    def segment(self, pages: list[StandardPage]) -> RegionSegmentationResult:
        return self.segment_source_lines(self._flatten(pages))

    def segment_source_lines(
        self, lines: list[RegionSourceLine]
    ) -> RegionSegmentationResult:
        """Classify an original-position stream without requiring parser pages."""

        current = DocumentRegionType.FRONT_MATTER
        in_toc = False
        toc_observed = False
        normative_marker_pending = False
        line_regions: dict[tuple[int, int], DocumentRegionType] = {}
        ordered: list[tuple[int, int, DocumentRegionType]] = []

        for position, item in enumerate(lines):
            line = item.text
            next_lines = self._next_significant(lines, position, limit=3)

            if self.TOC_HEADING.fullmatch(line):
                in_toc = True
                toc_observed = True
                assigned = DocumentRegionType.FRONT_MATTER
                self._record(item, assigned, line_regions, ordered)
                continue
            if in_toc:
                if not line or self._is_toc_entry(line):
                    self._record(
                        item, DocumentRegionType.FRONT_MATTER, line_regions, ordered
                    )
                    continue
                in_toc = False
            elif self._is_toc_entry(line):
                self._record(item, current, line_regions, ordered)
                continue

            if not line:
                self._record(item, current, line_regions, ordered)
                continue

            if self.NORMATIVE_MARKER.fullmatch(line):
                normative_marker_pending = True
                current = DocumentRegionType.OTHER
            elif normative_marker_pending and self._root_candidate(line, next_lines):
                current = DocumentRegionType.NORMATIVE_BODY
                normative_marker_pending = False
            elif self.OTHER_HEADING.fullmatch(line):
                current = DocumentRegionType.OTHER
                normative_marker_pending = False
            elif self.EXPLANATION.fullmatch(line) and self._valid_explanation(next_lines):
                current = DocumentRegionType.EXPLANATION
                normative_marker_pending = False
            elif (
                current == DocumentRegionType.NORMATIVE_BODY
                and self.APPENDIX.fullmatch(line)
                and self._valid_appendix(next_lines)
            ):
                current = DocumentRegionType.APPENDIX
            elif (
                current in {
                    DocumentRegionType.FRONT_MATTER,
                    DocumentRegionType.NORMATIVE_BODY,
                }
                and self._repealed_candidate(line, next_lines)
                and self._valid_repealed_list(next_lines)
            ):
                current = DocumentRegionType.REPEALED_LIST
            elif current == DocumentRegionType.FRONT_MATTER and (
                self._root_candidate(line, next_lines)
                or (
                    not toc_observed
                    and (
                        self.ARTICLE.match(line)
                        or self._chapter_start_candidate(line, next_lines)
                    )
                )
            ):
                current = DocumentRegionType.NORMATIVE_BODY
            elif current == DocumentRegionType.REPEALED_LIST and self._root_candidate(line, next_lines):
                current = DocumentRegionType.NORMATIVE_BODY
            elif current == DocumentRegionType.APPENDIX and self._root_candidate(line, next_lines):
                # A root-shaped heading after an appendix is ambiguous without an
                # explicit 正文 marker. Never promote it to normative evidence.
                current = DocumentRegionType.OTHER

            self._record(item, current, line_regions, ordered)

        return RegionSegmentationResult(
            regions=self._collapse(ordered), line_regions=line_regions
        )

    def _is_toc_entry(self, line: str) -> bool:
        return bool(self.TOC_ENTRY.search(line))

    def _root_candidate(self, line: str, next_lines: list[str]) -> bool:
        if self.ROOT_HEADING.fullmatch(line):
            return True
        combined = line + (next_lines[0] if next_lines else "")
        return self.ROOT_HEADING.fullmatch(combined) is not None

    def _chapter_start_candidate(self, line: str, next_lines: list[str]) -> bool:
        """Allow a chapter-scoped extract to start without a preceding TOC."""

        return self.CHAPTER_HEADING.match(line) is not None and any(
            self.ARTICLE.match(candidate) for candidate in next_lines
        )

    def _valid_explanation(self, next_lines: list[str]) -> bool:
        return any(
            self.CHAPTER_HEADING.match(line) or self.ARTICLE.match(line)
            for line in next_lines
        )

    def _valid_appendix(self, next_lines: list[str]) -> bool:
        return bool(next_lines) and (
            self.APPENDIX_ARTICLE.match(next_lines[0]) is not None
            or not self._is_toc_entry(next_lines[0])
        )

    def _valid_repealed_list(self, next_lines: list[str]) -> bool:
        return any(
            self.ARTICLE.match(line) or self.REPEALED_ITEM.match(line)
            for line in next_lines
        )

    def _repealed_candidate(self, line: str, next_lines: list[str]) -> bool:
        if self.REPEALED_HEADING.fullmatch(line):
            return True
        combined = line + (next_lines[0] if next_lines else "")
        return self.REPEALED_INTRO.search(combined) is not None

    @staticmethod
    def _flatten(pages: list[StandardPage]) -> list[RegionSourceLine]:
        result: list[RegionSourceLine] = []
        for page in pages:
            for line_index, raw_line in enumerate(page.text.splitlines()):
                result.append(
                    RegionSourceLine(
                        page_number=page.page_number,
                        line_index=line_index,
                        text=re.sub(r"\s+", " ", raw_line.strip()),
                    )
                )
        return result

    @staticmethod
    def _next_significant(
        lines: list[RegionSourceLine], position: int, *, limit: int
    ) -> list[str]:
        result: list[str] = []
        for item in lines[position + 1:]:
            if item.text:
                result.append(item.text)
            if len(result) == limit:
                break
        return result

    @staticmethod
    def _record(
        item: RegionSourceLine,
        region: DocumentRegionType,
        line_regions: dict[tuple[int, int], DocumentRegionType],
        ordered: list[tuple[int, int, DocumentRegionType]],
    ) -> None:
        line_regions[(item.page_number, item.line_index)] = region
        ordered.append((item.page_number, item.line_index, region))

    @staticmethod
    def _collapse(
        ordered: list[tuple[int, int, DocumentRegionType]]
    ) -> list[DocumentRegion]:
        if not ordered:
            return []
        result: list[DocumentRegion] = []
        start_page, start_line, active = ordered[0]
        previous_page, previous_line = start_page, start_line
        for page, line, region_type in ordered[1:]:
            if region_type != active:
                result.append(
                    DocumentRegion(
                        region_type=active,
                        source_page_start=start_page,
                        source_page_end=previous_page,
                        start_line_index=start_line,
                        end_line_index=previous_line,
                    )
                )
                start_page, start_line, active = page, line, region_type
            previous_page, previous_line = page, line
        result.append(
            DocumentRegion(
                region_type=active,
                source_page_start=start_page,
                source_page_end=previous_page,
                start_line_index=start_line,
                end_line_index=previous_line,
            )
        )
        return result
