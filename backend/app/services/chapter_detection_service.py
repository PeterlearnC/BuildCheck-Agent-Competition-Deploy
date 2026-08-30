"""Conservative chapter discovery based on contents and numbered headings."""

import re
from dataclasses import dataclass

from app.schemas.analysis import ChapterInfo, SectionPresence
from app.services.pdf_service import PDFPageText
from app.services.semantic_region_service import SemanticRegionService


@dataclass(frozen=True)
class HeadingCandidate:
    title: str
    level: int


class ChapterDetectionService:
    KNOWN_TITLES = (
        "工程概况", "编制依据", "编制说明及依据", "依据文件", "主要规范标准",
        "施工准备", "施工部署", "施工计划", "施工进度计划",
        "施工方法", "系统安装及施工", "施工工艺", "施工工艺技术", "安装施工", "专项施工方案",
        "施工技术", "施工方案", "质量保证措施", "质量标准", "安全保证措施",
        "施工安全保障措施",
        "安全防护措施", "文明施工", "职业健康安全", "环境保护", "应急预案", "成品保护",
        "计算书", "附图及计算书", "设计计算", "计算分析", "验算", "相关计算书",
    )
    CONTENTS_TITLE = re.compile(r"^\s*(?:目录|目\s*录|contents)\s*$", re.IGNORECASE)
    CONTENTS_PAGE_SUFFIX = re.compile(r"(?:\.{2,}|…{2,}|·{2,}|\s{2,})\s*\d+\s*$")
    CHAPTER_PREFIX = re.compile(r"^第[一二三四五六七八九十百]+[章节篇]\s*")
    CHINESE_PREFIX = re.compile(r"^([一二三四五六七八九十百]+)、\s*")
    PAREN_CHINESE_PREFIX = re.compile(r"^[（(]([一二三四五六七八九十百]+)[）)]\s*")
    ARABIC_COMMA_PREFIX = re.compile(r"^(\d+)、\s*")
    ARABIC_DOT_PREFIX = re.compile(r"^(\d+)[.．]\s+")
    ARABIC_SPACE_PREFIX = re.compile(r"^(\d+)\s+")
    DECIMAL_PREFIX = re.compile(r"^(\d+(?:\.\d+)+)(?:[、.．]\s*|\s+)")
    PARAMETER_PATTERN = re.compile(
        r"(?:MPa|Mpa|kPa|mm|cm|m²|m3|%|℃|°|~|～)", re.IGNORECASE
    )

    def __init__(self) -> None:
        self.semantic_regions = SemanticRegionService()

    def detect(self, pages: list[PDFPageText]) -> list[ChapterInfo]:
        contents_index = self._contents_page_index(pages)
        found: list[tuple[str, int, int]] = []
        if contents_index is not None:
            candidates = self._contents_candidates(pages, contents_index)
            body_start = self._body_start_after_contents(pages, contents_index)
            toc_matches = self._match_contents_to_body(pages, body_start, candidates)
            body_headings = self._fallback_headings(pages[body_start:])
            # Explicit body headings are authoritative. TOC-derived matches only
            # fill titles not found by the general body scanner.
            found = list(body_headings)
            body_titles = {item[0] for item in body_headings}
            found.extend(item for item in toc_matches if item[0] not in body_titles)
            found.sort(key=lambda item: item[1])
            found = self._deduplicate(found)
        else:
            found = self._fallback_headings(pages)
        return self._build_chapters(found, pages[-1].page_number if pages else 0)

    def _body_start_after_contents(
        self, pages: list[PDFPageText], contents_index: int
    ) -> int:
        last_contents = contents_index
        for index in range(contents_index, min(len(pages), contents_index + 5)):
            lines = [line.strip() for line in pages[index].text.splitlines() if line.strip()]
            if index == contents_index or any(
                self.CONTENTS_PAGE_SUFFIX.search(line) for line in lines
            ):
                last_contents = index
                continue
            break
        return min(last_contents + 1, len(pages))

    def _contents_page_index(self, pages: list[PDFPageText]) -> int | None:
        for index, page in enumerate(pages):
            if any(self.CONTENTS_TITLE.match(line) for line in page.text.splitlines()):
                return index
        return None

    def _contents_candidates(
        self, pages: list[PDFPageText], contents_index: int
    ) -> list[HeadingCandidate]:
        candidates: list[HeadingCandidate] = []
        seen: set[str] = set()
        for page in pages[contents_index : contents_index + 3]:
            for line in page.text.splitlines():
                cleaned = self.CONTENTS_PAGE_SUFFIX.sub("", line.strip()).strip()
                candidate = self._parse_heading(
                    cleaned, allow_known=True, allow_arabic_comma=True
                )
                if candidate and candidate.title not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.title)
        return candidates

    def _match_contents_to_body(
        self,
        pages: list[PDFPageText],
        body_start: int,
        candidates: list[HeadingCandidate],
    ) -> list[tuple[str, int, int]]:
        found: list[tuple[str, int, int]] = []
        search_pages = pages[body_start:]
        regions = self.semantic_regions.classify(pages)
        for candidate in candidates:
            for page in search_pages:
                if (
                    regions.get(page.page_number) == "calculation"
                    and not self.semantic_regions.is_calculation_heading(candidate.title)
                ):
                    continue
                if any(
                    (parsed := self._parse_heading(
                        line.strip(), allow_known=True, allow_arabic_comma=True
                    ))
                    and parsed.title == candidate.title
                    for line in page.text.splitlines()
                ):
                    found.append((candidate.title, page.page_number, candidate.level))
                    break
        found.sort(key=lambda item: item[1])
        return self._deduplicate(found)

    def _fallback_headings(self, pages: list[PDFPageText]) -> list[tuple[str, int, int]]:
        found: list[tuple[str, int, int]] = []
        regions = self.semantic_regions.classify(pages)
        for page in pages:
            for line in page.text.splitlines():
                if (
                    regions.get(page.page_number) == "calculation"
                    and not self.semantic_regions.is_calculation_heading(line.strip())
                ):
                    continue
                candidate = self._parse_heading(line.strip(), allow_known=True)
                if candidate:
                    found.append((candidate.title, page.page_number, candidate.level))
        return self._deduplicate(found)

    def _parse_heading(
        self, line: str, allow_known: bool, allow_arabic_comma: bool = False
    ) -> HeadingCandidate | None:
        if not line or len(line) > 60 or "：" in line or ":" in line:
            return None
        title = ""
        level = 1
        if match := self.CHAPTER_PREFIX.match(line):
            title = line[match.end() :].strip()
        elif match := self.CHINESE_PREFIX.match(line):
            title = line[match.end() :].strip()
        elif match := self.PAREN_CHINESE_PREFIX.match(line):
            title = line[match.end() :].strip()
            level = 2
        elif match := self.ARABIC_COMMA_PREFIX.match(line):
            title = line[match.end() :].strip()
            if not allow_arabic_comma and title.strip(" ：:。") not in self.KNOWN_TITLES:
                return None
        elif match := self.ARABIC_DOT_PREFIX.match(line):
            title = line[match.end() :].strip()
            if not allow_arabic_comma and title.strip(" ：:。") not in self.KNOWN_TITLES:
                return None
        elif match := self.ARABIC_SPACE_PREFIX.match(line):
            title = line[match.end() :].strip()
            if title.strip(" ：:。") not in self.KNOWN_TITLES:
                return None
        elif match := self.DECIMAL_PREFIX.match(line):
            number = match.group(1)
            level = number.count(".") + 1
            title = line[match.end() :].strip()
        elif allow_known and line.strip(" ：:。") in self.KNOWN_TITLES:
            title = line.strip(" ：:。")
        else:
            return None
        if not self._is_heading_title(title):
            return None
        return HeadingCandidate(title=title, level=level)

    def _is_heading_title(self, title: str) -> bool:
        if not 2 <= len(title) <= 30:
            return False
        if self.PARAMETER_PATTERN.search(title):
            return False
        if self.semantic_regions.is_formula_or_unit_expression(title):
            return False
        if re.fullmatch(
            r"(?:GB(?:/T)?|JGJ(?:/T)?|JTG|CJJ|DBJ?)\s*[A-Z]*\s*\d[\d.\-/]*",
            title,
            re.IGNORECASE,
        ):
            return False
        if title.startswith(("）", ")", "、", "。", "，", ",", "；", ";", "~", "～")):
            return False
        if re.match(r"^(?:采用|应当|应(?!急)|必须|不得|严禁|设置|安装|使用|检查|确保)", title):
            return False
        if title.endswith(("。", ".", "，", ",", "；", ";", "：", ":", "……", "…")):
            return False
        if re.fullmatch(r"[\d.\-~～/（）()]+", title):
            return False
        return True

    @staticmethod
    def _deduplicate(values: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        result: list[tuple[str, int, int]] = []
        seen: set[str] = set()
        for item in values:
            if item[0] not in seen:
                result.append(item)
                seen.add(item[0])
        return result

    @staticmethod
    def _build_chapters(
        found: list[tuple[str, int, int]], last_page: int
    ) -> list[ChapterInfo]:
        chapters: list[ChapterInfo] = []
        for index, (title, start_page, level) in enumerate(found):
            next_page = last_page + 1
            for _, following_page, following_level in found[index + 1 :]:
                if following_level <= level:
                    next_page = following_page
                    break
            chapters.append(
                ChapterInfo(
                    title=title,
                    start_page=start_page,
                    # Page-level ranges may overlap when the next heading starts
                    # below the previous section's final paragraphs on the same page.
                    end_page=max(start_page, min(next_page, last_page)),
                    level=level,
                )
            )
        return chapters

    @staticmethod
    def section_presence(chapters: list[ChapterInfo], pages: list[PDFPageText]) -> SectionPresence:
        # ChapterDetectionService is the primary source, but raw heading-like
        # lines are also retained as stable evidence when a valid local heading
        # was not promoted into the final chapter list.
        evidence = [chapter.title for chapter in chapters]
        presence_terms = re.compile(
            r"质量|验收标准|安全|危险源|环境|文明施工|扬尘|防噪音|绿色施工|应急"
        )
        numbered_heading = re.compile(
            r"^(?:第[一二三四五六七八九十百\d]+章|[一二三四五六七八九十]+、|"
            r"\d+(?:\.\d+)+[、.．]?|\d+[、.．])"
        )
        for page in pages:
            for raw_line in page.text.splitlines():
                line = raw_line.strip()
                if (
                    line
                    and len(line) <= 60
                    and presence_terms.search(line)
                    and (
                        numbered_heading.match(line)
                        or not re.search(r"[。；;，,]$", line)
                    )
                ):
                    evidence.append(line)
        searchable = "\n".join(evidence)
        return SectionPresence(
            quality=bool(
                re.search(r"质量|质量保证|质量标准|质量措施|工程质量|验收标准", searchable)
            ),
            safety=bool(re.search(r"安全|安全保障|安全措施|安全生产|危险源", searchable)),
            environment=bool(
                re.search(r"环境|环境保护|文明施工|扬尘|防噪音|绿色施工", searchable)
            ),
            emergency=bool(
                re.search(r"应急|应急处置|应急预案|应急救援|应急响应", searchable)
            ),
        )
