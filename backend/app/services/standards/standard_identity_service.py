"""Scored, conflict-aware canonical identity recovery for standards."""

from dataclasses import dataclass
import re
import unicodedata

from app.schemas.standards import (
    MetadataConfidence,
    StandardIdentityStatus,
    StandardPage,
)


@dataclass(frozen=True)
class StandardIdentityResult:
    canonical_code: str | None
    display_code: str | None
    standard_name: str | None
    status: StandardIdentityStatus
    confidence: MetadataConfidence
    reason: str


@dataclass(frozen=True)
class IdentityCandidate:
    code: str
    display_code: str
    name: str | None
    page: int
    line: int
    distance: int | None
    same_line: bool
    title_context: bool
    cover_context: bool
    candidate_score: int
    reference_context: bool


@dataclass(frozen=True)
class _CodeOccurrence:
    canonical: str
    display: str
    page_number: int
    line_index: int
    start: int
    end: int
    line: str


class StandardIdentityService:
    PREFIX = r"(?:GB|JGJ|CJJ)(?:\s*/\s*T)?|T\s*/\s*CECS"
    CODE_PATTERN = re.compile(
        rf"(?<![A-Z0-9])(?P<prefix>{PREFIX})\s*"
        r"(?P<number>\d{1,6}(?:\.\d+)?)\s*"
        r"[-—–－一]\s*(?P<year>(?:19|20)\d{2})(?!\d)",
        re.IGNORECASE,
    )
    NAME_SUFFIX = re.compile(
        r"(?:规范|标准|规程|指南|Standard|Code|Specification)$", re.IGNORECASE
    )
    TITLE_CONTEXT = re.compile(
        r"中华人民共和国(?:国家|行业)标准|标准编号|规范正式标题|正式标题"
    )
    REFERENCE_CONTEXT = re.compile(r"依据|按照|符合|引用|参见|执行")
    EXCLUDED_NAME = re.compile(
        r"中华人民共和国(?:国家|行业)标准|国家标准|行业标准|"
        r"住房和城乡建设部|国家市场监督管理总局|浏览专用|"
        r"发布|实施|批准|公告|标准编号|编号|实施日期|发布日期|前言"
    )

    def identify(self, pages: list[StandardPage]) -> StandardIdentityResult:
        cover_pages = sorted(pages, key=lambda page: page.page_number)[:3]
        occurrences = self._code_occurrences(cover_pages)
        if not occurrences:
            return StandardIdentityResult(
                None,
                None,
                self._standalone_name(cover_pages),
                StandardIdentityStatus.IDENTITY_UNCERTAIN,
                MetadataConfidence.LOW,
                "no supported standard-code candidate",
            )

        candidates = sorted(
            (self._score_candidate(item, cover_pages) for item in occurrences),
            key=lambda item: (-item.candidate_score, item.page, item.line, item.code),
        )
        top = candidates[0]
        runner_up = next(
            (item for item in candidates[1:] if item.code != top.code), None
        )
        conflicting = (
            runner_up is not None
            and top.candidate_score - runner_up.candidate_score < 30
        )
        clearly_bound = (
            top.name is not None
            and not top.reference_context
            and (top.same_line or (top.distance is not None and top.distance <= 2))
        )

        if conflicting:
            return StandardIdentityResult(
                None,
                None,
                None,
                StandardIdentityStatus.IDENTITY_UNCERTAIN,
                MetadataConfidence.LOW,
                "multiple standard-code candidates lack a decisive title binding",
            )
        if clearly_bound:
            return StandardIdentityResult(
                top.code,
                top.display_code,
                top.name,
                StandardIdentityStatus.CONFIRMED,
                MetadataConfidence.HIGH,
                "standard code is explicitly bound to a nearby formal title",
            )
        return StandardIdentityResult(
            top.code,
            top.display_code,
            top.name,
            StandardIdentityStatus.IDENTITY_UNCERTAIN,
            MetadataConfidence.MEDIUM if top.name else MetadataConfidence.LOW,
            "standard code candidate is not decisively bound to a formal title",
        )

    def canonicalize(self, value: str | None) -> str | None:
        occurrences = self._code_occurrences(
            [StandardPage(page_number=1, text=value or "")]
        )
        return occurrences[0].canonical if occurrences else None

    def display(self, value: str | None) -> str | None:
        occurrences = self._code_occurrences(
            [StandardPage(page_number=1, text=value or "")]
        )
        return occurrences[0].display if occurrences else None

    def _score_candidate(
        self, occurrence: _CodeOccurrence, pages: list[StandardPage]
    ) -> IdentityCandidate:
        page = next(item for item in pages if item.page_number == occurrence.page_number)
        lines = [unicodedata.normalize("NFKC", line).strip() for line in page.text.splitlines()]
        same_line_name = self._clean_name(
            occurrence.line[:occurrence.start] + occurrence.line[occurrence.end:]
        )
        name = same_line_name
        distance: int | None = 0 if same_line_name else None
        if name is None:
            nearby: list[tuple[int, str]] = []
            for offset in (1, -1, 2, -2, 3, -3):
                index = occurrence.line_index + offset
                if not 0 <= index < len(lines):
                    continue
                cleaned = self._clean_name(lines[index])
                if cleaned:
                    nearby.append((abs(offset), cleaned))
            for first in range(max(0, occurrence.line_index - 3), occurrence.line_index + 3):
                second = first + 1
                if second >= len(lines) or first == occurrence.line_index or second == occurrence.line_index:
                    continue
                cleaned = self._clean_name(lines[first] + lines[second])
                if cleaned:
                    nearby.append(
                        (
                            max(
                                abs(first - occurrence.line_index),
                                abs(second - occurrence.line_index),
                            ),
                            cleaned,
                        )
                    )
            if nearby:
                distance, name = max(
                    nearby, key=lambda item: (len(item[1]) - item[0] * 3, -item[0])
                )

        context = " ".join(
            lines[max(0, occurrence.line_index - 2): occurrence.line_index + 3]
        )
        title_context = bool(self.TITLE_CONTEXT.search(context))
        reference_context = bool(self.REFERENCE_CONTEXT.search(occurrence.line))
        cover_context = occurrence.page_number <= 2
        score = 100 - (occurrence.page_number - 1) * 20
        if same_line_name:
            score += 120
        elif name is not None and distance is not None:
            score += 95 - distance * 15
        if title_context:
            score += 25
        if cover_context:
            score += 20
        if reference_context:
            score -= 120
        return IdentityCandidate(
            code=occurrence.canonical,
            display_code=occurrence.display,
            name=name,
            page=occurrence.page_number,
            line=occurrence.line_index,
            distance=distance,
            same_line=same_line_name is not None,
            title_context=title_context,
            cover_context=cover_context,
            candidate_score=score,
            reference_context=reference_context,
        )

    def _code_occurrences(self, pages: list[StandardPage]) -> list[_CodeOccurrence]:
        result: list[_CodeOccurrence] = []
        for page in pages:
            for line_index, raw_line in enumerate(page.text.splitlines()):
                line = unicodedata.normalize("NFKC", raw_line).strip()
                for match in self.CODE_PATTERN.finditer(line):
                    prefix = re.sub(r"\s+", "", match.group("prefix")).upper()
                    number = match.group("number")
                    year = match.group("year")
                    result.append(
                        _CodeOccurrence(
                            canonical=f"{prefix}{number}-{year}",
                            display=f"{prefix} {number}-{year}",
                            page_number=page.page_number,
                            line_index=line_index,
                            start=match.start(),
                            end=match.end(),
                            line=line,
                        )
                    )
        return sorted(result, key=lambda item: (item.page_number, item.line_index))

    def _standalone_name(self, pages: list[StandardPage]) -> str | None:
        candidates: list[tuple[int, str]] = []
        for page in pages:
            for line in page.text.splitlines():
                cleaned = self._clean_name(line)
                if cleaned:
                    candidates.append((100 - page.page_number * 20 + len(cleaned), cleaned))
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def _clean_name(self, value: str | None) -> str | None:
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))
        compact = compact.strip("《》:：-—–－一")
        if not 4 <= len(compact) <= 80:
            return None
        if self.EXCLUDED_NAME.search(compact) or self.CODE_PATTERN.search(compact):
            return None
        return compact if self.NAME_SUFFIX.search(compact) else None
