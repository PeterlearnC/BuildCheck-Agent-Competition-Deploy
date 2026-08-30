"""Conservative cover-first metadata extraction for standards."""

from dataclasses import dataclass
from datetime import date
import re

from app.schemas.standards import (
    MetadataConfidence,
    StandardMetadataResult,
    StandardPage,
)
from app.services.standards.standard_identity_service import StandardIdentityService


@dataclass(frozen=True)
class _Candidate:
    value: str
    score: int
    page: int


class StandardMetadataService:
    CODE_PATTERN = re.compile(
        r"(?<![A-Z0-9])(?P<prefix>GB(?:/T)?|JGJ(?:/T)?|DB\d{1,2}(?:/T)?|T/CECS)"
        r"\s*(?P<body>[A-Z]?\s*\d{2,6}(?:\.\d+)?)\s*[-—–]\s*(?P<year>\d{4})(?!\d)",
        re.IGNORECASE,
    )
    EXPLICIT_EDITION = re.compile(r"(?<!\d)(19\d{2}|20\d{2})\s*年?版")
    DATE_PATTERN = re.compile(r"(19\d{2}|20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?")
    NAME_SUFFIX = re.compile(r"(?:规范|标准|规程|指南|Standard|Code|Specification)$", re.IGNORECASE)
    GENERIC_NAMES = {
        "中华人民共和国国家标准",
        "中华人民共和国行业标准",
        "国家标准",
        "行业标准",
        "团体标准",
    }
    EXCLUDED_NAME = re.compile(
        r"发布|实施|批准|住房和城乡建设部|国家市场监督管理总局|标准编号|编号|公告|前言"
    )

    def extract(self, pages: list[StandardPage]) -> StandardMetadataResult:
        cover_pages = sorted(pages, key=lambda item: item.page_number)[:3]
        identity = StandardIdentityService().identify(cover_pages)
        code_candidate = self._best_code(cover_pages)
        name_candidate = self._best_name(cover_pages)
        explicit_editions = self._explicit_editions(cover_pages)
        code_year = self._code_year(identity.canonical_code) if identity.canonical_code else None
        confidence = identity.confidence

        edition: str | None
        if explicit_editions:
            explicit = explicit_editions[0]
            if code_year and explicit != code_year:
                edition = None
                confidence = MetadataConfidence.LOW
            else:
                edition = explicit
        else:
            edition = code_year

        return StandardMetadataResult(
            standard_code=identity.display_code,
            standard_name=identity.standard_name,
            canonical_standard_code=identity.canonical_code,
            display_standard_code=identity.display_code,
            identity_status=identity.status,
            identity_confidence=identity.confidence,
            identity_reason=identity.reason,
            edition=edition,
            publish_date=self._labeled_date(cover_pages, r"发布(?:日期)?"),
            effective_date=self._labeled_date(cover_pages, r"实施(?:日期)?"),
            confidence=confidence,
        )

    def _best_code(self, pages: list[StandardPage]) -> _Candidate | None:
        candidates: list[_Candidate] = []
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                for match in self.CODE_PATTERN.finditer(line):
                    prefix = match.group("prefix").upper()
                    body = re.sub(r"\s+", "", match.group("body")).upper()
                    value = f"{prefix} {body}-{match.group('year')}"
                    context = " ".join(lines[max(0, index - 1): index + 2])
                    score = 120 - (page.page_number - 1) * 25
                    if re.search(r"标准编号|编号", context):
                        score += 50
                    if re.search(r"规范|标准|规程|Standard|Code", context, re.IGNORECASE):
                        score += 15
                    candidates.append(_Candidate(value, score, page.page_number))
        return max(candidates, key=lambda item: item.score, default=None)

    def _best_name(self, pages: list[StandardPage]) -> _Candidate | None:
        candidates: list[_Candidate] = []
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                for value in (line, self._pair(lines, index)):
                    if not value or not self._valid_name(value):
                        continue
                    score = 100 - (page.page_number - 1) * 25
                    score += min(len(value), 40)
                    if index <= 5:
                        score += 15
                    candidates.append(_Candidate(value, score, page.page_number))
        return max(candidates, key=lambda item: item.score, default=None)

    def _valid_name(self, value: str) -> bool:
        compact = re.sub(r"\s+", "", value).strip("《》")
        if not (4 <= len(compact) <= 80):
            return False
        if (
            compact in self.GENERIC_NAMES
            or any(compact.startswith(banner) for banner in self.GENERIC_NAMES)
            or self.EXCLUDED_NAME.search(compact)
        ):
            return False
        if self.CODE_PATTERN.search(compact):
            return False
        return bool(self.NAME_SUFFIX.search(compact))

    @staticmethod
    def _pair(lines: list[str], index: int) -> str | None:
        if index + 1 >= len(lines):
            return None
        first, second = lines[index], lines[index + 1]
        if not (2 <= len(first) <= 35 and 2 <= len(second) <= 45):
            return None
        return re.sub(r"\s+", "", first + second).strip("《》")

    def _explicit_editions(self, pages: list[StandardPage]) -> list[str]:
        result: list[str] = []
        for page in pages:
            result.extend(match.group(1) for match in self.EXPLICIT_EDITION.finditer(page.text))
        return result

    def _labeled_date(self, pages: list[StandardPage], label: str) -> date | None:
        pattern = re.compile(label + r"\s*[:：]?\s*" + self.DATE_PATTERN.pattern)
        for page in pages:
            match = pattern.search(page.text)
            if not match:
                continue
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                return None
        return None

    @staticmethod
    def _code_year(value: str) -> str | None:
        match = re.search(r"-(\d{4})$", value)
        return match.group(1) if match else None

    @staticmethod
    def _confidence(
        code: _Candidate | None, name: _Candidate | None
    ) -> MetadataConfidence:
        if code and name and max(code.page, name.page) <= 2:
            return MetadataConfidence.HIGH
        if code or name:
            return MetadataConfidence.MEDIUM
        return MetadataConfidence.UNKNOWN
