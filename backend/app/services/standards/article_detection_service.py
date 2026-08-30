"""High-confidence article-number candidate detection."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ArticleStart:
    article_number: str
    inline_content: str
    raw_line: str


class ArticleDetectionService:
    START = re.compile(
        r"^(?:第\s*)?"
        r"(?P<number>(?:\d+\.){2,3}\d+(?:-\d+)?|[A-Z]\.(?:\d+\.)+\d+(?:-\d+)?)"
        r"\s*(?:条)?\s*(?P<content>.*)$",
        re.IGNORECASE,
    )
    EXCLUDED_PREFIX = re.compile(r"^(?:表|图|公式)\s*[A-Z]?\d", re.IGNORECASE)
    DATE_NUMBER = re.compile(r"^(?:19|20)\d{2}\.\d{1,2}\.\d{1,2}$")
    ARTICLE_REFERENCE = re.compile(
        r"^(?:第\s*)?(?:\d+\.){2,3}\d+(?:-\d+)?条(?:所列|规定|中|所述|要求)"
    )

    def detect_start(self, line: str) -> ArticleStart | None:
        value = re.sub(r"\s+", " ", line.strip())
        if not value or self.EXCLUDED_PREFIX.match(value):
            return None
        if self.ARTICLE_REFERENCE.match(value):
            return None
        match = self.START.fullmatch(value)
        if not match:
            return None
        number = match.group("number").upper()
        if self.DATE_NUMBER.fullmatch(number):
            return None
        first = number.split(".", 1)[0]
        if first.isdigit() and int(first) >= 1900:
            return None
        return ArticleStart(number, match.group("content").strip(), value)
