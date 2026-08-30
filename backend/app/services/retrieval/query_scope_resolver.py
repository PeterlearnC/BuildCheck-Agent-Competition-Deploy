"""Resolve explicit standard identifiers into fail-closed retrieval scope."""

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata

from app.schemas.standards import StandardDocument
from app.services.standards.standard_identity_service import StandardIdentityService


class QueryScopeStatus(str, Enum):
    NO_EXPLICIT_SCOPE = "NO_EXPLICIT_SCOPE"
    RESOLVED_UNIQUE_SCOPE = "RESOLVED_UNIQUE_SCOPE"
    UNKNOWN_SCOPE = "UNKNOWN_SCOPE"
    AMBIGUOUS_SCOPE = "AMBIGUOUS_SCOPE"
    CONFLICTING_SCOPE = "CONFLICTING_SCOPE"
    INVALID_SCOPE = "INVALID_SCOPE"


@dataclass(frozen=True)
class QueryScopeResolution:
    status: QueryScopeStatus
    resolved_standard_ids: tuple[str, ...] | None
    consumed_scope_spans: tuple[tuple[int, int], ...]
    residual_query: str
    article_numbers: tuple[str, ...]
    reason: str

    @property
    def failed(self) -> bool:
        return self.status in {
            QueryScopeStatus.UNKNOWN_SCOPE,
            QueryScopeStatus.AMBIGUOUS_SCOPE,
            QueryScopeStatus.CONFLICTING_SCOPE,
            QueryScopeStatus.INVALID_SCOPE,
        }


class QueryScopeResolver:
    """Separate standard scope metadata from relevance-bearing query text."""

    ARTICLE_NUMBER = re.compile(r"(?<!\d)(?:\d+\.){2,3}\d+(?:-\d+)?(?!\d)")
    CODE_PREFIX = re.compile(
        r"(?<![A-Z0-9])(?:(?:GB|JGJ|CJJ)(?:\s*/\s*T)?|T\s*/\s*CECS)",
        re.IGNORECASE,
    )

    def __init__(self, identity: StandardIdentityService | None = None) -> None:
        self.identity = identity or StandardIdentityService()

    def resolve(
        self,
        query: str,
        documents: list[StandardDocument],
        caller_standard_ids: list[str] | None,
    ) -> QueryScopeResolution:
        normalized = unicodedata.normalize("NFKC", query or "")
        matches = list(self.identity.CODE_PATTERN.finditer(normalized))
        caller_scope = (
            tuple(dict.fromkeys(caller_standard_ids))
            if caller_standard_ids is not None
            else None
        )

        if not matches:
            article_numbers = tuple(
                dict.fromkeys(self.ARTICLE_NUMBER.findall(normalized))
            )
            if self._has_invalid_code_like_fragment(normalized, ()):
                return QueryScopeResolution(
                    QueryScopeStatus.INVALID_SCOPE,
                    None,
                    (),
                    normalized.strip(),
                    article_numbers,
                    "query contains a malformed standard-code expression",
                )
            return QueryScopeResolution(
                QueryScopeStatus.NO_EXPLICIT_SCOPE,
                caller_scope,
                (),
                normalized.strip(),
                article_numbers,
                "query contains no explicit standard scope",
            )

        spans = tuple((match.start(), match.end()) for match in matches)
        if self._has_invalid_code_like_fragment(normalized, spans):
            return QueryScopeResolution(
                QueryScopeStatus.INVALID_SCOPE,
                None,
                spans,
                self._without_spans(normalized, spans),
                (),
                "query contains a malformed standard-code expression",
            )

        canonical_codes = tuple(
            dict.fromkeys(
                code.casefold()
                for match in matches
                if (code := self.identity.canonicalize(match.group(0))) is not None
            )
        )
        residual_query = self._without_spans(normalized, spans)
        article_numbers = tuple(
            dict.fromkeys(self.ARTICLE_NUMBER.findall(residual_query))
        )
        if len(canonical_codes) != 1:
            return QueryScopeResolution(
                QueryScopeStatus.CONFLICTING_SCOPE,
                None,
                spans,
                residual_query,
                article_numbers,
                "query contains conflicting standard-code expressions",
            )

        by_code: dict[str, list[str]] = {}
        for document in documents:
            canonical = self.identity.canonicalize(
                document.canonical_standard_code
            ) or self.identity.canonicalize(document.standard_code)
            if canonical:
                by_code.setdefault(canonical.casefold(), []).append(document.standard_id)

        resolved = tuple(dict.fromkeys(by_code.get(canonical_codes[0], [])))
        if not resolved:
            return QueryScopeResolution(
                QueryScopeStatus.UNKNOWN_SCOPE,
                None,
                spans,
                residual_query,
                article_numbers,
                "query standard code is not available in the retrieval corpus",
            )
        if len(resolved) != 1:
            return QueryScopeResolution(
                QueryScopeStatus.AMBIGUOUS_SCOPE,
                None,
                spans,
                residual_query,
                article_numbers,
                "query standard code resolves to multiple retrieval documents",
            )
        if caller_scope is not None and resolved[0] not in caller_scope:
            return QueryScopeResolution(
                QueryScopeStatus.CONFLICTING_SCOPE,
                None,
                spans,
                residual_query,
                article_numbers,
                "query standard code conflicts with caller-supplied standard_ids",
            )
        return QueryScopeResolution(
            QueryScopeStatus.RESOLVED_UNIQUE_SCOPE,
            resolved,
            spans,
            residual_query,
            article_numbers,
            "query standard code resolved to a unique retrieval document",
        )

    def _has_invalid_code_like_fragment(
        self, normalized_query: str, valid_spans: tuple[tuple[int, int], ...]
    ) -> bool:
        for prefix in self.CODE_PREFIX.finditer(normalized_query):
            tail = normalized_query[prefix.end() : prefix.end() + 24]
            if not re.match(r"\s*\d", tail):
                continue
            if not any(start <= prefix.start() < end for start, end in valid_spans):
                return True
        return False

    @staticmethod
    def _without_spans(value: str, spans: tuple[tuple[int, int], ...]) -> str:
        characters = list(value)
        for start, end in spans:
            characters[start:end] = " " * (end - start)
        return " ".join("".join(characters).split())
