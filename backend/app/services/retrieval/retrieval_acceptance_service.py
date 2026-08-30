"""Deterministic relevance acceptance gate for standards retrieval."""

from collections import Counter
from dataclasses import dataclass
import math
import re

from app.schemas.standards_retrieval import RetrievalDecision, StandardSearchHit
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.query_normalization_service import QueryNormalizationService


@dataclass(frozen=True)
class RetrievalAcceptanceResult:
    decision: RetrievalDecision
    reason: str
    query_coverage: float


class RetrievalAcceptanceService:
    ARTICLE_NUMBER = re.compile(r"(?<!\d)(?:\d+\.){2,3}\d+(?:-\d+)?(?!\d)")

    def __init__(self, normalizer: QueryNormalizationService | None = None) -> None:
        self.normalizer = normalizer or QueryNormalizationService()

    def evaluate(
        self,
        query: str,
        records: list[RetrievalRecord],
        hits: list[StandardSearchHit],
    ) -> RetrievalAcceptanceResult:
        if not records or not hits:
            return RetrievalAcceptanceResult(
                RetrievalDecision.NO_MATCH, "no positive retrieval candidate", 0.0
            )
        normalized_query = self.normalizer.normalize(query)
        query_numbers = set(self.ARTICLE_NUMBER.findall(normalized_query))
        semantic_query = self.ARTICLE_NUMBER.sub(" ", normalized_query)
        semantic_tokens = self.normalizer.tokenize(semantic_query)
        if not semantic_tokens and not query_numbers:
            return RetrievalAcceptanceResult(
                RetrievalDecision.NO_MATCH, "query has no meaningful tokens", 0.0
            )

        corpus_token_sets = [self._record_tokens(record) for record in records]
        frequencies = Counter(
            token for tokens in corpus_token_sets for token in set(tokens)
        )
        top_tokens = self._hit_tokens(hits[0])
        total_mass = 0.0
        matched_mass = 0.0
        for token in semantic_tokens:
            idf = math.log(1 + (len(records) + 0.5) / (frequencies[token] + 0.5))
            total_mass += idf
            if token in top_tokens:
                matched_mass += idf
        coverage = matched_mass / total_mass if total_mass else 0.0

        exact_article_number = hits[0].article_number in query_numbers
        if exact_article_number and not semantic_tokens:
            return RetrievalAcceptanceResult(
                RetrievalDecision.ACCEPT,
                "exact article-number-only query",
                1.0,
            )
        normalized_content = self.normalizer.normalize(hits[0].content)
        exact_phrase = any(
            phrase in normalized_content
            for phrase in self.normalizer.cjk_phrases(semantic_query)
        )
        strong_term_match = any(
            len(token) >= 3 and token in top_tokens
            for token in semantic_tokens
            if any("\u3400" <= character <= "\u9fff" for character in token)
        )
        if exact_article_number and coverage >= 0.25 and strong_term_match:
            return RetrievalAcceptanceResult(
                RetrievalDecision.ACCEPT,
                "exact article number with sufficient semantic token coverage",
                round(coverage, 6),
            )
        if coverage >= 0.25 and strong_term_match:
            return RetrievalAcceptanceResult(
                RetrievalDecision.ACCEPT,
                "strong technical term with sufficient weighted coverage",
                round(coverage, 6),
            )
        if coverage >= 0.6 or (coverage >= 0.45 and exact_phrase):
            return RetrievalAcceptanceResult(
                RetrievalDecision.ACCEPT,
                "sufficient IDF-weighted query coverage",
                round(coverage, 6),
            )
        if coverage >= 0.30:
            return RetrievalAcceptanceResult(
                RetrievalDecision.LOW_CONFIDENCE,
                "partial IDF-weighted query coverage",
                round(coverage, 6),
            )
        return RetrievalAcceptanceResult(
            RetrievalDecision.NO_MATCH,
            "insufficient IDF-weighted query coverage",
            round(coverage, 6),
        )

    def _record_tokens(self, record: RetrievalRecord) -> set[str]:
        return set(
            self.normalizer.tokenize(
                " ".join(
                    value
                    for value in (
                        record.document.standard_code,
                        record.document.standard_name,
                        record.article.chapter_title,
                        record.article.article_number,
                        record.article.content,
                    )
                    if value
                )
            )
        )

    def _hit_tokens(self, hit: StandardSearchHit) -> set[str]:
        return set(
            self.normalizer.tokenize(
                " ".join(
                    value
                    for value in (
                        hit.standard_code,
                        hit.standard_name,
                        hit.chapter_title,
                        hit.article_number,
                        hit.content,
                    )
                    if value
                )
            )
        )
