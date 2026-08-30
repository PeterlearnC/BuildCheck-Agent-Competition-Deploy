"""Small deterministic relevance refinements for already retrieved candidates."""

import re

from app.schemas.standards_retrieval import StandardSearchHit
from app.services.retrieval.query_normalization_service import QueryNormalizationService


class RetrievalReranker:
    ARTICLE_NUMBER = re.compile(r"(?<!\d)(?:\d+\.){2,3}\d+(?:-\d+)?(?!\d)")

    def __init__(self, normalizer: QueryNormalizationService | None = None) -> None:
        self.normalizer = normalizer or QueryNormalizationService()

    def rerank(self, query: str, hits: list[StandardSearchHit]) -> list[StandardSearchHit]:
        query_normalized = self.normalizer.normalize(query)
        query_tokens = set(self.normalizer.tokenize(query))
        query_numbers = set(self.ARTICLE_NUMBER.findall(query_normalized))
        reranked: list[StandardSearchHit] = []
        for hit in hits:
            content = self.normalizer.normalize(hit.content)
            candidate_tokens = set(
                self.normalizer.tokenize(
                    " ".join(value for value in (hit.content, hit.chapter_title or "") if value)
                )
            )
            boost = 0.0
            boost += 4.0 * sum(
                phrase in content for phrase in self.normalizer.cjk_phrases(query_normalized)
            )
            if hit.article_number in query_numbers:
                boost += 10.0
            if query_tokens:
                boost += 3.0 * len(query_tokens.intersection(candidate_tokens)) / len(query_tokens)
            standard_code = self.normalizer.normalize(hit.standard_code or "")
            if standard_code and standard_code in query_normalized:
                boost += 5.0
            base = hit.hybrid_score or 0.0
            reranked.append(hit.model_copy(update={"rerank_score": base + boost / 100.0}))
        reranked.sort(
            key=lambda item: (
                -(item.rerank_score or 0.0),
                -(item.hybrid_score or 0.0),
                -(item.keyword_score or 0.0),
                -(item.vector_score or 0.0),
                item.standard_id,
                item.article_id,
            )
        )
        return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(reranked, 1)]
