"""Small deterministic BM25 implementation for parsed standard articles."""

from collections import Counter
import math
import re

from app.schemas.standards_retrieval import RetrievalMethod, StandardSearchHit
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.query_normalization_service import QueryNormalizationService


class KeywordRetriever:
    K1 = 1.5
    B = 0.75
    ARTICLE_NUMBER = re.compile(r"(?<!\d)(?:\d+\.){2,3}\d+(?:-\d+)?(?!\d)")

    def __init__(self, normalizer: QueryNormalizationService | None = None) -> None:
        self.normalizer = normalizer or QueryNormalizationService()

    def retrieve(
        self, query: str, records: list[RetrievalRecord], top_k: int
    ) -> list[StandardSearchHit]:
        if not records or top_k <= 0:
            return []
        query_tokens = self.normalizer.tokenize(query)
        if not query_tokens:
            return []
        field_tokens = [self._field_tokens(record) for record in records]
        lengths = [sum(tokens.values()) for tokens in field_tokens]
        average_length = sum(lengths) / len(lengths) if lengths else 1.0
        document_frequency = Counter(
            token for tokens in field_tokens for token in set(tokens)
        )
        scored: list[tuple[float, RetrievalRecord]] = []
        for record, tokens, length in zip(records, field_tokens, lengths, strict=True):
            score = 0.0
            for token in query_tokens:
                frequency = tokens.get(token, 0.0)
                if not frequency:
                    continue
                frequency_in_documents = document_frequency[token]
                inverse_document_frequency = math.log(
                    1 + (len(records) - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
                )
                denominator = frequency + self.K1 * (
                    1 - self.B + self.B * length / max(average_length, 1.0)
                )
                score += inverse_document_frequency * frequency * (self.K1 + 1) / denominator
            score += self._exact_boost(query, record, query_tokens)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].document.standard_id, item[1].article.sequence))
        return [
            make_hit(
                record,
                rank=index,
                methods=[RetrievalMethod.KEYWORD],
                keyword_score=score,
            )
            for index, (score, record) in enumerate(scored[:top_k], start=1)
        ]

    def _field_tokens(self, record: RetrievalRecord) -> Counter[str]:
        weighted: Counter[str] = Counter()
        for token in self.normalizer.tokenize(record.article.content):
            weighted[token] += 3
        for token in self.normalizer.tokenize(record.article.article_number):
            weighted[token] += 5
        for token in self.normalizer.tokenize(record.article.chapter_title or ""):
            weighted[token] += 2
        for token in self.normalizer.tokenize(
            " ".join(
                value
                for value in (record.document.standard_name, record.document.standard_code)
                if value
            )
        ):
            weighted[token] += 1
        return weighted

    def _exact_boost(
        self, query: str, record: RetrievalRecord, query_tokens: list[str]
    ) -> float:
        normalized_query = self.normalizer.normalize(query)
        normalized_content = self.normalizer.normalize(record.article.content)
        boost = 0.0
        for phrase in self.normalizer.cjk_phrases(normalized_query):
            if phrase in normalized_content:
                boost += 4.0
        article_numbers = self.ARTICLE_NUMBER.findall(normalized_query)
        if record.article.article_number in article_numbers:
            boost += 10.0
        normalized_code = self.normalizer.normalize(record.document.standard_code or "")
        if normalized_code and normalized_code in normalized_query:
            boost += 5.0
        record_tokens = set(self.normalizer.tokenize(record.article.content))
        if query_tokens:
            boost += len(record_tokens.intersection(query_tokens)) / len(query_tokens)
        return boost
