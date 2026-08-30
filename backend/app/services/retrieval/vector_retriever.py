"""Provider-independent in-memory cosine vector retrieval."""

import math

from app.schemas.standards_retrieval import RetrievalMethod, StandardSearchHit
from app.services.retrieval.embedding_provider import EmbeddingProvider, EmbeddingProviderError
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.retrieval_manifest_service import RetrievalManifestService
from app.services.retrieval.vector_store import LocalMemoryVectorStore, VectorStore


class VectorRetriever:
    def __init__(
        self,
        provider: EmbeddingProvider,
        store: VectorStore | None = None,
    ) -> None:
        self.provider = provider
        self.store = store or LocalMemoryVectorStore()
        self.manifests = RetrievalManifestService()

    def retrieve(
        self, query: str, records: list[RetrievalRecord], top_k: int
    ) -> list[StandardSearchHit]:
        if not records or top_k <= 0:
            return []
        self._ensure_index(records)
        query_vector = self.provider.embed_query(query)
        self._validate_vector(query_vector)
        if self._norm(query_vector) == 0:
            return []
        by_id = {record.article.article_id: record for record in records}
        scored = self.store.search(
            query_vector,
            top_k,
            allowed_ids=set(by_id),
        )
        return [
            make_hit(
                by_id[result.item_id],
                rank=index,
                methods=[RetrievalMethod.VECTOR],
                vector_score=result.score,
            )
            for index, result in enumerate(scored, start=1)
        ]

    def _ensure_index(self, records: list[RetrievalRecord]) -> None:
        expected = self.manifests.build(
            records,
            embedding_provider=f"{self.provider.provider_name}:{self.provider.provider_version}",
            embedding_model_id=self.provider.embedding_model_id,
            embedding_dimension=self.provider.embedding_dimension,
        )
        if self.manifests.compatible(expected, self.store.manifest()):
            return
        document_vectors = self.provider.embed_documents(
            [record.embedding_text for record in records]
        )
        if len(document_vectors) != len(records):
            raise EmbeddingProviderError("Embedding provider returned an unexpected document count.")
        embeddings: dict[str, list[float]] = {}
        for record, vector in zip(records, document_vectors, strict=True):
            self._validate_vector(vector)
            embeddings[record.article.article_id] = vector
        self.store.delete()
        self.store.add_embeddings(embeddings, expected)

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.provider.embedding_dimension:
            raise EmbeddingProviderError("Embedding dimension does not match provider metadata.")
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingProviderError("Embedding contains a non-finite value.")

    @staticmethod
    def _norm(vector: list[float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    @classmethod
    def _cosine(cls, left: list[float], right: list[float]) -> float:
        denominator = cls._norm(left) * cls._norm(right)
        if denominator == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / denominator
