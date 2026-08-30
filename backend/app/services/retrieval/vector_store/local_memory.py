"""Deterministic process-local vector index for tests and offline operation."""

import math

from app.services.retrieval.retrieval_manifest_service import RetrievalManifest
from app.services.retrieval.vector_store.base import (
    VectorSearchResult,
    VectorStore,
    VectorStoreError,
)


class LocalMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._embeddings: dict[str, tuple[float, ...]] = {}
        self._manifest: RetrievalManifest | None = None

    def add_embeddings(
        self,
        embeddings: dict[str, list[float]],
        manifest: RetrievalManifest,
    ) -> None:
        for item_id, vector in embeddings.items():
            self._validate(vector, manifest.embedding_dimension)
            self._embeddings[item_id] = tuple(vector)
        self._manifest = manifest.model_copy(deep=True)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        if top_k <= 0 or self._manifest is None:
            return []
        self._validate(query_embedding, self._manifest.embedding_dimension)
        scored: list[VectorSearchResult] = []
        for item_id, vector in self._embeddings.items():
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            score = self._cosine(query_embedding, vector)
            if score > 0:
                scored.append(VectorSearchResult(item_id=item_id, score=score))
        return sorted(scored, key=lambda item: (-item.score, item.item_id))[:top_k]

    def delete(self, item_ids: list[str] | None = None) -> None:
        if item_ids is None:
            self._embeddings.clear()
            self._manifest = None
            return
        for item_id in item_ids:
            self._embeddings.pop(item_id, None)
        if not self._embeddings:
            self._manifest = None

    def manifest(self) -> RetrievalManifest | None:
        return self._manifest.model_copy(deep=True) if self._manifest else None

    def persist(self) -> None:
        raise NotImplementedError("LocalMemoryVectorStore does not support persistence.")

    def load(self) -> None:
        raise NotImplementedError("LocalMemoryVectorStore does not support persistence.")

    @staticmethod
    def _validate(vector: list[float], dimension: int) -> None:
        if len(vector) != dimension:
            raise VectorStoreError("Vector dimension does not match the manifest.")
        if any(not math.isfinite(value) for value in vector):
            raise VectorStoreError("Vector contains a non-finite value.")

    @staticmethod
    def _cosine(left: list[float], right: tuple[float, ...]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
