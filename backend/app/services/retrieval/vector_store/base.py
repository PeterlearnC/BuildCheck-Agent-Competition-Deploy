"""Abstract vector index boundary independent of embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.services.retrieval.retrieval_manifest_service import RetrievalManifest


class VectorStoreError(ValueError):
    pass


@dataclass(frozen=True)
class VectorSearchResult:
    item_id: str
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add_embeddings(
        self,
        embeddings: dict[str, list[float]],
        manifest: RetrievalManifest,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, item_ids: list[str] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def manifest(self) -> RetrievalManifest | None:
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError
