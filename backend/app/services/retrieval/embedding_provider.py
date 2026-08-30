"""Provider-neutral interface for dense embeddings."""

from abc import ABC, abstractmethod


class EmbeddingProviderError(ValueError):
    pass


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_version(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_model_id(self) -> str:
        """Stable model identifier; dimensions alone do not identify a vector space."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
