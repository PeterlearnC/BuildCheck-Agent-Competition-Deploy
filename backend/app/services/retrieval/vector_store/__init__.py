"""Provider-neutral vector index contracts and local implementation."""

from app.services.retrieval.vector_store.base import (
    VectorSearchResult,
    VectorStore,
    VectorStoreError,
)
from app.services.retrieval.vector_store.local_memory import LocalMemoryVectorStore

__all__ = [
    "LocalMemoryVectorStore",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreError",
]
