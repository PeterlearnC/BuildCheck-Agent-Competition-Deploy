import math

import pytest

from app.services.retrieval.retrieval_manifest_service import RetrievalManifestService
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval.vector_store import LocalMemoryVectorStore, VectorStoreError
from tests.retrieval_helpers import MappingEmbeddingProvider, make_record


def _manifest(count=2):
    records = [make_record("std", f"1.1.{index}", "正文", sequence=index) for index in range(1, count + 1)]
    return RetrievalManifestService().build(records, embedding_provider="fake:1", embedding_dimension=2)


def test_add_embeddings_and_cosine_search() -> None:
    store = LocalMemoryVectorStore()
    store.add_embeddings({"a": [1.0, 0.0], "b": [0.1, 0.9]}, _manifest())
    assert [item.item_id for item in store.search([1.0, 0.0], 2)] == ["a", "b"]


def test_search_respects_top_k() -> None:
    store = LocalMemoryVectorStore()
    store.add_embeddings({"a": [1.0, 0.0], "b": [0.5, 0.5]}, _manifest())
    assert len(store.search([1.0, 0.0], 1)) == 1


def test_search_respects_allowed_ids() -> None:
    store = LocalMemoryVectorStore()
    store.add_embeddings({"a": [1.0, 0.0], "b": [0.5, 0.5]}, _manifest())
    assert [item.item_id for item in store.search([1.0, 0.0], 2, allowed_ids={"b"})] == ["b"]


def test_delete_selected_embedding() -> None:
    store = LocalMemoryVectorStore()
    store.add_embeddings({"a": [1.0, 0.0], "b": [0.5, 0.5]}, _manifest())
    store.delete(["a"])
    assert [item.item_id for item in store.search([1.0, 0.0], 2)] == ["b"]


def test_delete_all_clears_manifest() -> None:
    store = LocalMemoryVectorStore()
    store.add_embeddings({"a": [1.0, 0.0]}, _manifest(1))
    store.delete()
    assert store.manifest() is None
    assert store.search([1.0, 0.0], 1) == []


def test_manifest_is_defensively_copied() -> None:
    store = LocalMemoryVectorStore()
    store.add_embeddings({"a": [1.0, 0.0]}, _manifest(1))
    returned = store.manifest()
    returned.article_count = 99
    assert store.manifest().article_count == 1


def test_store_rejects_wrong_dimension_and_non_finite_values() -> None:
    store = LocalMemoryVectorStore()
    with pytest.raises(VectorStoreError, match="dimension"):
        store.add_embeddings({"a": [1.0]}, _manifest(1))
    with pytest.raises(VectorStoreError, match="non-finite"):
        store.add_embeddings({"a": [math.nan, 0.0]}, _manifest(1))


def test_vector_retriever_reuses_safe_index_cache() -> None:
    class CountingProvider(MappingEmbeddingProvider):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return super().embed_documents(texts)

    provider = CountingProvider([("query", [1.0, 0.0, 0.0]), ("正文", [1.0, 0.0, 0.0])])
    retriever = VectorRetriever(provider)
    records = [make_record("std", "1.1.1", "正文")]
    retriever.retrieve("query", records, 1)
    retriever.retrieve("query", records, 1)
    assert provider.calls == 1
