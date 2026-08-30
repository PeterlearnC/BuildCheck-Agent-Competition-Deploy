import math

import pytest

from app.services.retrieval.embedding_provider import EmbeddingProviderError
from app.services.retrieval.vector_retriever import VectorRetriever
from tests.retrieval_helpers import MappingEmbeddingProvider, make_record


def _semantic_provider() -> MappingEmbeddingProvider:
    return MappingEmbeddingProvider(
        [
            ("防止架体整体失稳", [1.0, 0.0, 0.0]),
            ("整体稳定", [1.0, 0.0, 0.0]),
            ("材料分类", [0.0, 1.0, 0.0]),
        ]
    )


def test_fake_semantic_mapping_retrieves_expected_article() -> None:
    records = [
        make_record("std", "1.1.1", "支撑体系应保持整体稳定。"),
        make_record("std", "1.1.2", "施工现场材料分类堆放。", sequence=2),
    ]
    hits = VectorRetriever(_semantic_provider()).retrieve("防止架体整体失稳", records, 2)
    assert hits[0].article_number == "1.1.1"


def test_cosine_similarity_orders_vectors() -> None:
    provider = MappingEmbeddingProvider(
        [("query", [1.0, 0.0, 0.0]), ("close", [0.9, 0.1, 0.0]), ("far", [0.2, 0.8, 0.0])]
    )
    records = [make_record("std", "1.1.1", "close"), make_record("std", "1.1.2", "far", sequence=2)]
    hits = VectorRetriever(provider).retrieve("query", records, 2)
    assert [hit.article_number for hit in hits] == ["1.1.1", "1.1.2"]


def test_zero_query_vector_returns_empty() -> None:
    provider = MappingEmbeddingProvider([("document", [1.0, 0.0, 0.0])])
    assert VectorRetriever(provider).retrieve("unknown", [make_record("std", "1.1.1", "document")], 1) == []


def test_dimension_mismatch_raises_clear_error() -> None:
    provider = MappingEmbeddingProvider([("query", [1.0, 0.0]), ("document", [1.0, 0.0, 0.0])], dimension=2)
    with pytest.raises(EmbeddingProviderError, match="dimension"):
        VectorRetriever(provider).retrieve("query", [make_record("std", "1.1.1", "document")], 1)


def test_non_finite_vector_is_rejected() -> None:
    provider = MappingEmbeddingProvider([("query", [1.0, 0.0, 0.0]), ("document", [math.nan, 0.0, 0.0])])
    with pytest.raises(EmbeddingProviderError, match="non-finite"):
        VectorRetriever(provider).retrieve("query", [make_record("std", "1.1.1", "document")], 1)


def test_empty_vector_corpus_returns_empty_without_provider_call() -> None:
    assert VectorRetriever(_semantic_provider()).retrieve("query", [], 5) == []
