from app.services.retrieval.retrieval_evaluation import evaluate_rankings
from app.services.retrieval.standards_index_service import (
    RETRIEVAL_INDEX_VERSION,
    RetrievalIndexManifest,
    StandardsIndexService,
)
from tests.retrieval_helpers import basic_records


def test_same_corpus_has_same_fingerprint() -> None:
    records = basic_records()
    service = StandardsIndexService()
    assert service.corpus_fingerprint(records) == service.corpus_fingerprint(list(reversed(records)))


def test_article_content_change_changes_fingerprint() -> None:
    records = basic_records()
    changed = records[0].article.model_copy(update={"content": "已修改的条文。"})
    changed_records = [records[0].__class__(records[0].document, changed), *records[1:]]
    assert StandardsIndexService().corpus_fingerprint(records) != StandardsIndexService().corpus_fingerprint(changed_records)


def test_article_provenance_change_changes_fingerprint() -> None:
    records = basic_records()
    changed = records[0].article.model_copy(update={"source_page_end": 9})
    changed_records = [records[0].__class__(records[0].document, changed), *records[1:]]
    assert StandardsIndexService().corpus_fingerprint(records) != StandardsIndexService().corpus_fingerprint(changed_records)


def test_fingerprint_is_stable_sha256_hex() -> None:
    fingerprint = StandardsIndexService().corpus_fingerprint(basic_records())
    assert len(fingerprint) == 64
    assert all(character in "0123456789abcdef" for character in fingerprint)


def test_index_version_change_invalidates_manifest() -> None:
    current = RetrievalIndexManifest("abc", RETRIEVAL_INDEX_VERSION, "fake", "1", 3)
    stale = RetrievalIndexManifest("abc", "old", "fake", "1", 3)
    assert StandardsIndexService.compatible(current, current)
    assert not StandardsIndexService.compatible(current, stale)


def test_provider_version_change_invalidates_manifest() -> None:
    current = RetrievalIndexManifest("abc", RETRIEVAL_INDEX_VERSION, "fake", "2", 3)
    stale = RetrievalIndexManifest("abc", RETRIEVAL_INDEX_VERSION, "fake", "1", 3)
    assert not StandardsIndexService.compatible(current, stale)


def test_retrieval_metrics_calculation() -> None:
    metrics = evaluate_rankings([["a", "b"], ["x", "c"]], [{"a"}, {"c"}])
    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_3 == 1.0
    assert metrics.mrr == 0.75
