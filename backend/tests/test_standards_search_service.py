from pathlib import Path

import pytest

from app.core.config import Settings
from app.schemas.standards_retrieval import StandardSearchRequest
from app.services.retrieval.standards_search_service import StandardsSearchService, UnknownStandardsError
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import basic_records


def _service(tmp_path: Path) -> StandardsSearchService:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    records = basic_records()
    documents = {record.document.standard_id: record.document for record in records}
    for standard_id, document in documents.items():
        repository.save_document(document)
        repository.save_articles(
            standard_id, [record.article for record in records if record.document.standard_id == standard_id]
        )
    return StandardsSearchService(repository)


def test_standard_filter_excludes_other_corpus(tmp_path: Path) -> None:
    response = _service(tmp_path).search(StandardSearchRequest(query="6.1.2", standard_ids=["std-b"]))
    assert response.hits
    assert {hit.standard_id for hit in response.hits} == {"std-b"}


def test_unknown_standard_filter_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(UnknownStandardsError):
        _service(tmp_path).search(StandardSearchRequest(query="连墙件", standard_ids=["missing"]))


def test_chapter_filter_is_applied(tmp_path: Path) -> None:
    response = _service(tmp_path).search(StandardSearchRequest(query="连墙件", chapter_numbers=["9.9"]))
    assert response.total_candidates == 0
    assert response.hits == []


def test_empty_repository_returns_empty_response(tmp_path: Path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "empty"))
    response = StandardsSearchService(repository).search(StandardSearchRequest(query="连墙件"))
    assert response.returned == 0
    assert response.hits == []
    assert len(response.corpus_fingerprint) == 64


def test_search_service_is_deterministic(tmp_path: Path) -> None:
    service = _service(tmp_path)
    request = StandardSearchRequest(query="连墙件设置", top_k=3)
    first = service.search(request)
    second = service.search(request)
    assert [(hit.article_id, hit.rank, hit.rerank_score) for hit in first.hits] == [
        (hit.article_id, hit.rank, hit.rerank_score) for hit in second.hits
    ]
