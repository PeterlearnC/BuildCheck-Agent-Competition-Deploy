from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import basic_records


client = TestClient(app)


@pytest.fixture(autouse=True)
def retrieval_repository(tmp_path: Path):
    settings = get_settings()
    previous = settings.standards_dir
    settings.standards_dir = tmp_path / "standards"
    repository = StandardRepository(settings)
    records = basic_records()
    for standard_id in {record.document.standard_id for record in records}:
        selected = [record for record in records if record.document.standard_id == standard_id]
        repository.save_document(selected[0].document)
        repository.save_articles(standard_id, [record.article for record in selected])
    yield
    settings.standards_dir = previous


def test_search_api_returns_200_and_ranked_hits() -> None:
    response = client.post("/api/v1/standards/search", json={"query": "连墙件设置", "top_k": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hits"][0]["standard_id"] == "std-a"
    assert body["hits"][0]["article_number"] == "6.1.2"


@pytest.mark.parametrize("top_k", [0, 21, 10000])
def test_search_api_validates_top_k(top_k: int) -> None:
    assert client.post("/api/v1/standards/search", json={"query": "连墙件", "top_k": top_k}).status_code == 422


def test_search_api_rejects_empty_query() -> None:
    assert client.post("/api/v1/standards/search", json={"query": " "}).status_code == 422


def test_search_api_preserves_provenance_and_text() -> None:
    hit = client.post("/api/v1/standards/search", json={"query": "连墙件设置", "top_k": 1}).json()["hits"][0]
    assert hit["content"] == "连墙件应按规定设置。"
    assert hit["source_page_start"] == 3
    assert hit["source_page_end"] == 3
    assert hit["article_number"] == "6.1.2"


def test_search_api_unknown_standard_filter_returns_404() -> None:
    response = client.post(
        "/api/v1/standards/search",
        json={"query": "连墙件", "standard_ids": ["missing"]},
    )
    assert response.status_code == 404


def test_search_api_has_no_hallucinated_article_numbers() -> None:
    body = client.post("/api/v1/standards/search", json={"query": "设置要求", "top_k": 20}).json()
    repository_numbers = {record.article.article_number for record in basic_records()}
    assert {hit["article_number"] for hit in body["hits"]}.issubset(repository_numbers)
