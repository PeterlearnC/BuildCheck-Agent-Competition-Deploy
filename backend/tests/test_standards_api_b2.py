import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import basic_records


client = TestClient(app)


@pytest.fixture(autouse=True)
def repository(tmp_path: Path):
    settings = get_settings()
    previous = settings.standards_dir
    settings.standards_dir = tmp_path / "standards"
    repo = StandardRepository(settings)
    records = basic_records()
    for standard_id in {record.document.standard_id for record in records}:
        selected = [record for record in records if record.document.standard_id == standard_id]
        repo.save_document(selected[0].document)
        repo.save_articles(standard_id, [record.article for record in selected])
    yield settings
    settings.standards_dir = previous


def test_registry_api_returns_empty_registry() -> None:
    response = client.get("/api/v1/standards/registry")
    assert response.status_code == 200
    assert response.json()["entries"] == []


def test_registry_api_returns_offline_entries(repository) -> None:
    (repository.standards_dir / "registry.json").write_text(
        json.dumps([{"standard_code": "JGJ 999-2024", "standard_name": "架体工程测试规范", "status": "ACTIVE"}], ensure_ascii=False),
        encoding="utf-8",
    )
    assert client.get("/api/v1/standards/registry").json()["entries"][0]["status"] == "ACTIVE"


def test_search_api_returns_evidence_and_manifest_version() -> None:
    body = client.post("/api/v1/standards/search", json={"query": "连墙件"}).json()
    assert body["manifest_version"]
    assert body["scope_warning"] == (
        "Search without explicit standard scope may include multiple standards"
    )
    assert body["hits"][0]["evidence"]["source_checksum"]
    assert body["hits"][0]["evidence"]["source_text"]


def test_registry_filter_allows_only_active_standard(repository) -> None:
    (repository.standards_dir / "registry.json").write_text(
        json.dumps([
            {"standard_code": "JGJ 999-2024", "standard_name": "架体", "status": "ACTIVE"},
            {"standard_code": "GB 888-2023", "standard_name": "管道", "status": "SUPERSEDED"},
        ]),
        encoding="utf-8",
    )
    body = client.post(
        "/api/v1/standards/search",
        json={"query": "6.1.2", "registry_filter": {}},
    ).json()
    assert body["hits"]
    assert {hit["standard_id"] for hit in body["hits"]} == {"std-a"}


def test_search_without_registry_filter_remains_backward_compatible() -> None:
    body = client.post("/api/v1/standards/search", json={"query": "6.1.2"}).json()
    assert {hit["standard_id"] for hit in body["hits"]} == {"std-a", "std-b"}
