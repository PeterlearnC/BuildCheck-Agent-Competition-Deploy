from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def temporary_standards_directory(tmp_path: Path):
    settings = get_settings()
    previous = settings.standards_dir
    settings.standards_dir = tmp_path / "standards"
    yield
    settings.standards_dir = previous


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Synthetic Construction Standard\nStandard Code JGJ 999-2024\n6 Construction\n6.1 General\n6.1.1 Components shall be connected.\n6.1.2 Installation shall be checked.",
        fontsize=10,
    )
    payload = document.tobytes()
    document.close()
    return payload


def _upload() -> str:
    response = client.post(
        "/api/v1/standards/upload",
        files={"file": ("synthetic.pdf", _pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()["standard_id"]


def test_upload_non_pdf_is_rejected() -> None:
    response = client.post(
        "/api/v1/standards/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 415


def test_unknown_standard_returns_404() -> None:
    assert client.get("/api/v1/standards/missing").status_code == 404
    assert client.post("/api/v1/standards/missing/parse").status_code == 404


def test_upload_parse_get_and_list() -> None:
    standard_id = _upload()
    parsed = client.post(f"/api/v1/standards/{standard_id}/parse")
    assert parsed.status_code == 200, parsed.text
    assert parsed.json()["summary"] == {"chapters": 2, "articles": 2}
    assert parsed.json()["cached"] is False
    detail = client.get(f"/api/v1/standards/{standard_id}").json()
    assert detail["article_count"] == 2
    listed = client.get("/api/v1/standards").json()["standards"]
    assert [item["standard"]["standard_id"] for item in listed] == [standard_id]


def test_parse_is_idempotent_and_force_bypasses_cache() -> None:
    standard_id = _upload()
    first = client.post(f"/api/v1/standards/{standard_id}/parse")
    second = client.post(f"/api/v1/standards/{standard_id}/parse")
    article_ids_before = [
        item["article_id"]
        for item in client.get(f"/api/v1/standards/{standard_id}/articles").json()["articles"]
    ]
    forced = client.post(f"/api/v1/standards/{standard_id}/parse?force=true")
    article_ids_after = [
        item["article_id"]
        for item in client.get(f"/api/v1/standards/{standard_id}/articles").json()["articles"]
    ]
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert forced.json()["cached"] is False
    assert first.json()["summary"] == forced.json()["summary"]
    assert article_ids_before == article_ids_after


def test_articles_pagination_and_chapter_filter() -> None:
    standard_id = _upload()
    client.post(f"/api/v1/standards/{standard_id}/parse")
    page = client.get(f"/api/v1/standards/{standard_id}/articles?limit=1&offset=1")
    assert page.status_code == 200
    assert page.json()["total"] == 2
    assert [item["article_number"] for item in page.json()["articles"]] == ["6.1.2"]
    filtered = client.get(
        f"/api/v1/standards/{standard_id}/articles?chapter_number=6.1"
    ).json()
    assert filtered["total"] == 2
