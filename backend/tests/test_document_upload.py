from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def use_temporary_upload_directory(tmp_path: Path) -> None:
    settings = get_settings()
    original_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path
    yield
    settings.upload_dir = original_upload_dir


def make_pdf(text: str | None = None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_non_pdf_upload_is_rejected() -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 415
    assert "Only PDF files" in response.json()["detail"]


def test_empty_pdf_upload_is_rejected() -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_invalid_pdf_upload_is_handled() -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("broken.pdf", b"this is not valid PDF data", "application/pdf")},
    )

    assert response.status_code == 400
    assert "damaged or cannot be opened" in response.json()["detail"]


def test_pdf_over_size_limit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_size", 8)

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.pdf", b"more than eight bytes", "application/pdf")},
    )

    assert response.status_code == 413
    assert "30 MB upload limit" in response.json()["detail"]


def test_pdf_without_text_layer_returns_clear_message() -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("scan.pdf", make_pdf(), "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "PDF contains little or no extractable text. "
        "OCR support is not enabled in V0.1."
    )


def test_valid_pdf_is_saved_and_parsed(tmp_path: Path) -> None:
    source_text = "Tunnel construction plan with extractable text."

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "../tunnel_construction_plan.pdf",
                make_pdf(source_text),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["filename"] == "tunnel_construction_plan.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["page_count"] == 1
    assert body["char_count"] >= len(source_text)
    assert source_text in body["text_preview"]
    assert body["pages"] == [{"page_number": 1, "text": f"{source_text}\n"}]
    assert (tmp_path / f"{body['document_id']}.pdf").is_file()
