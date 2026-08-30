import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.schemas.analysis import DocumentAnalysis
from app.services.completeness_review_cache_service import (
    CompletenessReviewCacheService,
)
from app.services.document_analysis_service import DocumentAnalysisService
from app.services.llm_service import LLMCallError, get_llm_service


client = TestClient(app)


class _NoLLM:
    def analyze_document(self, _pages):  # pragma: no cover - must not run
        raise AssertionError("Review with an analysis cache must not call the LLM.")


@pytest.fixture(autouse=True)
def temporary_review_directories(tmp_path: Path):
    settings = get_settings()
    old_upload = settings.upload_dir
    old_analysis = settings.analysis_dir
    old_review = settings.completeness_review_dir
    settings.upload_dir = tmp_path / "uploads"
    settings.analysis_dir = tmp_path / "analysis"
    settings.completeness_review_dir = tmp_path / "reviews" / "completeness"
    app.dependency_overrides[get_llm_service] = lambda: _NoLLM()
    yield settings
    app.dependency_overrides.clear()
    settings.upload_dir = old_upload
    settings.analysis_dir = old_analysis
    settings.completeness_review_dir = old_review


def _analysis(document_type: str = "通用施工方案") -> DocumentAnalysis:
    return DocumentAnalysis.model_validate(
        {
            "document_type": {
                "value": document_type,
                "source_page": 1,
                "source_text": document_type,
            },
            "project": {
                "project_name": {
                    "value": "测试项目",
                    "source_page": 2,
                    "source_text": "工程名称：测试项目",
                }
            },
            "chapters": [
                {"title": "工程概况", "start_page": 2},
                {"title": "质量保证措施", "start_page": 10},
            ],
            "section_presence": {"quality": True},
        }
    )


def _seed_document(settings, analysis: DocumentAnalysis | None = None) -> str:
    document_id = str(uuid4())
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    (settings.upload_dir / f"{document_id}.pdf").write_bytes(b"%PDF-placeholder")
    if analysis is not None:
        DocumentAnalysisService(_NoLLM(), settings=settings).save_cache(
            document_id, analysis
        )
    return document_id


def _post(document_id: str, force: bool = False):
    suffix = "?force=true" if force else ""
    return client.post(
        f"/api/v1/documents/{document_id}/review/completeness{suffix}"
    )


def test_post_completeness_review_returns_fourteen_consistent_checks(
    temporary_review_directories,
) -> None:
    document_id = _seed_document(temporary_review_directories, _analysis())

    response = _post(document_id)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["cached"] is False
    assert body["review"]["review_profile"] == "general_construction_plan"
    assert len(body["review"]["checks"]) == 14
    summary = body["review"]["summary"]
    assert summary["passed"] + summary["partial"] + summary["missing"] + summary[
        "not_applicable"
    ] == summary["total_checks"] == 14
    assert summary["applicable_checks"] == (
        summary["passed"] + summary["partial"] + summary["missing"]
    )
    expected_score = int(
        (
            (summary["passed"] + summary["partial"] * 0.5)
            / summary["applicable_checks"]
            * 100
        )
        + 0.5
    )
    assert summary["completeness_score"] == expected_score


def test_review_cache_transitions_false_to_true(temporary_review_directories) -> None:
    settings = temporary_review_directories
    document_id = _seed_document(settings, _analysis())

    first = _post(document_id)
    second = _post(document_id)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert first.json()["review"] == second.json()["review"]
    assert (settings.completeness_review_dir / f"{document_id}.json").is_file()


def test_force_bypasses_review_cache_only(
    temporary_review_directories, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = _seed_document(temporary_review_directories, _analysis())
    assert _post(document_id).status_code == 200

    def forbidden_analysis(*_args, **_kwargs):
        raise AssertionError("force review must not force document analysis")

    monkeypatch.setattr(DocumentAnalysisService, "analyze", forbidden_analysis)
    forced = _post(document_id, force=True)

    assert forced.status_code == 200
    assert forced.json()["cached"] is False


def test_analysis_fingerprint_change_invalidates_review_cache(
    temporary_review_directories,
) -> None:
    settings = temporary_review_directories
    document_id = _seed_document(settings, _analysis())
    assert _post(document_id).json()["cached"] is False
    assert _post(document_id).json()["cached"] is True

    changed = _analysis("构件专项施工方案")
    DocumentAnalysisService(_NoLLM(), settings=settings).save_cache(document_id, changed)
    refreshed = _post(document_id)

    assert refreshed.status_code == 200
    assert refreshed.json()["cached"] is False
    assert refreshed.json()["review"]["review_profile"] == "special_construction_plan"


def test_corrupt_review_cache_is_safely_recomputed(
    temporary_review_directories,
) -> None:
    settings = temporary_review_directories
    document_id = _seed_document(settings, _analysis())
    assert _post(document_id).status_code == 200
    cache_path = settings.completeness_review_dir / f"{document_id}.json"
    cache_path.write_text("{broken", encoding="utf-8")

    response = _post(document_id)

    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert json.loads(cache_path.read_text(encoding="utf-8"))["review_version"] == "v0.3"


def test_missing_review_cache_is_normally_generated(
    temporary_review_directories,
) -> None:
    settings = temporary_review_directories
    document_id = _seed_document(settings, _analysis())
    assert not (settings.completeness_review_dir / f"{document_id}.json").exists()

    response = _post(document_id)

    assert response.status_code == 200
    assert response.json()["cached"] is False


def test_unknown_document_uses_existing_404_shape() -> None:
    response = _post("00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}


def test_analysis_failure_does_not_create_review(
    temporary_review_directories, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = temporary_review_directories
    document_id = _seed_document(settings)

    def failed_analysis(*_args, **_kwargs):
        raise LLMCallError("synthetic analysis failure")

    monkeypatch.setattr(DocumentAnalysisService, "analyze", failed_analysis)
    response = _post(document_id)

    assert response.status_code == 502
    assert response.json()["detail"] == "synthetic analysis failure"
    assert not (settings.completeness_review_dir / f"{document_id}.json").exists()


def test_review_generates_analysis_only_when_cache_is_absent(
    temporary_review_directories, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = temporary_review_directories
    document_id = _seed_document(settings)
    calls = []

    def generated_analysis(_service, _pdf_path):
        calls.append("analyze")
        return _analysis()

    monkeypatch.setattr(DocumentAnalysisService, "analyze", generated_analysis)
    first = _post(document_id, force=True)
    second = _post(document_id, force=True)

    assert first.status_code == second.status_code == 200
    assert calls == ["analyze"]
    assert (settings.analysis_dir / f"{document_id}.json").is_file()


def test_fingerprint_is_stable_for_equivalent_model_content() -> None:
    first = DocumentAnalysis.model_validate(
        {"document_type": "施工方案", "section_presence": {"quality": True}}
    )
    second = DocumentAnalysis.model_validate(
        {"section_presence": {"quality": True}, "document_type": "施工方案"}
    )
    assert CompletenessReviewCacheService.analysis_fingerprint(
        first
    ) == CompletenessReviewCacheService.analysis_fingerprint(second)
