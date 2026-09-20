"""Controlled synthetic Case D exercises frozen D9-P1 through its public seam."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import socket

import pytest

from app.core.config import Settings
from app.services.pdf_service import PDFService
from app.services.review.internal_consistency_service import (
    InternalConsistencyReviewService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    PROJECT_ROOT / "backend/tests/fixtures/internal_consistency/case_d"
)
METADATA_PATH = FIXTURE_ROOT / "controlled-case-d.json"
P1_PATHS = {
    "backend/app/schemas/internal_consistency.py": (
        "e2ceb672b6707dfe11b4b4e8d28236b5e302eaa40844a1c50abe245b9bc3de6c"
    ),
    "backend/app/services/review/internal_consistency_service.py": (
        "cd67f6417b6d1e61d4f963e909d53480b3300454001b5097c1a5a2ba6f1b9235"
    ),
    "backend/tests/test_internal_consistency_d9.py": (
        "205d8ab4a6b28eddd6b9cf05a9e131b8d8984752afb9bf5c45b83d6e438131c4"
    ),
}
REAL_PLAN_IDS = {
    "f42886ef-83ec-454c-86e2-34d5c238ca0f",
    "04039d98-4131-422a-b29a-256bade04a6a",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def asset_path(metadata: dict) -> Path:
    return FIXTURE_ROOT / metadata["asset_filename"]


@pytest.fixture(scope="module")
def extraction(asset_path: Path):
    return PDFService().extract_text(asset_path)


@pytest.fixture(scope="module")
def review(metadata: dict):
    service = InternalConsistencyReviewService(
        settings=Settings(upload_dir=FIXTURE_ROOT)
    )
    return service.review_document(metadata["document_id"])


def test_case_d_asset_exists_with_stable_bytes(
    metadata: dict, asset_path: Path
) -> None:
    assert asset_path.is_file()
    assert asset_path.name == f'{metadata["document_id"]}.pdf'
    assert asset_path.stat().st_size == metadata["asset_size_bytes"] == 2514
    assert _sha256(asset_path) == metadata["asset_sha256"]


def test_case_d_is_explicitly_controlled_synthetic(metadata: dict) -> None:
    assert metadata["case_id"] == "CASE-D"
    assert metadata["classification"] == "CONTROLLED_SYNTHETIC"
    assert metadata["classification_zh"] == "受控合成演示案例"
    assert metadata["purpose"] == "INTERNAL_CONSISTENCY_DEMONSTRATION"
    assert "受控合成" in metadata["presentation_notice"]
    assert "合规结论权威" in metadata["presentation_notice"]


def test_case_d_has_two_page_machine_readable_text_layer(
    metadata: dict, extraction
) -> None:
    assert extraction.page_count == metadata["page_count"] == 2
    assert extraction.char_count > 0
    assert all(page.text.strip() for page in extraction.pages)
    assert all(page.image_count == 0 for page in extraction.pages)
    for page in extraction.pages:
        expected = metadata["expected_source_texts"][str(page.page_number)]
        assert expected in page.text
        assert "CONTROLLED_SYNTHETIC" in page.text
        assert "INTERNAL_CONSISTENCY_DEMONSTRATION" in page.text


def test_case_d_requires_neither_ocr_nor_network(metadata: dict) -> None:
    assert metadata["requires_ocr"] is False
    assert metadata["requires_network"] is False
    production_source = inspect.getsource(InternalConsistencyReviewService).lower()
    assert "services.ocr" not in production_source
    assert "requests." not in production_source
    assert "http://" not in production_source
    assert "https://" not in production_source


def test_case_d_uses_normal_server_side_review_document_path(
    metadata: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    original = PDFService.extract_text

    def counted_extract(self, pdf_path: Path, *, require_text: bool = True):
        calls.append(pdf_path.resolve())
        return original(self, pdf_path, require_text=require_text)

    monkeypatch.setattr(PDFService, "extract_text", counted_extract)
    service = InternalConsistencyReviewService(
        settings=Settings(upload_dir=FIXTURE_ROOT)
    )
    result = service.review_document(metadata["document_id"])
    assert result.document_id == metadata["document_id"]
    assert calls == [
        (FIXTURE_ROOT / metadata["asset_filename"]).resolve()
    ]


def test_case_d_review_result_matches_controlled_expectation(
    metadata: dict, review
) -> None:
    expected = metadata["expected_review"]
    assert review.document_sha256 == metadata["asset_sha256"]
    assert review.fact_count == expected["fact_count"] == 2
    assert review.candidate_count == expected["candidate_count"] == 1
    assert review.unresolved_count == expected["unresolved_count"] == 0
    candidate = review.candidates[0]
    assert candidate.status.value == expected["status"]
    assert candidate.relation.value == expected["relation"]
    assert candidate.reason.value == expected["reason"]
    assert candidate.review_status.value == expected["review_status"]


def test_case_d_frozen_profiles_and_value_groups_match(
    metadata: dict, review
) -> None:
    expected = metadata["expected_review"]
    candidate = review.candidates[0]
    assert candidate.object_profile_name.value == expected["object_profile"]
    assert candidate.object_profile_id == expected["object_profile_id"]
    assert candidate.parameter_profile_name.value == expected["parameter_profile"]
    assert candidate.parameter_profile_id == expected["parameter_profile_id"]
    assert candidate.normalized_unit.value == expected["normalized_unit"] == "mm"
    assert candidate.assertion_class.value == expected["assertion_class"]
    assert len(candidate.value_groups) == 2
    assert [str(group.numeric_value) for group in candidate.value_groups] == [
        "20",
        "30",
    ]
    facts = [fact for group in candidate.value_groups for fact in group.facts]
    assert len(facts) == 2
    assert len({fact.object_profile_id for fact in facts}) == 1
    assert len({fact.parameter_profile_id for fact in facts}) == 1
    assert len({fact.assertion_class for fact in facts}) == 1


def test_case_d_source_locator_authority_is_exact(
    metadata: dict, extraction, review
) -> None:
    pages = {page.page_number: page.text for page in extraction.pages}
    facts = [
        fact
        for group in review.candidates[0].value_groups
        for fact in group.facts
    ]
    assert {fact.physical_page for fact in facts} == {1, 2}
    for fact in facts:
        page_text = pages[fact.physical_page]
        expected_source = metadata["expected_source_texts"][str(fact.physical_page)]
        assert fact.source_text == expected_source
        assert page_text[fact.source_start : fact.source_end] == fact.source_text
        assert hashlib.sha256(fact.source_text.encode("utf-8")).hexdigest() == (
            fact.source_text_sha256
        )
        assert hashlib.sha256(page_text.encode("utf-8")).hexdigest() == (
            fact.page_text_sha256
        )
        assert fact.document_sha256 == metadata["asset_sha256"]
        assert fact.fact_id.startswith("consistencyfact_")


def test_case_d_candidate_and_review_identity_are_deterministic(
    metadata: dict, review
) -> None:
    repeated = InternalConsistencyReviewService(
        settings=Settings(upload_dir=FIXTURE_ROOT)
    ).review_document(metadata["document_id"])
    expected = metadata["expected_review"]
    assert review.candidates[0].candidate_id == expected["candidate_id"]
    assert repeated.candidates[0].candidate_id == expected["candidate_id"]
    assert review.review_id == repeated.review_id == expected["review_id"]
    assert review == repeated


def test_case_d_candidate_cannot_escalate_to_compliance_authority(review) -> None:
    candidate = review.candidates[0]
    payload = candidate.model_dump(mode="json")
    assert candidate.status.value != "NON_COMPLIANT"
    assert {
        "finding",
        "finding_id",
        "compliance",
        "decision",
        "severity",
        "risk",
        "risk_score",
        "unsafe",
    }.isdisjoint(payload)


def test_case_d_is_separate_from_qualified_real_plans(metadata: dict) -> None:
    raw_metadata = METADATA_PATH.read_text(encoding="utf-8")
    assert metadata["document_id"] not in REAL_PLAN_IDS
    assert REAL_PLAN_IDS.isdisjoint(raw_metadata.split())
    assert all(real_id not in raw_metadata for real_id in REAL_PLAN_IDS)


def test_case_d_review_uses_no_network(
    metadata: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("Case D attempted network access")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    result = InternalConsistencyReviewService(
        settings=Settings(upload_dir=FIXTURE_ROOT)
    ).review_document(metadata["document_id"])
    assert result.candidate_count == 1


def test_frozen_d9_p1_files_remain_byte_identical() -> None:
    for relative_path, expected_sha in P1_PATHS.items():
        assert _sha256(PROJECT_ROOT / relative_path) == expected_sha
