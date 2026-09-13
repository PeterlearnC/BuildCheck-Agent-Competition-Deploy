"""Qualification tests for the version-controlled D.2 production registry."""

import hashlib
import json
from pathlib import Path

import pytest

from app.schemas.standards import (
    StandardIdentityStatus,
    StandardParseStatus,
    StandardRegistryEntry,
    StandardRegistryStatus,
)
from app.services.pdf_service import PDFPageText, PDFService
from app.services.review.candidate_discovery_service import CandidateDiscoveryService
from app.services.review.standard_routing_service import StandardRoutingService
from app.services.standards.standard_identity_service import StandardIdentityService
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository


QUALIFIED_STANDARD_ID = "55023b3b-1b00-4000-8000-000000000001"
QUALIFIED_CANONICAL_CODE = "GB55023-2022"
QUALIFIED_SOURCE_SHA256 = (
    "58414579d5d6b70f24c0380d632ce659985043b3bb1121dbdbd026452942d567"
)
QUALIFIED_PLAN_SHA256 = (
    "5f3ebde3bd2c0a6d6dccb516c3029fd336558e198ebeccad07d29ca0d03fd8dc"
)

D1_FROZEN_SHA256 = {
    "backend/app/schemas/review_candidate.py": "4a180082f7c2b119728185782fba93a92fc7bf54f243c3fb3db9839b0642d009",
    "backend/app/services/review/candidate_discovery_service.py": "313a031bd6b2ae508cb5adb8b53298ae307017f4ab76b329b1fbf6618d346e0c",
    "backend/tests/test_review_candidate_discovery_d1.py": "0d1c39ef845c91b9c456733a07c4f1c55d2d8be90535eedfa1347968d297c2ae",
}
D2_PRODUCTION_P2_F3_AMENDMENT_SHA256 = {
    "backend/app/schemas/standard_route.py": "163760aebf540d4409c0787ccdcdf305df577ecb661350c27b869435ecf81b14",
    "backend/app/services/review/standard_routing_service.py": "2a65b388dba750128655185ae002bfb1838f12d90389fc509b768d659cef97ce",
}
D2_TEST_CONTRACT_TRANSITION_SHA256 = {
    "backend/tests/test_standard_routing_d2.py": "a5446f8bc9065be39d3758b074f6c7b8258803b443ad990aae597e4305f9e897",
}


class MemoryRepository:
    def __init__(self, documents):
        self.documents = list(documents)

    def list_documents(self):
        return list(self.documents)


def discover(source_text: str):
    return CandidateDiscoveryService().discover(
        document_id="d2-p1-production-registry-probe",
        document_sha256="d" * 64,
        pages=[PDFPageText(page_number=1, text=source_text)],
    )


def route(source_text: str):
    candidates = discover(source_text)
    assert len(candidates) == 1
    return StandardRoutingService().route(candidates[0])


def test_production_registry_is_exact_version_controlled_authority() -> None:
    service = StandardRegistryService()
    assert service.path.is_file()
    payload = json.loads(service.path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload == [
        {
            "standard_code": QUALIFIED_CANONICAL_CODE,
            "standard_name": "施工脚手架通用规范",
            "status": "ACTIVE",
        }
    ]
    entries = service.list_entries()
    assert len(entries) == 1
    assert entries[0].status == StandardRegistryStatus.ACTIVE
    assert StandardIdentityService().canonicalize(entries[0].standard_code) == (
        QUALIFIED_CANONICAL_CODE
    )


def test_registry_entry_resolves_only_installed_official_authority() -> None:
    repository = StandardRepository()
    documents = repository.list_documents()
    assert len(documents) == 1
    document = documents[0]
    assert document.standard_id == QUALIFIED_STANDARD_ID
    assert document.canonical_standard_code == QUALIFIED_CANONICAL_CODE
    assert document.identity_status == StandardIdentityStatus.CONFIRMED
    assert document.parse_status == StandardParseStatus.PARSED
    assert document.source_checksum == QUALIFIED_SOURCE_SHA256
    assert document.official_source_binding is not None
    assert document.official_source_binding.confirmed is True
    assert document.official_source_binding.source_checksum == document.source_checksum
    qualified, _ = StandardRoutingService()._qualified_scopes()
    assert len(qualified) == 1
    assert qualified[0].standard_id == QUALIFIED_STANDARD_ID
    assert qualified[0].canonical_standard_code == QUALIFIED_CANONICAL_CODE


@pytest.mark.parametrize(
    "source_text",
    [
        "可调托撑螺杆伸出钢管顶部不得大于200mm。",
        "螺杆外径与立杆钢管内径的间隙不大于3mm。",
    ],
)
def test_real_production_registry_routes_strong_scaffold_candidates(source_text) -> None:
    result = route(source_text)
    assert result.route_status.value == "ROUTED"
    assert result.selected_scope is not None
    assert result.selected_scope.standard_ids == [QUALIFIED_STANDARD_ID]
    assert result.scope_candidates[0].canonical_standard_code == QUALIFIED_CANONICAL_CODE


@pytest.mark.parametrize(
    "source_text",
    [
        "施工过程中应加强管理。",
        "材料应符合要求。",
        "应做好安全工作。",
    ],
)
def test_real_production_registry_does_not_route_generic_text(source_text) -> None:
    result = route(source_text)
    assert result.route_status.value == "NO_STANDARD_SCOPE"
    assert result.selected_scope is None


def test_registry_contains_no_article_or_decision_authority() -> None:
    payload = json.loads(StandardRegistryService().path.read_text(encoding="utf-8"))
    assert set(payload[0]) == {"standard_code", "standard_name", "status"}
    serialized = json.dumps(payload, ensure_ascii=False).casefold()
    prohibited = (
        "article_id",
        "article_number",
        "normative",
        "threshold",
        "operator",
        "requirement_id",
        "planfact",
        "decision",
        "finding",
        "review_action",
        "whole_document",
        "expected_result",
        "ocr",
        "llm",
    )
    assert all(item not in serialized for item in prohibited)


def test_unknown_extra_field_cannot_inject_uninstalled_authority() -> None:
    entry = StandardRegistryEntry.model_validate(
        {
            "standard_code": "UNINSTALLED-2026",
            "standard_name": "Uninstalled scope",
            "status": "ACTIVE",
            "standard_id": QUALIFIED_STANDARD_ID,
        }
    )
    assert not hasattr(entry, "standard_id")
    service = StandardRoutingService(
        registry=StandardRegistryService(entries=[entry])
    )
    assert service.route(discover("可调托撑不得大于200mm。")[0]).route_status.value == (
        "NO_STANDARD_SCOPE"
    )


def test_identical_duplicate_registry_entries_do_not_duplicate_scope_authority() -> None:
    entry = StandardRegistryService().list_entries()[0]
    service = StandardRoutingService(
        registry=StandardRegistryService(entries=[entry, entry.model_copy()])
    )
    result = service.route(discover("可调托撑不得大于200mm。")[0])
    assert result.route_status.value == "ROUTED"
    assert [item.standard_id for item in result.scope_candidates] == [
        QUALIFIED_STANDARD_ID
    ]


def test_conflicting_duplicate_name_cannot_replace_installed_authority() -> None:
    entry = StandardRegistryService().list_entries()[0]
    conflicting = entry.model_copy(update={"standard_name": "Untrusted display name"})
    service = StandardRoutingService(
        registry=StandardRegistryService(entries=[entry, conflicting])
    )
    result = service.route(discover("可调托撑不得大于200mm。")[0])
    assert result.route_status.value == "ROUTED"
    assert len(result.scope_candidates) == 1
    assert result.scope_candidates[0].standard_name == "施工脚手架通用规范"


@pytest.mark.parametrize("code", ["UNINSTALLED-2026", "GB55034-2022"])
def test_unknown_or_canonical_mismatch_registry_cannot_route(code) -> None:
    entry = StandardRegistryEntry(
        standard_code=code,
        standard_name="Unqualified scope",
        status=StandardRegistryStatus.ACTIVE,
    )
    service = StandardRoutingService(
        registry=StandardRegistryService(entries=[entry])
    )
    result = service.route(discover("可调托撑不得大于200mm。")[0])
    assert result.route_status.value == "NO_STANDARD_SCOPE"


def test_real_plan_strong_candidates_route_without_registry_injection() -> None:
    root = Path(__file__).resolve().parents[2]
    plan_path = root / "backend/data/uploads/f42886ef-83ec-454c-86e2-34d5c238ca0f.pdf"
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == QUALIFIED_PLAN_SHA256
    pages = PDFService().extract_text(plan_path).pages
    service = StandardRoutingService()
    page_17 = CandidateDiscoveryService().discover(
        document_id="f42886ef-83ec-454c-86e2-34d5c238ca0f",
        document_sha256=QUALIFIED_PLAN_SHA256,
        pages=[pages[16]],
    )
    page_46 = CandidateDiscoveryService().discover(
        document_id="f42886ef-83ec-454c-86e2-34d5c238ca0f",
        document_sha256=QUALIFIED_PLAN_SHA256,
        pages=[pages[45]],
    )
    support = next(item for item in page_17 if "可调托撑螺杆外径不得小于36mm" in item.source_text)
    interval = next(item for item in page_46 if "立杆钢管内径的间隙不大于3mm" in item.source_text)
    assert service.route(support).route_status.value == "ROUTED"
    assert service.route(interval).route_status.value == "ROUTED"


def test_d1_and_d2_authority_classes_remain_explicit() -> None:
    root = Path(__file__).resolve().parents[2]
    authority_classes = {
        "D1_FROZEN": D1_FROZEN_SHA256,
        "D2_PRODUCTION_P2_AMENDMENT_CANDIDATE": (
            D2_PRODUCTION_P2_F3_AMENDMENT_SHA256
        ),
        # Transitional test-contract identity only. Production invariants and
        # the focused semantic tests govern future qualified amendments.
        "D2_TEST_CONTRACT_AMENDED_FOR_PROVISIONED_REGISTRY_STATE": (
            D2_TEST_CONTRACT_TRANSITION_SHA256
        ),
    }
    for expected in authority_classes.values():
        actual = {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in expected
        }
        assert actual == expected
