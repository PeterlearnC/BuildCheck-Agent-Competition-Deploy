"""Focused safety and deterministic-identity tests for D.2 standard routing."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.ocr import OfficialSourceBinding
from app.schemas.review_candidate import (
    REVIEW_CANDIDATE_DISCOVERY_VERSION,
    REVIEW_CANDIDATE_IDENTITY_VERSION,
    CandidateSourceSpan,
    ReviewCandidate,
    ReviewCandidateClass,
    ReviewCandidateStatus,
    deterministic_candidate_id,
)
from app.schemas.standard_route import (
    QualifiedStandardScopeCandidate,
    StandardRoute,
    StandardRouteStatus,
    deterministic_route_id,
)
from app.schemas.standards import (
    MetadataConfidence,
    StandardDocument,
    StandardIdentityStatus,
    StandardParseStatus,
    StandardRegistryEntry,
    StandardRegistryStatus,
)
from app.services.review.standard_routing_service import StandardRoutingService
from app.services.standards.standard_registry_service import StandardRegistryService


SOURCE_SHA = "a" * 64
D1_PATHS = {
    "backend/app/schemas/review_candidate.py": "4a180082f7c2b119728185782fba93a92fc7bf54f243c3fb3db9839b0642d009",
    "backend/app/services/review/candidate_discovery_service.py": "313a031bd6b2ae508cb5adb8b53298ae307017f4ab76b329b1fbf6618d346e0c",
    "backend/tests/test_review_candidate_discovery_d1.py": "0d1c39ef845c91b9c456733a07c4f1c55d2d8be90535eedfa1347968d297c2ae",
}


class MemoryRepository:
    def __init__(self, documents=()):
        self.documents = list(documents)

    def list_documents(self):
        return list(self.documents)


def make_candidate(
    text="可调托撑螺杆伸出钢管顶部不得大于200mm。",
    *,
    candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
    document_sha="b" * 64,
    topic="可调托撑",
):
    text_sha = hashlib.sha256(text.encode()).hexdigest()
    span = CandidateSourceSpan(
        page_number=1,
        char_start=0,
        char_end=len(text),
        source_text=text,
        source_text_sha256=text_sha,
    )
    candidate_id = deterministic_candidate_id(
        document_sha256=document_sha,
        source_spans=[span],
        candidate_class=candidate_class,
    )
    status = (
        ReviewCandidateStatus.UNRESOLVED
        if candidate_class == ReviewCandidateClass.UNRESOLVED
        else ReviewCandidateStatus.DISCOVERED
    )
    return ReviewCandidate(
        identity_version=REVIEW_CANDIDATE_IDENTITY_VERSION,
        candidate_id=candidate_id,
        document_id="document-1",
        document_sha256=document_sha,
        source_spans=(span,),
        source_text=text,
        source_text_sha256=text_sha,
        topic=topic,
        candidate_class=candidate_class,
        discovery_version=REVIEW_CANDIDATE_DISCOVERY_VERSION,
        confidence=0.9,
        status=status,
    )


def make_document(standard_id="standard-1", checksum=SOURCE_SHA):
    binding = OfficialSourceBinding(
        source_checksum=checksum,
        canonical_standard_code="GB55023-2022",
        display_standard_code="GB 55023-2022",
        standard_name="建筑与市政工程施工质量控制通用规范",
        binding_reason="controlled test binding",
        binding_provenance="server-qualified test registry",
        confirmed=True,
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return StandardDocument(
        standard_id=standard_id,
        standard_code="GB 55023-2022",
        standard_name="建筑与市政工程施工质量控制通用规范",
        canonical_standard_code="GB55023-2022",
        identity_status=StandardIdentityStatus.CONFIRMED,
        identity_confidence=MetadataConfidence.HIGH,
        source_filename="qualified.pdf",
        source_checksum=checksum,
        page_count=1,
        parse_status=StandardParseStatus.PARSED,
        official_source_binding=binding,
        created_at=now,
        updated_at=now,
    )


def make_registry(code="GB 55023-2022", status=StandardRegistryStatus.ACTIVE):
    return StandardRegistryService(
        entries=[
            StandardRegistryEntry(
                standard_code=code,
                standard_name="建筑与市政工程施工质量控制通用规范",
                discipline="construction",
                status=status,
                jurisdiction="CN",
            )
        ]
    )


def service(documents=(None,), registry=None):
    docs = [make_document()] if documents == (None,) else list(documents)
    return StandardRoutingService(
        repository=MemoryRepository(docs), registry=registry or make_registry()
    )


def test_same_candidate_and_registry_produce_same_route_identity_and_output():
    candidate = make_candidate()
    first = service().route(candidate)
    second = service().route(candidate)
    assert first == second
    assert first.route_status == StandardRouteStatus.ROUTED


def test_changed_candidate_id_changes_route_identity():
    first = service().route(make_candidate(document_sha="b" * 64))
    second = service().route(make_candidate(document_sha="c" * 64))
    assert first.route_id != second.route_id


def test_changed_resolved_scope_changes_route_identity():
    first = service([make_document("standard-1")]).route(make_candidate())
    second = service([make_document("standard-2")]).route(make_candidate())
    assert first.route_id != second.route_id


def test_confidence_topic_status_and_reason_do_not_enter_route_identity():
    route = service().route(make_candidate())
    changed = route.model_copy(
        update={"routing_confidence": 0.1, "candidate_topic": "display-only", "routing_reason": "changed"}
    )
    assert changed.route_id == route.route_id


def test_schema_forbids_extra_fields():
    route = service().route(make_candidate())
    with pytest.raises(ValidationError):
        StandardRoute.model_validate({**route.model_dump(), "article_id": "3.2.1"})


def test_routed_requires_one_selected_qualified_scope():
    route = service().route(make_candidate())
    payload = route.model_dump()
    payload["selected_scope"] = None
    with pytest.raises(ValidationError):
        StandardRoute.model_validate(payload)


def test_unregistered_or_nonactive_standard_cannot_route():
    absent = service(registry=StandardRegistryService(entries=[])).route(make_candidate())
    inactive = service(registry=make_registry(status=StandardRegistryStatus.SUPERSEDED)).route(
        make_candidate()
    )
    assert absent.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert inactive.route_status == StandardRouteStatus.NO_STANDARD_SCOPE


def test_unqualified_document_cannot_route():
    document = make_document().model_copy(update={"official_source_binding": None})
    route = service([document]).route(make_candidate())
    assert route.route_status == StandardRouteStatus.NO_STANDARD_SCOPE


def test_multiple_equally_qualified_documents_are_ambiguous():
    route = service([make_document("standard-1"), make_document("standard-2")]).route(
        make_candidate()
    )
    assert route.route_status == StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE
    assert route.selected_scope is None
    assert len(route.scope_candidates) == 2


@pytest.mark.parametrize(
    "text,topic,candidate_class",
    [
        ("施工过程中应加强管理。", "一般审查目标", ReviewCandidateClass.CONSTRUCTION_REQUIREMENT),
        ("高处作业人员必须设置安全防护措施。", "安全防护", ReviewCandidateClass.SAFETY_REQUIREMENT),
        ("材料进场后应检查。", "材料", ReviewCandidateClass.INSPECTION_REQUIREMENT),
    ],
)
def test_generic_or_unsupported_domains_do_not_route(text, topic, candidate_class):
    route = service().route(
        make_candidate(text, topic=topic, candidate_class=candidate_class)
    )
    assert route.route_status == StandardRouteStatus.NO_STANDARD_SCOPE


@pytest.mark.parametrize(
    "text,topic",
    [
        ("可调托撑螺杆伸出钢管顶部不得大于200mm。", "可调托撑"),
        ("螺杆外径与立杆钢管内径的间隙不大于3mm。", "立杆"),
        ("模板支撑架搭设完成后应组织检查验收。", "模板支撑"),
    ],
)
def test_strong_scaffold_domains_route_through_qualified_registry(text, topic):
    route = service().route(make_candidate(text, topic=topic))
    assert route.route_status == StandardRouteStatus.ROUTED
    assert route.selected_scope.standard_ids == ["standard-1"]
    assert route.routing_confidence == 0.9


def test_route_contains_no_article_normative_or_compliance_authority():
    payload = service().route(make_candidate()).model_dump(mode="json")
    forbidden = {
        "article_id", "article_number", "normative_quotation", "threshold", "operator",
        "decision", "compliance", "finding_id", "review_action", "whole_document_status",
    }
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(StandardRoute.model_fields)


def test_route_does_not_mutate_d1_candidate():
    candidate = make_candidate()
    before = candidate.model_dump_json()
    service().route(candidate)
    assert candidate.model_dump_json() == before


def test_registry_fingerprint_changes_when_qualified_registry_changes():
    first = service().route(make_candidate())
    changed_registry = StandardRegistryService(
        entries=[
            StandardRegistryEntry(
                standard_code="GB 55023-2022",
                standard_name="建筑与市政工程施工质量控制通用规范",
                discipline="construction",
                status=StandardRegistryStatus.ACTIVE,
                jurisdiction="CN-LOCAL",
            )
        ]
    )
    second = service(registry=changed_registry).route(make_candidate())
    assert first.registry_fingerprint != second.registry_fingerprint
    assert first.route_id != second.route_id


def test_route_identity_recomputes_from_authoritative_fields():
    route = service().route(make_candidate())
    assert route.route_id == deterministic_route_id(
        candidate_id=route.candidate_id,
        scope_candidates=route.scope_candidates,
        registry_fingerprint=route.registry_fingerprint,
    )


def test_scope_candidate_schema_forbids_article_data():
    scope = service().route(make_candidate()).scope_candidates[0]
    with pytest.raises(ValidationError):
        QualifiedStandardScopeCandidate.model_validate(
            {**scope.model_dump(), "article_id": "5.1.2"}
        )


def test_non_review_candidate_input_is_rejected():
    with pytest.raises(TypeError):
        service().route({"candidate_id": "caller-controlled"})


def test_default_repository_registry_state_fails_closed(tmp_path):
    isolated_registry = StandardRegistryService(
        settings=Settings(standards_dir=tmp_path / "standards")
    )
    service = StandardRoutingService(
        repository=MemoryRepository([make_document()]),
        registry=isolated_registry,
    )
    assert not isolated_registry.path.exists()
    qualified_scopes, _ = service._qualified_scopes()
    assert qualified_scopes == ()
    route = service.route(make_candidate())
    assert route.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert route.selected_scope is None
    assert route.scope_candidates == ()


def test_d1_frozen_bytes_remain_unchanged():
    root = Path(__file__).resolve().parents[2]
    actual = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in D1_PATHS
    }
    assert actual == D1_PATHS


def test_production_d2_has_no_prohibited_authority_or_external_calls():
    root = Path(__file__).resolve().parents[2]
    production = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "backend/app/schemas/standard_route.py",
            "backend/app/services/review/standard_routing_service.py",
        )
    )
    forbidden = (
        "StandardsSearchService", "ComplianceComparisonService", "ReviewFindingService",
        "FindingReviewService", "openai", "deepseek", "httpx", "requests.", "subprocess",
        "article_number", "finding_id", "Case A", "Case B", "Case C",
    )
    assert all(token not in production for token in forbidden)
