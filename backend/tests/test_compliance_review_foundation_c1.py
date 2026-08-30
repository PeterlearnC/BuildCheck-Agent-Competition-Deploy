import hashlib
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.review import (
    get_compliance_review_service,
    get_review_unit_service,
)
from app.core.config import Settings
from app.main import app
from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.review_unit import ReviewUnit
from app.schemas.standards_retrieval import (
    RetrievalDecision,
    RetrievalMethod,
    StandardSearchResponse,
)
from app.schemas.standards_scope import StandardScope
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.review.compliance_review_service import ComplianceReviewService
from app.services.review.review_unit_service import (
    ReviewPageNotFoundError,
    ReviewSourceSpanError,
    ReviewSourceTextAmbiguousError,
    ReviewSourceTextNotFoundError,
    ReviewUnitService,
)
from app.services.standards.scope_validation_service import (
    ScopeValidationError,
    ScopeValidationResult,
    ScopeValidationService,
)
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import make_record


PLAN_TEXT = "Plan brace connection shall be secure."


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for lines in pages:
        page = document.new_page()
        for index, line in enumerate(lines):
            page.insert_text((72, 72 + index * 24), line)
    document.save(path)
    document.close()


def _stored_plan(tmp_path: Path, pages: list[list[str]]) -> tuple[Settings, str]:
    settings = Settings(upload_dir=tmp_path / "uploads")
    document_id = str(uuid4())
    _write_pdf(settings.upload_dir / f"{document_id}.pdf", pages)
    return settings, document_id


def _unit(
    *,
    document_id: str = "00000000-0000-0000-0000-000000000001",
    source_text: str = PLAN_TEXT,
    review_unit_id: str = "reviewunit_test",
    retrieval_query: str = "1.1.1 brace connection",
    standard_id: str = "std-a",
) -> ReviewUnit:
    return ReviewUnit(
        review_unit_id=review_unit_id,
        document_id=document_id,
        page_number=1,
        source_text=source_text,
        source_text_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        char_start=0,
        char_end=len(source_text),
        review_text=source_text,
        retrieval_query=retrieval_query,
        standard_scope=StandardScope(standard_ids=[standard_id]),
    )


def _accepted_hit():
    record = make_record(
        "std-a",
        "1.1.1",
        "brace connection shall be secure",
        standard_code="JGJ 100-2020",
        standard_name="Synthetic safety standard",
        chapter_number="1.1",
        page_start=4,
    )
    return make_hit(
        record,
        rank=1,
        methods=[RetrievalMethod.KEYWORD],
        keyword_score=1.0,
    )


def _response(
    decision: RetrievalDecision,
    *,
    hits=None,
    query: str = "1.1.1 brace connection",
) -> StandardSearchResponse:
    selected = list(hits or [])
    return StandardSearchResponse(
        query=query,
        normalized_query=query.casefold(),
        total_candidates=1,
        returned=len(selected),
        hits=selected,
        corpus_fingerprint="corpus-test",
        retrieval_version="retrieval-test",
        manifest_version="manifest-test",
        retrieval_decision=decision,
        acceptance_reason=f"synthetic {decision.value}",
        query_coverage=1.0 if decision == RetrievalDecision.ACCEPT else 0.3,
    )


class _StaticSearch:
    def __init__(self, response: StandardSearchResponse) -> None:
        self.response = response
        self.requests = []

    def search(self, request):
        self.requests.append(request)
        return self.response


class _StaticScopeValidator:
    def validate_scope(self, scope: StandardScope) -> ScopeValidationResult:
        return ScopeValidationResult(scope=scope)


class _TrustingReviewUnitVerifier:
    def verify(self, review_unit: ReviewUnit) -> ReviewUnit:
        return review_unit


def _binding(unit: ReviewUnit, *, review_unit_id: str | None = None):
    evidence = _accepted_hit().evidence
    return ReviewEvidenceBinding(
        review_unit_id=review_unit_id or unit.review_unit_id,
        document_id=unit.document_id,
        plan_page_number=unit.page_number,
        plan_char_start=unit.char_start,
        plan_char_end=unit.char_end,
        plan_source_text=unit.source_text,
        plan_source_text_sha256=unit.source_text_sha256,
        retrieval_decision=RetrievalDecision.ACCEPT,
        standard_evidence_id=evidence.id,
        standard_id=evidence.standard_id,
        standard_article_number=evidence.article_number,
        standard_page_start=evidence.source_page_start,
        standard_page_end=evidence.source_page_end,
        evidence=evidence,
    )


def test_review_unit_is_created_from_exact_physical_page_text(tmp_path: Path) -> None:
    settings, document_id = _stored_plan(tmp_path, [[PLAN_TEXT]])
    unit = ReviewUnitService(settings).create(
        document_id=document_id,
        page_number=1,
        source_text="brace connection",
        retrieval_query="connection requirement",
        standard_scope=StandardScope(standard_ids=["std-a"]),
    )

    assert unit.source_text == "brace connection"
    assert unit.review_text == unit.source_text
    assert unit.char_end - unit.char_start == len(unit.source_text)
    assert unit.source_text_sha256 == hashlib.sha256(unit.source_text.encode()).hexdigest()
    assert unit.review_unit_id.startswith("reviewunit_")


def test_review_unit_absent_text_fails_closed(tmp_path: Path) -> None:
    settings, document_id = _stored_plan(tmp_path, [[PLAN_TEXT]])
    with pytest.raises(ReviewSourceTextNotFoundError):
        ReviewUnitService(settings).create(
            document_id=document_id,
            page_number=1,
            source_text="not present",
            retrieval_query="connection requirement",
            standard_scope=StandardScope(standard_ids=["std-a"]),
        )


def test_review_unit_wrong_physical_page_fails_closed(tmp_path: Path) -> None:
    settings, document_id = _stored_plan(tmp_path, [[PLAN_TEXT]])
    with pytest.raises(ReviewPageNotFoundError):
        ReviewUnitService(settings).create(
            document_id=document_id,
            page_number=2,
            source_text=PLAN_TEXT,
            retrieval_query="connection requirement",
            standard_scope=StandardScope(standard_ids=["std-a"]),
        )


def test_review_unit_duplicate_text_requires_explicit_span(tmp_path: Path) -> None:
    phrase = "repeated plan statement"
    settings, document_id = _stored_plan(tmp_path, [[phrase, phrase]])
    service = ReviewUnitService(settings)
    with pytest.raises(ReviewSourceTextAmbiguousError):
        service.create(
            document_id=document_id,
            page_number=1,
            source_text=phrase,
            retrieval_query="statement requirement",
            standard_scope=StandardScope(standard_ids=["std-a"]),
        )

    page_text = service.pdf_service.extract_text(
        settings.upload_dir / f"{document_id}.pdf"
    ).pages[0].text
    second_start = page_text.rfind(phrase)
    unit = service.create(
        document_id=document_id,
        page_number=1,
        source_text=phrase,
        char_start=second_start,
        retrieval_query="statement requirement",
        standard_scope=StandardScope(standard_ids=["std-a"]),
    )
    assert unit.char_start == second_start


def test_review_unit_explicit_wrong_span_fails_closed(tmp_path: Path) -> None:
    settings, document_id = _stored_plan(tmp_path, [[PLAN_TEXT]])
    with pytest.raises(ReviewSourceSpanError):
        ReviewUnitService(settings).create(
            document_id=document_id,
            page_number=1,
            source_text="brace connection",
            char_start=0,
            retrieval_query="connection requirement",
            standard_scope=StandardScope(standard_ids=["std-a"]),
        )


def test_review_unit_identity_is_stable_and_span_sensitive(tmp_path: Path) -> None:
    phrase = "repeated plan statement"
    settings, document_id = _stored_plan(tmp_path, [[phrase, phrase]])
    service = ReviewUnitService(settings)
    page_text = service.pdf_service.extract_text(
        settings.upload_dir / f"{document_id}.pdf"
    ).pages[0].text
    starts = [page_text.find(phrase), page_text.rfind(phrase)]
    units = [
        service.create(
            document_id=document_id,
            page_number=1,
            source_text=phrase,
            char_start=start,
            retrieval_query="statement requirement",
            standard_scope=StandardScope(standard_ids=["std-a"]),
        )
        for start in starts
    ]
    repeated = service.create(
        document_id=document_id,
        page_number=1,
        source_text=phrase,
        char_start=starts[0],
        retrieval_query="a different query does not affect source identity",
        standard_scope=StandardScope(standard_ids=["std-a"]),
    )
    assert units[0].review_unit_id == repeated.review_unit_id
    assert units[0].review_unit_id != units[1].review_unit_id


def test_review_unit_schema_rejects_hash_or_review_text_drift() -> None:
    payload = _unit().model_dump()
    payload["source_text_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="source_text_sha256"):
        ReviewUnit.model_validate(payload)
    payload = _unit().model_dump()
    payload["review_text"] = "rewritten text"
    with pytest.raises(ValidationError, match="exact grounded source_text"):
        ReviewUnit.model_validate(payload)


def test_accept_maps_only_to_needs_comparison_with_dual_provenance() -> None:
    unit = _unit()
    search = _StaticSearch(_response(RetrievalDecision.ACCEPT, hits=[_accepted_hit()]))
    result = ComplianceReviewService(
        search, _StaticScopeValidator(), _TrustingReviewUnitVerifier()
    ).review(unit)

    assert result.status == ComplianceReviewStatus.NEEDS_COMPARISON
    assert result.status not in {
        ComplianceReviewStatus.COMPLIANT,
        ComplianceReviewStatus.NON_COMPLIANT,
    }
    assert len(result.evidence_bindings) == 1
    binding = result.evidence_bindings[0]
    assert binding.review_unit_id == unit.review_unit_id
    assert binding.plan_source_text == unit.source_text
    assert binding.standard_evidence_id == binding.evidence.id
    assert search.requests[0].standard_ids == ["std-a"]


@pytest.mark.parametrize(
    "decision", [RetrievalDecision.LOW_CONFIDENCE, RetrievalDecision.NO_MATCH]
)
def test_non_accept_maps_to_insufficient_without_evidence(decision) -> None:
    result = ComplianceReviewService(
        _StaticSearch(_response(decision)),
        _StaticScopeValidator(),
        _TrustingReviewUnitVerifier(),
    ).review(_unit())
    assert result.status == ComplianceReviewStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence_bindings == []


@pytest.mark.parametrize(
    "status",
    [ComplianceReviewStatus.COMPLIANT, ComplianceReviewStatus.NON_COMPLIANT],
)
def test_c1_schema_rejects_final_compliance_judgments(status) -> None:
    unit = _unit()
    with pytest.raises(ValidationError, match="cannot emit"):
        ComplianceReviewResult(
            review_unit=unit,
            status=status,
            retrieval_decision=RetrievalDecision.ACCEPT,
            evidence_bindings=[_binding(unit)],
            reason="not permitted in C.1",
        )


@pytest.mark.parametrize(
    "status",
    [ComplianceReviewStatus.COMPLIANT, ComplianceReviewStatus.NON_COMPLIANT],
)
def test_final_judgment_without_plan_or_evidence_cannot_be_serialized(status) -> None:
    with pytest.raises(ValidationError):
        ComplianceReviewResult.model_validate(
            {
                "status": status,
                "retrieval_decision": RetrievalDecision.ACCEPT,
                "evidence_bindings": [],
                "reason": "unsupported final judgment",
            }
        )


def test_needs_comparison_rejects_missing_evidence() -> None:
    with pytest.raises(ValidationError, match="requires ACCEPT evidence"):
        ComplianceReviewResult(
            review_unit=_unit(),
            status=ComplianceReviewStatus.NEEDS_COMPARISON,
            retrieval_decision=RetrievalDecision.ACCEPT,
            evidence_bindings=[],
            reason="missing evidence",
        )


@pytest.mark.parametrize(
    "decision", [RetrievalDecision.LOW_CONFIDENCE, RetrievalDecision.NO_MATCH]
)
def test_non_accept_rejects_normative_evidence_binding(decision) -> None:
    unit = _unit()
    with pytest.raises(ValidationError, match="cannot expose"):
        ComplianceReviewResult(
            review_unit=unit,
            status=ComplianceReviewStatus.INSUFFICIENT_EVIDENCE,
            retrieval_decision=decision,
            evidence_bindings=[_binding(unit)],
            reason="not authoritative",
        )


def test_result_rejects_mismatched_review_unit_binding() -> None:
    unit = _unit()
    with pytest.raises(ValidationError, match="review_unit_id"):
        ComplianceReviewResult(
            review_unit=unit,
            status=ComplianceReviewStatus.NEEDS_COMPARISON,
            retrieval_decision=RetrievalDecision.ACCEPT,
            evidence_bindings=[_binding(unit, review_unit_id="reviewunit_other")],
            reason="mismatched unit",
        )


def test_binding_rejects_fabricated_evidence_id() -> None:
    unit = _unit()
    payload = _binding(unit).model_dump()
    payload["standard_evidence_id"] = "fabricated"
    with pytest.raises(ValidationError, match="EvidenceEnvelope.id"):
        ReviewEvidenceBinding.model_validate(payload)


def test_standard_evidence_without_plan_provenance_cannot_be_bound() -> None:
    payload = _binding(_unit()).model_dump()
    for field in (
        "review_unit_id",
        "document_id",
        "plan_page_number",
        "plan_char_start",
        "plan_char_end",
        "plan_source_text",
        "plan_source_text_sha256",
    ):
        payload.pop(field)
    with pytest.raises(ValidationError):
        ReviewEvidenceBinding.model_validate(payload)


def test_compliance_service_reverifies_plan_source_before_retrieval() -> None:
    class _RejectingVerifier:
        def verify(self, _review_unit):
            raise ReviewSourceSpanError("not grounded")

    search = _StaticSearch(_response(RetrievalDecision.ACCEPT, hits=[_accepted_hit()]))
    service = ComplianceReviewService(
        search, _StaticScopeValidator(), _RejectingVerifier()
    )
    with pytest.raises(ReviewSourceSpanError, match="not grounded"):
        service.review(_unit())
    assert search.requests == []


def test_wrong_standard_explicit_target_remains_fail_closed(tmp_path: Path) -> None:
    settings = Settings(standards_dir=tmp_path / "standards")
    repository = StandardRepository(settings)
    records = [
        make_record(
            "std-a",
            "1.1.1",
            "brace connection shall be secure",
            standard_code="JGJ 100-2020",
            standard_name="Synthetic standard A",
        ),
        make_record(
            "std-b",
            "3.2.5",
            "same numbered requirement in another standard",
            standard_code="JGJ 200-2020",
            standard_name="Synthetic standard B",
        ),
    ]
    for record in records:
        repository.save_document(record.document)
        repository.save_articles(record.document.standard_id, [record.article])
    registry = StandardRegistryService(settings, entries=[])
    service = ComplianceReviewService(
        StandardsSearchService(repository, registry=registry),
        ScopeValidationService(repository, registry),
        _TrustingReviewUnitVerifier(),
    )
    result = service.review(
        _unit(retrieval_query="JGJ 100-2020 3.2.5", standard_id="std-a")
    )
    assert result.status == ComplianceReviewStatus.INSUFFICIENT_EVIDENCE
    assert result.retrieval_decision == RetrievalDecision.NO_MATCH
    assert result.evidence_bindings == []


def test_conflicting_and_unknown_scope_fail_closed(tmp_path: Path) -> None:
    settings = Settings(standards_dir=tmp_path / "standards")
    repository = StandardRepository(settings)
    record = make_record(
        "std-a",
        "1.1.1",
        "brace connection shall be secure",
        standard_code="JGJ 100-2020",
        standard_name="Synthetic standard A",
    )
    repository.save_document(record.document)
    repository.save_articles("std-a", [record.article])
    registry = StandardRegistryService(settings, entries=[])
    service = ComplianceReviewService(
        StandardsSearchService(repository, registry=registry),
        ScopeValidationService(repository, registry),
        _TrustingReviewUnitVerifier(),
    )

    conflict = service.review(
        _unit(retrieval_query="JGJ 200-2020 1.1.1", standard_id="std-a")
    )
    assert conflict.status == ComplianceReviewStatus.INSUFFICIENT_EVIDENCE
    assert conflict.evidence_bindings == []
    with pytest.raises(ScopeValidationError, match="Unknown standard_id"):
        service.review(_unit(standard_id="missing"))


@pytest.fixture
def compliance_api(tmp_path: Path):
    settings, document_id = _stored_plan(tmp_path, [[PLAN_TEXT]])
    settings.standards_dir = tmp_path / "standards"
    repository = StandardRepository(settings)
    records = [
        make_record(
            "std-a",
            "1.1.1",
            "brace connection shall be secure",
            standard_code="JGJ 100-2020",
            standard_name="Synthetic standard A",
            page_start=4,
        ),
        make_record(
            "std-b",
            "3.2.5",
            "other standard requirement",
            standard_code="JGJ 200-2020",
            standard_name="Synthetic standard B",
            page_start=5,
        ),
    ]
    for record in records:
        repository.save_document(record.document)
        repository.save_articles(record.document.standard_id, [record.article])
    registry = StandardRegistryService(settings, entries=[])
    service = ComplianceReviewService(
        StandardsSearchService(repository, registry=registry),
        ScopeValidationService(repository, registry),
        ReviewUnitService(settings),
    )
    app.dependency_overrides[get_review_unit_service] = lambda: ReviewUnitService(
        settings
    )
    app.dependency_overrides[get_compliance_review_service] = lambda: service
    yield TestClient(app), settings, document_id
    app.dependency_overrides.clear()


def _api_payload(query: str = "JGJ 100-2020 1.1.1 brace connection"):
    return {
        "page_number": 1,
        "source_text": "brace connection",
        "retrieval_query": query,
        "standard_ids": ["std-a"],
    }


def test_compliance_api_accept_traceability_chain(compliance_api) -> None:
    client, _, document_id = compliance_api
    response = client.post(
        f"/api/v1/documents/{document_id}/review/compliance",
        json=_api_payload(),
    )
    assert response.status_code == 200, response.text
    result = response.json()
    unit = result["review_unit"]
    binding = result["evidence_bindings"][0]
    assert result["status"] == "NEEDS_COMPARISON"
    assert result["retrieval_decision"] == "ACCEPT"
    assert unit["document_id"] == document_id
    assert unit["page_number"] == 1
    assert unit["source_text"] == "brace connection"
    assert unit["char_end"] - unit["char_start"] == len(unit["source_text"])
    assert binding["review_unit_id"] == unit["review_unit_id"]
    assert binding["plan_source_text_sha256"] == unit["source_text_sha256"]
    assert binding["standard_evidence_id"] == binding["evidence"]["id"]
    assert binding["standard_article_number"] == "1.1.1"


@pytest.mark.parametrize(
    ("document_id", "payload", "status_code"),
    [
        ("invalid", _api_payload(), 404),
        (None, {**_api_payload(), "page_number": 2}, 422),
        (None, {**_api_payload(), "source_text": "not on page"}, 422),
    ],
)
def test_compliance_api_rejects_invalid_plan_source(
    compliance_api, document_id, payload, status_code
) -> None:
    client, _, stored_id = compliance_api
    response = client.post(
        f"/api/v1/documents/{document_id or stored_id}/review/compliance",
        json=payload,
    )
    assert response.status_code == status_code


def test_compliance_api_rejects_ambiguous_source_text(compliance_api) -> None:
    client, settings, _ = compliance_api
    document_id = str(uuid4())
    _write_pdf(
        settings.upload_dir / f"{document_id}.pdf",
        [["duplicate statement", "duplicate statement"]],
    )
    response = client.post(
        f"/api/v1/documents/{document_id}/review/compliance",
        json={
            **_api_payload(),
            "source_text": "duplicate statement",
        },
    )
    assert response.status_code == 409


def test_compliance_api_no_match_and_scope_conflict_are_insufficient(
    compliance_api,
) -> None:
    client, _, document_id = compliance_api
    for query in ("JGJ 100-2020 9.9.9", "JGJ 200-2020 1.1.1"):
        response = client.post(
            f"/api/v1/documents/{document_id}/review/compliance",
            json=_api_payload(query),
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "INSUFFICIENT_EVIDENCE"
        assert result["retrieval_decision"] == "NO_MATCH"
        assert result["evidence_bindings"] == []


def test_compliance_api_low_confidence_exposes_no_evidence(compliance_api) -> None:
    client, _, document_id = compliance_api
    low_service = ComplianceReviewService(
        _StaticSearch(_response(RetrievalDecision.LOW_CONFIDENCE)),
        _StaticScopeValidator(),
        _TrustingReviewUnitVerifier(),
    )
    app.dependency_overrides[get_compliance_review_service] = lambda: low_service
    response = client.post(
        f"/api/v1/documents/{document_id}/review/compliance",
        json=_api_payload("partial terminology"),
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "INSUFFICIENT_EVIDENCE"
    assert response.json()["evidence_bindings"] == []


def test_compliance_api_unknown_scope_fails_closed(compliance_api) -> None:
    client, _, document_id = compliance_api
    response = client.post(
        f"/api/v1/documents/{document_id}/review/compliance",
        json={**_api_payload(), "standard_ids": ["missing"]},
    )
    assert response.status_code == 422


def test_existing_completeness_endpoint_remains_separate(compliance_api) -> None:
    client, _, _ = compliance_api
    missing_document_id = str(uuid4())
    response = client.post(
        f"/api/v1/documents/{missing_document_id}/review/completeness"
    )
    assert response.status_code == 404
