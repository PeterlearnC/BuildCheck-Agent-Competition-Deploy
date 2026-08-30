import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.review import get_review_finding_service
from app.core.config import Settings
from app.main import app
from app.schemas.compliance_comparison import (
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonReasonCode,
    ComplianceComparisonRequest,
    SourceSpanSelector,
)
from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.review_finding import ReviewFinding, ReviewFindingResponse
from app.schemas.review_unit import ReviewUnit
from app.schemas.standards_retrieval import RetrievalDecision, RetrievalMethod
from app.schemas.standards_scope import StandardScope
from app.services.retrieval.hit_factory import make_hit
from app.services.review.compliance_comparison_service import ComplianceComparisonService
from app.services.review.plan_fact_service import PlanFactService
from app.services.review.requirement_service import RequirementService
from app.services.review.review_finding_service import (
    ReviewFindingAuthorityMismatchError,
    ReviewFindingAuthorityUnavailableError,
    ReviewFindingService,
    ReviewFindingSourceChangedError,
)
from app.services.review.review_unit_service import ReviewDocumentNotFoundError
from tests.retrieval_helpers import make_record


DOCUMENT_ID = "00000000-0000-0000-0000-000000000003"
STANDARD_TEXT = "Platform width must be at least 1.2 m."
PLAN_TEXT = "Platform width is 1.5 m."


class _TrustingVerifier:
    def verify(self, review_unit):
        return review_unit


class _StaticReviewUnitService:
    def __init__(self, unit):
        self.unit = unit
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        return self.unit


class _StaticComplianceReviewService:
    def __init__(self, preparation):
        self.preparation = preparation
        self.calls = 0

    def review(self, _unit, *, top_k=5):
        self.calls += 1
        return self.preparation


class _StaticFindingService:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error

    def create(self, _document_id, _request):
        if self.error:
            raise self.error
        return self.response


class _MutatingComparisonService:
    def __init__(self, delegate, path: Path):
        self.delegate = delegate
        self.path = path

    def compare(self, *args, **kwargs):
        response = self.delegate.compare(*args, **kwargs)
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        return response


def _unit(plan_text: str = PLAN_TEXT) -> ReviewUnit:
    return ReviewUnit(
        review_unit_id="reviewunit_c3",
        document_id=DOCUMENT_ID,
        page_number=2,
        source_text=plan_text,
        source_text_sha256=hashlib.sha256(plan_text.encode()).hexdigest(),
        char_start=10,
        char_end=10 + len(plan_text),
        review_text=plan_text,
        retrieval_query="STD-A 1.1.1",
        standard_scope=StandardScope(standard_ids=["std-a"]),
    )


def _binding(unit: ReviewUnit, standard_text: str = STANDARD_TEXT):
    record = make_record(
        "std-a",
        "1.1.1",
        standard_text,
        standard_code="STD-A",
        standard_name="Synthetic numeric standard",
        page_start=7,
    )
    evidence = make_hit(
        record,
        rank=1,
        methods=[RetrievalMethod.KEYWORD],
        keyword_score=1.0,
    ).evidence
    return ReviewEvidenceBinding(
        review_unit_id=unit.review_unit_id,
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


def _preparation(
    *,
    unit: ReviewUnit | None = None,
    standard_text: str = STANDARD_TEXT,
    decision: RetrievalDecision = RetrievalDecision.ACCEPT,
) -> ComplianceReviewResult:
    unit = unit or _unit()
    if decision == RetrievalDecision.ACCEPT:
        return ComplianceReviewResult(
            review_unit=unit,
            status=ComplianceReviewStatus.NEEDS_COMPARISON,
            retrieval_decision=decision,
            evidence_bindings=[_binding(unit, standard_text)],
            reason="accepted",
        )
    return ComplianceReviewResult(
        review_unit=unit,
        status=ComplianceReviewStatus.INSUFFICIENT_EVIDENCE,
        retrieval_decision=decision,
        evidence_bindings=[],
        reason="not authoritative",
    )


def _selector(text: str, selected: str | None = None) -> SourceSpanSelector:
    selected = selected or text
    start = text.index(selected)
    return SourceSpanSelector(
        char_start=start,
        char_end=start + len(selected),
        source_text=selected,
    )


def _comparison_service() -> ComplianceComparisonService:
    return ComplianceComparisonService(
        RequirementService(), PlanFactService(_TrustingVerifier())
    )


def _request(
    preparation: ComplianceReviewResult,
    *,
    requirement_text: str = STANDARD_TEXT,
    fact_text: str | None = "1.5 m",
) -> ComplianceComparisonRequest:
    evidence_text = preparation.evidence_bindings[0].evidence.source_text
    return ComplianceComparisonRequest(
        page_number=preparation.review_unit.page_number,
        source_text=preparation.review_unit.source_text,
        char_start=preparation.review_unit.char_start,
        retrieval_query=preparation.review_unit.retrieval_query,
        standard_ids=["std-a"],
        evidence_id=preparation.evidence_bindings[0].standard_evidence_id,
        requirement=_selector(evidence_text, requirement_text),
        plan_facts=(
            []
            if fact_text is None
            else [_selector(preparation.review_unit.source_text, fact_text)]
        ),
    )


def _service(
    tmp_path: Path,
    *,
    preparation: ComplianceReviewResult | None = None,
    comparison_service=None,
) -> tuple[ReviewFindingService, Path]:
    preparation = preparation or _preparation()
    path = tmp_path / f"{DOCUMENT_ID}.pdf"
    path.write_bytes(b"authoritative stored pdf bytes")
    return (
        ReviewFindingService(
            settings=Settings(upload_dir=tmp_path),
            review_unit_service=_StaticReviewUnitService(preparation.review_unit),
            compliance_review_service=_StaticComplianceReviewService(preparation),
            comparison_service=comparison_service or _comparison_service(),
        ),
        path,
    )


def _finding(
    tmp_path: Path,
    *,
    plan_text: str = PLAN_TEXT,
    fact_text: str | None = "1.5 m",
) -> ReviewFindingResponse:
    preparation = _preparation(unit=_unit(plan_text))
    service, _ = _service(tmp_path, preparation=preparation)
    return service.create(
        DOCUMENT_ID,
        _request(preparation, fact_text=fact_text),
    )


@pytest.mark.parametrize(
    ("plan_text", "fact_text", "expected"),
    [
        ("Platform width is 1.5 m.", "1.5 m", ComparisonDecision.COMPLIANT),
        ("Platform width is 1.0 m.", "1.0 m", ComparisonDecision.NON_COMPLIANT),
        ("Platform width is not stated.", None, ComparisonDecision.INSUFFICIENT_INFORMATION),
    ],
)
def test_finding_preserves_all_c2_decisions(
    tmp_path, plan_text, fact_text, expected
) -> None:
    finding = _finding(tmp_path, plan_text=plan_text, fact_text=fact_text).finding
    result = finding.authoritative_comparison_response.comparison
    assert finding.decision == expected == result.decision
    assert finding.reason_code == result.reason_code
    assert finding.decision_scope == ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT


def test_service_reconstructs_c1_and_c2_instead_of_accepting_authority(tmp_path) -> None:
    preparation = _preparation()
    units = _StaticReviewUnitService(preparation.review_unit)
    reviews = _StaticComplianceReviewService(preparation)
    path = tmp_path / f"{DOCUMENT_ID}.pdf"
    path.write_bytes(b"pdf")
    service = ReviewFindingService(
        settings=Settings(upload_dir=tmp_path),
        review_unit_service=units,
        compliance_review_service=reviews,
        comparison_service=_comparison_service(),
    )
    response = service.create(DOCUMENT_ID, _request(preparation))
    assert units.calls == 1
    assert reviews.calls == 1
    assert response.finding.authoritative_comparison_response.comparison is not None


def test_plan_and_standard_citations_are_exact_authoritative_projections(tmp_path) -> None:
    finding = _finding(tmp_path).finding
    authority = finding.authoritative_comparison_response
    unit = authority.preparation.review_unit
    evidence = authority.evidence_binding.evidence
    requirement = authority.requirement
    stored_pdf = tmp_path / f"{DOCUMENT_ID}.pdf"
    assert finding.document_sha256 == hashlib.sha256(stored_pdf.read_bytes()).hexdigest()
    assert finding.plan_citation.source_text == unit.source_text
    assert finding.plan_citation.source_text_sha256 == unit.source_text_sha256
    assert finding.plan_citation.plan_facts[0].source_text == "1.5 m"
    assert finding.standard_citation.evidence_id == evidence.id
    assert finding.standard_citation.source_checksum == evidence.source_checksum
    assert finding.standard_citation.requirement_text == requirement.requirement_text
    assert authority.evidence_binding.evidence.raw_source_span_refs == evidence.raw_source_span_refs


def test_finding_identity_and_summary_are_exactly_repeatable(tmp_path) -> None:
    preparation = _preparation()
    service, _ = _service(tmp_path, preparation=preparation)
    request = _request(preparation)
    first = service.create(DOCUMENT_ID, request).finding
    second = service.create(DOCUMENT_ID, request).finding
    assert first == second
    assert first.finding_id == second.finding_id
    assert first.summary == second.summary
    assert first.decision.value in first.summary
    assert first.reason_code.value in first.summary
    assert first.decision_scope.value in first.summary
    assert "whole plan" not in first.summary.lower()
    assert "plan is compliant" not in first.summary.lower()


def test_non_accept_c1_creates_no_finding_authority(tmp_path) -> None:
    preparation = _preparation(decision=RetrievalDecision.NO_MATCH)
    service, _ = _service(tmp_path, preparation=preparation)
    request = ComplianceComparisonRequest(
        page_number=2,
        source_text=preparation.review_unit.source_text,
        char_start=preparation.review_unit.char_start,
        retrieval_query="STD-A 9.9.9",
        standard_ids=["std-a"],
        evidence_id="not-authoritative",
        requirement=_selector(STANDARD_TEXT),
        plan_facts=[],
    )
    with pytest.raises(ReviewFindingAuthorityUnavailableError):
        service.create(DOCUMENT_ID, request)


def test_local_absence_remains_insufficient_not_non_compliant(tmp_path) -> None:
    finding = _finding(
        tmp_path, plan_text="Platform width is not stated.", fact_text=None
    ).finding
    assert finding.decision == ComparisonDecision.INSUFFICIENT_INFORMATION
    assert finding.reason_code == ComparisonReasonCode.PLAN_FACT_MISSING
    assert "violat" not in finding.summary.lower()
    assert "non-compliant" not in finding.summary.lower()


def test_pdf_mutation_during_reconstruction_fails_closed(tmp_path) -> None:
    preparation = _preparation()
    base = _comparison_service()
    service, path = _service(tmp_path, preparation=preparation)
    service.comparison_service = _MutatingComparisonService(base, path)
    with pytest.raises(ReviewFindingSourceChangedError):
        service.create(DOCUMENT_ID, _request(preparation))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comparison_id", "comparison_fabricated"),
        ("decision", ComparisonDecision.NON_COMPLIANT),
        ("reason_code", ComparisonReasonCode.NUMERIC_LIMIT_VIOLATED),
        ("decision_scope", "WHOLE_PLAN"),
        ("review_unit_id", "reviewunit_fabricated"),
        ("evidence_id", "evidence_fabricated"),
        ("requirement_id", "requirement_fabricated"),
        ("summary", "The whole plan is compliant."),
    ],
)
def test_finding_rejects_tampered_authority_fields(tmp_path, field, value) -> None:
    payload = _finding(tmp_path).finding.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        ReviewFinding.model_validate(payload)


@pytest.mark.parametrize(
    ("citation", "field", "value"),
    [
        ("plan_citation", "source_text", "tampered plan text"),
        ("plan_citation", "source_text_sha256", "0" * 64),
        ("standard_citation", "standard_id", "wrong-standard"),
        ("standard_citation", "article_id", "wrong-article"),
        ("standard_citation", "evidence_id", "wrong-evidence"),
        ("standard_citation", "requirement_text", "tampered requirement"),
        ("standard_citation", "requirement_text_sha256", "0" * 64),
    ],
)
def test_finding_rejects_tampered_citations(
    tmp_path, citation, field, value
) -> None:
    payload = _finding(tmp_path).finding.model_dump()
    payload[citation][field] = value
    with pytest.raises(ValidationError):
        ReviewFinding.model_validate(payload)


def test_plan_fact_citation_outside_review_unit_is_rejected(tmp_path) -> None:
    payload = _finding(tmp_path).finding.model_dump()
    fact = payload["plan_citation"]["plan_facts"][0]
    fact["char_start"] = len(payload["plan_citation"]["source_text"]) + 1
    fact["char_end"] = fact["char_start"] + len(fact["source_text"])
    with pytest.raises(ValidationError, match="outside"):
        ReviewFinding.model_validate(payload)


@pytest.mark.parametrize(
    ("authority_path", "value"),
    [
        (("requirement", "requirement_text"), "outside evidence"),
        (("plan_facts", 0, "source_text"), "outside review unit"),
    ],
)
def test_tampered_embedded_c2_provenance_is_rejected(
    tmp_path, authority_path, value
) -> None:
    payload = _finding(tmp_path).finding.model_dump()
    target = payload["authoritative_comparison_response"]
    for part in authority_path[:-1]:
        target = target[part]
    target[authority_path[-1]] = value
    with pytest.raises(ValidationError):
        ReviewFinding.model_validate(payload)


def test_multiple_requirements_produce_distinct_findings(tmp_path) -> None:
    standard_text = (
        "Platform width must be at least 1.2 m. "
        "Platform height must be at least 2 m."
    )
    plan_text = "Platform width is 1.5 m and platform height is 2.5 m."
    preparation = _preparation(unit=_unit(plan_text), standard_text=standard_text)
    service, _ = _service(tmp_path, preparation=preparation)
    first = service.create(
        DOCUMENT_ID,
        _request(
            preparation,
            requirement_text="Platform width must be at least 1.2 m.",
            fact_text="1.5 m",
        ),
    ).finding
    second = service.create(
        DOCUMENT_ID,
        _request(
            preparation,
            requirement_text="Platform height must be at least 2 m.",
            fact_text="2.5 m",
        ),
    ).finding
    assert first.requirement_id != second.requirement_id
    assert first.comparison_id != second.comparison_id
    assert first.finding_id != second.finding_id


def test_api_returns_typed_source_grounded_finding(tmp_path) -> None:
    response_model = _finding(tmp_path)
    app.dependency_overrides[get_review_finding_service] = lambda: _StaticFindingService(
        response=response_model
    )
    try:
        response = TestClient(app).post(
            f"/api/v1/documents/{DOCUMENT_ID}/review/compliance/findings",
            json=_request(_preparation()).model_dump(mode="json"),
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    assert response.json()["finding"]["decision"] == "COMPLIANT"
    assert response.json()["finding"]["authoritative_comparison_response"]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ReviewDocumentNotFoundError("missing"), 404),
        (ReviewFindingAuthorityUnavailableError("no C.2 authority"), 422),
        (ReviewFindingSourceChangedError("changed"), 422),
        (ReviewFindingAuthorityMismatchError("invalid authority"), 500),
    ],
)
def test_api_maps_finding_failures_without_fabricating_response(
    error, status_code
) -> None:
    app.dependency_overrides[get_review_finding_service] = lambda: _StaticFindingService(
        error=error
    )
    try:
        response = TestClient(app).post(
            f"/api/v1/documents/{DOCUMENT_ID}/review/compliance/findings",
            json=_request(_preparation()).model_dump(mode="json"),
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == status_code
    assert "finding" not in response.json()


def test_existing_c1_and_c2_endpoints_remain_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/v1/documents/{document_id}/review/compliance" in paths
    assert "/api/v1/documents/{document_id}/review/compliance/compare" in paths
    assert "/api/v1/documents/{document_id}/review/compliance/findings" in paths
