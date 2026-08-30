import hashlib
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.review import (
    get_compliance_comparison_service,
    get_compliance_review_service,
    get_review_unit_service,
)
from app.main import app
from app.schemas.compliance_comparison import (
    ApplicabilityStatus,
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonMethod,
    ComparisonReasonCode,
    ComparisonResult,
    PlanFact,
    SourceSpanSelector,
)
from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.review_unit import ReviewUnit
from app.schemas.standards_retrieval import RetrievalDecision, RetrievalMethod
from app.schemas.standards_scope import StandardScope
from app.services.retrieval.hit_factory import make_hit
from app.services.review.compliance_comparison_service import (
    ComplianceComparisonService,
    EvidenceSelectionError,
)
from app.services.review.plan_fact_service import PlanFactError, PlanFactService
from app.services.review.requirement_service import (
    RequirementService,
    RequirementSpanError,
)
from app.services.review.review_unit_service import (
    ReviewDocumentNotFoundError,
    ReviewPageNotFoundError,
    ReviewSourceTextNotFoundError,
)
from tests.retrieval_helpers import make_record


STANDARD_TEXT = "作业平台宽度不得小于 1.2米。"
PLAN_TEXT = "方案明确作业平台宽度为 1.5米。"


class _TrustingVerifier:
    def verify(self, review_unit):
        return review_unit


def _unit(source_text: str = PLAN_TEXT) -> ReviewUnit:
    return ReviewUnit(
        review_unit_id="reviewunit_c2",
        document_id="00000000-0000-0000-0000-000000000002",
        page_number=2,
        source_text=source_text,
        source_text_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        char_start=10,
        char_end=10 + len(source_text),
        review_text=source_text,
        retrieval_query="JGJ 100-2020 1.1.1",
        standard_scope=StandardScope(standard_ids=["std-a"]),
    )


def _binding(unit: ReviewUnit, standard_text: str = STANDARD_TEXT):
    record = make_record(
        "std-a",
        "1.1.1",
        standard_text,
        standard_code="JGJ 100-2020",
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


def _service() -> ComplianceComparisonService:
    return ComplianceComparisonService(
        RequirementService(), PlanFactService(_TrustingVerifier())
    )


def _compare(
    *,
    plan_text: str = PLAN_TEXT,
    standard_text: str = STANDARD_TEXT,
    fact_text: str | None = "1.5米",
):
    unit = _unit(plan_text)
    preparation = _preparation(unit=unit, standard_text=standard_text)
    evidence_text = preparation.evidence_bindings[0].evidence.source_text
    return _service().compare(
        preparation,
        evidence_id=preparation.evidence_bindings[0].standard_evidence_id,
        requirement_selector=_selector(evidence_text, standard_text),
        plan_fact_selectors=[] if fact_text is None else [_selector(plan_text, fact_text)],
    )


def test_exact_requirement_is_grounded_and_identity_is_repeatable() -> None:
    binding = _binding(_unit())
    service = RequirementService()
    selector = _selector(binding.evidence.source_text, STANDARD_TEXT)
    requirement = service.select(binding, selector)
    repeated = service.select(binding, selector)
    assert requirement == repeated
    assert requirement.requirement_text == STANDARD_TEXT
    assert requirement.requirement_text in binding.evidence.source_text
    assert requirement.evidence_id == binding.evidence.id
    assert requirement.requirement_id.startswith("requirement_")
    assert requirement.value == Decimal("1.2")


def test_different_requirement_span_changes_identity() -> None:
    text = "说明。作业平台宽度不得小于 1.2米。"
    binding = _binding(_unit(), text)
    service = RequirementService()
    whole = service.select(binding, _selector(binding.evidence.source_text))
    part = service.select(
        binding, _selector(binding.evidence.source_text, STANDARD_TEXT)
    )
    assert whole.requirement_id != part.requirement_id


@pytest.mark.parametrize(
    "selector",
    [
        SourceSpanSelector(char_start=0, char_end=3, source_text="不存在"),
        SourceSpanSelector(char_start=1, char_end=1 + len(STANDARD_TEXT), source_text=STANDARD_TEXT),
    ],
)
def test_requirement_outside_or_wrong_span_is_rejected(selector) -> None:
    with pytest.raises(RequirementSpanError):
        RequirementService().select(_binding(_unit()), selector)


def test_multi_obligation_requirement_is_not_decisive() -> None:
    text = "高度不得超过 2米；宽度不得小于 1米"
    result = _compare(plan_text="高度为 1米", standard_text=text, fact_text="1米")
    assert result.comparison.decision == ComparisonDecision.INSUFFICIENT_INFORMATION
    assert result.comparison.reason_code == (
        ComparisonReasonCode.REQUIREMENT_DECOMPOSITION_UNRESOLVED
    )


def test_exact_plan_fact_is_grounded_repeatable_and_span_sensitive() -> None:
    unit = _unit("数值 1.5米，复核值 1.5米")
    service = PlanFactService(_TrustingVerifier())
    first = service.create_many(unit, [_selector(unit.source_text, "1.5米")])[0]
    repeated = service.create_many(unit, [_selector(unit.source_text, "1.5米")])[0]
    second_start = unit.source_text.rindex("1.5米")
    second = service.create_many(
        unit,
        [SourceSpanSelector(char_start=second_start, char_end=second_start + 4, source_text="1.5米")],
    )[0]
    assert first == repeated
    assert first.plan_fact_id != second.plan_fact_id
    assert first.normalized_value == Decimal("1.5")


def test_plan_fact_wrong_span_is_rejected() -> None:
    with pytest.raises(PlanFactError):
        PlanFactService(_TrustingVerifier()).create_many(
            _unit(),
            [SourceSpanSelector(char_start=0, char_end=4, source_text="9.9米")],
        )


def test_fabricated_normalized_plan_value_fails_recomputation() -> None:
    unit = _unit()
    service = PlanFactService(_TrustingVerifier())
    fact = service.create_many(unit, [_selector(unit.source_text, "1.5米")])[0]
    payload = fact.model_dump()
    payload["normalized_value"] = Decimal("99")
    fabricated = PlanFact.model_validate(payload)
    with pytest.raises(PlanFactError, match="recomputation"):
        service.verify_many(unit, [fabricated])


@pytest.mark.parametrize(
    ("standard_text", "plan_text", "fact_text", "decision"),
    [
        ("高度不得超过 2米", "高度为 1.5米", "1.5米", ComparisonDecision.COMPLIANT),
        ("高度不得超过 2米", "高度为 3米", "3米", ComparisonDecision.NON_COMPLIANT),
        ("宽度不得小于 1.2米", "宽度为 1米", "1米", ComparisonDecision.NON_COMPLIANT),
        ("宽度不得小于 1.2米", "宽度为 1.2米", "1.2米", ComparisonDecision.COMPLIANT),
    ],
)
def test_supported_numeric_comparisons(
    standard_text, plan_text, fact_text, decision
) -> None:
    response = _compare(
        plan_text=plan_text, standard_text=standard_text, fact_text=fact_text
    )
    assert response.comparison.decision == decision
    assert response.comparison.decision_scope == (
        ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT
    )
    assert response.comparison.comparison_method == ComparisonMethod.DETERMINISTIC_NUMERIC


def test_missing_local_fact_is_information_insufficient_not_violation() -> None:
    response = _compare(fact_text=None)
    assert response.comparison.decision == ComparisonDecision.INSUFFICIENT_INFORMATION
    assert response.comparison.reason_code == ComparisonReasonCode.PLAN_FACT_MISSING
    assert response.comparison.decision != ComparisonDecision.NON_COMPLIANT


def test_ambiguous_plan_numeric_source_is_information_insufficient() -> None:
    response = _compare(plan_text="宽度为 1米至2米", fact_text="1米至2米")
    assert response.comparison.reason_code == ComparisonReasonCode.PLAN_FACT_AMBIGUOUS


def test_unsupported_unit_conversion_is_information_insufficient() -> None:
    response = _compare(plan_text="宽度为 120厘米", fact_text="120厘米")
    assert response.comparison.reason_code == ComparisonReasonCode.UNSUPPORTED_UNIT


def test_conditional_requirement_has_unresolved_applicability() -> None:
    text = "当设置平台时，宽度不得小于 1.2米"
    response = _compare(standard_text=text)
    assert response.comparison.decision == ComparisonDecision.INSUFFICIENT_INFORMATION
    assert response.comparison.reason_code == ComparisonReasonCode.CONDITION_UNRESOLVED
    assert response.comparison.applicability_status == ApplicabilityStatus.UNRESOLVED


def test_unsupported_atomic_semantic_requirement_fails_closed() -> None:
    text = "构件应设置可靠防护。"
    response = _compare(standard_text=text)
    assert response.comparison.reason_code == ComparisonReasonCode.UNSUPPORTED_REQUIREMENT_CLASS


@pytest.mark.parametrize(
    "decision", [RetrievalDecision.NO_MATCH, RetrievalDecision.LOW_CONFIDENCE]
)
def test_non_accept_c1_stops_before_normative_comparison(decision) -> None:
    response = _service().compare(
        _preparation(decision=decision),
        evidence_id="fabricated",
        requirement_selector=_selector(STANDARD_TEXT),
        plan_fact_selectors=[],
    )
    assert response.preparation.retrieval_decision == decision
    assert response.evidence_binding is None
    assert response.comparison is None


def test_unknown_evidence_id_is_rejected() -> None:
    with pytest.raises(EvidenceSelectionError):
        _service().compare(
            _preparation(),
            evidence_id="fabricated",
            requirement_selector=_selector(STANDARD_TEXT),
            plan_fact_selectors=[],
        )


def test_decisive_result_rejects_scope_widening_or_missing_fact() -> None:
    response = _compare()
    payload = response.comparison.model_dump()
    payload["decision_scope"] = "WHOLE_DOCUMENT"
    with pytest.raises(ValidationError):
        ComparisonResult.model_validate(payload)
    payload = response.comparison.model_dump()
    payload["plan_facts_used"] = []
    with pytest.raises(ValidationError, match="requires grounded"):
        ComparisonResult.model_validate(payload)


def test_unresolved_applicability_cannot_be_made_decisive() -> None:
    response = _compare()
    payload = response.comparison.model_dump()
    payload["applicability_status"] = ApplicabilityStatus.UNRESOLVED
    with pytest.raises(ValidationError, match="proven applicability"):
        ComparisonResult.model_validate(payload)


def test_same_supported_comparison_is_exactly_repeatable() -> None:
    first = _compare()
    second = _compare()
    assert first == second
    assert first.comparison.comparison_id == second.comparison.comparison_id


class _StaticReviewUnitService:
    def __init__(self, unit):
        self.unit = unit

    def create(self, **_kwargs):
        return self.unit


class _StaticComplianceReviewService:
    def __init__(self, preparation):
        self.preparation = preparation

    def review(self, _unit, *, top_k=5):
        return self.preparation


class _RejectingReviewUnitService:
    def __init__(self, error):
        self.error = error

    def create(self, **_kwargs):
        raise self.error


def _api_payload(preparation: ComplianceReviewResult):
    binding = preparation.evidence_bindings[0]
    return {
        "page_number": 2,
        "source_text": PLAN_TEXT,
        "retrieval_query": "JGJ 100-2020 1.1.1",
        "standard_ids": ["std-a"],
        "evidence_id": binding.standard_evidence_id,
        "requirement": _selector(
            binding.evidence.source_text, STANDARD_TEXT
        ).model_dump(),
        "plan_facts": [_selector(PLAN_TEXT, "1.5米").model_dump()],
    }


def test_compare_api_reconstructs_c1_and_returns_local_result() -> None:
    preparation = _preparation()
    app.dependency_overrides[get_review_unit_service] = lambda: _StaticReviewUnitService(
        preparation.review_unit
    )
    app.dependency_overrides[get_compliance_review_service] = lambda: (
        _StaticComplianceReviewService(preparation)
    )
    app.dependency_overrides[get_compliance_comparison_service] = _service
    try:
        response = TestClient(app).post(
            f"/api/v1/documents/{preparation.review_unit.document_id}/review/compliance/compare",
            json=_api_payload(preparation),
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preparation"]["status"] == "NEEDS_COMPARISON"
    assert body["comparison"]["decision"] == "COMPLIANT"
    assert body["comparison"]["decision_scope"] == "REVIEW_UNIT_REQUIREMENT"
    assert body["evidence_binding"]["standard_evidence_id"] == body["requirement"]["evidence_id"]


def test_compare_api_non_accept_exposes_no_c2_authority() -> None:
    preparation = _preparation(decision=RetrievalDecision.NO_MATCH)
    app.dependency_overrides[get_review_unit_service] = lambda: _StaticReviewUnitService(
        preparation.review_unit
    )
    app.dependency_overrides[get_compliance_review_service] = lambda: (
        _StaticComplianceReviewService(preparation)
    )
    app.dependency_overrides[get_compliance_comparison_service] = _service
    payload = {
        "page_number": 2,
        "source_text": PLAN_TEXT,
        "retrieval_query": "JGJ 100-2020 9.9.9",
        "standard_ids": ["std-a"],
        "evidence_id": "not-authoritative",
        "requirement": _selector(STANDARD_TEXT).model_dump(),
        "plan_facts": [],
    }
    try:
        response = TestClient(app).post(
            f"/api/v1/documents/{preparation.review_unit.document_id}/review/compliance/compare",
            json=payload,
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["comparison"] is None
    assert response.json()["evidence_binding"] is None


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ReviewDocumentNotFoundError("missing document"), 404),
        (ReviewPageNotFoundError("wrong page"), 422),
        (ReviewSourceTextNotFoundError("absent source"), 422),
    ],
)
def test_compare_api_rejects_ungrounded_plan_source(error, expected_status) -> None:
    preparation = _preparation()
    app.dependency_overrides[get_review_unit_service] = lambda: (
        _RejectingReviewUnitService(error)
    )
    try:
        response = TestClient(app).post(
            f"/api/v1/documents/{preparation.review_unit.document_id}/review/compliance/compare",
            json=_api_payload(preparation),
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == expected_status


@pytest.mark.parametrize("invalid_selection", ["requirement", "plan_fact"])
def test_compare_api_rejects_source_spans_outside_authority(invalid_selection) -> None:
    preparation = _preparation()
    app.dependency_overrides[get_review_unit_service] = lambda: _StaticReviewUnitService(
        preparation.review_unit
    )
    app.dependency_overrides[get_compliance_review_service] = lambda: (
        _StaticComplianceReviewService(preparation)
    )
    app.dependency_overrides[get_compliance_comparison_service] = _service
    payload = _api_payload(preparation)
    if invalid_selection == "requirement":
        payload["requirement"] = {
            "char_start": 0,
            "char_end": len(STANDARD_TEXT),
            "source_text": STANDARD_TEXT,
        }
    else:
        payload["plan_facts"] = [
            {"char_start": 0, "char_end": 4, "source_text": "9.9米"}
        ]
    try:
        response = TestClient(app).post(
            f"/api/v1/documents/{preparation.review_unit.document_id}/review/compliance/compare",
            json=payload,
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


def test_existing_c1_endpoint_and_contract_remain_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/v1/documents/{document_id}/review/compliance" in paths
    assert "/api/v1/documents/{document_id}/review/compliance/compare" in paths


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("不大于", "<="),
        ("不得大于", "<="),
        ("不应大于", "<="),
        ("应不大于", "<="),
        ("小于等于", "<="),
        ("≤", "<="),
        ("不小于", ">="),
        ("不得小于", ">="),
        ("不应小于", ">="),
        ("应不小于", ">="),
        ("大于等于", ">="),
        ("≥", ">="),
        ("应小于", "<"),
        ("<", "<"),
        ("应大于", ">"),
        (">", ">"),
    ],
)
def test_required_numeric_operator_direction_contract(phrase, expected) -> None:
    text = f"限值{phrase} 20米"
    binding = _binding(_unit(), text)
    requirement = RequirementService().select(
        binding, _selector(binding.evidence.source_text, text)
    )
    assert requirement.operator is not None
    assert requirement.operator.value == expected


@pytest.mark.parametrize(
    ("phrase", "actual", "expected"),
    [
        ("小于等于", 15, ComparisonDecision.COMPLIANT),
        ("小于等于", 20, ComparisonDecision.COMPLIANT),
        ("小于等于", 25, ComparisonDecision.NON_COMPLIANT),
        ("≤", 15, ComparisonDecision.COMPLIANT),
        ("≤", 20, ComparisonDecision.COMPLIANT),
        ("≤", 25, ComparisonDecision.NON_COMPLIANT),
        ("大于等于", 25, ComparisonDecision.COMPLIANT),
        ("大于等于", 20, ComparisonDecision.COMPLIANT),
        ("大于等于", 15, ComparisonDecision.NON_COMPLIANT),
        ("≥", 25, ComparisonDecision.COMPLIANT),
        ("≥", 20, ComparisonDecision.COMPLIANT),
        ("≥", 15, ComparisonDecision.NON_COMPLIANT),
        ("应小于", 15, ComparisonDecision.COMPLIANT),
        ("应小于", 20, ComparisonDecision.NON_COMPLIANT),
        ("<", 15, ComparisonDecision.COMPLIANT),
        ("<", 20, ComparisonDecision.NON_COMPLIANT),
        ("应大于", 25, ComparisonDecision.COMPLIANT),
        ("应大于", 20, ComparisonDecision.NON_COMPLIANT),
        (">", 25, ComparisonDecision.COMPLIANT),
        (">", 20, ComparisonDecision.NON_COMPLIANT),
    ],
)
def test_textual_and_symbol_numeric_outcomes(phrase, actual, expected) -> None:
    standard_text = f"限值{phrase} 20米"
    plan_text = f"计划值为 {actual}米"
    response = _compare(
        standard_text=standard_text,
        plan_text=plan_text,
        fact_text=f"{actual}米",
    )
    assert response.comparison.decision == expected


def test_overlapping_operator_forms_use_longest_direction() -> None:
    cases = {
        "小于等于": "<=",
        "大于等于": ">=",
        "不得大于": "<=",
        "不得小于": ">=",
    }
    for phrase, expected in cases.items():
        text = f"限值{phrase} 20米"
        binding = _binding(_unit(), text)
        requirement = RequirementService().select(
            binding, _selector(binding.evidence.source_text, text)
        )
        assert requirement.operator is not None
        assert requirement.operator.value == expected
