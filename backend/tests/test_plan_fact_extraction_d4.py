import hashlib
import inspect
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.compliance_comparison import (
    PlanFactParsingStatus,
    PlanFactType,
)
from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.normative_requirement import (
    NormativeRequirement,
    NumericOperator,
    RequirementAtomicity,
    RequirementExtractionMethod,
    RequirementModality,
)
from app.schemas.plan_fact_extraction import (
    PlanAssertionKind,
    PlanFactExtractionReason,
    PlanFactExtractionResult,
    PlanFactExtractionStatus,
    deterministic_extraction_id,
)
from app.schemas.requirement_decomposition import (
    RequirementDecompositionMethod,
    RequirementDecompositionResult,
    RequirementDecompositionStatus,
    deterministic_decomposition_id,
)
from app.schemas.review_candidate import (
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
    StandardRoutingMethod,
    deterministic_route_id,
)
from app.schemas.standards_scope import StandardScope
from app.schemas.standards_retrieval import RetrievalDecision, RetrievalMethod
from app.services.pdf_service import PDFExtractionResult, PDFPageText
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.review.plan_fact_extraction_service import (
    PlanFactExtractionAuthorityError,
    PlanFactExtractionService,
)
from app.services.review.plan_fact_service import PlanFactError, PlanFactService
from app.services.review.review_unit_service import ReviewUnitService
from app.services.review.routing_context_service import RoutingContextService
from app.services.standards.standard_repository import StandardRepository


DOCUMENT_ID = "00000000-0000-0000-0000-000000000044"
STANDARD_ID = "synthetic-standard"
SOURCE_CHECKSUM = "c" * 64
REGISTRY_FINGERPRINT = "d" * 64


class _StaticPDFService:
    def __init__(self, page_text: str) -> None:
        self.page_text = page_text

    def extract_text(self, _path):
        page = PDFPageText(page_number=1, text=self.page_text)
        return PDFExtractionResult(
            page_count=1,
            char_count=len(self.page_text),
            text=self.page_text,
            pages=[page],
        )


class _StaticDecompositionService:
    def __init__(self, result: RequirementDecompositionResult) -> None:
        self.result = result
        self.calls = 0

    def decompose(self, _candidate, _route, _preparation):
        self.calls += 1
        return self.result


def _requirement(text: str, operator: NumericOperator, value: str, unit: str):
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    return NormativeRequirement(
        requirement_id="requirement_" + hashlib.sha256(
            f"{text}|{operator.value}|{value}|{unit}".encode()
        ).hexdigest(),
        evidence_id="evidence-synthetic",
        standard_id=STANDARD_ID,
        article_id="article-synthetic",
        article_number="1.1.1",
        requirement_char_start=0,
        requirement_char_end=len(text),
        requirement_text=text,
        requirement_text_sha256=text_hash,
        modality=RequirementModality.NUMERIC_LIMIT,
        atomicity=RequirementAtomicity.PROVEN,
        operator=operator,
        value=Decimal(value),
        unit=unit,
        extraction_method=RequirementExtractionMethod.EXACT_SPAN_DETERMINISTIC,
        extraction_version="test-frozen-requirement-authority",
    )


def _candidate(text: str, document_sha: str, *, start: int = 0):
    span = CandidateSourceSpan(
        page_number=1,
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    candidate_id = deterministic_candidate_id(
        document_sha256=document_sha,
        source_spans=(span,),
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
    )
    return ReviewCandidate(
        candidate_id=candidate_id,
        document_id=DOCUMENT_ID,
        document_sha256=document_sha,
        source_spans=(span,),
        source_text=text,
        source_text_sha256=span.source_text_sha256,
        topic="numeric_control",
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
        confidence=1.0,
        status=ReviewCandidateStatus.DISCOVERED,
    )


def _route(candidate: ReviewCandidate):
    scope = QualifiedStandardScopeCandidate(
        standard_id=STANDARD_ID,
        standard_code="GB 10000-2026",
        canonical_standard_code="GB10000-2026",
        standard_name="Synthetic standard",
        source_checksum=SOURCE_CHECKSUM,
    )
    route_id = deterministic_route_id(
        candidate_id=candidate.candidate_id,
        scope_candidates=(scope,),
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )
    return StandardRoute(
        route_id=route_id,
        candidate_id=candidate.candidate_id,
        route_status=StandardRouteStatus.ROUTED,
        candidate_topic=candidate.topic,
        candidate_class=candidate.candidate_class,
        scope_candidates=(scope,),
        selected_scope=StandardScope(standard_ids=[STANDARD_ID]),
        routing_method=StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES,
        routing_confidence=1.0,
        routing_reason="synthetic qualified scope",
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )


def _decomposition(candidate, route, requirement):
    decomposition_id = deterministic_decomposition_id(
        candidate_id=candidate.candidate_id,
        route_id=route.route_id,
        evidence_ids=(requirement.evidence_id,),
        requirements=(requirement,),
        unresolved_spans=(),
    )
    return RequirementDecompositionResult(
        decomposition_id=decomposition_id,
        candidate_id=candidate.candidate_id,
        route_id=route.route_id,
        document_id=candidate.document_id,
        document_sha256=candidate.document_sha256,
        standard_id=STANDARD_ID,
        standard_code="GB 10000-2026",
        canonical_standard_code="GB10000-2026",
        standard_name="Synthetic standard",
        evidence_ids=(requirement.evidence_id,),
        requirements=(requirement,),
        status=RequirementDecompositionStatus.DECOMPOSED,
        decomposition_method=RequirementDecompositionMethod.EXACT_SOURCE_DETERMINISTIC,
    )


def _fixture(tmp_path, plan_text: str, requirement: NormativeRequirement, *, page_text=None):
    page_text = page_text if page_text is not None else plan_text
    pdf_bytes = b"server-controlled-plan-pdf"
    document_sha = hashlib.sha256(pdf_bytes).hexdigest()
    settings = Settings(upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True)
    (settings.upload_dir / f"{DOCUMENT_ID}.pdf").write_bytes(pdf_bytes)
    start = page_text.index(plan_text)
    candidate = _candidate(plan_text, document_sha, start=start)
    route = _route(candidate)
    pdf_service = _StaticPDFService(page_text)
    unit_service = ReviewUnitService(settings=settings, pdf_service=pdf_service)
    unit = unit_service.create(
        document_id=DOCUMENT_ID,
        page_number=1,
        source_text=plan_text,
        char_start=start,
        retrieval_query="synthetic evidence",
        standard_scope=route.selected_scope,
    )
    preparation = ComplianceReviewResult.model_construct(review_unit=unit)
    decomposition = _decomposition(candidate, route, requirement)
    d3 = _StaticDecompositionService(decomposition)
    service = PlanFactExtractionService(
        decomposition_service=d3,
        routing_context_service=RoutingContextService(
            settings=settings, pdf_service=pdf_service
        ),
        review_unit_service=unit_service,
        plan_fact_service=PlanFactService(unit_service),
    )
    return service, candidate, route, preparation, d3, settings, pdf_service


def _extract(tmp_path, plan_text, requirement_text, operator, value, unit="mm", **kwargs):
    requirement = _requirement(requirement_text, operator, value, unit)
    fixture = _fixture(tmp_path, plan_text, requirement, **kwargs)
    service, candidate, route, preparation, *_ = fixture
    result = service.extract(
        candidate, route, preparation, requirement_id=requirement.requirement_id
    )
    return result, fixture, requirement


@pytest.mark.parametrize(
    ("plan_text", "requirement_text", "operator", "assertion", "expected"),
    [
        ("托撑插入立杆内长度为160mm。", "托撑插入立杆内长度不得小于150mm。", NumericOperator.GE, PlanAssertionKind.EXACT_VALUE, "160"),
        ("螺杆与立杆钢管间隙不得大于3mm。", "螺杆与立杆钢管间隙不得大于2.5mm。", NumericOperator.LE, PlanAssertionKind.UPPER_BOUND_CONTROL, "3"),
        ("托撑插入立杆内长度不小于150mm。", "托撑插入立杆内长度不得小于150mm。", NumericOperator.GE, PlanAssertionKind.LOWER_BOUND_CONTROL, "150"),
        ("螺杆与立杆钢管间隙设计值为3mm。", "螺杆与立杆钢管间隙不得大于2.5mm。", NumericOperator.LE, PlanAssertionKind.DESIGN_CONTROL, "3"),
    ],
)
def test_supported_assertion_kinds(
    tmp_path, plan_text, requirement_text, operator, assertion, expected
):
    result, fixture, _ = _extract(
        tmp_path, plan_text, requirement_text, operator, "150" if operator == NumericOperator.GE else "2.5"
    )
    assert result.status == PlanFactExtractionStatus.EXTRACTED
    assert result.binding.assertion_kind == assertion
    assert result.binding.plan_fact.normalized_value == Decimal(expected)
    assert fixture[4].calls == 1


def test_same_object_descriptive_value_is_eligible(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "建筑高度为90m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    assert result.status == PlanFactExtractionStatus.EXTRACTED
    assert result.binding.plan_fact.normalized_value == Decimal("90")


@pytest.mark.parametrize(
    "plan_text",
    ["钢管长度为6m。", "板厚为12mm。", "建筑高度为90m。"],
)
def test_unrelated_descriptive_numbers_are_not_bound(tmp_path, plan_text):
    result, *_ = _extract(
        tmp_path,
        plan_text,
        "托撑插入立杆内长度不得小于150mm。",
        NumericOperator.GE,
        "150",
    )
    assert result.status == PlanFactExtractionStatus.NOT_FOUND
    assert result.binding is None


def test_semicolon_segments_select_only_matching_object(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "钢管长度为6m；托撑伸出长度不得大于200mm。",
        "托撑伸出长度不得大于250mm。",
        NumericOperator.LE,
        "250",
    )
    assert result.status == PlanFactExtractionStatus.EXTRACTED
    assert result.binding.plan_fact.normalized_value == Decimal("200")
    assert result.binding.source_text == "托撑伸出长度不得大于200mm。"


def test_two_matching_facts_are_ambiguous(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "建筑高度为90m；建筑高度为95m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    assert result.status == PlanFactExtractionStatus.AMBIGUOUS
    assert result.reason == PlanFactExtractionReason.MULTIPLE_MATCHING_FACTS
    assert result.binding is None


@pytest.mark.parametrize(
    ("plan_text", "reason"),
    [
        ("螺杆与立杆钢管间隙为2mm和3mm。", PlanFactExtractionReason.MULTIPLE_VALUES),
        ("钢管外径为48.3×3.6mm。", PlanFactExtractionReason.COMPOUND_DIMENSION_UNSUPPORTED),
        ("建筑高度为80~90m。", PlanFactExtractionReason.RANGE_UNSUPPORTED),
    ],
)
def test_multiple_compound_and_range_values_fail_closed(tmp_path, plan_text, reason):
    requirement_text = (
        "钢管外径不得大于50mm。" if "外径" in plan_text else "建筑高度不得大于100m。"
    )
    if "间隙" in plan_text:
        requirement_text = "螺杆与立杆钢管间隙不得大于4mm。"
    unit = "m" if "高度" in plan_text else "mm"
    result, *_ = _extract(
        tmp_path, plan_text, requirement_text, NumericOperator.LE, "100", unit=unit
    )
    assert result.status in {
        PlanFactExtractionStatus.AMBIGUOUS,
        PlanFactExtractionStatus.UNSUPPORTED,
    }
    assert result.reason == reason
    assert result.binding is None


@pytest.mark.parametrize(
    "plan_text",
    [
        "当X时，间距不得大于1m。",
        "在X情况下，间距不得大于1m。",
        "在端部，间距不得大于1m。",
        "在顶部，间距不得大于1m。",
        "于节点处，间距不得大于1m。",
        "不同规格分别采用构件间距1m。",
        "施工过程中构件间距不得大于1m。",
    ],
)
def test_conditions_and_locations_are_unsupported(tmp_path, plan_text):
    result, *_ = _extract(
        tmp_path,
        plan_text,
        "构件间距不得大于1m。",
        NumericOperator.LE,
        "1",
        unit="m",
    )
    assert result.status == PlanFactExtractionStatus.UNSUPPORTED
    assert result.binding is None


@pytest.mark.parametrize(
    "plan_text",
    ["螺杆不得采用200mm。", "螺杆不采用200mm。", "螺杆禁止采用200mm。"],
)
def test_prohibited_choice_is_not_an_asserted_value(tmp_path, plan_text):
    result, *_ = _extract(
        tmp_path,
        plan_text,
        "螺杆长度不得大于250mm。",
        NumericOperator.LE,
        "250",
    )
    assert result.status == PlanFactExtractionStatus.UNSUPPORTED
    assert result.reason == PlanFactExtractionReason.PROHIBITED_VALUE_NOT_FACT


def test_opposite_direction_control_is_unsupported(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "建筑高度不小于90m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    assert result.status == PlanFactExtractionStatus.UNSUPPORTED
    assert result.reason == PlanFactExtractionReason.ASSERTION_INCOMPATIBLE


def test_normative_threshold_cannot_fill_silent_plan(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "托撑插入立杆内长度按设计执行。",
        "托撑插入立杆内长度不得小于150mm。",
        NumericOperator.GE,
        "150",
    )
    assert result.status == PlanFactExtractionStatus.NOT_FOUND
    assert result.binding is None


def test_routing_context_number_outside_candidate_has_zero_fact_authority(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "托撑插入立杆内长度按设计执行。",
        "托撑插入立杆内长度不得小于150mm。",
        NumericOperator.GE,
        "150",
        page_text="1、可调托撑伸出长度不得大于200mm。\n托撑插入立杆内长度按设计执行。",
    )
    assert result.status == PlanFactExtractionStatus.NOT_FOUND


def test_wrong_stored_document_sha_fails_closed(tmp_path):
    requirement = _requirement(
        "托撑插入立杆内长度不得小于150mm。", NumericOperator.GE, "150", "mm"
    )
    fixture = _fixture(tmp_path, "托撑插入立杆内长度为160mm。", requirement)
    service, candidate, route, preparation, _, settings, _ = fixture
    (settings.upload_dir / f"{DOCUMENT_ID}.pdf").write_bytes(b"substituted document")
    result = service.extract(
        candidate, route, preparation, requirement_id=requirement.requirement_id
    )
    assert result.status == PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED
    assert result.binding is None


def test_duplicate_text_uses_recorded_candidate_offset(tmp_path):
    text = "建筑高度为90m。"
    page = f"前文{text}后文{text}"
    requirement = _requirement("建筑高度不得大于100m。", NumericOperator.LE, "100", "m")
    fixture = _fixture(tmp_path, text, requirement, page_text=page[page.rfind(text):])
    service, candidate, route, preparation, *_ = fixture
    result = service.extract(
        candidate, route, preparation, requirement_id=requirement.requirement_id
    )
    assert result.status == PlanFactExtractionStatus.EXTRACTED
    assert result.binding.page_char_start == candidate.source_spans[0].char_start


def test_planfact_semantic_tampering_is_rejected(tmp_path):
    result, fixture, _ = _extract(
        tmp_path,
        "建筑高度为90m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    fact = result.binding.plan_fact
    tampered = fact.model_copy(
        update={
            "normalized_value": Decimal("200"),
            "unit": "cm",
            "fact_type": PlanFactType.NUMERIC,
            "parsing_status": PlanFactParsingStatus.PROVEN,
        }
    )
    with pytest.raises(PlanFactError, match="recomputation"):
        fixture[0].plan_fact_service.verify_many(
            fixture[3].review_unit, [tampered]
        )


def test_extraction_identity_is_deterministic_and_recomputed(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "建筑高度为90m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    clone = PlanFactExtractionResult.model_validate(result.model_dump())
    assert clone.extraction_id == result.extraction_id
    with pytest.raises(ValidationError, match="canonical"):
        PlanFactExtractionResult.model_validate(
            {**result.model_dump(), "extraction_id": "planfactextraction_" + "0" * 64}
        )


def test_forged_decomposition_is_not_a_public_authority_parameter():
    parameters = inspect.signature(PlanFactExtractionService.extract).parameters
    assert "decomposition" not in parameters
    assert "routing_context" not in parameters
    assert "page_text" not in parameters


def test_recomputed_d3_identity_mismatch_is_rejected(tmp_path):
    requirement = _requirement("建筑高度不得大于100m。", NumericOperator.LE, "100", "m")
    fixture = _fixture(tmp_path, "建筑高度为90m。", requirement)
    service, candidate, route, preparation, d3, *_ = fixture
    d3.result = d3.result.model_copy(update={"candidate_id": "candidate_" + "0" * 64})
    with pytest.raises(PlanFactExtractionAuthorityError, match="does not match"):
        service.extract(
            candidate, route, preparation, requirement_id=requirement.requirement_id
        )


def test_result_has_no_comparison_or_finding_authority(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "建筑高度为90m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    assert not set(type(result).model_fields).intersection(
        {"decision", "comparison", "finding", "score", "review_action"}
    )


def test_schema_is_frozen_and_forbids_extra(tmp_path):
    result, *_ = _extract(
        tmp_path,
        "建筑高度为90m。",
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    with pytest.raises(ValidationError):
        PlanFactExtractionResult.model_validate({**result.model_dump(), "extra": True})
    with pytest.raises(ValidationError):
        result.status = PlanFactExtractionStatus.NOT_FOUND


def test_source_span_and_hash_are_exact(tmp_path):
    text = "前缀；建筑高度为90m。"
    result, *_ = _extract(
        tmp_path,
        text,
        "建筑高度不得大于100m。",
        NumericOperator.LE,
        "100",
        unit="m",
    )
    assert result.binding.source_text == "建筑高度为90m。"
    assert result.binding.source_text_sha256 == hashlib.sha256(
        result.binding.source_text.encode()
    ).hexdigest()
    assert result.binding.page_char_end - result.binding.page_char_start == len(
        result.binding.source_text
    )


def test_static_isolation_has_no_forbidden_authority():
    source = inspect.getsource(PlanFactExtractionService)
    for forbidden in (
        "ComplianceComparisonResult",
        "ReviewFinding",
        "external LLM",
        "http://",
        "https://",
    ):
        assert forbidden not in source


def _real_candidate(document_id, document_sha, page, start, text):
    span = CandidateSourceSpan(
        page_number=page,
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    candidate_id = deterministic_candidate_id(
        document_sha256=document_sha,
        source_spans=(span,),
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
    )
    return ReviewCandidate(
        candidate_id=candidate_id,
        document_id=document_id,
        document_sha256=document_sha,
        source_spans=(span,),
        source_text=text,
        source_text_sha256=span.source_text_sha256,
        topic="numeric_control",
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
        confidence=1.0,
        status=ReviewCandidateStatus.DISCOVERED,
    )


def _real_route(candidate, document):
    scope = QualifiedStandardScopeCandidate(
        standard_id=document.standard_id,
        standard_code=document.standard_code,
        canonical_standard_code=document.canonical_standard_code,
        standard_name=document.standard_name,
        source_checksum=document.source_checksum,
    )
    route_id = deterministic_route_id(
        candidate_id=candidate.candidate_id,
        scope_candidates=(scope,),
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )
    return StandardRoute(
        route_id=route_id,
        candidate_id=candidate.candidate_id,
        route_status=StandardRouteStatus.ROUTED,
        candidate_topic=candidate.topic,
        candidate_class=candidate.candidate_class,
        scope_candidates=(scope,),
        selected_scope=StandardScope(standard_ids=[document.standard_id]),
        routing_method=StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES,
        routing_confidence=1.0,
        routing_reason="repository-qualified real scope",
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )


def _real_preparation(candidate, route, record):
    unit = ReviewUnitService().create(
        document_id=candidate.document_id,
        page_number=candidate.source_spans[0].page_number,
        source_text=candidate.source_text,
        char_start=candidate.source_spans[0].char_start,
        retrieval_query="repository-qualified normative evidence",
        standard_scope=route.selected_scope,
    )
    evidence = make_hit(
        record,
        rank=1,
        methods=[RetrievalMethod.KEYWORD],
        keyword_score=1.0,
    ).evidence
    binding = ReviewEvidenceBinding(
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
    return ComplianceReviewResult(
        review_unit=unit,
        status=ComplianceReviewStatus.NEEDS_COMPARISON,
        retrieval_decision=RetrievalDecision.ACCEPT,
        evidence_bindings=[binding],
        reason="repository-qualified real evidence",
    )


def _real_extract(*, document_id, document_sha, page, start, text, article_number):
    repository = StandardRepository()
    standard_id = "55023b3b-1b00-4000-8000-000000000001"
    document = repository.load_document(standard_id)
    article = next(
        item
        for item in repository.load_articles(standard_id)
        if item.article_number == article_number
    )
    candidate = _real_candidate(document_id, document_sha, page, start, text)
    route = _real_route(candidate, document)
    preparation = _real_preparation(
        candidate, route, RetrievalRecord(document=document, article=article)
    )
    decomposition = PlanFactExtractionService().decomposition_service.decompose(
        candidate, route, preparation
    )
    requirement = decomposition.requirements[0]
    result = PlanFactExtractionService().extract(
        candidate, route, preparation, requirement_id=requirement.requirement_id
    )
    return result, candidate, requirement


def test_real_case_a_extracts_candidate_local_150mm():
    result, candidate, requirement = _real_extract(
        document_id="f42886ef-83ec-454c-86e2-34d5c238ca0f",
        document_sha="5f3ebde3bd2c0a6d6dccb516c3029fd336558e198ebeccad07d29ca0d03fd8dc",
        page=17,
        start=417,
        text="插入立杆内的长度不得小于150mm。",
        article_number="4.4.15",
    )
    assert candidate.candidate_id == (
        "candidate_697a922f6d4d27fe14911815890b62d035854af5b9870779f23aca13638c12e4"
    )
    assert result.status == PlanFactExtractionStatus.EXTRACTED
    assert result.binding.assertion_kind == PlanAssertionKind.LOWER_BOUND_CONTROL
    assert (result.binding.plan_fact.normalized_value, result.binding.plan_fact.unit) == (
        Decimal("150"),
        "mm",
    )
    assert requirement.value == Decimal("150")
    assert result.binding.source_text == candidate.source_text


def test_real_case_b_extracts_candidate_local_3mm_not_nearby_200mm():
    result, candidate, requirement = _real_extract(
        document_id="f42886ef-83ec-454c-86e2-34d5c238ca0f",
        document_sha="5f3ebde3bd2c0a6d6dccb516c3029fd336558e198ebeccad07d29ca0d03fd8dc",
        page=46,
        start=311,
        text="螺杆外径与立杆钢管内径的间隙不大于3mm，",
        article_number="4.4.16",
    )
    assert candidate.candidate_id == (
        "candidate_ea698a231218480ed14848aa9971a2b70a63ae22600a65a6cc96048849f7f6f1"
    )
    assert result.status == PlanFactExtractionStatus.EXTRACTED
    assert result.binding.assertion_kind == PlanAssertionKind.UPPER_BOUND_CONTROL
    assert (result.binding.plan_fact.normalized_value, result.binding.plan_fact.unit) == (
        Decimal("3"),
        "mm",
    )
    assert requirement.value == Decimal("2.5")
    assert "200mm" not in result.binding.source_text


def test_real_case_c_unrelated_two_spacing_values_are_not_bound():
    result, _, _ = _real_extract(
        document_id="4441b5a1-a167-4e26-9711-d90a8e85f71d",
        document_sha="b0f8c55d2f02b9c826dd1c8001a56582b470bd80ec5d5ad8368235dcb0058462",
        page=23,
        start=322,
        text="1、连墙件采用刚性连接，垂直间距为3.60m，水平间距为4.5m。",
        article_number="4.4.15",
    )
    assert result.status == PlanFactExtractionStatus.NOT_FOUND
    assert result.binding is None
