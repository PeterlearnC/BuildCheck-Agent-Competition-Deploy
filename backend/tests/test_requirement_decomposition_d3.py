import hashlib
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.normative_requirement import NumericOperator
from app.schemas.ocr import OfficialSourceBinding
from app.schemas.requirement_decomposition import (
    RequirementDecompositionStatus,
    UnresolvedRequirementReason,
    deterministic_decomposition_id,
)
from app.schemas.review_candidate import (
    CandidateSourceSpan,
    ReviewCandidate,
    ReviewCandidateClass,
    ReviewCandidateStatus,
    deterministic_candidate_id,
)
from app.schemas.review_unit import ReviewUnit
from app.schemas.standard_route import (
    QualifiedStandardScopeCandidate,
    StandardRoute,
    StandardRouteStatus,
    StandardRoutingMethod,
    deterministic_route_id,
)
from app.schemas.standards import (
    MetadataConfidence,
    StandardDocument,
    StandardIdentityStatus,
)
from app.schemas.standards_retrieval import RetrievalDecision, RetrievalMethod
from app.schemas.standards_scope import StandardScope
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.review.requirement_decomposition_service import (
    RequirementDecompositionAuthorityError,
    RequirementDecompositionService,
)
from app.services.review.requirement_service import RequirementService
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import make_record


PLAN_TEXT = "螺杆长度不得小于20mm。"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000031"
DOCUMENT_SHA = "a" * 64
REGISTRY_FINGERPRINT = "b" * 64


def _repository(tmp_path, article_text: str, *, article_number: str = "1.1.1"):
    settings = Settings(standards_dir=tmp_path / "standards")
    repository = StandardRepository(settings)
    base = make_record(
        "std-a",
        article_number,
        article_text,
        standard_code="GB 10000-2026",
        standard_name="Synthetic qualified standard",
        page_start=3,
    )
    checksum = base.document.source_checksum
    official = OfficialSourceBinding(
        source_checksum=checksum,
        canonical_standard_code="GB10000-2026",
        display_standard_code="GB 10000-2026",
        standard_name="Synthetic qualified standard",
        binding_reason="deterministic test authority",
        binding_provenance="repository fixture",
        confirmed=True,
    )
    payload = base.document.model_dump()
    payload.update(
        standard_code="GB 10000-2026",
        canonical_standard_code="GB10000-2026",
        standard_display_code="GB 10000-2026",
        standard_name="Synthetic qualified standard",
        identity_status=StandardIdentityStatus.CONFIRMED,
        identity_confidence=MetadataConfidence.HIGH,
        official_source_binding=official,
    )
    document = StandardDocument.model_validate(payload)
    repository.save_document(document)
    repository.save_articles(document.standard_id, [base.article])
    return repository, RetrievalRecord(document=document, article=base.article)


def _candidate(source_text: str = PLAN_TEXT, *, document_sha: str = DOCUMENT_SHA):
    span = CandidateSourceSpan(
        page_number=1,
        char_start=0,
        char_end=len(source_text),
        source_text=source_text,
        source_text_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
    )
    candidate_id = deterministic_candidate_id(
        document_sha256=document_sha,
        source_spans=[span],
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
    )
    return ReviewCandidate(
        candidate_id=candidate_id,
        document_id=DOCUMENT_ID,
        document_sha256=document_sha,
        source_spans=(span,),
        source_text=source_text,
        source_text_sha256=span.source_text_sha256,
        topic="numeric_control",
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
        confidence=1.0,
        status=ReviewCandidateStatus.DISCOVERED,
    )


def _route(candidate: ReviewCandidate, document, *, fingerprint=REGISTRY_FINGERPRINT):
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
        registry_fingerprint=fingerprint,
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
        routing_reason="qualified deterministic test scope",
        registry_fingerprint=fingerprint,
    )


def _unit(candidate: ReviewCandidate, standard_id="std-a"):
    span = candidate.source_spans[0]
    return ReviewUnit(
        review_unit_id="reviewunit_d3",
        document_id=candidate.document_id,
        page_number=span.page_number,
        source_text=candidate.source_text,
        source_text_sha256=candidate.source_text_sha256,
        char_start=span.char_start,
        char_end=span.char_end,
        review_text=candidate.source_text,
        retrieval_query="qualified normative requirement",
        standard_scope=StandardScope(standard_ids=[standard_id]),
    )


def _binding(unit: ReviewUnit, record: RetrievalRecord):
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


def _preparation(candidate, record, *, binding=None):
    unit = _unit(candidate, record.document.standard_id)
    return ComplianceReviewResult(
        review_unit=unit,
        status=ComplianceReviewStatus.NEEDS_COMPARISON,
        retrieval_decision=RetrievalDecision.ACCEPT,
        evidence_bindings=[binding or _binding(unit, record)],
        reason="accepted repository evidence",
    )


def _decompose(tmp_path, article_text, *, article_number="1.1.1"):
    repository, record = _repository(
        tmp_path, article_text, article_number=article_number
    )
    candidate = _candidate()
    route = _route(candidate, record.document)
    preparation = _preparation(candidate, record)
    result = RequirementDecompositionService(repository).decompose(
        candidate, route, preparation
    )
    return result, repository, record, candidate, route, preparation


def test_single_numeric_minimum_is_verified(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "插入长度不得小于150mm。")
    requirement = result.requirements[0]
    assert result.status == RequirementDecompositionStatus.DECOMPOSED
    assert requirement.operator == NumericOperator.GE
    assert requirement.value == Decimal("150")
    assert requirement.unit == "mm"


def test_single_numeric_maximum_is_verified(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "间隙不得大于2.5mm。")
    requirement = result.requirements[0]
    assert requirement.operator == NumericOperator.LE
    assert requirement.value == Decimal("2.5")
    assert requirement.unit == "mm"


def test_two_semicolon_requirements_are_ordered(tmp_path) -> None:
    result, *_ = _decompose(
        tmp_path, "插入长度不得小于150mm；伸出长度不得大于200mm。"
    )
    assert result.status == RequirementDecompositionStatus.DECOMPOSED
    assert [item.value for item in result.requirements] == [
        Decimal("150"),
        Decimal("200"),
    ]
    assert result.requirements[0].requirement_char_start < result.requirements[1].requirement_char_start


def test_proven_plus_qualitative_is_partial(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "长度不得小于150mm；伸出长度应经计算确定。")
    assert result.status == RequirementDecompositionStatus.PARTIAL
    assert len(result.requirements) == 1
    assert result.unresolved_spans[0].reason == (
        UnresolvedRequirementReason.QUALITATIVE_UNSUPPORTED
    )


@pytest.mark.parametrize(
    "text",
    [
        "当钢管直径为42mm时，伸出长度不得大于200mm。",
        "在施工荷载超过设计值情况下，支架高度不得大于10m。",
        "对于临时支架，间距不得大于1m。",
    ],
)
def test_conditional_numeric_clause_is_unresolved(tmp_path, text) -> None:
    result, *_ = _decompose(tmp_path, text)
    assert result.status == RequirementDecompositionStatus.UNRESOLVED
    assert result.unresolved_spans[0].reason == (
        UnresolvedRequirementReason.CONDITIONAL_UNSUPPORTED
    )


@pytest.mark.parametrize(
    "text",
    ["除特殊部位外，间距不得大于1m。", "施工过程中，间距不得大于1m。"],
)
def test_exception_or_scope_is_unresolved(tmp_path, text) -> None:
    result, *_ = _decompose(tmp_path, text)
    assert result.status == RequirementDecompositionStatus.UNRESOLVED
    assert result.unresolved_spans[0].reason == (
        UnresolvedRequirementReason.EXCEPTION_SCOPE_UNRESOLVED
    )


@pytest.mark.parametrize(
    "text",
    [
        "在端部，间距不得大于1m。",
        "在顶部，间距不得大于1m。",
        "在节点处，间距不得大于1m。",
        "于端部，间距不得大于1m。",
        "在该区域内，间距不得大于1m。",
        "在连接部位，间距不得大于1m。",
    ],
)
def test_leading_location_qualifier_is_unresolved(tmp_path, text) -> None:
    result, _, record, *_ = _decompose(tmp_path, text)
    assert result.status == RequirementDecompositionStatus.UNRESOLVED
    assert not result.requirements
    assert len(result.unresolved_spans) == 1
    unresolved = result.unresolved_spans[0]
    assert unresolved.reason == (
        UnresolvedRequirementReason.EXCEPTION_SCOPE_UNRESOLVED
    )
    assert unresolved.source_text == record.article.source_text
    assert record.article.source_text[
        unresolved.char_start : unresolved.char_end
    ] == unresolved.source_text


def test_unconditional_counterpart_remains_decomposed(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "间距不得大于1m。")
    requirement = result.requirements[0]
    assert result.status == RequirementDecompositionStatus.DECOMPOSED
    assert (requirement.operator, requirement.value, requirement.unit) == (
        NumericOperator.LE,
        Decimal("1"),
        "m",
    )


@pytest.mark.parametrize(
    "text",
    [
        "在役设备间距不得大于1m。",
        "位于端部的构件间距不得大于1m。",
    ],
)
def test_lexical_zai_or_yu_without_qualifier_boundary_is_not_overmatched(
    tmp_path, text
) -> None:
    result, *_ = _decompose(tmp_path, text)
    assert result.status == RequirementDecompositionStatus.DECOMPOSED
    assert result.requirements[0].operator == NumericOperator.LE


def test_ambiguous_non_obligation_is_unresolved(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "结构说明文字。")
    assert result.status == RequirementDecompositionStatus.UNRESOLVED
    assert result.unresolved_spans[0].reason == UnresolvedRequirementReason.AMBIGUOUS_SPLIT


def test_guarded_conjunction_requires_independent_subject(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "立杆高度应核定，且不得大于10m。")
    assert result.status == RequirementDecompositionStatus.UNRESOLVED
    assert len(result.unresolved_spans) == 1


def test_guarded_conjunction_with_independent_subject_can_split(tmp_path) -> None:
    result, *_ = _decompose(
        tmp_path, "插入长度不得小于150mm，且伸出长度不得大于200mm。"
    )
    assert result.status == RequirementDecompositionStatus.DECOMPOSED
    assert len(result.requirements) == 2


def test_exact_requirement_and_unresolved_spans_match_repository(tmp_path) -> None:
    result, _, record, *_ = _decompose(tmp_path, "长度不得小于150mm；应进行验收。")
    source = record.article.source_text
    for item in (*result.requirements, *result.unresolved_spans):
        start = getattr(item, "requirement_char_start", getattr(item, "char_start", None))
        end = getattr(item, "requirement_char_end", getattr(item, "char_end", None))
        text = getattr(item, "requirement_text", getattr(item, "source_text", None))
        assert source[start:end] == text


def test_same_authority_has_same_identity(tmp_path) -> None:
    result, repository, record, candidate, route, preparation = _decompose(
        tmp_path, "长度不得小于150mm。"
    )
    repeated = RequirementDecompositionService(repository).decompose(
        candidate, route, preparation
    )
    assert result == repeated


def test_changed_candidate_changes_identity(tmp_path) -> None:
    result, repository, record, _, _, _ = _decompose(tmp_path, "长度不得小于150mm。")
    candidate = _candidate(document_sha="c" * 64)
    route = _route(candidate, record.document)
    changed = RequirementDecompositionService(repository).decompose(
        candidate, route, _preparation(candidate, record)
    )
    assert changed.decomposition_id != result.decomposition_id


def test_changed_route_changes_identity(tmp_path) -> None:
    result, repository, record, candidate, _, preparation = _decompose(
        tmp_path, "长度不得小于150mm。"
    )
    changed_route = _route(candidate, record.document, fingerprint="c" * 64)
    changed = RequirementDecompositionService(repository).decompose(
        candidate, changed_route, preparation
    )
    assert changed.decomposition_id != result.decomposition_id


def test_changed_threshold_changes_requirement_and_decomposition_identity(tmp_path) -> None:
    first, *_ = _decompose(tmp_path / "first", "长度不得小于150mm。")
    second, *_ = _decompose(tmp_path / "second", "长度不得小于151mm。")
    assert first.requirements[0].requirement_id != second.requirements[0].requirement_id
    assert first.decomposition_id != second.decomposition_id


def test_changed_evidence_changes_decomposition_identity(tmp_path) -> None:
    first, *_ = _decompose(
        tmp_path / "first", "长度不得小于150mm。", article_number="1.1.1"
    )
    second, *_ = _decompose(
        tmp_path / "second", "长度不得小于150mm。", article_number="1.1.2"
    )
    assert first.evidence_ids != second.evidence_ids
    assert first.decomposition_id != second.decomposition_id


def test_document_timestamps_do_not_affect_identity(tmp_path) -> None:
    first, *_ = _decompose(tmp_path / "first", "长度不得小于150mm。")
    second, *_ = _decompose(tmp_path / "second", "长度不得小于150mm。")
    assert first.decomposition_id == second.decomposition_id


def test_requirement_order_is_identity_bearing(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "长度不得小于150mm；长度不得大于200mm。")
    reversed_id = deterministic_decomposition_id(
        candidate_id=result.candidate_id,
        route_id=result.route_id,
        evidence_ids=result.evidence_ids,
        requirements=tuple(reversed(result.requirements)),
        unresolved_spans=result.unresolved_spans,
    )
    assert reversed_id != result.decomposition_id


def test_unresolved_reason_is_identity_bearing(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "应进行验收。")
    span = result.unresolved_spans[0]
    payload = span.model_dump()
    payload["reason"] = UnresolvedRequirementReason.AMBIGUOUS_SPLIT
    with pytest.raises(ValidationError, match="unresolved_span_id"):
        type(span).model_validate(payload)


def test_result_and_unresolved_span_reject_extra_fields(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "应进行验收。")
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(result).model_validate({**result.model_dump(), "score": 1})
    span = result.unresolved_spans[0]
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(span).model_validate({**span.model_dump(), "decision": "COMPLIANT"})


def test_off_by_one_unresolved_span_is_rejected(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "应进行验收。")
    span = result.unresolved_spans[0]
    with pytest.raises(ValidationError, match="offsets"):
        type(span).model_validate({**span.model_dump(), "char_end": span.char_end + 1})


def test_same_text_at_two_offsets_keeps_distinct_identity(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "间距不得大于1m；间距不得大于1m。")
    assert len(result.requirements) == 2
    assert result.requirements[0].requirement_id != result.requirements[1].requirement_id


def test_candidate_route_mismatch_is_rejected(tmp_path) -> None:
    _, repository, record, candidate, _, preparation = _decompose(
        tmp_path, "长度不得小于150mm。"
    )
    other = _candidate(document_sha="c" * 64)
    with pytest.raises(RequirementDecompositionAuthorityError, match="StandardRoute"):
        RequirementDecompositionService(repository).decompose(
            other, _route(candidate, record.document), preparation
        )


def test_candidate_preparation_mismatch_is_rejected(tmp_path) -> None:
    _, repository, record, candidate, route, _ = _decompose(
        tmp_path, "长度不得小于150mm。"
    )
    other = _candidate("另一候选文本。")
    with pytest.raises(RequirementDecompositionAuthorityError, match="preparation"):
        RequirementDecompositionService(repository).decompose(
            candidate, route, _preparation(other, record)
        )


@pytest.mark.parametrize("decision", [RetrievalDecision.LOW_CONFIDENCE, RetrievalDecision.NO_MATCH])
def test_non_accept_retrieval_returns_source_not_qualified(tmp_path, decision) -> None:
    repository, record = _repository(tmp_path, "长度不得小于150mm。")
    candidate = _candidate()
    route = _route(candidate, record.document)
    preparation = ComplianceReviewResult(
        review_unit=_unit(candidate),
        status=ComplianceReviewStatus.INSUFFICIENT_EVIDENCE,
        retrieval_decision=decision,
        evidence_bindings=[],
        reason="not accepted",
    )
    result = RequirementDecompositionService(repository).decompose(
        candidate, route, preparation
    )
    assert result.status == RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED
    assert not result.requirements and not result.evidence_ids


def test_missing_or_unqualified_standard_is_rejected(tmp_path) -> None:
    repository, record = _repository(tmp_path, "长度不得小于150mm。")
    candidate = _candidate()
    route = _route(candidate, record.document)
    repository.save_document(
        record.document.model_copy(
            update={"identity_status": StandardIdentityStatus.IDENTITY_UNCERTAIN}
        )
    )
    with pytest.raises(RequirementDecompositionAuthorityError, match="qualified"):
        RequirementDecompositionService(repository).decompose(
            candidate, route, _preparation(candidate, record)
        )


@pytest.mark.parametrize("field", ["source_text", "article_id", "standard_id"])
def test_fabricated_evidence_is_rejected(tmp_path, field) -> None:
    _, repository, record, candidate, route, preparation = _decompose(
        tmp_path, "长度不得小于150mm。"
    )
    binding = preparation.evidence_bindings[0]
    evidence_update = {
        "source_text": binding.evidence.source_text + "伪造",
        "article_id": "fake-article",
        "standard_id": "fake-standard",
    }[field]
    evidence = binding.evidence.model_copy(update={field: evidence_update})
    altered = binding.model_copy(
        update={
            "evidence": evidence,
            "standard_evidence_id": evidence.id,
            "standard_id": evidence.standard_id,
            "standard_article_number": evidence.article_number,
        }
    )
    forged = preparation.model_copy(update={"evidence_bindings": [altered]})
    with pytest.raises(RequirementDecompositionAuthorityError):
        RequirementDecompositionService(repository).decompose(candidate, route, forged)


def test_plan_threshold_has_no_normative_authority(tmp_path) -> None:
    repository, record = _repository(tmp_path, "构造应经计算确定。")
    candidate = _candidate("按GB55023第4.4.15条，最小150mm")
    result = RequirementDecompositionService(repository).decompose(
        candidate, _route(candidate, record.document), _preparation(candidate, record)
    )
    assert not result.requirements


def test_context_route_threshold_has_no_normative_authority(tmp_path) -> None:
    repository, record = _repository(tmp_path, "构造应经计算确定。")
    candidate = _candidate("螺杆不得小于200mm")
    exact = _route(candidate, record.document)
    payload = exact.model_dump()
    payload.update(
        routing_method=StandardRoutingMethod.CONTEXT_ASSISTED_DETERMINISTIC,
        routing_context_id="routingcontext_" + "d" * 64,
    )
    payload["route_id"] = deterministic_route_id(
        candidate_id=candidate.candidate_id,
        scope_candidates=exact.scope_candidates,
        registry_fingerprint=exact.registry_fingerprint,
        routing_context_id=payload["routing_context_id"],
    )
    route = StandardRoute.model_validate(payload)
    result = RequirementDecompositionService(repository).decompose(
        candidate, route, _preparation(candidate, record)
    )
    assert not result.requirements


def test_no_comparison_or_finding_fields_exist(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "长度不得小于150mm。")
    fields = set(type(result).model_fields)
    assert not fields.intersection(
        {"decision", "comparison", "finding", "score", "review_action"}
    )


def _real_case(article_number: str):
    repository = StandardRepository()
    standard_id = "55023b3b-1b00-4000-8000-000000000001"
    document = repository.load_document(standard_id)
    article = next(
        item
        for item in repository.load_articles(standard_id)
        if item.article_number == article_number
    )
    record = RetrievalRecord(document=document, article=article)
    candidate = _candidate()
    route = _route(candidate, document)
    result = RequirementDecompositionService(repository).decompose(
        candidate, route, _preparation(candidate, record)
    )
    return result, record


def test_real_article_4415_is_partial_and_preserves_conditions() -> None:
    result, record = _real_case("4.4.15")
    assert result.status == RequirementDecompositionStatus.PARTIAL
    assert [(item.operator, item.value, item.unit) for item in result.requirements] == [
        (NumericOperator.GE, Decimal("150"), "mm")
    ]
    assert len(result.unresolved_spans) == 3
    assert sum(
        item.reason == UnresolvedRequirementReason.CONDITIONAL_UNSUPPORTED
        for item in result.unresolved_spans
    ) == 2
    assert all(item.source_text in record.article.source_text for item in result.unresolved_spans)


def test_real_article_4416_is_decomposed() -> None:
    result, _ = _real_case("4.4.16")
    assert result.status == RequirementDecompositionStatus.DECOMPOSED
    requirement = result.requirements[0]
    assert (requirement.operator, requirement.value, requirement.unit) == (
        NumericOperator.LE,
        Decimal("2.5"),
        "mm",
    )


def test_case_a_requirement_remains_verifiable_by_frozen_service() -> None:
    result, record = _real_case("4.4.15")
    candidate = _candidate()
    binding = _preparation(candidate, record).evidence_bindings[0]
    assert RequirementService().verify(result.requirements[0], binding) == result.requirements[0]


def test_case_b_requirement_remains_verifiable_by_frozen_service() -> None:
    result, record = _real_case("4.4.16")
    candidate = _candidate()
    binding = _preparation(candidate, record).evidence_bindings[0]
    assert RequirementService().verify(result.requirements[0], binding) == result.requirements[0]


def test_case_c_condition_remains_fail_closed(tmp_path) -> None:
    result, *_ = _decompose(
        tmp_path, "当钢管直径为42mm时，伸出长度不得大于200mm。"
    )
    assert result.status == RequirementDecompositionStatus.UNRESOLVED
    assert not result.requirements


def test_units_are_source_preserved_without_conversion(tmp_path) -> None:
    result, *_ = _decompose(tmp_path, "插入长度不得小于150mm。")
    requirement = result.requirements[0]
    assert requirement.value == Decimal("150")
    assert requirement.unit == "mm"
    assert not hasattr(requirement, "converted_value")


def test_semantic_tampering_remains_rejected_by_frozen_requirement_service(
    tmp_path,
) -> None:
    result, _, _, candidate, _, preparation = _decompose(
        tmp_path, "长度不得小于150mm。"
    )
    requirement = result.requirements[0]
    tampered = requirement.model_copy(
        update={
            "operator": NumericOperator.LE,
            "value": Decimal("200"),
            "unit": "cm",
        }
    )
    with pytest.raises(ValueError, match="recomputation"):
        RequirementService().verify(tampered, preparation.evidence_bindings[0])
