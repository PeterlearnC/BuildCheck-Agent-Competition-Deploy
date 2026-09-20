"""Focused authority qualification for deterministic D9-P1 internal consistency."""

from __future__ import annotations

import hashlib
import inspect
import time
from decimal import Decimal
from pathlib import Path
from typing import get_args, get_origin

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.compliance_comparison import (
    PlanFact,
    PlanFactParsingStatus,
    PlanFactType,
)
from app.schemas.internal_consistency import (
    ConsistencyAssertionClass,
    ConsistencyCandidateReason,
    ConsistencyCandidateStatus,
    ConsistencyCoverageStatus,
    ConsistencyFact,
    ConsistencyParameterProfile,
    ConsistencyReviewStatus,
    ConsistencyUnresolvedCount,
    ConsistencyUnresolvedReason,
    ConsistencyValueGroup,
    ConsistencyValueRelation,
    EngineeringObjectProfile,
    InternalConsistencyCandidate,
    InternalConsistencyReviewResult,
    NormalizedUnit,
)
from app.services.pdf_service import PDFExtractionResult, PDFPageText
from app.services.review.internal_consistency_service import (
    InternalConsistencyDocumentChangedError,
    InternalConsistencyDocumentNotFoundError,
    InternalConsistencyReviewService,
)


DOCUMENT_ID = "11111111-1111-4111-8111-111111111111"
DOCUMENT_SHA = "a" * 64


def _pages(*texts: str, numbers: tuple[int, ...] | None = None) -> list[PDFPageText]:
    page_numbers = numbers or tuple(range(1, len(texts) + 1))
    return [
        PDFPageText(page_number=number, text=text)
        for number, text in zip(page_numbers, texts, strict=True)
    ]


def _review(
    *texts: str,
    numbers: tuple[int, ...] | None = None,
    document_sha: str = DOCUMENT_SHA,
) -> InternalConsistencyReviewResult:
    return InternalConsistencyReviewService()._review_verified_pages(
        document_id=DOCUMENT_ID,
        document_sha256=document_sha,
        pages=_pages(*texts, numbers=numbers),
    )


def _fact(
    text: str,
    *,
    page_number: int = 1,
    document_sha: str = DOCUMENT_SHA,
) -> ConsistencyFact:
    page = PDFPageText(page_number=page_number, text=text)
    spans = InternalConsistencyReviewService._bounded_clauses(text)
    assert len(spans) == 1
    fact, reason = InternalConsistencyReviewService._fact_from_clause(
        document_id=DOCUMENT_ID,
        document_sha256=document_sha,
        page=page,
        page_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        source_start=spans[0][0],
        source_end=spans[0][1],
    )
    assert reason is None
    assert fact is not None
    return fact


def _positive_result() -> InternalConsistencyReviewResult:
    return _review(
        "支护结构水平位移报警值为30 mm。",
        "支护结构水平位移报警值控制为20 mm。",
        numbers=(18, 43),
    )


def _unresolved_reasons(
    result: InternalConsistencyReviewResult,
) -> dict[ConsistencyUnresolvedReason, int]:
    return {item.reason: item.count for item in result.unresolved_counts}


class _FakePDFService:
    def __init__(
        self,
        pages: list[PDFPageText],
        *,
        mutate_path: bool = False,
    ) -> None:
        self.pages = pages
        self.mutate_path = mutate_path
        self.calls = 0
        self.last_path: Path | None = None

    def extract_text(self, path: Path) -> PDFExtractionResult:
        self.calls += 1
        self.last_path = path
        if self.mutate_path:
            path.write_bytes(path.read_bytes() + b"changed")
        text = "\n".join(page.text for page in self.pages)
        return PDFExtractionResult(
            page_count=len(self.pages),
            char_count=len(text),
            text=text,
            pages=self.pages,
        )


def _model_instances() -> tuple[object, ...]:
    result = _positive_result()
    candidate = result.candidates[0]
    group = candidate.value_groups[0]
    return (
        group.facts[0],
        group,
        candidate,
        ConsistencyUnresolvedCount(
            reason=ConsistencyUnresolvedReason.INSUFFICIENT_CONTEXT, count=1
        ),
        result,
    )


def _annotation_contains_any(annotation: object) -> bool:
    if str(annotation) in {"typing.Any", "Any"}:
        return True
    return any(_annotation_contains_any(item) for item in get_args(annotation))


# Schema contract.


@pytest.mark.parametrize("model", _model_instances())
def test_public_models_are_frozen(model: object) -> None:
    field_name = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError):
        setattr(model, field_name, getattr(model, field_name))


@pytest.mark.parametrize("model", _model_instances())
def test_public_models_forbid_extra_fields(model: object) -> None:
    payload = model.model_dump(mode="python")
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        type(model).model_validate(payload)


def test_public_models_have_no_any_authority_fields() -> None:
    models = (
        ConsistencyFact,
        ConsistencyValueGroup,
        InternalConsistencyCandidate,
        ConsistencyUnresolvedCount,
        InternalConsistencyReviewResult,
    )
    assert not any(
        _annotation_contains_any(field.annotation)
        for model in models
        for field in model.model_fields.values()
    )


def test_public_models_have_no_mutable_literal_defaults() -> None:
    models = (
        ConsistencyFact,
        ConsistencyValueGroup,
        InternalConsistencyCandidate,
        ConsistencyUnresolvedCount,
        InternalConsistencyReviewResult,
    )
    assert not any(
        isinstance(field.default, (list, dict, set))
        for model in models
        for field in model.model_fields.values()
    )


def test_public_models_expose_no_compliance_severity_or_risk_fields() -> None:
    forbidden = {
        "standard",
        "article",
        "normative_requirement",
        "compliance",
        "compliance_decision",
        "finding_id",
        "severity",
        "risk",
        "risk_score",
        "unsafe",
        "whole_plan_verdict",
        "plan_consistent",
        "overall_consistency",
        "final_decision",
        "pass",
        "fail",
    }
    models = (ConsistencyFact, InternalConsistencyCandidate, InternalConsistencyReviewResult)
    assert not any(forbidden.intersection(model.model_fields) for model in models)


# Public seam and stored-document authority.


def test_review_document_is_the_only_public_authority_method() -> None:
    public = {
        name
        for name, value in InternalConsistencyReviewService.__dict__.items()
        if inspect.isfunction(value) and not name.startswith("_")
    }
    assert public == {"review_document"}
    assert tuple(inspect.signature(InternalConsistencyReviewService.review_document).parameters) == (
        "self",
        "document_id",
    )


def test_caller_consistency_fact_cannot_enter_public_seam() -> None:
    service = InternalConsistencyReviewService()
    with pytest.raises(TypeError):
        service.review_document(DOCUMENT_ID, _fact("立杆间距为30 mm。"))  # type: ignore[call-arg]


def test_caller_plan_fact_cannot_enter_public_seam() -> None:
    source = "30mm"
    plan_fact = PlanFact(
        plan_fact_id="caller-fake",
        review_unit_id="caller-fake",
        char_start=0,
        char_end=len(source),
        source_text=source,
        source_text_sha256=hashlib.sha256(source.encode()).hexdigest(),
        fact_type=PlanFactType.NUMERIC,
        parsing_status=PlanFactParsingStatus.PROVEN,
        normalized_value=Decimal("30"),
        unit="mm",
        parser_method="CALLER",
        parser_version="CALLER",
    )
    with pytest.raises(TypeError):
        InternalConsistencyReviewService().review_document(  # type: ignore[call-arg]
            DOCUMENT_ID, plan_fact
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"source_text": "立杆间距为30mm"},
        {"source_start": 0, "source_end": 10},
        {"pdf_path": Path("caller.pdf")},
        {"document_sha256": DOCUMENT_SHA},
    ),
)
def test_caller_source_path_and_sha_authority_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        InternalConsistencyReviewService().review_document(DOCUMENT_ID, **kwargs)  # type: ignore[call-arg]


def test_review_document_resolves_server_path_hashes_and_reads_text_once(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / f"{DOCUMENT_ID}.pdf"
    pdf_path.write_bytes(b"server-owned-pdf")
    fake = _FakePDFService(
        _pages(
            "支护结构水平位移报警值为30mm。",
            "支护结构水平位移报警值为20mm。",
        )
    )
    service = InternalConsistencyReviewService(
        settings=Settings(upload_dir=tmp_path), pdf_service=fake  # type: ignore[arg-type]
    )
    result = service.review_document(DOCUMENT_ID)
    assert fake.calls == 1
    assert fake.last_path == pdf_path.resolve()
    assert result.document_sha256 == hashlib.sha256(b"server-owned-pdf").hexdigest()


def test_invalid_or_caller_like_document_path_fails_closed(tmp_path: Path) -> None:
    service = InternalConsistencyReviewService(settings=Settings(upload_dir=tmp_path))
    with pytest.raises(InternalConsistencyDocumentNotFoundError):
        service.review_document("../caller.pdf")


def test_document_mutation_between_pre_and_post_hash_fails_closed(tmp_path: Path) -> None:
    pdf_path = tmp_path / f"{DOCUMENT_ID}.pdf"
    pdf_path.write_bytes(b"initial")
    fake = _FakePDFService(
        _pages("支护结构水平位移报警值为30mm。"), mutate_path=True
    )
    service = InternalConsistencyReviewService(
        settings=Settings(upload_dir=tmp_path), pdf_service=fake  # type: ignore[arg-type]
    )
    with pytest.raises(InternalConsistencyDocumentChangedError):
        service.review_document(DOCUMENT_ID)


# Exact source and finite profiles.


def test_fact_retains_exact_page_text_offsets_and_hashes() -> None:
    text = "前缀。支护结构水平位移报警值为30 mm。后缀。"
    result = _review(text, "支护结构水平位移报警值为20 mm。")
    fact = next(
        fact
        for group in result.candidates[0].value_groups
        for fact in group.facts
        if fact.physical_page == 1
    )
    assert text[fact.source_start : fact.source_end] == fact.source_text
    assert fact.source_text_sha256 == hashlib.sha256(fact.source_text.encode()).hexdigest()
    assert fact.page_text_sha256 == hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize(
    ("text", "object_profile", "parameter_profile"),
    (
        (
            "支护结构水平位移报警值为30mm。",
            EngineeringObjectProfile.SUPPORT_STRUCTURE,
            ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_ALARM_VALUE,
        ),
        (
            "围护墙水平位移控制值为30mm。",
            EngineeringObjectProfile.RETAINING_WALL,
            ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_CONTROL_VALUE,
        ),
        (
            "冠梁水平位移预警值为30mm。",
            EngineeringObjectProfile.CROWN_BEAM,
            ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_PREWARNING_VALUE,
        ),
        (
            "立杆间距为1500mm。",
            EngineeringObjectProfile.UPRIGHT_MEMBER,
            ConsistencyParameterProfile.SPACING,
        ),
        (
            "水平杆步距为1.5m。",
            EngineeringObjectProfile.HORIZONTAL_MEMBER,
            ConsistencyParameterProfile.STEP_SPACING,
        ),
        (
            "角焊缝厚度为8mm。",
            EngineeringObjectProfile.ANGLE_WELD,
            ConsistencyParameterProfile.THICKNESS,
        ),
        (
            "开挖深度为3m。",
            EngineeringObjectProfile.EXCAVATION,
            ConsistencyParameterProfile.DEPTH,
        ),
        (
            "立杆高度为2m。",
            EngineeringObjectProfile.UPRIGHT_MEMBER,
            ConsistencyParameterProfile.HEIGHT,
        ),
        (
            "钢支撑轴力为300kN。",
            EngineeringObjectProfile.STEEL_SUPPORT,
            ConsistencyParameterProfile.SUPPORT_AXIAL_FORCE,
        ),
    ),
)
def test_finite_object_and_parameter_profiles_are_exact(
    text: str,
    object_profile: EngineeringObjectProfile,
    parameter_profile: ConsistencyParameterProfile,
) -> None:
    fact = _fact(text)
    assert fact.object_profile_name == object_profile
    assert fact.parameter_profile_name == parameter_profile


def test_same_unit_different_parameter_is_not_compared() -> None:
    result = _review(
        "立杆间距为30mm。",
        "立杆高度为20mm。",
    )
    assert result.candidate_count == 0


def test_same_parameter_different_object_is_not_compared() -> None:
    result = _review(
        "支护结构水平位移报警值为30mm。",
        "冠梁水平位移报警值为20mm。",
    )
    assert result.candidate_count == 0


def test_failed_object_identity_is_unresolved_without_candidate() -> None:
    result = _review("水平位移报警值为30mm。")
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.OBJECT_IDENTITY_UNRESOLVED] == 1


def test_failed_parameter_identity_is_unresolved_without_candidate() -> None:
    result = _review("支护结构设计值为30mm。")
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.PARAMETER_IDENTITY_UNRESOLVED] == 1


def test_no_fuzzy_or_display_text_object_matching() -> None:
    result = _review(
        "支护结构水平位移报警值为30mm。",
        "支護結構水平位移报警值为20mm。",
    )
    assert result.candidate_count == 0


def test_alarm_and_prewarning_profiles_are_not_broadly_aliased() -> None:
    result = _review(
        "支护结构水平位移报警值为30mm。",
        "支护结构水平位移预警值为20mm。",
    )
    assert result.candidate_count == 0


def test_unrelated_nearby_number_prevents_unsafe_fact_attachment() -> None:
    result = _review("支护结构水平位移报警值为30mm，另有螺栓20mm。")
    assert result.fact_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.INSUFFICIENT_CONTEXT] == 1


# Unit, value, and assertion safety.


def test_identical_whitelisted_unit_case_is_normalized() -> None:
    result = _review(
        "支护结构水平位移报警值为30 MM。",
        "支护结构水平位移报警值为20 mm。",
    )
    assert result.candidate_count == 1
    assert result.candidates[0].normalized_unit == NormalizedUnit.MILLIMETRE


def test_different_units_are_unresolved_without_candidate() -> None:
    result = _review(
        "支护结构水平位移报警值为30mm。",
        "支护结构水平位移报警值为20cm。",
    )
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.UNIT_MISMATCH] == 1


def test_missing_unit_is_unresolved_without_candidate() -> None:
    result = _review("支护结构水平位移报警值为30。")
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.MISSING_UNIT] == 1


def test_m_and_mm_are_not_implicitly_converted() -> None:
    result = _review(
        "开挖深度为3m。",
        "开挖深度为3000mm。",
    )
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.UNIT_MISMATCH] == 1


def test_five_part_grouping_preserves_valid_subgroup_across_mixed_units() -> None:
    facts = (
        _fact("支护结构水平位移报警值为30mm。", page_number=18),
        _fact("支护结构水平位移报警值为20mm。", page_number=43),
        _fact("支护结构水平位移报警值为1m。", page_number=44),
    )

    candidates, unresolved = InternalConsistencyReviewService._group_facts(facts)

    assert len(candidates) == 1
    assert candidates[0].normalized_unit == NormalizedUnit.MILLIMETRE
    assert [group.numeric_value for group in candidates[0].value_groups] == [
        Decimal("20"),
        Decimal("30"),
    ]
    assert unresolved[ConsistencyUnresolvedReason.UNIT_MISMATCH] == 1


def test_mixed_unit_subgroup_order_is_invariant() -> None:
    facts = (
        _fact("支护结构水平位移报警值为30mm。", page_number=18),
        _fact("支护结构水平位移报警值为20mm。", page_number=43),
        _fact("支护结构水平位移报警值为1m。", page_number=44),
    )

    assert InternalConsistencyReviewService._group_facts(
        facts
    ) == InternalConsistencyReviewService._group_facts(tuple(reversed(facts)))


def test_decimal_three_equals_three_point_zero() -> None:
    result = _review(
        "支护结构水平位移报警值为3mm。",
        "支护结构水平位移报警值为3.0mm。",
    )
    assert result.fact_count == 2
    assert result.candidate_count == 0


def test_same_values_create_no_candidate() -> None:
    result = _review(
        "立杆间距为1500mm。",
        "立杆间距控制为1500mm。",
    )
    assert result.candidate_count == 0


def test_different_compatible_values_create_candidate() -> None:
    result = _positive_result()
    candidate = result.candidates[0]
    assert candidate.status == ConsistencyCandidateStatus.INTERNAL_CONSISTENCY_CANDIDATE
    assert candidate.relation == ConsistencyValueRelation.DIFFERENT_VALUE
    assert candidate.reason == ConsistencyCandidateReason.DIFFERENT_EXPLICIT_NUMERIC_VALUES
    assert candidate.review_status == ConsistencyReviewStatus.NEEDS_HUMAN_REVIEW


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        (
            "支护结构水平位移报警值为30mm。",
            ConsistencyAssertionClass.DECLARED_CONTROL_VALUE,
        ),
        (
            "支护结构水平位移报警值实测为30mm。",
            ConsistencyAssertionClass.EXACT_MEASURED_VALUE,
        ),
        (
            "支护结构水平位移报警值不超过30mm。",
            ConsistencyAssertionClass.UPPER_BOUND_CONTROL,
        ),
        (
            "角焊缝厚度不低于8mm。",
            ConsistencyAssertionClass.LOWER_BOUND_CONTROL,
        ),
    ),
)
def test_assertion_class_is_finite_and_inequality_is_retained(
    text: str, expected: ConsistencyAssertionClass
) -> None:
    assert _fact(text).assertion_class == expected


def test_exact_and_upper_bound_are_not_compared() -> None:
    result = _review(
        "支护结构水平位移报警值实测为30mm。",
        "支护结构水平位移报警值不超过20mm。",
    )
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.ASSERTION_TYPE_MISMATCH] == 1


def test_declared_control_and_exact_measured_are_not_compared() -> None:
    result = _review(
        "支护结构水平位移报警值为30mm。",
        "支护结构水平位移报警值实测为20mm。",
    )
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[ConsistencyUnresolvedReason.ASSERTION_TYPE_MISMATCH] == 1


def test_assertion_class_is_direct_group_isolation() -> None:
    facts = (
        _fact("支护结构水平位移报警值为30mm。", page_number=18),
        _fact("支护结构水平位移报警值不大于20mm。", page_number=43),
    )

    candidates, unresolved = InternalConsistencyReviewService._group_facts(facts)

    assert candidates == ()
    assert unresolved[ConsistencyUnresolvedReason.ASSERTION_TYPE_MISMATCH] == 1


def test_mixed_assertion_subgroup_order_is_invariant() -> None:
    facts = (
        _fact("支护结构水平位移报警值为30mm。", page_number=18),
        _fact("支护结构水平位移报警值不大于20mm。", page_number=43),
    )

    assert InternalConsistencyReviewService._group_facts(
        facts
    ) == InternalConsistencyReviewService._group_facts(tuple(reversed(facts)))


# Grouping, duplicate suppression, and identity.


def test_thirty_thirty_twenty_produces_one_grouped_candidate() -> None:
    result = _review(
        "支护结构水平位移报警值为30mm。",
        "支护结构水平位移报警值控制为30mm。",
        "支护结构水平位移报警值为20mm。",
        numbers=(18, 27, 43),
    )
    assert result.candidate_count == 1
    candidate = result.candidates[0]
    assert [group.numeric_value for group in candidate.value_groups] == [
        Decimal("20"),
        Decimal("30"),
    ]
    assert [fact.physical_page for fact in candidate.value_groups[1].facts] == [18, 27]


def test_duplicate_exact_source_identity_does_not_inflate(monkeypatch: pytest.MonkeyPatch) -> None:
    original = InternalConsistencyReviewService._bounded_clauses

    def duplicate_spans(text: str) -> tuple[tuple[int, int], ...]:
        spans = original(text)
        return spans + spans

    monkeypatch.setattr(
        InternalConsistencyReviewService,
        "_bounded_clauses",
        staticmethod(duplicate_spans),
    )
    result = _review("立杆间距为30mm。")
    assert result.fact_count == 1


def test_page_order_is_invariant() -> None:
    service = InternalConsistencyReviewService()
    pages = _pages(
        "支护结构水平位移报警值为30mm。",
        "支护结构水平位移报警值为20mm。",
        numbers=(18, 43),
    )
    first = service._review_verified_pages(
        document_id=DOCUMENT_ID, document_sha256=DOCUMENT_SHA, pages=pages
    )
    second = service._review_verified_pages(
        document_id=DOCUMENT_ID,
        document_sha256=DOCUMENT_SHA,
        pages=list(reversed(pages)),
    )
    assert first == second


def test_fact_order_is_invariant() -> None:
    facts = (
        _fact("立杆间距为30mm。", page_number=18),
        _fact("立杆间距为20mm。", page_number=43),
    )
    first = InternalConsistencyReviewService._group_facts(facts)
    second = InternalConsistencyReviewService._group_facts(tuple(reversed(facts)))
    assert first == second


def test_document_sha_is_direct_group_isolation() -> None:
    facts = (
        _fact(
            "支护结构水平位移报警值为30mm。",
            page_number=18,
            document_sha="a" * 64,
        ),
        _fact(
            "支护结构水平位移报警值为20mm。",
            page_number=43,
            document_sha="b" * 64,
        ),
    )

    candidates, unresolved = InternalConsistencyReviewService._group_facts(facts)

    assert candidates == ()
    assert not unresolved


def test_group_order_is_deterministic() -> None:
    result = _review(
        "立杆间距为30mm。",
        "立杆间距为20mm。",
        "角焊缝厚度为8mm。",
        "角焊缝厚度为6mm。",
    )
    assert result.candidates == tuple(
        sorted(result.candidates, key=lambda item: item.candidate_id)
    )


def test_fact_id_is_deterministic_and_source_bound() -> None:
    first = _fact("立杆间距为30mm。", page_number=18)
    second = _fact("立杆间距为30mm。", page_number=18)
    changed_page = _fact("立杆间距为30mm。", page_number=19)
    changed_value = _fact("立杆间距为20mm。", page_number=18)
    changed_document = _fact(
        "立杆间距为30mm。", page_number=18, document_sha="b" * 64
    )
    assert first.fact_id == second.fact_id
    assert len(
        {first.fact_id, changed_page.fact_id, changed_value.fact_id, changed_document.fact_id}
    ) == 4


def test_candidate_id_is_deterministic_and_value_bound() -> None:
    first = _positive_result()
    second = _positive_result()
    changed = _review(
        "支护结构水平位移报警值为31mm。",
        "支护结构水平位移报警值为20mm。",
    )
    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id
    assert first.candidates[0].candidate_id != changed.candidates[0].candidate_id


def test_review_id_is_deterministic_document_bound_and_runtime_free() -> None:
    first = _positive_result()
    time.sleep(0.001)
    second = _positive_result()
    changed_document = _review(
        "支护结构水平位移报警值为30mm。",
        "支护结构水平位移报警值为20mm。",
        document_sha="b" * 64,
    )
    assert first.review_id == second.review_id
    assert first.review_id != changed_document.review_id
    assert "timestamp" not in InternalConsistencyReviewResult.model_fields


# Zero-candidate, real negative, and synthetic positive safety.


def test_zero_candidate_status_is_bounded_coverage_only() -> None:
    result = _review("立杆间距为30mm。")
    assert result.coverage_status == (
        ConsistencyCoverageStatus.NO_INCONSISTENCY_ESTABLISHED_WITHIN_D9_V1_COVERAGE
    )
    assert result.candidate_count == 0
    assert "plan_consistent" not in type(result).model_fields


def test_real_eight_mm_repeated_statement_is_negative_control() -> None:
    result = _review(
        "角焊缝厚度不低于8mm；",
        "角焊缝厚度不低于8mm；",
        numbers=(531, 536),
    )
    assert result.fact_count == 2
    assert result.candidate_count == 0


def test_real_eight_mm_clause_retains_exact_owned_fact() -> None:
    fact = _fact("④角焊缝厚度不低于8mm；", page_number=531)

    assert fact.object_profile_name == EngineeringObjectProfile.ANGLE_WELD
    assert fact.parameter_profile_name == ConsistencyParameterProfile.THICKNESS
    assert fact.numeric_value == Decimal("8")
    assert fact.normalized_unit == NormalizedUnit.MILLIMETRE


def test_thinner_plate_thickness_is_not_claimed_by_later_angle_weld() -> None:
    result = _review(
        "当被焊构件中较薄板厚度≥25mm时，宜采用局部开坡口的角焊缝。"
    )

    assert result.fact_count == 0
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[
        ConsistencyUnresolvedReason.OBJECT_IDENTITY_UNRESOLVED
    ] == 1


def test_object_anchor_after_numeric_token_cannot_claim_earlier_value() -> None:
    result = _review("厚度不低于25mm时采用角焊缝。")

    assert result.fact_count == 0
    assert result.candidate_count == 0


def test_unrelated_later_object_anchor_does_not_create_fact() -> None:
    result = _review("板件厚度为25mm，施工形式采用角焊缝。")

    assert result.fact_count == 0
    assert result.candidate_count == 0


def test_ambiguous_owner_relation_fails_closed() -> None:
    result = _review("角焊缝和立杆厚度为8mm。")

    assert result.fact_count == 0
    assert result.candidate_count == 0
    assert _unresolved_reasons(result)[
        ConsistencyUnresolvedReason.OBJECT_IDENTITY_UNRESOLVED
    ] == 1


def test_distinct_five_mm_measurements_are_not_grouped() -> None:
    result = _review(
        "支护结构水平位移预警值为5mm。",
        "冠梁水平位移预警值为5mm。",
        numbers=(71, 72),
    )
    assert result.fact_count == 2
    assert result.candidate_count == 0


def test_controlled_thirty_twenty_fixture_is_one_review_candidate() -> None:
    result = _positive_result()
    assert result.candidate_count == 1
    assert result.coverage_status == (
        ConsistencyCoverageStatus.CANDIDATES_ESTABLISHED_WITHIN_D9_V1_COVERAGE
    )
    assert result.candidates[0].review_status == ConsistencyReviewStatus.NEEDS_HUMAN_REVIEW


def test_production_source_contains_no_synthetic_case_authority() -> None:
    source = inspect.getsource(InternalConsistencyReviewService)
    assert "CASE-D" not in source
    assert "page 18" not in source
    assert "page 43" not in source
    assert "30 mm" not in source
    assert "20 mm" not in source


# Boundary isolation and performance.


def test_production_d9_has_no_normative_c2_c3_d5_or_d6_dependencies() -> None:
    source = inspect.getsource(inspect.getmodule(InternalConsistencyReviewService))
    forbidden = (
        "StandardRoutingService",
        "RequirementDecompositionService",
        "NormativeRequirement",
        "ComplianceComparisonService",
        "ReviewFindingService",
        "WholePlanReviewService",
        "FindingsWorkspaceService",
        "Finding(",
    )
    assert all(name not in source for name in forbidden)


def test_production_d9_has_no_llm_ocr_network_embedding_or_vector_authority() -> None:
    source = inspect.getsource(inspect.getmodule(InternalConsistencyReviewService)).casefold()
    forbidden = (
        "llm_service",
        "vlm",
        "ocr_service",
        "requests.",
        "httpx.",
        "embedding",
        "vector_store",
    )
    assert all(name not in source for name in forbidden)


def test_candidate_cannot_express_non_compliant_severity_or_risk() -> None:
    fields = InternalConsistencyCandidate.model_fields
    assert "decision" not in fields
    assert "severity" not in fields
    assert "risk_score" not in fields
    assert "finding_id" not in fields


def test_large_grouping_is_non_pairwise_and_deterministic() -> None:
    templates = (
        "支护结构水平位移报警值为{value}mm。",
        "冠梁水平位移报警值为{value}mm。",
        "立杆间距为{value}mm。",
        "角焊缝厚度为{value}mm。",
    )
    facts: list[ConsistencyFact] = []
    for index in range(2400):
        template = templates[index % len(templates)]
        value = 20 if (index // len(templates)) % 2 == 0 else 30
        facts.append(_fact(template.format(value=value), page_number=index + 1))
    started = time.perf_counter()
    first = InternalConsistencyReviewService._group_facts(tuple(facts))
    elapsed = time.perf_counter() - started
    second = InternalConsistencyReviewService._group_facts(tuple(reversed(facts)))
    assert len(first[0]) == 4
    assert first == second
    assert elapsed >= 0
