"""D.1 deterministic candidate discovery and authority-boundary tests."""

import hashlib
import inspect

import pytest
from pydantic import ValidationError

from app.schemas.review_candidate import (
    REVIEW_CANDIDATE_DISCOVERY_VERSION,
    REVIEW_CANDIDATE_IDENTITY_VERSION,
    CandidateSourceSpan,
    ReviewCandidate,
    ReviewCandidateClass,
    ReviewCandidateDiscoveryMethod,
    ReviewCandidateStatus,
    deterministic_candidate_id,
)
from app.services.pdf_service import PDFPageText
from app.services.review.candidate_discovery_service import (
    GENERAL_TOPIC,
    MAX_CANDIDATE_CHARS,
    CandidateDiscoveryInputError,
    CandidateDiscoveryService,
)
import app.services.review.candidate_discovery_service as discovery_module


DOCUMENT_ID = "document-d1-test"
DOCUMENT_SHA = "a" * 64


def discover(*texts: str, document_sha256: str = DOCUMENT_SHA):
    pages = [
        PDFPageText(page_number=index, text=text)
        for index, text in enumerate(texts, 1)
    ]
    return CandidateDiscoveryService().discover(
        document_id=DOCUMENT_ID,
        document_sha256=document_sha256,
        pages=pages,
    )


def make_candidate(
    *,
    source_text: str = "立杆间距不大于1500mm。",
    char_start: int = 0,
    candidate_class: ReviewCandidateClass = ReviewCandidateClass.NUMERIC_CONTROL,
    confidence: float = 0.9,
) -> ReviewCandidate:
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    span = CandidateSourceSpan(
        page_number=1,
        char_start=char_start,
        char_end=char_start + len(source_text),
        source_text=source_text,
        source_text_sha256=source_hash,
    )
    status = (
        ReviewCandidateStatus.UNRESOLVED
        if candidate_class == ReviewCandidateClass.UNRESOLVED
        else ReviewCandidateStatus.DISCOVERED
    )
    return ReviewCandidate(
        identity_version=REVIEW_CANDIDATE_IDENTITY_VERSION,
        candidate_id=deterministic_candidate_id(
            document_sha256=DOCUMENT_SHA,
            source_spans=[span],
            candidate_class=candidate_class,
        ),
        document_id=DOCUMENT_ID,
        document_sha256=DOCUMENT_SHA,
        source_spans=(span,),
        source_text=source_text,
        source_text_sha256=source_hash,
        topic="立杆",
        candidate_class=candidate_class,
        discovery_method=ReviewCandidateDiscoveryMethod.DETERMINISTIC_SIGNAL_RULES,
        discovery_version=REVIEW_CANDIDATE_DISCOVERY_VERSION,
        confidence=confidence,
        status=status,
    )


def test_same_input_has_identical_candidate_ids_and_order() -> None:
    pages = [
        PDFPageText(page_number=2, text="脚手架搭设完成后必须进行检查验收。"),
        PDFPageText(page_number=1, text="立杆间距不大于1500mm。"),
    ]
    service = CandidateDiscoveryService()
    first = service.discover(
        document_id=DOCUMENT_ID, document_sha256=DOCUMENT_SHA, pages=pages
    )
    second = service.discover(
        document_id=DOCUMENT_ID, document_sha256=DOCUMENT_SHA, pages=pages
    )
    assert first == second
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]


def test_changed_source_text_changes_source_and_candidate_identity() -> None:
    first = discover("间隙不大于3mm。")[0]
    second = discover("间隙不大于4mm。")[0]
    assert first.source_text_sha256 != second.source_text_sha256
    assert first.candidate_id != second.candidate_id


def test_changed_source_span_changes_candidate_identity() -> None:
    first = make_candidate(char_start=0)
    second = make_candidate(char_start=12)
    assert first.candidate_id != second.candidate_id


def test_changed_candidate_class_changes_candidate_identity() -> None:
    numeric = make_candidate(candidate_class=ReviewCandidateClass.NUMERIC_CONTROL)
    structural = make_candidate(
        candidate_class=ReviewCandidateClass.STRUCTURAL_CONFIGURATION
    )
    assert numeric.candidate_id != structural.candidate_id


def test_confidence_change_does_not_change_candidate_identity() -> None:
    lower = make_candidate(confidence=0.55)
    higher = make_candidate(confidence=0.95)
    assert lower.candidate_id == higher.candidate_id


def test_exact_source_span_reconstructs_candidate_text() -> None:
    page = "前置说明。立杆间距不大于1500mm。后续说明。"
    candidate = discover(page)[0]
    span = candidate.source_spans[0]
    assert page[span.char_start : span.char_end] == span.source_text
    assert span.source_text == candidate.source_text


def test_invalid_source_span_and_candidate_text_are_rejected() -> None:
    text = "间隙不大于3mm。"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with pytest.raises(ValidationError, match="span"):
        CandidateSourceSpan(
            page_number=1,
            char_start=0,
            char_end=len(text) + 1,
            source_text=text,
            source_text_sha256=digest,
        )
    valid = make_candidate()
    with pytest.raises(ValidationError, match="source_text"):
        ReviewCandidate.model_validate(
            {**valid.model_dump(mode="json"), "source_text": "fabricated"}
        )


def test_numeric_control_is_discovered_without_judgment() -> None:
    candidate = discover("间隙不大于3mm。")[0]
    assert candidate.candidate_class == ReviewCandidateClass.NUMERIC_CONTROL
    assert candidate.status == ReviewCandidateStatus.DISCOVERED
    assert "decision" not in candidate.model_dump()


@pytest.mark.parametrize(
    "text",
    [
        "可调托撑螺杆伸出钢管顶部不得大于200mm。",
        "混凝土强度不低于30MPa。",
        "施工荷载不得超过3kN。",
    ],
)
def test_explicit_numeric_limit_takes_classification_priority(text: str) -> None:
    candidate = discover(text)[0]
    assert candidate.candidate_class == ReviewCandidateClass.NUMERIC_CONTROL


@pytest.mark.parametrize(
    "text",
    [
        "建筑高度为90m。",
        "建筑面积为50000m²。",
        "结构高度为30m。",
        "楼层层高为4.5m。",
        "钢管直径为48.3mm。",
        "共有3个施工区域。",
    ],
)
def test_descriptive_numeric_fact_is_not_promoted_to_numeric_control(text: str) -> None:
    candidates = discover(text)
    assert all(
        candidate.candidate_class != ReviewCandidateClass.NUMERIC_CONTROL
        for candidate in candidates
    )


@pytest.mark.parametrize(
    "text", ["3", "参数3", "编号为2026", "3mm", "2026年8月31日。"]
)
def test_bare_number_or_unit_without_review_context_is_not_candidate(text: str) -> None:
    assert discover(text) == []


def test_inspection_candidate_is_discovered() -> None:
    candidate = discover("模板支撑架搭设完成后应组织检查验收。")[0]
    assert candidate.candidate_class == ReviewCandidateClass.INSPECTION_REQUIREMENT


def test_safety_candidate_is_discovered() -> None:
    candidate = discover("高处作业人员必须设置安全防护措施。")[0]
    assert candidate.candidate_class == ReviewCandidateClass.SAFETY_REQUIREMENT


def test_procedural_candidate_is_discovered() -> None:
    candidate = discover("模板应按顺序安装并拆除。")[0]
    assert candidate.candidate_class == ReviewCandidateClass.PROCEDURAL_REQUIREMENT


def test_material_and_structural_candidates_are_conservative() -> None:
    material = discover("钢管材料必须具有产品合格证明。")[0]
    structural = discover("立杆应纵横成排设置。")[0]
    assert material.candidate_class == ReviewCandidateClass.MATERIAL_REQUIREMENT
    assert structural.candidate_class == ReviewCandidateClass.STRUCTURAL_CONFIGURATION


def test_ambiguous_obligation_is_preserved_as_unresolved() -> None:
    candidate = discover("应予以妥善处理。")[0]
    assert candidate.candidate_class == ReviewCandidateClass.UNRESOLVED
    assert candidate.status == ReviewCandidateStatus.UNRESOLVED
    assert candidate.topic == GENERAL_TOPIC


def test_multiple_triggers_in_same_span_are_suppressed() -> None:
    candidates = discover("立杆间距不大于1500mm且最大高度不超过20m。")
    assert len(candidates) == 1
    assert candidates[0].candidate_class == ReviewCandidateClass.NUMERIC_CONTROL


def test_deterministic_order_uses_physical_page_and_span() -> None:
    pages = [
        PDFPageText(page_number=3, text="作业人员必须设置安全防护设施。"),
        PDFPageText(page_number=1, text="间隙不大于3mm。随后必须检查脚手架。"),
        PDFPageText(page_number=2, text="立杆应纵横成排设置。"),
    ]
    result = CandidateDiscoveryService().discover(
        document_id=DOCUMENT_ID, document_sha256=DOCUMENT_SHA, pages=pages
    )
    ordering = [
        (item.source_spans[0].page_number, item.source_spans[0].char_start)
        for item in result
    ]
    assert ordering == sorted(ordering)


def test_long_context_is_bounded_and_still_exact() -> None:
    page = "前置内容" * 80 + "立杆间距不大于1500mm" + "后续内容" * 80
    candidate = discover(page)[0]
    span = candidate.source_spans[0]
    assert len(candidate.source_text) <= MAX_CANDIDATE_CHARS
    assert page[span.char_start : span.char_end] == candidate.source_text


def test_schema_forbids_extra_fields() -> None:
    payload = make_candidate().model_dump(mode="json")
    payload["decision"] = "NON_COMPLIANT"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ReviewCandidate.model_validate(payload)


def test_schema_contains_no_compliance_or_downstream_authority_fields() -> None:
    prohibited = {
        "decision",
        "finding_id",
        "standard_id",
        "article_id",
        "requirement_id",
        "compliance",
    }
    assert prohibited.isdisjoint(ReviewCandidate.model_fields)


def test_input_requires_verified_identity_and_pdf_page_records() -> None:
    service = CandidateDiscoveryService()
    with pytest.raises(CandidateDiscoveryInputError, match="SHA-256"):
        service.discover(document_id=DOCUMENT_ID, document_sha256="bad", pages=[])
    with pytest.raises(CandidateDiscoveryInputError, match="PDFPageText"):
        service.discover(
            document_id=DOCUMENT_ID,
            document_sha256=DOCUMENT_SHA,
            pages=[{"page_number": 1, "text": "间隙不大于3mm。"}],
        )


def test_duplicate_or_invalid_physical_pages_are_rejected() -> None:
    service = CandidateDiscoveryService()
    with pytest.raises(CandidateDiscoveryInputError, match="positive and unique"):
        service.discover(
            document_id=DOCUMENT_ID,
            document_sha256=DOCUMENT_SHA,
            pages=[
                PDFPageText(page_number=1, text="间隙不大于3mm。"),
                PDFPageText(page_number=1, text="立杆应设置。"),
            ],
        )


def test_discovery_confidence_is_bounded_and_not_compliance_confidence() -> None:
    resolved = discover("间隙不大于3mm。")[0]
    unresolved = discover("应予以妥善处理。")[0]
    assert 0.0 <= resolved.confidence <= 0.95
    assert 0.0 <= unresolved.confidence <= 0.55


def test_d1_module_has_no_prohibited_authority_or_external_integrations() -> None:
    source = inspect.getsource(discovery_module)
    prohibited = (
        "StandardsSearchService",
        "ComplianceReviewService",
        "ComplianceComparisonService",
        "ReviewFindingService",
        "FindingReviewService",
        "app.services.ocr",
        "LLMService",
        "openai",
        "deepseek",
        "httpx",
        "requests",
        "Path(",
        "open(",
        "finding_993e",
        "finding_5308",
        "finding_9c02",
    )
    assert all(value not in source for value in prohibited)
