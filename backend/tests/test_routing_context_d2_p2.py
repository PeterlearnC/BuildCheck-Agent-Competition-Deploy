"""Focused qualification tests for source-grounded D.2-P2 routing context."""

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.review_candidate import ReviewCandidate
from app.schemas.routing_context import (
    RoutingContext,
    RoutingContextBoundaryMethod,
    deterministic_routing_context_id,
)
from app.schemas.standard_route import StandardRouteStatus, StandardRoutingMethod
from app.core.config import Settings
from app.services.pdf_service import PDFExtractionResult, PDFPageText, PDFService
from app.services.review.candidate_discovery_service import CandidateDiscoveryService
from app.services.review.routing_context_service import (
    MAX_ROUTING_CONTEXT_CHARS,
    RoutingContextError,
    RoutingContextDocumentBindingError,
    RoutingContextService,
)
from app.services.review.standard_routing_service import StandardRoutingService
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository


DOCUMENT_ID = "d2-p2-routing-context"
DOCUMENT_SHA = "d" * 64
QUALIFIED_PLAN_ID = "f42886ef-83ec-454c-86e2-34d5c238ca0f"
QUALIFIED_PLAN_SHA = "5f3ebde3bd2c0a6d6dccb516c3029fd336558e198ebeccad07d29ca0d03fd8dc"
QUALIFIED_STANDARD_ID = "55023b3b-1b00-4000-8000-000000000001"


class MemoryRepository:
    def __init__(self, documents):
        self.documents = list(documents)

    def list_documents(self):
        return list(self.documents)


class FixedPagePDFService:
    def __init__(self, page: PDFPageText) -> None:
        self.page = page

    def extract_text(self, _pdf_path: Path) -> PDFExtractionResult:
        return PDFExtractionResult(
            page_count=1,
            char_count=len(self.page.text),
            text=self.page.text,
            pages=[self.page],
        )


def discover_target(
    page_text: str,
    target: str,
    *,
    page_number: int = 1,
    document_id: str = DOCUMENT_ID,
    document_sha: str = DOCUMENT_SHA,
) -> tuple[PDFPageText, ReviewCandidate]:
    page = PDFPageText(page_number=page_number, text=page_text)
    candidates = CandidateDiscoveryService().discover(
        document_id=document_id,
        document_sha256=document_sha,
        pages=[page],
    )
    return page, next(item for item in candidates if target in item.source_text)


def build_context(
    page_text: str = "1、可调托撑安装要求，螺杆长度不得小于150mm。",
    target: str = "螺杆长度不得小于150mm",
    **candidate_kwargs,
) -> tuple[ReviewCandidate, RoutingContext]:
    page, candidate = discover_target(
        page_text, target, **candidate_kwargs
    )
    return candidate, RoutingContextService()._build_from_verified_page(
        candidate=candidate, page=page
    )


class FixedRoutingContextService:
    def __init__(self, context: RoutingContext) -> None:
        self.context = context

    def build_for_candidate(self, *, candidate: ReviewCandidate) -> RoutingContext:
        return RoutingContextService().verify(candidate=candidate, context=self.context)


def route_with_server_context(
    candidate: ReviewCandidate,
    context: RoutingContext,
    *,
    repository=None,
    registry=None,
):
    service = StandardRoutingService(repository=repository, registry=registry)
    service._routing_context_service = FixedRoutingContextService(context)
    return service.route(candidate)


def context_id_for(context: RoutingContext, **changes) -> str:
    values = {
        "candidate_id": context.candidate_id,
        "document_sha256": context.document_sha256,
        "page_number": context.page_number,
        "context_start": context.context_start,
        "context_end": context.context_end,
        "context_text_sha256": context.context_text_sha256,
        "candidate_relative_start": context.candidate_relative_start,
        "candidate_relative_end": context.candidate_relative_end,
        "candidate_source_text_sha256": context.candidate_source_text_sha256,
        "source_page_text_sha256": context.source_page_text_sha256,
        "source_binding_method": context.source_binding_method,
        "boundary_method": context.boundary_method,
        "context_identity_version": context.context_identity_version,
        "boundary_version": context.boundary_version,
    }
    values.update(changes)
    return deterministic_routing_context_id(**values)


def test_context_identity_is_deterministic() -> None:
    first_candidate, first = build_context()
    second_candidate, second = build_context()
    assert first_candidate == second_candidate
    assert first == second


def test_changed_context_text_changes_context_identity() -> None:
    candidate, first = build_context("1、可调托撑安装要求，螺杆长度不得小于150mm。")
    changed_page = PDFPageText(
        page_number=1, text="1、模板支撑安装要求，螺杆长度不得小于150mm。"
    )
    second = RoutingContextService()._build_from_verified_page(
        candidate=candidate, page=changed_page
    )
    assert first.routing_context_id != second.routing_context_id


def test_changed_boundary_method_changes_context_identity() -> None:
    _, context = build_context()
    changed = context_id_for(
        context, boundary_method=RoutingContextBoundaryMethod.EXACT_ONLY
    )
    assert changed != context.routing_context_id


def test_changed_candidate_identity_changes_context_identity() -> None:
    _, first = build_context(document_sha="d" * 64)
    _, second = build_context(document_sha="e" * 64)
    assert first.context_text == second.context_text
    assert first.routing_context_id != second.routing_context_id


def test_context_schema_forbids_extra_fields() -> None:
    _, context = build_context()
    with pytest.raises(ValidationError):
        RoutingContext.model_validate({**context.model_dump(), "standard_id": "caller"})


def test_exact_candidate_slice_and_hash_are_preserved() -> None:
    candidate, context = build_context()
    assert context.context_text[
        context.candidate_relative_start : context.candidate_relative_end
    ] == candidate.source_text
    assert context.candidate_source_text_sha256 == candidate.source_text_sha256


def test_context_schema_rejects_candidate_slice_sha_mismatch() -> None:
    _, context = build_context()
    with pytest.raises(ValidationError):
        RoutingContext.model_validate(
            {**context.model_dump(), "candidate_source_text_sha256": "a" * 64}
        )


def test_context_builder_rejects_page_offset_or_text_mismatch() -> None:
    page, candidate = discover_target(
        "1、可调托撑安装要求，螺杆长度不得小于150mm。", "螺杆长度"
    )
    span = candidate.source_spans[0]
    shifted_span = span.model_copy(
        update={"char_start": span.char_start + 1, "char_end": span.char_end + 1}
    )
    shifted = candidate.model_copy(update={"source_spans": (shifted_span,)})
    with pytest.raises(RoutingContextError):
        RoutingContextService()._build_from_verified_page(candidate=shifted, page=page)
    changed_page = PDFPageText(page_number=1, text=page.text.replace("长度", "宽度"))
    with pytest.raises(RoutingContextError):
        RoutingContextService()._build_from_verified_page(
            candidate=candidate, page=changed_page
        )


def test_context_builder_rejects_cross_page_input() -> None:
    _, candidate = discover_target("1、设备要求，螺杆长度不得小于20mm。", "螺杆长度")
    with pytest.raises(RoutingContextError):
        RoutingContextService()._build_from_verified_page(
            candidate=candidate,
            page=PDFPageText(page_number=2, text=candidate.source_text),
        )


def test_production_builder_rejects_cross_document_matching_page(tmp_path) -> None:
    document_id = "11111111-1111-4111-8111-111111111111"
    page_text = "1、可调托撑安装要求，螺杆长度不得小于20mm。"
    page, candidate = discover_target(
        page_text,
        "螺杆长度",
        document_id=document_id,
        document_sha=hashlib.sha256(b"document-a").hexdigest(),
    )
    (tmp_path / f"{document_id}.pdf").write_bytes(b"document-b")
    service = RoutingContextService(
        settings=Settings(upload_dir=tmp_path),
        pdf_service=FixedPagePDFService(page),
    )
    with pytest.raises(RoutingContextDocumentBindingError, match="SHA"):
        service.build_for_candidate(candidate=candidate)


def test_production_builder_rejects_wrong_candidate_document_sha(tmp_path) -> None:
    document_id = "22222222-2222-4222-8222-222222222222"
    stored_bytes = b"authoritative-document"
    page_text = "1、可调托撑安装要求，螺杆长度不得小于20mm。"
    page, candidate = discover_target(
        page_text,
        "螺杆长度",
        document_id=document_id,
        document_sha="a" * 64,
    )
    (tmp_path / f"{document_id}.pdf").write_bytes(stored_bytes)
    service = RoutingContextService(
        settings=Settings(upload_dir=tmp_path),
        pdf_service=FixedPagePDFService(page),
    )
    with pytest.raises(RoutingContextDocumentBindingError, match="SHA"):
        service.build_for_candidate(candidate=candidate)


def test_production_builder_accepts_verified_stored_document(tmp_path) -> None:
    document_id = "33333333-3333-4333-8333-333333333333"
    stored_bytes = b"authoritative-document"
    page_text = "1、可调托撑安装要求，螺杆长度不得小于20mm。"
    page, candidate = discover_target(
        page_text,
        "螺杆长度",
        document_id=document_id,
        document_sha=hashlib.sha256(stored_bytes).hexdigest(),
    )
    (tmp_path / f"{document_id}.pdf").write_bytes(stored_bytes)
    service = RoutingContextService(
        settings=Settings(upload_dir=tmp_path),
        pdf_service=FixedPagePDFService(page),
    )
    context = service.build_for_candidate(candidate=candidate)
    assert context.document_id == document_id
    assert context.document_sha256 == hashlib.sha256(stored_bytes).hexdigest()
    assert context.source_page_text_sha256 == hashlib.sha256(
        page_text.encode("utf-8")
    ).hexdigest()
    assert not hasattr(service, "build")


def test_public_router_rejects_caller_supplied_routing_context() -> None:
    candidate, forged_context = build_context()
    service = StandardRoutingService()
    with pytest.raises(TypeError):
        service.route(candidate, forged_context)
    assert service.route(candidate).route_status == StandardRouteStatus.NO_STANDARD_SCOPE


def test_numbered_item_owner_and_next_peer_boundary_are_deterministic() -> None:
    text = "1、可调托撑安装要求，螺杆长度不得小于150mm。\n2、材料应符合要求。"
    candidate, context = build_context(text)
    assert context.boundary_method == RoutingContextBoundaryMethod.NUMBERED_ITEM
    assert context.context_text.startswith("1、可调托撑")
    assert "2、材料" not in context.context_text
    assert context.context_end <= len(text)
    assert candidate.source_text in context.context_text


@pytest.mark.parametrize("marker", ["1、", "1．", "1. ", "1.2 "])
def test_minimum_supported_owner_marker_forms(marker) -> None:
    _, context = build_context(f"{marker}可调托撑安装要求，螺杆长度不得小于150mm。")
    assert context.boundary_method == RoutingContextBoundaryMethod.NUMBERED_ITEM


def test_decimal_measurement_is_not_an_owner_marker() -> None:
    _, context = build_context(
        "4.5m范围介绍可调托撑。\n螺杆长度不得小于20mm。", "螺杆长度"
    )
    assert context.boundary_method == RoutingContextBoundaryMethod.EXACT_ONLY
    assert context.context_text == "螺杆长度不得小于20mm。"


def test_candidate_before_first_owner_uses_exact_only() -> None:
    _, context = build_context(
        "螺杆长度不得小于20mm。\n1、可调托撑安装要求。", "螺杆长度"
    )
    assert context.boundary_method == RoutingContextBoundaryMethod.EXACT_ONLY


def test_owner_larger_than_280_fails_closed_to_exact_only() -> None:
    text = "1、可调托撑" + "甲" * 300 + "，螺杆长度不得小于20mm。"
    candidate, context = build_context(text, "螺杆长度")
    assert len(context.context_text) <= MAX_ROUTING_CONTEXT_CHARS
    assert context.boundary_method == RoutingContextBoundaryMethod.EXACT_ONLY
    assert context.context_text == candidate.source_text


def test_context_anchor_more_than_64_characters_away_does_not_route() -> None:
    text = "1、可调托撑" + "甲" * 65 + "，螺杆长度不得小于20mm。"
    candidate, context = build_context(text, "螺杆长度")
    result = route_with_server_context(candidate, context)
    assert context.boundary_method == RoutingContextBoundaryMethod.NUMBERED_ITEM
    assert result.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert result.routing_context_id is None


@pytest.mark.parametrize(
    "generic",
    [
        "施工过程中应加强管理。",
        "材料应符合要求。",
        "应做好安全工作。",
        "工程质量应满足要求。",
        "施工完成后应检查。",
    ],
)
def test_generic_candidate_cannot_inherit_scaffold_owner(generic) -> None:
    text = "1、可调托撑、立杆和钢管安装要求，" + generic
    candidate, context = build_context(text, generic.rstrip("。"))
    result = route_with_server_context(candidate, context)
    assert result.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert result.routing_context_id is None


def test_next_peer_item_cannot_inherit_previous_scaffold_owner() -> None:
    text = "1、可调托撑与钢管安装要求。\n2、设备螺杆长度不得小于20mm。"
    candidate, context = build_context(text, "螺杆长度")
    assert context.context_text.startswith("2、设备")
    assert "可调托撑" not in context.context_text
    assert route_with_server_context(candidate, context).route_status == (
        StandardRouteStatus.NO_STANDARD_SCOPE
    )


def test_scaffold_heavy_page_without_owner_remains_exact_only() -> None:
    text = "本页介绍可调托撑、立杆和钢管。\n螺杆长度不得小于20mm。"
    candidate, context = build_context(text, "螺杆长度")
    assert context.boundary_method == RoutingContextBoundaryMethod.EXACT_ONLY
    assert route_with_server_context(candidate, context).route_status == (
        StandardRouteStatus.NO_STANDARD_SCOPE
    )


def test_unrelated_owner_with_weak_screw_term_does_not_route() -> None:
    candidate, context = build_context(
        "1、机电设备维护要求，螺杆长度不得小于20mm。", "螺杆长度"
    )
    assert route_with_server_context(candidate, context).route_status == (
        StandardRouteStatus.NO_STANDARD_SCOPE
    )


@pytest.mark.parametrize(
    "reference",
    [
        "GB55023-2022",
        "施工脚手架通用规范",
        "《施工脚手架通用规范》",
        "GB55023-2022 施工脚手架通用规范 第4.4.15条",
        "依据假冒施工脚手架规范",
    ],
)
def test_plan_standard_reference_cannot_supply_context_domain_signal(reference) -> None:
    candidate, context = build_context(
        f"1、{reference}\n螺杆长度不得小于20mm。", "螺杆长度"
    )
    result = route_with_server_context(candidate, context)
    assert result.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert result.selected_scope is None


@pytest.mark.parametrize(
    "title",
    [
        "假冒施工脚手架规范",
        "某某施工脚手架规范",
        "临时脚手架安全技术规程",
        "虚构模板支撑架标准",
        "未知脚手架施工规范",
    ],
)
def test_unknown_bare_title_cannot_supply_context_domain_signal(title) -> None:
    candidate, context = build_context(
        f"1、{title}\n螺杆长度不得小于20mm。", "螺杆长度不得小于20mm"
    )
    service = StandardRoutingService()
    qualified, _ = service._qualified_scopes()
    eligible = service._eligible_domain_text(context.context_text, qualified)
    assert title not in eligible
    result = route_with_server_context(candidate, context)
    assert result.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert result.selected_scope is None


def test_unknown_bare_title_before_reference_verb_is_negative_only() -> None:
    candidate, context = build_context(
        "1、某某脚手架规范规定，设置可调托撑。\n螺杆长度不得小于20mm。",
        "螺杆长度不得小于20mm",
    )
    service = StandardRoutingService()
    qualified, _ = service._qualified_scopes()
    eligible = service._eligible_domain_text(context.context_text, qualified)
    assert "某某脚手架规范" not in eligible
    assert "可调托撑" in eligible
    assert route_with_server_context(candidate, context).route_status == (
        StandardRouteStatus.ROUTED
    )


@pytest.mark.parametrize(
    "reference",
    [
        "施工脚手架，通用规范",
        "施工脚手架 通用规范",
        "GB55023-2022，施工脚手架，通用规范",
        "《施工脚手架通用规范》GB55023-2022",
    ],
)
def test_split_known_standard_name_cannot_supply_domain_signal(reference) -> None:
    candidate, context = build_context(
        f"1、{reference}\n螺杆长度不得小于20mm。", "螺杆长度"
    )
    assert route_with_server_context(candidate, context).route_status == (
        StandardRouteStatus.NO_STANDARD_SCOPE
    )


def test_spaced_article_reference_is_masked_without_selecting_scope() -> None:
    service = StandardRoutingService()
    qualified, _ = service._qualified_scopes()
    text = "第 4.4.15 条"
    masked = service._eligible_domain_text(text, qualified)
    assert masked == " " * len(text)


@pytest.mark.parametrize(
    "engineering_text",
    [
        "脚手架搭设应规范操作。",
        "模板支撑架应按标准化流程检查。",
        "可调托撑安装应符合相关要求。",
        "脚手架施工应符合规范。",
        "脚手架施工应符合某某规范。",
        "脚手架搭设必须符合现行规范要求。",
        "模板支撑架施工应按相关标准执行。",
    ],
)
def test_ordinary_engineering_prose_is_not_overmasked(engineering_text) -> None:
    page, candidate = discover_target(engineering_text, engineering_text.rstrip("。"))
    service = StandardRoutingService()
    qualified, _ = service._qualified_scopes()
    eligible = service._eligible_domain_text(candidate.source_text, qualified)
    assert eligible == candidate.source_text
    assert service.route(candidate).route_status == StandardRouteStatus.ROUTED


def test_plan_standard_reference_cannot_supply_exact_span_domain_signal() -> None:
    page, candidate = discover_target(
        "依据《施工脚手架通用规范》，螺杆长度不得小于20mm。",
        "螺杆长度",
    )
    assert page.text.startswith(candidate.source_text, candidate.source_spans[0].char_start)
    assert StandardRoutingService().route(candidate).route_status == (
        StandardRouteStatus.NO_STANDARD_SCOPE
    )


def test_engineering_domain_signal_outside_reference_remains_eligible() -> None:
    candidate, context = build_context(
        "1、依据《施工脚手架通用规范》要求，设置可调托撑。\n螺杆长度不得小于20mm。",
        "螺杆长度",
    )
    result = route_with_server_context(candidate, context)
    assert result.route_status == StandardRouteStatus.ROUTED
    assert result.selected_scope.standard_ids == [QUALIFIED_STANDARD_ID]
    assert result.routing_method == StandardRoutingMethod.CONTEXT_ASSISTED_DETERMINISTIC


@pytest.fixture(scope="module")
def qualified_plan_pages():
    root = Path(__file__).resolve().parents[2]
    path = root / f"backend/data/uploads/{QUALIFIED_PLAN_ID}.pdf"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == QUALIFIED_PLAN_SHA
    return PDFService().extract_text(path).pages


@pytest.mark.parametrize(
    "page_number,target,marker",
    [
        (17, "150mm", "插入"),
        (46, "200mm", "伸出"),
        (46, "36mm", "不低于"),
    ],
)
def test_real_plan_context_recovers_diagnosed_misses(
    qualified_plan_pages, page_number, target, marker
) -> None:
    page = qualified_plan_pages[page_number - 1]
    candidates = CandidateDiscoveryService().discover(
        document_id=QUALIFIED_PLAN_ID,
        document_sha256=QUALIFIED_PLAN_SHA,
        pages=[page],
    )
    candidate = next(
        item for item in candidates if target in item.source_text and marker in item.source_text
    )
    context = RoutingContextService().build_for_candidate(candidate=candidate)
    service = StandardRoutingService()
    recovered = service.route(candidate)
    assert recovered.route_status == StandardRouteStatus.ROUTED
    assert recovered.selected_scope.standard_ids == [QUALIFIED_STANDARD_ID]
    assert recovered.routing_method == StandardRoutingMethod.CONTEXT_ASSISTED_DETERMINISTIC
    assert recovered.routing_context_id == context.routing_context_id


@pytest.mark.parametrize(
    "page_number,target",
    [(17, "可调托撑螺杆外径不得小于36mm"), (46, "立杆钢管内径的间隙不大于3mm")],
)
def test_existing_exact_routes_and_route_ids_ignore_unused_context(
    qualified_plan_pages, page_number, target
) -> None:
    page = qualified_plan_pages[page_number - 1]
    candidate = next(
        item
        for item in CandidateDiscoveryService().discover(
            document_id=QUALIFIED_PLAN_ID,
            document_sha256=QUALIFIED_PLAN_SHA,
            pages=[page],
        )
        if target in item.source_text
    )
    context = RoutingContextService().build_for_candidate(candidate=candidate)
    without_context = StandardRoutingService().route(candidate)
    with_context = route_with_server_context(candidate, context)
    assert without_context == with_context
    assert with_context.routing_method == StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES
    assert with_context.routing_context_id is None


def test_context_assisted_route_identity_binds_used_context() -> None:
    first_candidate, first_context = build_context(
        "1、可调托撑安装要求，螺杆长度不得小于150mm。"
    )
    second_page = PDFPageText(
        page_number=1, text="1、模板支撑安装要求，螺杆长度不得小于150mm。"
    )
    second_context = RoutingContextService()._build_from_verified_page(
        candidate=first_candidate, page=second_page
    )
    first_route = route_with_server_context(first_candidate, first_context)
    second_route = route_with_server_context(first_candidate, second_context)
    assert first_route.route_status == second_route.route_status == StandardRouteStatus.ROUTED
    assert first_route.route_id != second_route.route_id


def test_missing_registry_still_fails_closed_with_context() -> None:
    candidate, context = build_context()
    result = route_with_server_context(
        candidate, context, registry=StandardRegistryService(entries=[])
    )
    assert result.route_status == StandardRouteStatus.NO_STANDARD_SCOPE
    assert result.selected_scope is None


def test_context_ambiguity_never_selects_first_scope() -> None:
    candidate, context = build_context()
    document = StandardRepository().list_documents()[0]
    second = document.model_copy(update={"standard_id": "qualified-standard-2"})
    result = route_with_server_context(
        candidate, context, repository=MemoryRepository([document, second])
    )
    assert result.route_status == StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE
    assert result.selected_scope is None
    assert len(result.scope_candidates) == 2


def test_context_output_contains_no_article_normative_or_decision_authority() -> None:
    candidate, context = build_context()
    route = route_with_server_context(candidate, context)
    fields = set(type(context).model_fields) | set(type(route).model_fields)
    prohibited = {
        "standard_id", "article_id", "article_number", "normative_text",
        "threshold", "operator", "requirement_id", "plan_fact", "decision",
        "compliance", "finding_id", "review_action", "whole_document_status",
    }
    assert prohibited.isdisjoint(fields)


def test_p2_production_has_no_external_or_semantic_authority_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    production = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "backend/app/schemas/routing_context.py",
            "backend/app/services/review/routing_context_service.py",
            "backend/app/schemas/standard_route.py",
            "backend/app/services/review/standard_routing_service.py",
        )
    ).casefold()
    prohibited = (
        "standardssearchservice", "compliancecomparisonservice", "reviewfindingservice",
        "findingreviewservice", "openai", "deepseek", "httpx", "requests.",
        "subprocess", "article_number", "finding_id", "unit conversion",
        "case a", "case b", "case c",
    )
    assert all(token not in production for token in prohibited)
