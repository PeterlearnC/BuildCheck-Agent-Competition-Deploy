"""D.6 source-grounded, read-only findings-workspace qualification tests."""

from __future__ import annotations

import hashlib
import inspect
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.findings_workspace import (
    FindingWorkspaceItem,
    FindingsWorkspaceCounts,
    FindingsWorkspaceResult,
    InternalErrorWorkspaceItem,
    PlanSourceLocator,
    ReviewGapSource,
    ReviewGapWorkspaceItem,
    StandardSourceLocator,
    WorkspaceItemKind,
    derive_workspace_counts,
    deterministic_workspace_id,
    workspace_item_authority_id,
    workspace_item_sort_key,
)
from app.schemas.plan_fact_extraction import PlanFactExtractionStatus
from app.schemas.requirement_decomposition import (
    RequirementDecompositionResult,
    RequirementDecompositionStatus,
    UnresolvedRequirementReason,
    UnresolvedRequirementSpan,
    deterministic_decomposition_id,
    deterministic_unresolved_span_id,
)
from app.schemas.standard_route import StandardRouteStatus
from app.schemas.whole_plan_review import (
    WholePlanCandidateTerminalState,
    WholePlanCandidateTrace,
    WholePlanRequirementTerminalState,
    WholePlanReviewResult,
    derive_coverage_counts,
    deterministic_candidate_trace_id,
    deterministic_whole_plan_review_id,
)
from app.services.review.findings_workspace_service import (
    FindingsWorkspaceAuthorityInvariantError,
    FindingsWorkspaceService,
    FindingsWorkspaceSourceNavigationError,
)
from tests.test_whole_plan_review_d5 import _service as d5_service


class _D5:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def review_document(self, document_id):
        self.calls.append(document_id)
        return self.result


class _Repository:
    def __init__(self, result, *, wrong_checksum=False, wrong_article=False):
        self.documents = {}
        self.articles = {}
        self.document_calls = []
        self.article_calls = []
        for trace in result.candidate_traces:
            if trace.route is None or not trace.route.scope_candidates:
                continue
            scope = trace.route.scope_candidates[0]
            checksum = "0" * 64 if wrong_checksum else scope.source_checksum
            self.documents[scope.standard_id] = SimpleNamespace(
                standard_id=scope.standard_id,
                standard_code=scope.standard_code,
                canonical_standard_code=scope.canonical_standard_code,
                source_checksum=checksum,
            )
            if trace.decomposition is None:
                continue
            sources = (*trace.decomposition.requirements, *trace.decomposition.unresolved_spans)
            grouped = {}
            for source in sources:
                grouped.setdefault(source.article_id, []).append(source)
            by_id = {}
            for source_id, article_sources in grouped.items():
                max_end = max(
                    getattr(value, "requirement_char_end", getattr(value, "char_end", 0))
                    for value in article_sources
                )
                characters = [" "] * max_end
                for value in article_sources:
                    start = getattr(value, "requirement_char_start", getattr(value, "char_start", 0))
                    text = getattr(value, "requirement_text", getattr(value, "source_text", ""))
                    characters[start : start + len(text)] = text
                source = article_sources[0]
                article_id = "wrong" if wrong_article else source.article_id
                by_id[source_id] = SimpleNamespace(
                    article_id=article_id,
                    standard_id=scope.standard_id,
                    article_number=source.article_number,
                    source_page_start=3,
                    source_page_end=3,
                    source_text="".join(characters),
                )
            self.articles[scope.standard_id] = list(by_id.values())

    def load_document(self, standard_id):
        self.document_calls.append(standard_id)
        return self.documents.get(standard_id)

    def load_articles(self, standard_id):
        self.article_calls.append(standard_id)
        return self.articles.get(standard_id, [])


def _workspace(tmp_path, **kwargs):
    env = d5_service(tmp_path, **kwargs)
    result = env.service.review_document(env.document_id)
    d5 = _D5(result)
    repository = _Repository(result)
    service = FindingsWorkspaceService(
        settings=env.settings,
        whole_plan_review_service=d5,
        standard_repository=repository,
    )
    return SimpleNamespace(
        env=env,
        d5_result=result,
        d5=d5,
        repository=repository,
        service=service,
        workspace=service.build_workspace(env.document_id),
    )


@pytest.mark.parametrize(
    "model",
    [
        PlanSourceLocator,
        StandardSourceLocator,
        FindingWorkspaceItem,
        ReviewGapWorkspaceItem,
        InternalErrorWorkspaceItem,
        FindingsWorkspaceCounts,
        FindingsWorkspaceResult,
    ],
)
def test_public_models_are_frozen_and_extra_forbid(model):
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"


@pytest.mark.parametrize(
    "forbidden",
    [
        "overall_verdict",
        "overall_compliance",
        "safety_score",
        "compliance_score",
        "pass_rate",
        "risk_percentage",
        "risk_score",
        "severity",
    ],
)
def test_public_schema_contains_no_verdict_score_or_severity(forbidden):
    names = {
        name
        for model in (
            FindingsWorkspaceResult,
            FindingsWorkspaceCounts,
            FindingWorkspaceItem,
            ReviewGapWorkspaceItem,
            InternalErrorWorkspaceItem,
        )
        for name in model.model_fields
    }
    assert forbidden not in names


def test_public_seam_accepts_only_document_id():
    signature = inspect.signature(FindingsWorkspaceService.build_workspace)
    assert list(signature.parameters) == ["self", "document_id"]


@pytest.mark.parametrize(
    "forbidden",
    [
        "whole_plan_review_result",
        "candidate_traces",
        "findings",
        "comparisons",
        "plan_facts",
        "items",
        "counts",
        "pages",
        "severity",
        "sort_mode",
        "search_query",
    ],
)
def test_public_seam_rejects_caller_authority(tmp_path, forbidden):
    env = _workspace(tmp_path)
    with pytest.raises(TypeError):
        env.service.build_workspace(env.env.document_id, **{forbidden: object()})


def test_d5_is_invoked_exactly_once(tmp_path):
    env = _workspace(tmp_path)
    assert env.d5.calls == [env.env.document_id]


def test_empty_document_projects_no_items_without_a_verdict(tmp_path):
    env = _workspace(tmp_path, candidate_texts=())
    assert env.workspace.items == ()
    assert env.workspace.counts.findings == 0
    assert env.workspace.counts.review_gaps == 0
    assert "verdict" not in env.workspace.model_dump(mode="json")


@pytest.mark.parametrize(
    "decision",
    [
        ComparisonDecision.COMPLIANT,
        ComparisonDecision.NON_COMPLIANT,
        ComparisonDecision.INSUFFICIENT_INFORMATION,
    ],
)
def test_all_local_decisions_project_as_findings(tmp_path, decision):
    env = _workspace(tmp_path, decisions=[decision])
    item = env.workspace.items[0]
    trace = env.d5_result.candidate_traces[0].requirement_traces[0]
    assert isinstance(item, FindingWorkspaceItem)
    assert item.decision == decision
    assert item.finding_id == trace.finding_id
    assert item.comparison_id == trace.comparison_id


def test_finding_retains_exact_planfact_source_location(tmp_path):
    env = _workspace(tmp_path)
    item = env.workspace.items[0]
    binding = env.d5_result.candidate_traces[0].requirement_traces[0].extraction.binding
    assert item.plan_source.page_char_start == binding.page_char_start
    assert item.plan_source.page_char_end == binding.page_char_end
    assert item.plan_source.source_text == binding.source_text
    assert item.plan_source.plan_fact_id == binding.plan_fact.plan_fact_id


def test_finding_retains_exact_standard_requirement_location(tmp_path):
    env = _workspace(tmp_path)
    item = env.workspace.items[0]
    requirement = env.d5_result.candidate_traces[0].decomposition.requirements[0]
    assert item.standard_source.article_id == requirement.article_id
    assert item.standard_source.article_char_start == requirement.requirement_char_start
    assert item.standard_source.article_char_end == requirement.requirement_char_end
    assert item.standard_source.source_text == requirement.requirement_text


@pytest.mark.parametrize(
    "route_status, terminal",
    [
        (StandardRouteStatus.NO_STANDARD_SCOPE, WholePlanCandidateTerminalState.NO_STANDARD_SCOPE),
        (
            StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE,
            WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE,
        ),
    ],
)
def test_routing_stops_are_review_gaps(tmp_path, route_status, terminal):
    env = _workspace(tmp_path, route_statuses=[route_status])
    item = env.workspace.items[0]
    assert isinstance(item, ReviewGapWorkspaceItem)
    assert item.gap_source == ReviewGapSource.CANDIDATE_TERMINAL
    assert item.candidate_terminal_state == terminal
    assert item.standard_source is None


def test_evidence_stop_is_review_gap(tmp_path):
    from app.schemas.standards_retrieval import RetrievalDecision

    env = _workspace(tmp_path, retrieval=RetrievalDecision.NO_MATCH)
    item = env.workspace.items[0]
    assert item.candidate_terminal_state == WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED
    assert env.workspace.counts.evidence_gaps == 1


def test_requirement_source_stop_is_review_gap(tmp_path):
    env = _workspace(
        tmp_path, decomposition_status=RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED
    )
    item = env.workspace.items[0]
    assert item.candidate_terminal_state == (
        WholePlanCandidateTerminalState.REQUIREMENT_SOURCE_NOT_QUALIFIED
    )


@pytest.mark.parametrize(
    "status, terminal",
    [
        (PlanFactExtractionStatus.NOT_FOUND, WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND),
        (PlanFactExtractionStatus.AMBIGUOUS, WholePlanRequirementTerminalState.PLAN_FACT_AMBIGUOUS),
        (PlanFactExtractionStatus.UNSUPPORTED, WholePlanRequirementTerminalState.PLAN_FACT_UNSUPPORTED),
        (
            PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED,
            WholePlanRequirementTerminalState.PLAN_FACT_SOURCE_NOT_QUALIFIED,
        ),
    ],
)
def test_planfact_stops_are_requirement_gaps(tmp_path, status, terminal):
    env = _workspace(tmp_path, extraction_statuses=[status])
    item = env.workspace.items[0]
    assert item.gap_source == ReviewGapSource.REQUIREMENT_TERMINAL
    assert item.requirement_terminal_state == terminal
    assert item.requirement_id is not None
    assert item.standard_source.requirement_id == item.requirement_id
    assert env.workspace.counts.plan_fact_gaps == 1


@pytest.mark.parametrize(
    "status",
    [RequirementDecompositionStatus.UNRESOLVED, RequirementDecompositionStatus.PARTIAL],
)
def test_each_unresolved_span_is_visible_as_review_gap(tmp_path, status):
    env = _workspace(tmp_path, decomposition_status=status)
    unresolved = env.d5_result.candidate_traces[0].decomposition.unresolved_spans
    gaps = [item for item in env.workspace.items if isinstance(item, ReviewGapWorkspaceItem)]
    assert {item.unresolved_span_id for item in gaps} == {
        item.unresolved_span_id for item in unresolved
    }


def test_partial_keeps_finding_and_unresolved_gap(tmp_path):
    env = _workspace(tmp_path, decomposition_status=RequirementDecompositionStatus.PARTIAL)
    assert [item.item_kind for item in env.workspace.items] == [
        WorkspaceItemKind.REVIEW_GAP,
        WorkspaceItemKind.FINDING,
    ]
    assert env.workspace.counts.findings == 1
    assert env.workspace.counts.partial_unresolved_gaps == 1


def test_two_unresolved_spans_create_two_distinct_gap_items(tmp_path):
    env = _workspace(tmp_path, decomposition_status=RequirementDecompositionStatus.PARTIAL)
    original = env.d5_result.candidate_traces[0]
    decomposition = original.decomposition
    first = decomposition.unresolved_spans[0]
    text = "shall"
    values = dict(
        evidence_id=first.evidence_id,
        article_id=first.article_id,
        article_number=first.article_number,
        char_start=8,
        char_end=8 + len(text),
        source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        reason=UnresolvedRequirementReason.CONDITIONAL_UNSUPPORTED,
    )
    second = UnresolvedRequirementSpan(
        unresolved_span_id=deterministic_unresolved_span_id(**values),
        source_text=text,
        **values,
    )
    unresolved = (*decomposition.unresolved_spans, second)
    decomposition_values = dict(
        candidate_id=decomposition.candidate_id,
        route_id=decomposition.route_id,
        evidence_ids=decomposition.evidence_ids,
        requirements=(),
        unresolved_spans=unresolved,
    )
    new_decomposition = RequirementDecompositionResult(
        **{
            **decomposition.model_dump(
                exclude={"decomposition_id", "requirements", "unresolved_spans", "status"}
            ),
            "decomposition_id": deterministic_decomposition_id(**decomposition_values),
            "requirements": (),
            "unresolved_spans": unresolved,
            "status": RequirementDecompositionStatus.UNRESOLVED,
        }
    )
    trace_values = dict(
        candidate=original.candidate,
        route=original.route,
        review_unit_id=original.review_unit_id,
        preparation_status=original.preparation_status,
        retrieval_decision=original.retrieval_decision,
        evidence_ids=original.evidence_ids,
        decomposition=new_decomposition,
        requirement_traces=(),
        terminal_state=WholePlanCandidateTerminalState.REQUIREMENT_UNRESOLVED,
        internal_error=original.internal_error,
    )
    new_trace = WholePlanCandidateTrace(
        candidate_trace_id=deterministic_candidate_trace_id(**trace_values),
        **trace_values,
    )
    traces = (new_trace,)
    counts = derive_coverage_counts(traces)
    run_values = dict(
        document_id=env.d5_result.document_id,
        document_sha256=env.d5_result.document_sha256,
        processing_state=env.d5_result.processing_state,
        candidate_traces=traces,
    )
    result = WholePlanReviewResult(
        whole_plan_review_id=deterministic_whole_plan_review_id(**run_values),
        coverage_counts=counts,
        **run_values,
    )
    service = FindingsWorkspaceService(
        settings=env.env.settings,
        whole_plan_review_service=_D5(result),
        standard_repository=_Repository(result),
    )
    workspace = service.build_workspace(result.document_id)
    gaps = [item for item in workspace.items if item.item_kind == WorkspaceItemKind.REVIEW_GAP]
    assert len(gaps) == 2
    assert len({item.review_gap_id for item in gaps}) == 2


def test_candidate_internal_error_is_projected_without_raw_text(tmp_path):
    from app.schemas.whole_plan_review import WholePlanStage

    env = _workspace(tmp_path, routing_failure=WholePlanStage.ROUTING)
    item = env.workspace.items[0]
    assert isinstance(item, InternalErrorWorkspaceItem)
    assert item.error.stage == WholePlanStage.ROUTING
    assert set(type(item.error).model_fields) == {"stage", "code"}
    assert env.workspace.counts.internal_errors == 1


def test_requirement_internal_error_is_projected(tmp_path):
    env = _workspace(tmp_path, extraction_failure_at=0)
    item = env.workspace.items[0]
    assert isinstance(item, InternalErrorWorkspaceItem)
    assert item.requirement_trace_id is not None
    assert item.requirement_id is not None


def test_mixed_finding_gap_and_error_all_remain_visible(tmp_path):
    first = _workspace(
        tmp_path / "a",
        candidate_texts=("gap shall not exceed 3mm.", "height shall not exceed 5m."),
        route_statuses=[StandardRouteStatus.ROUTED, StandardRouteStatus.NO_STANDARD_SCOPE],
    )
    assert {item.item_kind for item in first.workspace.items} == {
        WorkspaceItemKind.FINDING,
        WorkspaceItemKind.REVIEW_GAP,
    }
    second = _workspace(tmp_path / "b", routing_failure=__import__(
        "app.schemas.whole_plan_review", fromlist=["WholePlanStage"]
    ).WholePlanStage.ROUTING)
    assert second.workspace.items[0].item_kind == WorkspaceItemKind.INTERNAL_ERROR


def test_zero_findings_with_gaps_never_becomes_compliant(tmp_path):
    env = _workspace(tmp_path, route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE])
    payload = env.workspace.model_dump(mode="json")
    assert payload["counts"]["findings"] == 0
    assert payload["counts"]["review_gaps"] == 1
    assert "plan_compliant" not in payload


def test_workspace_and_item_identities_are_repeatable(tmp_path):
    env = _workspace(tmp_path)
    again = env.service.build_workspace(env.env.document_id)
    assert again.workspace_id == env.workspace.workspace_id
    assert [workspace_item_authority_id(item) for item in again.items] == [
        workspace_item_authority_id(item) for item in env.workspace.items
    ]


def test_workspace_identity_binds_ordered_complete_items(tmp_path):
    env = _workspace(tmp_path)
    assert env.workspace.workspace_id == deterministic_workspace_id(
        whole_plan_review_id=env.workspace.whole_plan_review_id,
        items=env.workspace.items,
    )


def test_default_order_is_exact_source_order(tmp_path):
    env = _workspace(
        tmp_path,
        candidate_texts=("gap shall not exceed 3mm.", "height shall not exceed 5m."),
    )
    assert env.workspace.items == tuple(
        sorted(env.workspace.items, key=workspace_item_sort_key)
    )
    assert [item.plan_source.page_char_start for item in env.workspace.items] == sorted(
        item.plan_source.page_char_start for item in env.workspace.items
    )


def test_same_text_at_distinct_offsets_is_not_merged(tmp_path):
    text = "gap shall not exceed 3mm."
    env = _workspace(tmp_path, candidate_texts=(text, text))
    assert len(env.workspace.items) == 2
    assert len({item.candidate_id for item in env.workspace.items}) == 2
    assert len({item.plan_source.page_char_start for item in env.workspace.items}) == 2


def test_counts_are_derived_from_complete_items(tmp_path):
    env = _workspace(tmp_path, decomposition_status=RequirementDecompositionStatus.PARTIAL)
    assert env.workspace.counts == derive_workspace_counts(env.workspace.items)
    assert env.workspace.counts.findings == env.d5_result.coverage_counts.findings_created
    assert env.workspace.counts.internal_errors == env.d5_result.coverage_counts.internal_errors


@pytest.mark.parametrize(
    "decision, field",
    [
        (ComparisonDecision.COMPLIANT, "finding_compliant"),
        (ComparisonDecision.NON_COMPLIANT, "finding_non_compliant"),
        (ComparisonDecision.INSUFFICIENT_INFORMATION, "finding_insufficient_information"),
    ],
)
def test_local_decision_counts_reconcile(tmp_path, decision, field):
    env = _workspace(tmp_path, decisions=[decision])
    assert getattr(env.workspace.counts, field) == 1
    assert env.workspace.counts.findings == 1


def test_standard_metadata_checksum_mismatch_fails_closed(tmp_path):
    env = _workspace(tmp_path)
    service = FindingsWorkspaceService(
        settings=env.env.settings,
        whole_plan_review_service=_D5(env.d5_result),
        standard_repository=_Repository(env.d5_result, wrong_checksum=True),
    )
    with pytest.raises(FindingsWorkspaceSourceNavigationError):
        service.build_workspace(env.env.document_id)


def test_standard_article_mismatch_fails_closed(tmp_path):
    env = _workspace(tmp_path)
    service = FindingsWorkspaceService(
        settings=env.env.settings,
        whole_plan_review_service=_D5(env.d5_result),
        standard_repository=_Repository(env.d5_result, wrong_article=True),
    )
    with pytest.raises(FindingsWorkspaceSourceNavigationError):
        service.build_workspace(env.env.document_id)


def test_standard_lookup_is_cached_per_qualified_identity(tmp_path):
    env = _workspace(tmp_path, requirement_count=2)
    assert env.repository.document_calls == [env.repository.document_calls[0]]
    assert env.repository.article_calls == [env.repository.article_calls[0]]


def test_d5_cross_document_result_is_rejected(tmp_path):
    env = _workspace(tmp_path)
    with pytest.raises(FindingsWorkspaceAuthorityInvariantError):
        env.service.build_workspace("different-document")


def test_duplicate_finding_identity_is_rejected_by_result_schema(tmp_path):
    env = _workspace(tmp_path)
    item = env.workspace.items[0]
    with pytest.raises(ValidationError):
        FindingsWorkspaceResult(
            **{
                **env.workspace.model_dump(),
                "items": (item, item),
                "counts": derive_workspace_counts((item, item)),
            }
        )


@pytest.mark.parametrize("field", ["page_char_start", "page_char_end", "source_text", "source_text_sha256"])
def test_plan_locator_tampering_is_rejected(tmp_path, field):
    env = _workspace(tmp_path)
    data = env.workspace.items[0].plan_source.model_dump()
    data[field] = {"page_char_start": 1, "page_char_end": 1, "source_text": "wrong", "source_text_sha256": "0" * 64}[field]
    with pytest.raises(ValidationError):
        PlanSourceLocator(**data)


@pytest.mark.parametrize("field", ["article_char_start", "article_char_end", "source_text", "source_text_sha256"])
def test_standard_locator_tampering_is_rejected(tmp_path, field):
    env = _workspace(tmp_path)
    data = env.workspace.items[0].standard_source.model_dump()
    data[field] = {"article_char_start": 1, "article_char_end": 1, "source_text": "wrong", "source_text_sha256": "0" * 64}[field]
    with pytest.raises(ValidationError):
        StandardSourceLocator(**data)


@pytest.mark.parametrize(
    "module_name",
    [
        "app.services.review.compliance_comparison_service",
        "app.services.review.review_finding_service",
        "app.services.review.review_action_service",
        "requests",
        "httpx",
        "openai",
        "rapidocr",
    ],
)
def test_projection_service_has_no_forbidden_runtime_import(module_name):
    import app.services.review.findings_workspace_service as module

    source = inspect.getsource(module)
    assert module_name not in source


@pytest.mark.parametrize(
    "term",
    [
        "HIGH",
        "MEDIUM",
        "LOW",
        "PLAN_COMPLIANT",
        "PLAN_NON_COMPLIANT",
        "risk_score",
        "pass_rate",
        "report_export",
    ],
)
def test_production_schema_does_not_invent_authority(term):
    import app.schemas.findings_workspace as schema_module

    assert term not in inspect.getsource(schema_module)


def test_case_a_compatible_compliant_finding(tmp_path):
    env = _workspace(tmp_path, decisions=[ComparisonDecision.COMPLIANT])
    assert env.workspace.items[0].item_kind == WorkspaceItemKind.FINDING
    assert env.workspace.items[0].decision == ComparisonDecision.COMPLIANT


def test_case_b_separates_plan_and_normative_sources(tmp_path):
    env = _workspace(tmp_path, decisions=[ComparisonDecision.NON_COMPLIANT])
    trace = env.d5_result.candidate_traces[0]
    requirement = trace.decomposition.requirements[0]
    normative_text = "gap shall not exceed 2.5mm"
    normative = type(requirement)(
        **{
            **requirement.model_dump(
                exclude={
                    "requirement_char_end",
                    "requirement_text",
                    "requirement_text_sha256",
                    "value",
                }
            ),
            "requirement_char_end": requirement.requirement_char_start + len(normative_text),
            "requirement_text": normative_text,
            "requirement_text_sha256": hashlib.sha256(normative_text.encode()).hexdigest(),
            "value": Decimal("2.5"),
        }
    )
    decomposition = RequirementDecompositionResult(
        **{
            **trace.decomposition.model_dump(exclude={"requirements"}),
            "requirements": (normative,),
        }
    )
    changed_trace = WholePlanCandidateTrace(
        **{
            **trace.model_dump(exclude={"decomposition"}),
            "decomposition": decomposition,
        }
    )
    result = WholePlanReviewResult(
        **{
            **env.d5_result.model_dump(exclude={"candidate_traces"}),
            "candidate_traces": (changed_trace,),
        }
    )
    workspace = FindingsWorkspaceService(
        settings=env.env.settings,
        whole_plan_review_service=_D5(result),
        standard_repository=_Repository(result),
    ).build_workspace(result.document_id)
    item = workspace.items[0]
    assert item.plan_source.source_text == "gap shall not exceed 3mm."
    assert item.standard_source.source_text == normative_text
    assert "200" not in item.plan_source.source_text
    assert "200" not in item.standard_source.source_text


def test_case_c_is_gap_without_fabricated_downstream_authority(tmp_path):
    env = _workspace(tmp_path, extraction_statuses=[PlanFactExtractionStatus.NOT_FOUND])
    item = env.workspace.items[0]
    assert isinstance(item, ReviewGapWorkspaceItem)
    assert item.requirement_terminal_state == WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND
    payload = item.model_dump(mode="json")
    assert "finding_id" not in payload
    assert "comparison_id" not in payload
    assert "plan_fact_id" not in payload
