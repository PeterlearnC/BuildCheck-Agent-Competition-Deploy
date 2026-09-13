"""D.7-P2 deterministic semantic ReportModel qualification tests."""

from __future__ import annotations

import inspect
from datetime import timedelta
from pathlib import Path
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.finding_review import FindingReviewDisposition, ReportInclusionStatus
from app.schemas.findings_workspace import WorkspaceItemKind, deterministic_workspace_id
from app.schemas.human_review import (
    ReviewGapHumanAction,
)
from app.schemas.review_report import (
    ExcludedReportItemReference,
    ReportCounts,
    ReportCoverageStatement,
    ReportDocumentIdentity,
    ReportFindingEntry,
    ReportGapEntry,
    ReportInclusionBasis,
    ReportInternalErrorEntry,
    ReportProcessingSummary,
    ReviewReportModel,
    derive_report_counts,
)
from app.schemas.standard_route import StandardRouteStatus
from app.schemas.whole_plan_review import WholePlanStage
from app.services.review.human_review_service import (
    HumanReviewService,
    HumanReviewStaleWorkspaceError,
)
from app.services.review.review_report_service import (
    ReviewReportAuthorityInvariantError,
    ReviewReportService,
)
from tests.test_human_review_d7 import (
    FIXED_TIME,
    _WorkspaceService,
    _command,
    _human_env,
)
from tests.test_findings_workspace_d6 import _workspace


class _P1Authority:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def review_document(self, document_id, commands):
        self.calls.append((document_id, commands))
        if self.error is not None:
            raise self.error
        return self.result


def _report_env(
    tmp_path,
    *,
    kind=WorkspaceItemKind.FINDING,
    decision=None,
    inclusion=None,
    action=None,
    reviewed=True,
    note="Human note.",
):
    source, _, p1 = _human_env(tmp_path, kind=kind, decision=decision)
    commands = ()
    if reviewed:
        commands = (
            _command(
                source.workspace,
                action=action,
                inclusion=inclusion or ReportInclusionStatus.UNDECIDED,
                note=note,
            ),
        )
    human = p1.review_document(source.env.document_id, commands)
    authority = _P1Authority(human)
    service = ReviewReportService(human_review_service=authority)
    report = service.build_report(source.env.document_id, commands)
    return source, human, authority, service, report


PUBLIC_MODELS = (
    ReportDocumentIdentity,
    ReportProcessingSummary,
    ReportFindingEntry,
    ReportGapEntry,
    ReportInternalErrorEntry,
    ExcludedReportItemReference,
    ReportCounts,
    ReportCoverageStatement,
    ReviewReportModel,
)


@pytest.mark.parametrize("model", PUBLIC_MODELS)
def test_all_public_models_are_frozen_and_extra_forbid(model):
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"


def test_public_models_use_no_mutable_defaults_or_any_authority_fields():
    for model in PUBLIC_MODELS:
        for field in model.model_fields.values():
            assert not isinstance(field.default, (list, dict, set))
            annotation = field.annotation
            assert annotation is not Any
            assert Any not in get_args(annotation)


@pytest.mark.parametrize(
    "forbidden",
    [
        "whole_plan_verdict",
        "overall_compliance",
        "final_compliance",
        "effective_compliance",
        "severity",
        "risk_score",
        "compliance_percentage",
        "renderer",
        "output_path",
    ],
)
def test_schema_has_no_verdict_risk_score_or_formatting_fields(forbidden):
    names = {name for model in PUBLIC_MODELS for name in model.model_fields}
    assert forbidden not in names


def test_tuple_collections_are_immutable():
    for name in ("findings", "review_gaps", "internal_errors", "excluded_items"):
        annotation = ReviewReportModel.model_fields[name].annotation
        assert get_origin(annotation) is tuple


def test_extra_fields_rejected(tmp_path):
    report = _report_env(tmp_path)[-1]
    payload = report.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ReviewReportModel.model_validate(payload)


@pytest.mark.parametrize("model", PUBLIC_MODELS)
def test_public_models_are_runtime_immutable(tmp_path, model):
    report = _report_env(tmp_path)[-1]
    instances = {
        ReportDocumentIdentity: report.document,
        ReportProcessingSummary: report.processing,
        ReportFindingEntry: report.findings[0],
        ReportGapEntry: _report_env(
            tmp_path / "gap", kind=WorkspaceItemKind.REVIEW_GAP
        )[-1].review_gaps[0],
        ReportInternalErrorEntry: _report_env(
            tmp_path / "error", kind=WorkspaceItemKind.INTERNAL_ERROR
        )[-1].internal_errors[0],
        ExcludedReportItemReference: _report_env(
            tmp_path / "excluded", inclusion=ReportInclusionStatus.EXCLUDE
        )[-1].excluded_items[0],
        ReportCounts: report.counts,
        ReportCoverageStatement: report.coverage,
        ReviewReportModel: report,
    }
    instance = instances[model]
    field = next(iter(model.model_fields))
    with pytest.raises(ValidationError):
        setattr(instance, field, getattr(instance, field))


def test_public_seam_accepts_only_document_and_typed_commands():
    signature = inspect.signature(ReviewReportService.build_report)
    assert list(signature.parameters) == ["self", "document_id", "commands"]


@pytest.mark.parametrize(
    "forbidden",
    [
        "human_review_result",
        "workspace",
        "items",
        "findings",
        "entries",
        "counts",
        "machine_decision",
        "source_locator",
    ],
)
def test_public_seam_rejects_caller_authority(tmp_path, forbidden):
    source, _, _, service, _ = _report_env(tmp_path)
    with pytest.raises(TypeError):
        service.build_report(source.env.document_id, (), **{forbidden: object()})


def test_p1_called_exactly_once_with_exact_inputs(tmp_path):
    source, _, authority, _, _ = _report_env(tmp_path)
    assert len(authority.calls) == 1
    assert authority.calls[0][0] == source.env.document_id
    assert len(authority.calls[0][1]) == 1


def test_invalid_p1_type_and_cross_document_result_abort(tmp_path):
    source, human, authority, service, _ = _report_env(tmp_path)
    authority.result = object()
    with pytest.raises(ReviewReportAuthorityInvariantError):
        service.build_report(source.env.document_id, ())
    authority.result = human
    with pytest.raises(ReviewReportAuthorityInvariantError):
        service.build_report("another-document", ())


def test_stale_p1_failure_propagates_without_report():
    stale = HumanReviewStaleWorkspaceError("stale")
    authority = _P1Authority(error=stale)
    service = ReviewReportService(human_review_service=authority)
    with pytest.raises(HumanReviewStaleWorkspaceError):
        service.build_report("document", ())
    assert len(authority.calls) == 1


@pytest.mark.parametrize("decision", list(ComparisonDecision))
def test_finding_entry_retains_exact_machine_authority(tmp_path, decision):
    source, _, _, _, report = _report_env(tmp_path, decision=decision)
    item = source.workspace.items[0]
    entry = report.findings[0]
    assert entry.item_kind == WorkspaceItemKind.FINDING
    assert entry.source_item_id == item.finding_id == entry.finding_id
    assert entry.comparison_id == item.comparison_id
    assert entry.machine_decision == item.decision == decision
    assert entry.reason_code == item.reason_code
    assert entry.plan_source == item.plan_source
    assert entry.standard_source == item.standard_source


@pytest.mark.parametrize("decision", list(ComparisonDecision))
def test_human_action_never_rewrites_machine_decision(tmp_path, decision):
    _, _, _, _, report = _report_env(
        tmp_path,
        decision=decision,
        action=FindingReviewDisposition.REJECTED,
    )
    entry = report.findings[0]
    assert entry.machine_decision == decision
    assert entry.human_review.human_action.value == "REJECTED"


def test_finding_entry_contains_no_reconstructed_c3_semantics(tmp_path):
    entry = _report_env(tmp_path)[-1].findings[0]
    forbidden = {
        "summary",
        "narrative",
        "severity",
        "risk_category",
        "missing_information",
        "recommendation",
        "citation_narrative",
    }
    assert forbidden.isdisjoint(type(entry).model_fields)


def test_gap_entry_retains_exact_status_and_human_metadata(tmp_path):
    source, _, _, _, report = _report_env(
        tmp_path,
        kind=WorkspaceItemKind.REVIEW_GAP,
        action=ReviewGapHumanAction.NEEDS_FOLLOW_UP,
    )
    item = source.workspace.items[0]
    entry = report.review_gaps[0]
    assert entry.review_gap_id == item.review_gap_id
    assert entry.gap_source == item.gap_source
    assert entry.candidate_terminal_state == item.candidate_terminal_state
    assert entry.human_review.human_action.value == "NEEDS_FOLLOW_UP"
    assert not report.findings


@pytest.mark.parametrize(
    "route_status",
    [StandardRouteStatus.NO_STANDARD_SCOPE, StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE],
)
def test_candidate_gap_variants_remain_exact_statuses(tmp_path, route_status):
    source = _workspace(tmp_path, route_statuses=[route_status])
    human = HumanReviewService(
        findings_workspace_service=_WorkspaceService(source.workspace),
        clock=lambda: FIXED_TIME,
    ).review_document(source.env.document_id, ())
    report = ReviewReportService._project_report(human)
    assert report.review_gaps[0].candidate_terminal_state.value == route_status.value
    assert not report.findings


def test_error_entry_retains_finite_stage_and_code(tmp_path):
    source, _, _, _, report = _report_env(
        tmp_path, kind=WorkspaceItemKind.INTERNAL_ERROR
    )
    item = source.workspace.items[0]
    entry = report.internal_errors[0]
    assert entry.internal_error_item_id == item.internal_error_item_id
    assert entry.error_stage == item.error.stage
    assert entry.error_code == item.error.code
    assert not report.findings
    assert "exception" not in entry.model_dump(mode="json")


@pytest.mark.parametrize(
    ("kind", "section"),
    [
        (WorkspaceItemKind.FINDING, "findings"),
        (WorkspaceItemKind.REVIEW_GAP, "review_gaps"),
        (WorkspaceItemKind.INTERNAL_ERROR, "internal_errors"),
    ],
)
@pytest.mark.parametrize(
    ("inclusion", "basis"),
    [
        (ReportInclusionStatus.INCLUDE, ReportInclusionBasis.EXPLICIT_INCLUDE),
        (
            ReportInclusionStatus.UNDECIDED,
            ReportInclusionBasis.UNDECIDED_DEFAULT_INCLUDE,
        ),
    ],
)
def test_include_and_undecided_create_detailed_entries(
    tmp_path, kind, section, inclusion, basis
):
    report = _report_env(tmp_path, kind=kind, inclusion=inclusion)[-1]
    entry = getattr(report, section)[0]
    assert entry.inclusion_basis == basis
    assert not report.excluded_items


@pytest.mark.parametrize(
    ("kind", "section"),
    [
        (WorkspaceItemKind.FINDING, "findings"),
        (WorkspaceItemKind.REVIEW_GAP, "review_gaps"),
        (WorkspaceItemKind.INTERNAL_ERROR, "internal_errors"),
    ],
)
def test_unreviewed_items_are_detailed_without_invented_human_metadata(
    tmp_path, kind, section
):
    report = _report_env(tmp_path, kind=kind, reviewed=False)[-1]
    entry = getattr(report, section)[0]
    assert entry.inclusion_basis == ReportInclusionBasis.UNREVIEWED_DEFAULT_INCLUDE
    assert entry.human_review is None
    assert report.counts.human_unreviewed == 1


@pytest.mark.parametrize(
    ("kind", "section", "status_field"),
    [
        (WorkspaceItemKind.FINDING, "findings", "machine_decision"),
        (WorkspaceItemKind.REVIEW_GAP, "review_gaps", "gap_source"),
        (WorkspaceItemKind.INTERNAL_ERROR, "internal_errors", "error_code"),
    ],
)
def test_exclusion_omits_detail_but_preserves_audit_authority(
    tmp_path, kind, section, status_field
):
    report = _report_env(
        tmp_path, kind=kind, inclusion=ReportInclusionStatus.EXCLUDE
    )[-1]
    assert not getattr(report, section)
    excluded = report.excluded_items[0]
    assert excluded.item_kind == kind
    assert excluded.report_inclusion == ReportInclusionStatus.EXCLUDE
    assert excluded.review_record_id.startswith("humanreview_")
    assert getattr(excluded, status_field) is not None
    assert report.counts.human_reviewed == 1


def test_excluded_non_compliant_finding_remains_accounted(tmp_path):
    report = _report_env(
        tmp_path,
        decision=ComparisonDecision.NON_COMPLIANT,
        inclusion=ReportInclusionStatus.EXCLUDE,
    )[-1]
    assert report.excluded_items[0].machine_decision == ComparisonDecision.NON_COMPLIANT
    assert report.counts.workspace_findings == 1
    assert report.counts.machine_finding_non_compliant == 1
    assert report.counts.included_findings == 0
    assert report.counts.excluded_findings == 1


@pytest.mark.parametrize("kind", list(WorkspaceItemKind))
def test_every_machine_item_has_exactly_one_accounting_path(tmp_path, kind):
    included = _report_env(tmp_path / "included", kind=kind, reviewed=False)[-1]
    excluded = _report_env(
        tmp_path / "excluded", kind=kind, inclusion=ReportInclusionStatus.EXCLUDE
    )[-1]
    assert sum(
        len(section)
        for section in (
            included.findings,
            included.review_gaps,
            included.internal_errors,
            included.excluded_items,
        )
    ) == 1
    assert sum(
        len(section)
        for section in (
            excluded.findings,
            excluded.review_gaps,
            excluded.internal_errors,
            excluded.excluded_items,
        )
    ) == 1


def test_report_counts_are_derived_and_reconcile_with_workspace(tmp_path):
    report = _report_env(tmp_path, decision=ComparisonDecision.COMPLIANT)[-1]
    assert report.counts == derive_report_counts(
        findings=report.findings,
        review_gaps=report.review_gaps,
        internal_errors=report.internal_errors,
        excluded_items=report.excluded_items,
    )
    assert report.counts.workspace_findings == report.processing.workspace_counts.findings
    assert report.counts.human_reviewed == report.processing.human_review_counts.reviewed_items


@pytest.mark.parametrize(
    "field",
    [
        "workspace_findings",
        "machine_finding_compliant",
        "human_reviewed",
        "human_unreviewed",
        "included_findings",
        "excluded_findings",
    ],
)
def test_tampered_report_counts_are_rejected(tmp_path, field):
    report = _report_env(tmp_path, decision=ComparisonDecision.COMPLIANT)[-1]
    payload = report.model_dump()
    payload["counts"][field] += 1
    with pytest.raises(ValidationError):
        ReviewReportModel.model_validate(payload)


@pytest.mark.parametrize(
    "forbidden", ["percentage", "ratio", "score", "overall", "verdict", "pass_rate"]
)
def test_counts_and_root_have_no_ratio_or_whole_plan_fields(forbidden):
    names = set(ReportCounts.model_fields) | set(ReviewReportModel.model_fields)
    assert all(forbidden not in name.lower() for name in names)


def test_coverage_statement_is_fixed_and_conservative():
    coverage = ReportCoverageStatement()
    lowered = coverage.statement.lower()
    assert "separately preserves" in lowered
    assert "compliant" not in lowered
    assert "all requirements" not in lowered
    assert "no problems" not in lowered


def test_workflow_completeness_is_explicitly_human_only(tmp_path):
    report = _report_env(tmp_path, reviewed=False)[-1]
    assert report.processing.human_review_workflow_completeness.value == "UNREVIEWED"
    assert "compliance" not in ReportProcessingSummary.model_fields


def test_reviewer_identity_note_and_audit_time_are_exact_passthrough(tmp_path):
    note = "Reviewer-authored observation."
    report = _report_env(tmp_path, note=note)[-1]
    review = report.findings[0].human_review
    assert review.reviewer_id == "reviewer-one"
    assert review.reviewer_identity_assurance.value == "CALLER_ASSERTED_UNVERIFIED"
    assert review.reviewer_note == note
    assert review.created_at == FIXED_TIME


def test_created_at_does_not_change_entry_or_report_identity(tmp_path):
    source, authority, first_p1 = _human_env(tmp_path)
    command = _command(source.workspace)
    first = first_p1.review_document(source.env.document_id, (command,))
    later_p1 = HumanReviewService(
        findings_workspace_service=authority,
        clock=lambda: FIXED_TIME + timedelta(days=1),
    )
    second = later_p1.review_document(source.env.document_id, (command,))
    first_report = ReviewReportService._project_report(first)
    second_report = ReviewReportService._project_report(second)
    assert first.records[0].created_at != second.records[0].created_at
    assert first_report.findings[0].report_entry_id == second_report.findings[0].report_entry_id
    assert first_report.report_model_id == second_report.report_model_id


def test_report_identities_are_deterministic(tmp_path):
    _, human, _, _, first = _report_env(tmp_path)
    second = ReviewReportService._project_report(human)
    assert first.report_model_id == second.report_model_id
    assert first.findings[0].report_entry_id == second.findings[0].report_entry_id


def test_inclusion_and_review_set_changes_change_report_identity(tmp_path):
    source, authority, p1 = _human_env(tmp_path)
    undecided = p1.review_document(
        source.env.document_id, (_command(source.workspace),)
    )
    included = p1.review_document(
        source.env.document_id,
        (_command(source.workspace, inclusion=ReportInclusionStatus.INCLUDE),),
    )
    first = ReviewReportService._project_report(undecided)
    second = ReviewReportService._project_report(included)
    assert undecided.review_set_id != included.review_set_id
    assert first.report_model_id != second.report_model_id
    assert first.findings[0].report_entry_id != second.findings[0].report_entry_id


def test_source_locators_are_lossless_and_same_text_is_not_a_merge_key(tmp_path):
    source, _, _, _, report = _report_env(tmp_path)
    item = source.workspace.items[0]
    entry = report.findings[0]
    assert entry.plan_source.model_dump() == item.plan_source.model_dump()
    assert entry.standard_source.model_dump() == item.standard_source.model_dump()
    assert entry.source_item_id == item.finding_id


def test_same_text_at_distinct_offsets_remains_two_report_entries(tmp_path):
    source = _workspace(
        tmp_path,
        candidate_texts=("same requirement text", "same requirement text"),
        decisions=(ComparisonDecision.COMPLIANT, ComparisonDecision.COMPLIANT),
    )
    human = HumanReviewService(
        findings_workspace_service=_WorkspaceService(source.workspace),
        clock=lambda: FIXED_TIME,
    ).review_document(source.env.document_id, ())
    report = ReviewReportService._project_report(human)
    assert len(report.findings) == 2
    assert report.findings[0].plan_source.source_text == report.findings[1].plan_source.source_text
    assert report.findings[0].plan_source.page_char_start != report.findings[1].plan_source.page_char_start
    assert report.findings[0].source_item_id != report.findings[1].source_item_id
    assert report.findings[0].report_entry_id != report.findings[1].report_entry_id


def test_mixed_finding_and_gap_preserve_separate_sections_and_global_ordinals(tmp_path):
    source = _workspace(
        tmp_path,
        candidate_texts=("first requirement", "second requirement"),
        route_statuses=(StandardRouteStatus.ROUTED, StandardRouteStatus.NO_STANDARD_SCOPE),
    )
    human = HumanReviewService(
        findings_workspace_service=_WorkspaceService(source.workspace),
        clock=lambda: FIXED_TIME,
    ).review_document(source.env.document_id, ())
    report = ReviewReportService._project_report(human)
    assert len(report.findings) == 1
    assert len(report.review_gaps) == 1
    assert sorted(
        [report.findings[0].source_ordinal, report.review_gaps[0].source_ordinal]
    ) == [0, 1]


def test_report_rejects_document_locator_substitution(tmp_path):
    report = _report_env(tmp_path)[-1]
    payload = report.model_dump()
    payload["findings"][0]["plan_source"]["document_id"] = "wrong-document"
    with pytest.raises(ValidationError):
        ReviewReportModel.model_validate(payload)


def test_report_rejects_missing_or_duplicate_accounting(tmp_path):
    report = _report_env(tmp_path)[-1]
    missing = report.model_dump()
    missing["findings"] = []
    with pytest.raises(ValidationError):
        ReviewReportModel.model_validate(missing)
    duplicate = report.model_dump()
    duplicate["findings"] = tuple(duplicate["findings"]) + (
        dict(duplicate["findings"][0]),
    )
    with pytest.raises(ValidationError):
        ReviewReportModel.model_validate(duplicate)


def test_tampered_entry_and_report_identity_rejected(tmp_path):
    report = _report_env(tmp_path)[-1]
    entry_payload = report.findings[0].model_dump()
    entry_payload["report_entry_id"] = "reportentry_" + "0" * 64
    with pytest.raises(ValidationError):
        ReportFindingEntry.model_validate(entry_payload)
    report_payload = report.model_dump()
    report_payload["report_model_id"] = "reviewreport_" + "0" * 64
    with pytest.raises(ValidationError):
        ReviewReportModel.model_validate(report_payload)


def test_tampered_excluded_reference_identity_rejected(tmp_path):
    report = _report_env(
        tmp_path, inclusion=ReportInclusionStatus.EXCLUDE
    )[-1]
    payload = report.excluded_items[0].model_dump()
    payload["excluded_reference_id"] = "excludedreportitem_" + "0" * 64
    with pytest.raises(ValidationError):
        ExcludedReportItemReference.model_validate(payload)


def test_zero_item_workspace_is_empty_semantic_projection(tmp_path):
    source, _, p1 = _human_env(tmp_path)
    workspace = source.workspace.model_copy(
        update={
            "items": (),
            "workspace_id": deterministic_workspace_id(
                whole_plan_review_id=source.workspace.whole_plan_review_id,
                items=(),
            ),
            "counts": source.workspace.counts.model_copy(
                update={
                    "findings": 0,
                    "finding_compliant": 0,
                    "finding_non_compliant": 0,
                    "finding_insufficient_information": 0,
                }
            ),
            "source_coverage_counts": source.workspace.source_coverage_counts.model_copy(
                update={
                    "findings_created": 0,
                    "comparisons_completed": 0,
                    "comparison_compliant": 0,
                    "comparison_non_compliant": 0,
                    "comparison_insufficient_information": 0,
                }
            ),
        }
    )
    # Revalidate the altered synthetic authority before P1 consumes it.
    workspace = type(source.workspace).model_validate(workspace.model_dump())
    human_service = HumanReviewService(
        findings_workspace_service=_WorkspaceService(workspace),
        clock=lambda: FIXED_TIME,
    )
    human = human_service.review_document(source.env.document_id, ())
    report = ReviewReportService._project_report(human)
    assert not report.findings and not report.review_gaps
    assert not report.internal_errors and not report.excluded_items
    assert report.counts.human_unreviewed == 0
    assert report.processing.human_review_workflow_completeness.value == "REVIEWED"
    serialized = report.model_dump_json().lower()
    assert "plan compliant" not in serialized
    assert "no problems found" not in serialized


def test_case_a_compliant_machine_and_rejected_human_remain_separate(tmp_path):
    report = _report_env(
        tmp_path,
        decision=ComparisonDecision.COMPLIANT,
        action=FindingReviewDisposition.REJECTED,
    )[-1]
    assert report.findings[0].machine_decision == ComparisonDecision.COMPLIANT
    assert report.findings[0].human_review.human_action.value == "REJECTED"


def test_case_b_non_compliant_sources_are_exact_without_nearby_contamination(tmp_path):
    source, _, _, _, report = _report_env(
        tmp_path, decision=ComparisonDecision.NON_COMPLIANT
    )
    entry = report.findings[0]
    assert entry.machine_decision == ComparisonDecision.NON_COMPLIANT
    assert entry.plan_source == source.workspace.items[0].plan_source
    assert entry.standard_source == source.workspace.items[0].standard_source
    assert "200" not in entry.plan_source.source_text


def test_case_c_is_gap_without_machine_fabrication(tmp_path):
    report = _report_env(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)[-1]
    assert len(report.review_gaps) == 1
    assert not report.findings
    assert report.review_gaps[0].human_review.machine_decision is None
    assert report.review_gaps[0].human_review.finding_id is None
    assert report.review_gaps[0].human_review.comparison_id is None


def test_production_has_no_direct_machine_service_or_output_dependencies():
    root = Path(__file__).parents[1] / "app"
    production = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "schemas/review_report.py",
            "services/review/review_report_service.py",
        )
    )
    forbidden = (
        "ComplianceComparisonService",
        "ReviewFindingService",
        "FindingReviewService",
        "CandidateDiscoveryService",
        "StandardRoutingService",
        "RequirementDecompositionService",
        "PlanFactExtractionService",
        "WholePlanReviewService",
        "FindingsWorkspaceService",
        "reportlab",
        "weasyprint",
        "python-docx",
        "wkhtmltopdf",
        "FastAPI",
        "Redis",
    )
    assert all(token not in production for token in forbidden)


@pytest.mark.parametrize(
    "token",
    [
        "GB55023",
        "4.4.15",
        "4.4.16",
        "HIGH",
        "MEDIUM",
        "LOW",
        "PLAN_COMPLIANT",
        "PLAN_NON_COMPLIANT",
    ],
)
def test_no_competition_severity_or_verdict_authority_in_production(token):
    root = Path(__file__).parents[1] / "app"
    production = (root / "schemas/review_report.py").read_text(encoding="utf-8") + (
        root / "services/review/review_report_service.py"
    ).read_text(encoding="utf-8")
    assert token not in production
