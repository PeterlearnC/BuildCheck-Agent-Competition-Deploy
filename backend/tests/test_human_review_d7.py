"""D.7-P1 stateless human-review authority qualification tests."""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.finding_review import (
    MAX_REVIEWER_ID_LENGTH,
    MAX_REVIEWER_NOTE_LENGTH,
    FindingReviewDisposition,
    ReportInclusionStatus,
    ReviewerIdentityAssurance,
)
from app.schemas.findings_workspace import WorkspaceItemKind, workspace_item_authority_id
from app.schemas.human_review import (
    FindingWorkspaceReviewCommand,
    HumanReviewCompleteness,
    HumanReviewCounts,
    HumanReviewResult,
    InternalErrorHumanAction,
    InternalErrorWorkspaceReviewCommand,
    ReviewGapHumanAction,
    ReviewGapWorkspaceReviewCommand,
    WorkspaceHumanAction,
    WorkspaceItemReviewRecord,
    deterministic_review_record_id,
    deterministic_review_set_id,
    deterministic_reviewer_note_sha256,
    deterministic_workspace_item_authority_sha256,
)
from app.schemas.standard_route import StandardRouteStatus
from app.schemas.whole_plan_review import WholePlanStage
from app.services.review.human_review_service import (
    HumanReviewAuthorityInvariantError,
    HumanReviewCommandError,
    HumanReviewConflictError,
    HumanReviewService,
    HumanReviewStaleWorkspaceError,
    HumanReviewTargetError,
)
from tests.test_findings_workspace_d6 import _workspace


FIXED_TIME = datetime(2026, 9, 3, 3, 30, 1, 123456, tzinfo=timezone.utc)


class _WorkspaceService:
    def __init__(self, workspace):
        self.workspace = workspace
        self.calls = []

    def build_workspace(self, document_id):
        self.calls.append(document_id)
        return self.workspace


def _human_env(tmp_path, *, kind=WorkspaceItemKind.FINDING, decision=None):
    kwargs = {}
    if kind == WorkspaceItemKind.REVIEW_GAP:
        kwargs["route_statuses"] = [StandardRouteStatus.NO_STANDARD_SCOPE]
    elif kind == WorkspaceItemKind.INTERNAL_ERROR:
        kwargs["routing_failure"] = WholePlanStage.ROUTING
    elif decision is not None:
        kwargs["decisions"] = [decision]
    source = _workspace(tmp_path, **kwargs)
    authority = _WorkspaceService(source.workspace)
    service = HumanReviewService(
        findings_workspace_service=authority, clock=lambda: FIXED_TIME
    )
    return source, authority, service


def _command(workspace, *, action=None, reviewer_id="reviewer-one", note="Reviewed.", inclusion=ReportInclusionStatus.UNDECIDED):
    item = workspace.items[0]
    common = dict(
        expected_workspace_id=workspace.workspace_id,
        target_item_id=workspace_item_authority_id(item),
        reviewer_id=reviewer_id,
        reviewer_note=note,
        report_inclusion=inclusion,
    )
    if item.item_kind == WorkspaceItemKind.FINDING:
        return FindingWorkspaceReviewCommand(
            target_item_kind=item.item_kind,
            disposition=action or FindingReviewDisposition.ACCEPTED,
            **common,
        )
    if item.item_kind == WorkspaceItemKind.REVIEW_GAP:
        return ReviewGapWorkspaceReviewCommand(
            target_item_kind=item.item_kind,
            action=action or ReviewGapHumanAction.ACKNOWLEDGED,
            **common,
        )
    return InternalErrorWorkspaceReviewCommand(
        target_item_kind=item.item_kind,
        action=action or InternalErrorHumanAction.ACKNOWLEDGED,
        **common,
    )


@pytest.mark.parametrize(
    "model",
    [
        FindingWorkspaceReviewCommand,
        ReviewGapWorkspaceReviewCommand,
        InternalErrorWorkspaceReviewCommand,
        WorkspaceItemReviewRecord,
        HumanReviewCounts,
        HumanReviewResult,
    ],
)
def test_all_public_models_are_frozen_extra_forbid(model):
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"


def test_public_models_have_no_any_authority_or_mutable_defaults():
    import app.schemas.human_review as schema

    models = [
        value
        for value in vars(schema).values()
        if inspect.isclass(value)
        and issubclass(value, BaseModel)
        and value.__module__ == schema.__name__
    ]
    for model in models:
        for field in model.model_fields.values():
            assert field.annotation is not __import__("typing").Any
            assert not isinstance(field.default, (list, dict, set))


@pytest.mark.parametrize(
    "forbidden",
    [
        "final_decision",
        "final_compliance",
        "overall_verdict",
        "whole_plan_compliant",
        "severity",
        "risk_score",
        "report_id",
        "export_path",
    ],
)
def test_public_models_have_no_verdict_risk_or_report_fields(forbidden):
    names = set()
    for model in (WorkspaceItemReviewRecord, HumanReviewResult, HumanReviewCounts):
        names.update(model.model_fields)
    assert forbidden not in names


def test_public_seam_accepts_only_document_and_typed_commands():
    signature = inspect.signature(HumanReviewService.review_document)
    assert list(signature.parameters) == ["self", "document_id", "commands"]


@pytest.mark.parametrize(
    "forbidden",
    ["workspace", "finding", "machine_decision", "source_text", "source_offsets"],
)
def test_public_seam_rejects_caller_machine_authority(tmp_path, forbidden):
    source, _, service = _human_env(tmp_path)
    with pytest.raises(TypeError):
        service.review_document(source.env.document_id, (), **{forbidden: object()})


def test_commands_must_be_typed_tuple(tmp_path):
    source, _, service = _human_env(tmp_path)
    with pytest.raises(HumanReviewCommandError):
        service.review_document(source.env.document_id, [])
    with pytest.raises(HumanReviewCommandError):
        service.review_document(source.env.document_id, ({"target": "fake"},))


def test_d6_called_exactly_once(tmp_path):
    source, authority, service = _human_env(tmp_path)
    service.review_document(source.env.document_id, ())
    assert authority.calls == [source.env.document_id]


def test_invalid_d6_type_and_document_mismatch_abort(tmp_path):
    source, authority, service = _human_env(tmp_path)
    authority.workspace = object()
    with pytest.raises(HumanReviewAuthorityInvariantError):
        service.review_document(source.env.document_id, ())
    authority.workspace = source.workspace.model_copy(update={"document_id": "other"})
    with pytest.raises(HumanReviewAuthorityInvariantError):
        service.review_document(source.env.document_id, ())


def test_stale_workspace_rejected(tmp_path):
    source, _, service = _human_env(tmp_path)
    command = _command(source.workspace).model_copy(
        update={"expected_workspace_id": "findingsworkspace_" + "0" * 64}
    )
    with pytest.raises(HumanReviewStaleWorkspaceError):
        service.review_document(source.env.document_id, (command,))


def test_unknown_target_and_wrong_kind_rejected(tmp_path):
    source, _, service = _human_env(tmp_path)
    command = _command(source.workspace).model_copy(update={"target_item_id": "unknown"})
    with pytest.raises(HumanReviewTargetError):
        service.review_document(source.env.document_id, (command,))
    wrong = ReviewGapWorkspaceReviewCommand(
        expected_workspace_id=source.workspace.workspace_id,
        target_item_id=workspace_item_authority_id(source.workspace.items[0]),
        target_item_kind=WorkspaceItemKind.REVIEW_GAP,
        reviewer_id="reviewer-one",
        action=ReviewGapHumanAction.ACKNOWLEDGED,
    )
    with pytest.raises(HumanReviewTargetError):
        service.review_document(source.env.document_id, (wrong,))


@pytest.mark.parametrize("disposition", list(FindingReviewDisposition))
@pytest.mark.parametrize("inclusion", list(ReportInclusionStatus))
def test_finding_review_reuses_c4_vocab_and_preserves_machine(tmp_path, disposition, inclusion):
    source, _, service = _human_env(
        tmp_path, decision=ComparisonDecision.NON_COMPLIANT
    )
    result = service.review_document(
        source.env.document_id,
        (_command(source.workspace, action=disposition, inclusion=inclusion),),
    )
    record = result.records[0]
    item = source.workspace.items[0]
    assert record.human_action.value == disposition.value
    assert record.report_inclusion == inclusion
    assert record.machine_decision == ComparisonDecision.NON_COMPLIANT
    assert record.finding_id == item.finding_id
    assert record.comparison_id == item.comparison_id
    assert result.workspace_authority == source.workspace


@pytest.mark.parametrize("decision", list(ComparisonDecision))
def test_every_machine_decision_remains_independent_of_human_rejection(tmp_path, decision):
    source, _, service = _human_env(tmp_path, decision=decision)
    result = service.review_document(
        source.env.document_id,
        (_command(source.workspace, action=FindingReviewDisposition.REJECTED),),
    )
    assert result.records[0].machine_decision == decision
    assert result.records[0].human_action == WorkspaceHumanAction.REJECTED


@pytest.mark.parametrize("action", list(ReviewGapHumanAction))
def test_gap_actions_preserve_gap_without_compliance(tmp_path, action):
    source, _, service = _human_env(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)
    result = service.review_document(
        source.env.document_id, (_command(source.workspace, action=action),)
    )
    record = result.records[0]
    assert record.target_item_kind == WorkspaceItemKind.REVIEW_GAP
    assert record.human_action.value == action.value
    assert record.machine_decision is None
    assert record.finding_id is None
    assert result.workspace_authority.items[0].item_kind == WorkspaceItemKind.REVIEW_GAP


@pytest.mark.parametrize("action", list(InternalErrorHumanAction))
def test_error_actions_preserve_error_without_compliance_or_risk(tmp_path, action):
    source, _, service = _human_env(tmp_path, kind=WorkspaceItemKind.INTERNAL_ERROR)
    result = service.review_document(
        source.env.document_id, (_command(source.workspace, action=action),)
    )
    record = result.records[0]
    assert record.target_item_kind == WorkspaceItemKind.INTERNAL_ERROR
    assert record.human_action.value == action.value
    assert record.machine_decision is None
    assert result.workspace_authority.items[0].item_kind == WorkspaceItemKind.INTERNAL_ERROR
    assert "risk" not in record.model_dump(mode="json")


def test_cross_kind_actions_rejected_by_record_contract(tmp_path):
    source, _, service = _human_env(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)
    record = service.review_document(
        source.env.document_id, (_command(source.workspace),)
    ).records[0]
    payload = record.model_dump()
    payload["human_action"] = WorkspaceHumanAction.ACCEPTED
    payload["review_record_id"] = deterministic_review_record_id(
        document_id=record.document_id,
        document_sha256=record.document_sha256,
        workspace_id=record.workspace_id,
        target_item_id=record.target_item_id,
        target_item_authority_sha256=record.target_item_authority_sha256,
        target_item_kind=record.target_item_kind,
        human_action=WorkspaceHumanAction.ACCEPTED,
        report_inclusion=record.report_inclusion,
        reviewer_id=record.reviewer_id,
        reviewer_note_sha256=record.reviewer_note_sha256,
    )
    with pytest.raises(ValidationError):
        WorkspaceItemReviewRecord.model_validate(payload)


def test_note_hash_and_unverified_identity_assurance(tmp_path):
    source, _, service = _human_env(tmp_path)
    note = "需要后续人工确认。"
    record = service.review_document(
        source.env.document_id, (_command(source.workspace, note=note),)
    ).records[0]
    assert record.reviewer_note == note
    assert record.reviewer_note_sha256 == hashlib.sha256(note.encode()).hexdigest()
    assert record.reviewer_identity_assurance == ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED


@pytest.mark.parametrize("reviewer", ["", " ", " leading", "trailing ", "x" * (MAX_REVIEWER_ID_LENGTH + 1)])
def test_reviewer_bounds_match_c4(tmp_path, reviewer):
    source, _, _ = _human_env(tmp_path)
    with pytest.raises(ValidationError):
        _command(source.workspace, reviewer_id=reviewer)


def test_note_bound_matches_c4(tmp_path):
    source, _, _ = _human_env(tmp_path)
    with pytest.raises(ValidationError):
        _command(source.workspace, note="x" * (MAX_REVIEWER_NOTE_LENGTH + 1))


def test_created_at_is_utc_audit_metadata_excluded_from_identity(tmp_path):
    source, authority, first_service = _human_env(tmp_path)
    command = _command(source.workspace)
    first = first_service.review_document(source.env.document_id, (command,)).records[0]
    later_service = HumanReviewService(
        findings_workspace_service=authority,
        clock=lambda: FIXED_TIME + timedelta(days=1),
    )
    second = later_service.review_document(source.env.document_id, (command,)).records[0]
    assert first.review_record_id == second.review_record_id
    assert first.created_at != second.created_at
    assert first.created_at.utcoffset() == timedelta(0)


def test_naive_server_clock_rejected(tmp_path):
    source, authority, _ = _human_env(tmp_path)
    service = HumanReviewService(
        findings_workspace_service=authority,
        clock=lambda: FIXED_TIME.replace(tzinfo=None),
    )
    with pytest.raises(HumanReviewCommandError, match="timezone-aware"):
        service.review_document(source.env.document_id, (_command(source.workspace),))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_record_id", "humanreview_" + "0" * 64),
        ("reviewer_note_sha256", "0" * 64),
        ("target_item_authority_sha256", "0" * 64),
    ],
)
def test_record_semantic_tampering_rejected(tmp_path, field, value):
    source, _, service = _human_env(tmp_path)
    result = service.review_document(source.env.document_id, (_command(source.workspace),))
    payload = result.records[0].model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        WorkspaceItemReviewRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("disposition", FindingReviewDisposition.REJECTED),
        ("report_inclusion", ReportInclusionStatus.EXCLUDE),
        ("reviewer_note", "Different note"),
        ("reviewer_id", "reviewer-two"),
    ],
)
def test_semantic_changes_change_record_identity(tmp_path, change, value):
    source, _, service = _human_env(tmp_path)
    first_command = _command(source.workspace)
    first = service.review_document(source.env.document_id, (first_command,)).records[0]
    kwargs = {}
    if change == "disposition": kwargs["action"] = value
    elif change == "report_inclusion": kwargs["inclusion"] = value
    elif change == "reviewer_note": kwargs["note"] = value
    else: kwargs["reviewer_id"] = value
    second = service.review_document(
        source.env.document_id, (_command(source.workspace, **kwargs),)
    ).records[0]
    assert first.review_record_id != second.review_record_id


def test_identical_duplicate_is_idempotent(tmp_path):
    source, _, service = _human_env(tmp_path)
    command = _command(source.workspace)
    result = service.review_document(source.env.document_id, (command, command))
    assert len(result.records) == 1
    assert result.counts.reviewed_items == 1


@pytest.mark.parametrize(
    "other",
    [
        {"action": FindingReviewDisposition.REJECTED},
        {"inclusion": ReportInclusionStatus.EXCLUDE},
        {"note": "conflicting"},
        {"reviewer_id": "reviewer-two"},
    ],
)
def test_conflicting_commands_are_rejected(tmp_path, other):
    source, _, service = _human_env(tmp_path)
    with pytest.raises(HumanReviewConflictError):
        service.review_document(
            source.env.document_id,
            (_command(source.workspace), _command(source.workspace, **other)),
        )


def test_command_order_does_not_change_record_order_or_set_identity(tmp_path):
    source = _workspace(
        tmp_path,
        candidate_texts=("first gap 1mm.", "second gap 2mm."),
        route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE] * 2,
    )
    authority = _WorkspaceService(source.workspace)
    service = HumanReviewService(
        findings_workspace_service=authority, clock=lambda: FIXED_TIME
    )
    commands = tuple(
        ReviewGapWorkspaceReviewCommand(
            expected_workspace_id=source.workspace.workspace_id,
            target_item_id=workspace_item_authority_id(item),
            target_item_kind=WorkspaceItemKind.REVIEW_GAP,
            reviewer_id="reviewer-one",
            action=ReviewGapHumanAction.ACKNOWLEDGED,
        )
        for item in source.workspace.items
    )
    first = service.review_document(source.env.document_id, commands)
    second = service.review_document(source.env.document_id, tuple(reversed(commands)))
    assert [r.review_record_id for r in first.records] == [r.review_record_id for r in second.records]
    assert first.review_set_id == second.review_set_id


def test_unreviewed_partial_and_reviewed_completeness(tmp_path):
    source = _workspace(
        tmp_path,
        candidate_texts=("first gap 1mm.", "second gap 2mm."),
        route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE] * 2,
    )
    authority = _WorkspaceService(source.workspace)
    service = HumanReviewService(findings_workspace_service=authority, clock=lambda: FIXED_TIME)
    commands = tuple(
        ReviewGapWorkspaceReviewCommand(
            expected_workspace_id=source.workspace.workspace_id,
            target_item_id=workspace_item_authority_id(item),
            target_item_kind=WorkspaceItemKind.REVIEW_GAP,
            reviewer_id="reviewer-one",
            action=ReviewGapHumanAction.ACKNOWLEDGED,
        ) for item in source.workspace.items
    )
    none = service.review_document(source.env.document_id, ())
    partial = service.review_document(source.env.document_id, commands[:1])
    full = service.review_document(source.env.document_id, commands)
    assert none.completeness == HumanReviewCompleteness.UNREVIEWED
    assert partial.completeness == HumanReviewCompleteness.PARTIALLY_REVIEWED
    assert full.completeness == HumanReviewCompleteness.REVIEWED
    assert none.unreviewed_item_ids == tuple(workspace_item_authority_id(i) for i in source.workspace.items)
    assert partial.unreviewed_item_ids == (workspace_item_authority_id(source.workspace.items[1]),)
    assert full.unreviewed_item_ids == ()


def test_zero_item_workspace_is_workflow_reviewed_not_compliant(tmp_path):
    source = _workspace(tmp_path, candidate_texts=())
    authority = _WorkspaceService(source.workspace)
    result = HumanReviewService(findings_workspace_service=authority).review_document(
        source.env.document_id, ()
    )
    assert result.completeness == HumanReviewCompleteness.REVIEWED
    assert result.counts.workspace_items == 0
    assert set(HumanReviewResult.model_fields).isdisjoint(
        {"plan_compliant", "overall_compliance", "final_compliance", "verdict"}
    )


def test_result_identity_counts_and_workspace_authority_revalidate(tmp_path):
    source, _, service = _human_env(tmp_path)
    result = service.review_document(source.env.document_id, (_command(source.workspace),))
    assert result.review_set_id == deterministic_review_set_id(
        workspace_id=result.workspace_id, records=result.records
    )
    assert result.counts.workspace_items == 1
    assert result.counts.reviewed_items == 1
    assert result.counts.unreviewed_items == 0
    assert HumanReviewResult.model_validate(result.model_dump()) == result
    with pytest.raises(ValidationError):
        result.workspace_id = "findingsworkspace_" + "0" * 64


def test_exclude_does_not_erase_machine_item(tmp_path):
    source, _, service = _human_env(tmp_path)
    result = service.review_document(
        source.env.document_id,
        (_command(source.workspace, inclusion=ReportInclusionStatus.EXCLUDE),),
    )
    assert len(result.workspace_authority.items) == 1
    assert result.records[0].report_inclusion == ReportInclusionStatus.EXCLUDE


def test_same_text_distinct_items_remain_distinct_targets(tmp_path):
    text = "same gap 1mm."
    source = _workspace(
        tmp_path,
        candidate_texts=(text, text),
        route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE] * 2,
    )
    ids = tuple(workspace_item_authority_id(item) for item in source.workspace.items)
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_target_authority_hash_is_exact_and_deterministic(tmp_path):
    source, _, service = _human_env(tmp_path)
    first = service.review_document(source.env.document_id, (_command(source.workspace),))
    second = service.review_document(source.env.document_id, (_command(source.workspace),))
    expected = deterministic_workspace_item_authority_sha256(source.workspace.items[0])
    assert first.records[0].target_item_authority_sha256 == expected
    assert second.records[0].review_record_id == first.records[0].review_record_id


def test_no_c2_c3_c4_reconstruction_or_report_persistence_boundaries():
    from pathlib import Path

    production = Path("backend/app/services/review/human_review_service.py").read_text(encoding="utf-8")
    schema = Path("backend/app/schemas/human_review.py").read_text(encoding="utf-8")
    for forbidden in (
        "ComplianceComparisonService",
        "ReviewFindingService",
        "FindingReviewService",
        "ComplianceComparisonRequest",
        "ReportModel",
        "report_id",
        "sqlalchemy",
        "redis",
        "celery",
        "FastAPI",
        "APIRouter",
        "httpx",
        "requests",
        "openai",
        "OCR",
    ):
        assert forbidden not in production
        assert forbidden not in schema


@pytest.mark.parametrize("decision", [ComparisonDecision.COMPLIANT, ComparisonDecision.NON_COMPLIANT])
def test_cases_a_and_b_preserve_machine_axis(tmp_path, decision):
    source, _, service = _human_env(tmp_path, decision=decision)
    record = service.review_document(
        source.env.document_id,
        (_command(source.workspace, action=FindingReviewDisposition.REJECTED),),
    ).records[0]
    assert record.machine_decision == decision
    assert record.human_action == WorkspaceHumanAction.REJECTED


def test_case_c_remains_review_gap_without_machine_decision(tmp_path):
    source, _, service = _human_env(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)
    record = service.review_document(
        source.env.document_id,
        (_command(source.workspace, action=ReviewGapHumanAction.NEEDS_FOLLOW_UP),),
    ).records[0]
    assert record.target_item_kind == WorkspaceItemKind.REVIEW_GAP
    assert record.machine_decision is None
    assert record.finding_id is None


@pytest.mark.parametrize(
    ("command_type", "payload"),
    [
        (
            ReviewGapWorkspaceReviewCommand,
            {"action": FindingReviewDisposition.ACCEPTED},
        ),
        (
            InternalErrorWorkspaceReviewCommand,
            {"action": FindingReviewDisposition.NEEDS_INFORMATION},
        ),
    ],
)
def test_item_kind_action_vocabularies_do_not_leak(tmp_path, command_type, payload):
    kind = (
        WorkspaceItemKind.REVIEW_GAP
        if command_type is ReviewGapWorkspaceReviewCommand
        else WorkspaceItemKind.INTERNAL_ERROR
    )
    source, _, _ = _human_env(tmp_path, kind=kind)
    item = source.workspace.items[0]
    with pytest.raises(ValidationError):
        command_type(
            expected_workspace_id=source.workspace.workspace_id,
            target_item_id=workspace_item_authority_id(item),
            target_item_kind=kind,
            reviewer_id="reviewer-one",
            **payload,
        )


@pytest.mark.parametrize(
    "forbidden",
    ["workspace_item", "finding", "comparison", "plan_fact", "source_text"],
)
def test_command_schema_rejects_caller_authored_machine_fields(tmp_path, forbidden):
    source, _, _ = _human_env(tmp_path)
    payload = _command(source.workspace).model_dump()
    payload[forbidden] = "caller-authored"
    with pytest.raises(ValidationError):
        FindingWorkspaceReviewCommand.model_validate(payload)


def test_production_contains_no_competition_specific_business_authority():
    from pathlib import Path

    text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "backend/app/schemas/human_review.py",
            "backend/app/services/review/human_review_service.py",
        )
    )
    for forbidden in (
        "4.4.15",
        "4.4.16",
        "GB55023",
        "150mm",
        "200mm",
        "2.5mm",
        "candidate_697a",
        "candidate_ea698",
    ):
        assert forbidden not in text
