"""Stateless human-review metadata over frozen D.6 workspace authority."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.finding_review import (
    MAX_REVIEWER_ID_LENGTH,
    MAX_REVIEWER_NOTE_LENGTH,
    FindingReviewDisposition,
    ReportInclusionStatus,
    ReviewerIdentityAssurance,
)
from app.schemas.findings_workspace import (
    FindingWorkspaceItem,
    FindingsWorkspaceResult,
    WorkspaceItem,
    WorkspaceItemKind,
    workspace_item_authority_id,
)


HUMAN_REVIEW_RECORD_VERSION = "v0.1-d.7-p1-human-review-record"
HUMAN_REVIEW_SET_VERSION = "v0.1-d.7-p1-human-review-set"


class ReviewGapHumanAction(str, Enum):
    """Workflow handling only; none of these actions resolves a review gap."""

    ACKNOWLEDGED = "ACKNOWLEDGED"
    NEEDS_FOLLOW_UP = "NEEDS_FOLLOW_UP"
    DEFERRED = "DEFERRED"


class InternalErrorHumanAction(str, Enum):
    """Operational handling only; these actions carry no compliance meaning."""

    ACKNOWLEDGED = "ACKNOWLEDGED"
    REQUIRES_RERUN = "REQUIRES_RERUN"
    DEFERRED = "DEFERRED"


class WorkspaceHumanAction(str, Enum):
    """Finite serialized action vocabulary for immutable review records."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEEDS_INFORMATION = "NEEDS_INFORMATION"
    DEFERRED = "DEFERRED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    NEEDS_FOLLOW_UP = "NEEDS_FOLLOW_UP"
    REQUIRES_RERUN = "REQUIRES_RERUN"


class HumanReviewCompleteness(str, Enum):
    """Human workflow completion only, never a whole-plan verdict."""

    UNREVIEWED = "UNREVIEWED"
    PARTIALLY_REVIEWED = "PARTIALLY_REVIEWED"
    REVIEWED = "REVIEWED"


class _CommandBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_workspace_id: str = Field(pattern=r"^findingsworkspace_[0-9a-f]{64}$")
    target_item_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1, max_length=MAX_REVIEWER_ID_LENGTH)
    reviewer_note: str = Field(default="", max_length=MAX_REVIEWER_NOTE_LENGTH)
    report_inclusion: ReportInclusionStatus = ReportInclusionStatus.UNDECIDED

    @model_validator(mode="after")
    def validate_reviewer_id(self):
        if not self.reviewer_id.strip() or self.reviewer_id != self.reviewer_id.strip():
            raise ValueError("reviewer_id must be non-blank without edge whitespace.")
        return self


class FindingWorkspaceReviewCommand(_CommandBase):
    target_item_kind: Literal[WorkspaceItemKind.FINDING] = WorkspaceItemKind.FINDING
    disposition: FindingReviewDisposition


class ReviewGapWorkspaceReviewCommand(_CommandBase):
    target_item_kind: Literal[WorkspaceItemKind.REVIEW_GAP] = WorkspaceItemKind.REVIEW_GAP
    action: ReviewGapHumanAction


class InternalErrorWorkspaceReviewCommand(_CommandBase):
    target_item_kind: Literal[WorkspaceItemKind.INTERNAL_ERROR] = (
        WorkspaceItemKind.INTERNAL_ERROR
    )
    action: InternalErrorHumanAction


WorkspaceItemReviewCommand = Annotated[
    FindingWorkspaceReviewCommand
    | ReviewGapWorkspaceReviewCommand
    | InternalErrorWorkspaceReviewCommand,
    Field(discriminator="target_item_kind"),
]


def deterministic_reviewer_note_sha256(note: str) -> str:
    return hashlib.sha256(note.encode("utf-8")).hexdigest()


def deterministic_workspace_item_authority_sha256(item: WorkspaceItem) -> str:
    serialized = json.dumps(
        item.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def deterministic_review_record_id(
    *,
    document_id: str,
    document_sha256: str,
    workspace_id: str,
    target_item_id: str,
    target_item_authority_sha256: str,
    target_item_kind: WorkspaceItemKind,
    human_action: WorkspaceHumanAction,
    report_inclusion: ReportInclusionStatus,
    reviewer_id: str,
    reviewer_note_sha256: str,
    record_version: str = HUMAN_REVIEW_RECORD_VERSION,
) -> str:
    payload = {
        "record_version": record_version,
        "document_id": document_id,
        "document_sha256": document_sha256,
        "workspace_id": workspace_id,
        "target_item_id": target_item_id,
        "target_item_authority_sha256": target_item_authority_sha256,
        "target_item_kind": target_item_kind.value,
        "human_action": human_action.value,
        "report_inclusion": report_inclusion.value,
        "reviewer_id": reviewer_id,
        "reviewer_note_sha256": reviewer_note_sha256,
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "humanreview_" + hashlib.sha256(raw).hexdigest()


_FINDING_ACTIONS = {item.value for item in FindingReviewDisposition}
_GAP_ACTIONS = {item.value for item in ReviewGapHumanAction}
_ERROR_ACTIONS = {item.value for item in InternalErrorHumanAction}


class WorkspaceItemReviewRecord(BaseModel):
    """Human metadata referencing, never replacing, one D.6 workspace item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_record_id: str = Field(pattern=r"^humanreview_[0-9a-f]{64}$")
    record_version: Literal[HUMAN_REVIEW_RECORD_VERSION] = HUMAN_REVIEW_RECORD_VERSION
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_id: str = Field(pattern=r"^findingsworkspace_[0-9a-f]{64}$")
    target_item_id: str = Field(min_length=1)
    target_item_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_item_kind: WorkspaceItemKind
    machine_decision: ComparisonDecision | None = None
    finding_id: str | None = None
    comparison_id: str | None = None
    human_action: WorkspaceHumanAction
    report_inclusion: ReportInclusionStatus
    reviewer_id: str = Field(min_length=1, max_length=MAX_REVIEWER_ID_LENGTH)
    reviewer_identity_assurance: Literal[
        ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
    ] = ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
    reviewer_note: str = Field(max_length=MAX_REVIEWER_NOTE_LENGTH)
    reviewer_note_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime

    @model_validator(mode="after")
    def validate_record(self):
        if not self.reviewer_id.strip() or self.reviewer_id != self.reviewer_id.strip():
            raise ValueError("reviewer_id must be non-blank without edge whitespace.")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware UTC audit metadata.")
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must use UTC.")
        if self.reviewer_note_sha256 != deterministic_reviewer_note_sha256(
            self.reviewer_note
        ):
            raise ValueError("Reviewer note hash does not match reviewer_note.")

        allowed = {
            WorkspaceItemKind.FINDING: _FINDING_ACTIONS,
            WorkspaceItemKind.REVIEW_GAP: _GAP_ACTIONS,
            WorkspaceItemKind.INTERNAL_ERROR: _ERROR_ACTIONS,
        }[self.target_item_kind]
        if self.human_action.value not in allowed:
            raise ValueError("Human action is incompatible with workspace item kind.")

        finding_fields = (
            self.machine_decision,
            self.finding_id,
            self.comparison_id,
        )
        if self.target_item_kind == WorkspaceItemKind.FINDING:
            if any(value is None for value in finding_fields):
                raise ValueError("Finding review must retain machine Finding authority.")
            if self.finding_id != self.target_item_id:
                raise ValueError("Finding review target must be the original finding_id.")
        elif any(value is not None for value in finding_fields):
            raise ValueError("Non-Finding review cannot contain machine Finding authority.")

        expected_id = deterministic_review_record_id(
            document_id=self.document_id,
            document_sha256=self.document_sha256,
            workspace_id=self.workspace_id,
            target_item_id=self.target_item_id,
            target_item_authority_sha256=self.target_item_authority_sha256,
            target_item_kind=self.target_item_kind,
            human_action=self.human_action,
            report_inclusion=self.report_inclusion,
            reviewer_id=self.reviewer_id,
            reviewer_note_sha256=self.reviewer_note_sha256,
            record_version=self.record_version,
        )
        if self.review_record_id != expected_id:
            raise ValueError("Review record identity does not match semantic inputs.")
        return self


class HumanReviewCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_items: int = Field(ge=0)
    reviewed_items: int = Field(ge=0)
    unreviewed_items: int = Field(ge=0)
    findings_reviewed: int = Field(ge=0)
    review_gaps_reviewed: int = Field(ge=0)
    internal_errors_reviewed: int = Field(ge=0)


def derive_human_review_counts(
    workspace: FindingsWorkspaceResult,
    records: tuple[WorkspaceItemReviewRecord, ...],
) -> HumanReviewCounts:
    return HumanReviewCounts(
        workspace_items=len(workspace.items),
        reviewed_items=len(records),
        unreviewed_items=len(workspace.items) - len(records),
        findings_reviewed=sum(
            record.target_item_kind == WorkspaceItemKind.FINDING for record in records
        ),
        review_gaps_reviewed=sum(
            record.target_item_kind == WorkspaceItemKind.REVIEW_GAP for record in records
        ),
        internal_errors_reviewed=sum(
            record.target_item_kind == WorkspaceItemKind.INTERNAL_ERROR
            for record in records
        ),
    )


def derive_review_completeness(
    *, workspace_item_count: int, reviewed_item_count: int
) -> HumanReviewCompleteness:
    if workspace_item_count == 0:
        return HumanReviewCompleteness.REVIEWED
    if reviewed_item_count == 0:
        return HumanReviewCompleteness.UNREVIEWED
    if reviewed_item_count == workspace_item_count:
        return HumanReviewCompleteness.REVIEWED
    return HumanReviewCompleteness.PARTIALLY_REVIEWED


def deterministic_review_set_id(
    *,
    workspace_id: str,
    records: tuple[WorkspaceItemReviewRecord, ...],
    review_set_version: str = HUMAN_REVIEW_SET_VERSION,
) -> str:
    payload = {
        "review_set_version": review_set_version,
        "workspace_id": workspace_id,
        "ordered_review_record_ids": [record.review_record_id for record in records],
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "humanreviewset_" + hashlib.sha256(raw).hexdigest()


class HumanReviewResult(BaseModel):
    """Complete stateless human workflow projection over one exact D.6 workspace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_set_version: Literal[HUMAN_REVIEW_SET_VERSION] = HUMAN_REVIEW_SET_VERSION
    review_set_id: str = Field(pattern=r"^humanreviewset_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_id: str = Field(pattern=r"^findingsworkspace_[0-9a-f]{64}$")
    workspace_authority: FindingsWorkspaceResult
    completeness: HumanReviewCompleteness
    records: tuple[WorkspaceItemReviewRecord, ...] = ()
    unreviewed_item_ids: tuple[str, ...] = ()
    counts: HumanReviewCounts

    @model_validator(mode="after")
    def validate_result(self):
        workspace = self.workspace_authority
        if (
            self.document_id != workspace.document_id
            or self.document_sha256 != workspace.document_sha256
            or self.workspace_id != workspace.workspace_id
        ):
            raise ValueError("Human review result differs from D.6 workspace authority.")

        item_ids = tuple(workspace_item_authority_id(item) for item in workspace.items)
        item_by_id = dict(zip(item_ids, workspace.items, strict=True))
        record_ids = tuple(record.target_item_id for record in self.records)
        record_id_set = set(record_ids)
        if len(record_ids) != len(record_id_set):
            raise ValueError("One effective human review record is allowed per item.")
        if any(target not in item_by_id for target in record_ids):
            raise ValueError("Review record targets an item outside the workspace.")
        ordered_record_ids = tuple(
            item_id for item_id in item_ids if item_id in record_id_set
        )
        if record_ids != ordered_record_ids:
            raise ValueError("Review records must follow D.6 workspace source order.")
        for record in self.records:
            item = item_by_id[record.target_item_id]
            if (
                record.document_id != self.document_id
                or record.document_sha256 != self.document_sha256
                or record.workspace_id != self.workspace_id
                or record.target_item_kind != item.item_kind
                or record.target_item_authority_sha256
                != deterministic_workspace_item_authority_sha256(item)
            ):
                raise ValueError("Review record does not match target item authority.")
            if isinstance(item, FindingWorkspaceItem) and (
                record.machine_decision != item.decision
                or record.finding_id != item.finding_id
                or record.comparison_id != item.comparison_id
            ):
                raise ValueError("Human record altered retained machine Finding authority.")

        expected_unreviewed = tuple(
            item_id for item_id in item_ids if item_id not in record_id_set
        )
        if self.unreviewed_item_ids != expected_unreviewed:
            raise ValueError("Unreviewed items must be the exact workspace complement.")
        if self.counts != derive_human_review_counts(workspace, self.records):
            raise ValueError("Human review counts must derive from records and workspace.")
        if self.completeness != derive_review_completeness(
            workspace_item_count=len(item_ids), reviewed_item_count=len(record_ids)
        ):
            raise ValueError("Review completeness must describe workflow coverage only.")
        if self.review_set_id != deterministic_review_set_id(
            workspace_id=self.workspace_id,
            records=self.records,
            review_set_version=self.review_set_version,
        ):
            raise ValueError("Review-set identity does not match ordered records.")
        return self
