"""Read-only, source-grounded projection contracts for D.6 workspaces."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_comparison import ComparisonDecision, ComparisonReasonCode
from app.schemas.requirement_decomposition import (
    RequirementDecompositionStatus,
    UnresolvedRequirementReason,
)
from app.schemas.whole_plan_review import (
    WholePlanCandidateTerminalState,
    WholePlanCoverageCounts,
    WholePlanInternalError,
    WholePlanProcessingState,
    WholePlanRequirementTerminalState,
)


FINDINGS_WORKSPACE_IDENTITY_VERSION = "v0.1-d.6-findings-workspace"
FINDINGS_WORKSPACE_PROJECTION_VERSION = "v0.1-d.6-read-only-projection"
REVIEW_GAP_IDENTITY_VERSION = "v0.1-d.6-review-gap"
INTERNAL_ERROR_ITEM_IDENTITY_VERSION = "v0.1-d.6-internal-error-item"


class WorkspaceItemKind(str, Enum):
    FINDING = "FINDING"
    REVIEW_GAP = "REVIEW_GAP"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ReviewGapSource(str, Enum):
    CANDIDATE_TERMINAL = "CANDIDATE_TERMINAL"
    UNRESOLVED_SPAN = "UNRESOLVED_SPAN"
    REQUIREMENT_TERMINAL = "REQUIREMENT_TERMINAL"


class PlanSourceLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    physical_page: int = Field(gt=0)
    page_char_start: int = Field(ge=0)
    page_char_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_unit_id: str | None = None
    plan_fact_id: str | None = None

    @model_validator(mode="after")
    def validate_exact_source(self):
        if self.page_char_end != self.page_char_start + len(self.source_text):
            raise ValueError("Plan source locator span does not match exact text.")
        expected = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected:
            raise ValueError("Plan source locator hash does not match exact text.")
        if (self.review_unit_id is None) != (self.plan_fact_id is None):
            raise ValueError("ReviewUnit and PlanFact identities must appear together.")
        return self


class StandardSourceLocator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    standard_id: str = Field(min_length=1)
    standard_code: str = Field(min_length=1)
    canonical_standard_code: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    article_id: str = Field(min_length=1)
    article_number: str = Field(min_length=1)
    source_page_start: int = Field(gt=0)
    source_page_end: int = Field(gt=0)
    evidence_id: str = Field(min_length=1)
    requirement_id: str | None = None
    unresolved_span_id: str | None = None
    article_char_start: int = Field(ge=0)
    article_char_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    navigation_level: Literal["TEXT_PAGE_INTERVAL"] = "TEXT_PAGE_INTERVAL"

    @model_validator(mode="after")
    def validate_exact_source(self):
        if self.source_page_end < self.source_page_start:
            raise ValueError("Standard source page interval is reversed.")
        if self.article_char_end != self.article_char_start + len(self.source_text):
            raise ValueError("Standard source locator span does not match exact text.")
        expected = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected:
            raise ValueError("Standard source locator hash does not match exact text.")
        if (self.requirement_id is None) == (self.unresolved_span_id is None):
            raise ValueError(
                "Standard locator requires exactly one requirement or unresolved span."
            )
        return self


class FindingWorkspaceItem(BaseModel):
    """Projection of retained D.5/C.3 identity, never a replacement Finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_kind: Literal[WorkspaceItemKind.FINDING] = WorkspaceItemKind.FINDING
    finding_id: str = Field(pattern=r"^finding_[0-9a-f]{64}$")
    comparison_id: str = Field(min_length=1)
    decision: ComparisonDecision
    reason_code: ComparisonReasonCode
    candidate_trace_id: str = Field(pattern=r"^candidatetrace_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    requirement_trace_id: str = Field(pattern=r"^requirementtrace_[0-9a-f]{64}$")
    requirement_id: str = Field(min_length=1)
    plan_fact_id: str = Field(min_length=1)
    route_id: str = Field(pattern=r"^route_[0-9a-f]{64}$")
    plan_source: PlanSourceLocator
    standard_source: StandardSourceLocator

    @model_validator(mode="after")
    def validate_cross_source_identity(self):
        if self.candidate_id != self.plan_source.candidate_id:
            raise ValueError("Finding plan locator belongs to another candidate.")
        if self.requirement_id != self.standard_source.requirement_id:
            raise ValueError("Finding standard locator belongs to another requirement.")
        if self.plan_fact_id != self.plan_source.plan_fact_id:
            raise ValueError("Finding plan locator belongs to another PlanFact.")
        return self


_CANDIDATE_GAP_TERMINALS = {
    WholePlanCandidateTerminalState.NO_STANDARD_SCOPE,
    WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE,
    WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED,
    WholePlanCandidateTerminalState.REQUIREMENT_SOURCE_NOT_QUALIFIED,
}
_REQUIREMENT_GAP_TERMINALS = {
    WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND,
    WholePlanRequirementTerminalState.PLAN_FACT_AMBIGUOUS,
    WholePlanRequirementTerminalState.PLAN_FACT_UNSUPPORTED,
    WholePlanRequirementTerminalState.PLAN_FACT_SOURCE_NOT_QUALIFIED,
}


def _canonical_identity(prefix: str, payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(raw).hexdigest()


def deterministic_review_gap_id(
    *,
    whole_plan_review_id: str,
    document_id: str,
    document_sha256: str,
    candidate_trace_id: str,
    requirement_trace_id: str | None,
    unresolved_span_id: str | None,
    gap_source: ReviewGapSource,
    candidate_terminal_state: WholePlanCandidateTerminalState | None,
    requirement_terminal_state: WholePlanRequirementTerminalState | None,
    decomposition_status: RequirementDecompositionStatus | None,
    unresolved_reason: UnresolvedRequirementReason | None,
    identity_version: str = REVIEW_GAP_IDENTITY_VERSION,
    projection_version: str = FINDINGS_WORKSPACE_PROJECTION_VERSION,
) -> str:
    return _canonical_identity(
        "reviewgap_",
        {
            "identity_version": identity_version,
            "projection_version": projection_version,
            "whole_plan_review_id": whole_plan_review_id,
            "document_id": document_id,
            "document_sha256": document_sha256,
            "candidate_trace_id": candidate_trace_id,
            "requirement_trace_id": requirement_trace_id,
            "unresolved_span_id": unresolved_span_id,
            "gap_source": gap_source.value,
            "candidate_terminal_state": (
                candidate_terminal_state.value if candidate_terminal_state else None
            ),
            "requirement_terminal_state": (
                requirement_terminal_state.value if requirement_terminal_state else None
            ),
            "decomposition_status": (
                decomposition_status.value if decomposition_status else None
            ),
            "unresolved_reason": (
                unresolved_reason.value if unresolved_reason else None
            ),
        },
    )


class ReviewGapWorkspaceItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_kind: Literal[WorkspaceItemKind.REVIEW_GAP] = WorkspaceItemKind.REVIEW_GAP
    identity_version: Literal[REVIEW_GAP_IDENTITY_VERSION] = REVIEW_GAP_IDENTITY_VERSION
    review_gap_id: str = Field(pattern=r"^reviewgap_[0-9a-f]{64}$")
    whole_plan_review_id: str = Field(pattern=r"^wholeplanreview_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_trace_id: str = Field(pattern=r"^candidatetrace_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    requirement_trace_id: str | None = None
    requirement_id: str | None = None
    unresolved_span_id: str | None = None
    gap_source: ReviewGapSource
    candidate_terminal_state: WholePlanCandidateTerminalState | None = None
    requirement_terminal_state: WholePlanRequirementTerminalState | None = None
    decomposition_status: RequirementDecompositionStatus | None = None
    unresolved_reason: UnresolvedRequirementReason | None = None
    plan_source: PlanSourceLocator
    standard_source: StandardSourceLocator | None = None
    projection_version: Literal[FINDINGS_WORKSPACE_PROJECTION_VERSION] = (
        FINDINGS_WORKSPACE_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_gap_contract_and_identity(self):
        if self.candidate_id != self.plan_source.candidate_id:
            raise ValueError("Gap plan locator belongs to another candidate.")
        if self.gap_source == ReviewGapSource.CANDIDATE_TERMINAL:
            if (
                self.candidate_terminal_state not in _CANDIDATE_GAP_TERMINALS
                or any(
                    value is not None
                    for value in (
                        self.requirement_trace_id,
                        self.requirement_id,
                        self.unresolved_span_id,
                        self.requirement_terminal_state,
                        self.decomposition_status,
                        self.unresolved_reason,
                        self.standard_source,
                    )
                )
            ):
                raise ValueError("Candidate-level review gap has contradictory authority.")
        elif self.gap_source == ReviewGapSource.UNRESOLVED_SPAN:
            if (
                self.unresolved_span_id is None
                or self.decomposition_status
                not in {
                    RequirementDecompositionStatus.UNRESOLVED,
                    RequirementDecompositionStatus.PARTIAL,
                }
                or self.unresolved_reason is None
                or self.standard_source is None
                or self.standard_source.unresolved_span_id != self.unresolved_span_id
                or any(
                    value is not None
                    for value in (
                        self.requirement_trace_id,
                        self.requirement_id,
                        self.candidate_terminal_state,
                        self.requirement_terminal_state,
                    )
                )
            ):
                raise ValueError("Unresolved-span review gap has contradictory authority.")
        elif (
            self.requirement_terminal_state not in _REQUIREMENT_GAP_TERMINALS
            or self.requirement_trace_id is None
            or self.requirement_id is None
            or self.standard_source is None
            or self.standard_source.requirement_id != self.requirement_id
            or any(
                value is not None
                for value in (
                    self.candidate_terminal_state,
                    self.unresolved_span_id,
                    self.decomposition_status,
                    self.unresolved_reason,
                )
            )
        ):
            raise ValueError("Requirement-level review gap has contradictory authority.")
        expected = deterministic_review_gap_id(
            whole_plan_review_id=self.whole_plan_review_id,
            document_id=self.document_id,
            document_sha256=self.document_sha256,
            candidate_trace_id=self.candidate_trace_id,
            requirement_trace_id=self.requirement_trace_id,
            unresolved_span_id=self.unresolved_span_id,
            gap_source=self.gap_source,
            candidate_terminal_state=self.candidate_terminal_state,
            requirement_terminal_state=self.requirement_terminal_state,
            decomposition_status=self.decomposition_status,
            unresolved_reason=self.unresolved_reason,
            identity_version=self.identity_version,
            projection_version=self.projection_version,
        )
        if self.review_gap_id != expected:
            raise ValueError("review_gap_id does not match frozen gap authority.")
        return self


def deterministic_internal_error_item_id(
    *,
    whole_plan_review_id: str,
    document_id: str,
    document_sha256: str,
    candidate_trace_id: str,
    requirement_trace_id: str | None,
    error: WholePlanInternalError,
    identity_version: str = INTERNAL_ERROR_ITEM_IDENTITY_VERSION,
    projection_version: str = FINDINGS_WORKSPACE_PROJECTION_VERSION,
) -> str:
    return _canonical_identity(
        "workspaceerror_",
        {
            "identity_version": identity_version,
            "projection_version": projection_version,
            "whole_plan_review_id": whole_plan_review_id,
            "document_id": document_id,
            "document_sha256": document_sha256,
            "candidate_trace_id": candidate_trace_id,
            "requirement_trace_id": requirement_trace_id,
            "stage": error.stage.value,
            "code": error.code.value,
        },
    )


class InternalErrorWorkspaceItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_kind: Literal[WorkspaceItemKind.INTERNAL_ERROR] = WorkspaceItemKind.INTERNAL_ERROR
    identity_version: Literal[INTERNAL_ERROR_ITEM_IDENTITY_VERSION] = (
        INTERNAL_ERROR_ITEM_IDENTITY_VERSION
    )
    internal_error_item_id: str = Field(pattern=r"^workspaceerror_[0-9a-f]{64}$")
    whole_plan_review_id: str = Field(pattern=r"^wholeplanreview_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_trace_id: str = Field(pattern=r"^candidatetrace_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    requirement_trace_id: str | None = None
    requirement_id: str | None = None
    error: WholePlanInternalError
    plan_source: PlanSourceLocator
    projection_version: Literal[FINDINGS_WORKSPACE_PROJECTION_VERSION] = (
        FINDINGS_WORKSPACE_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_error_contract_and_identity(self):
        if self.candidate_id != self.plan_source.candidate_id:
            raise ValueError("Error plan locator belongs to another candidate.")
        if (self.requirement_trace_id is None) != (self.requirement_id is None):
            raise ValueError("Requirement error identities must appear together.")
        expected = deterministic_internal_error_item_id(
            whole_plan_review_id=self.whole_plan_review_id,
            document_id=self.document_id,
            document_sha256=self.document_sha256,
            candidate_trace_id=self.candidate_trace_id,
            requirement_trace_id=self.requirement_trace_id,
            error=self.error,
            identity_version=self.identity_version,
            projection_version=self.projection_version,
        )
        if self.internal_error_item_id != expected:
            raise ValueError("Internal-error item ID does not match frozen authority.")
        return self


WorkspaceItem = Annotated[
    FindingWorkspaceItem | ReviewGapWorkspaceItem | InternalErrorWorkspaceItem,
    Field(discriminator="item_kind"),
]


def workspace_item_authority_id(item: WorkspaceItem) -> str:
    if isinstance(item, FindingWorkspaceItem):
        return item.finding_id
    if isinstance(item, ReviewGapWorkspaceItem):
        return item.review_gap_id
    return item.internal_error_item_id


def workspace_item_sort_key(item: WorkspaceItem) -> tuple[object, ...]:
    standard = getattr(item, "standard_source", None)
    kind_rank = {
        WorkspaceItemKind.FINDING: 0,
        WorkspaceItemKind.REVIEW_GAP: 1,
        WorkspaceItemKind.INTERNAL_ERROR: 2,
    }[item.item_kind]
    return (
        item.plan_source.physical_page,
        item.plan_source.page_char_start,
        item.plan_source.page_char_end,
        item.candidate_id,
        standard.source_page_start if standard else 0,
        standard.article_char_start if standard else 0,
        kind_rank,
        workspace_item_authority_id(item),
    )


class FindingsWorkspaceCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: int = Field(ge=0)
    review_gaps: int = Field(ge=0)
    internal_errors: int = Field(ge=0)
    finding_compliant: int = Field(ge=0)
    finding_non_compliant: int = Field(ge=0)
    finding_insufficient_information: int = Field(ge=0)
    routing_gaps: int = Field(ge=0)
    evidence_gaps: int = Field(ge=0)
    requirement_gaps: int = Field(ge=0)
    plan_fact_gaps: int = Field(ge=0)
    partial_unresolved_gaps: int = Field(ge=0)


def derive_workspace_counts(items: tuple[WorkspaceItem, ...]) -> FindingsWorkspaceCounts:
    counts = {name: 0 for name in FindingsWorkspaceCounts.model_fields}
    for item in items:
        if isinstance(item, FindingWorkspaceItem):
            counts["findings"] += 1
            key = {
                ComparisonDecision.COMPLIANT: "finding_compliant",
                ComparisonDecision.NON_COMPLIANT: "finding_non_compliant",
                ComparisonDecision.INSUFFICIENT_INFORMATION: (
                    "finding_insufficient_information"
                ),
            }[item.decision]
            counts[key] += 1
        elif isinstance(item, InternalErrorWorkspaceItem):
            counts["internal_errors"] += 1
        else:
            counts["review_gaps"] += 1
            if item.gap_source == ReviewGapSource.CANDIDATE_TERMINAL:
                if item.candidate_terminal_state in {
                    WholePlanCandidateTerminalState.NO_STANDARD_SCOPE,
                    WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE,
                }:
                    counts["routing_gaps"] += 1
                elif item.candidate_terminal_state == (
                    WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED
                ):
                    counts["evidence_gaps"] += 1
                else:
                    counts["requirement_gaps"] += 1
            elif item.gap_source == ReviewGapSource.UNRESOLVED_SPAN:
                counts["requirement_gaps"] += 1
                if item.decomposition_status == RequirementDecompositionStatus.PARTIAL:
                    counts["partial_unresolved_gaps"] += 1
            else:
                counts["plan_fact_gaps"] += 1
    return FindingsWorkspaceCounts(**counts)


def deterministic_workspace_id(
    *,
    whole_plan_review_id: str,
    items: tuple[WorkspaceItem, ...],
    identity_version: str = FINDINGS_WORKSPACE_IDENTITY_VERSION,
    projection_version: str = FINDINGS_WORKSPACE_PROJECTION_VERSION,
) -> str:
    return _canonical_identity(
        "findingsworkspace_",
        {
            "identity_version": identity_version,
            "projection_version": projection_version,
            "whole_plan_review_id": whole_plan_review_id,
            "ordered_item_authority_ids": [
                workspace_item_authority_id(item) for item in items
            ],
        },
    )


class FindingsWorkspaceResult(BaseModel):
    """Complete authoritative projection; filtering and pagination are external views."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[FINDINGS_WORKSPACE_IDENTITY_VERSION] = (
        FINDINGS_WORKSPACE_IDENTITY_VERSION
    )
    projection_version: Literal[FINDINGS_WORKSPACE_PROJECTION_VERSION] = (
        FINDINGS_WORKSPACE_PROJECTION_VERSION
    )
    workspace_id: str = Field(pattern=r"^findingsworkspace_[0-9a-f]{64}$")
    whole_plan_review_id: str = Field(pattern=r"^wholeplanreview_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_state: WholePlanProcessingState
    items: tuple[WorkspaceItem, ...] = ()
    counts: FindingsWorkspaceCounts
    source_coverage_counts: WholePlanCoverageCounts

    @model_validator(mode="after")
    def validate_complete_projection(self):
        authority_ids = [workspace_item_authority_id(item) for item in self.items]
        if len(authority_ids) != len(set(authority_ids)):
            raise ValueError("Workspace item authority identities must be unique.")
        finding_ids = [
            item.finding_id
            for item in self.items
            if isinstance(item, FindingWorkspaceItem)
        ]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("One C.3 finding ID may map to only one workspace item.")
        if self.items != tuple(sorted(self.items, key=workspace_item_sort_key)):
            raise ValueError("Workspace items must retain canonical source ordering.")
        if any(
            item.plan_source.document_id != self.document_id
            or item.plan_source.document_sha256 != self.document_sha256
            for item in self.items
        ):
            raise ValueError("Workspace item belongs to another document authority.")
        expected_counts = derive_workspace_counts(self.items)
        if self.counts != expected_counts:
            raise ValueError("Workspace counts must derive from the complete item set.")
        if self.counts.findings != self.source_coverage_counts.findings_created:
            raise ValueError("Workspace findings do not reconcile with D.5 authority.")
        if self.counts.internal_errors != self.source_coverage_counts.internal_errors:
            raise ValueError("Workspace errors do not reconcile with D.5 authority.")
        expected_plan_fact_gaps = sum(
            (
                self.source_coverage_counts.plan_fact_not_found,
                self.source_coverage_counts.plan_fact_ambiguous,
                self.source_coverage_counts.plan_fact_unsupported,
                self.source_coverage_counts.plan_fact_source_not_qualified,
            )
        )
        if self.counts.plan_fact_gaps != expected_plan_fact_gaps:
            raise ValueError("Workspace PlanFact gaps do not reconcile with D.5 authority.")
        if self.counts.routing_gaps != (
            self.source_coverage_counts.no_standard_scope
            + self.source_coverage_counts.ambiguous_standard_scope
        ):
            raise ValueError("Workspace routing gaps do not reconcile with D.5 authority.")
        expected_id = deterministic_workspace_id(
            whole_plan_review_id=self.whole_plan_review_id,
            items=self.items,
            identity_version=self.identity_version,
            projection_version=self.projection_version,
        )
        if self.workspace_id != expected_id:
            raise ValueError("workspace_id does not match complete D.5 projection.")
        return self
