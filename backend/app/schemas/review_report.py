"""Deterministic semantic report projection over frozen human-review authority."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_comparison import ComparisonDecision, ComparisonReasonCode
from app.schemas.finding_review import ReportInclusionStatus
from app.schemas.findings_workspace import (
    FindingsWorkspaceCounts,
    PlanSourceLocator,
    ReviewGapSource,
    StandardSourceLocator,
    WorkspaceItemKind,
)
from app.schemas.human_review import (
    HumanReviewCompleteness,
    HumanReviewCounts,
    WorkspaceItemReviewRecord,
)
from app.schemas.requirement_decomposition import (
    RequirementDecompositionStatus,
    UnresolvedRequirementReason,
)
from app.schemas.whole_plan_review import (
    WholePlanCandidateTerminalState,
    WholePlanCoverageCounts,
    WholePlanInternalErrorCode,
    WholePlanProcessingState,
    WholePlanRequirementTerminalState,
    WholePlanStage,
)


REVIEW_REPORT_PROJECTION_VERSION = "v0.1-d.7-p2-semantic-report"
REPORT_ENTRY_IDENTITY_VERSION = "v0.1-d.7-p2-report-entry"
EXCLUDED_REFERENCE_IDENTITY_VERSION = "v0.1-d.7-p2-excluded-reference"
REPORT_COVERAGE_CODE = "QUALIFIED_PATHS_WITH_GAPS_ERRORS_AND_EXCLUSIONS_RETAINED"
REPORT_COVERAGE_STATEMENT = (
    "This report separately preserves machine Findings, unresolved review gaps, "
    "internal processing errors, human review state, and excluded-item accounting."
)
UNREVIEWED_SEMANTIC_MARKER = "UNREVIEWED"


class ReportInclusionBasis(str, Enum):
    EXPLICIT_INCLUDE = "EXPLICIT_INCLUDE"
    UNDECIDED_DEFAULT_INCLUDE = "UNDECIDED_DEFAULT_INCLUDE"
    UNREVIEWED_DEFAULT_INCLUDE = "UNREVIEWED_DEFAULT_INCLUDE"


class ReportDocumentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    whole_plan_review_id: str = Field(pattern=r"^wholeplanreview_[0-9a-f]{64}$")
    workspace_id: str = Field(pattern=r"^findingsworkspace_[0-9a-f]{64}$")
    review_set_id: str = Field(pattern=r"^humanreviewset_[0-9a-f]{64}$")


class ReportProcessingSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    processing_state: WholePlanProcessingState
    workspace_counts: FindingsWorkspaceCounts
    source_coverage_counts: WholePlanCoverageCounts
    human_review_workflow_completeness: HumanReviewCompleteness
    human_review_counts: HumanReviewCounts


def _canonical_id(prefix: str, payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(raw).hexdigest()


def _review_semantic_id(review: WorkspaceItemReviewRecord | None) -> str:
    return review.review_record_id if review is not None else UNREVIEWED_SEMANTIC_MARKER


def _expected_inclusion_basis(
    review: WorkspaceItemReviewRecord | None,
) -> ReportInclusionBasis:
    if review is None:
        return ReportInclusionBasis.UNREVIEWED_DEFAULT_INCLUDE
    if review.report_inclusion == ReportInclusionStatus.INCLUDE:
        return ReportInclusionBasis.EXPLICIT_INCLUDE
    if review.report_inclusion == ReportInclusionStatus.UNDECIDED:
        return ReportInclusionBasis.UNDECIDED_DEFAULT_INCLUDE
    raise ValueError("An explicitly excluded item cannot be a detailed report entry.")


def deterministic_report_entry_id(
    *,
    source_item_id: str,
    target_item_authority_sha256: str,
    item_kind: WorkspaceItemKind,
    review_record_id: str,
    inclusion_basis: ReportInclusionBasis,
    identity_version: str = REPORT_ENTRY_IDENTITY_VERSION,
    projection_version: str = REVIEW_REPORT_PROJECTION_VERSION,
) -> str:
    return _canonical_id(
        "reportentry_",
        {
            "identity_version": identity_version,
            "projection_version": projection_version,
            "source_item_id": source_item_id,
            "target_item_authority_sha256": target_item_authority_sha256,
            "item_kind": item_kind.value,
            "review_record_id": review_record_id,
            "inclusion_basis": inclusion_basis.value,
        },
    )


class _ReportEntryBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[REPORT_ENTRY_IDENTITY_VERSION] = (
        REPORT_ENTRY_IDENTITY_VERSION
    )
    projection_version: Literal[REVIEW_REPORT_PROJECTION_VERSION] = (
        REVIEW_REPORT_PROJECTION_VERSION
    )
    report_entry_id: str = Field(pattern=r"^reportentry_[0-9a-f]{64}$")
    source_ordinal: int = Field(ge=0)
    source_item_id: str = Field(min_length=1)
    target_item_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inclusion_basis: ReportInclusionBasis
    human_review: WorkspaceItemReviewRecord | None = None

    def validate_projection_identity(self, item_kind: WorkspaceItemKind) -> None:
        expected_basis = _expected_inclusion_basis(self.human_review)
        if self.inclusion_basis != expected_basis:
            raise ValueError("Detailed report inclusion basis is inconsistent.")
        if self.human_review is not None and (
            self.human_review.target_item_id != self.source_item_id
            or self.human_review.target_item_kind != item_kind
            or self.human_review.target_item_authority_sha256
            != self.target_item_authority_sha256
        ):
            raise ValueError("Human review does not match report source authority.")
        expected_id = deterministic_report_entry_id(
            source_item_id=self.source_item_id,
            target_item_authority_sha256=self.target_item_authority_sha256,
            item_kind=item_kind,
            review_record_id=_review_semantic_id(self.human_review),
            inclusion_basis=self.inclusion_basis,
            identity_version=self.identity_version,
            projection_version=self.projection_version,
        )
        if self.report_entry_id != expected_id:
            raise ValueError("Report entry identity does not match semantic inputs.")


class ReportFindingEntry(_ReportEntryBase):
    item_kind: Literal[WorkspaceItemKind.FINDING] = WorkspaceItemKind.FINDING
    finding_id: str = Field(pattern=r"^finding_[0-9a-f]{64}$")
    comparison_id: str = Field(min_length=1)
    machine_decision: ComparisonDecision
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
    def validate_finding_entry(self):
        if self.source_item_id != self.finding_id:
            raise ValueError("Finding report entry must retain original finding_id.")
        if (
            self.candidate_id != self.plan_source.candidate_id
            or self.requirement_id != self.standard_source.requirement_id
            or self.plan_fact_id != self.plan_source.plan_fact_id
        ):
            raise ValueError("Finding report source authority is inconsistent.")
        if self.human_review is not None and (
            self.human_review.machine_decision != self.machine_decision
            or self.human_review.finding_id != self.finding_id
            or self.human_review.comparison_id != self.comparison_id
        ):
            raise ValueError("Finding human review altered machine authority.")
        self.validate_projection_identity(WorkspaceItemKind.FINDING)
        return self


class ReportGapEntry(_ReportEntryBase):
    item_kind: Literal[WorkspaceItemKind.REVIEW_GAP] = WorkspaceItemKind.REVIEW_GAP
    review_gap_id: str = Field(pattern=r"^reviewgap_[0-9a-f]{64}$")
    gap_source: ReviewGapSource
    candidate_trace_id: str = Field(pattern=r"^candidatetrace_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    requirement_trace_id: str | None = None
    requirement_id: str | None = None
    unresolved_span_id: str | None = None
    candidate_terminal_state: WholePlanCandidateTerminalState | None = None
    requirement_terminal_state: WholePlanRequirementTerminalState | None = None
    decomposition_status: RequirementDecompositionStatus | None = None
    unresolved_reason: UnresolvedRequirementReason | None = None
    plan_source: PlanSourceLocator
    standard_source: StandardSourceLocator | None = None

    @model_validator(mode="after")
    def validate_gap_entry(self):
        if self.source_item_id != self.review_gap_id:
            raise ValueError("Gap report entry must retain original review_gap_id.")
        if self.candidate_id != self.plan_source.candidate_id:
            raise ValueError("Gap report locator belongs to another candidate.")
        self.validate_projection_identity(WorkspaceItemKind.REVIEW_GAP)
        return self


class ReportInternalErrorEntry(_ReportEntryBase):
    item_kind: Literal[WorkspaceItemKind.INTERNAL_ERROR] = (
        WorkspaceItemKind.INTERNAL_ERROR
    )
    internal_error_item_id: str = Field(pattern=r"^workspaceerror_[0-9a-f]{64}$")
    candidate_trace_id: str = Field(pattern=r"^candidatetrace_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    requirement_trace_id: str | None = None
    requirement_id: str | None = None
    error_stage: WholePlanStage
    error_code: WholePlanInternalErrorCode
    plan_source: PlanSourceLocator

    @model_validator(mode="after")
    def validate_error_entry(self):
        if self.source_item_id != self.internal_error_item_id:
            raise ValueError("Error report entry must retain original item identity.")
        if self.candidate_id != self.plan_source.candidate_id:
            raise ValueError("Error report locator belongs to another candidate.")
        self.validate_projection_identity(WorkspaceItemKind.INTERNAL_ERROR)
        return self


def deterministic_excluded_reference_id(
    *,
    source_item_id: str,
    target_item_authority_sha256: str,
    item_kind: WorkspaceItemKind,
    machine_decision: ComparisonDecision | None,
    gap_source: ReviewGapSource | None,
    candidate_terminal_state: WholePlanCandidateTerminalState | None,
    requirement_terminal_state: WholePlanRequirementTerminalState | None,
    decomposition_status: RequirementDecompositionStatus | None,
    unresolved_reason: UnresolvedRequirementReason | None,
    error_stage: WholePlanStage | None,
    error_code: WholePlanInternalErrorCode | None,
    review_record_id: str,
    identity_version: str = EXCLUDED_REFERENCE_IDENTITY_VERSION,
    projection_version: str = REVIEW_REPORT_PROJECTION_VERSION,
) -> str:
    return _canonical_id(
        "excludedreportitem_",
        {
            "identity_version": identity_version,
            "projection_version": projection_version,
            "source_item_id": source_item_id,
            "target_item_authority_sha256": target_item_authority_sha256,
            "item_kind": item_kind.value,
            "machine_decision": machine_decision.value if machine_decision else None,
            "gap_source": gap_source.value if gap_source else None,
            "candidate_terminal_state": (
                candidate_terminal_state.value if candidate_terminal_state else None
            ),
            "requirement_terminal_state": (
                requirement_terminal_state.value
                if requirement_terminal_state
                else None
            ),
            "decomposition_status": (
                decomposition_status.value if decomposition_status else None
            ),
            "unresolved_reason": unresolved_reason.value if unresolved_reason else None,
            "error_stage": error_stage.value if error_stage else None,
            "error_code": error_code.value if error_code else None,
            "review_record_id": review_record_id,
            "report_inclusion": ReportInclusionStatus.EXCLUDE.value,
        },
    )


class ExcludedReportItemReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[EXCLUDED_REFERENCE_IDENTITY_VERSION] = (
        EXCLUDED_REFERENCE_IDENTITY_VERSION
    )
    projection_version: Literal[REVIEW_REPORT_PROJECTION_VERSION] = (
        REVIEW_REPORT_PROJECTION_VERSION
    )
    excluded_reference_id: str = Field(
        pattern=r"^excludedreportitem_[0-9a-f]{64}$"
    )
    source_ordinal: int = Field(ge=0)
    source_item_id: str = Field(min_length=1)
    target_item_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_kind: WorkspaceItemKind
    machine_decision: ComparisonDecision | None = None
    gap_source: ReviewGapSource | None = None
    candidate_terminal_state: WholePlanCandidateTerminalState | None = None
    requirement_terminal_state: WholePlanRequirementTerminalState | None = None
    decomposition_status: RequirementDecompositionStatus | None = None
    unresolved_reason: UnresolvedRequirementReason | None = None
    error_stage: WholePlanStage | None = None
    error_code: WholePlanInternalErrorCode | None = None
    review_record_id: str = Field(pattern=r"^humanreview_[0-9a-f]{64}$")
    report_inclusion: Literal[ReportInclusionStatus.EXCLUDE] = (
        ReportInclusionStatus.EXCLUDE
    )

    @model_validator(mode="after")
    def validate_excluded_reference(self):
        finding_values = (self.machine_decision,)
        gap_values = (
            self.gap_source,
            self.candidate_terminal_state,
            self.requirement_terminal_state,
            self.decomposition_status,
            self.unresolved_reason,
        )
        error_values = (self.error_stage, self.error_code)
        if self.item_kind == WorkspaceItemKind.FINDING:
            if self.machine_decision is None or any(
                value is not None for value in gap_values + error_values
            ):
                raise ValueError("Excluded Finding reference has contradictory status.")
        elif self.item_kind == WorkspaceItemKind.REVIEW_GAP:
            if self.gap_source is None or any(
                value is not None for value in finding_values + error_values
            ):
                raise ValueError("Excluded gap reference has contradictory status.")
        elif any(value is not None for value in finding_values + gap_values) or any(
            value is None for value in error_values
        ):
            raise ValueError("Excluded error reference has contradictory status.")
        expected = deterministic_excluded_reference_id(
            source_item_id=self.source_item_id,
            target_item_authority_sha256=self.target_item_authority_sha256,
            item_kind=self.item_kind,
            machine_decision=self.machine_decision,
            gap_source=self.gap_source,
            candidate_terminal_state=self.candidate_terminal_state,
            requirement_terminal_state=self.requirement_terminal_state,
            decomposition_status=self.decomposition_status,
            unresolved_reason=self.unresolved_reason,
            error_stage=self.error_stage,
            error_code=self.error_code,
            review_record_id=self.review_record_id,
            identity_version=self.identity_version,
            projection_version=self.projection_version,
        )
        if self.excluded_reference_id != expected:
            raise ValueError("Excluded reference identity does not match authority.")
        return self


class ReportCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_findings: int = Field(ge=0)
    workspace_review_gaps: int = Field(ge=0)
    workspace_internal_errors: int = Field(ge=0)
    machine_finding_compliant: int = Field(ge=0)
    machine_finding_non_compliant: int = Field(ge=0)
    machine_finding_insufficient_information: int = Field(ge=0)
    human_reviewed: int = Field(ge=0)
    human_unreviewed: int = Field(ge=0)
    included_findings: int = Field(ge=0)
    included_review_gaps: int = Field(ge=0)
    included_internal_errors: int = Field(ge=0)
    excluded_findings: int = Field(ge=0)
    excluded_review_gaps: int = Field(ge=0)
    excluded_internal_errors: int = Field(ge=0)


class ReportCoverageStatement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: Literal[REPORT_COVERAGE_CODE] = REPORT_COVERAGE_CODE
    statement: Literal[REPORT_COVERAGE_STATEMENT] = REPORT_COVERAGE_STATEMENT


def derive_report_counts(
    *,
    findings: tuple[ReportFindingEntry, ...],
    review_gaps: tuple[ReportGapEntry, ...],
    internal_errors: tuple[ReportInternalErrorEntry, ...],
    excluded_items: tuple[ExcludedReportItemReference, ...],
) -> ReportCounts:
    excluded_findings = sum(
        item.item_kind == WorkspaceItemKind.FINDING for item in excluded_items
    )
    excluded_gaps = sum(
        item.item_kind == WorkspaceItemKind.REVIEW_GAP for item in excluded_items
    )
    excluded_errors = sum(
        item.item_kind == WorkspaceItemKind.INTERNAL_ERROR for item in excluded_items
    )
    reviewed = sum(
        entry.human_review is not None
        for section in (findings, review_gaps, internal_errors)
        for entry in section
    ) + len(excluded_items)
    unreviewed = sum(
        entry.human_review is None
        for section in (findings, review_gaps, internal_errors)
        for entry in section
    )
    finding_decisions = [entry.machine_decision for entry in findings] + [
        item.machine_decision
        for item in excluded_items
        if item.item_kind == WorkspaceItemKind.FINDING
    ]
    return ReportCounts(
        workspace_findings=len(findings) + excluded_findings,
        workspace_review_gaps=len(review_gaps) + excluded_gaps,
        workspace_internal_errors=len(internal_errors) + excluded_errors,
        machine_finding_compliant=sum(
            decision == ComparisonDecision.COMPLIANT for decision in finding_decisions
        ),
        machine_finding_non_compliant=sum(
            decision == ComparisonDecision.NON_COMPLIANT
            for decision in finding_decisions
        ),
        machine_finding_insufficient_information=sum(
            decision == ComparisonDecision.INSUFFICIENT_INFORMATION
            for decision in finding_decisions
        ),
        human_reviewed=reviewed,
        human_unreviewed=unreviewed,
        included_findings=len(findings),
        included_review_gaps=len(review_gaps),
        included_internal_errors=len(internal_errors),
        excluded_findings=excluded_findings,
        excluded_review_gaps=excluded_gaps,
        excluded_internal_errors=excluded_errors,
    )


def deterministic_report_model_id(
    *,
    document: ReportDocumentIdentity,
    findings: tuple[ReportFindingEntry, ...],
    review_gaps: tuple[ReportGapEntry, ...],
    internal_errors: tuple[ReportInternalErrorEntry, ...],
    excluded_items: tuple[ExcludedReportItemReference, ...],
    counts: ReportCounts,
    coverage: ReportCoverageStatement,
    projection_version: str = REVIEW_REPORT_PROJECTION_VERSION,
) -> str:
    return _canonical_id(
        "reviewreport_",
        {
            "projection_version": projection_version,
            "document_sha256": document.document_sha256,
            "whole_plan_review_id": document.whole_plan_review_id,
            "workspace_id": document.workspace_id,
            "review_set_id": document.review_set_id,
            "ordered_included_entry_ids": [
                entry.report_entry_id
                for section in (findings, review_gaps, internal_errors)
                for entry in section
            ],
            "ordered_excluded_reference_ids": [
                item.excluded_reference_id for item in excluded_items
            ],
            "counts": counts.model_dump(mode="json"),
            "coverage": coverage.model_dump(mode="json"),
        },
    )


class ReviewReportModel(BaseModel):
    """Complete semantic report content; formatting is outside this contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_version: Literal[REVIEW_REPORT_PROJECTION_VERSION] = (
        REVIEW_REPORT_PROJECTION_VERSION
    )
    report_model_id: str = Field(pattern=r"^reviewreport_[0-9a-f]{64}$")
    document: ReportDocumentIdentity
    processing: ReportProcessingSummary
    findings: tuple[ReportFindingEntry, ...] = ()
    review_gaps: tuple[ReportGapEntry, ...] = ()
    internal_errors: tuple[ReportInternalErrorEntry, ...] = ()
    excluded_items: tuple[ExcludedReportItemReference, ...] = ()
    counts: ReportCounts
    coverage: ReportCoverageStatement

    @model_validator(mode="after")
    def validate_complete_report(self):
        sections = (self.findings, self.review_gaps, self.internal_errors)
        if any(
            section != tuple(sorted(section, key=lambda item: item.source_ordinal))
            for section in sections
        ) or self.excluded_items != tuple(
            sorted(self.excluded_items, key=lambda item: item.source_ordinal)
        ):
            raise ValueError("Report sections must preserve workspace source order.")
        all_items = [entry for section in sections for entry in section] + list(
            self.excluded_items
        )
        source_ids = [item.source_item_id for item in all_items]
        ordinals = [item.source_ordinal for item in all_items]
        if len(source_ids) != len(set(source_ids)) or len(ordinals) != len(
            set(ordinals)
        ):
            raise ValueError("Every workspace item must have one report accounting path.")
        if sorted(ordinals) != list(range(len(all_items))):
            raise ValueError("Report accounting must cover every workspace source ordinal.")
        expected_counts = derive_report_counts(
            findings=self.findings,
            review_gaps=self.review_gaps,
            internal_errors=self.internal_errors,
            excluded_items=self.excluded_items,
        )
        if self.counts != expected_counts:
            raise ValueError("Report counts must derive from semantic report items.")
        workspace_counts = self.processing.workspace_counts
        if (
            self.counts.workspace_findings != workspace_counts.findings
            or self.counts.workspace_review_gaps != workspace_counts.review_gaps
            or self.counts.workspace_internal_errors != workspace_counts.internal_errors
            or self.counts.machine_finding_compliant
            != workspace_counts.finding_compliant
            or self.counts.machine_finding_non_compliant
            != workspace_counts.finding_non_compliant
            or self.counts.machine_finding_insufficient_information
            != workspace_counts.finding_insufficient_information
            or self.counts.human_reviewed
            != self.processing.human_review_counts.reviewed_items
            or self.counts.human_unreviewed
            != self.processing.human_review_counts.unreviewed_items
        ):
            raise ValueError("Report accounting differs from frozen upstream counts.")
        for section in sections:
            for entry in section:
                if entry.plan_source.document_id != self.document.document_id or (
                    entry.plan_source.document_sha256 != self.document.document_sha256
                ):
                    raise ValueError("Report source belongs to another document.")
                if entry.human_review is not None and (
                    entry.human_review.document_id != self.document.document_id
                    or entry.human_review.document_sha256
                    != self.document.document_sha256
                    or entry.human_review.workspace_id != self.document.workspace_id
                ):
                    raise ValueError("Report human review belongs to another authority.")
        expected_id = deterministic_report_model_id(
            document=self.document,
            findings=self.findings,
            review_gaps=self.review_gaps,
            internal_errors=self.internal_errors,
            excluded_items=self.excluded_items,
            counts=self.counts,
            coverage=self.coverage,
            projection_version=self.projection_version,
        )
        if self.report_model_id != expected_id:
            raise ValueError("ReportModel identity does not match semantic content.")
        return self
