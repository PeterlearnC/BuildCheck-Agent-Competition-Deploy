"""Build one semantic report from one server-created human-review result."""

from __future__ import annotations

from app.schemas.finding_review import ReportInclusionStatus
from app.schemas.findings_workspace import (
    FindingWorkspaceItem,
    InternalErrorWorkspaceItem,
    ReviewGapWorkspaceItem,
    WorkspaceItemKind,
    workspace_item_authority_id,
)
from app.schemas.human_review import (
    HumanReviewResult,
    WorkspaceItemReviewCommand,
    deterministic_workspace_item_authority_sha256,
)
from app.schemas.review_report import (
    ExcludedReportItemReference,
    ReportCoverageStatement,
    ReportDocumentIdentity,
    ReportFindingEntry,
    ReportGapEntry,
    ReportInclusionBasis,
    ReportInternalErrorEntry,
    ReportProcessingSummary,
    ReviewReportModel,
    UNREVIEWED_SEMANTIC_MARKER,
    derive_report_counts,
    deterministic_excluded_reference_id,
    deterministic_report_entry_id,
    deterministic_report_model_id,
)
from app.services.review.human_review_service import HumanReviewService


class ReviewReportError(ValueError):
    pass


class ReviewReportAuthorityInvariantError(ReviewReportError):
    pass


class ReviewReportService:
    """Delegate once to P1, then project immutable semantic report content."""

    def __init__(
        self, *, human_review_service: HumanReviewService | None = None
    ) -> None:
        self.human_review_service = human_review_service or HumanReviewService()

    def build_report(
        self,
        document_id: str,
        commands: tuple[WorkspaceItemReviewCommand, ...],
    ) -> ReviewReportModel:
        human_review = self.human_review_service.review_document(document_id, commands)
        if not isinstance(human_review, HumanReviewResult):
            raise ReviewReportAuthorityInvariantError(
                "P1 returned an invalid human-review authority type."
            )
        if human_review.document_id != document_id:
            raise ReviewReportAuthorityInvariantError(
                "P1 human-review authority belongs to another document."
            )
        return self._project_report(human_review)

    @staticmethod
    def _project_report(human_review: HumanReviewResult) -> ReviewReportModel:
        workspace = human_review.workspace_authority
        records_by_item = {
            record.target_item_id: record for record in human_review.records
        }
        if len(records_by_item) != len(human_review.records):
            raise ReviewReportAuthorityInvariantError(
                "P1 contains duplicate effective review targets."
            )

        findings = []
        gaps = []
        errors = []
        excluded = []
        for source_ordinal, item in enumerate(workspace.items):
            item_id = workspace_item_authority_id(item)
            review = records_by_item.get(item_id)
            authority_hash = (
                review.target_item_authority_sha256
                if review is not None
                else ReviewReportService._item_authority_sha256(item)
            )
            if review is not None and review.report_inclusion == (
                ReportInclusionStatus.EXCLUDE
            ):
                excluded.append(
                    ReviewReportService._excluded_reference(
                        item=item,
                        source_ordinal=source_ordinal,
                        authority_hash=authority_hash,
                        review_record_id=review.review_record_id,
                    )
                )
                continue

            inclusion_basis = ReviewReportService._inclusion_basis(review)
            review_record_id = (
                review.review_record_id
                if review is not None
                else UNREVIEWED_SEMANTIC_MARKER
            )
            entry_id = deterministic_report_entry_id(
                source_item_id=item_id,
                target_item_authority_sha256=authority_hash,
                item_kind=item.item_kind,
                review_record_id=review_record_id,
                inclusion_basis=inclusion_basis,
            )
            common = {
                "report_entry_id": entry_id,
                "source_ordinal": source_ordinal,
                "source_item_id": item_id,
                "target_item_authority_sha256": authority_hash,
                "inclusion_basis": inclusion_basis,
                "human_review": review,
            }
            if isinstance(item, FindingWorkspaceItem):
                findings.append(
                    ReportFindingEntry(
                        **common,
                        finding_id=item.finding_id,
                        comparison_id=item.comparison_id,
                        machine_decision=item.decision,
                        reason_code=item.reason_code,
                        candidate_trace_id=item.candidate_trace_id,
                        candidate_id=item.candidate_id,
                        requirement_trace_id=item.requirement_trace_id,
                        requirement_id=item.requirement_id,
                        plan_fact_id=item.plan_fact_id,
                        route_id=item.route_id,
                        plan_source=item.plan_source,
                        standard_source=item.standard_source,
                    )
                )
            elif isinstance(item, ReviewGapWorkspaceItem):
                gaps.append(
                    ReportGapEntry(
                        **common,
                        review_gap_id=item.review_gap_id,
                        gap_source=item.gap_source,
                        candidate_trace_id=item.candidate_trace_id,
                        candidate_id=item.candidate_id,
                        requirement_trace_id=item.requirement_trace_id,
                        requirement_id=item.requirement_id,
                        unresolved_span_id=item.unresolved_span_id,
                        candidate_terminal_state=item.candidate_terminal_state,
                        requirement_terminal_state=item.requirement_terminal_state,
                        decomposition_status=item.decomposition_status,
                        unresolved_reason=item.unresolved_reason,
                        plan_source=item.plan_source,
                        standard_source=item.standard_source,
                    )
                )
            elif isinstance(item, InternalErrorWorkspaceItem):
                errors.append(
                    ReportInternalErrorEntry(
                        **common,
                        internal_error_item_id=item.internal_error_item_id,
                        candidate_trace_id=item.candidate_trace_id,
                        candidate_id=item.candidate_id,
                        requirement_trace_id=item.requirement_trace_id,
                        requirement_id=item.requirement_id,
                        error_stage=item.error.stage,
                        error_code=item.error.code,
                        plan_source=item.plan_source,
                    )
                )
            else:
                raise ReviewReportAuthorityInvariantError(
                    "P1 workspace contains an unknown item type."
                )

        finding_tuple = tuple(findings)
        gap_tuple = tuple(gaps)
        error_tuple = tuple(errors)
        excluded_tuple = tuple(excluded)
        document = ReportDocumentIdentity(
            document_id=human_review.document_id,
            document_sha256=human_review.document_sha256,
            whole_plan_review_id=workspace.whole_plan_review_id,
            workspace_id=human_review.workspace_id,
            review_set_id=human_review.review_set_id,
        )
        processing = ReportProcessingSummary(
            processing_state=workspace.processing_state,
            workspace_counts=workspace.counts,
            source_coverage_counts=workspace.source_coverage_counts,
            human_review_workflow_completeness=human_review.completeness,
            human_review_counts=human_review.counts,
        )
        counts = derive_report_counts(
            findings=finding_tuple,
            review_gaps=gap_tuple,
            internal_errors=error_tuple,
            excluded_items=excluded_tuple,
        )
        coverage = ReportCoverageStatement()
        report_id = deterministic_report_model_id(
            document=document,
            findings=finding_tuple,
            review_gaps=gap_tuple,
            internal_errors=error_tuple,
            excluded_items=excluded_tuple,
            counts=counts,
            coverage=coverage,
        )
        return ReviewReportModel(
            report_model_id=report_id,
            document=document,
            processing=processing,
            findings=finding_tuple,
            review_gaps=gap_tuple,
            internal_errors=error_tuple,
            excluded_items=excluded_tuple,
            counts=counts,
            coverage=coverage,
        )

    @staticmethod
    def _item_authority_sha256(item):
        return deterministic_workspace_item_authority_sha256(item)

    @staticmethod
    def _inclusion_basis(review) -> ReportInclusionBasis:
        if review is None:
            return ReportInclusionBasis.UNREVIEWED_DEFAULT_INCLUDE
        if review.report_inclusion == ReportInclusionStatus.INCLUDE:
            return ReportInclusionBasis.EXPLICIT_INCLUDE
        if review.report_inclusion == ReportInclusionStatus.UNDECIDED:
            return ReportInclusionBasis.UNDECIDED_DEFAULT_INCLUDE
        raise ReviewReportAuthorityInvariantError(
            "Excluded review cannot enter a detailed report section."
        )

    @staticmethod
    def _excluded_reference(
        *, item, source_ordinal: int, authority_hash: str, review_record_id: str
    ) -> ExcludedReportItemReference:
        values = {
            "machine_decision": None,
            "gap_source": None,
            "candidate_terminal_state": None,
            "requirement_terminal_state": None,
            "decomposition_status": None,
            "unresolved_reason": None,
            "error_stage": None,
            "error_code": None,
        }
        if isinstance(item, FindingWorkspaceItem):
            values["machine_decision"] = item.decision
        elif isinstance(item, ReviewGapWorkspaceItem):
            values.update(
                gap_source=item.gap_source,
                candidate_terminal_state=item.candidate_terminal_state,
                requirement_terminal_state=item.requirement_terminal_state,
                decomposition_status=item.decomposition_status,
                unresolved_reason=item.unresolved_reason,
            )
        elif isinstance(item, InternalErrorWorkspaceItem):
            values.update(error_stage=item.error.stage, error_code=item.error.code)
        else:
            raise ReviewReportAuthorityInvariantError(
                "Cannot account for an unknown excluded item type."
            )
        item_id = workspace_item_authority_id(item)
        excluded_id = deterministic_excluded_reference_id(
            source_item_id=item_id,
            target_item_authority_sha256=authority_hash,
            item_kind=WorkspaceItemKind(item.item_kind),
            review_record_id=review_record_id,
            **values,
        )
        return ExcludedReportItemReference(
            excluded_reference_id=excluded_id,
            source_ordinal=source_ordinal,
            source_item_id=item_id,
            target_item_authority_sha256=authority_hash,
            item_kind=item.item_kind,
            review_record_id=review_record_id,
            **values,
        )
