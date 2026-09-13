"""Apply human workflow metadata to one server-created D.6 workspace."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from app.schemas.finding_review import ReviewerIdentityAssurance
from app.schemas.findings_workspace import (
    FindingWorkspaceItem,
    FindingsWorkspaceResult,
    WorkspaceItemKind,
    workspace_item_authority_id,
)
from app.schemas.human_review import (
    FindingWorkspaceReviewCommand,
    HumanReviewResult,
    InternalErrorWorkspaceReviewCommand,
    ReviewGapWorkspaceReviewCommand,
    WorkspaceHumanAction,
    WorkspaceItemReviewCommand,
    WorkspaceItemReviewRecord,
    derive_human_review_counts,
    derive_review_completeness,
    deterministic_review_record_id,
    deterministic_review_set_id,
    deterministic_reviewer_note_sha256,
    deterministic_workspace_item_authority_sha256,
)
from app.services.review.findings_workspace_service import FindingsWorkspaceService


class HumanReviewError(ValueError):
    pass


class HumanReviewCommandError(HumanReviewError):
    pass


class HumanReviewStaleWorkspaceError(HumanReviewError):
    pass


class HumanReviewTargetError(HumanReviewError):
    pass


class HumanReviewConflictError(HumanReviewError):
    pass


class HumanReviewAuthorityInvariantError(HumanReviewError):
    pass


_COMMAND_TYPES = (
    FindingWorkspaceReviewCommand,
    ReviewGapWorkspaceReviewCommand,
    InternalErrorWorkspaceReviewCommand,
)


class HumanReviewService:
    """Build exactly one D.6 workspace and apply stateless typed commands."""

    def __init__(
        self,
        *,
        findings_workspace_service: FindingsWorkspaceService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.findings_workspace_service = (
            findings_workspace_service or FindingsWorkspaceService()
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def review_document(
        self,
        document_id: str,
        commands: tuple[WorkspaceItemReviewCommand, ...],
    ) -> HumanReviewResult:
        """Resolve all targets from one current server-created D.6 workspace."""

        if not isinstance(commands, tuple) or any(
            not isinstance(command, _COMMAND_TYPES) for command in commands
        ):
            raise HumanReviewCommandError("commands must be a tuple of typed commands.")

        workspace = self.findings_workspace_service.build_workspace(document_id)
        self._require_workspace_document(workspace, document_id)
        items_by_id = {
            workspace_item_authority_id(item): item for item in workspace.items
        }
        if len(items_by_id) != len(workspace.items):
            raise HumanReviewAuthorityInvariantError(
                "D.6 workspace contains duplicate item authority identities."
            )

        effective_commands = {}
        for command in commands:
            if command.expected_workspace_id != workspace.workspace_id:
                raise HumanReviewStaleWorkspaceError(
                    "Review command targets a stale or different workspace."
                )
            item = items_by_id.get(command.target_item_id)
            if item is None:
                raise HumanReviewTargetError(
                    "Review command target does not exist in the current workspace."
                )
            if command.target_item_kind != item.item_kind:
                raise HumanReviewTargetError(
                    "Review command item kind does not match current workspace authority."
                )
            existing = effective_commands.get(command.target_item_id)
            if existing is not None:
                if existing != command:
                    raise HumanReviewConflictError(
                        "Conflicting human review commands target the same workspace item."
                    )
                continue
            effective_commands[command.target_item_id] = command

        records = []
        unreviewed_item_ids = []
        for item in workspace.items:
            item_id = workspace_item_authority_id(item)
            command = effective_commands.get(item_id)
            if command is None:
                unreviewed_item_ids.append(item_id)
                continue
            records.append(self._record(workspace, item, command))

        record_tuple = tuple(records)
        counts = derive_human_review_counts(workspace, record_tuple)
        completeness = derive_review_completeness(
            workspace_item_count=len(workspace.items),
            reviewed_item_count=len(record_tuple),
        )
        return HumanReviewResult(
            review_set_id=deterministic_review_set_id(
                workspace_id=workspace.workspace_id, records=record_tuple
            ),
            document_id=workspace.document_id,
            document_sha256=workspace.document_sha256,
            workspace_id=workspace.workspace_id,
            workspace_authority=workspace,
            completeness=completeness,
            records=record_tuple,
            unreviewed_item_ids=tuple(unreviewed_item_ids),
            counts=counts,
        )

    @staticmethod
    def _require_workspace_document(
        workspace: FindingsWorkspaceResult, document_id: str
    ) -> None:
        if not isinstance(workspace, FindingsWorkspaceResult):
            raise HumanReviewAuthorityInvariantError(
                "D.6 returned an invalid workspace authority type."
            )
        if workspace.document_id != document_id:
            raise HumanReviewAuthorityInvariantError(
                "D.6 workspace belongs to another document."
            )

    def _record(self, workspace, item, command) -> WorkspaceItemReviewRecord:
        created_at = self.clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise HumanReviewCommandError("Server clock must return timezone-aware UTC.")
        created_at = created_at.astimezone(timezone.utc)
        note_hash = deterministic_reviewer_note_sha256(command.reviewer_note)
        authority_hash = deterministic_workspace_item_authority_sha256(item)
        if isinstance(command, FindingWorkspaceReviewCommand):
            human_action = WorkspaceHumanAction(command.disposition.value)
        else:
            human_action = WorkspaceHumanAction(command.action.value)

        machine_decision = None
        finding_id = None
        comparison_id = None
        if isinstance(item, FindingWorkspaceItem):
            machine_decision = item.decision
            finding_id = item.finding_id
            comparison_id = item.comparison_id

        record_id = deterministic_review_record_id(
            document_id=workspace.document_id,
            document_sha256=workspace.document_sha256,
            workspace_id=workspace.workspace_id,
            target_item_id=workspace_item_authority_id(item),
            target_item_authority_sha256=authority_hash,
            target_item_kind=WorkspaceItemKind(item.item_kind),
            human_action=human_action,
            report_inclusion=command.report_inclusion,
            reviewer_id=command.reviewer_id,
            reviewer_note_sha256=note_hash,
        )
        return WorkspaceItemReviewRecord(
            review_record_id=record_id,
            document_id=workspace.document_id,
            document_sha256=workspace.document_sha256,
            workspace_id=workspace.workspace_id,
            target_item_id=workspace_item_authority_id(item),
            target_item_authority_sha256=authority_hash,
            target_item_kind=item.item_kind,
            machine_decision=machine_decision,
            finding_id=finding_id,
            comparison_id=comparison_id,
            human_action=human_action,
            report_inclusion=command.report_inclusion,
            reviewer_id=command.reviewer_id,
            reviewer_identity_assurance=(
                ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
            ),
            reviewer_note=command.reviewer_note,
            reviewer_note_sha256=note_hash,
            created_at=created_at,
        )
