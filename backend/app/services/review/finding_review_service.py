"""Create one stateless human-review event from server-reconstructed C.3 authority."""

from collections.abc import Callable
from datetime import datetime, timezone

from app.schemas.compliance_comparison import ComplianceComparisonRequest
from app.schemas.finding_review import (
    REVIEW_EVENT_VERSION,
    FindingReviewCommand,
    FindingReviewEvent,
    FindingReviewProjection,
    ReviewerIdentityAssurance,
    deterministic_finding_authority_sha256,
    deterministic_review_event_id,
    deterministic_reviewer_note_sha256,
)
from app.schemas.review_finding import ReviewFindingResponse
from app.services.review.review_finding_service import ReviewFindingService


class FindingReviewError(ValueError):
    pass


class FindingReviewCommandError(FindingReviewError):
    pass


class FindingReviewAuthorityMismatchError(FindingReviewError):
    pass


class FindingReviewService:
    """Validate one command without persistence, aggregation, or authority mutation."""

    def __init__(
        self,
        *,
        review_finding_service: ReviewFindingService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.review_finding_service = review_finding_service or ReviewFindingService()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def create(
        self,
        *,
        document_id: str,
        finding_id: str,
        comparison_request: ComplianceComparisonRequest,
        command: FindingReviewCommand,
    ) -> FindingReviewProjection:
        if not isinstance(comparison_request, ComplianceComparisonRequest):
            raise FindingReviewCommandError(
                "C.4 requires a typed C.3 reconstruction request, not Finding authority."
            )
        if not isinstance(command, FindingReviewCommand):
            raise FindingReviewCommandError("C.4 requires a typed workflow command.")

        response = self.review_finding_service.create(document_id, comparison_request)
        if not isinstance(response, ReviewFindingResponse):
            raise FindingReviewAuthorityMismatchError(
                "C.3 returned an invalid Finding response type."
            )
        finding = response.finding
        if finding.document_id != document_id:
            raise FindingReviewAuthorityMismatchError(
                "Reconstructed Finding does not belong to the requested document."
            )
        if finding.finding_id != finding_id:
            raise FindingReviewAuthorityMismatchError(
                "Referenced finding_id does not match reconstructed C.3 authority."
            )

        occurred_at = self.clock()
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise FindingReviewCommandError("Server clock must return timezone-aware UTC.")
        occurred_at = occurred_at.astimezone(timezone.utc)
        note_hash = deterministic_reviewer_note_sha256(command.reviewer_note)
        authority_hash = deterministic_finding_authority_sha256(finding)
        assurance = ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
        event = FindingReviewEvent(
            review_event_id=deterministic_review_event_id(
                document_id=document_id,
                finding_id=finding.finding_id,
                reviewer_id=command.reviewer_id,
                reviewer_identity_assurance=assurance,
                disposition=command.disposition,
                report_inclusion=command.report_inclusion,
                reviewer_note_sha256=note_hash,
                occurred_at=occurred_at,
            ),
            review_event_version=REVIEW_EVENT_VERSION,
            document_id=document_id,
            finding_id=finding.finding_id,
            finding_authority_sha256=authority_hash,
            reviewer_id=command.reviewer_id,
            reviewer_identity_assurance=assurance,
            disposition=command.disposition,
            report_inclusion=command.report_inclusion,
            reviewer_note=command.reviewer_note,
            reviewer_note_sha256=note_hash,
            occurred_at=occurred_at,
        )
        return FindingReviewProjection(
            authoritative_finding=finding,
            review_event=event,
        )
