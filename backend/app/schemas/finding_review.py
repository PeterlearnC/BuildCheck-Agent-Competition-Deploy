"""Stateless human-workflow events over immutable C.3 ReviewFinding authority."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.review_finding import ReviewFinding


REVIEW_EVENT_VERSION = "v0.1-c.4-f1-stateless-review-event"
MAX_REVIEWER_ID_LENGTH = 128
MAX_REVIEWER_NOTE_LENGTH = 4000


class FindingReviewDisposition(str, Enum):
    """Human handling only; these values never replace a C.3 decision.

    ACCEPTED means the reviewer accepts this exact Finding for workflow
    consideration, never that a document passed or failed. REJECTED means the
    reviewer declines to use it in this workflow, not that C.3 is technically
    false. NEEDS_INFORMATION requests more evidence without rewriting C.3's
    reason or decision. DEFERRED postpones handling without changing authority.
    """

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NEEDS_INFORMATION = "NEEDS_INFORMATION"
    DEFERRED = "DEFERRED"


class ReportInclusionStatus(str, Enum):
    """Independent workflow metadata, never inferred from disposition."""

    UNDECIDED = "UNDECIDED"
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"


class ReviewerIdentityAssurance(str, Enum):
    """C.4-F1 has no authentication authority beyond a caller assertion."""

    CALLER_ASSERTED_UNVERIFIED = "CALLER_ASSERTED_UNVERIFIED"


class FindingReviewCommand(BaseModel):
    """Caller workflow input with no Finding, identity, time, or history authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer_id: str = Field(min_length=1, max_length=MAX_REVIEWER_ID_LENGTH)
    disposition: FindingReviewDisposition
    report_inclusion: ReportInclusionStatus = ReportInclusionStatus.UNDECIDED
    reviewer_note: str = Field(default="", max_length=MAX_REVIEWER_NOTE_LENGTH)

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_id_is_bounded_and_explicit(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("reviewer_id must be non-blank without edge whitespace.")
        return value


def deterministic_reviewer_note_sha256(note: str) -> str:
    return hashlib.sha256(note.encode("utf-8")).hexdigest()


def deterministic_finding_authority_sha256(finding: ReviewFinding) -> str:
    serialized = json.dumps(
        finding.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def canonical_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware.")
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def deterministic_review_event_id(
    *,
    document_id: str,
    finding_id: str,
    reviewer_id: str,
    reviewer_identity_assurance: ReviewerIdentityAssurance,
    disposition: FindingReviewDisposition,
    report_inclusion: ReportInclusionStatus,
    reviewer_note_sha256: str,
    occurred_at: datetime,
) -> str:
    payload = {
        "review_event_version": REVIEW_EVENT_VERSION,
        "document_id": document_id,
        "finding_id": finding_id,
        "reviewer_id": reviewer_id,
        "reviewer_identity_assurance": reviewer_identity_assurance.value,
        "disposition": disposition.value,
        "report_inclusion": report_inclusion.value,
        "reviewer_note_sha256": reviewer_note_sha256,
        "occurred_at": canonical_utc_timestamp(occurred_at),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "reviewevent_" + hashlib.sha256(serialized).hexdigest()


class FindingReviewEvent(BaseModel):
    """One immutable, stateless human action referencing one C.3 Finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_event_id: str
    review_event_version: Literal[REVIEW_EVENT_VERSION] = REVIEW_EVENT_VERSION
    document_id: str
    finding_id: str
    finding_authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1, max_length=MAX_REVIEWER_ID_LENGTH)
    reviewer_identity_assurance: Literal[
        ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
    ] = ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
    disposition: FindingReviewDisposition
    report_inclusion: ReportInclusionStatus
    reviewer_note: str = Field(max_length=MAX_REVIEWER_NOTE_LENGTH)
    reviewer_note_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    occurred_at: datetime

    @model_validator(mode="after")
    def validate_event_identity(self):
        if not self.reviewer_id.strip() or self.reviewer_id != self.reviewer_id.strip():
            raise ValueError("reviewer_id must be non-blank without edge whitespace.")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware UTC.")
        if self.occurred_at.utcoffset() != timedelta(0):
            raise ValueError("occurred_at must use UTC.")
        expected_note_hash = deterministic_reviewer_note_sha256(self.reviewer_note)
        if self.reviewer_note_sha256 != expected_note_hash:
            raise ValueError("Reviewer note hash does not match reviewer_note.")
        expected_id = deterministic_review_event_id(
            document_id=self.document_id,
            finding_id=self.finding_id,
            reviewer_id=self.reviewer_id,
            reviewer_identity_assurance=ReviewerIdentityAssurance(
                self.reviewer_identity_assurance
            ),
            disposition=self.disposition,
            report_inclusion=self.report_inclusion,
            reviewer_note_sha256=self.reviewer_note_sha256,
            occurred_at=self.occurred_at,
        )
        if self.review_event_id != expected_id:
            raise ValueError("Review event identity does not match canonical inputs.")
        return self


class FindingReviewProjection(BaseModel):
    """Keep immutable C.3 authority separate from one human workflow event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    authoritative_finding: ReviewFinding
    review_event: FindingReviewEvent

    @model_validator(mode="after")
    def validate_authority_reference(self):
        finding = self.authoritative_finding
        event = self.review_event
        if event.document_id != finding.document_id:
            raise ValueError("Review event document differs from C.3 Finding authority.")
        if event.finding_id != finding.finding_id:
            raise ValueError("Review event finding_id differs from C.3 authority.")
        expected_hash = deterministic_finding_authority_sha256(finding)
        if event.finding_authority_sha256 != expected_hash:
            raise ValueError("Review event Finding authority hash does not match C.3.")
        return self
