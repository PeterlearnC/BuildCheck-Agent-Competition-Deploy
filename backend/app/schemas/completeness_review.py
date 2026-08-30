"""Explainable completeness-review models built on V0.2 analysis output."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ReviewStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ReviewSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    IMPORTANT = "IMPORTANT"


class EvidenceSourceType(str, Enum):
    CHAPTER = "chapter"
    FIELD = "field"
    SECTION_PRESENCE = "section_presence"
    TEXT = "text"


class CompletenessEvidence(BaseModel):
    source_type: EvidenceSourceType
    source_page: int | None = None
    source_text: str | None = None
    matched_title: str | None = None
    field_path: str | None = None


class CompletenessCheckResult(BaseModel):
    check_id: str
    title: str
    description: str
    status: ReviewStatus
    severity: ReviewSeverity
    reason: str
    evidence: list[CompletenessEvidence] = Field(default_factory=list)
    suggestion: str | None = None

    @model_validator(mode="after")
    def require_evidence_for_positive_findings(self):
        if self.status in {ReviewStatus.PASS, ReviewStatus.PARTIAL} and not self.evidence:
            raise ValueError("PASS and PARTIAL completeness results require evidence.")
        return self


class CompletenessSummary(BaseModel):
    total_checks: int
    applicable_checks: int
    passed: int
    partial: int
    missing: int
    not_applicable: int
    completeness_score: int


class CompletenessReview(BaseModel):
    analysis_document_type: str | None = None
    review_profile: str
    summary: CompletenessSummary
    checks: list[CompletenessCheckResult]
    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class CompletenessReviewResponse(BaseModel):
    success: bool = True
    document_id: str
    cached: bool = False
    review: CompletenessReview


class CompletenessReviewCache(BaseModel):
    review_version: str
    document_id: str
    analysis_fingerprint: str
    review: CompletenessReview
