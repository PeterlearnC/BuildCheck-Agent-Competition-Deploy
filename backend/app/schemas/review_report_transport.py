"""Strict HTTP transport inputs for frozen D.7 report artifacts."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.human_review import WorkspaceItemReviewCommand
from app.schemas.review_report_artifact import ReviewReportArtifactFormat


MAX_REVIEW_COMMANDS = 2000
MAX_REPORT_ARTIFACT_BYTES = 64 * 1024 * 1024


class ReviewReportDelivery(str, Enum):
    """HTTP presentation behavior only; it never changes artifact bytes."""

    INLINE = "INLINE"
    ATTACHMENT = "ATTACHMENT"


class ReviewReportTransportRequest(BaseModel):
    """Caller-supplied workflow commands plus presentation selection only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commands: tuple[WorkspaceItemReviewCommand, ...] = Field(
        default=(), max_length=MAX_REVIEW_COMMANDS
    )
    format: ReviewReportArtifactFormat
    delivery: ReviewReportDelivery
