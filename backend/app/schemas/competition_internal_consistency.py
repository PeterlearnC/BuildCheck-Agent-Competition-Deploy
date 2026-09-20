"""Presentation-only contracts for the controlled Case D demonstration."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.internal_consistency import (
    ConsistencyAssertionClass,
    ConsistencyCandidateReason,
    ConsistencyCandidateStatus,
    ConsistencyCoverageStatus,
    ConsistencyReviewStatus,
    ConsistencyValueRelation,
    NormalizedUnit,
)


class _FrozenConsistencyPresentationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompetitionConsistencySourceView(_FrozenConsistencyPresentationModel):
    physical_page: int = Field(gt=0)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fact_id: str = Field(pattern=r"^consistencyfact_[0-9a-f]{64}$")


class CompetitionConsistencyValueGroupView(_FrozenConsistencyPresentationModel):
    value: Decimal
    unit: NormalizedUnit
    sources: tuple[CompetitionConsistencySourceView, ...] = Field(min_length=1)


class CompetitionConsistencyCandidateView(_FrozenConsistencyPresentationModel):
    candidate_id: str = Field(pattern=r"^consistencycandidate_[0-9a-f]{64}$")
    status: ConsistencyCandidateStatus
    relation: ConsistencyValueRelation
    reason: ConsistencyCandidateReason
    review_status: ConsistencyReviewStatus
    object_display_name: str = Field(min_length=1)
    parameter_display_name: str = Field(min_length=1)
    unit: NormalizedUnit
    assertion_class: ConsistencyAssertionClass
    value_groups: tuple[CompetitionConsistencyValueGroupView, ...] = Field(min_length=2)


class CompetitionInternalConsistencyResponse(_FrozenConsistencyPresentationModel):
    case_id: Literal["CASE-D"] = "CASE-D"
    classification: Literal["CONTROLLED_SYNTHETIC"] = "CONTROLLED_SYNTHETIC"
    display_label: Literal["受控合成演示案例"] = "受控合成演示案例"
    purpose: Literal["INTERNAL_CONSISTENCY_DEMONSTRATION"] = (
        "INTERNAL_CONSISTENCY_DEMONSTRATION"
    )
    document_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_id: str = Field(pattern=r"^consistencyreview_[0-9a-f]{64}$")
    coverage_status: ConsistencyCoverageStatus
    candidate_count: int = Field(ge=0)
    candidates: tuple[CompetitionConsistencyCandidateView, ...]
