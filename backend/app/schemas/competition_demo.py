"""Typed presentation contracts for the qualified competition demo.

Successful responses are a strict union of live C.3 Findings and live D.6
ReviewGaps.  The transport layer does not create either machine authority.
"""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.compliance_comparison import (
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonReasonCode,
    MissingInformationItem,
)
from app.schemas.findings_workspace import ReviewGapSource
from app.schemas.whole_plan_review import WholePlanCandidateTerminalState


class _FrozenPresentationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompetitionReadinessCheck(_FrozenPresentationModel):
    key: str
    label: str
    ready: bool
    detail: str


class CompetitionPlanSummary(_FrozenPresentationModel):
    document_id: str
    display_name: str
    pdf_sha256: str
    verified: bool
    case_count: int = Field(ge=0)


class CompetitionFindingCaseSummary(_FrozenPresentationModel):
    case_id: str
    item_kind: Literal["FINDING"] = "FINDING"
    label: str
    description: str
    document_id: str
    page_number: int = Field(ge=1)
    standard_code: str
    article_number: str


class CompetitionReviewGapCaseSummary(_FrozenPresentationModel):
    case_id: str
    item_kind: Literal["REVIEW_GAP"] = "REVIEW_GAP"
    label: str
    description: str
    document_id: str
    page_number: int = Field(ge=1)


CompetitionCaseSummary = Annotated[
    CompetitionFindingCaseSummary | CompetitionReviewGapCaseSummary,
    Field(discriminator="item_kind"),
]


class CompetitionDemoMetadata(_FrozenPresentationModel):
    mode: Literal["QUALIFIED_SAMPLE_MODE"] = "QUALIFIED_SAMPLE_MODE"
    schema_version: str
    status: Literal["READY", "NOT_READY"]
    readiness_checks: tuple[CompetitionReadinessCheck, ...]
    plans: tuple[CompetitionPlanSummary, ...]
    cases: tuple[CompetitionCaseSummary, ...]
    sample_mode_notice: str
    scope_notice: str


class CompetitionPlanFactEvidence(_FrozenPresentationModel):
    plan_fact_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    source_text: str
    source_text_sha256: str
    normalized_value: Decimal | bool | str | None = None
    unit: str | None = None


class CompetitionPlanEvidence(_FrozenPresentationModel):
    document_display_name: str
    document_id: str
    pdf_sha256: str
    page_number: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    exact_text: str
    text_sha256: str
    review_unit_id: str
    plan_facts: tuple[CompetitionPlanFactEvidence, ...]


class CompetitionStandardEvidence(_FrozenPresentationModel):
    standard_id: str
    standard_code: str
    article_number: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    exact_requirement_text: str
    requirement_text_sha256: str
    official_source_sha256: str
    evidence_id: str
    requirement_id: str


class CompetitionTechnicalProvenance(_FrozenPresentationModel):
    finding_id: str
    comparison_id: str
    document_id: str
    document_sha256: str
    standard_id: str
    standard_source_sha256: str
    evidence_id: str
    requirement_id: str
    review_unit_id: str
    plan_fact_ids: tuple[str, ...]
    decision_scope: ComparisonDecisionScope


class CompetitionFindingCaseRunResult(_FrozenPresentationModel):
    status: Literal["SUCCESS"] = "SUCCESS"
    message: str
    case_id: str
    item_kind: Literal["FINDING"] = "FINDING"
    finding_id: str
    decision: ComparisonDecision
    decision_label: str
    reason_code: ComparisonReasonCode
    decision_scope: ComparisonDecisionScope
    summary: str
    missing_information: tuple[MissingInformationItem, ...]
    local_explanation: str
    scope_notice: str
    plan_evidence: CompetitionPlanEvidence
    standard_evidence: CompetitionStandardEvidence
    technical_provenance: CompetitionTechnicalProvenance


class CompetitionReviewGapPlanEvidence(_FrozenPresentationModel):
    document_display_name: str
    document_id: str
    pdf_sha256: str
    physical_page: int = Field(ge=1)
    page_char_start: int = Field(ge=0)
    page_char_end: int = Field(gt=0)
    exact_text: str
    text_sha256: str
    candidate_id: str


class CompetitionReviewGapTechnicalProvenance(_FrozenPresentationModel):
    workspace_id: str
    whole_plan_review_id: str
    review_gap_id: str
    document_id: str
    document_sha256: str
    candidate_trace_id: str
    candidate_id: str
    identity_version: str
    projection_version: str


class CompetitionReviewGapCaseRunResult(_FrozenPresentationModel):
    status: Literal["SUCCESS"] = "SUCCESS"
    message: str
    case_id: str
    item_kind: Literal["REVIEW_GAP"] = "REVIEW_GAP"
    review_gap_id: str
    terminal_class: Literal[ReviewGapSource.CANDIDATE_TERMINAL]
    terminal_status: Literal[WholePlanCandidateTerminalState.NO_STANDARD_SCOPE]
    local_explanation: str
    scope_notice: str
    finding_absent: Literal[True] = True
    comparison_absent: Literal[True] = True
    decision_absent: Literal[True] = True
    standard_authority_absent: Literal[True] = True
    article_authority_absent: Literal[True] = True
    requirement_authority_absent: Literal[True] = True
    plan_evidence: CompetitionReviewGapPlanEvidence
    technical_provenance: CompetitionReviewGapTechnicalProvenance


CompetitionCaseRunResult = Annotated[
    CompetitionFindingCaseRunResult | CompetitionReviewGapCaseRunResult,
    Field(discriminator="item_kind"),
]
