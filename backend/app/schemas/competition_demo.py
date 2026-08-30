"""Typed presentation contracts for the competition demonstration UI.

These contracts project live C.3 authority for presentation. They do not
create document-level compliance authority or accept caller-supplied review
inputs.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.compliance_comparison import (
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonReasonCode,
    MissingInformationItem,
)


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


class CompetitionCaseSummary(_FrozenPresentationModel):
    case_id: str
    label: str
    description: str
    document_id: str
    page_number: int = Field(ge=1)
    standard_code: str
    article_number: str


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


class CompetitionCaseRunResult(_FrozenPresentationModel):
    status: Literal["SUCCESS"] = "SUCCESS"
    message: str
    case_id: str
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
