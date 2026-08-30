"""Fail-closed contracts for local ReviewUnit x requirement comparison."""

import hashlib
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_review import (
    ComplianceReviewRequest,
    ComplianceReviewResult,
    ReviewEvidenceBinding,
)
from app.schemas.normative_requirement import NormativeRequirement


class PlanFactType(str, Enum):
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    ACTION = "ACTION"
    SEQUENCE = "SEQUENCE"
    UNSUPPORTED = "UNSUPPORTED"


class PlanFactParsingStatus(str, Enum):
    PROVEN = "PROVEN"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"


class PlanFact(BaseModel):
    """An exact ReviewUnit subspan with server-derived normalized data."""

    model_config = ConfigDict(frozen=True)

    plan_fact_id: str
    review_unit_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str
    source_text_sha256: str
    fact_type: PlanFactType
    parsing_status: PlanFactParsingStatus
    normalized_value: Decimal | bool | str | None = None
    unit: str | None = None
    parser_method: str
    parser_version: str

    @model_validator(mode="after")
    def validate_source_identity_and_normalization(self):
        if not self.source_text:
            raise ValueError("PlanFact source text must not be empty.")
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("PlanFact span does not match source_text.")
        expected_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected_hash:
            raise ValueError("PlanFact source hash does not match source_text.")
        if self.fact_type == PlanFactType.NUMERIC:
            if self.parsing_status == PlanFactParsingStatus.PROVEN:
                if not isinstance(self.normalized_value, Decimal) or not self.unit:
                    raise ValueError("A proven numeric PlanFact requires value and unit.")
            elif self.normalized_value is not None or self.unit is not None:
                raise ValueError("An unresolved PlanFact cannot carry normalized data.")
        elif self.normalized_value is not None or self.unit is not None:
            raise ValueError("Unsupported PlanFacts cannot carry normalized data.")
        return self


class ComparisonDecision(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


class ComparisonDecisionScope(str, Enum):
    REVIEW_UNIT_REQUIREMENT = "REVIEW_UNIT_REQUIREMENT"


class ApplicabilityStatus(str, Enum):
    APPLICABLE = "APPLICABLE"
    UNRESOLVED = "UNRESOLVED"


class ComparisonReasonCode(str, Enum):
    NUMERIC_LIMIT_SATISFIED = "NUMERIC_LIMIT_SATISFIED"
    NUMERIC_LIMIT_VIOLATED = "NUMERIC_LIMIT_VIOLATED"
    APPLICABILITY_UNRESOLVED = "APPLICABILITY_UNRESOLVED"
    PLAN_FACT_MISSING = "PLAN_FACT_MISSING"
    PLAN_FACT_AMBIGUOUS = "PLAN_FACT_AMBIGUOUS"
    REQUIREMENT_DECOMPOSITION_UNRESOLVED = "REQUIREMENT_DECOMPOSITION_UNRESOLVED"
    UNSUPPORTED_REQUIREMENT_CLASS = "UNSUPPORTED_REQUIREMENT_CLASS"
    UNSUPPORTED_UNIT = "UNSUPPORTED_UNIT"
    CONDITION_UNRESOLVED = "CONDITION_UNRESOLVED"
    SEQUENCE_INCOMPLETE = "SEQUENCE_INCOMPLETE"
    MULTIPLE_REQUIREMENTS_UNRESOLVED = "MULTIPLE_REQUIREMENTS_UNRESOLVED"


class ComparisonMethod(str, Enum):
    DETERMINISTIC_NUMERIC = "DETERMINISTIC_NUMERIC"
    FAIL_CLOSED = "FAIL_CLOSED"


class StructuredComparisonReason(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    plan_fact_ids: list[str] = Field(default_factory=list)
    actual_value: Decimal | None = None
    operator: str | None = None
    required_value: Decimal | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def numeric_details_are_complete(self):
        numeric = (self.actual_value, self.operator, self.required_value, self.unit)
        if any(value is not None for value in numeric) and any(
            value is None for value in numeric
        ):
            raise ValueError("Structured numeric reason details must be complete.")
        return self


class MissingInformationItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ComparisonReasonCode
    detail: str


class ComparisonResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    comparison_id: str
    decision: ComparisonDecision
    decision_scope: ComparisonDecisionScope
    review_unit_id: str
    requirement_id: str
    evidence_id: str
    plan_facts_used: list[str] = Field(default_factory=list)
    requirement_text: str
    reason_code: ComparisonReasonCode
    reason: StructuredComparisonReason
    missing_information: list[MissingInformationItem] = Field(default_factory=list)
    applicability_status: ApplicabilityStatus
    comparison_method: ComparisonMethod
    comparison_version: str

    @model_validator(mode="after")
    def decisive_results_are_fully_supported(self):
        decisive = self.decision in {
            ComparisonDecision.COMPLIANT,
            ComparisonDecision.NON_COMPLIANT,
        }
        if decisive:
            if self.decision_scope != ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT:
                raise ValueError("Decisive comparison scope may not be widened.")
            if self.applicability_status != ApplicabilityStatus.APPLICABLE:
                raise ValueError("Decisive comparison requires proven applicability.")
            if self.missing_information:
                raise ValueError("Decisive comparison cannot have missing information.")
            if self.comparison_method != ComparisonMethod.DETERMINISTIC_NUMERIC:
                raise ValueError("Decisive comparison requires a supported method.")
            if not self.plan_facts_used:
                raise ValueError("Decisive comparison requires grounded PlanFacts.")
            expected_reason = (
                ComparisonReasonCode.NUMERIC_LIMIT_SATISFIED
                if self.decision == ComparisonDecision.COMPLIANT
                else ComparisonReasonCode.NUMERIC_LIMIT_VIOLATED
            )
            if self.reason_code != expected_reason:
                raise ValueError("Decisive comparison reason is inconsistent with decision.")
        elif not self.missing_information:
            raise ValueError(
                "INSUFFICIENT_INFORMATION requires structured missing information."
            )
        elif self.reason_code in {
            ComparisonReasonCode.NUMERIC_LIMIT_SATISFIED,
            ComparisonReasonCode.NUMERIC_LIMIT_VIOLATED,
        }:
            raise ValueError("Insufficient comparison cannot use a decisive reason code.")
        if any(
            fact_id not in self.plan_facts_used for fact_id in self.reason.plan_fact_ids
        ):
            raise ValueError("Comparison reason references an unbound PlanFact.")
        return self


class SourceSpanSelector(BaseModel):
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str

    @model_validator(mode="after")
    def span_matches_text(self):
        if not self.source_text:
            raise ValueError("Selected source text must not be empty.")
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("Selected span does not match source_text.")
        return self


class ComplianceComparisonRequest(ComplianceReviewRequest):
    evidence_id: str
    requirement: SourceSpanSelector
    plan_facts: list[SourceSpanSelector] = Field(default_factory=list)


class ComplianceComparisonResponse(BaseModel):
    """C.1 reconstruction plus an optional authoritative C.2 comparison."""

    preparation: ComplianceReviewResult
    evidence_binding: ReviewEvidenceBinding | None = None
    requirement: NormativeRequirement | None = None
    plan_facts: list[PlanFact] = Field(default_factory=list)
    comparison: ComparisonResult | None = None

    @model_validator(mode="after")
    def validate_complete_provenance_chain(self):
        if self.comparison is None:
            if self.evidence_binding or self.requirement or self.plan_facts:
                raise ValueError("A stopped comparison cannot expose partial C.2 authority.")
            return self
        if not self.evidence_binding or not self.requirement:
            raise ValueError("A comparison requires evidence and requirement authority.")
        unit = self.preparation.review_unit
        binding = self.evidence_binding
        requirement = self.requirement
        result = self.comparison
        if binding not in self.preparation.evidence_bindings:
            raise ValueError("Selected evidence binding is not in C.1 preparation.")
        if result.review_unit_id != unit.review_unit_id:
            raise ValueError("Comparison ReviewUnit identity mismatch.")
        if result.evidence_id != binding.standard_evidence_id:
            raise ValueError("Comparison evidence identity mismatch.")
        if result.requirement_id != requirement.requirement_id:
            raise ValueError("Comparison requirement identity mismatch.")
        evidence = binding.evidence
        if (
            requirement.evidence_id != evidence.id
            or requirement.standard_id != evidence.standard_id
            or requirement.article_id != evidence.article_id
            or requirement.article_number != evidence.article_number
            or requirement.requirement_char_end > len(evidence.source_text)
            or not evidence.source_text.startswith(
                requirement.requirement_text, requirement.requirement_char_start
            )
        ):
            raise ValueError("Requirement provenance does not match EvidenceEnvelope.")
        fact_ids = [fact.plan_fact_id for fact in self.plan_facts]
        if result.plan_facts_used != fact_ids:
            raise ValueError("Comparison PlanFact identities mismatch.")
        if any(fact.review_unit_id != unit.review_unit_id for fact in self.plan_facts):
            raise ValueError("PlanFact ReviewUnit identity mismatch.")
        if any(
            fact.char_end > len(unit.source_text)
            or not unit.source_text.startswith(fact.source_text, fact.char_start)
            for fact in self.plan_facts
        ):
            raise ValueError("PlanFact source does not match grounded ReviewUnit.")
        return self
