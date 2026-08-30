"""Immutable source-grounded presentation contracts for C.3 review findings."""

import hashlib
import json
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_comparison import (
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonReasonCode,
    ComplianceComparisonResponse,
    MissingInformationItem,
)


FINDING_VERSION = "v0.1-c.3-f1-source-grounded"


class PlanFactCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_fact_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str
    source_text_sha256: str
    normalized_value: Decimal | bool | str | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_source_identity(self):
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("PlanFact citation span does not match source_text.")
        expected = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected:
            raise ValueError("PlanFact citation hash does not match source_text.")
        return self


class PlanCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(gt=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str
    source_text_sha256: str
    review_unit_id: str
    plan_facts: list[PlanFactCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_grounded_spans(self):
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("Plan citation span does not match source_text.")
        expected = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected:
            raise ValueError("Plan citation hash does not match source_text.")
        for fact in self.plan_facts:
            if fact.char_end > len(self.source_text) or not self.source_text.startswith(
                fact.source_text, fact.char_start
            ):
                raise ValueError("PlanFact citation lies outside the ReviewUnit source.")
        return self


class StandardCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    standard_id: str
    standard_code: str | None = None
    canonical_standard_code: str | None = None
    article_id: str
    article_number: str
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    evidence_id: str
    source_checksum: str
    requirement_id: str
    requirement_char_start: int = Field(ge=0)
    requirement_char_end: int = Field(gt=0)
    requirement_text: str
    requirement_text_sha256: str

    @model_validator(mode="after")
    def validate_requirement_identity(self):
        if self.page_end < self.page_start:
            raise ValueError("Standard citation page interval is invalid.")
        if self.requirement_char_end != (
            self.requirement_char_start + len(self.requirement_text)
        ):
            raise ValueError("Requirement citation span does not match requirement_text.")
        expected = hashlib.sha256(self.requirement_text.encode("utf-8")).hexdigest()
        if self.requirement_text_sha256 != expected:
            raise ValueError("Requirement citation hash does not match requirement_text.")
        return self


def deterministic_finding_summary(authority: ComplianceComparisonResponse) -> str:
    """Render only exact authoritative source and typed C.2 outcome fields."""
    result = authority.comparison
    requirement = authority.requirement
    if result is None or requirement is None:
        raise ValueError("A deterministic finding summary requires C.2 authority.")
    unit = authority.preparation.review_unit
    summary = (
        f'Plan evidence states: "{unit.source_text}". '
        f'Standard requirement states: "{requirement.requirement_text}". '
        f"Local {result.decision_scope.value} decision: {result.decision.value}. "
        f"Reason: {result.reason_code.value}."
    )
    if result.missing_information:
        details = "; ".join(item.detail for item in result.missing_information)
        summary += f" Missing or unresolved information: {details}"
    return summary


def deterministic_finding_id(
    *,
    comparison_id: str,
    decision: ComparisonDecision,
    reason_code: ComparisonReasonCode,
    decision_scope: ComparisonDecisionScope,
    document_sha256: str,
) -> str:
    payload = {
        "finding_version": FINDING_VERSION,
        "comparison_id": comparison_id,
        "decision": decision.value,
        "reason_code": reason_code.value,
        "decision_scope": decision_scope.value,
        "document_sha256": document_sha256,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "finding_" + hashlib.sha256(serialized).hexdigest()


class ReviewFinding(BaseModel):
    """A one-to-one, non-decisional projection of authoritative C.2 output."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    finding_version: Literal[FINDING_VERSION] = FINDING_VERSION
    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison_id: str
    decision: ComparisonDecision
    decision_scope: ComparisonDecisionScope
    reason_code: ComparisonReasonCode
    review_unit_id: str
    evidence_id: str
    requirement_id: str
    plan_fact_ids: list[str] = Field(default_factory=list)
    plan_citation: PlanCitation
    standard_citation: StandardCitation
    summary: str
    missing_information: list[MissingInformationItem] = Field(default_factory=list)
    authoritative_comparison_response: ComplianceComparisonResponse

    @model_validator(mode="after")
    def preserve_complete_c2_authority(self):
        authority = self.authoritative_comparison_response
        result = authority.comparison
        binding = authority.evidence_binding
        requirement = authority.requirement
        if result is None or binding is None or requirement is None:
            raise ValueError("ReviewFinding requires complete authoritative C.2 output.")
        unit = authority.preparation.review_unit
        evidence = binding.evidence

        if self.document_id != unit.document_id:
            raise ValueError("Finding document identity differs from ReviewUnit.")
        if (
            self.comparison_id != result.comparison_id
            or self.decision != result.decision
            or self.reason_code != result.reason_code
            or self.decision_scope != result.decision_scope
        ):
            raise ValueError("Finding decision authority differs from C.2 result.")
        if self.decision_scope != ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT:
            raise ValueError("Finding decision scope may not be widened.")
        if (
            self.review_unit_id != unit.review_unit_id
            or self.evidence_id != evidence.id
            or self.requirement_id != requirement.requirement_id
            or self.plan_fact_ids != result.plan_facts_used
            or self.missing_information != result.missing_information
        ):
            raise ValueError("Finding provenance identities differ from C.2 authority.")

        plan = self.plan_citation
        if (
            plan.document_id != unit.document_id
            or plan.document_sha256 != self.document_sha256
            or plan.page_number != unit.page_number
            or plan.char_start != unit.char_start
            or plan.char_end != unit.char_end
            or plan.source_text != unit.source_text
            or plan.source_text_sha256 != unit.source_text_sha256
            or plan.review_unit_id != unit.review_unit_id
        ):
            raise ValueError("Plan citation differs from grounded ReviewUnit authority.")
        expected_fact_citations = [
            PlanFactCitation(
                plan_fact_id=fact.plan_fact_id,
                char_start=fact.char_start,
                char_end=fact.char_end,
                source_text=fact.source_text,
                source_text_sha256=fact.source_text_sha256,
                normalized_value=fact.normalized_value,
                unit=fact.unit,
            )
            for fact in authority.plan_facts
        ]
        if plan.plan_facts != expected_fact_citations:
            raise ValueError("PlanFact citations differ from C.2 authority.")

        standard = self.standard_citation
        if (
            standard.standard_id != evidence.standard_id
            or standard.standard_code != evidence.standard_code
            or standard.canonical_standard_code != evidence.canonical_standard_code
            or standard.article_id != evidence.article_id
            or standard.article_number != evidence.article_number
            or standard.page_start != evidence.source_page_start
            or standard.page_end != evidence.source_page_end
            or standard.evidence_id != evidence.id
            or standard.source_checksum != evidence.source_checksum
            or standard.requirement_id != requirement.requirement_id
            or standard.requirement_char_start != requirement.requirement_char_start
            or standard.requirement_char_end != requirement.requirement_char_end
            or standard.requirement_text != requirement.requirement_text
            or standard.requirement_text_sha256 != requirement.requirement_text_sha256
        ):
            raise ValueError("Standard citation differs from C.2 evidence authority.")
        if requirement.requirement_char_end > len(evidence.source_text) or not (
            evidence.source_text.startswith(
                requirement.requirement_text, requirement.requirement_char_start
            )
        ):
            raise ValueError("Requirement citation lies outside EvidenceEnvelope.")

        expected_id = deterministic_finding_id(
            comparison_id=result.comparison_id,
            decision=result.decision,
            reason_code=result.reason_code,
            decision_scope=result.decision_scope,
            document_sha256=self.document_sha256,
        )
        if self.finding_id != expected_id:
            raise ValueError("Finding identity does not match authoritative inputs.")
        if self.summary != deterministic_finding_summary(authority):
            raise ValueError("Finding summary is not the deterministic C.2 projection.")
        return self


class ReviewFindingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    finding: ReviewFinding
