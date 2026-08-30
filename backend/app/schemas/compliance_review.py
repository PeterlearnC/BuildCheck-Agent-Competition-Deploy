"""Fail-closed contracts for the C.1 compliance-review foundation."""

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.evidence import EvidenceEnvelope
from app.schemas.review_unit import ReviewUnit
from app.schemas.standards_retrieval import RetrievalDecision


class ComplianceReviewStatus(str, Enum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEEDS_COMPARISON = "NEEDS_COMPARISON"
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"


class ComplianceReviewRequest(BaseModel):
    page_number: int = Field(gt=0)
    source_text: str
    char_start: int | None = Field(default=None, ge=0)
    retrieval_query: str
    standard_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("source_text", "retrieval_query")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Compliance review text fields must not be blank.")
        return value

    @field_validator("standard_ids")
    @classmethod
    def normalize_standard_ids(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not normalized:
            raise ValueError("Compliance review requires at least one standard_id.")
        return normalized


class ReviewEvidenceBinding(BaseModel):
    """A dual-sided binding without rewriting the immutable standard evidence."""

    model_config = ConfigDict(frozen=True)

    review_unit_id: str
    document_id: str
    plan_page_number: int = Field(gt=0)
    plan_char_start: int = Field(ge=0)
    plan_char_end: int = Field(gt=0)
    plan_source_text: str
    plan_source_text_sha256: str
    retrieval_decision: RetrievalDecision
    standard_evidence_id: str
    standard_id: str
    standard_article_number: str
    standard_page_start: int = Field(gt=0)
    standard_page_end: int = Field(gt=0)
    evidence: EvidenceEnvelope

    @model_validator(mode="after")
    def binding_is_consistent(self):
        if self.retrieval_decision != RetrievalDecision.ACCEPT:
            raise ValueError("Only ACCEPT-qualified evidence may be bound.")
        if self.plan_char_end != self.plan_char_start + len(self.plan_source_text):
            raise ValueError("Bound plan source span does not match its text.")
        source_hash = hashlib.sha256(self.plan_source_text.encode("utf-8")).hexdigest()
        if self.plan_source_text_sha256 != source_hash:
            raise ValueError("Bound plan source hash does not match its text.")
        if self.standard_evidence_id != self.evidence.id:
            raise ValueError("standard_evidence_id does not match EvidenceEnvelope.id.")
        if self.standard_id != self.evidence.standard_id:
            raise ValueError("Bound standard_id does not match EvidenceEnvelope.")
        if self.standard_article_number != self.evidence.article_number:
            raise ValueError("Bound article number does not match EvidenceEnvelope.")
        if self.standard_page_start != self.evidence.source_page_start:
            raise ValueError("Bound standard start page does not match EvidenceEnvelope.")
        if self.standard_page_end != self.evidence.source_page_end:
            raise ValueError("Bound standard end page does not match EvidenceEnvelope.")
        return self


class ComplianceReviewResult(BaseModel):
    """C.1 result: evidence readiness only, never a final compliance judgment."""

    review_unit: ReviewUnit
    status: ComplianceReviewStatus
    retrieval_decision: RetrievalDecision
    evidence_bindings: list[ReviewEvidenceBinding] = Field(default_factory=list)
    reason: str
    scope_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_c1_safety_contract(self):
        if self.status in {
            ComplianceReviewStatus.COMPLIANT,
            ComplianceReviewStatus.NON_COMPLIANT,
        }:
            raise ValueError("C.1 cannot emit a final compliance judgment.")

        if self.retrieval_decision == RetrievalDecision.ACCEPT:
            if self.status != ComplianceReviewStatus.NEEDS_COMPARISON:
                raise ValueError("ACCEPT evidence requires NEEDS_COMPARISON status.")
            if not self.evidence_bindings:
                raise ValueError("NEEDS_COMPARISON requires ACCEPT evidence bindings.")
        else:
            if self.status != ComplianceReviewStatus.INSUFFICIENT_EVIDENCE:
                raise ValueError("Non-ACCEPT retrieval requires INSUFFICIENT_EVIDENCE.")
            if self.evidence_bindings:
                raise ValueError("Non-ACCEPT retrieval cannot expose evidence bindings.")

        for binding in self.evidence_bindings:
            if binding.review_unit_id != self.review_unit.review_unit_id:
                raise ValueError("Evidence binding review_unit_id does not match result unit.")
            if binding.document_id != self.review_unit.document_id:
                raise ValueError("Evidence binding document_id does not match result unit.")
            if binding.plan_page_number != self.review_unit.page_number:
                raise ValueError("Evidence binding plan page does not match result unit.")
            if (
                binding.plan_char_start != self.review_unit.char_start
                or binding.plan_char_end != self.review_unit.char_end
                or binding.plan_source_text != self.review_unit.source_text
                or binding.plan_source_text_sha256
                != self.review_unit.source_text_sha256
            ):
                raise ValueError("Evidence binding plan provenance does not match result unit.")
        return self
