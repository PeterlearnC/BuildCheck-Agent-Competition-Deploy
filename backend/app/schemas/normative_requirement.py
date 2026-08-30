"""Exact-source normative requirements for deterministic local comparison."""

import hashlib
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RequirementModality(str, Enum):
    REQUIRED = "REQUIRED"
    PROHIBITED = "PROHIBITED"
    NUMERIC_LIMIT = "NUMERIC_LIMIT"
    SEQUENCE = "SEQUENCE"
    CONDITIONAL = "CONDITIONAL"
    UNSUPPORTED = "UNSUPPORTED"


class RequirementAtomicity(str, Enum):
    PROVEN = "PROVEN"
    UNRESOLVED = "UNRESOLVED"


class NumericOperator(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "="


class RequirementExtractionMethod(str, Enum):
    EXACT_SPAN_DETERMINISTIC = "EXACT_SPAN_DETERMINISTIC"


class NormativeRequirement(BaseModel):
    """An immutable exact evidence subspan; never a rewritten requirement."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    evidence_id: str
    standard_id: str
    article_id: str
    article_number: str
    requirement_char_start: int = Field(ge=0)
    requirement_char_end: int = Field(gt=0)
    requirement_text: str
    requirement_text_sha256: str
    modality: RequirementModality
    atomicity: RequirementAtomicity
    condition_text: str | None = None
    subject: str | None = None
    action: str | None = None
    operator: NumericOperator | None = None
    value: Decimal | None = None
    unit: str | None = None
    exception_text: str | None = None
    extraction_method: RequirementExtractionMethod
    extraction_version: str

    @model_validator(mode="after")
    def validate_source_identity_and_structure(self):
        if not self.requirement_text:
            raise ValueError("Requirement text must not be empty.")
        if self.requirement_char_end != (
            self.requirement_char_start + len(self.requirement_text)
        ):
            raise ValueError("Requirement span does not match requirement_text.")
        expected_hash = hashlib.sha256(
            self.requirement_text.encode("utf-8")
        ).hexdigest()
        if self.requirement_text_sha256 != expected_hash:
            raise ValueError("Requirement text hash does not match requirement_text.")
        numeric_fields = (self.operator, self.value, self.unit)
        if self.modality == RequirementModality.NUMERIC_LIMIT:
            if self.atomicity != RequirementAtomicity.PROVEN:
                raise ValueError("A numeric limit must be an atomic requirement.")
            if any(value is None for value in numeric_fields):
                raise ValueError("A numeric limit requires operator, value, and unit.")
        elif any(value is not None for value in numeric_fields):
            raise ValueError("Only numeric limits may carry numeric constraint fields.")
        return self
