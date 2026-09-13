"""Source-grounded, comparison-free PlanFact extraction contracts for D.4."""

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_comparison import (
    PlanFact,
    PlanFactParsingStatus,
    PlanFactType,
)


PLAN_FACT_EXTRACTION_IDENTITY_VERSION = "v0.1-d.4-plan-fact-extraction"
PLAN_FACT_EXTRACTION_VERSION = "v0.1-d.4-conservative-exact-candidate"


class PlanFactExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    UNSUPPORTED = "UNSUPPORTED"
    SOURCE_NOT_QUALIFIED = "SOURCE_NOT_QUALIFIED"


class PlanAssertionKind(str, Enum):
    EXACT_VALUE = "EXACT_VALUE"
    UPPER_BOUND_CONTROL = "UPPER_BOUND_CONTROL"
    LOWER_BOUND_CONTROL = "LOWER_BOUND_CONTROL"
    DESIGN_CONTROL = "DESIGN_CONTROL"


class PlanFactExtractionReason(str, Enum):
    SOURCE_AUTHORITY_FAILED = "SOURCE_AUTHORITY_FAILED"
    REQUIREMENT_NOT_PROVEN = "REQUIREMENT_NOT_PROVEN"
    NO_NUMERIC_FACT = "NO_NUMERIC_FACT"
    OBJECT_MISMATCH = "OBJECT_MISMATCH"
    MULTIPLE_VALUES = "MULTIPLE_VALUES"
    MULTIPLE_MATCHING_FACTS = "MULTIPLE_MATCHING_FACTS"
    CONDITIONAL_UNSUPPORTED = "CONDITIONAL_UNSUPPORTED"
    LOCATION_QUALIFIER_UNSUPPORTED = "LOCATION_QUALIFIER_UNSUPPORTED"
    PROHIBITED_VALUE_NOT_FACT = "PROHIBITED_VALUE_NOT_FACT"
    RANGE_UNSUPPORTED = "RANGE_UNSUPPORTED"
    COMPOUND_DIMENSION_UNSUPPORTED = "COMPOUND_DIMENSION_UNSUPPORTED"
    ASSERTION_INCOMPATIBLE = "ASSERTION_INCOMPATIBLE"
    ASSERTION_UNSUPPORTED = "ASSERTION_UNSUPPORTED"


def deterministic_measurement_profile_id(
    *, profile_name: str, measurement_anchors: tuple[str, ...], object_anchors: tuple[str, ...]
) -> str:
    payload = {
        "profile_name": profile_name,
        "measurement_anchors": list(measurement_anchors),
        "object_anchors": list(object_anchors),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "measurementprofile_" + hashlib.sha256(canonical).hexdigest()


class ExtractedPlanFactBinding(BaseModel):
    """One exact candidate subspan accepted by frozen PlanFact authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_fact: PlanFact
    review_unit_id: str = Field(pattern=r"^reviewunit_[0-9a-f]{64}$")
    routing_context_id: str = Field(pattern=r"^routingcontext_[0-9a-f]{64}$")
    source_page_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_unit_char_start: int = Field(ge=0)
    review_unit_char_end: int = Field(gt=0)
    page_char_start: int = Field(ge=0)
    page_char_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assertion_kind: PlanAssertionKind
    measurement_profile_id: str = Field(
        pattern=r"^measurementprofile_[0-9a-f]{64}$"
    )
    measurement_profile_name: str = Field(min_length=1)
    measurement_anchors: tuple[str, ...] = Field(min_length=1)
    object_anchors: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binding(self):
        if self.review_unit_char_end != self.review_unit_char_start + len(
            self.source_text
        ):
            raise ValueError("ReviewUnit-relative extraction span does not match text.")
        if self.page_char_end != self.page_char_start + len(self.source_text):
            raise ValueError("Page-relative extraction span does not match text.")
        expected_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected_hash:
            raise ValueError("Extraction source SHA-256 does not match exact text.")
        fact = self.plan_fact
        if (
            fact.review_unit_id != self.review_unit_id
            or fact.char_start != self.review_unit_char_start
            or fact.char_end != self.review_unit_char_end
            or fact.source_text != self.source_text
            or fact.source_text_sha256 != self.source_text_sha256
            or fact.fact_type != PlanFactType.NUMERIC
            or fact.parsing_status != PlanFactParsingStatus.PROVEN
        ):
            raise ValueError("PlanFact does not equal the exact verified extraction span.")
        expected_profile = deterministic_measurement_profile_id(
            profile_name=self.measurement_profile_name,
            measurement_anchors=self.measurement_anchors,
            object_anchors=self.object_anchors,
        )
        if self.measurement_profile_id != expected_profile:
            raise ValueError("Measurement profile identity is not canonical.")
        return self


def deterministic_extraction_id(
    *,
    candidate_id: str,
    document_id: str,
    document_sha256: str,
    physical_page: int,
    review_unit_id: str,
    route_id: str,
    decomposition_id: str,
    requirement_id: str,
    status: PlanFactExtractionStatus,
    binding: ExtractedPlanFactBinding | None,
    reason: PlanFactExtractionReason | None,
    identity_version: str = PLAN_FACT_EXTRACTION_IDENTITY_VERSION,
    extraction_version: str = PLAN_FACT_EXTRACTION_VERSION,
) -> str:
    payload = {
        "identity_version": identity_version,
        "candidate_id": candidate_id,
        "document_id": document_id,
        "document_sha256": document_sha256,
        "physical_page": physical_page,
        "review_unit_id": review_unit_id,
        "route_id": route_id,
        "decomposition_id": decomposition_id,
        "requirement_id": requirement_id,
        "status": status.value,
        "plan_fact_id": binding.plan_fact.plan_fact_id if binding else None,
        "source_text_sha256": binding.source_text_sha256 if binding else None,
        "page_char_start": binding.page_char_start if binding else None,
        "page_char_end": binding.page_char_end if binding else None,
        "assertion_kind": binding.assertion_kind.value if binding else None,
        "measurement_profile_id": binding.measurement_profile_id if binding else None,
        "reason": reason.value if reason else None,
        "extraction_version": extraction_version,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "planfactextraction_" + hashlib.sha256(canonical).hexdigest()


class PlanFactExtractionResult(BaseModel):
    """D.4 output: one verified plan fact or a finite fail-closed status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[PLAN_FACT_EXTRACTION_IDENTITY_VERSION] = (
        PLAN_FACT_EXTRACTION_IDENTITY_VERSION
    )
    extraction_id: str = Field(pattern=r"^planfactextraction_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_page: int = Field(gt=0)
    review_unit_id: str = Field(pattern=r"^reviewunit_[0-9a-f]{64}$")
    route_id: str = Field(pattern=r"^route_[0-9a-f]{64}$")
    decomposition_id: str = Field(pattern=r"^decomposition_[0-9a-f]{64}$")
    requirement_id: str = Field(min_length=1)
    status: PlanFactExtractionStatus
    binding: ExtractedPlanFactBinding | None = None
    reason: PlanFactExtractionReason | None = None
    extraction_method: Literal["EXACT_CANDIDATE_DETERMINISTIC"] = (
        "EXACT_CANDIDATE_DETERMINISTIC"
    )
    extraction_version: Literal[PLAN_FACT_EXTRACTION_VERSION] = (
        PLAN_FACT_EXTRACTION_VERSION
    )

    @model_validator(mode="after")
    def validate_result(self):
        if self.status == PlanFactExtractionStatus.EXTRACTED:
            if self.binding is None or self.reason is not None:
                raise ValueError("EXTRACTED requires exactly one binding and no reason.")
            if self.binding.review_unit_id != self.review_unit_id:
                raise ValueError("Extraction and PlanFact ReviewUnit identities differ.")
        elif self.binding is not None or self.reason is None:
            raise ValueError("Fail-closed extraction requires no PlanFact and one reason.")
        expected = deterministic_extraction_id(
            candidate_id=self.candidate_id,
            document_id=self.document_id,
            document_sha256=self.document_sha256,
            physical_page=self.physical_page,
            review_unit_id=self.review_unit_id,
            route_id=self.route_id,
            decomposition_id=self.decomposition_id,
            requirement_id=self.requirement_id,
            status=self.status,
            binding=self.binding,
            reason=self.reason,
            identity_version=self.identity_version,
            extraction_version=self.extraction_version,
        )
        if self.extraction_id != expected:
            raise ValueError("extraction_id does not match canonical authority identity.")
        return self
