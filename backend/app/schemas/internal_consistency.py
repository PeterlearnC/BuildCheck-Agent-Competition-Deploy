"""Deterministic, non-normative contracts for D9 internal consistency review."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


INTERNAL_CONSISTENCY_SCHEMA_VERSION = "v0.1-d9-internal-consistency"
INTERNAL_CONSISTENCY_EXTRACTOR_VERSION = "v0.1-d9-finite-numeric-profiles"
INTERNAL_CONSISTENCY_PROFILE_VERSION = "v0.1-d9-exact-profiles"


class EngineeringObjectProfile(str, Enum):
    SUPPORT_STRUCTURE = "SUPPORT_STRUCTURE"
    RETAINING_WALL = "RETAINING_WALL"
    CROWN_BEAM = "CROWN_BEAM"
    UPRIGHT_MEMBER = "UPRIGHT_MEMBER"
    HORIZONTAL_MEMBER = "HORIZONTAL_MEMBER"
    EXCAVATION = "EXCAVATION"
    STEEL_SUPPORT = "STEEL_SUPPORT"
    ANGLE_WELD = "ANGLE_WELD"


class ConsistencyParameterProfile(str, Enum):
    HORIZONTAL_DISPLACEMENT_ALARM_VALUE = (
        "HORIZONTAL_DISPLACEMENT_ALARM_VALUE"
    )
    HORIZONTAL_DISPLACEMENT_PREWARNING_VALUE = (
        "HORIZONTAL_DISPLACEMENT_PREWARNING_VALUE"
    )
    HORIZONTAL_DISPLACEMENT_CONTROL_VALUE = (
        "HORIZONTAL_DISPLACEMENT_CONTROL_VALUE"
    )
    SPACING = "SPACING"
    STEP_SPACING = "STEP_SPACING"
    THICKNESS = "THICKNESS"
    DEPTH = "DEPTH"
    HEIGHT = "HEIGHT"
    SUPPORT_AXIAL_FORCE = "SUPPORT_AXIAL_FORCE"
    MONITORING_FREQUENCY = "MONITORING_FREQUENCY"


class NormalizedUnit(str, Enum):
    MILLIMETRE = "mm"
    CENTIMETRE = "cm"
    METRE = "m"
    NEWTON = "N"
    KILONEWTON = "kN"
    KILOPASCAL = "kPa"
    MEGAPASCAL = "MPa"
    PERCENT = "%"
    CELSIUS = "℃"


class ConsistencyAssertionClass(str, Enum):
    DECLARED_CONTROL_VALUE = "DECLARED_CONTROL_VALUE"
    EXACT_MEASURED_VALUE = "EXACT_MEASURED_VALUE"
    UPPER_BOUND_CONTROL = "UPPER_BOUND_CONTROL"
    LOWER_BOUND_CONTROL = "LOWER_BOUND_CONTROL"


class ConsistencyCandidateStatus(str, Enum):
    INTERNAL_CONSISTENCY_CANDIDATE = "INTERNAL_CONSISTENCY_CANDIDATE"


class ConsistencyValueRelation(str, Enum):
    DIFFERENT_VALUE = "DIFFERENT_VALUE"


class ConsistencyCandidateReason(str, Enum):
    DIFFERENT_EXPLICIT_NUMERIC_VALUES = "DIFFERENT_EXPLICIT_NUMERIC_VALUES"


class ConsistencyReviewStatus(str, Enum):
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class ConsistencyCoverageStatus(str, Enum):
    CANDIDATES_ESTABLISHED_WITHIN_D9_V1_COVERAGE = (
        "CANDIDATES_ESTABLISHED_WITHIN_D9_V1_COVERAGE"
    )
    NO_INCONSISTENCY_ESTABLISHED_WITHIN_D9_V1_COVERAGE = (
        "NO_INCONSISTENCY_ESTABLISHED_WITHIN_D9_V1_COVERAGE"
    )


class ConsistencyUnresolvedReason(str, Enum):
    OBJECT_IDENTITY_UNRESOLVED = "OBJECT_IDENTITY_UNRESOLVED"
    PARAMETER_IDENTITY_UNRESOLVED = "PARAMETER_IDENTITY_UNRESOLVED"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    MISSING_UNIT = "MISSING_UNIT"
    ASSERTION_TYPE_MISMATCH = "ASSERTION_TYPE_MISMATCH"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


def canonical_decimal(value: Decimal) -> str:
    """Return one non-exponential identity representation for a finite Decimal."""

    if not value.is_finite():
        raise ValueError("Consistency numeric values must be finite.")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def deterministic_profile_id(
    *, profile_kind: Literal["object", "parameter"], profile_name: str, anchors: tuple[str, ...]
) -> str:
    payload = {
        "profile_version": INTERNAL_CONSISTENCY_PROFILE_VERSION,
        "profile_kind": profile_kind,
        "profile_name": profile_name,
        "ordered_anchors": list(anchors),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    prefix = "objectprofile_" if profile_kind == "object" else "parameterprofile_"
    return prefix + hashlib.sha256(canonical).hexdigest()


def deterministic_fact_id(
    *,
    document_sha256: str,
    physical_page: int,
    page_text_sha256: str,
    source_start: int,
    source_end: int,
    source_text_sha256: str,
    object_profile_id: str,
    parameter_profile_id: str,
    numeric_value: Decimal,
    normalized_unit: NormalizedUnit,
    assertion_class: ConsistencyAssertionClass,
) -> str:
    payload = {
        "schema_version": INTERNAL_CONSISTENCY_SCHEMA_VERSION,
        "extractor_version": INTERNAL_CONSISTENCY_EXTRACTOR_VERSION,
        "document_sha256": document_sha256,
        "physical_page": physical_page,
        "page_text_sha256": page_text_sha256,
        "source_start": source_start,
        "source_end": source_end,
        "source_text_sha256": source_text_sha256,
        "object_profile_id": object_profile_id,
        "parameter_profile_id": parameter_profile_id,
        "numeric_value": canonical_decimal(numeric_value),
        "normalized_unit": normalized_unit.value,
        "assertion_class": assertion_class.value,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "consistencyfact_" + hashlib.sha256(canonical).hexdigest()


class ConsistencyFact(BaseModel):
    """One exact source-bound D9 projection; never normative authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[INTERNAL_CONSISTENCY_SCHEMA_VERSION] = (
        INTERNAL_CONSISTENCY_SCHEMA_VERSION
    )
    fact_id: str = Field(pattern=r"^consistencyfact_[0-9a-f]{64}$")
    document_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_page: int = Field(gt=0)
    page_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_profile_id: str = Field(pattern=r"^objectprofile_[0-9a-f]{64}$")
    object_profile_name: EngineeringObjectProfile
    object_anchors: tuple[str, ...] = Field(min_length=1)
    parameter_profile_id: str = Field(
        pattern=r"^parameterprofile_[0-9a-f]{64}$"
    )
    parameter_profile_name: ConsistencyParameterProfile
    parameter_anchors: tuple[str, ...] = Field(min_length=1)
    numeric_value: Decimal
    normalized_unit: NormalizedUnit
    assertion_class: ConsistencyAssertionClass
    extractor_version: Literal[INTERNAL_CONSISTENCY_EXTRACTOR_VERSION] = (
        INTERNAL_CONSISTENCY_EXTRACTOR_VERSION
    )
    original_plan_fact_id: str | None = Field(
        default=None, pattern=r"^planfact_[0-9a-f]{64}$"
    )

    @field_validator("object_anchors", "parameter_anchors")
    @classmethod
    def anchors_are_explicit(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or item != item.strip() for item in value):
            raise ValueError("Profile anchors must be explicit and trimmed.")
        if len(value) != len(set(value)):
            raise ValueError("Profile anchors must be unique and ordered.")
        return value

    @field_validator("numeric_value")
    @classmethod
    def numeric_value_is_finite(cls, value: Decimal) -> Decimal:
        canonical_decimal(value)
        return value

    @model_validator(mode="after")
    def validate_authority(self):
        if self.source_end != self.source_start + len(self.source_text):
            raise ValueError("ConsistencyFact source span does not match exact text.")
        expected_source_hash = hashlib.sha256(
            self.source_text.encode("utf-8")
        ).hexdigest()
        if self.source_text_sha256 != expected_source_hash:
            raise ValueError("ConsistencyFact source SHA-256 does not match text.")
        expected_object_profile = deterministic_profile_id(
            profile_kind="object",
            profile_name=self.object_profile_name.value,
            anchors=self.object_anchors,
        )
        expected_parameter_profile = deterministic_profile_id(
            profile_kind="parameter",
            profile_name=self.parameter_profile_name.value,
            anchors=self.parameter_anchors,
        )
        if self.object_profile_id != expected_object_profile:
            raise ValueError("Object profile identity is not canonical.")
        if self.parameter_profile_id != expected_parameter_profile:
            raise ValueError("Parameter profile identity is not canonical.")
        expected_fact_id = deterministic_fact_id(
            document_sha256=self.document_sha256,
            physical_page=self.physical_page,
            page_text_sha256=self.page_text_sha256,
            source_start=self.source_start,
            source_end=self.source_end,
            source_text_sha256=self.source_text_sha256,
            object_profile_id=self.object_profile_id,
            parameter_profile_id=self.parameter_profile_id,
            numeric_value=self.numeric_value,
            normalized_unit=self.normalized_unit,
            assertion_class=self.assertion_class,
        )
        if self.fact_id != expected_fact_id:
            raise ValueError("fact_id does not match canonical source authority.")
        return self


def fact_source_sort_key(fact: ConsistencyFact) -> tuple[int, int, int, str]:
    return (fact.physical_page, fact.source_start, fact.source_end, fact.fact_id)


class ConsistencyValueGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    numeric_value: Decimal
    normalized_unit: NormalizedUnit
    fact_ids: tuple[str, ...] = Field(min_length=1)
    facts: tuple[ConsistencyFact, ...] = Field(min_length=1)

    @field_validator("numeric_value")
    @classmethod
    def numeric_value_is_finite(cls, value: Decimal) -> Decimal:
        canonical_decimal(value)
        return value

    @model_validator(mode="after")
    def validate_group(self):
        expected_facts = tuple(sorted(self.facts, key=fact_source_sort_key))
        if self.facts != expected_facts:
            raise ValueError("Value-group facts must have deterministic source order.")
        if len({fact.fact_id for fact in self.facts}) != len(self.facts):
            raise ValueError("Value-group facts must be unique.")
        if self.fact_ids != tuple(fact.fact_id for fact in self.facts):
            raise ValueError("Value-group fact IDs do not match exact facts.")
        if any(
            fact.numeric_value != self.numeric_value
            or fact.normalized_unit != self.normalized_unit
            for fact in self.facts
        ):
            raise ValueError("Value-group facts must share value and normalized unit.")
        return self


def deterministic_candidate_id(
    *,
    document_sha256: str,
    object_profile_id: str,
    parameter_profile_id: str,
    normalized_unit: NormalizedUnit,
    assertion_class: ConsistencyAssertionClass,
    value_groups: tuple[ConsistencyValueGroup, ...],
) -> str:
    payload = {
        "schema_version": INTERNAL_CONSISTENCY_SCHEMA_VERSION,
        "document_sha256": document_sha256,
        "object_profile_id": object_profile_id,
        "parameter_profile_id": parameter_profile_id,
        "normalized_unit": normalized_unit.value,
        "assertion_class": assertion_class.value,
        "ordered_value_groups": [
            {
                "numeric_value": canonical_decimal(group.numeric_value),
                "ordered_fact_ids": list(group.fact_ids),
            }
            for group in value_groups
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "consistencycandidate_" + hashlib.sha256(canonical).hexdigest()


class InternalConsistencyCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[INTERNAL_CONSISTENCY_SCHEMA_VERSION] = (
        INTERNAL_CONSISTENCY_SCHEMA_VERSION
    )
    candidate_id: str = Field(pattern=r"^consistencycandidate_[0-9a-f]{64}$")
    document_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_profile_id: str = Field(pattern=r"^objectprofile_[0-9a-f]{64}$")
    object_profile_name: EngineeringObjectProfile
    object_anchors: tuple[str, ...] = Field(min_length=1)
    parameter_profile_id: str = Field(
        pattern=r"^parameterprofile_[0-9a-f]{64}$"
    )
    parameter_profile_name: ConsistencyParameterProfile
    parameter_anchors: tuple[str, ...] = Field(min_length=1)
    normalized_unit: NormalizedUnit
    assertion_class: ConsistencyAssertionClass
    value_groups: tuple[ConsistencyValueGroup, ...] = Field(min_length=2)
    status: Literal[
        ConsistencyCandidateStatus.INTERNAL_CONSISTENCY_CANDIDATE
    ] = ConsistencyCandidateStatus.INTERNAL_CONSISTENCY_CANDIDATE
    relation: Literal[ConsistencyValueRelation.DIFFERENT_VALUE] = (
        ConsistencyValueRelation.DIFFERENT_VALUE
    )
    reason: Literal[
        ConsistencyCandidateReason.DIFFERENT_EXPLICIT_NUMERIC_VALUES
    ] = ConsistencyCandidateReason.DIFFERENT_EXPLICIT_NUMERIC_VALUES
    review_status: Literal[ConsistencyReviewStatus.NEEDS_HUMAN_REVIEW] = (
        ConsistencyReviewStatus.NEEDS_HUMAN_REVIEW
    )

    @model_validator(mode="after")
    def validate_candidate(self):
        expected_groups = tuple(
            sorted(
                self.value_groups,
                key=lambda group: (
                    group.numeric_value,
                    group.normalized_unit.value,
                    group.fact_ids,
                ),
            )
        )
        if self.value_groups != expected_groups:
            raise ValueError("Candidate value groups must be deterministically ordered.")
        if len({group.numeric_value for group in self.value_groups}) != len(
            self.value_groups
        ):
            raise ValueError("Candidate value groups must have distinct values.")
        for group in self.value_groups:
            for fact in group.facts:
                if (
                    fact.document_id != self.document_id
                    or fact.document_sha256 != self.document_sha256
                    or fact.object_profile_id != self.object_profile_id
                    or fact.object_profile_name != self.object_profile_name
                    or fact.object_anchors != self.object_anchors
                    or fact.parameter_profile_id != self.parameter_profile_id
                    or fact.parameter_profile_name != self.parameter_profile_name
                    or fact.parameter_anchors != self.parameter_anchors
                    or fact.normalized_unit != self.normalized_unit
                    or fact.assertion_class != self.assertion_class
                ):
                    raise ValueError("Candidate facts do not share exact D9 authority.")
        expected_id = deterministic_candidate_id(
            document_sha256=self.document_sha256,
            object_profile_id=self.object_profile_id,
            parameter_profile_id=self.parameter_profile_id,
            normalized_unit=self.normalized_unit,
            assertion_class=self.assertion_class,
            value_groups=self.value_groups,
        )
        if self.candidate_id != expected_id:
            raise ValueError("candidate_id does not match canonical grouped authority.")
        return self


class ConsistencyUnresolvedCount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: ConsistencyUnresolvedReason
    count: int = Field(gt=0)


def deterministic_review_id(
    *,
    document_sha256: str,
    source_page_count: int,
    fact_count: int,
    candidates: tuple[InternalConsistencyCandidate, ...],
    unresolved_counts: tuple[ConsistencyUnresolvedCount, ...],
) -> str:
    payload = {
        "schema_version": INTERNAL_CONSISTENCY_SCHEMA_VERSION,
        "extractor_version": INTERNAL_CONSISTENCY_EXTRACTOR_VERSION,
        "document_sha256": document_sha256,
        "source_page_count": source_page_count,
        "fact_count": fact_count,
        "ordered_candidate_ids": [item.candidate_id for item in candidates],
        "ordered_unresolved_counts": [
            {"reason": item.reason.value, "count": item.count}
            for item in unresolved_counts
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "consistencyreview_" + hashlib.sha256(canonical).hexdigest()


class InternalConsistencyReviewResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[INTERNAL_CONSISTENCY_SCHEMA_VERSION] = (
        INTERNAL_CONSISTENCY_SCHEMA_VERSION
    )
    extractor_version: Literal[INTERNAL_CONSISTENCY_EXTRACTOR_VERSION] = (
        INTERNAL_CONSISTENCY_EXTRACTOR_VERSION
    )
    review_id: str = Field(pattern=r"^consistencyreview_[0-9a-f]{64}$")
    document_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_page_count: int = Field(gt=0)
    fact_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    coverage_status: ConsistencyCoverageStatus
    candidates: tuple[InternalConsistencyCandidate, ...] = ()
    unresolved_counts: tuple[ConsistencyUnresolvedCount, ...] = ()

    @model_validator(mode="after")
    def validate_result(self):
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count does not match candidates.")
        if self.unresolved_count != sum(item.count for item in self.unresolved_counts):
            raise ValueError("unresolved_count does not match finite reason counts.")
        if self.fact_count < len(
            {
                fact.fact_id
                for candidate in self.candidates
                for group in candidate.value_groups
                for fact in group.facts
            }
        ):
            raise ValueError("fact_count cannot omit candidate source facts.")
        expected_candidates = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        if self.candidates != expected_candidates:
            raise ValueError("Candidates must have deterministic identity order.")
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate identities must be unique.")
        if any(
            item.document_id != self.document_id
            or item.document_sha256 != self.document_sha256
            for item in self.candidates
        ):
            raise ValueError("Candidate document authority differs from review authority.")
        expected_unresolved = tuple(
            sorted(self.unresolved_counts, key=lambda item: item.reason.value)
        )
        if self.unresolved_counts != expected_unresolved:
            raise ValueError("Unresolved counts must have deterministic reason order.")
        if len({item.reason for item in self.unresolved_counts}) != len(
            self.unresolved_counts
        ):
            raise ValueError("Unresolved reason counts must be unique.")
        expected_status = (
            ConsistencyCoverageStatus.CANDIDATES_ESTABLISHED_WITHIN_D9_V1_COVERAGE
            if self.candidates
            else ConsistencyCoverageStatus.NO_INCONSISTENCY_ESTABLISHED_WITHIN_D9_V1_COVERAGE
        )
        if self.coverage_status != expected_status:
            raise ValueError("Coverage status cannot widen D9 V1 evidence.")
        expected_review_id = deterministic_review_id(
            document_sha256=self.document_sha256,
            source_page_count=self.source_page_count,
            fact_count=self.fact_count,
            candidates=self.candidates,
            unresolved_counts=self.unresolved_counts,
        )
        if self.review_id != expected_review_id:
            raise ValueError("review_id does not match deterministic D9 authority.")
        return self
