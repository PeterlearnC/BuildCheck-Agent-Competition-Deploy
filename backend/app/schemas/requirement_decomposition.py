"""Source-grounded requirement-decomposition contracts for D.3."""

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.normative_requirement import (
    NormativeRequirement,
    RequirementAtomicity,
    RequirementModality,
)


DECOMPOSITION_IDENTITY_VERSION = "v0.1-d.3-requirement-decomposition"
DECOMPOSITION_VERSION = "v0.1-d.3-conservative-exact-spans"
UNRESOLVED_SPAN_IDENTITY_VERSION = "v0.1-d.3-unresolved-span"


class RequirementDecompositionStatus(str, Enum):
    DECOMPOSED = "DECOMPOSED"
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    SOURCE_NOT_QUALIFIED = "SOURCE_NOT_QUALIFIED"


class RequirementDecompositionMethod(str, Enum):
    EXACT_SOURCE_DETERMINISTIC = "EXACT_SOURCE_DETERMINISTIC"


class UnresolvedRequirementReason(str, Enum):
    QUALITATIVE_UNSUPPORTED = "QUALITATIVE_UNSUPPORTED"
    CONDITIONAL_UNSUPPORTED = "CONDITIONAL_UNSUPPORTED"
    COMPOUND_STRUCTURE_UNRESOLVED = "COMPOUND_STRUCTURE_UNRESOLVED"
    EXCEPTION_SCOPE_UNRESOLVED = "EXCEPTION_SCOPE_UNRESOLVED"
    AMBIGUOUS_SPLIT = "AMBIGUOUS_SPLIT"
    UNSUPPORTED_NUMERIC_FORM = "UNSUPPORTED_NUMERIC_FORM"


def deterministic_unresolved_span_id(
    *,
    evidence_id: str,
    article_id: str,
    article_number: str,
    char_start: int,
    char_end: int,
    source_text_sha256: str,
    reason: UnresolvedRequirementReason,
) -> str:
    payload = {
        "identity_version": UNRESOLVED_SPAN_IDENTITY_VERSION,
        "evidence_id": evidence_id,
        "article_id": article_id,
        "article_number": article_number,
        "char_start": char_start,
        "char_end": char_end,
        "source_text_sha256": source_text_sha256,
        "reason": reason.value,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "unresolvedspan_" + hashlib.sha256(canonical).hexdigest()


class UnresolvedRequirementSpan(BaseModel):
    """An exact qualified article span that D.3 refuses to overgeneralize."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unresolved_span_id: str = Field(pattern=r"^unresolvedspan_[0-9a-f]{64}$")
    evidence_id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    article_number: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: UnresolvedRequirementReason

    @model_validator(mode="after")
    def validate_exact_span_identity(self):
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("Unresolved span offsets do not match source_text.")
        text_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != text_hash:
            raise ValueError("Unresolved span SHA-256 does not match source_text.")
        expected = deterministic_unresolved_span_id(
            evidence_id=self.evidence_id,
            article_id=self.article_id,
            article_number=self.article_number,
            char_start=self.char_start,
            char_end=self.char_end,
            source_text_sha256=self.source_text_sha256,
            reason=self.reason,
        )
        if self.unresolved_span_id != expected:
            raise ValueError("unresolved_span_id does not match canonical identity.")
        return self


def deterministic_decomposition_id(
    *,
    candidate_id: str,
    route_id: str,
    evidence_ids: tuple[str, ...] | list[str],
    requirements: tuple[NormativeRequirement, ...] | list[NormativeRequirement],
    unresolved_spans: tuple[UnresolvedRequirementSpan, ...]
    | list[UnresolvedRequirementSpan],
    identity_version: str = DECOMPOSITION_IDENTITY_VERSION,
    decomposition_version: str = DECOMPOSITION_VERSION,
) -> str:
    payload = {
        "identity_version": identity_version,
        "candidate_id": candidate_id,
        "route_id": route_id,
        "ordered_evidence_ids": list(evidence_ids),
        "ordered_requirement_ids": [item.requirement_id for item in requirements],
        "ordered_unresolved_span_ids": [
            item.unresolved_span_id for item in unresolved_spans
        ],
        "decomposition_version": decomposition_version,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "decomposition_" + hashlib.sha256(canonical).hexdigest()


class RequirementDecompositionResult(BaseModel):
    """D.3 requirement authority only; never a compliance result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[DECOMPOSITION_IDENTITY_VERSION] = (
        DECOMPOSITION_IDENTITY_VERSION
    )
    decomposition_id: str = Field(pattern=r"^decomposition_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    route_id: str = Field(pattern=r"^route_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    standard_id: str = Field(min_length=1)
    standard_code: str = Field(min_length=1)
    canonical_standard_code: str = Field(min_length=1)
    standard_name: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    requirements: tuple[NormativeRequirement, ...] = ()
    unresolved_spans: tuple[UnresolvedRequirementSpan, ...] = ()
    status: RequirementDecompositionStatus
    decomposition_method: Literal[
        RequirementDecompositionMethod.EXACT_SOURCE_DETERMINISTIC
    ] = RequirementDecompositionMethod.EXACT_SOURCE_DETERMINISTIC
    decomposition_version: Literal[DECOMPOSITION_VERSION] = DECOMPOSITION_VERSION

    @model_validator(mode="after")
    def validate_result_authority(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Evidence identities must be unique.")
        if len({item.requirement_id for item in self.requirements}) != len(
            self.requirements
        ):
            raise ValueError("Requirement identities must be unique.")
        if len({item.unresolved_span_id for item in self.unresolved_spans}) != len(
            self.unresolved_spans
        ):
            raise ValueError("Unresolved span identities must be unique.")
        evidence_set = set(self.evidence_ids)
        for requirement in self.requirements:
            if requirement.evidence_id not in evidence_set:
                raise ValueError("Requirement evidence is absent from the result.")
            if requirement.standard_id != self.standard_id:
                raise ValueError("Requirement standard does not match the result.")
            if (
                requirement.atomicity != RequirementAtomicity.PROVEN
                or requirement.modality != RequirementModality.NUMERIC_LIMIT
            ):
                raise ValueError("Only proven numeric requirements are D.3 outputs.")
        for span in self.unresolved_spans:
            if span.evidence_id not in evidence_set:
                raise ValueError("Unresolved-span evidence is absent from the result.")

        proven = bool(self.requirements)
        unresolved = bool(self.unresolved_spans)
        if self.status == RequirementDecompositionStatus.DECOMPOSED:
            valid_status = proven and not unresolved
        elif self.status == RequirementDecompositionStatus.PARTIAL:
            valid_status = proven and unresolved
        elif self.status == RequirementDecompositionStatus.UNRESOLVED:
            valid_status = not proven and unresolved
        else:
            valid_status = not proven and not unresolved and not self.evidence_ids
        if not valid_status:
            raise ValueError("Decomposition status does not match its outputs.")

        expected = deterministic_decomposition_id(
            candidate_id=self.candidate_id,
            route_id=self.route_id,
            evidence_ids=self.evidence_ids,
            requirements=self.requirements,
            unresolved_spans=self.unresolved_spans,
            identity_version=self.identity_version,
            decomposition_version=self.decomposition_version,
        )
        if self.decomposition_id != expected:
            raise ValueError("decomposition_id does not match canonical identity.")
        return self
