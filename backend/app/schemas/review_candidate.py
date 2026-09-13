"""Source-grounded, non-authoritative review-candidate contracts for D.1."""

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REVIEW_CANDIDATE_IDENTITY_VERSION = "v0.1-d.1-review-candidate"
REVIEW_CANDIDATE_DISCOVERY_VERSION = "v0.1-d.1-deterministic-signals"


class ReviewCandidateStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    UNRESOLVED = "UNRESOLVED"


class ReviewCandidateClass(str, Enum):
    NUMERIC_CONTROL = "NUMERIC_CONTROL"
    CONSTRUCTION_REQUIREMENT = "CONSTRUCTION_REQUIREMENT"
    MATERIAL_REQUIREMENT = "MATERIAL_REQUIREMENT"
    STRUCTURAL_CONFIGURATION = "STRUCTURAL_CONFIGURATION"
    INSPECTION_REQUIREMENT = "INSPECTION_REQUIREMENT"
    SAFETY_REQUIREMENT = "SAFETY_REQUIREMENT"
    PROCEDURAL_REQUIREMENT = "PROCEDURAL_REQUIREMENT"
    UNRESOLVED = "UNRESOLVED"


class ReviewCandidateDiscoveryMethod(str, Enum):
    DETERMINISTIC_SIGNAL_RULES = "DETERMINISTIC_SIGNAL_RULES"


class CandidateSourceSpan(BaseModel):
    """One exact, reconstructable source span from one physical parsed page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(gt=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_span_identity(self):
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("Candidate source span does not match source_text length.")
        expected = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected:
            raise ValueError("Candidate source_text_sha256 does not match source_text.")
        return self


def deterministic_candidate_id(
    *,
    document_sha256: str,
    source_spans: list[CandidateSourceSpan] | tuple[CandidateSourceSpan, ...],
    candidate_class: ReviewCandidateClass,
    identity_version: str = REVIEW_CANDIDATE_IDENTITY_VERSION,
    discovery_version: str = REVIEW_CANDIDATE_DISCOVERY_VERSION,
) -> str:
    """Hash authority-bearing source identity; exclude confidence and status."""

    payload = {
        "identity_version": identity_version,
        "document_sha256": document_sha256,
        "ordered_source_spans": [
            {
                "page_number": span.page_number,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "source_text_sha256": span.source_text_sha256,
            }
            for span in source_spans
        ],
        "candidate_class": candidate_class.value,
        "discovery_version": discovery_version,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "candidate_" + hashlib.sha256(canonical).hexdigest()


class ReviewCandidate(BaseModel):
    """An exact plan span worth later review, never a compliance judgment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[REVIEW_CANDIDATE_IDENTITY_VERSION] = (
        REVIEW_CANDIDATE_IDENTITY_VERSION
    )
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spans: tuple[CandidateSourceSpan, ...] = Field(min_length=1)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic: str = Field(min_length=1)
    candidate_class: ReviewCandidateClass
    discovery_method: Literal[
        ReviewCandidateDiscoveryMethod.DETERMINISTIC_SIGNAL_RULES
    ] = ReviewCandidateDiscoveryMethod.DETERMINISTIC_SIGNAL_RULES
    discovery_version: Literal[REVIEW_CANDIDATE_DISCOVERY_VERSION] = (
        REVIEW_CANDIDATE_DISCOVERY_VERSION
    )
    confidence: float = Field(ge=0.0, le=1.0)
    status: ReviewCandidateStatus

    @field_validator("document_id", "topic")
    @classmethod
    def text_fields_are_explicit(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("Candidate identity text must be non-blank without edge whitespace.")
        return value

    @model_validator(mode="after")
    def validate_candidate_authority(self):
        # D.1 emits bounded single-page candidates. The plural field keeps the
        # identity contract ready for a separately qualified multi-span phase.
        if len(self.source_spans) != 1:
            raise ValueError("D.1 candidates require exactly one source span.")
        span = self.source_spans[0]
        if self.source_text != span.source_text:
            raise ValueError("Candidate source_text must equal its exact source span.")
        expected_text_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected_text_hash:
            raise ValueError("Candidate source_text_sha256 does not match source_text.")
        expected_id = deterministic_candidate_id(
            document_sha256=self.document_sha256,
            source_spans=self.source_spans,
            candidate_class=self.candidate_class,
            identity_version=self.identity_version,
            discovery_version=self.discovery_version,
        )
        if self.candidate_id != expected_id:
            raise ValueError("candidate_id does not match canonical source identity.")
        unresolved = self.candidate_class == ReviewCandidateClass.UNRESOLVED
        if unresolved != (self.status == ReviewCandidateStatus.UNRESOLVED):
            raise ValueError("UNRESOLVED class and status must be paired.")
        return self
