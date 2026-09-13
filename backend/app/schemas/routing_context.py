"""Source-grounded supplemental routing context for D.2-P2."""

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ROUTING_CONTEXT_IDENTITY_VERSION = "v0.1-d.2-p2-routing-context"
ROUTING_CONTEXT_BOUNDARY_VERSION = "v0.1-d.2-p2-numbered-item"


class RoutingContextBoundaryMethod(str, Enum):
    EXACT_ONLY = "EXACT_ONLY"
    NUMBERED_ITEM = "NUMBERED_ITEM"


class RoutingContextSourceBindingMethod(str, Enum):
    STORED_DOCUMENT_SHA_VERIFIED = "STORED_DOCUMENT_SHA_VERIFIED"


def deterministic_routing_context_id(
    *,
    candidate_id: str,
    document_sha256: str,
    page_number: int,
    context_start: int,
    context_end: int,
    context_text_sha256: str,
    candidate_relative_start: int,
    candidate_relative_end: int,
    candidate_source_text_sha256: str,
    source_page_text_sha256: str,
    source_binding_method: RoutingContextSourceBindingMethod,
    boundary_method: RoutingContextBoundaryMethod,
    context_identity_version: str = ROUTING_CONTEXT_IDENTITY_VERSION,
    boundary_version: str = ROUTING_CONTEXT_BOUNDARY_VERSION,
) -> str:
    """Hash exact context provenance; never include routing outcomes or standards."""

    payload = {
        "context_identity_version": context_identity_version,
        "candidate_id": candidate_id,
        "document_sha256": document_sha256,
        "page_number": page_number,
        "context_start": context_start,
        "context_end": context_end,
        "context_text_sha256": context_text_sha256,
        "candidate_relative_start": candidate_relative_start,
        "candidate_relative_end": candidate_relative_end,
        "candidate_source_text_sha256": candidate_source_text_sha256,
        "source_page_text_sha256": source_page_text_sha256,
        "source_binding_method": source_binding_method.value,
        "boundary_method": boundary_method.value,
        "boundary_version": boundary_version,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "routingcontext_" + hashlib.sha256(canonical).hexdigest()


class RoutingContext(BaseModel):
    """Verified plan text used only to improve deterministic scope routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_identity_version: Literal[ROUTING_CONTEXT_IDENTITY_VERSION] = (
        ROUTING_CONTEXT_IDENTITY_VERSION
    )
    routing_context_id: str = Field(pattern=r"^routingcontext_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(gt=0)
    context_start: int = Field(ge=0)
    context_end: int = Field(gt=0)
    context_text: str = Field(min_length=1, max_length=280)
    context_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_relative_start: int = Field(ge=0)
    candidate_relative_end: int = Field(gt=0)
    candidate_source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_page_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_method: Literal[
        RoutingContextSourceBindingMethod.STORED_DOCUMENT_SHA_VERIFIED
    ] = RoutingContextSourceBindingMethod.STORED_DOCUMENT_SHA_VERIFIED
    boundary_method: RoutingContextBoundaryMethod
    boundary_version: Literal[ROUTING_CONTEXT_BOUNDARY_VERSION] = (
        ROUTING_CONTEXT_BOUNDARY_VERSION
    )

    @field_validator("document_id")
    @classmethod
    def document_id_is_explicit(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("Routing context document_id must be explicit and trimmed.")
        return value

    @model_validator(mode="after")
    def validate_context_identity(self):
        if self.context_end != self.context_start + len(self.context_text):
            raise ValueError("Routing context span does not match context_text length.")
        if not (
            0 <= self.candidate_relative_start
            < self.candidate_relative_end
            <= len(self.context_text)
        ):
            raise ValueError("Candidate-relative offsets must lie inside routing context.")
        expected_context_hash = hashlib.sha256(
            self.context_text.encode("utf-8")
        ).hexdigest()
        if self.context_text_sha256 != expected_context_hash:
            raise ValueError("context_text_sha256 does not match context_text.")
        candidate_slice = self.context_text[
            self.candidate_relative_start : self.candidate_relative_end
        ]
        expected_candidate_hash = hashlib.sha256(
            candidate_slice.encode("utf-8")
        ).hexdigest()
        if self.candidate_source_text_sha256 != expected_candidate_hash:
            raise ValueError("Routing context does not preserve the candidate source slice.")
        expected_id = deterministic_routing_context_id(
            candidate_id=self.candidate_id,
            document_sha256=self.document_sha256,
            page_number=self.page_number,
            context_start=self.context_start,
            context_end=self.context_end,
            context_text_sha256=self.context_text_sha256,
            candidate_relative_start=self.candidate_relative_start,
            candidate_relative_end=self.candidate_relative_end,
            candidate_source_text_sha256=self.candidate_source_text_sha256,
            source_page_text_sha256=self.source_page_text_sha256,
            source_binding_method=self.source_binding_method,
            boundary_method=self.boundary_method,
            context_identity_version=self.context_identity_version,
            boundary_version=self.boundary_version,
        )
        if self.routing_context_id != expected_id:
            raise ValueError("routing_context_id does not match canonical context identity.")
        return self
