"""Qualified-scope routing proposals for D.2; never normative evidence."""

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.review_candidate import ReviewCandidateClass
from app.schemas.standards_scope import StandardScope


STANDARD_ROUTE_IDENTITY_VERSION = "v0.1-d.2-standard-route"
STANDARD_ROUTING_VERSION = "v0.1-d.2-deterministic-scope"


class StandardRouteStatus(str, Enum):
    ROUTED = "ROUTED"
    NO_STANDARD_SCOPE = "NO_STANDARD_SCOPE"
    AMBIGUOUS_STANDARD_SCOPE = "AMBIGUOUS_STANDARD_SCOPE"


class StandardRoutingMethod(str, Enum):
    DETERMINISTIC_DOMAIN_RULES = "DETERMINISTIC_DOMAIN_RULES"
    CONTEXT_ASSISTED_DETERMINISTIC = "CONTEXT_ASSISTED_DETERMINISTIC"


class QualifiedStandardScopeCandidate(BaseModel):
    """One installed standard resolved through an ACTIVE server registry entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    standard_id: str = Field(min_length=1)
    standard_code: str = Field(min_length=1)
    canonical_standard_code: str = Field(min_length=1)
    standard_name: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_status: Literal["ACTIVE"] = "ACTIVE"

    @field_validator(
        "standard_id", "standard_code", "canonical_standard_code", "standard_name"
    )
    @classmethod
    def text_is_trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Qualified scope identity text must be trimmed.")
        return value


def deterministic_route_id(
    *,
    candidate_id: str,
    scope_candidates: tuple[QualifiedStandardScopeCandidate, ...]
    | list[QualifiedStandardScopeCandidate],
    registry_fingerprint: str,
    route_identity_version: str = STANDARD_ROUTE_IDENTITY_VERSION,
    routing_version: str = STANDARD_ROUTING_VERSION,
    routing_context_id: str | None = None,
) -> str:
    """Hash stable routing authority; exclude status, labels, reason and confidence."""

    payload = {
        "route_identity_version": route_identity_version,
        "candidate_id": candidate_id,
        "routing_version": routing_version,
        "ordered_scope_candidates": [
            {
                "standard_id": item.standard_id,
                "canonical_standard_code": item.canonical_standard_code,
                "source_checksum": item.source_checksum,
            }
            for item in scope_candidates
        ],
        "registry_fingerprint": registry_fingerprint,
    }
    # Omitting an unused context preserves all qualified exact-span identities.
    if routing_context_id is not None:
        payload["routing_context_id"] = routing_context_id
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "route_" + hashlib.sha256(canonical).hexdigest()


class StandardRoute(BaseModel):
    """A validated standard-scope proposal, not an article or compliance result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route_identity_version: Literal[STANDARD_ROUTE_IDENTITY_VERSION] = (
        STANDARD_ROUTE_IDENTITY_VERSION
    )
    route_id: str = Field(pattern=r"^route_[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate_[0-9a-f]{64}$")
    routing_version: Literal[STANDARD_ROUTING_VERSION] = STANDARD_ROUTING_VERSION
    route_status: StandardRouteStatus
    candidate_topic: str = Field(min_length=1)
    candidate_class: ReviewCandidateClass
    scope_candidates: tuple[QualifiedStandardScopeCandidate, ...] = ()
    selected_scope: StandardScope | None = None
    routing_method: StandardRoutingMethod = (
        StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES
    )
    routing_confidence: float = Field(ge=0.0, le=1.0)
    routing_reason: str = Field(min_length=1)
    registry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    routing_context_id: str | None = Field(
        default=None, pattern=r"^routingcontext_[0-9a-f]{64}$"
    )

    @field_validator("candidate_topic", "routing_reason")
    @classmethod
    def route_text_is_trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("Routing metadata text must be trimmed.")
        return value

    @model_validator(mode="after")
    def validate_route_contract(self):
        ordered = tuple(
            sorted(
                self.scope_candidates,
                key=lambda item: (
                    item.canonical_standard_code,
                    item.standard_id,
                    item.source_checksum,
                ),
            )
        )
        if self.scope_candidates != ordered or len(
            {item.standard_id for item in self.scope_candidates}
        ) != len(self.scope_candidates):
            raise ValueError("Scope candidates must be unique and canonically ordered.")

        if self.route_status == StandardRouteStatus.ROUTED:
            if len(self.scope_candidates) != 1 or self.selected_scope is None:
                raise ValueError("ROUTED requires one qualified selected scope.")
            if self.selected_scope.standard_ids != [self.scope_candidates[0].standard_id]:
                raise ValueError("Selected scope must identify the resolved standard.")
        elif self.route_status == StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE:
            if len(self.scope_candidates) < 2 or self.selected_scope is not None:
                raise ValueError("AMBIGUOUS_STANDARD_SCOPE requires multiple unselected scopes.")
        elif self.scope_candidates or self.selected_scope is not None:
            raise ValueError("NO_STANDARD_SCOPE cannot contain a selected standard.")

        uses_context = (
            self.routing_method == StandardRoutingMethod.CONTEXT_ASSISTED_DETERMINISTIC
        )
        if uses_context != (self.routing_context_id is not None):
            raise ValueError(
                "Context-assisted routes require exactly one routing_context_id binding."
            )

        expected = deterministic_route_id(
            candidate_id=self.candidate_id,
            scope_candidates=self.scope_candidates,
            registry_fingerprint=self.registry_fingerprint,
            route_identity_version=self.route_identity_version,
            routing_version=self.routing_version,
            routing_context_id=self.routing_context_id,
        )
        if self.route_id != expected:
            raise ValueError("route_id does not match canonical routing identity.")
        return self
