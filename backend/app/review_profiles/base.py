"""Profile configuration primitives for completeness review."""

from dataclasses import dataclass

from app.schemas.completeness_review import ReviewSeverity


@dataclass(frozen=True)
class CheckDefinition:
    check_id: str
    title: str
    description: str
    severity: ReviewSeverity
    required: bool = True
    applicability_rule: str = "always"


@dataclass(frozen=True)
class ReviewProfile:
    profile_id: str
    checks: tuple[CheckDefinition, ...]
