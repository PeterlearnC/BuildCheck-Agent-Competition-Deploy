"""Configured completeness-review profiles."""

from app.review_profiles.base import ReviewProfile
from app.review_profiles.general_construction_plan import GENERAL_CONSTRUCTION_PLAN
from app.review_profiles.special_construction_plan import SPECIAL_CONSTRUCTION_PLAN


def select_review_profile(document_type: str | None) -> ReviewProfile:
    value = (document_type or "").strip()
    if "专项施工方案" in value or "专项方案" in value:
        return SPECIAL_CONSTRUCTION_PLAN
    return GENERAL_CONSTRUCTION_PLAN


__all__ = [
    "GENERAL_CONSTRUCTION_PLAN",
    "SPECIAL_CONSTRUCTION_PLAN",
    "ReviewProfile",
    "select_review_profile",
]
