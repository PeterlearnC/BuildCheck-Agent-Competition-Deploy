"""General construction-plan completeness profile."""

from app.review_profiles.base import ReviewProfile
from app.review_profiles.common import ALL_COMMON_CHECKS


GENERAL_CONSTRUCTION_PLAN = ReviewProfile(
    profile_id="general_construction_plan",
    checks=ALL_COMMON_CHECKS,
)
