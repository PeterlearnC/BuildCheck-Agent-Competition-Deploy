"""Special construction-plan completeness profile."""

from dataclasses import replace

from app.review_profiles.base import ReviewProfile
from app.review_profiles.common import ALL_COMMON_CHECKS


SPECIAL_CHECKS = tuple(
    replace(check, required=True)
    if check.check_id == "CR-009"
    else check
    for check in ALL_COMMON_CHECKS
)


SPECIAL_CONSTRUCTION_PLAN = ReviewProfile(
    profile_id="special_construction_plan",
    checks=SPECIAL_CHECKS,
)
