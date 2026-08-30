"""Profile-driven completeness review over existing document analysis."""

import math

from app.review_profiles import ReviewProfile, select_review_profile
from app.schemas.analysis import DocumentAnalysis
from app.schemas.completeness_review import (
    CompletenessReview,
    CompletenessSummary,
    ReviewStatus,
)
from app.services.completeness_checks import (
    BatchACompletenessChecks,
    BatchBCompletenessChecks,
)


class CompletenessReviewService:
    STATUS_SCORE = {
        ReviewStatus.PASS: 1.0,
        ReviewStatus.PARTIAL: 0.5,
        ReviewStatus.MISSING: 0.0,
    }

    def review(self, analysis: DocumentAnalysis) -> CompletenessReview:
        document_type = (
            str(analysis.document_type.value)
            if analysis.document_type is not None
            and analysis.document_type.value is not None
            else None
        )
        profile = select_review_profile(document_type)
        checks = self._run_profile(profile, analysis)
        return CompletenessReview(
            analysis_document_type=document_type,
            review_profile=profile.profile_id,
            summary=self.summarize(checks),
            checks=checks,
        )

    @staticmethod
    def _run_profile(profile: ReviewProfile, analysis: DocumentAnalysis):
        detector_groups = (
            BatchACompletenessChecks(analysis),
            BatchBCompletenessChecks(analysis),
        )
        results = []
        for definition in profile.checks:
            detector = next(
                (
                    group
                    for group in detector_groups
                    if definition.check_id in group.detectors
                ),
                None,
            )
            if detector is None:
                raise ValueError(
                    f"Unsupported completeness check: {definition.check_id}"
                )
            results.append(detector.run(definition))
        return results

    @classmethod
    def summarize(cls, checks) -> CompletenessSummary:
        counts = {status: 0 for status in ReviewStatus}
        for check in checks:
            counts[check.status] += 1
        applicable = len(checks) - counts[ReviewStatus.NOT_APPLICABLE]
        points = sum(cls.STATUS_SCORE.get(check.status, 0.0) for check in checks)
        score = math.floor((points / applicable * 100) + 0.5) if applicable else 0
        return CompletenessSummary(
            total_checks=len(checks),
            applicable_checks=applicable,
            passed=counts[ReviewStatus.PASS],
            partial=counts[ReviewStatus.PARTIAL],
            missing=counts[ReviewStatus.MISSING],
            not_applicable=counts[ReviewStatus.NOT_APPLICABLE],
            completeness_score=score,
        )
