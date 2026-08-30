import json
from pathlib import Path

import pytest

from app.schemas.analysis import DocumentAnalysis
from app.services.completeness_review_service import CompletenessReviewService


TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden_reviews"
FIXTURE_DIR = TESTS_DIR / "fixtures" / "golden_reviews"
GOLDEN_FILES = sorted(GOLDEN_DIR.glob("*.json"))


def assert_golden_review(analysis: DocumentAnalysis, golden: dict) -> None:
    review = CompletenessReviewService().review(analysis)
    assert review.review_profile == golden["review_profile"]
    statuses = {check.check_id: check.status.value for check in review.checks}
    for check_id, expected in golden.get("must_status", {}).items():
        assert statuses[check_id] == expected, (
            f"{check_id}: expected {expected}, got {statuses[check_id]}"
        )
    for check_id, forbidden in golden.get("must_not_status", {}).items():
        assert statuses[check_id] not in forbidden, (
            f"{check_id}: forbidden status {statuses[check_id]}"
        )
    score_rule = golden.get("score", {})
    if "equals" in score_rule:
        assert review.summary.completeness_score == score_rule["equals"]
    if "min" in score_rule:
        assert review.summary.completeness_score >= score_rule["min"]
    if "max" in score_rule:
        assert review.summary.completeness_score <= score_rule["max"]


@pytest.mark.parametrize("golden_path", GOLDEN_FILES, ids=lambda path: path.stem)
def test_completeness_review_golden(golden_path: Path) -> None:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    fixture = DocumentAnalysis.model_validate_json(
        (FIXTURE_DIR / golden["fixture"]).read_text(encoding="utf-8")
    )
    assert_golden_review(fixture, golden)
