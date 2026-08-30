import pytest
from pydantic import ValidationError

from app.review_profiles import select_review_profile
from app.schemas.analysis import DocumentAnalysis
from app.schemas.completeness_review import (
    CompletenessCheckResult,
    CompletenessEvidence,
    EvidenceSourceType,
    ReviewSeverity,
    ReviewStatus,
)
from app.services.completeness_review_service import CompletenessReviewService


def _review(payload: dict) -> dict[str, CompletenessCheckResult]:
    review = CompletenessReviewService().review(DocumentAnalysis.model_validate(payload))
    return {check.check_id: check for check in review.checks}


def _chapter(title: str, page: int = 5) -> dict:
    return {"title": title, "start_page": page, "end_page": page, "level": 1}


def _result(status: ReviewStatus, check_id: str) -> CompletenessCheckResult:
    evidence = []
    if status in {ReviewStatus.PASS, ReviewStatus.PARTIAL}:
        evidence = [
            CompletenessEvidence(
                source_type=EvidenceSourceType.FIELD,
                field_path="test.value",
                matched_title="test",
            )
        ]
    return CompletenessCheckResult(
        check_id=check_id,
        title=check_id,
        description="test",
        status=status,
        severity=ReviewSeverity.INFO,
        reason="test",
        evidence=evidence,
    )


def test_profile_selection_prefers_specific_special_plan() -> None:
    assert (
        select_review_profile("构件工程专项施工方案").profile_id
        == "special_construction_plan"
    )
    assert (
        select_review_profile("机电安装施工组织设计").profile_id
        == "general_construction_plan"
    )
    assert select_review_profile(None).profile_id == "general_construction_plan"


def test_cr001_overview_chapter_and_metadata_pass_with_page_evidence() -> None:
    checks = _review(
        {
            "chapters": [_chapter("工程概况", 7)],
            "project": {
                "project_name": {
                    "value": "某建设项目",
                    "source_page": 8,
                    "source_text": "工程名称：某建设项目",
                }
            },
        }
    )

    result = checks["CR-001"]
    assert result.status == ReviewStatus.PASS
    assert {(item.source_type, item.source_page) for item in result.evidence} == {
        (EvidenceSourceType.CHAPTER, 7),
        (EvidenceSourceType.FIELD, 8),
    }
    assert next(
        item for item in result.evidence if item.source_type == EvidenceSourceType.FIELD
    ).source_text == "工程名称：某建设项目"


def test_cr001_metadata_without_overview_chapter_is_partial() -> None:
    result = _review(
        {"project": {"project_location": {"value": "某市某区", "source_page": 4}}}
    )["CR-001"]

    assert result.status == ReviewStatus.PARTIAL
    assert result.evidence[0].field_path == "project.project_location"


def test_cr001_without_chapter_or_metadata_is_missing() -> None:
    result = _review({})["CR-001"]

    assert result.status == ReviewStatus.MISSING
    assert result.evidence == []


def test_cr002_empty_basis_is_missing() -> None:
    result = _review({})["CR-002"]

    assert result.status == ReviewStatus.MISSING
    assert result.evidence == []


def test_cr002_two_basis_items_pass_without_inventing_pages() -> None:
    result = _review(
        {"compilation_basis": ["设计图纸", "施工合同"]}
    )["CR-002"]

    assert result.status == ReviewStatus.PASS
    assert len(result.evidence) == 2
    assert all(item.source_page is None for item in result.evidence)


def test_cr002_basis_chapter_and_one_item_pass() -> None:
    result = _review(
        {
            "chapters": [_chapter("编制依据", 5)],
            "compilation_basis": [
                {
                    "value": "施工合同",
                    "source_page": 6,
                    "source_text": "1.1 施工合同",
                }
            ],
        }
    )["CR-002"]

    assert result.status == ReviewStatus.PASS
    assert {item.source_page for item in result.evidence} == {5, 6}


def test_cr002_basis_chapter_without_items_is_partial() -> None:
    result = _review({"chapters": [_chapter("编制依据", 5)]})["CR-002"]

    assert result.status == ReviewStatus.PARTIAL
    assert len(result.evidence) == 1
    assert "未提取到有效依据条目" in result.reason


def test_cr002_one_item_without_basis_chapter_is_partial() -> None:
    result = _review({"compilation_basis": ["施工合同"]})["CR-002"]

    assert result.status == ReviewStatus.PARTIAL
    assert len(result.evidence) == 1
    assert "缺少明确章节边界" in result.reason


def test_cr003_schedule_chapter_without_dates_is_partial() -> None:
    result = _review(
        {"chapters": [_chapter("脚手架工程施工进度计划", 13)]}
    )["CR-003"]

    assert result.status == ReviewStatus.PARTIAL
    assert result.evidence[0].source_page == 13
    assert "未从文本层提取到可靠起止日期" in result.reason


def test_cr003_schedule_field_and_chapter_pass() -> None:
    result = _review(
        {
            "chapters": [_chapter("施工进度计划", 10)],
            "schedule": {
                "plan_start_date": {
                    "value": "2026-01-10",
                    "source_page": 11,
                    "source_text": "计划2026年1月10日开工",
                }
            },
        }
    )["CR-003"]

    assert result.status == ReviewStatus.PASS
    assert {item.source_page for item in result.evidence} == {10, 11}


def test_cr003_without_schedule_chapter_or_fields_is_missing() -> None:
    result = _review({})["CR-003"]

    assert result.status == ReviewStatus.MISSING
    assert result.evidence == []


@pytest.mark.parametrize(
    ("chapters", "expected"),
    [
        (["施工准备"], ReviewStatus.PASS),
        (["技术准备", "材料准备"], ReviewStatus.PASS),
        (["技术准备"], ReviewStatus.PARTIAL),
        ([], ReviewStatus.MISSING),
    ],
)
def test_cr004_preparation_coverage(chapters: list[str], expected: ReviewStatus) -> None:
    result = _review(
        {"chapters": [_chapter(title, index + 5) for index, title in enumerate(chapters)]}
    )["CR-004"]

    assert result.status == expected


def test_cr005_material_records_and_plan_chapter_pass() -> None:
    result = _review(
        {
            "chapters": [_chapter("主要材料与设备计划", 12)],
            "main_materials": [
                {
                    "name": "钢管",
                    "specification": "48×3.0mm",
                    "source_page": 13,
                    "source_text": "钢管 48×3.0mm",
                }
            ],
        }
    )["CR-005"]

    assert result.status == ReviewStatus.PASS
    assert {item.source_page for item in result.evidence} == {12, 13}


def test_cr005_material_record_without_plan_chapter_is_partial() -> None:
    result = _review({"main_materials": [{"name": "钢管"}]})["CR-005"]

    assert result.status == ReviewStatus.PARTIAL
    assert result.evidence[0].field_path == "main_materials[0]"


def test_cr005_without_material_chapter_or_records_is_missing() -> None:
    result = _review({})["CR-005"]

    assert result.status == ReviewStatus.MISSING
    assert result.evidence == []


def test_cr006_three_methods_and_method_chapter_pass() -> None:
    result = _review(
        {
            "chapters": [_chapter("施工工艺技术", 20)],
            "main_methods": [
                {"name": "构件定位", "source_page": 21},
                {"name": "构件安装", "source_page": 22},
                {"name": "构件拆除", "source_page": 23},
            ],
        }
    )["CR-006"]

    assert result.status == ReviewStatus.PASS
    assert len(result.evidence) == 4


def test_cr006_empty_methods_is_missing_even_with_heading() -> None:
    result = _review({"chapters": [_chapter("施工方法", 20)]})["CR-006"]

    assert result.status == ReviewStatus.MISSING
    assert result.evidence[0].source_page == 20


def test_cr006_few_methods_are_partial() -> None:
    result = _review(
        {"main_methods": [{"name": "构件安装", "source_page": 21}]}
    )["CR-006"]

    assert result.status == ReviewStatus.PARTIAL
    assert result.evidence[0].source_page == 21


@pytest.mark.parametrize("status", [ReviewStatus.PASS, ReviewStatus.PARTIAL])
def test_positive_review_status_requires_evidence(status: ReviewStatus) -> None:
    with pytest.raises(ValidationError, match="require evidence"):
        CompletenessCheckResult(
            check_id="CR-X",
            title="test",
            description="test",
            status=status,
            severity=ReviewSeverity.INFO,
            reason="test",
        )


@pytest.mark.parametrize(
    "status", [ReviewStatus.MISSING, ReviewStatus.NOT_APPLICABLE]
)
def test_non_positive_status_allows_empty_evidence(status: ReviewStatus) -> None:
    result = CompletenessCheckResult(
        check_id="CR-X",
        title="test",
        description="test",
        status=status,
        severity=ReviewSeverity.INFO,
        reason="test",
    )

    assert result.evidence == []


def test_score_is_transparent_and_not_applicable_is_excluded() -> None:
    checks = [
        *[_result(ReviewStatus.PASS, f"P-{index}") for index in range(8)],
        *[_result(ReviewStatus.PARTIAL, f"R-{index}") for index in range(2)],
        _result(ReviewStatus.MISSING, "M-1"),
        _result(ReviewStatus.NOT_APPLICABLE, "N-1"),
    ]

    summary = CompletenessReviewService.summarize(checks)

    assert summary.total_checks == 12
    assert summary.applicable_checks == 11
    assert summary.passed == 8
    assert summary.partial == 2
    assert summary.missing == 1
    assert summary.not_applicable == 1
    assert summary.completeness_score == 82
