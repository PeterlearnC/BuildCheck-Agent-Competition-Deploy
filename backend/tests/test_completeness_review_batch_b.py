import pytest

from app.review_profiles import select_review_profile
from app.schemas.analysis import DocumentAnalysis
from app.schemas.completeness_review import EvidenceSourceType, ReviewStatus
from app.services.completeness_evidence_service import CompletenessEvidenceService
from app.services.completeness_review_service import CompletenessReviewService


def _chapter(title: str, page: int = 5) -> dict:
    return {"title": title, "start_page": page, "end_page": page, "level": 1}


def _review(payload: dict):
    review = CompletenessReviewService().review(DocumentAnalysis.model_validate(payload))
    return review, {check.check_id: check for check in review.checks}


def _status(payload: dict, check_id: str) -> ReviewStatus:
    return _review(payload)[1][check_id].status


def test_cr007_quality_chapter_and_section_pass() -> None:
    review, checks = _review(
        {
            "chapters": [_chapter("质量保证措施", 20)],
            "section_presence": {"quality": True},
        }
    )
    result = checks["CR-007"]
    assert result.status == ReviewStatus.PASS
    assert result.suggestion is None
    assert review.summary.total_checks == 14


def test_cr007_only_quality_section_is_partial() -> None:
    assert _status(
        {"section_presence": {"quality": True}}, "CR-007"
    ) == ReviewStatus.PARTIAL


def test_cr007_quality_acceptance_standard_only_is_missing() -> None:
    assert _status(
        {
            "standards": [
                {"name": "建筑工程施工质量验收统一标准", "code": "GB50300-2013"}
            ]
        },
        "CR-007",
    ) == ReviewStatus.MISSING


def test_cr008_safety_chapter_and_section_pass() -> None:
    assert _status(
        {
            "chapters": [_chapter("安全技术措施")],
            "section_presence": {"safety": True},
        },
        "CR-008",
    ) == ReviewStatus.PASS


def test_cr008_only_safety_chapter_is_partial() -> None:
    assert _status(
        {"chapters": [_chapter("安全管理措施")]}, "CR-008"
    ) == ReviewStatus.PARTIAL


def test_cr008_safety_standard_only_is_missing() -> None:
    assert _status(
        {
            "standards": [
                {"name": "建筑施工安全检查标准", "code": "JGJ59-2011"}
            ]
        },
        "CR-008",
    ) == ReviewStatus.MISSING


def test_cr009_special_without_emergency_is_missing() -> None:
    assert _status(
        {"document_type": "构件工程专项施工方案"}, "CR-009"
    ) == ReviewStatus.MISSING


def test_cr009_general_without_emergency_is_not_applicable() -> None:
    assert _status(
        {"document_type": "构件工程施工方案"}, "CR-009"
    ) == ReviewStatus.NOT_APPLICABLE


def test_cr009_emergency_chapter_and_section_pass() -> None:
    assert _status(
        {
            "document_type": "构件工程专项施工方案",
            "chapters": [_chapter("事故应急预案")],
            "section_presence": {"emergency": True},
        },
        "CR-009",
    ) == ReviewStatus.PASS


def test_cr009_emergency_phone_alone_does_not_trigger_positive_status() -> None:
    assert _status(
        {
            "document_type": "构件工程专项施工方案",
            "chapters": [_chapter("应急联系电话：119、120")],
        },
        "CR-009",
    ) == ReviewStatus.MISSING


def test_cr010_environment_chapter_and_section_pass() -> None:
    assert _status(
        {
            "chapters": [_chapter("环境保护措施")],
            "section_presence": {"environment": True},
        },
        "CR-010",
    ) == ReviewStatus.PASS


def test_cr010_only_civilized_construction_is_partial() -> None:
    assert _status(
        {"chapters": [_chapter("文明施工措施")]}, "CR-010"
    ) == ReviewStatus.PARTIAL


def test_cr010_without_evidence_is_missing() -> None:
    assert _status({}, "CR-010") == ReviewStatus.MISSING


def test_cr011_organization_and_responsibilities_pass() -> None:
    assert _status(
        {
            "chapters": [
                _chapter("项目组织机构", 8),
                _chapter("岗位职责", 9),
            ]
        },
        "CR-011",
    ) == ReviewStatus.PASS


def test_cr011_labor_plan_only_is_partial() -> None:
    assert _status(
        {"chapters": [_chapter("劳动力计划")]}, "CR-011"
    ) == ReviewStatus.PARTIAL


def test_cr011_participant_company_field_only_is_missing() -> None:
    assert _status(
        {"project": {"construction_unit": "某建设单位"}}, "CR-011"
    ) == ReviewStatus.MISSING


def test_cr012_acceptance_standard_only_is_missing() -> None:
    assert _status(
        {
            "standards": [
                {"name": "某工程施工质量验收规范", "code": "GB50000-2020"}
            ]
        },
        "CR-012",
    ) == ReviewStatus.MISSING


def test_cr012_acceptance_basis_only_is_missing() -> None:
    assert _status(
        {"compilation_basis": ["《某工程施工质量验收规范》"]}, "CR-012"
    ) == ReviewStatus.MISSING


def test_cr012_heading_and_arrangement_pass() -> None:
    assert _status(
        {
            "chapters": [
                _chapter("检查验收", 30),
                _chapter("验收程序与验收人员", 31),
            ]
        },
        "CR-012",
    ) == ReviewStatus.PASS


def test_cr012_isolated_accepted_before_next_step_does_not_pass() -> None:
    assert _status(
        {"chapters": [_chapter("经验收合格后方可进行下一道工序")]},
        "CR-012",
    ) == ReviewStatus.PARTIAL


def test_cr012_multiple_acceptance_arrangements_without_heading_are_partial() -> None:
    assert _status(
        {
            "chapters": [
                _chapter("验收流程", 30),
                _chapter("验收记录", 31),
            ]
        },
        "CR-012",
    ) == ReviewStatus.PARTIAL


@pytest.mark.parametrize(
    "project",
    [
        {"supervision_unit": "某监理单位"},
        {"supervision_unit": "某监控单位"},
    ],
)
def test_cr013_participant_or_monitoring_company_is_not_applicable(project: dict) -> None:
    assert _status({"project": project}, "CR-013") == ReviewStatus.NOT_APPLICABLE


def test_cr013_monitoring_plan_chapter_pass() -> None:
    assert _status(
        {"chapters": [_chapter("施工监测方案")]}, "CR-013"
    ) == ReviewStatus.PASS


def test_cr013_deformation_monitoring_sentence_is_partial() -> None:
    assert _status(
        {"chapters": [_chapter("施工期间对结构变形进行监测")]}, "CR-013"
    ) == ReviewStatus.PARTIAL


def test_cr014_calculation_book_chapter_pass() -> None:
    assert _status(
        {"chapters": [_chapter("附图及计算书")]}, "CR-014"
    ) == ReviewStatus.PASS


def test_cr014_without_calculation_is_not_applicable() -> None:
    assert _status({}, "CR-014") == ReviewStatus.NOT_APPLICABLE


def test_optional_strong_evidence_is_pass() -> None:
    assert _status(
        {"chapters": [_chapter("监测监控方案")]}, "CR-013"
    ) == ReviewStatus.PASS


def test_optional_weak_evidence_is_partial() -> None:
    assert _status(
        {"chapters": [_chapter("持续量测构件位移变化")]}, "CR-013"
    ) == ReviewStatus.PARTIAL


def test_optional_without_evidence_is_not_applicable() -> None:
    assert _status({}, "CR-013") == ReviewStatus.NOT_APPLICABLE


def test_section_presence_evidence_has_no_invented_page() -> None:
    result = _review({"section_presence": {"quality": True}})[1]["CR-007"]
    evidence = next(
        item
        for item in result.evidence
        if item.source_type == EvidenceSourceType.SECTION_PRESENCE
    )
    assert evidence.source_page is None
    assert evidence.field_path == "section_presence.quality"


def test_evidence_dedup_is_shared_and_normalizes_basic_width_and_spacing() -> None:
    service = CompletenessEvidenceService(DocumentAnalysis())
    first = service.chapters(())
    assert first == []
    evidence_type = type(
        _review({"chapters": [_chapter("质量保证措施")]})[1]["CR-007"].evidence[0]
    )
    evidence = service.deduplicate(
        [
            evidence_type(
                source_type="chapter",
                source_page=8,
                matched_title="质量 保证措施",
                field_path="chapters",
            ),
            evidence_type(
                source_type="chapter",
                source_page=8,
                matched_title="质量　保证措施",
                field_path="chapters",
            ),
        ]
    )
    assert len(evidence) == 1


def test_duplicate_chapter_matches_are_deduplicated_in_check_result() -> None:
    result = _review(
        {
            "chapters": [
                _chapter("质量保证措施", 8),
                _chapter("质量保证措施", 8),
            ],
            "section_presence": {"quality": True},
        }
    )[1]["CR-007"]
    assert len(result.evidence) == 2


def test_general_profile_required_matrix() -> None:
    matrix = {
        check.check_id: check.required
        for check in select_review_profile("普通施工组织设计").checks
    }
    assert matrix == {
        **{f"CR-{index:03d}": True for index in range(1, 9)},
        "CR-009": False,
        "CR-010": True,
        "CR-011": True,
        "CR-012": True,
        "CR-013": False,
        "CR-014": False,
    }


def test_special_profile_required_matrix() -> None:
    matrix = {
        check.check_id: check.required
        for check in select_review_profile("构件专项施工方案").checks
    }
    assert matrix == {
        **{f"CR-{index:03d}": True for index in range(1, 13)},
        "CR-013": False,
        "CR-014": False,
    }


def test_fourteen_check_score_excludes_not_applicable() -> None:
    review, _ = _review(
        {
            "document_type": "普通施工方案",
            "chapters": [
                _chapter("质量保证措施"),
                _chapter("安全保证措施"),
            ],
            "section_presence": {"quality": True, "safety": True},
        }
    )
    summary = review.summary
    assert summary.total_checks == 14
    assert summary.applicable_checks == (
        summary.passed + summary.partial + summary.missing
    )
    assert summary.not_applicable == 3
    expected = int(
        ((summary.passed + summary.partial * 0.5) / summary.applicable_checks * 100)
        + 0.5
    )
    assert summary.completeness_score == expected
