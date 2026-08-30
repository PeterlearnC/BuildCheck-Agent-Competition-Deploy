"""Completeness check definitions shared by construction-plan profiles."""

from app.review_profiles.base import CheckDefinition
from app.schemas.completeness_review import ReviewSeverity


BATCH_A_CHECKS = (
    CheckDefinition(
        "CR-001",
        "工程概况",
        "检查方案是否包含工程概况章节及可追溯的项目基本信息。",
        ReviewSeverity.IMPORTANT,
    ),
    CheckDefinition(
        "CR-002",
        "编制依据",
        "检查方案是否列出明确的编制依据、标准或依据文件。",
        ReviewSeverity.IMPORTANT,
    ),
    CheckDefinition(
        "CR-003",
        "施工计划 / 施工进度",
        "检查方案是否包含施工计划或进度安排及可靠的工期信息。",
        ReviewSeverity.WARNING,
    ),
    CheckDefinition(
        "CR-004",
        "施工准备",
        "检查技术、材料、人员或机具等施工准备内容。",
        ReviewSeverity.WARNING,
    ),
    CheckDefinition(
        "CR-005",
        "主要材料与设备",
        "检查主要材料、设备或施工机具是否有明确计划或清单。",
        ReviewSeverity.WARNING,
    ),
    CheckDefinition(
        "CR-006",
        "施工工艺 / 施工方法",
        "检查主要施工流程、关键工序及施工方法是否得到明确说明。",
        ReviewSeverity.IMPORTANT,
    ),
)


BATCH_B_CHECKS = (
    CheckDefinition(
        "CR-007",
        "质量保证措施",
        "检查方案是否包含可定位的质量保证、质量管理或质量控制措施。",
        ReviewSeverity.IMPORTANT,
    ),
    CheckDefinition(
        "CR-008",
        "安全保证措施",
        "检查方案是否包含可定位的安全保证、安全管理或安全技术措施。",
        ReviewSeverity.IMPORTANT,
    ),
    CheckDefinition(
        "CR-009",
        "应急处置",
        "检查方案是否包含应急预案、应急响应或事故处置安排。",
        ReviewSeverity.IMPORTANT,
        required=False,
        applicability_rule="profile_required",
    ),
    CheckDefinition(
        "CR-010",
        "文明施工 / 环境保护",
        "检查文明施工、绿色施工或环境保护措施是否得到明确说明。",
        ReviewSeverity.WARNING,
    ),
    CheckDefinition(
        "CR-011",
        "施工组织 / 人员职责",
        "检查施工组织机构、管理人员及岗位职责是否得到说明。",
        ReviewSeverity.WARNING,
    ),
    CheckDefinition(
        "CR-012",
        "验收要求",
        "检查方案自身是否包含验收程序、条件、人员或记录等安排。",
        ReviewSeverity.WARNING,
    ),
    CheckDefinition(
        "CR-013",
        "监测监控",
        "检查是否包含与工程结构或施工变形相关的监测安排。",
        ReviewSeverity.WARNING,
        required=False,
        applicability_rule="evidence_present",
    ),
    CheckDefinition(
        "CR-014",
        "计算书 / 验算",
        "检查既有章节结构中是否存在计算书、设计计算或验算内容。",
        ReviewSeverity.WARNING,
        required=False,
        applicability_rule="evidence_present",
    ),
)


ALL_COMMON_CHECKS = (*BATCH_A_CHECKS, *BATCH_B_CHECKS)
