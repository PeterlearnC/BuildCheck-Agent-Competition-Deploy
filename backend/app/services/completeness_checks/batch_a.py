"""Deterministic CR-001 through CR-006 completeness checks."""

from collections.abc import Callable

from app.review_profiles.base import CheckDefinition
from app.schemas.analysis import DocumentAnalysis
from app.schemas.completeness_review import (
    CompletenessCheckResult,
    CompletenessEvidence,
    ReviewStatus,
)
from app.services.completeness_evidence_service import CompletenessEvidenceService


class BatchACompletenessChecks:
    def __init__(self, analysis: DocumentAnalysis) -> None:
        self.analysis = analysis
        self.evidence = CompletenessEvidenceService(analysis)
        self.detectors: dict[
            str, Callable[[CheckDefinition], CompletenessCheckResult]
        ] = {
            "CR-001": self.project_overview,
            "CR-002": self.compilation_basis,
            "CR-003": self.schedule,
            "CR-004": self.preparation,
            "CR-005": self.materials_and_equipment,
            "CR-006": self.methods,
        }

    def run(self, definition: CheckDefinition) -> CompletenessCheckResult:
        detector = self.detectors.get(definition.check_id)
        if detector is None:
            raise ValueError(f"Unsupported completeness check: {definition.check_id}")
        return detector(definition)

    def project_overview(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters((r"工程概况", r"项目概况", r"工程简介"))
        fields = self.evidence.project_fields()
        evidence = [*chapters, *fields]
        if chapters and fields:
            return self._result(
                definition,
                ReviewStatus.PASS,
                "发现工程概况章节及可追溯的项目基本信息。",
                evidence,
            )
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现工程概况章节或项目字段，二者未形成完整证据链。",
                evidence,
                "建议补充明确的工程概况章节，并核对项目名称、地点等基本信息。",
            )
        return self._missing(
            definition,
            "未识别到工程概况章节或可靠的项目基本信息。",
            "建议补充工程概况、项目名称、建设地点及主要参建信息。",
        )

    def compilation_basis(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (r"编制依据", r"编制说明及依据", r"依据文件", r"主要规范标准")
        )
        items = self.evidence.compilation_basis()
        evidence = [*chapters, *items]
        has_basis_chapter = bool(chapters)
        basis_item_count = len(items)
        if (has_basis_chapter and basis_item_count >= 1) or (
            not has_basis_chapter and basis_item_count >= 2
        ):
            reason = (
                f"发现明确编制依据章节及 {basis_item_count} 项有效依据。"
                if has_basis_chapter
                else f"未识别到明确章节，但发现 {basis_item_count} 项有效编制依据。"
            )
            return self._result(
                definition,
                ReviewStatus.PASS,
                reason,
                evidence,
            )
        if evidence:
            reason = (
                "发现明确编制依据章节，但未提取到有效依据条目。"
                if has_basis_chapter
                else "仅发现一项有效依据，缺少明确章节边界或足够内容。"
            )
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                reason,
                evidence,
                "建议补充适用的设计文件、合同文件及主要规范标准。",
            )
        return self._missing(
            definition,
            "未识别到编制依据章节或有效依据条目。",
            "建议增加编制依据章节并列明主要设计文件、合同及规范标准。",
        )

    def schedule(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (r"施工计划", r"施工进度", r"进度计划", r"工期计划")
        )
        fields = self.evidence.schedule_fields()
        evidence = [*chapters, *fields]
        if chapters and fields:
            return self._result(
                definition,
                ReviewStatus.PASS,
                "发现施工计划或进度章节，并提取到可靠的工期字段。",
                evidence,
            )
        if evidence:
            reason = (
                "发现施工进度计划章节，但未从文本层提取到可靠起止日期。"
                if chapters
                else "提取到工期字段，但未识别到明确的施工计划或进度章节。"
            )
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                reason,
                evidence,
                "建议人工核查施工进度计划表，并补充可读取的起止日期信息。",
            )
        return self._missing(
            definition,
            "未识别到施工计划、进度章节或可靠工期信息。",
            "建议补充施工进度安排、计划起止日期及关键节点。",
        )

    def preparation(self, definition: CheckDefinition) -> CompletenessCheckResult:
        broad = self.evidence.chapters((r"施工准备", r"施工部署", r"前期准备"))
        components = self.evidence.chapters(
            (r"技术准备", r"材料准备", r"人员准备", r"劳动力准备", r"机具准备", r"机械.*准备")
        )
        evidence = self.evidence.deduplicate([*broad, *components])
        component_titles = {item.matched_title for item in components}
        if broad or len(component_titles) >= 2:
            return self._result(
                definition,
                ReviewStatus.PASS,
                "发现综合施工准备章节或多个准备工作分项。",
                evidence,
            )
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅识别到单项施工准备内容。",
                evidence,
                "建议补充技术、材料、人员及机具等施工准备内容。",
            )
        return self._missing(
            definition,
            "未识别到施工准备或施工部署内容。",
            "建议补充技术、材料、人员和机具准备安排。",
        )

    def materials_and_equipment(
        self, definition: CheckDefinition
    ) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (
                r"材料.*(?:计划|准备)",
                r"主要材料",
                r"材料与设备",
                r"主要机具",
                r"施工机械",
                r"设备.*计划",
            )
        )
        items = self.evidence.materials()
        evidence = [*chapters, *items]
        if chapters and items:
            return self._result(
                definition,
                ReviewStatus.PASS,
                f"发现材料或设备计划章节及 {len(items)} 项结构化材料记录。",
                evidence,
            )
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现材料设备章节或结构化材料记录，证据覆盖不完整。",
                evidence,
                "建议补充主要材料、设备和机具清单及必要的规格用途信息。",
            )
        return self._missing(
            definition,
            "未识别到主要材料、设备或施工机具内容。",
            "建议补充主要材料与设备计划、规格及用途。",
        )

    def methods(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (r"施工工艺", r"施工方法", r"工艺流程", r"安装施工", r"施工工艺技术")
        )
        methods = self.evidence.methods()
        evidence = [*chapters, *methods]
        if len(methods) >= 3 and chapters:
            return self._result(
                definition,
                ReviewStatus.PASS,
                f"发现明确的施工工艺或方法章节及 {len(methods)} 项施工方法。",
                evidence,
            )
        if methods:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "已提取施工方法，但方法数量或章节证据尚不充分。",
                evidence,
                "建议完善施工工艺章节，补充主要施工流程、关键工序及技术要求。",
            )
        return self._missing(
            definition,
            "未识别到可靠的施工工艺或施工方法内容。",
            "未识别到明确的施工工艺或施工方法内容，建议补充主要施工流程、关键工序及技术要求。",
            evidence=chapters,
        )

    @staticmethod
    def _result(
        definition: CheckDefinition,
        status: ReviewStatus,
        reason: str,
        evidence: list[CompletenessEvidence],
        suggestion: str | None = None,
    ) -> CompletenessCheckResult:
        return CompletenessCheckResult(
            check_id=definition.check_id,
            title=definition.title,
            description=definition.description,
            status=status,
            severity=definition.severity,
            reason=reason,
            evidence=evidence,
            suggestion=suggestion,
        )

    @classmethod
    def _missing(
        cls,
        definition: CheckDefinition,
        reason: str,
        suggestion: str,
        evidence: list[CompletenessEvidence] | None = None,
    ) -> CompletenessCheckResult:
        return cls._result(
            definition,
            ReviewStatus.MISSING,
            reason,
            evidence or [],
            suggestion,
        )
