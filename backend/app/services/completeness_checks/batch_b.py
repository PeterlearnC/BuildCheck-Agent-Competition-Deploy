"""Deterministic CR-007 through CR-014 completeness checks."""

from collections.abc import Callable

from app.review_profiles.base import CheckDefinition
from app.schemas.analysis import DocumentAnalysis
from app.schemas.completeness_review import (
    CompletenessCheckResult,
    CompletenessEvidence,
    ReviewStatus,
)
from app.services.completeness_evidence_service import CompletenessEvidenceService


class BatchBCompletenessChecks:
    def __init__(self, analysis: DocumentAnalysis) -> None:
        self.analysis = analysis
        self.evidence = CompletenessEvidenceService(analysis)
        self.detectors: dict[
            str, Callable[[CheckDefinition], CompletenessCheckResult]
        ] = {
            "CR-007": self.quality,
            "CR-008": self.safety,
            "CR-009": self.emergency,
            "CR-010": self.environment,
            "CR-011": self.organization,
            "CR-012": self.acceptance,
            "CR-013": self.monitoring,
            "CR-014": self.calculation,
        }

    def run(self, definition: CheckDefinition) -> CompletenessCheckResult:
        detector = self.detectors.get(definition.check_id)
        if detector is None:
            raise ValueError(f"Unsupported completeness check: {definition.check_id}")
        return detector(definition)

    def quality(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (
                r"质量保证(?:措施|体系)",
                r"质量管理(?:措施|体系)",
                r"质量控制(?:措施)?",
                r"质量标准",
            )
        )
        section = self.evidence.section_presence("quality")
        evidence = self.evidence.deduplicate([*chapters, *section])
        if chapters and section:
            return self._result(definition, ReviewStatus.PASS, "发现明确质量措施章节，且质量章节存在性已确认。", evidence)
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现质量措施章节或质量章节存在性证据，证据链尚不完整。",
                evidence,
                "建议补充可定位的质量管理体系、控制措施及质量标准。",
            )
        return self._no_evidence(definition, "未识别到质量保证或质量控制措施。", "建议补充质量保证体系、质量控制措施及检查要求。")

    def safety(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (
                r"安全(?:生产)?保证措施",
                r"安全技术措施",
                r"安全管理(?:措施|体系)",
                r"安全保证体系",
            )
        )
        section = self.evidence.section_presence("safety")
        evidence = self.evidence.deduplicate([*chapters, *section])
        if chapters and section:
            return self._result(definition, ReviewStatus.PASS, "发现明确安全措施章节，且安全章节存在性已确认。", evidence)
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现安全措施章节或安全章节存在性证据，证据链尚不完整。",
                evidence,
                "建议补充可定位的安全管理体系、技术措施及责任安排。",
            )
        return self._no_evidence(definition, "未识别到安全保证或安全技术措施。", "建议补充安全保证体系、管理措施及关键工序安全技术要求。")

    def emergency(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (
                r"应急处置",
                r"应急预案",
                r"应急救援",
                r"应急措施",
                r"事故应急",
                r"应急响应",
                r"应急组织",
            )
        )
        section = self.evidence.section_presence("emergency")
        evidence = self.evidence.deduplicate([*chapters, *section])
        if chapters and section:
            return self._result(definition, ReviewStatus.PASS, "发现明确应急章节，且应急章节存在性已确认。", evidence)
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现应急章节或应急章节存在性证据，应急安排证据尚不完整。",
                evidence,
                "建议补充应急组织、响应流程、处置措施及必要的联络安排。",
            )
        return self._no_evidence(definition, "未识别到可靠的应急处置内容。", "建议补充应急预案、救援组织、响应流程及事故处置措施。")

    def environment(self, definition: CheckDefinition) -> CompletenessCheckResult:
        chapters = self.evidence.chapters(
            (
                r"文明施工",
                r"绿色施工",
                r"环境保护",
                r"环保措施",
                r"扬尘控制",
                r"噪声控制",
                r"水土保持",
            )
        )
        section = self.evidence.section_presence("environment")
        evidence = self.evidence.deduplicate([*chapters, *section])
        if chapters and section:
            return self._result(definition, ReviewStatus.PASS, "发现文明施工或环境保护章节，且环境章节存在性已确认。", evidence)
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现文明施工、环境保护章节或章节存在性证据中的一部分。",
                evidence,
                "建议同时完善文明施工和环境保护措施，并覆盖扬尘、噪声等主要影响。",
            )
        return self._no_evidence(definition, "未识别到文明施工或环境保护内容。", "建议补充文明施工、环境保护及主要污染控制措施。")

    def organization(self, definition: CheckDefinition) -> CompletenessCheckResult:
        organization = self.evidence.chapters(
            (
                r"施工组织(?:管理|机构)?",
                r"项目组织机构",
                r"管理组织机构",
            )
        )
        responsibilities = self.evidence.chapters(
            (r"岗位职责", r"人员职责", r"职责分工")
        )
        supporting = self.evidence.chapters(
            (r"项目管理人员", r"劳动力配置", r"劳动力计划", r"人员配置", r"管理人员表")
        )
        evidence = self.evidence.deduplicate(
            [*organization, *responsibilities, *supporting]
        )
        if organization and responsibilities:
            return self._result(definition, ReviewStatus.PASS, "发现组织机构及岗位或人员职责等互补内容。", evidence)
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现组织机构、职责或人员配置中的部分内容。",
                evidence,
                "建议补充施工组织结构、管理人员配置及清晰的岗位职责分工。",
            )
        return self._no_evidence(definition, "未识别到施工组织或人员职责内容。", "建议补充施工组织机构、项目管理人员及岗位职责分工。")

    def acceptance(self, definition: CheckDefinition) -> CompletenessCheckResult:
        headings = self.evidence.chapters(
            (r"检查验收", r"验收要求", r"验收管理")
        )
        arrangements = self.evidence.chapters(
            (r"验收程序", r"验收条件", r"验收人员", r"验收内容", r"验收流程", r"验收记录", r"验收标准")
        )
        weak = self.evidence.chapters(
            (r"验收合格后", r"经.*验收.*合格", r"报.*验收")
        )
        evidence = self.evidence.deduplicate([*headings, *arrangements, *weak])
        if headings and arrangements:
            return self._result(definition, ReviewStatus.PASS, "发现明确验收章节及程序、条件、人员或记录等实际验收安排。", evidence)
        if evidence:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "发现方案验收相关内容，但尚未形成明确且完整的验收安排。",
                evidence,
                "建议补充验收程序、验收条件、参与人员、验收内容及记录要求。",
            )
        return self._no_evidence(definition, "未识别到方案自身的验收安排；规范名称中的“验收”不作为本项证据。", "建议补充检查验收章节，并明确程序、条件、人员、内容及记录要求。")

    def monitoring(self, definition: CheckDefinition) -> CompletenessCheckResult:
        headings = self.evidence.chapters(
            (r"监测监控", r"监测方案", r"施工监测", r"变形监测", r"沉降监测", r"位移监测", r"监控量测")
        )
        weak = self.evidence.chapters(
            (
                r"(?:变形|沉降|位移|应力|挠度).{0,10}(?:监测|量测)",
                r"(?:监测|量测).{0,10}(?:变形|沉降|位移|应力|挠度)",
            )
        )
        evidence = self.evidence.deduplicate([*headings, *weak])
        if headings:
            return self._result(definition, ReviewStatus.PASS, "发现明确的工程监测或监控量测章节。", evidence)
        if weak:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "发现少量与工程变形或受力相关的监测内容，但缺少明确监测章节。",
                evidence,
                "建议明确监测项目、测点、频率、预警条件及处置流程。",
            )
        return self._no_evidence(definition, "未识别到可靠的工程监测内容。", "建议根据方案适用性确认是否需要补充工程监测安排。")

    def calculation(self, definition: CheckDefinition) -> CompletenessCheckResult:
        headings = self.evidence.chapters(
            (r"计算书", r"附图及计算书", r"设计计算", r"验算", r"计算分析", r"荷载计算")
        )
        weak = self.evidence.chapters(
            (r"经计算", r"计算结果", r"计算复核")
        )
        evidence = self.evidence.deduplicate([*headings, *weak])
        if headings:
            return self._result(definition, ReviewStatus.PASS, "在既有章节结构中发现明确计算书、设计计算或验算标题。", evidence)
        if weak:
            return self._result(
                definition,
                ReviewStatus.PARTIAL,
                "仅发现可靠的零散计算语义，未识别到明确计算区域或标题。",
                evidence,
                "建议根据方案适用性确认并完善计算书或验算章节。",
            )
        return self._no_evidence(definition, "既有章节结构中未识别到计算书或验算内容。", "建议根据方案适用性确认是否需要附计算书或验算资料。")

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
    def _no_evidence(
        cls,
        definition: CheckDefinition,
        reason: str,
        suggestion: str,
    ) -> CompletenessCheckResult:
        if definition.required:
            return cls._result(
                definition, ReviewStatus.MISSING, reason, [], suggestion
            )
        return cls._result(
            definition,
            ReviewStatus.NOT_APPLICABLE,
            reason,
            [],
            None,
        )
