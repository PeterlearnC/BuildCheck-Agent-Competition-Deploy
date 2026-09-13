"""Read-only competition presentation service over the qualified F1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.competition_demo import (
    CompetitionCaseRunResult,
    CompetitionDemoMetadata,
    CompetitionFindingCaseRunResult,
    CompetitionFindingCaseSummary,
    CompetitionPlanEvidence,
    CompetitionPlanFactEvidence,
    CompetitionPlanSummary,
    CompetitionReadinessCheck,
    CompetitionReviewGapCaseRunResult,
    CompetitionReviewGapCaseSummary,
    CompetitionReviewGapPlanEvidence,
    CompetitionReviewGapTechnicalProvenance,
    CompetitionStandardEvidence,
    CompetitionTechnicalProvenance,
)
from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.findings_workspace import ReviewGapWorkspaceItem
from app.schemas.review_finding import ReviewFinding
from app.services.review.compliance_comparison_service import (
    ComplianceComparisonError,
)
from app.services.review.findings_workspace_service import (
    FindingsWorkspaceError,
    FindingsWorkspaceService,
)
from app.services.review.plan_fact_service import PlanFactError
from app.services.review.review_finding_service import (
    ReviewFindingAuthorityMismatchError,
    ReviewFindingAuthorityUnavailableError,
    ReviewFindingSourceChangedError,
)
from app.services.review.requirement_service import RequirementSelectionError
from app.services.review.review_unit_service import ReviewDocumentNotFoundError, ReviewUnitError
from app.services.standards.standard_repository import StandardRepository
from scripts.prepare_competition_demo import (
    CompetitionBootstrapError,
    DemoFindingCase,
    DemoManifest,
    DemoReviewGapCase,
    QualifiedCorpusPackage,
    _assert_live_finding,
    _finding_service,
    _repository_payload,
    build_comparison_request,
    canonical_sha256,
    load_corpus_package,
    load_manifest,
    select_qualified_review_gap,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "competition" / "demo-manifest.json"

SAMPLE_MODE_NOTICE = (
    "当前演示使用已验证的真实施工方案与已资格化规范证据，"
    "系统实时重建局部审查结果。"
)
SCOPE_NOTICE = (
    "本结果仅针对当前方案片段与单项规范要求的局部比较，"
    "不代表整份方案总体合规状态。"
)
DEMO_SCOPE_NOTICE = (
    "本演示仅呈现固定局部机器结果，不代表整份方案总体合规状态或完整审查覆盖。"
)
REVIEW_GAP_SCOPE_NOTICE = (
    "该项是机器审查工作区中的明确审查缺口，不是合规判断，"
    "也不代表整份方案的覆盖或批准状态。"
)
REVIEW_GAP_EXPLANATION = (
    "方案中已识别出一个局部审查目标，但当前资格化机器管线无法建立足够的"
    "规范范围权威，因此没有进入规范要求分解或合规比较。"
)

DECISION_LABELS: dict[ComparisonDecision, str] = {
    ComparisonDecision.COMPLIANT: "局部符合",
    ComparisonDecision.NON_COMPLIANT: "局部不符合",
    ComparisonDecision.INSUFFICIENT_INFORMATION: "局部信息不足",
}

DECISION_EXPLANATIONS: dict[ComparisonDecision, str] = {
    ComparisonDecision.COMPLIANT: "当前方案片段满足该项原子规范要求。",
    ComparisonDecision.NON_COMPLIANT: "当前方案控制值与该项规范限值发生局部冲突。",
    ComparisonDecision.INSUFFICIENT_INFORMATION: (
        "现有证据不足以形成该局部规范要求的确定性违规结论。"
    ),
}

# Presentation-only copy; it never supplies a decision or Finding identity.
CASE_PRESENTATION: dict[str, tuple[str, str]] = {
    "CASE-A": ("可调托撑插入长度", "对比方案片段中的插入长度与规范最小限值。"),
    "CASE-B": ("立杆钢管间隙控制", "对比方案控制上限与规范允许上限。"),
    "CASE-C": ("套管预埋水平标高控制", "展示当前标准范围无法建立时形成的明确审查缺口。"),
}

PLAN_PRESENTATION: dict[str, str] = {
    "f42886ef-83ec-454c-86e2-34d5c238ca0f": (
        "山东汇金国际金融中心建设项目 · 贝雷梁高支模专项施工方案"
    ),
    "04039d98-4131-422a-b29a-256bade04a6a": (
        "和景家园 · 施工组织设计"
    ),
}


class CompetitionDemoError(RuntimeError):
    """Base class for safe route error mapping."""


class CompetitionDemoCaseNotFoundError(CompetitionDemoError):
    pass


class CompetitionDemoNotReadyError(CompetitionDemoError):
    pass


class CompetitionDemoAuthorityDriftError(CompetitionDemoError):
    pass


class CompetitionDemoExecutionError(CompetitionDemoError):
    pass


class CompetitionDemoInternalError(CompetitionDemoError):
    pass


@dataclass(frozen=True)
class _RuntimeSnapshot:
    manifest: DemoManifest | None
    corpus: QualifiedCorpusPackage | None
    checks: tuple[CompetitionReadinessCheck, ...]
    verified_documents: frozenset[str]

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(check.ready for check in self.checks)


class CompetitionDemoService:
    """Expose fixed competition cases without accepting authority overrides."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def _check(key: str, label: str, ready: bool, detail: str) -> CompetitionReadinessCheck:
        return CompetitionReadinessCheck(key=key, label=label, ready=ready, detail=detail)

    def _runtime_snapshot(self) -> _RuntimeSnapshot:
        checks: list[CompetitionReadinessCheck] = []
        manifest: DemoManifest | None = None
        corpus: QualifiedCorpusPackage | None = None

        try:
            manifest = load_manifest(MANIFEST_PATH)
        except (OSError, ValueError, CompetitionBootstrapError):
            checks.append(self._check("manifest_authority", "演示清单权威", False, "固定演示清单未通过权威校验。"))
        else:
            checks.append(self._check("manifest_authority", "演示清单权威", True, "固定演示清单已验证。"))

        if manifest is None:
            checks.extend(
                (
                    self._check("corpus_authority", "规范库权威", False, "等待演示清单权威可用。"),
                    self._check("plan_authorities", "施工方案权威", False, "等待演示清单权威可用。"),
                    self._check("case_definitions", "局部审查目标", False, "等待演示清单权威可用。"),
                )
            )
            return _RuntimeSnapshot(None, None, tuple(checks), frozenset())

        try:
            corpus = load_corpus_package(manifest, PROJECT_ROOT)
            repository = StandardRepository(self.settings)
            installed_payload = _repository_payload(repository, manifest.standard.standard_id)
            installed_identity = canonical_sha256(installed_payload)
            packaged_runtime_identity = canonical_sha256(
                corpus.parse_result.model_dump(mode="json")
            )
            if installed_identity != packaged_runtime_identity:
                raise CompetitionBootstrapError("installed corpus identity mismatch")
            document = installed_payload["document"]
            if document["standard_code"] != manifest.standard.standard_code:
                raise CompetitionBootstrapError("installed standard code mismatch")
            if document["source_checksum"] != manifest.standard.source_sha256:
                raise CompetitionBootstrapError("installed source checksum mismatch")
            if len(installed_payload["articles"]) != manifest.standard.qualified_article_count:
                raise CompetitionBootstrapError("installed article count mismatch")
        except (OSError, KeyError, ValueError, CompetitionBootstrapError):
            checks.append(self._check("corpus_authority", "规范库权威", False, "已安装规范库未通过资格化权威校验。"))
        else:
            checks.append(self._check("corpus_authority", "规范库权威", True, "资格化规范库及条文已验证。"))

        verified_documents: set[str] = set()
        for plan in manifest.plans:
            plan_path = self.settings.upload_dir / f"{plan.document_id}.pdf"
            try:
                if not plan_path.is_file() or sha256_file(plan_path) != plan.pdf_sha256:
                    raise CompetitionBootstrapError("plan authority mismatch")
            except (OSError, CompetitionBootstrapError):
                continue
            verified_documents.add(plan.document_id)

        plans_ready = len(verified_documents) == len(manifest.plans)
        checks.append(
            self._check(
                "plan_authorities",
                "施工方案权威",
                plans_ready,
                "两份真实施工方案均已完成字节级验证。" if plans_ready else "施工方案缺失或字节权威不匹配。",
            )
        )

        plan_ids = {plan.document_id for plan in manifest.plans}
        cases_ready = (
            len(manifest.cases) == 3
            and len({case.case_id for case in manifest.cases}) == 3
            and all(case.case_id in CASE_PRESENTATION for case in manifest.cases)
            and all(case.document_id in plan_ids for case in manifest.cases)
            and all(plan.document_id in PLAN_PRESENTATION for plan in manifest.plans)
        )
        checks.append(
            self._check(
                "case_definitions",
                "局部审查目标",
                cases_ready,
                "三个固定局部审查目标已验证。" if cases_ready else "局部审查目标定义无效。",
            )
        )
        return _RuntimeSnapshot(manifest, corpus, tuple(checks), frozenset(verified_documents))

    def metadata(self) -> CompetitionDemoMetadata:
        snapshot = self._runtime_snapshot()
        manifest = snapshot.manifest
        if manifest is None:
            return CompetitionDemoMetadata(
                schema_version="unavailable",
                status="NOT_READY",
                readiness_checks=snapshot.checks,
                plans=(),
                cases=(),
                sample_mode_notice=SAMPLE_MODE_NOTICE,
                scope_notice=DEMO_SCOPE_NOTICE,
            )

        case_counts = {
            plan.document_id: sum(case.document_id == plan.document_id for case in manifest.cases)
            for plan in manifest.plans
        }
        plans = tuple(
            CompetitionPlanSummary(
                document_id=plan.document_id,
                display_name=PLAN_PRESENTATION[plan.document_id],
                pdf_sha256=plan.pdf_sha256,
                verified=plan.document_id in snapshot.verified_documents,
                case_count=case_counts[plan.document_id],
            )
            for plan in manifest.plans
        )
        case_summaries = []
        for case in manifest.cases:
            presentation = CASE_PRESENTATION[case.case_id]
            if isinstance(case, DemoFindingCase):
                case_summaries.append(
                    CompetitionFindingCaseSummary(
                        case_id=case.case_id,
                        label=presentation[0],
                        description=presentation[1],
                        document_id=case.document_id,
                        page_number=case.request.page_number,
                        standard_code=manifest.standard.standard_code,
                        article_number=case.article_number,
                    )
                )
            else:
                case_summaries.append(
                    CompetitionReviewGapCaseSummary(
                        case_id=case.case_id,
                        label=presentation[0],
                        description=presentation[1],
                        document_id=case.document_id,
                        page_number=case.plan_locator.physical_page,
                    )
                )
        return CompetitionDemoMetadata(
            schema_version=manifest.schema_version,
            status="READY" if snapshot.ready else "NOT_READY",
            readiness_checks=snapshot.checks,
            plans=plans,
            cases=tuple(case_summaries),
            sample_mode_notice=SAMPLE_MODE_NOTICE,
            scope_notice=DEMO_SCOPE_NOTICE,
        )

    def run_case(self, case_id: str) -> CompetitionCaseRunResult:
        snapshot = self._runtime_snapshot()
        if snapshot.manifest is None:
            raise CompetitionDemoNotReadyError("competition runtime is not ready")
        case = next((item for item in snapshot.manifest.cases if item.case_id == case_id), None)
        if case is None:
            raise CompetitionDemoCaseNotFoundError("unknown competition case")
        if not snapshot.ready:
            raise CompetitionDemoNotReadyError("competition runtime is not ready")

        if isinstance(case, DemoReviewGapCase):
            return self._run_review_gap_case(case, snapshot.manifest)

        try:
            response = _finding_service(self.settings).create(case.document_id, build_comparison_request(case))
            finding = response.finding
            _assert_live_finding(case, finding)
        except CompetitionBootstrapError as exc:
            raise CompetitionDemoAuthorityDriftError("live finding no longer matches qualified authority") from exc
        except ReviewFindingAuthorityMismatchError as exc:
            raise CompetitionDemoInternalError("C.3 authority invariant failed") from exc
        except (ReviewDocumentNotFoundError, ReviewFindingSourceChangedError) as exc:
            raise CompetitionDemoNotReadyError("verified plan authority changed") from exc
        except (
            ReviewFindingAuthorityUnavailableError,
            ReviewUnitError,
            ComplianceComparisonError,
            PlanFactError,
            RequirementSelectionError,
        ) as exc:
            raise CompetitionDemoExecutionError("live C.3 reconstruction failed safely") from exc
        except Exception as exc:  # pragma: no cover - impossible internal invariant boundary
            raise CompetitionDemoInternalError("unexpected competition invariant failure") from exc
        return self._project_finding(case, snapshot.manifest, finding)

    def _run_review_gap_case(
        self,
        case: DemoReviewGapCase,
        manifest: DemoManifest,
    ) -> CompetitionReviewGapCaseRunResult:
        plan = next(item for item in manifest.plans if item.document_id == case.document_id)
        try:
            workspace = FindingsWorkspaceService(settings=self.settings).build_workspace(
                case.document_id
            )
            gap = select_qualified_review_gap(case, plan, workspace)
        except CompetitionBootstrapError as exc:
            raise CompetitionDemoAuthorityDriftError(
                "live ReviewGap no longer matches qualified plan authority"
            ) from exc
        except ReviewDocumentNotFoundError as exc:
            raise CompetitionDemoNotReadyError("verified plan authority changed") from exc
        except FindingsWorkspaceError as exc:
            raise CompetitionDemoExecutionError(
                "live D.6 reconstruction failed safely"
            ) from exc
        except Exception as exc:  # pragma: no cover - internal invariant boundary
            raise CompetitionDemoInternalError(
                "unexpected D.6 competition invariant failure"
            ) from exc
        return self._project_review_gap(case, manifest, workspace.workspace_id, gap)

    @staticmethod
    def _project_finding(
        case: DemoFindingCase,
        manifest: DemoManifest,
        finding: ReviewFinding,
    ) -> CompetitionFindingCaseRunResult:
        plan = next(item for item in manifest.plans if item.document_id == case.document_id)
        plan_citation = finding.plan_citation
        standard_citation = finding.standard_citation
        plan_facts = tuple(
            CompetitionPlanFactEvidence(
                plan_fact_id=fact.plan_fact_id,
                char_start=fact.char_start,
                char_end=fact.char_end,
                source_text=fact.source_text,
                source_text_sha256=fact.source_text_sha256,
                normalized_value=fact.normalized_value,
                unit=fact.unit,
            )
            for fact in plan_citation.plan_facts
        )
        return CompetitionFindingCaseRunResult(
            message="局部审查结果已生成",
            case_id=case.case_id,
            finding_id=finding.finding_id,
            decision=finding.decision,
            decision_label=DECISION_LABELS[finding.decision],
            reason_code=finding.reason_code,
            decision_scope=finding.decision_scope,
            summary=finding.summary,
            missing_information=tuple(finding.missing_information),
            local_explanation=DECISION_EXPLANATIONS[finding.decision],
            scope_notice=SCOPE_NOTICE,
            plan_evidence=CompetitionPlanEvidence(
                document_display_name=PLAN_PRESENTATION[plan.document_id],
                document_id=plan_citation.document_id,
                pdf_sha256=plan_citation.document_sha256,
                page_number=plan_citation.page_number,
                char_start=plan_citation.char_start,
                char_end=plan_citation.char_end,
                exact_text=plan_citation.source_text,
                text_sha256=plan_citation.source_text_sha256,
                review_unit_id=plan_citation.review_unit_id,
                plan_facts=plan_facts,
            ),
            standard_evidence=CompetitionStandardEvidence(
                standard_id=standard_citation.standard_id,
                standard_code=standard_citation.standard_code,
                article_number=standard_citation.article_number,
                page_start=standard_citation.page_start,
                page_end=standard_citation.page_end,
                exact_requirement_text=standard_citation.requirement_text,
                requirement_text_sha256=standard_citation.requirement_text_sha256,
                official_source_sha256=standard_citation.source_checksum,
                evidence_id=standard_citation.evidence_id,
                requirement_id=standard_citation.requirement_id,
            ),
            technical_provenance=CompetitionTechnicalProvenance(
                finding_id=finding.finding_id,
                comparison_id=finding.comparison_id,
                document_id=finding.document_id,
                document_sha256=finding.document_sha256,
                standard_id=standard_citation.standard_id,
                standard_source_sha256=standard_citation.source_checksum,
                evidence_id=standard_citation.evidence_id,
                requirement_id=standard_citation.requirement_id,
                review_unit_id=plan_citation.review_unit_id,
                plan_fact_ids=tuple(fact.plan_fact_id for fact in plan_citation.plan_facts),
                decision_scope=finding.decision_scope,
            ),
        )

    @staticmethod
    def _project_review_gap(
        case: DemoReviewGapCase,
        manifest: DemoManifest,
        workspace_id: str,
        gap: ReviewGapWorkspaceItem,
    ) -> CompetitionReviewGapCaseRunResult:
        plan = next(item for item in manifest.plans if item.document_id == case.document_id)
        source = gap.plan_source
        return CompetitionReviewGapCaseRunResult(
            message="明确审查缺口已由机器工作区生成",
            case_id=case.case_id,
            review_gap_id=gap.review_gap_id,
            terminal_class=gap.gap_source,
            terminal_status=gap.candidate_terminal_state,
            local_explanation=REVIEW_GAP_EXPLANATION,
            scope_notice=REVIEW_GAP_SCOPE_NOTICE,
            finding_absent=getattr(gap, "finding_id", None) is None,
            comparison_absent=getattr(gap, "comparison_id", None) is None,
            decision_absent=getattr(gap, "decision", None) is None,
            standard_authority_absent=gap.standard_source is None,
            article_authority_absent=gap.standard_source is None,
            requirement_authority_absent=gap.requirement_id is None,
            plan_evidence=CompetitionReviewGapPlanEvidence(
                document_display_name=PLAN_PRESENTATION[plan.document_id],
                document_id=source.document_id,
                pdf_sha256=source.document_sha256,
                physical_page=source.physical_page,
                page_char_start=source.page_char_start,
                page_char_end=source.page_char_end,
                exact_text=source.source_text,
                text_sha256=source.source_text_sha256,
                candidate_id=source.candidate_id,
            ),
            technical_provenance=CompetitionReviewGapTechnicalProvenance(
                workspace_id=workspace_id,
                whole_plan_review_id=gap.whole_plan_review_id,
                review_gap_id=gap.review_gap_id,
                document_id=gap.document_id,
                document_sha256=gap.document_sha256,
                candidate_trace_id=gap.candidate_trace_id,
                candidate_id=gap.candidate_id,
                identity_version=gap.identity_version,
                projection_version=gap.projection_version,
            ),
        )


def get_competition_demo_service() -> CompetitionDemoService:
    return CompetitionDemoService()
