"""D.5 synchronous orchestration over frozen source-authority stages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

from app.core.config import Settings, get_settings
from app.schemas.compliance_comparison import (
    ComplianceComparisonRequest,
    SourceSpanSelector,
)
from app.schemas.compliance_review import ComplianceReviewStatus
from app.schemas.plan_fact_extraction import PlanFactExtractionStatus
from app.schemas.requirement_decomposition import RequirementDecompositionStatus
from app.schemas.standard_route import StandardRouteStatus
from app.schemas.standards_retrieval import RetrievalDecision
from app.schemas.whole_plan_review import (
    WHOLE_PLAN_CANDIDATE_TRACE_VERSION,
    WHOLE_PLAN_REQUIREMENT_TRACE_VERSION,
    WHOLE_PLAN_REVIEW_IDENTITY_VERSION,
    WHOLE_PLAN_REVIEW_VERSION,
    WholePlanCandidateTerminalState,
    WholePlanCandidateTrace,
    WholePlanInternalError,
    WholePlanInternalErrorCode,
    WholePlanProcessingState,
    WholePlanRequirementTerminalState,
    WholePlanRequirementTrace,
    WholePlanReviewResult,
    WholePlanStage,
    derive_coverage_counts,
    deterministic_candidate_trace_id,
    deterministic_requirement_trace_id,
    deterministic_whole_plan_review_id,
)
from app.services.pdf_service import PDFService
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.review.candidate_discovery_service import CandidateDiscoveryService
from app.services.review.compliance_comparison_service import ComplianceComparisonService
from app.services.review.compliance_review_service import ComplianceReviewService
from app.services.review.plan_fact_extraction_service import PlanFactExtractionService
from app.services.review.plan_fact_service import PlanFactService
from app.services.review.requirement_decomposition_service import (
    RequirementDecompositionService,
)
from app.services.review.requirement_service import RequirementService
from app.services.review.review_finding_service import ReviewFindingService
from app.services.review.review_unit_service import ReviewUnitService
from app.services.review.routing_context_service import RoutingContextService
from app.services.review.standard_routing_service import StandardRoutingService
from app.services.standards.scope_validation_service import ScopeValidationService
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository


class WholePlanReviewError(RuntimeError):
    pass


class WholePlanDocumentNotFoundError(WholePlanReviewError):
    pass


class WholePlanDocumentSourceChangedError(WholePlanReviewError):
    pass


class WholePlanAuthorityInvariantError(WholePlanReviewError):
    pass


class WholePlanCandidateInfrastructureError(WholePlanReviewError):
    """Explicitly classified candidate-local infrastructure failure for DI/testing."""

    def __init__(self, stage: WholePlanStage) -> None:
        self.stage = stage
        super().__init__(stage.value)


class WholePlanReviewService:
    """Review one server-stored document without creating a plan-level verdict."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        pdf_service: PDFService | None = None,
        candidate_discovery_service: CandidateDiscoveryService | None = None,
        routing_service: StandardRoutingService | None = None,
        review_unit_service: ReviewUnitService | None = None,
        compliance_review_service: ComplianceReviewService | None = None,
        decomposition_service: RequirementDecompositionService | None = None,
        extraction_service: PlanFactExtractionService | None = None,
        comparison_service: ComplianceComparisonService | None = None,
        finding_service: ReviewFindingService | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k < 1 or top_k > 20:
            raise ValueError("D.5 top_k must remain within the frozen C.1 contract.")
        self.settings = settings or get_settings()
        self.pdf_service = pdf_service or PDFService()
        self.candidate_discovery_service = (
            candidate_discovery_service or CandidateDiscoveryService()
        )

        repository = StandardRepository(self.settings)
        registry = StandardRegistryService(self.settings)
        self.routing_service = routing_service or StandardRoutingService(
            repository=repository, registry=registry
        )
        self.review_unit_service = review_unit_service or ReviewUnitService(
            settings=self.settings, pdf_service=self.pdf_service
        )
        search_service = StandardsSearchService(
            repository=repository, registry=registry
        )
        scope_validator = ScopeValidationService(
            repository=repository, registry=registry
        )
        self.compliance_review_service = (
            compliance_review_service
            or ComplianceReviewService(
                search_service=search_service,
                scope_validator=scope_validator,
                review_unit_service=self.review_unit_service,
            )
        )
        requirement_service = RequirementService()
        self.decomposition_service = (
            decomposition_service
            or RequirementDecompositionService(
                repository=repository, requirement_service=requirement_service
            )
        )
        plan_fact_service = PlanFactService(self.review_unit_service)
        self.extraction_service = (
            extraction_service
            or PlanFactExtractionService(
                decomposition_service=self.decomposition_service,
                routing_context_service=RoutingContextService(
                    settings=self.settings, pdf_service=self.pdf_service
                ),
                review_unit_service=self.review_unit_service,
                plan_fact_service=plan_fact_service,
            )
        )
        self.comparison_service = comparison_service or ComplianceComparisonService(
            requirement_service=requirement_service,
            plan_fact_service=plan_fact_service,
        )
        self.finding_service = finding_service or ReviewFindingService(
            settings=self.settings,
            review_unit_service=self.review_unit_service,
            compliance_review_service=self.compliance_review_service,
            comparison_service=self.comparison_service,
        )
        self.top_k = top_k

    def review_document(self, document_id: str) -> WholePlanReviewResult:
        """Reconstruct every stage from one contained server document identity."""

        pdf_path = self._require_stored_pdf(document_id)
        initial_sha = self._sha256(pdf_path)
        extracted = self.pdf_service.extract_text(pdf_path)
        candidates = self.candidate_discovery_service.discover(
            document_id=document_id,
            document_sha256=initial_sha,
            pages=extracted.pages,
        )
        self._validate_candidate_order(candidates)

        traces = tuple(self._review_candidate(candidate) for candidate in candidates)
        if self._sha256(pdf_path) != initial_sha:
            raise WholePlanDocumentSourceChangedError(
                "Stored plan PDF changed during whole-plan review."
            )

        processing_state = (
            WholePlanProcessingState.COMPLETED_WITH_INTERNAL_ERRORS
            if self._contains_internal_error(traces)
            else WholePlanProcessingState.COMPLETED
        )
        counts = derive_coverage_counts(traces)
        review_id = deterministic_whole_plan_review_id(
            document_id=document_id,
            document_sha256=initial_sha,
            processing_state=processing_state,
            candidate_traces=traces,
        )
        return WholePlanReviewResult(
            identity_version=WHOLE_PLAN_REVIEW_IDENTITY_VERSION,
            whole_plan_review_id=review_id,
            document_id=document_id,
            document_sha256=initial_sha,
            orchestrator_version=WHOLE_PLAN_REVIEW_VERSION,
            processing_state=processing_state,
            candidate_traces=traces,
            coverage_counts=counts,
        )

    def _review_candidate(self, candidate) -> WholePlanCandidateTrace:
        route = None
        unit = None
        preparation = None
        decomposition = None
        evidence_ids: tuple[str, ...] = ()
        requirement_traces: tuple[WholePlanRequirementTrace, ...] = ()
        try:
            route = self.routing_service.route(candidate)
            if route.route_status == StandardRouteStatus.NO_STANDARD_SCOPE:
                return self._candidate_trace(
                    candidate=candidate,
                    route=route,
                    terminal_state=WholePlanCandidateTerminalState.NO_STANDARD_SCOPE,
                )
            if route.route_status == StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE:
                return self._candidate_trace(
                    candidate=candidate,
                    route=route,
                    terminal_state=(
                        WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE
                    ),
                )
            if route.route_status != StandardRouteStatus.ROUTED or route.selected_scope is None:
                raise WholePlanAuthorityInvariantError(
                    "D.2 returned an unsupported route contract."
                )

            span = candidate.source_spans[0]
            unit = self.review_unit_service.create(
                document_id=candidate.document_id,
                page_number=span.page_number,
                source_text=candidate.source_text,
                char_start=span.char_start,
                retrieval_query=candidate.source_text,
                standard_scope=route.selected_scope,
            )
            unit = self.review_unit_service.verify(unit)
            self._require_unit_matches_candidate(candidate, unit)

            preparation = self.compliance_review_service.review(
                unit, top_k=self.top_k
            )
            evidence_ids = tuple(
                binding.standard_evidence_id
                for binding in preparation.evidence_bindings
            )
            if (
                preparation.retrieval_decision != RetrievalDecision.ACCEPT
                or preparation.status != ComplianceReviewStatus.NEEDS_COMPARISON
            ):
                return self._candidate_trace(
                    candidate=candidate,
                    route=route,
                    review_unit_id=unit.review_unit_id,
                    preparation_status=preparation.status,
                    retrieval_decision=preparation.retrieval_decision,
                    evidence_ids=evidence_ids,
                    terminal_state=(
                        WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED
                    ),
                )

            decomposition = self.decomposition_service.decompose(
                candidate, route, preparation
            )
            if decomposition.status == RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED:
                return self._candidate_trace(
                    candidate=candidate,
                    route=route,
                    review_unit_id=unit.review_unit_id,
                    preparation_status=preparation.status,
                    retrieval_decision=preparation.retrieval_decision,
                    evidence_ids=evidence_ids,
                    decomposition=decomposition,
                    terminal_state=(
                        WholePlanCandidateTerminalState.REQUIREMENT_SOURCE_NOT_QUALIFIED
                    ),
                )
            if decomposition.status == RequirementDecompositionStatus.UNRESOLVED:
                return self._candidate_trace(
                    candidate=candidate,
                    route=route,
                    review_unit_id=unit.review_unit_id,
                    preparation_status=preparation.status,
                    retrieval_decision=preparation.retrieval_decision,
                    evidence_ids=evidence_ids,
                    decomposition=decomposition,
                    terminal_state=(
                        WholePlanCandidateTerminalState.REQUIREMENT_UNRESOLVED
                    ),
                )

            requirement_traces = tuple(
                self._review_requirement(
                    candidate=candidate,
                    route=route,
                    preparation=preparation,
                    requirement=requirement,
                )
                for requirement in decomposition.requirements
            )
            self._require_requirement_order(decomposition, requirement_traces)
            return self._candidate_trace(
                candidate=candidate,
                route=route,
                review_unit_id=unit.review_unit_id,
                preparation_status=preparation.status,
                retrieval_decision=preparation.retrieval_decision,
                evidence_ids=evidence_ids,
                decomposition=decomposition,
                requirement_traces=requirement_traces,
                terminal_state=WholePlanCandidateTerminalState.REQUIREMENTS_PROCESSED,
            )
        except WholePlanCandidateInfrastructureError as exc:
            return self._candidate_trace(
                candidate=candidate,
                route=route,
                review_unit_id=unit.review_unit_id if unit else None,
                preparation_status=preparation.status if preparation else None,
                retrieval_decision=(
                    preparation.retrieval_decision if preparation else None
                ),
                evidence_ids=evidence_ids,
                decomposition=decomposition,
                requirement_traces=requirement_traces,
                terminal_state=WholePlanCandidateTerminalState.INTERNAL_ERROR,
                internal_error=self._internal_error(exc.stage),
            )

    def _review_requirement(self, *, candidate, route, preparation, requirement):
        extraction = None
        plan_fact_id = None
        comparison_result = None
        try:
            extraction = self.extraction_service.extract(
                candidate,
                route,
                preparation,
                requirement_id=requirement.requirement_id,
            )
            terminal = {
                PlanFactExtractionStatus.NOT_FOUND: (
                    WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND
                ),
                PlanFactExtractionStatus.AMBIGUOUS: (
                    WholePlanRequirementTerminalState.PLAN_FACT_AMBIGUOUS
                ),
                PlanFactExtractionStatus.UNSUPPORTED: (
                    WholePlanRequirementTerminalState.PLAN_FACT_UNSUPPORTED
                ),
                PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED: (
                    WholePlanRequirementTerminalState.PLAN_FACT_SOURCE_NOT_QUALIFIED
                ),
            }.get(extraction.status)
            if terminal is not None:
                return self._requirement_trace(
                    requirement_id=requirement.requirement_id,
                    extraction=extraction,
                    terminal_state=terminal,
                )
            if extraction.status != PlanFactExtractionStatus.EXTRACTED:
                raise WholePlanAuthorityInvariantError(
                    "D.4 returned an unsupported extraction status."
                )
            binding = extraction.binding
            if binding is None:
                raise WholePlanAuthorityInvariantError(
                    "D.4 EXTRACTED result has no verified PlanFact binding."
                )
            plan_fact = binding.plan_fact
            plan_fact_id = plan_fact.plan_fact_id
            requirement_selector = SourceSpanSelector(
                char_start=requirement.requirement_char_start,
                char_end=requirement.requirement_char_end,
                source_text=requirement.requirement_text,
            )
            plan_fact_selector = SourceSpanSelector(
                char_start=plan_fact.char_start,
                char_end=plan_fact.char_end,
                source_text=plan_fact.source_text,
            )
            comparison = self.comparison_service.compare(
                preparation,
                evidence_id=requirement.evidence_id,
                requirement_selector=requirement_selector,
                plan_fact_selectors=[plan_fact_selector],
            )
            self._require_c2_matches_d3_d4(
                comparison, requirement.requirement_id, plan_fact_id
            )
            comparison_result = comparison.comparison
            if comparison_result is None:
                raise WholePlanAuthorityInvariantError(
                    "C.2 returned no authoritative comparison after D.4 extraction."
                )

            request = ComplianceComparisonRequest(
                page_number=preparation.review_unit.page_number,
                source_text=preparation.review_unit.source_text,
                char_start=preparation.review_unit.char_start,
                retrieval_query=preparation.review_unit.retrieval_query,
                standard_ids=preparation.review_unit.standard_scope.standard_ids,
                top_k=self.top_k,
                evidence_id=requirement.evidence_id,
                requirement=requirement_selector,
                plan_facts=[plan_fact_selector],
            )
            finding = self.finding_service.create(candidate.document_id, request).finding
            if (
                finding.comparison_id != comparison_result.comparison_id
                or finding.requirement_id != requirement.requirement_id
                or finding.plan_fact_ids != [plan_fact_id]
                or finding.authoritative_comparison_response.comparison is None
                or finding.authoritative_comparison_response.comparison.comparison_id
                != comparison_result.comparison_id
            ):
                raise WholePlanAuthorityInvariantError(
                    "C.3 reconstruction does not match preceding C.2 authority."
                )
            return self._requirement_trace(
                requirement_id=requirement.requirement_id,
                extraction=extraction,
                plan_fact_id=plan_fact_id,
                comparison_id=comparison_result.comparison_id,
                comparison_decision=comparison_result.decision,
                comparison_reason_code=comparison_result.reason_code,
                finding_id=finding.finding_id,
                terminal_state=WholePlanRequirementTerminalState.FINDING_CREATED,
            )
        except WholePlanCandidateInfrastructureError as exc:
            return self._requirement_trace(
                requirement_id=requirement.requirement_id,
                extraction=extraction,
                plan_fact_id=plan_fact_id,
                comparison_id=(
                    comparison_result.comparison_id if comparison_result else None
                ),
                comparison_decision=(
                    comparison_result.decision if comparison_result else None
                ),
                comparison_reason_code=(
                    comparison_result.reason_code if comparison_result else None
                ),
                terminal_state=WholePlanRequirementTerminalState.INTERNAL_ERROR,
                internal_error=self._internal_error(exc.stage),
            )

    @staticmethod
    def _require_c2_matches_d3_d4(comparison, requirement_id, plan_fact_id) -> None:
        if (
            comparison.requirement is None
            or comparison.requirement.requirement_id != requirement_id
            or len(comparison.plan_facts) != 1
            or comparison.plan_facts[0].plan_fact_id != plan_fact_id
        ):
            raise WholePlanAuthorityInvariantError(
                "C.2 reconstruction does not match D.3/D.4 authority."
            )

    @staticmethod
    def _require_unit_matches_candidate(candidate, unit) -> None:
        span = candidate.source_spans[0]
        if (
            unit.document_id != candidate.document_id
            or unit.page_number != span.page_number
            or unit.char_start != span.char_start
            or unit.char_end != span.char_end
            or unit.source_text != candidate.source_text
            or unit.source_text_sha256 != candidate.source_text_sha256
        ):
            raise WholePlanAuthorityInvariantError(
                "ReviewUnit does not match exact D.1 source authority."
            )

    @staticmethod
    def _require_requirement_order(decomposition, traces) -> None:
        expected = tuple(item.requirement_id for item in decomposition.requirements)
        actual = tuple(item.requirement_id for item in traces)
        if actual != expected or len(set(actual)) != len(actual):
            raise WholePlanAuthorityInvariantError(
                "Requirement traces do not preserve unique D.3 order."
            )

    @staticmethod
    def _validate_candidate_order(candidates) -> None:
        expected = sorted(
            candidates,
            key=lambda item: (
                item.source_spans[0].page_number,
                item.source_spans[0].char_start,
                item.source_spans[0].char_end,
                item.candidate_class.value,
                item.candidate_id,
            ),
        )
        candidate_ids = [item.candidate_id for item in candidates]
        if candidates != expected or len(candidate_ids) != len(set(candidate_ids)):
            raise WholePlanAuthorityInvariantError(
                "D.1 candidates are not uniquely source ordered."
            )

    def _require_stored_pdf(self, document_id: str) -> Path:
        try:
            parsed = UUID(document_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise WholePlanDocumentNotFoundError(
                "Stored plan document was not found."
            ) from exc
        if str(parsed) != document_id:
            raise WholePlanDocumentNotFoundError("Stored plan document was not found.")
        upload_root = self.settings.upload_dir.resolve()
        path = (upload_root / f"{document_id}.pdf").resolve()
        if path.parent != upload_root or not path.is_file():
            raise WholePlanDocumentNotFoundError("Stored plan document was not found.")
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise WholePlanDocumentSourceChangedError(
                "Stored plan PDF could not be read consistently."
            ) from exc
        return digest.hexdigest()

    @staticmethod
    def _contains_internal_error(traces) -> bool:
        return any(
            trace.internal_error is not None
            or any(item.internal_error is not None for item in trace.requirement_traces)
            for trace in traces
        )

    @staticmethod
    def _internal_error(stage: WholePlanStage) -> WholePlanInternalError:
        return WholePlanInternalError(
            stage=stage,
            code=WholePlanInternalErrorCode.CANDIDATE_LOCAL_INFRASTRUCTURE,
        )

    @staticmethod
    def _candidate_trace(**values) -> WholePlanCandidateTrace:
        defaults = {
            "route": None,
            "review_unit_id": None,
            "preparation_status": None,
            "retrieval_decision": None,
            "evidence_ids": (),
            "decomposition": None,
            "requirement_traces": (),
            "internal_error": None,
        }
        for key, default in defaults.items():
            values.setdefault(key, default)
        trace_id = deterministic_candidate_trace_id(
            trace_version=WHOLE_PLAN_CANDIDATE_TRACE_VERSION,
            **values,
        )
        return WholePlanCandidateTrace(
            candidate_trace_id=trace_id,
            trace_version=WHOLE_PLAN_CANDIDATE_TRACE_VERSION,
            **values,
        )

    @staticmethod
    def _requirement_trace(**values) -> WholePlanRequirementTrace:
        defaults = {
            "extraction": None,
            "plan_fact_id": None,
            "comparison_id": None,
            "comparison_decision": None,
            "comparison_reason_code": None,
            "finding_id": None,
            "internal_error": None,
        }
        for key, default in defaults.items():
            values.setdefault(key, default)
        trace_id = deterministic_requirement_trace_id(
            trace_version=WHOLE_PLAN_REQUIREMENT_TRACE_VERSION,
            **values,
        )
        return WholePlanRequirementTrace(
            requirement_trace_id=trace_id,
            trace_version=WHOLE_PLAN_REQUIREMENT_TRACE_VERSION,
            **values,
        )
