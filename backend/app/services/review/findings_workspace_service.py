"""D.6 read-only projection of one server-created D.5 review result."""

from __future__ import annotations

import hashlib

from app.core.config import Settings, get_settings
from app.schemas.findings_workspace import (
    FINDINGS_WORKSPACE_IDENTITY_VERSION,
    FINDINGS_WORKSPACE_PROJECTION_VERSION,
    FindingWorkspaceItem,
    FindingsWorkspaceResult,
    InternalErrorWorkspaceItem,
    PlanSourceLocator,
    ReviewGapSource,
    ReviewGapWorkspaceItem,
    StandardSourceLocator,
    derive_workspace_counts,
    deterministic_internal_error_item_id,
    deterministic_review_gap_id,
    deterministic_workspace_id,
    workspace_item_sort_key,
)
from app.schemas.requirement_decomposition import RequirementDecompositionStatus
from app.schemas.standard_route import StandardRouteStatus
from app.schemas.whole_plan_review import (
    WholePlanCandidateTerminalState,
    WholePlanRequirementTerminalState,
    WholePlanReviewResult,
)
from app.services.review.whole_plan_review_service import WholePlanReviewService
from app.services.standards.standard_repository import StandardRepository


class FindingsWorkspaceError(RuntimeError):
    pass


class FindingsWorkspaceAuthorityInvariantError(FindingsWorkspaceError):
    pass


class FindingsWorkspaceSourceNavigationError(FindingsWorkspaceError):
    pass


_CANDIDATE_GAP_STATES = {
    WholePlanCandidateTerminalState.NO_STANDARD_SCOPE,
    WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE,
    WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED,
    WholePlanCandidateTerminalState.REQUIREMENT_SOURCE_NOT_QUALIFIED,
}
_REQUIREMENT_GAP_STATES = {
    WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND,
    WholePlanRequirementTerminalState.PLAN_FACT_AMBIGUOUS,
    WholePlanRequirementTerminalState.PLAN_FACT_UNSUPPORTED,
    WholePlanRequirementTerminalState.PLAN_FACT_SOURCE_NOT_QUALIFIED,
}


class FindingsWorkspaceService:
    """Build a complete workspace from one server-authoritative document ID."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        whole_plan_review_service: WholePlanReviewService | None = None,
        standard_repository: StandardRepository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.whole_plan_review_service = (
            whole_plan_review_service or WholePlanReviewService(settings=self.settings)
        )
        self.standard_repository = standard_repository or StandardRepository(self.settings)

    def build_workspace(self, document_id: str) -> FindingsWorkspaceResult:
        """Invoke frozen D.5 exactly once, then project its complete result."""

        review = self.whole_plan_review_service.review_document(document_id)
        if review.document_id != document_id:
            raise FindingsWorkspaceAuthorityInvariantError(
                "D.5 returned authority for another document."
            )
        return self._project_workspace(review)

    def _project_workspace(
        self, review: WholePlanReviewResult
    ) -> FindingsWorkspaceResult:
        items = []
        standard_cache: dict[tuple[str, str], tuple[object, dict[str, object]]] = {}
        finding_ids: set[str] = set()

        for candidate_trace in review.candidate_traces:
            plan_source = self._candidate_plan_source(candidate_trace)

            if candidate_trace.internal_error is not None:
                error_id = deterministic_internal_error_item_id(
                    whole_plan_review_id=review.whole_plan_review_id,
                    document_id=review.document_id,
                    document_sha256=review.document_sha256,
                    candidate_trace_id=candidate_trace.candidate_trace_id,
                    requirement_trace_id=None,
                    error=candidate_trace.internal_error,
                )
                items.append(
                    InternalErrorWorkspaceItem(
                        internal_error_item_id=error_id,
                        whole_plan_review_id=review.whole_plan_review_id,
                        document_id=review.document_id,
                        document_sha256=review.document_sha256,
                        candidate_trace_id=candidate_trace.candidate_trace_id,
                        candidate_id=candidate_trace.candidate.candidate_id,
                        error=candidate_trace.internal_error,
                        plan_source=plan_source,
                    )
                )
                continue

            if candidate_trace.terminal_state in _CANDIDATE_GAP_STATES:
                items.append(
                    self._candidate_gap(review, candidate_trace, plan_source)
                )
                continue

            decomposition = candidate_trace.decomposition
            if decomposition is None:
                raise FindingsWorkspaceAuthorityInvariantError(
                    "Post-routing D.5 trace lacks D.3 authority."
                )

            if decomposition.status in {
                RequirementDecompositionStatus.UNRESOLVED,
                RequirementDecompositionStatus.PARTIAL,
            }:
                for unresolved in decomposition.unresolved_spans:
                    standard_source = self._standard_source(
                        candidate_trace,
                        source=unresolved,
                        requirement_id=None,
                        unresolved_span_id=unresolved.unresolved_span_id,
                        cache=standard_cache,
                    )
                    gap_id = deterministic_review_gap_id(
                        whole_plan_review_id=review.whole_plan_review_id,
                        document_id=review.document_id,
                        document_sha256=review.document_sha256,
                        candidate_trace_id=candidate_trace.candidate_trace_id,
                        requirement_trace_id=None,
                        unresolved_span_id=unresolved.unresolved_span_id,
                        gap_source=ReviewGapSource.UNRESOLVED_SPAN,
                        candidate_terminal_state=None,
                        requirement_terminal_state=None,
                        decomposition_status=decomposition.status,
                        unresolved_reason=unresolved.reason,
                    )
                    items.append(
                        ReviewGapWorkspaceItem(
                            review_gap_id=gap_id,
                            whole_plan_review_id=review.whole_plan_review_id,
                            document_id=review.document_id,
                            document_sha256=review.document_sha256,
                            candidate_trace_id=candidate_trace.candidate_trace_id,
                            candidate_id=candidate_trace.candidate.candidate_id,
                            unresolved_span_id=unresolved.unresolved_span_id,
                            gap_source=ReviewGapSource.UNRESOLVED_SPAN,
                            decomposition_status=decomposition.status,
                            unresolved_reason=unresolved.reason,
                            plan_source=plan_source,
                            standard_source=standard_source,
                        )
                    )

            requirements = {item.requirement_id: item for item in decomposition.requirements}
            for requirement_trace in candidate_trace.requirement_traces:
                requirement = requirements.get(requirement_trace.requirement_id)
                if requirement is None:
                    raise FindingsWorkspaceAuthorityInvariantError(
                        "D.5 requirement trace is absent from D.3 authority."
                    )
                standard_source = self._standard_source(
                    candidate_trace,
                    source=requirement,
                    requirement_id=requirement.requirement_id,
                    unresolved_span_id=None,
                    cache=standard_cache,
                )
                if requirement_trace.internal_error is not None:
                    error_id = deterministic_internal_error_item_id(
                        whole_plan_review_id=review.whole_plan_review_id,
                        document_id=review.document_id,
                        document_sha256=review.document_sha256,
                        candidate_trace_id=candidate_trace.candidate_trace_id,
                        requirement_trace_id=requirement_trace.requirement_trace_id,
                        error=requirement_trace.internal_error,
                    )
                    items.append(
                        InternalErrorWorkspaceItem(
                            internal_error_item_id=error_id,
                            whole_plan_review_id=review.whole_plan_review_id,
                            document_id=review.document_id,
                            document_sha256=review.document_sha256,
                            candidate_trace_id=candidate_trace.candidate_trace_id,
                            candidate_id=candidate_trace.candidate.candidate_id,
                            requirement_trace_id=requirement_trace.requirement_trace_id,
                            requirement_id=requirement.requirement_id,
                            error=requirement_trace.internal_error,
                            plan_source=plan_source,
                        )
                    )
                elif requirement_trace.terminal_state in _REQUIREMENT_GAP_STATES:
                    gap_id = deterministic_review_gap_id(
                        whole_plan_review_id=review.whole_plan_review_id,
                        document_id=review.document_id,
                        document_sha256=review.document_sha256,
                        candidate_trace_id=candidate_trace.candidate_trace_id,
                        requirement_trace_id=requirement_trace.requirement_trace_id,
                        unresolved_span_id=None,
                        gap_source=ReviewGapSource.REQUIREMENT_TERMINAL,
                        candidate_terminal_state=None,
                        requirement_terminal_state=requirement_trace.terminal_state,
                        decomposition_status=None,
                        unresolved_reason=None,
                    )
                    items.append(
                        ReviewGapWorkspaceItem(
                            review_gap_id=gap_id,
                            whole_plan_review_id=review.whole_plan_review_id,
                            document_id=review.document_id,
                            document_sha256=review.document_sha256,
                            candidate_trace_id=candidate_trace.candidate_trace_id,
                            candidate_id=candidate_trace.candidate.candidate_id,
                            requirement_trace_id=requirement_trace.requirement_trace_id,
                            requirement_id=requirement.requirement_id,
                            gap_source=ReviewGapSource.REQUIREMENT_TERMINAL,
                            requirement_terminal_state=requirement_trace.terminal_state,
                            plan_source=plan_source,
                            standard_source=standard_source,
                        )
                    )
                elif requirement_trace.terminal_state == (
                    WholePlanRequirementTerminalState.FINDING_CREATED
                ):
                    if requirement_trace.finding_id in finding_ids:
                        raise FindingsWorkspaceAuthorityInvariantError(
                            "Duplicate C.3 finding identity in D.5 authority."
                        )
                    finding_ids.add(requirement_trace.finding_id)
                    extraction = requirement_trace.extraction
                    if extraction is None or extraction.binding is None:
                        raise FindingsWorkspaceAuthorityInvariantError(
                            "Finding trace lacks verified D.4 source authority."
                        )
                    binding = extraction.binding
                    finding_plan_source = PlanSourceLocator(
                        document_id=review.document_id,
                        document_sha256=review.document_sha256,
                        candidate_id=candidate_trace.candidate.candidate_id,
                        physical_page=extraction.physical_page,
                        page_char_start=binding.page_char_start,
                        page_char_end=binding.page_char_end,
                        source_text=binding.source_text,
                        source_text_sha256=binding.source_text_sha256,
                        review_unit_id=binding.review_unit_id,
                        plan_fact_id=binding.plan_fact.plan_fact_id,
                    )
                    items.append(
                        FindingWorkspaceItem(
                            finding_id=requirement_trace.finding_id,
                            comparison_id=requirement_trace.comparison_id,
                            decision=requirement_trace.comparison_decision,
                            reason_code=requirement_trace.comparison_reason_code,
                            candidate_trace_id=candidate_trace.candidate_trace_id,
                            candidate_id=candidate_trace.candidate.candidate_id,
                            requirement_trace_id=requirement_trace.requirement_trace_id,
                            requirement_id=requirement.requirement_id,
                            plan_fact_id=requirement_trace.plan_fact_id,
                            route_id=candidate_trace.route.route_id,
                            plan_source=finding_plan_source,
                            standard_source=standard_source,
                        )
                    )
                else:
                    raise FindingsWorkspaceAuthorityInvariantError(
                        "Unsupported D.5 requirement terminal state."
                    )

        ordered = tuple(sorted(items, key=workspace_item_sort_key))
        counts = derive_workspace_counts(ordered)
        workspace_id = deterministic_workspace_id(
            whole_plan_review_id=review.whole_plan_review_id,
            items=ordered,
        )
        return FindingsWorkspaceResult(
            identity_version=FINDINGS_WORKSPACE_IDENTITY_VERSION,
            projection_version=FINDINGS_WORKSPACE_PROJECTION_VERSION,
            workspace_id=workspace_id,
            whole_plan_review_id=review.whole_plan_review_id,
            document_id=review.document_id,
            document_sha256=review.document_sha256,
            processing_state=review.processing_state,
            items=ordered,
            counts=counts,
            source_coverage_counts=review.coverage_counts,
        )

    @staticmethod
    def _candidate_plan_source(candidate_trace) -> PlanSourceLocator:
        candidate = candidate_trace.candidate
        if len(candidate.source_spans) != 1:
            raise FindingsWorkspaceAuthorityInvariantError(
                "D.6 requires the frozen single exact D.1 source span."
            )
        span = candidate.source_spans[0]
        return PlanSourceLocator(
            document_id=candidate.document_id,
            document_sha256=candidate.document_sha256,
            candidate_id=candidate.candidate_id,
            physical_page=span.page_number,
            page_char_start=span.char_start,
            page_char_end=span.char_end,
            source_text=candidate.source_text,
            source_text_sha256=candidate.source_text_sha256,
        )

    @staticmethod
    def _candidate_gap(review, candidate_trace, plan_source):
        gap_id = deterministic_review_gap_id(
            whole_plan_review_id=review.whole_plan_review_id,
            document_id=review.document_id,
            document_sha256=review.document_sha256,
            candidate_trace_id=candidate_trace.candidate_trace_id,
            requirement_trace_id=None,
            unresolved_span_id=None,
            gap_source=ReviewGapSource.CANDIDATE_TERMINAL,
            candidate_terminal_state=candidate_trace.terminal_state,
            requirement_terminal_state=None,
            decomposition_status=None,
            unresolved_reason=None,
        )
        return ReviewGapWorkspaceItem(
            review_gap_id=gap_id,
            whole_plan_review_id=review.whole_plan_review_id,
            document_id=review.document_id,
            document_sha256=review.document_sha256,
            candidate_trace_id=candidate_trace.candidate_trace_id,
            candidate_id=candidate_trace.candidate.candidate_id,
            gap_source=ReviewGapSource.CANDIDATE_TERMINAL,
            candidate_terminal_state=candidate_trace.terminal_state,
            plan_source=plan_source,
        )

    def _standard_source(
        self,
        candidate_trace,
        *,
        source,
        requirement_id: str | None,
        unresolved_span_id: str | None,
        cache: dict,
    ) -> StandardSourceLocator:
        route = candidate_trace.route
        decomposition = candidate_trace.decomposition
        if (
            route is None
            or route.route_status != StandardRouteStatus.ROUTED
            or len(route.scope_candidates) != 1
            or decomposition is None
        ):
            raise FindingsWorkspaceSourceNavigationError(
                "Standard navigation requires one frozen routed scope."
            )
        scope = route.scope_candidates[0]
        if scope.standard_id != decomposition.standard_id:
            raise FindingsWorkspaceSourceNavigationError(
                "D.2 and D.3 standard identities differ."
            )
        key = (scope.standard_id, scope.source_checksum)
        cached = cache.get(key)
        if cached is None:
            document = self.standard_repository.load_document(scope.standard_id)
            if (
                document is None
                or document.source_checksum != scope.source_checksum
                or document.standard_code != scope.standard_code
                or document.canonical_standard_code != scope.canonical_standard_code
            ):
                raise FindingsWorkspaceSourceNavigationError(
                    "Qualified standard metadata/checksum mismatch."
                )
            articles = self.standard_repository.load_articles(scope.standard_id)
            by_id = {item.article_id: item for item in articles}
            if len(by_id) != len(articles):
                raise FindingsWorkspaceSourceNavigationError(
                    "Duplicate stored standard article identity."
                )
            cached = (document, by_id)
            cache[key] = cached
        document, by_id = cached
        article = by_id.get(source.article_id)
        if (
            article is None
            or article.standard_id != scope.standard_id
            or article.article_number != source.article_number
        ):
            raise FindingsWorkspaceSourceNavigationError(
                "Qualified standard article identity mismatch."
            )
        start = (
            source.requirement_char_start
            if requirement_id is not None
            else source.char_start
        )
        end = (
            source.requirement_char_end
            if requirement_id is not None
            else source.char_end
        )
        text = (
            source.requirement_text
            if requirement_id is not None
            else source.source_text
        )
        text_sha = (
            source.requirement_text_sha256
            if requirement_id is not None
            else source.source_text_sha256
        )
        if end != start + len(text) or article.source_text[start:end] != text:
            raise FindingsWorkspaceSourceNavigationError(
                "Exact standard article source span mismatch."
            )
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_sha:
            raise FindingsWorkspaceSourceNavigationError(
                "Exact standard source hash mismatch."
            )
        return StandardSourceLocator(
            standard_id=scope.standard_id,
            standard_code=scope.standard_code,
            canonical_standard_code=scope.canonical_standard_code,
            source_checksum=document.source_checksum,
            article_id=source.article_id,
            article_number=source.article_number,
            source_page_start=article.source_page_start,
            source_page_end=article.source_page_end,
            evidence_id=source.evidence_id,
            requirement_id=requirement_id,
            unresolved_span_id=unresolved_span_id,
            article_char_start=start,
            article_char_end=end,
            source_text=text,
            source_text_sha256=text_sha,
        )
