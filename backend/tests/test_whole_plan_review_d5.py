"""D.5 deterministic whole-plan orchestration and authority-boundary tests."""

from __future__ import annotations

import hashlib
import inspect
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.compliance_comparison import (
    ComparisonDecision,
    ComparisonReasonCode,
    PlanFact,
    PlanFactParsingStatus,
    PlanFactType,
)
from app.schemas.compliance_review import (
    ComplianceReviewResult,
    ComplianceReviewStatus,
    ReviewEvidenceBinding,
)
from app.schemas.normative_requirement import (
    NormativeRequirement,
    NumericOperator,
    RequirementAtomicity,
    RequirementExtractionMethod,
    RequirementModality,
)
from app.schemas.plan_fact_extraction import (
    ExtractedPlanFactBinding,
    PlanAssertionKind,
    PlanFactExtractionReason,
    PlanFactExtractionResult,
    PlanFactExtractionStatus,
    deterministic_extraction_id,
    deterministic_measurement_profile_id,
)
from app.schemas.requirement_decomposition import (
    RequirementDecompositionResult,
    RequirementDecompositionStatus,
    UnresolvedRequirementReason,
    UnresolvedRequirementSpan,
    deterministic_decomposition_id,
    deterministic_unresolved_span_id,
)
from app.schemas.review_candidate import (
    CandidateSourceSpan,
    ReviewCandidate,
    ReviewCandidateClass,
    ReviewCandidateStatus,
    deterministic_candidate_id,
)
from app.schemas.review_unit import ReviewUnit
from app.schemas.standard_route import (
    QualifiedStandardScopeCandidate,
    StandardRoute,
    StandardRouteStatus,
    StandardRoutingMethod,
    deterministic_route_id,
)
from app.schemas.standards_scope import StandardScope
from app.schemas.standards_retrieval import RetrievalDecision, RetrievalMethod
from app.schemas.whole_plan_review import (
    WholePlanCandidateTerminalState,
    WholePlanCoverageCounts,
    WholePlanProcessingState,
    WholePlanRequirementTerminalState,
    WholePlanReviewResult,
    WholePlanStage,
)
from app.services.pdf_service import PDFExtractionResult, PDFPageText
from app.services.retrieval.hit_factory import make_hit
from app.services.review.review_unit_service import ReviewUnitService
from app.services.review.whole_plan_review_service import (
    WholePlanAuthorityInvariantError,
    WholePlanCandidateInfrastructureError,
    WholePlanDocumentNotFoundError,
    WholePlanDocumentSourceChangedError,
    WholePlanReviewService,
)
from tests.retrieval_helpers import make_record


STANDARD_ID = "std-d5"
REGISTRY_FINGERPRINT = "d" * 64
STANDARD_CHECKSUM = "c" * 64


class _StaticPDFService:
    def __init__(self, pages: tuple[str, ...]) -> None:
        self.pages = pages

    def extract_text(self, _path: Path) -> PDFExtractionResult:
        page_items = [
            PDFPageText(page_number=index, text=text)
            for index, text in enumerate(self.pages, 1)
        ]
        return PDFExtractionResult(
            page_count=len(page_items),
            char_count=sum(len(item.text) for item in page_items),
            text="\n".join(self.pages),
            pages=page_items,
        )


class _MutatingPDFService(_StaticPDFService):
    def __init__(self, pages: tuple[str, ...], path: Path) -> None:
        super().__init__(pages)
        self.path = path

    def extract_text(self, path: Path) -> PDFExtractionResult:
        result = super().extract_text(path)
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        return result


class _Discovery:
    def __init__(self, candidates) -> None:
        self.candidates = list(candidates)
        self.calls = 0

    def discover(self, *, document_id, document_sha256, pages):
        self.calls += 1
        assert document_id
        assert len(document_sha256) == 64
        assert pages
        return list(self.candidates)


class _Routing:
    def __init__(self, routes, *, failure_stage=None) -> None:
        self.routes = routes
        self.failure_stage = failure_stage
        self.calls = []

    def route(self, candidate):
        self.calls.append(candidate.candidate_id)
        if self.failure_stage and len(self.calls) == 1:
            raise WholePlanCandidateInfrastructureError(self.failure_stage)
        value = self.routes[candidate.candidate_id]
        return value(candidate) if callable(value) else value


class _Review:
    def __init__(self, decision=RetrievalDecision.ACCEPT) -> None:
        self.decision = decision
        self.calls = []

    def review(self, unit, *, top_k=5):
        self.calls.append((unit.review_unit_id, top_k))
        if self.decision != RetrievalDecision.ACCEPT:
            return ComplianceReviewResult(
                review_unit=unit,
                status=ComplianceReviewStatus.INSUFFICIENT_EVIDENCE,
                retrieval_decision=self.decision,
                evidence_bindings=[],
                reason="expected fail-closed retrieval",
            )
        return _accepted_preparation(unit)


class _Decomposition:
    def __init__(self, status=RequirementDecompositionStatus.DECOMPOSED, count=1):
        self.status = status
        self.count = count
        self.calls = []

    def decompose(self, candidate, route, preparation):
        self.calls.append(candidate.candidate_id)
        return _decomposition(candidate, route, preparation, self.status, self.count)


class _Extraction:
    def __init__(
        self,
        statuses=None,
        *,
        decomposition_status=RequirementDecompositionStatus.DECOMPOSED,
        requirement_count=1,
        failure_at=None,
    ) -> None:
        self.statuses = list(statuses or [PlanFactExtractionStatus.EXTRACTED])
        self.decomposition_status = decomposition_status
        self.requirement_count = requirement_count
        self.failure_at = failure_at
        self.calls = []

    def extract(self, candidate, route, preparation, *, requirement_id):
        index = len(self.calls)
        self.calls.append(requirement_id)
        if self.failure_at == index:
            raise WholePlanCandidateInfrastructureError(
                WholePlanStage.PLAN_FACT_EXTRACTION
            )
        status = self.statuses[min(index, len(self.statuses) - 1)]
        return _extraction(
            candidate,
            route,
            preparation,
            requirement_id,
            status,
            self.decomposition_status,
            self.requirement_count,
        )


class _Comparison:
    def __init__(
        self,
        decisions=None,
        *,
        mismatch_requirement=False,
        mismatch_fact=False,
        failure_at=None,
    ) -> None:
        self.decisions = list(decisions or [ComparisonDecision.COMPLIANT])
        self.mismatch_requirement = mismatch_requirement
        self.mismatch_fact = mismatch_fact
        self.failure_at = failure_at
        self.calls = []

    def compare(
        self,
        preparation,
        *,
        evidence_id,
        requirement_selector,
        plan_fact_selectors,
    ):
        index = len(self.calls)
        if self.failure_at == index:
            raise WholePlanCandidateInfrastructureError(
                WholePlanStage.COMPLIANCE_COMPARISON
            )
        available = _requirements(preparation, 2)
        requirement = next(
            item
            for item in available
            if item.requirement_text == requirement_selector.source_text
        )
        requirement_index = available.index(requirement)
        fact = _plan_fact(preparation.review_unit, plan_fact_selectors[0], 0)
        if self.mismatch_requirement:
            requirement = requirement.model_copy(update={"requirement_id": "wrong"})
        if self.mismatch_fact:
            fact = fact.model_copy(update={"plan_fact_id": "wrong"})
        decision = self.decisions[
            min(requirement_index, len(self.decisions) - 1)
        ]
        reason = {
            ComparisonDecision.COMPLIANT: ComparisonReasonCode.NUMERIC_LIMIT_SATISFIED,
            ComparisonDecision.NON_COMPLIANT: ComparisonReasonCode.NUMERIC_LIMIT_VIOLATED,
            ComparisonDecision.INSUFFICIENT_INFORMATION: (
                ComparisonReasonCode.UNSUPPORTED_UNIT
            ),
        }[decision]
        comparison = SimpleNamespace(
            comparison_id=_prefixed_id(
                "comparison_",
                preparation.review_unit.review_unit_id + requirement.requirement_id,
            ),
            decision=decision,
            reason_code=reason,
        )
        response = SimpleNamespace(
            requirement=requirement,
            plan_facts=[fact],
            comparison=comparison,
        )
        self.calls.append(response)
        return response


class _Finding:
    def __init__(self, comparisons: _Comparison, *, mismatch=False, failure_at=None):
        self.comparisons = comparisons
        self.mismatch = mismatch
        self.failure_at = failure_at
        self.calls = []

    def create(self, document_id, request):
        index = len(self.calls)
        if self.failure_at == index:
            raise WholePlanCandidateInfrastructureError(WholePlanStage.REVIEW_FINDING)
        comparison_response = self.comparisons.calls[-1]
        comparison = comparison_response.comparison
        comparison_id = "comparison_wrong" if self.mismatch else comparison.comparison_id
        finding = SimpleNamespace(
            finding_id=_prefixed_id("finding_", comparison.comparison_id),
            comparison_id=comparison_id,
            requirement_id=comparison_response.requirement.requirement_id,
            plan_fact_ids=[comparison_response.plan_facts[0].plan_fact_id],
            authoritative_comparison_response=SimpleNamespace(
                comparison=SimpleNamespace(comparison_id=comparison_id)
            ),
        )
        self.calls.append((document_id, request, finding))
        return SimpleNamespace(finding=finding)


def _prefixed_id(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()


def _stored(tmp_path, page_text="gap shall not exceed 3mm."):
    settings = Settings(upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True)
    document_id = str(uuid4())
    path = settings.upload_dir / f"{document_id}.pdf"
    path.write_bytes(b"qualified-stored-plan")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return settings, document_id, path, sha, (page_text,)


def _candidate(document_id, sha, text, *, start=0, page=1):
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    span = CandidateSourceSpan(
        page_number=page,
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        source_text_sha256=source_hash,
    )
    return ReviewCandidate(
        candidate_id=deterministic_candidate_id(
            document_sha256=sha,
            source_spans=(span,),
            candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
        ),
        document_id=document_id,
        document_sha256=sha,
        source_spans=(span,),
        source_text=text,
        source_text_sha256=source_hash,
        topic="clearance",
        candidate_class=ReviewCandidateClass.NUMERIC_CONTROL,
        confidence=1.0,
        status=ReviewCandidateStatus.DISCOVERED,
    )


def _scope(index=0):
    suffix = chr(ord("a") + index)
    return QualifiedStandardScopeCandidate(
        standard_id=STANDARD_ID + suffix,
        standard_code="STD-" + suffix,
        canonical_standard_code="STD-" + suffix,
        standard_name="Synthetic " + suffix,
        source_checksum=STANDARD_CHECKSUM,
    )


def _route(candidate, status=StandardRouteStatus.ROUTED):
    scopes = {
        StandardRouteStatus.ROUTED: (_scope(),),
        StandardRouteStatus.NO_STANDARD_SCOPE: (),
        StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE: (_scope(), _scope(1)),
    }[status]
    route_id = deterministic_route_id(
        candidate_id=candidate.candidate_id,
        scope_candidates=scopes,
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )
    return StandardRoute(
        route_id=route_id,
        candidate_id=candidate.candidate_id,
        route_status=status,
        candidate_topic=candidate.topic,
        candidate_class=candidate.candidate_class,
        scope_candidates=scopes,
        selected_scope=(
            StandardScope(standard_ids=[scopes[0].standard_id])
            if status == StandardRouteStatus.ROUTED
            else None
        ),
        routing_method=StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES,
        routing_confidence=1.0 if status == StandardRouteStatus.ROUTED else 0.0,
        routing_reason="deterministic test route",
        registry_fingerprint=REGISTRY_FINGERPRINT,
    )


def _accepted_preparation(unit: ReviewUnit) -> ComplianceReviewResult:
    record = make_record(
        unit.standard_scope.standard_ids[0],
        "1.1",
        "gap shall not exceed 2mm; height shall not exceed 4m",
        standard_code="STD-a",
        standard_name="Synthetic a",
        page_start=3,
    )
    evidence = make_hit(
        record,
        rank=1,
        methods=[RetrievalMethod.KEYWORD],
        keyword_score=1.0,
    ).evidence
    binding = ReviewEvidenceBinding(
        review_unit_id=unit.review_unit_id,
        document_id=unit.document_id,
        plan_page_number=unit.page_number,
        plan_char_start=unit.char_start,
        plan_char_end=unit.char_end,
        plan_source_text=unit.source_text,
        plan_source_text_sha256=unit.source_text_sha256,
        retrieval_decision=RetrievalDecision.ACCEPT,
        standard_evidence_id=evidence.id,
        standard_id=evidence.standard_id,
        standard_article_number=evidence.article_number,
        standard_page_start=evidence.source_page_start,
        standard_page_end=evidence.source_page_end,
        evidence=evidence,
    )
    return ComplianceReviewResult(
        review_unit=unit,
        status=ComplianceReviewStatus.NEEDS_COMPARISON,
        retrieval_decision=RetrievalDecision.ACCEPT,
        evidence_bindings=[binding],
        reason="accepted test evidence",
    )


def _requirements(preparation, count):
    evidence = preparation.evidence_bindings[0].evidence
    texts = ("gap shall not exceed 2mm", "height shall not exceed 4m")
    values = (Decimal("2"), Decimal("4"))
    requirements = []
    for index in range(count):
        text = texts[index]
        start = evidence.source_text.index(text)
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        requirements.append(
            NormativeRequirement(
                requirement_id=_prefixed_id("requirement_", evidence.id + str(index)),
                evidence_id=evidence.id,
                standard_id=evidence.standard_id,
                article_id=evidence.article_id,
                article_number=evidence.article_number,
                requirement_char_start=start,
                requirement_char_end=start + len(text),
                requirement_text=text,
                requirement_text_sha256=text_hash,
                modality=RequirementModality.NUMERIC_LIMIT,
                atomicity=RequirementAtomicity.PROVEN,
                operator=NumericOperator.LE,
                value=values[index],
                unit="mm" if index == 0 else "m",
                extraction_method=RequirementExtractionMethod.EXACT_SPAN_DETERMINISTIC,
                extraction_version="test",
            )
        )
    return tuple(requirements)


def _requirement_from_preparation(preparation, index):
    return _requirements(preparation, index + 1)[index]


def _unresolved(preparation):
    evidence = preparation.evidence_bindings[0].evidence
    text = evidence.source_text[:5]
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    values = dict(
        evidence_id=evidence.id,
        article_id=evidence.article_id,
        article_number=evidence.article_number,
        char_start=0,
        char_end=len(text),
        source_text_sha256=source_hash,
        reason=UnresolvedRequirementReason.QUALITATIVE_UNSUPPORTED,
    )
    return UnresolvedRequirementSpan(
        unresolved_span_id=deterministic_unresolved_span_id(**values),
        source_text=text,
        **values,
    )


def _decomposition(candidate, route, preparation, status, count):
    if status == RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED:
        evidence_ids = ()
        requirements = ()
        unresolved = ()
    else:
        evidence_ids = tuple(
            item.standard_evidence_id for item in preparation.evidence_bindings
        )
        requirements = (
            _requirements(preparation, count)
            if status
            in {
                RequirementDecompositionStatus.DECOMPOSED,
                RequirementDecompositionStatus.PARTIAL,
            }
            else ()
        )
        unresolved = (
            (_unresolved(preparation),)
            if status
            in {
                RequirementDecompositionStatus.UNRESOLVED,
                RequirementDecompositionStatus.PARTIAL,
            }
            else ()
        )
    values = dict(
        candidate_id=candidate.candidate_id,
        route_id=route.route_id,
        evidence_ids=evidence_ids,
        requirements=requirements,
        unresolved_spans=unresolved,
    )
    scope = route.scope_candidates[0]
    return RequirementDecompositionResult(
        decomposition_id=deterministic_decomposition_id(**values),
        document_id=candidate.document_id,
        document_sha256=candidate.document_sha256,
        standard_id=scope.standard_id,
        standard_code=scope.standard_code,
        canonical_standard_code=scope.canonical_standard_code,
        standard_name=scope.standard_name,
        status=status,
        **values,
    )


def _plan_fact(unit, selector, index):
    text_hash = hashlib.sha256(selector.source_text.encode()).hexdigest()
    return PlanFact(
        plan_fact_id=_prefixed_id("planfact_", unit.review_unit_id + str(index)),
        review_unit_id=unit.review_unit_id,
        char_start=selector.char_start,
        char_end=selector.char_end,
        source_text=selector.source_text,
        source_text_sha256=text_hash,
        fact_type=PlanFactType.NUMERIC,
        parsing_status=PlanFactParsingStatus.PROVEN,
        normalized_value=Decimal("3"),
        unit="mm",
        parser_method="EXACT_SPAN_DETERMINISTIC",
        parser_version="test",
    )


def _extraction(
    candidate,
    route,
    preparation,
    requirement_id,
    status,
    decomposition_status,
    requirement_count,
):
    binding = None
    reason = None
    if status == PlanFactExtractionStatus.EXTRACTED:
        unit = preparation.review_unit
        selector = SimpleNamespace(
            char_start=0,
            char_end=len(unit.source_text),
            source_text=unit.source_text,
        )
        fact = _plan_fact(unit, selector, 0)
        profile_values = dict(
            profile_name="SYNTHETIC_CLEARANCE",
            measurement_anchors=("gap",),
            object_anchors=("component",),
        )
        binding = ExtractedPlanFactBinding(
            plan_fact=fact,
            review_unit_id=unit.review_unit_id,
            routing_context_id=_prefixed_id("routingcontext_", candidate.candidate_id),
            source_page_text_sha256=candidate.source_text_sha256,
            review_unit_char_start=0,
            review_unit_char_end=len(unit.source_text),
            page_char_start=candidate.source_spans[0].char_start,
            page_char_end=candidate.source_spans[0].char_end,
            source_text=unit.source_text,
            source_text_sha256=candidate.source_text_sha256,
            assertion_kind=PlanAssertionKind.UPPER_BOUND_CONTROL,
            measurement_profile_id=deterministic_measurement_profile_id(
                **profile_values
            ),
            measurement_profile_name=profile_values["profile_name"],
            measurement_anchors=profile_values["measurement_anchors"],
            object_anchors=profile_values["object_anchors"],
        )
    else:
        reason = {
            PlanFactExtractionStatus.NOT_FOUND: PlanFactExtractionReason.NO_NUMERIC_FACT,
            PlanFactExtractionStatus.AMBIGUOUS: (
                PlanFactExtractionReason.MULTIPLE_MATCHING_FACTS
            ),
            PlanFactExtractionStatus.UNSUPPORTED: (
                PlanFactExtractionReason.ASSERTION_UNSUPPORTED
            ),
            PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED: (
                PlanFactExtractionReason.SOURCE_AUTHORITY_FAILED
            ),
        }[status]
    decomposition = _decomposition(
        candidate,
        route,
        preparation,
        decomposition_status,
        requirement_count,
    )
    values = dict(
        candidate_id=candidate.candidate_id,
        document_id=candidate.document_id,
        document_sha256=candidate.document_sha256,
        physical_page=candidate.source_spans[0].page_number,
        review_unit_id=preparation.review_unit.review_unit_id,
        route_id=route.route_id,
        decomposition_id=decomposition.decomposition_id,
        requirement_id=requirement_id,
        status=status,
        binding=binding,
        reason=reason,
    )
    return PlanFactExtractionResult(
        extraction_id=deterministic_extraction_id(**values), **values
    )


def _service(
    tmp_path,
    *,
    candidate_texts=("gap shall not exceed 3mm.",),
    route_statuses=None,
    retrieval=RetrievalDecision.ACCEPT,
    decomposition_status=RequirementDecompositionStatus.DECOMPOSED,
    requirement_count=1,
    extraction_statuses=None,
    decisions=None,
    routing_failure=None,
    extraction_failure_at=None,
    comparison_mismatch_requirement=False,
    comparison_mismatch_fact=False,
    finding_mismatch=False,
    comparison_failure_at=None,
    finding_failure_at=None,
):
    page_text = "\n".join(candidate_texts) or "ordinary plan text"
    settings, document_id, path, sha, pages = _stored(tmp_path, page_text)
    candidates = []
    cursor = 0
    for text in candidate_texts:
        start = page_text.index(text, cursor)
        candidates.append(_candidate(document_id, sha, text, start=start))
        cursor = start + len(text)
    statuses = list(route_statuses or [StandardRouteStatus.ROUTED] * len(candidates))
    routes = {
        candidate.candidate_id: _route(candidate, statuses[index])
        for index, candidate in enumerate(candidates)
    }
    pdf = _StaticPDFService(pages)
    units = ReviewUnitService(settings=settings, pdf_service=pdf)
    review = _Review(retrieval)
    decomposition = _Decomposition(decomposition_status, requirement_count)
    extraction = _Extraction(
        extraction_statuses,
        decomposition_status=decomposition_status,
        requirement_count=requirement_count,
        failure_at=extraction_failure_at,
    )
    comparison = _Comparison(
        decisions,
        mismatch_requirement=comparison_mismatch_requirement,
        mismatch_fact=comparison_mismatch_fact,
        failure_at=comparison_failure_at,
    )
    finding = _Finding(
        comparison, mismatch=finding_mismatch, failure_at=finding_failure_at
    )
    service = WholePlanReviewService(
        settings=settings,
        pdf_service=pdf,
        candidate_discovery_service=_Discovery(candidates),
        routing_service=_Routing(routes, failure_stage=routing_failure),
        review_unit_service=units,
        compliance_review_service=review,
        decomposition_service=decomposition,
        extraction_service=extraction,
        comparison_service=comparison,
        finding_service=finding,
    )
    return SimpleNamespace(
        service=service,
        settings=settings,
        document_id=document_id,
        path=path,
        sha=sha,
        candidates=candidates,
        routes=routes,
        review=review,
        decomposition=decomposition,
        extraction=extraction,
        comparison=comparison,
        finding=finding,
    )


def test_public_seam_accepts_only_document_id():
    signature = inspect.signature(WholePlanReviewService.review_document)
    assert list(signature.parameters) == ["self", "document_id"]


@pytest.mark.parametrize(
    "forbidden",
    ["pages", "candidates", "route", "requirement", "plan_fact", "comparison", "finding"],
)
def test_public_seam_rejects_caller_stage_authority(tmp_path, forbidden):
    env = _service(tmp_path)
    with pytest.raises(TypeError):
        env.service.review_document(env.document_id, **{forbidden: object()})


@pytest.mark.parametrize(
    "model",
    [WholePlanReviewResult, WholePlanCoverageCounts],
)
def test_public_models_are_frozen_and_extra_forbid(model):
    assert model.model_config["frozen"] is True
    assert model.model_config["extra"] == "forbid"


def test_all_d5_models_are_frozen_and_extra_forbid():
    import app.schemas.whole_plan_review as schema

    models = [
        value
        for value in vars(schema).values()
        if inspect.isclass(value)
        and issubclass(value, schema.BaseModel)
        and value.__module__ == schema.__name__
    ]
    assert models
    assert all(model.model_config["frozen"] for model in models)
    assert all(model.model_config["extra"] == "forbid" for model in models)


def test_result_has_no_plan_verdict_or_ratio_fields():
    fields = set(WholePlanReviewResult.model_fields) | set(
        WholePlanCoverageCounts.model_fields
    )
    forbidden = {
        "plan_compliant",
        "plan_non_compliant",
        "overall_verdict",
        "compliance_score",
        "pass_rate",
        "risk_score",
        "coverage_percentage",
    }
    assert fields.isdisjoint(forbidden)


def test_missing_document_fails_closed(tmp_path):
    settings = Settings(upload_dir=tmp_path / "uploads")
    with pytest.raises(WholePlanDocumentNotFoundError):
        WholePlanReviewService(settings=settings).review_document(str(uuid4()))


@pytest.mark.parametrize(
    "document_id",
    ["../escape", "..\\escape", "C:\\escape.pdf", "//server/share", "not-a-uuid"],
)
def test_path_escape_cannot_establish_authority(tmp_path, document_id):
    settings = Settings(upload_dir=tmp_path / "uploads")
    with pytest.raises(WholePlanDocumentNotFoundError):
        WholePlanReviewService(settings=settings).review_document(document_id)


def test_document_mutation_aborts_without_result(tmp_path):
    env = _service(tmp_path, candidate_texts=())
    mutating = _MutatingPDFService(("ordinary plan text",), env.path)
    env.service.pdf_service = mutating
    with pytest.raises(WholePlanDocumentSourceChangedError):
        env.service.review_document(env.document_id)


def test_no_candidate_document_is_completed_not_compliant(tmp_path):
    env = _service(tmp_path, candidate_texts=())
    result = env.service.review_document(env.document_id)
    assert result.processing_state == WholePlanProcessingState.COMPLETED
    assert result.candidate_traces == ()
    assert result.coverage_counts.candidates_discovered == 0
    assert "compliant" not in WholePlanReviewResult.model_fields


@pytest.mark.parametrize(
    ("status", "terminal", "count_field"),
    [
        (
            StandardRouteStatus.NO_STANDARD_SCOPE,
            WholePlanCandidateTerminalState.NO_STANDARD_SCOPE,
            "no_standard_scope",
        ),
        (
            StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE,
            WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE,
            "ambiguous_standard_scope",
        ),
    ],
)
def test_routing_stop_is_preserved(tmp_path, status, terminal, count_field):
    env = _service(tmp_path, route_statuses=[status])
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0]
    assert trace.route.route_status == status
    assert trace.terminal_state == terminal
    assert getattr(result.coverage_counts, count_field) == 1
    assert trace.requirement_traces == ()


def test_routing_stop_does_not_abort_later_candidate(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("first gap 1mm.", "gap shall not exceed 3mm."),
        route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE, StandardRouteStatus.ROUTED],
    )
    result = env.service.review_document(env.document_id)
    assert [item.terminal_state for item in result.candidate_traces] == [
        WholePlanCandidateTerminalState.NO_STANDARD_SCOPE,
        WholePlanCandidateTerminalState.REQUIREMENTS_PROCESSED,
    ]


def test_source_distinct_equal_text_is_not_silently_merged(tmp_path):
    text = "gap shall not exceed 3mm."
    env = _service(tmp_path, candidate_texts=(text, text))
    result = env.service.review_document(env.document_id)
    assert len(result.candidate_traces) == 2
    assert result.candidate_traces[0].candidate.source_spans[0].char_start != (
        result.candidate_traces[1].candidate.source_spans[0].char_start
    )


@pytest.mark.parametrize("decision", [RetrievalDecision.NO_MATCH, RetrievalDecision.LOW_CONFIDENCE])
def test_c1_nonqualified_evidence_is_preserved(tmp_path, decision):
    env = _service(tmp_path, retrieval=decision)
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0]
    assert trace.terminal_state == WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED
    assert trace.retrieval_decision == decision
    assert trace.decomposition is None


@pytest.mark.parametrize(
    ("status", "terminal", "count_field"),
    [
        (
            RequirementDecompositionStatus.UNRESOLVED,
            WholePlanCandidateTerminalState.REQUIREMENT_UNRESOLVED,
            "requirement_unresolved",
        ),
        (
            RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED,
            WholePlanCandidateTerminalState.REQUIREMENT_SOURCE_NOT_QUALIFIED,
            "requirement_source_not_qualified",
        ),
    ],
)
def test_d3_stop_status_is_preserved(tmp_path, status, terminal, count_field):
    env = _service(tmp_path, decomposition_status=status)
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0]
    assert trace.decomposition.status == status
    assert trace.terminal_state == terminal
    assert getattr(result.coverage_counts, count_field) == 1
    assert trace.requirement_traces == ()


def test_d3_partial_processes_proven_and_retains_unresolved(tmp_path):
    env = _service(tmp_path, decomposition_status=RequirementDecompositionStatus.PARTIAL)
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0]
    assert trace.decomposition.unresolved_spans
    assert len(trace.requirement_traces) == 1
    assert result.coverage_counts.requirement_partial == 1


@pytest.mark.parametrize(
    ("status", "terminal", "count_field"),
    [
        (
            PlanFactExtractionStatus.NOT_FOUND,
            WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND,
            "plan_fact_not_found",
        ),
        (
            PlanFactExtractionStatus.AMBIGUOUS,
            WholePlanRequirementTerminalState.PLAN_FACT_AMBIGUOUS,
            "plan_fact_ambiguous",
        ),
        (
            PlanFactExtractionStatus.UNSUPPORTED,
            WholePlanRequirementTerminalState.PLAN_FACT_UNSUPPORTED,
            "plan_fact_unsupported",
        ),
        (
            PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED,
            WholePlanRequirementTerminalState.PLAN_FACT_SOURCE_NOT_QUALIFIED,
            "plan_fact_source_not_qualified",
        ),
    ],
)
def test_d4_stop_status_is_preserved_without_c2_c3(
    tmp_path, status, terminal, count_field
):
    env = _service(tmp_path, extraction_statuses=[status])
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0].requirement_traces[0]
    assert trace.terminal_state == terminal
    assert trace.comparison_id is None
    assert trace.finding_id is None
    assert getattr(result.coverage_counts, count_field) == 1
    assert env.comparison.calls == []
    assert env.finding.calls == []


@pytest.mark.parametrize(
    "decision",
    [
        ComparisonDecision.COMPLIANT,
        ComparisonDecision.NON_COMPLIANT,
        ComparisonDecision.INSUFFICIENT_INFORMATION,
    ],
)
def test_c2_decision_and_c3_finding_are_preserved(tmp_path, decision):
    env = _service(tmp_path, decisions=[decision])
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0].requirement_traces[0]
    assert trace.terminal_state == WholePlanRequirementTerminalState.FINDING_CREATED
    assert trace.comparison_decision == decision
    assert trace.comparison_id
    assert trace.finding_id
    assert result.coverage_counts.comparisons_completed == 1
    assert result.coverage_counts.findings_created == 1


@pytest.mark.parametrize(
    "mismatch",
    ["requirement", "fact"],
)
def test_c2_identity_mismatch_is_authority_invariant(tmp_path, mismatch):
    env = _service(
        tmp_path,
        comparison_mismatch_requirement=mismatch == "requirement",
        comparison_mismatch_fact=mismatch == "fact",
    )
    with pytest.raises(WholePlanAuthorityInvariantError):
        env.service.review_document(env.document_id)


def test_c3_comparison_identity_mismatch_is_authority_invariant(tmp_path):
    env = _service(tmp_path, finding_mismatch=True)
    with pytest.raises(WholePlanAuthorityInvariantError):
        env.service.review_document(env.document_id)


def test_two_requirements_create_two_independent_traces(tmp_path):
    env = _service(
        tmp_path,
        requirement_count=2,
        extraction_statuses=[
            PlanFactExtractionStatus.EXTRACTED,
            PlanFactExtractionStatus.NOT_FOUND,
        ],
        decisions=[ComparisonDecision.COMPLIANT],
    )
    result = env.service.review_document(env.document_id)
    traces = result.candidate_traces[0].requirement_traces
    assert len(traces) == 2
    assert traces[0].terminal_state == WholePlanRequirementTerminalState.FINDING_CREATED
    assert traces[1].terminal_state == WholePlanRequirementTerminalState.PLAN_FACT_NOT_FOUND
    assert traces[0].requirement_id != traces[1].requirement_id


def test_mixed_document_preserves_all_candidate_results(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("first gap 1mm.", "gap shall not exceed 3mm."),
        route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE, StandardRouteStatus.ROUTED],
    )
    result = env.service.review_document(env.document_id)
    assert result.coverage_counts.candidates_discovered == 2
    assert result.coverage_counts.no_standard_scope == 1
    assert result.coverage_counts.findings_created == 1


def test_supported_candidate_local_error_is_recorded_and_later_candidate_continues(
    tmp_path,
):
    env = _service(
        tmp_path,
        candidate_texts=("first gap 1mm.", "gap shall not exceed 3mm."),
        routing_failure=WholePlanStage.ROUTING,
    )
    result = env.service.review_document(env.document_id)
    assert result.processing_state == WholePlanProcessingState.COMPLETED_WITH_INTERNAL_ERRORS
    assert result.candidate_traces[0].terminal_state == WholePlanCandidateTerminalState.INTERNAL_ERROR
    assert result.candidate_traces[1].terminal_state == WholePlanCandidateTerminalState.REQUIREMENTS_PROCESSED
    assert result.coverage_counts.internal_errors == 1


@pytest.mark.parametrize("stage", ["extraction", "comparison", "finding"])
def test_requirement_local_error_is_finite_and_non_authoritative(tmp_path, stage):
    kwargs = {
        "extraction_failure_at": 0 if stage == "extraction" else None,
        "comparison_failure_at": 0 if stage == "comparison" else None,
        "finding_failure_at": 0 if stage == "finding" else None,
    }
    env = _service(tmp_path, **kwargs)
    result = env.service.review_document(env.document_id)
    trace = result.candidate_traces[0].requirement_traces[0]
    assert trace.terminal_state == WholePlanRequirementTerminalState.INTERNAL_ERROR
    assert trace.internal_error is not None
    assert result.processing_state == WholePlanProcessingState.COMPLETED_WITH_INTERNAL_ERRORS
    assert trace.finding_id is None


def test_programming_error_is_not_swallowed(tmp_path):
    env = _service(tmp_path)

    def broken(_candidate):
        raise AssertionError("programming defect")

    env.service.routing_service.route = broken
    with pytest.raises(AssertionError):
        env.service.review_document(env.document_id)


def test_same_authority_produces_same_order_and_id(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("first gap 1mm.", "gap shall not exceed 3mm."),
    )
    first = env.service.review_document(env.document_id)
    second = env.service.review_document(env.document_id)
    assert first == second
    assert first.whole_plan_review_id == second.whole_plan_review_id
    assert [item.candidate_trace_id for item in first.candidate_traces] == [
        item.candidate_trace_id for item in second.candidate_traces
    ]


def test_counts_are_derived_and_cannot_be_tampered(tmp_path):
    env = _service(tmp_path)
    result = env.service.review_document(env.document_id)
    payload = result.model_dump()
    payload["coverage_counts"]["findings_created"] = 0
    with pytest.raises(ValidationError):
        WholePlanReviewResult.model_validate(payload)


def test_result_and_trace_ids_cannot_be_overridden(tmp_path):
    env = _service(tmp_path)
    result = env.service.review_document(env.document_id)
    payload = result.model_dump()
    payload["whole_plan_review_id"] = "wholeplanreview_" + "0" * 64
    with pytest.raises(ValidationError):
        WholePlanReviewResult.model_validate(payload)
    payload = result.model_dump()
    payload["candidate_traces"][0]["candidate_trace_id"] = "candidatetrace_" + "0" * 64
    with pytest.raises(ValidationError):
        WholePlanReviewResult.model_validate(payload)


def test_invalid_extra_field_is_rejected(tmp_path):
    env = _service(tmp_path)
    payload = env.service.review_document(env.document_id).model_dump()
    payload["overall_verdict"] = "COMPLIANT"
    with pytest.raises(ValidationError):
        WholePlanReviewResult.model_validate(payload)


def test_cross_document_candidate_cannot_substitute_same_text(tmp_path):
    env = _service(tmp_path)
    other_id = str(uuid4())
    forged = _candidate(other_id, env.sha, env.candidates[0].source_text)
    env.service.candidate_discovery_service = _Discovery([forged])
    env.service.routing_service = _Routing(
        {forged.candidate_id: _route(forged, StandardRouteStatus.NO_STANDARD_SCOPE)}
    )
    with pytest.raises(ValidationError):
        env.service.review_document(env.document_id)


def test_schema_rejects_processing_state_without_matching_errors(tmp_path):
    env = _service(tmp_path)
    payload = env.service.review_document(env.document_id).model_dump()
    payload["processing_state"] = WholePlanProcessingState.COMPLETED_WITH_INTERNAL_ERRORS
    with pytest.raises(ValidationError):
        WholePlanReviewResult.model_validate(payload)


def test_static_service_has_no_comparison_finding_or_c4_construction_logic():
    source = inspect.getsource(WholePlanReviewService)
    forbidden = (
        "FindingReviewService",
        "reviewer_note",
        "report_inclusion",
        "Celery",
        "Redis",
        "http://",
        "https://",
        "OCR",
        "LLM",
        "PLAN_COMPLIANT",
        "PLAN_NON_COMPLIANT",
    )
    assert all(token not in source for token in forbidden)
    assert "comparison_service.compare" in source
    assert "finding_service.create" in source


def test_production_has_no_competition_specific_authority():
    import app.services.review.whole_plan_review_service as module

    source = inspect.getsource(module)
    forbidden = (
        "CASE-A",
        "CASE-B",
        "CASE-C",
        "GB55023",
        "4.4.15",
        "4.4.16",
        "candidate_697a",
        "candidate_ea698",
    )
    assert all(token not in source for token in forbidden)


def test_case_a_shape_reaches_full_safe_path(tmp_path):
    env = _service(tmp_path, candidate_texts=("insertion length not less than 150mm.",))
    result = env.service.review_document(env.document_id)
    assert result.coverage_counts.plan_fact_extracted == 1
    assert result.coverage_counts.findings_created == 1


def test_case_b_shape_preserves_plan_value_separate_from_normative(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("screw and upright tube gap shall not exceed 3mm.",),
        decisions=[ComparisonDecision.NON_COMPLIANT],
    )
    result = env.service.review_document(env.document_id)
    assert result.coverage_counts.comparison_non_compliant == 1
    extraction = result.candidate_traces[0].requirement_traces[0].extraction
    assert extraction.binding.plan_fact.normalized_value == Decimal("3")


def test_case_c_shape_remains_not_found_without_fabricated_finding(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("vertical spacing 3.60m and horizontal spacing 4.5m.",),
        extraction_statuses=[PlanFactExtractionStatus.NOT_FOUND],
    )
    result = env.service.review_document(env.document_id)
    assert result.coverage_counts.plan_fact_not_found == 1
    assert result.coverage_counts.comparisons_completed == 0
    assert result.coverage_counts.findings_created == 0


def test_no_database_api_background_or_persistence_imports():
    import app.services.review.whole_plan_review_service as module

    source = inspect.getsource(module)
    forbidden = ("fastapi", "sqlalchemy", "Session", "websocket", "background", "commit(")
    assert all(token not in source for token in forbidden)


def test_candidate_source_order_is_preserved(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("first gap 1mm.", "second gap 2mm.", "third gap 3mm."),
        route_statuses=[StandardRouteStatus.NO_STANDARD_SCOPE] * 3,
    )
    result = env.service.review_document(env.document_id)
    offsets = [
        trace.candidate.source_spans[0].char_start for trace in result.candidate_traces
    ]
    assert offsets == sorted(offsets)


def test_reordered_d1_output_is_authority_invariant(tmp_path):
    env = _service(
        tmp_path,
        candidate_texts=("first gap 1mm.", "second gap 2mm."),
    )
    env.service.candidate_discovery_service = _Discovery(list(reversed(env.candidates)))
    with pytest.raises(WholePlanAuthorityInvariantError):
        env.service.review_document(env.document_id)
