"""Deterministic, non-verdict contracts for D.5 whole-plan orchestration."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.compliance_comparison import ComparisonDecision, ComparisonReasonCode
from app.schemas.compliance_review import ComplianceReviewStatus
from app.schemas.plan_fact_extraction import (
    PlanFactExtractionResult,
    PlanFactExtractionStatus,
)
from app.schemas.requirement_decomposition import (
    RequirementDecompositionResult,
    RequirementDecompositionStatus,
)
from app.schemas.review_candidate import ReviewCandidate
from app.schemas.standard_route import StandardRoute, StandardRouteStatus
from app.schemas.standards_retrieval import RetrievalDecision


WHOLE_PLAN_REVIEW_IDENTITY_VERSION = "v0.1-d.5-whole-plan-review"
WHOLE_PLAN_REVIEW_VERSION = "v0.1-d.5-synchronous-frozen-stage-orchestration"
WHOLE_PLAN_CANDIDATE_TRACE_VERSION = "v0.1-d.5-candidate-trace"
WHOLE_PLAN_REQUIREMENT_TRACE_VERSION = "v0.1-d.5-requirement-trace"


class WholePlanProcessingState(str, Enum):
    """Execution completeness only; never a plan-compliance conclusion."""

    COMPLETED = "COMPLETED"
    COMPLETED_WITH_INTERNAL_ERRORS = "COMPLETED_WITH_INTERNAL_ERRORS"


class WholePlanCandidateTerminalState(str, Enum):
    NO_STANDARD_SCOPE = "NO_STANDARD_SCOPE"
    AMBIGUOUS_STANDARD_SCOPE = "AMBIGUOUS_STANDARD_SCOPE"
    EVIDENCE_NOT_QUALIFIED = "EVIDENCE_NOT_QUALIFIED"
    REQUIREMENT_SOURCE_NOT_QUALIFIED = "REQUIREMENT_SOURCE_NOT_QUALIFIED"
    REQUIREMENT_UNRESOLVED = "REQUIREMENT_UNRESOLVED"
    REQUIREMENTS_PROCESSED = "REQUIREMENTS_PROCESSED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class WholePlanRequirementTerminalState(str, Enum):
    PLAN_FACT_NOT_FOUND = "PLAN_FACT_NOT_FOUND"
    PLAN_FACT_AMBIGUOUS = "PLAN_FACT_AMBIGUOUS"
    PLAN_FACT_UNSUPPORTED = "PLAN_FACT_UNSUPPORTED"
    PLAN_FACT_SOURCE_NOT_QUALIFIED = "PLAN_FACT_SOURCE_NOT_QUALIFIED"
    FINDING_CREATED = "FINDING_CREATED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class WholePlanStage(str, Enum):
    ROUTING = "ROUTING"
    REVIEW_UNIT = "REVIEW_UNIT"
    EVIDENCE_PREPARATION = "EVIDENCE_PREPARATION"
    REQUIREMENT_DECOMPOSITION = "REQUIREMENT_DECOMPOSITION"
    PLAN_FACT_EXTRACTION = "PLAN_FACT_EXTRACTION"
    COMPLIANCE_COMPARISON = "COMPLIANCE_COMPARISON"
    REVIEW_FINDING = "REVIEW_FINDING"


class WholePlanInternalErrorCode(str, Enum):
    CANDIDATE_LOCAL_INFRASTRUCTURE = "CANDIDATE_LOCAL_INFRASTRUCTURE"


class WholePlanInternalError(BaseModel):
    """A stable non-authoritative error classification without exception text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: WholePlanStage
    code: WholePlanInternalErrorCode


def _identity(prefix: str, payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(canonical).hexdigest()


def deterministic_requirement_trace_id(
    *,
    requirement_id: str,
    extraction: PlanFactExtractionResult | None,
    plan_fact_id: str | None,
    comparison_id: str | None,
    comparison_decision: ComparisonDecision | None,
    comparison_reason_code: ComparisonReasonCode | None,
    finding_id: str | None,
    terminal_state: WholePlanRequirementTerminalState,
    internal_error: WholePlanInternalError | None,
    trace_version: str = WHOLE_PLAN_REQUIREMENT_TRACE_VERSION,
) -> str:
    return _identity(
        "requirementtrace_",
        {
            "trace_version": trace_version,
            "requirement_id": requirement_id,
            "extraction_id": extraction.extraction_id if extraction else None,
            "extraction_status": extraction.status.value if extraction else None,
            "plan_fact_id": plan_fact_id,
            "comparison_id": comparison_id,
            "comparison_decision": (
                comparison_decision.value if comparison_decision else None
            ),
            "comparison_reason_code": (
                comparison_reason_code.value if comparison_reason_code else None
            ),
            "finding_id": finding_id,
            "terminal_state": terminal_state.value,
            "internal_error": (
                internal_error.model_dump(mode="json") if internal_error else None
            ),
        },
    )


class WholePlanRequirementTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_version: Literal[WHOLE_PLAN_REQUIREMENT_TRACE_VERSION] = (
        WHOLE_PLAN_REQUIREMENT_TRACE_VERSION
    )
    requirement_trace_id: str = Field(pattern=r"^requirementtrace_[0-9a-f]{64}$")
    requirement_id: str = Field(min_length=1)
    extraction: PlanFactExtractionResult | None = None
    plan_fact_id: str | None = None
    comparison_id: str | None = None
    comparison_decision: ComparisonDecision | None = None
    comparison_reason_code: ComparisonReasonCode | None = None
    finding_id: str | None = None
    terminal_state: WholePlanRequirementTerminalState
    internal_error: WholePlanInternalError | None = None

    @model_validator(mode="after")
    def validate_stage_contract(self):
        comparison_fields = (
            self.comparison_id,
            self.comparison_decision,
            self.comparison_reason_code,
        )
        if any(value is None for value in comparison_fields) and any(
            value is not None for value in comparison_fields
        ):
            raise ValueError("Comparison identity, decision and reason must be complete.")

        if self.terminal_state == WholePlanRequirementTerminalState.INTERNAL_ERROR:
            if self.internal_error is None or self.finding_id is not None:
                raise ValueError("INTERNAL_ERROR requires one stable error and no finding.")
        else:
            if self.internal_error is not None or self.extraction is None:
                raise ValueError("A normal requirement trace requires D.4 authority only.")
            expected_terminal = {
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
                PlanFactExtractionStatus.EXTRACTED: (
                    WholePlanRequirementTerminalState.FINDING_CREATED
                ),
            }[self.extraction.status]
            if self.terminal_state != expected_terminal:
                raise ValueError("Requirement terminal state contradicts D.4 status.")
            if self.extraction.requirement_id != self.requirement_id:
                raise ValueError("D.4 requirement identity does not match its trace.")
            if self.extraction.status == PlanFactExtractionStatus.EXTRACTED:
                binding = self.extraction.binding
                if binding is None:
                    raise ValueError("EXTRACTED D.4 authority requires a binding.")
                if (
                    self.plan_fact_id != binding.plan_fact.plan_fact_id
                    or any(value is None for value in comparison_fields)
                    or self.finding_id is None
                ):
                    raise ValueError("A completed requirement requires D.4/C.2/C.3 identities.")
            elif any(
                value is not None
                for value in (
                    self.plan_fact_id,
                    *comparison_fields,
                    self.finding_id,
                )
            ):
                raise ValueError("A stopped D.4 branch cannot expose downstream authority.")

        expected = deterministic_requirement_trace_id(
            requirement_id=self.requirement_id,
            extraction=self.extraction,
            plan_fact_id=self.plan_fact_id,
            comparison_id=self.comparison_id,
            comparison_decision=self.comparison_decision,
            comparison_reason_code=self.comparison_reason_code,
            finding_id=self.finding_id,
            terminal_state=self.terminal_state,
            internal_error=self.internal_error,
            trace_version=self.trace_version,
        )
        if self.requirement_trace_id != expected:
            raise ValueError("requirement_trace_id does not match canonical identity.")
        return self


def deterministic_candidate_trace_id(
    *,
    candidate: ReviewCandidate,
    route: StandardRoute | None,
    review_unit_id: str | None,
    preparation_status: ComplianceReviewStatus | None,
    retrieval_decision: RetrievalDecision | None,
    evidence_ids: tuple[str, ...],
    decomposition: RequirementDecompositionResult | None,
    requirement_traces: tuple[WholePlanRequirementTrace, ...],
    terminal_state: WholePlanCandidateTerminalState,
    internal_error: WholePlanInternalError | None,
    trace_version: str = WHOLE_PLAN_CANDIDATE_TRACE_VERSION,
) -> str:
    return _identity(
        "candidatetrace_",
        {
            "trace_version": trace_version,
            "candidate_id": candidate.candidate_id,
            "route_id": route.route_id if route else None,
            "route_status": route.route_status.value if route else None,
            "review_unit_id": review_unit_id,
            "preparation_status": (
                preparation_status.value if preparation_status else None
            ),
            "retrieval_decision": retrieval_decision.value if retrieval_decision else None,
            "ordered_evidence_ids": list(evidence_ids),
            "decomposition_id": decomposition.decomposition_id if decomposition else None,
            "decomposition_status": (
                decomposition.status.value if decomposition else None
            ),
            "ordered_requirement_trace_ids": [
                item.requirement_trace_id for item in requirement_traces
            ],
            "terminal_state": terminal_state.value,
            "internal_error": (
                internal_error.model_dump(mode="json") if internal_error else None
            ),
        },
    )


class WholePlanCandidateTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_version: Literal[WHOLE_PLAN_CANDIDATE_TRACE_VERSION] = (
        WHOLE_PLAN_CANDIDATE_TRACE_VERSION
    )
    candidate_trace_id: str = Field(pattern=r"^candidatetrace_[0-9a-f]{64}$")
    candidate: ReviewCandidate
    route: StandardRoute | None = None
    review_unit_id: str | None = None
    preparation_status: ComplianceReviewStatus | None = None
    retrieval_decision: RetrievalDecision | None = None
    evidence_ids: tuple[str, ...] = ()
    decomposition: RequirementDecompositionResult | None = None
    requirement_traces: tuple[WholePlanRequirementTrace, ...] = ()
    terminal_state: WholePlanCandidateTerminalState
    internal_error: WholePlanInternalError | None = None

    @model_validator(mode="after")
    def validate_candidate_pipeline(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Candidate evidence identities must be unique and ordered.")
        if len({item.requirement_trace_id for item in self.requirement_traces}) != len(
            self.requirement_traces
        ):
            raise ValueError("Requirement trace identities must be unique.")

        if self.terminal_state == WholePlanCandidateTerminalState.INTERNAL_ERROR:
            if self.internal_error is None:
                raise ValueError("Candidate INTERNAL_ERROR requires a stable error.")
        else:
            if self.internal_error is not None or self.route is None:
                raise ValueError("A normal candidate trace requires D.2 authority.")
            if self.route.candidate_id != self.candidate.candidate_id:
                raise ValueError("D.2 route does not belong to the candidate.")
            route_terminals = {
                StandardRouteStatus.NO_STANDARD_SCOPE: (
                    WholePlanCandidateTerminalState.NO_STANDARD_SCOPE
                ),
                StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE: (
                    WholePlanCandidateTerminalState.AMBIGUOUS_STANDARD_SCOPE
                ),
            }
            if self.route.route_status in route_terminals:
                if self.terminal_state != route_terminals[self.route.route_status] or any(
                    value is not None
                    for value in (
                        self.review_unit_id,
                        self.preparation_status,
                        self.retrieval_decision,
                        self.decomposition,
                    )
                ) or self.evidence_ids or self.requirement_traces:
                    raise ValueError("A stopped D.2 trace exposes downstream authority.")
            elif self.route.route_status == StandardRouteStatus.ROUTED:
                if (
                    self.review_unit_id is None
                    or self.preparation_status is None
                    or self.retrieval_decision is None
                ):
                    raise ValueError("A routed trace requires ReviewUnit and C.1 status.")
                if self.terminal_state == WholePlanCandidateTerminalState.EVIDENCE_NOT_QUALIFIED:
                    if (
                        self.retrieval_decision == RetrievalDecision.ACCEPT
                        or self.evidence_ids
                        or self.decomposition is not None
                        or self.requirement_traces
                    ):
                        raise ValueError("Evidence stop contains downstream authority.")
                else:
                    if self.decomposition is None:
                        raise ValueError("A post-C.1 trace requires D.3 authority.")
                    if self.decomposition.candidate_id != self.candidate.candidate_id:
                        raise ValueError("D.3 decomposition belongs to another candidate.")
                    if self.decomposition.route_id != self.route.route_id:
                        raise ValueError("D.3 decomposition belongs to another route.")
                    expected_terminal = {
                        RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED: (
                            WholePlanCandidateTerminalState.REQUIREMENT_SOURCE_NOT_QUALIFIED
                        ),
                        RequirementDecompositionStatus.UNRESOLVED: (
                            WholePlanCandidateTerminalState.REQUIREMENT_UNRESOLVED
                        ),
                        RequirementDecompositionStatus.DECOMPOSED: (
                            WholePlanCandidateTerminalState.REQUIREMENTS_PROCESSED
                        ),
                        RequirementDecompositionStatus.PARTIAL: (
                            WholePlanCandidateTerminalState.REQUIREMENTS_PROCESSED
                        ),
                    }[self.decomposition.status]
                    if self.terminal_state != expected_terminal:
                        raise ValueError("Candidate terminal state contradicts D.3 status.")
                    expected_requirements = tuple(
                        item.requirement_id for item in self.decomposition.requirements
                    )
                    actual_requirements = tuple(
                        item.requirement_id for item in self.requirement_traces
                    )
                    if actual_requirements != expected_requirements:
                        raise ValueError("Requirement traces must preserve exact D.3 order.")
                    for requirement_trace in self.requirement_traces:
                        extraction = requirement_trace.extraction
                        if extraction is not None and (
                            extraction.candidate_id != self.candidate.candidate_id
                            or extraction.document_id != self.candidate.document_id
                            or extraction.document_sha256
                            != self.candidate.document_sha256
                            or extraction.route_id != self.route.route_id
                            or extraction.decomposition_id
                            != self.decomposition.decomposition_id
                        ):
                            raise ValueError(
                                "D.4 extraction does not bind the candidate D.3 authority."
                            )

        expected = deterministic_candidate_trace_id(
            candidate=self.candidate,
            route=self.route,
            review_unit_id=self.review_unit_id,
            preparation_status=self.preparation_status,
            retrieval_decision=self.retrieval_decision,
            evidence_ids=self.evidence_ids,
            decomposition=self.decomposition,
            requirement_traces=self.requirement_traces,
            terminal_state=self.terminal_state,
            internal_error=self.internal_error,
            trace_version=self.trace_version,
        )
        if self.candidate_trace_id != expected:
            raise ValueError("candidate_trace_id does not match canonical identity.")
        return self


class WholePlanCoverageCounts(BaseModel):
    """Processing counts only; no ratio or plan-level conclusion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates_discovered: int = Field(ge=0)
    routed: int = Field(ge=0)
    no_standard_scope: int = Field(ge=0)
    ambiguous_standard_scope: int = Field(ge=0)
    qualified_evidence: int = Field(ge=0)
    requirement_decomposed: int = Field(ge=0)
    requirement_partial: int = Field(ge=0)
    requirement_unresolved: int = Field(ge=0)
    requirement_source_not_qualified: int = Field(ge=0)
    plan_fact_extracted: int = Field(ge=0)
    plan_fact_not_found: int = Field(ge=0)
    plan_fact_ambiguous: int = Field(ge=0)
    plan_fact_unsupported: int = Field(ge=0)
    plan_fact_source_not_qualified: int = Field(ge=0)
    comparisons_completed: int = Field(ge=0)
    findings_created: int = Field(ge=0)
    comparison_compliant: int = Field(ge=0)
    comparison_non_compliant: int = Field(ge=0)
    comparison_insufficient_information: int = Field(ge=0)
    internal_errors: int = Field(ge=0)


def derive_coverage_counts(
    traces: tuple[WholePlanCandidateTrace, ...],
) -> WholePlanCoverageCounts:
    counts = {name: 0 for name in WholePlanCoverageCounts.model_fields}
    counts["candidates_discovered"] = len(traces)
    for trace in traces:
        if trace.internal_error is not None:
            counts["internal_errors"] += 1
        if trace.route is not None:
            route_key = {
                StandardRouteStatus.ROUTED: "routed",
                StandardRouteStatus.NO_STANDARD_SCOPE: "no_standard_scope",
                StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE: (
                    "ambiguous_standard_scope"
                ),
            }[trace.route.route_status]
            counts[route_key] += 1
        if (
            trace.preparation_status == ComplianceReviewStatus.NEEDS_COMPARISON
            and trace.retrieval_decision == RetrievalDecision.ACCEPT
        ):
            counts["qualified_evidence"] += 1
        if trace.decomposition is not None:
            decomposition_key = {
                RequirementDecompositionStatus.DECOMPOSED: "requirement_decomposed",
                RequirementDecompositionStatus.PARTIAL: "requirement_partial",
                RequirementDecompositionStatus.UNRESOLVED: "requirement_unresolved",
                RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED: (
                    "requirement_source_not_qualified"
                ),
            }[trace.decomposition.status]
            counts[decomposition_key] += 1
        for requirement_trace in trace.requirement_traces:
            if requirement_trace.internal_error is not None:
                counts["internal_errors"] += 1
            extraction = requirement_trace.extraction
            if extraction is not None:
                extraction_key = {
                    PlanFactExtractionStatus.EXTRACTED: "plan_fact_extracted",
                    PlanFactExtractionStatus.NOT_FOUND: "plan_fact_not_found",
                    PlanFactExtractionStatus.AMBIGUOUS: "plan_fact_ambiguous",
                    PlanFactExtractionStatus.UNSUPPORTED: "plan_fact_unsupported",
                    PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED: (
                        "plan_fact_source_not_qualified"
                    ),
                }[extraction.status]
                counts[extraction_key] += 1
            if requirement_trace.comparison_id is not None:
                counts["comparisons_completed"] += 1
                decision_key = {
                    ComparisonDecision.COMPLIANT: "comparison_compliant",
                    ComparisonDecision.NON_COMPLIANT: "comparison_non_compliant",
                    ComparisonDecision.INSUFFICIENT_INFORMATION: (
                        "comparison_insufficient_information"
                    ),
                }[requirement_trace.comparison_decision]
                counts[decision_key] += 1
            if requirement_trace.finding_id is not None:
                counts["findings_created"] += 1
    return WholePlanCoverageCounts(**counts)


def deterministic_whole_plan_review_id(
    *,
    document_id: str,
    document_sha256: str,
    processing_state: WholePlanProcessingState,
    candidate_traces: tuple[WholePlanCandidateTrace, ...],
    identity_version: str = WHOLE_PLAN_REVIEW_IDENTITY_VERSION,
    orchestrator_version: str = WHOLE_PLAN_REVIEW_VERSION,
) -> str:
    return _identity(
        "wholeplanreview_",
        {
            "identity_version": identity_version,
            "document_id": document_id,
            "document_sha256": document_sha256,
            "orchestrator_version": orchestrator_version,
            "ordered_candidate_trace_ids": [
                item.candidate_trace_id for item in candidate_traces
            ],
            "processing_state": processing_state.value,
        },
    )


class WholePlanReviewResult(BaseModel):
    """One deterministic processing record, explicitly not a plan verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[WHOLE_PLAN_REVIEW_IDENTITY_VERSION] = (
        WHOLE_PLAN_REVIEW_IDENTITY_VERSION
    )
    whole_plan_review_id: str = Field(pattern=r"^wholeplanreview_[0-9a-f]{64}$")
    document_id: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    orchestrator_version: Literal[WHOLE_PLAN_REVIEW_VERSION] = WHOLE_PLAN_REVIEW_VERSION
    processing_state: WholePlanProcessingState
    candidate_traces: tuple[WholePlanCandidateTrace, ...] = ()
    coverage_counts: WholePlanCoverageCounts

    @model_validator(mode="after")
    def validate_run_identity_and_counts(self):
        candidate_ids = [item.candidate.candidate_id for item in self.candidate_traces]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate identities must be unique within one run.")
        if any(
            item.candidate.document_id != self.document_id
            or item.candidate.document_sha256 != self.document_sha256
            for item in self.candidate_traces
        ):
            raise ValueError("Candidate authority differs from whole-plan source authority.")
        expected_state = (
            WholePlanProcessingState.COMPLETED_WITH_INTERNAL_ERRORS
            if any(
                item.internal_error is not None
                or any(req.internal_error is not None for req in item.requirement_traces)
                for item in self.candidate_traces
            )
            else WholePlanProcessingState.COMPLETED
        )
        if self.processing_state != expected_state:
            raise ValueError("Processing state does not match recorded internal errors.")
        if self.coverage_counts != derive_coverage_counts(self.candidate_traces):
            raise ValueError("Coverage counts must be derived from final traces.")
        expected_id = deterministic_whole_plan_review_id(
            document_id=self.document_id,
            document_sha256=self.document_sha256,
            processing_state=self.processing_state,
            candidate_traces=self.candidate_traces,
            identity_version=self.identity_version,
            orchestrator_version=self.orchestrator_version,
        )
        if self.whole_plan_review_id != expected_id:
            raise ValueError("whole_plan_review_id does not match canonical identity.")
        return self
