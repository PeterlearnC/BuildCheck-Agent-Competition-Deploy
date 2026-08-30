"""Deterministic, local ReviewUnit x requirement comparison orchestration."""

import hashlib
import json
from decimal import Decimal

from app.schemas.compliance_comparison import (
    ApplicabilityStatus,
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonMethod,
    ComparisonReasonCode,
    ComparisonResult,
    ComplianceComparisonResponse,
    MissingInformationItem,
    PlanFact,
    PlanFactParsingStatus,
    PlanFactType,
    SourceSpanSelector,
    StructuredComparisonReason,
)
from app.schemas.compliance_review import ComplianceReviewResult, ComplianceReviewStatus
from app.schemas.normative_requirement import (
    NormativeRequirement,
    NumericOperator,
    RequirementAtomicity,
    RequirementModality,
)
from app.schemas.standards_retrieval import RetrievalDecision
from app.services.review.plan_fact_service import PlanFactService
from app.services.review.requirement_service import RequirementService


COMPARISON_VERSION = "v0.1-c.2-f1-deterministic-numeric"
COMPARISON_SCOPE_VERSION = "v0.1-review-unit-requirement"


class ComplianceComparisonError(ValueError):
    pass


class EvidenceSelectionError(ComplianceComparisonError):
    pass


class ComplianceComparisonService:
    def __init__(
        self,
        requirement_service: RequirementService | None = None,
        plan_fact_service: PlanFactService | None = None,
    ) -> None:
        self.requirement_service = requirement_service or RequirementService()
        self.plan_fact_service = plan_fact_service or PlanFactService()

    def compare(
        self,
        preparation: ComplianceReviewResult,
        *,
        evidence_id: str,
        requirement_selector: SourceSpanSelector,
        plan_fact_selectors: list[SourceSpanSelector],
    ) -> ComplianceComparisonResponse:
        if preparation.retrieval_decision != RetrievalDecision.ACCEPT:
            if preparation.status != ComplianceReviewStatus.INSUFFICIENT_EVIDENCE:
                raise ComplianceComparisonError(
                    "Non-ACCEPT C.1 preparation has an invalid status."
                )
            return ComplianceComparisonResponse(preparation=preparation)
        if preparation.status != ComplianceReviewStatus.NEEDS_COMPARISON:
            raise ComplianceComparisonError(
                "ACCEPT C.1 preparation must be NEEDS_COMPARISON."
            )

        selected = [
            binding
            for binding in preparation.evidence_bindings
            if binding.standard_evidence_id == evidence_id
        ]
        if len(selected) != 1:
            raise EvidenceSelectionError(
                "Selected evidence_id does not uniquely identify an ACCEPT binding."
            )
        binding = selected[0]
        requirement = self.requirement_service.select(binding, requirement_selector)
        requirement = self.requirement_service.verify(requirement, binding)
        plan_facts = self.plan_fact_service.create_many(
            preparation.review_unit, plan_fact_selectors
        )
        plan_facts = self.plan_fact_service.verify_many(
            preparation.review_unit, plan_facts
        )
        result = self._compare_numeric_or_fail_closed(
            preparation=preparation,
            requirement=requirement,
            plan_facts=plan_facts,
        )
        return ComplianceComparisonResponse(
            preparation=preparation,
            evidence_binding=binding,
            requirement=requirement,
            plan_facts=plan_facts,
            comparison=result,
        )

    def _compare_numeric_or_fail_closed(
        self,
        *,
        preparation: ComplianceReviewResult,
        requirement: NormativeRequirement,
        plan_facts: list[PlanFact],
    ) -> ComparisonResult:
        if requirement.atomicity != RequirementAtomicity.PROVEN:
            reason_code = (
                ComparisonReasonCode.CONDITION_UNRESOLVED
                if requirement.modality == RequirementModality.CONDITIONAL
                else ComparisonReasonCode.REQUIREMENT_DECOMPOSITION_UNRESOLVED
            )
            return self._insufficient(
                preparation,
                requirement,
                plan_facts,
                reason_code=reason_code,
                missing=[
                    "A single unconditional atomic requirement was not deterministically proven."
                ],
                applicability=ApplicabilityStatus.UNRESOLVED,
            )
        if requirement.modality != RequirementModality.NUMERIC_LIMIT:
            return self._insufficient(
                preparation,
                requirement,
                plan_facts,
                reason_code=ComparisonReasonCode.UNSUPPORTED_REQUIREMENT_CLASS,
                missing=["The selected atomic requirement class is not supported by C.2-F1."],
                applicability=ApplicabilityStatus.APPLICABLE,
            )
        if not plan_facts:
            return self._insufficient(
                preparation,
                requirement,
                plan_facts,
                reason_code=ComparisonReasonCode.PLAN_FACT_MISSING,
                missing=["A grounded numeric PlanFact is required."],
                applicability=ApplicabilityStatus.APPLICABLE,
            )
        proven_numeric = [
            fact
            for fact in plan_facts
            if fact.fact_type == PlanFactType.NUMERIC
            and fact.parsing_status == PlanFactParsingStatus.PROVEN
        ]
        if len(plan_facts) != 1 or len(proven_numeric) != 1:
            return self._insufficient(
                preparation,
                requirement,
                plan_facts,
                reason_code=ComparisonReasonCode.PLAN_FACT_AMBIGUOUS,
                missing=["Exactly one unambiguous grounded numeric PlanFact is required."],
                applicability=ApplicabilityStatus.APPLICABLE,
            )
        fact = proven_numeric[0]
        if self._normalized_unit(requirement.unit) != self._normalized_unit(fact.unit):
            return self._insufficient(
                preparation,
                requirement,
                plan_facts,
                reason_code=ComparisonReasonCode.UNSUPPORTED_UNIT,
                missing=["Standard and plan units must match exactly; conversion is unsupported."],
                applicability=ApplicabilityStatus.APPLICABLE,
            )
        if not isinstance(fact.normalized_value, Decimal):
            raise ComplianceComparisonError(
                "A proven numeric PlanFact has an invalid normalized value."
            )
        if requirement.value is None or requirement.operator is None:
            raise ComplianceComparisonError(
                "A numeric requirement is missing deterministic constraint fields."
            )

        satisfied = self._evaluate(
            fact.normalized_value, requirement.operator, requirement.value
        )
        decision = (
            ComparisonDecision.COMPLIANT
            if satisfied
            else ComparisonDecision.NON_COMPLIANT
        )
        reason_code = (
            ComparisonReasonCode.NUMERIC_LIMIT_SATISFIED
            if satisfied
            else ComparisonReasonCode.NUMERIC_LIMIT_VIOLATED
        )
        fact_ids = [fact.plan_fact_id for fact in plan_facts]
        return ComparisonResult(
            comparison_id=self._comparison_id(
                preparation.review_unit.review_unit_id,
                requirement.requirement_id,
                fact_ids,
            ),
            decision=decision,
            decision_scope=ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT,
            review_unit_id=preparation.review_unit.review_unit_id,
            requirement_id=requirement.requirement_id,
            evidence_id=requirement.evidence_id,
            plan_facts_used=fact_ids,
            requirement_text=requirement.requirement_text,
            reason_code=reason_code,
            reason=StructuredComparisonReason(
                summary=(
                    "The grounded plan numeric value satisfies the exact local limit."
                    if satisfied
                    else "The grounded plan numeric value affirmatively violates the exact local limit."
                ),
                plan_fact_ids=fact_ids,
                actual_value=fact.normalized_value,
                operator=requirement.operator.value,
                required_value=requirement.value,
                unit=requirement.unit,
            ),
            missing_information=[],
            applicability_status=ApplicabilityStatus.APPLICABLE,
            comparison_method=ComparisonMethod.DETERMINISTIC_NUMERIC,
            comparison_version=COMPARISON_VERSION,
        )

    def _insufficient(
        self,
        preparation: ComplianceReviewResult,
        requirement: NormativeRequirement,
        plan_facts: list[PlanFact],
        *,
        reason_code: ComparisonReasonCode,
        missing: list[str],
        applicability: ApplicabilityStatus,
    ) -> ComparisonResult:
        fact_ids = [fact.plan_fact_id for fact in plan_facts]
        return ComparisonResult(
            comparison_id=self._comparison_id(
                preparation.review_unit.review_unit_id,
                requirement.requirement_id,
                fact_ids,
            ),
            decision=ComparisonDecision.INSUFFICIENT_INFORMATION,
            decision_scope=ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT,
            review_unit_id=preparation.review_unit.review_unit_id,
            requirement_id=requirement.requirement_id,
            evidence_id=requirement.evidence_id,
            plan_facts_used=fact_ids,
            requirement_text=requirement.requirement_text,
            reason_code=reason_code,
            reason=StructuredComparisonReason(
                summary="Deterministic local comparison cannot prove a decisive result."
            ),
            missing_information=[
                MissingInformationItem(code=reason_code, detail=detail)
                for detail in missing
            ],
            applicability_status=applicability,
            comparison_method=ComparisonMethod.FAIL_CLOSED,
            comparison_version=COMPARISON_VERSION,
        )

    @staticmethod
    def _evaluate(
        actual: Decimal, operator: NumericOperator, limit: Decimal
    ) -> bool:
        operations = {
            NumericOperator.LT: actual < limit,
            NumericOperator.LE: actual <= limit,
            NumericOperator.GT: actual > limit,
            NumericOperator.GE: actual >= limit,
            NumericOperator.EQ: actual == limit,
        }
        return operations[operator]

    @staticmethod
    def _normalized_unit(unit: str | None) -> str | None:
        return unit.strip().casefold() if unit is not None else None

    @staticmethod
    def _comparison_id(
        review_unit_id: str, requirement_id: str, plan_fact_ids: list[str]
    ) -> str:
        payload = {
            "identity_version": COMPARISON_VERSION,
            "review_unit_id": review_unit_id,
            "requirement_id": requirement_id,
            "ordered_plan_fact_ids": plan_fact_ids,
            "comparison_method_version": COMPARISON_VERSION,
            "decision_scope_version": COMPARISON_SCOPE_VERSION,
        }
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "comparison_" + hashlib.sha256(serialized).hexdigest()
