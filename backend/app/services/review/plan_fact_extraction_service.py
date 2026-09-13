"""Deterministic, source-bound extraction of one comparison-ready PlanFact."""

import hashlib
import re
from dataclasses import dataclass

from app.schemas.compliance_comparison import (
    PlanFactParsingStatus,
    SourceSpanSelector,
)
from app.schemas.compliance_review import ComplianceReviewResult
from app.schemas.normative_requirement import (
    NormativeRequirement,
    NumericOperator,
    RequirementAtomicity,
    RequirementModality,
)
from app.schemas.plan_fact_extraction import (
    PLAN_FACT_EXTRACTION_VERSION,
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
)
from app.schemas.review_candidate import ReviewCandidate
from app.schemas.standard_route import StandardRoute
from app.services.review.plan_fact_service import PlanFactService
from app.services.review.requirement_decomposition_service import (
    RequirementDecompositionService,
)
from app.services.review.review_unit_service import ReviewUnitService
from app.services.review.routing_context_service import RoutingContextService


_ATOMIC_CLAUSE = re.compile(r"[^。！？；;]+[。！？；;]?")
_CONDITIONAL = re.compile(
    r"^\s*(?:(?:当|对于|仅当|若|如果|在[^，。；;]{0,32}情况下)[^，。；;]*[，,]"
    r"|不同规格分别|施工过程中)"
)
_LOCATION = re.compile(r"^\s*(?:在|于)[^，。；;]{1,32}[，,]")
_PROHIBITED_CHOICE = re.compile(r"(?:不得|不应|禁止|不)\s*(?:采用|选用|使用)\s*\d")
_COMPOUND_DIMENSION = re.compile(r"\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?")
_RANGE = re.compile(r"\d+(?:\.\d+)?\s*(?:[-~～至到])\s*\d+(?:\.\d+)?")
_UPPER = re.compile(r"(?:不得大于|不大于|不得超过|不超过|至多|控制在[^，。；;]{0,20}以内)")
_LOWER = re.compile(r"(?:不得小于|不小于|不得低于|不低于|至少)")
_DESIGN = re.compile(r"(?:设计值|控制值|设计控制值)\s*(?:为|取)|采用[^，。；;]{0,20}控制")
_EXACT = re.compile(r"(?:为|等于|取值为|实测(?:为)?|测得)\s*[+-]?\d")
_NUMBER_WITH_UNIT_VIEW = re.compile(
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(?:%|℃|°C|mm|cm|m|kg)", re.I
)

_MEASUREMENTS = ("长度", "间隙", "高度", "厚度", "距离", "间距", "外径", "内径")
_OBJECTS = (
    "可调托撑",
    "托撑",
    "螺杆",
    "立杆",
    "钢管",
    "建筑",
    "板",
    "构件",
    "连墙件",
    "节点",
)


class PlanFactExtractionError(ValueError):
    pass


class PlanFactExtractionAuthorityError(PlanFactExtractionError):
    pass


@dataclass(frozen=True)
class _MeasurementProfile:
    name: str
    measurements: tuple[str, ...]
    objects: tuple[str, ...]


@dataclass(frozen=True)
class _RejectedClause:
    reason: PlanFactExtractionReason
    ambiguous: bool = False


class PlanFactExtractionService:
    """Discover candidate-local facts while keeping frozen C.2 as fact authority."""

    def __init__(
        self,
        *,
        decomposition_service: RequirementDecompositionService | None = None,
        routing_context_service: RoutingContextService | None = None,
        review_unit_service: ReviewUnitService | None = None,
        plan_fact_service: PlanFactService | None = None,
    ) -> None:
        self.decomposition_service = (
            decomposition_service or RequirementDecompositionService()
        )
        self.routing_context_service = (
            routing_context_service or RoutingContextService()
        )
        self.review_unit_service = review_unit_service or ReviewUnitService()
        self.plan_fact_service = plan_fact_service or PlanFactService(
            self.review_unit_service
        )

    def extract(
        self,
        candidate: ReviewCandidate,
        route: StandardRoute,
        preparation: ComplianceReviewResult,
        *,
        requirement_id: str,
    ) -> PlanFactExtractionResult:
        """Recompute D.3 and stored-plan authority before examining exact plan text."""

        if not isinstance(candidate, ReviewCandidate):
            raise TypeError("D.4 requires a typed ReviewCandidate.")
        if not isinstance(route, StandardRoute):
            raise TypeError("D.4 requires a typed StandardRoute.")
        if not isinstance(preparation, ComplianceReviewResult):
            raise TypeError("D.4 requires a server-produced C.1 preparation contract.")
        if not requirement_id:
            raise ValueError("D.4 requires an explicit proven requirement identity.")

        decomposition = self.decomposition_service.decompose(
            candidate, route, preparation
        )
        self._validate_decomposition(candidate, route, decomposition)
        requirement = next(
            (
                item
                for item in decomposition.requirements
                if item.requirement_id == requirement_id
            ),
            None,
        )
        if (
            decomposition.status == RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED
            or requirement is None
        ):
            return self._result(
                candidate=candidate,
                route=route,
                preparation=preparation,
                decomposition=decomposition,
                requirement_id=requirement_id,
                status=PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED,
                reason=PlanFactExtractionReason.REQUIREMENT_NOT_PROVEN,
            )
        if (
            requirement.atomicity != RequirementAtomicity.PROVEN
            or requirement.modality != RequirementModality.NUMERIC_LIMIT
        ):
            return self._result(
                candidate=candidate,
                route=route,
                preparation=preparation,
                decomposition=decomposition,
                requirement_id=requirement_id,
                status=PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED,
                reason=PlanFactExtractionReason.REQUIREMENT_NOT_PROVEN,
            )

        try:
            context = self.routing_context_service.build_for_candidate(
                candidate=candidate
            )
            self.routing_context_service.verify(candidate=candidate, context=context)
            unit = self.review_unit_service.verify(preparation.review_unit)
            self._validate_plan_authority(candidate, preparation, context, unit)
        except (OSError, ValueError) as exc:
            return self._result(
                candidate=candidate,
                route=route,
                preparation=preparation,
                decomposition=decomposition,
                requirement_id=requirement_id,
                status=PlanFactExtractionStatus.SOURCE_NOT_QUALIFIED,
                reason=PlanFactExtractionReason.SOURCE_AUTHORITY_FAILED,
            )

        bindings: list[ExtractedPlanFactBinding] = []
        rejected: list[_RejectedClause] = []
        for start, end in self._atomic_spans(unit.source_text):
            binding, rejection = self._extract_clause(
                candidate=candidate,
                requirement=requirement,
                unit=unit,
                context=context,
                start=start,
                end=end,
            )
            if binding is not None:
                bindings.append(binding)
            elif rejection is not None:
                rejected.append(rejection)

        if len(bindings) == 1:
            return self._result(
                candidate=candidate,
                route=route,
                preparation=preparation,
                decomposition=decomposition,
                requirement_id=requirement_id,
                status=PlanFactExtractionStatus.EXTRACTED,
                binding=bindings[0],
            )
        if len(bindings) > 1:
            return self._result(
                candidate=candidate,
                route=route,
                preparation=preparation,
                decomposition=decomposition,
                requirement_id=requirement_id,
                status=PlanFactExtractionStatus.AMBIGUOUS,
                reason=PlanFactExtractionReason.MULTIPLE_MATCHING_FACTS,
            )
        ambiguous = next((item for item in rejected if item.ambiguous), None)
        if ambiguous is not None:
            return self._result(
                candidate=candidate,
                route=route,
                preparation=preparation,
                decomposition=decomposition,
                requirement_id=requirement_id,
                status=PlanFactExtractionStatus.AMBIGUOUS,
                reason=ambiguous.reason,
            )
        unsafe = next(
            (
                item
                for item in rejected
                if item.reason
                not in {
                    PlanFactExtractionReason.NO_NUMERIC_FACT,
                    PlanFactExtractionReason.OBJECT_MISMATCH,
                }
            ),
            None,
        )
        return self._result(
            candidate=candidate,
            route=route,
            preparation=preparation,
            decomposition=decomposition,
            requirement_id=requirement_id,
            status=(
                PlanFactExtractionStatus.UNSUPPORTED
                if unsafe
                else PlanFactExtractionStatus.NOT_FOUND
            ),
            reason=(unsafe.reason if unsafe else PlanFactExtractionReason.NO_NUMERIC_FACT),
        )

    @staticmethod
    def _validate_decomposition(
        candidate: ReviewCandidate,
        route: StandardRoute,
        decomposition: RequirementDecompositionResult,
    ) -> None:
        if not isinstance(decomposition, RequirementDecompositionResult):
            raise PlanFactExtractionAuthorityError(
                "D.3 did not return a validated decomposition contract."
            )
        if (
            decomposition.candidate_id != candidate.candidate_id
            or decomposition.route_id != route.route_id
            or decomposition.document_id != candidate.document_id
            or decomposition.document_sha256 != candidate.document_sha256
        ):
            raise PlanFactExtractionAuthorityError(
                "Recomputed D.3 authority does not match D.4 inputs."
            )

    @staticmethod
    def _validate_plan_authority(candidate, preparation, context, unit) -> None:
        span = candidate.source_spans[0]
        if (
            unit != preparation.review_unit
            or unit.document_id != candidate.document_id
            or unit.page_number != span.page_number
            or unit.char_start != span.char_start
            or unit.char_end != span.char_end
            or unit.source_text != candidate.source_text
            or unit.source_text_sha256 != candidate.source_text_sha256
            or context.document_sha256 != candidate.document_sha256
            or context.page_number != span.page_number
            or context.source_page_text_sha256 == ""
        ):
            raise PlanFactExtractionAuthorityError(
                "Stored source, ReviewUnit and candidate authority do not match."
            )

    def _extract_clause(
        self, *, candidate, requirement, unit, context, start: int, end: int
    ):
        text = unit.source_text[start:end]
        view = re.sub(r"\s+", "", text)
        if not _NUMBER_WITH_UNIT_VIEW.search(view):
            return None, _RejectedClause(PlanFactExtractionReason.NO_NUMERIC_FACT)
        if _CONDITIONAL.search(view):
            return None, _RejectedClause(
                PlanFactExtractionReason.CONDITIONAL_UNSUPPORTED
            )
        if _LOCATION.search(view):
            return None, _RejectedClause(
                PlanFactExtractionReason.LOCATION_QUALIFIER_UNSUPPORTED
            )
        if _PROHIBITED_CHOICE.search(view):
            return None, _RejectedClause(
                PlanFactExtractionReason.PROHIBITED_VALUE_NOT_FACT
            )
        if _COMPOUND_DIMENSION.search(view):
            return None, _RejectedClause(
                PlanFactExtractionReason.COMPOUND_DIMENSION_UNSUPPORTED
            )
        if _RANGE.search(view):
            return None, _RejectedClause(PlanFactExtractionReason.RANGE_UNSUPPORTED)

        profile = self._measurement_profile(requirement.requirement_text, view)
        if profile is None:
            return None, _RejectedClause(PlanFactExtractionReason.OBJECT_MISMATCH)
        assertion = self._assertion_kind(view, profile)
        if assertion is None:
            return None, _RejectedClause(
                PlanFactExtractionReason.ASSERTION_UNSUPPORTED
            )
        if not self._compatible(requirement.operator, assertion):
            return None, _RejectedClause(
                PlanFactExtractionReason.ASSERTION_INCOMPATIBLE
            )

        selector = SourceSpanSelector(
            char_start=start, char_end=end, source_text=text
        )
        facts = self.plan_fact_service.create_many(unit, [selector])
        self.plan_fact_service.verify_many(unit, facts)
        fact = facts[0]
        if fact.parsing_status == PlanFactParsingStatus.AMBIGUOUS:
            return None, _RejectedClause(
                PlanFactExtractionReason.MULTIPLE_VALUES, ambiguous=True
            )
        if fact.parsing_status != PlanFactParsingStatus.PROVEN:
            return None, _RejectedClause(
                PlanFactExtractionReason.ASSERTION_UNSUPPORTED
            )
        profile_id = deterministic_measurement_profile_id(
            profile_name=profile.name,
            measurement_anchors=profile.measurements,
            object_anchors=profile.objects,
        )
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return ExtractedPlanFactBinding(
            plan_fact=fact,
            review_unit_id=unit.review_unit_id,
            routing_context_id=context.routing_context_id,
            source_page_text_sha256=context.source_page_text_sha256,
            review_unit_char_start=start,
            review_unit_char_end=end,
            page_char_start=candidate.source_spans[0].char_start + start,
            page_char_end=candidate.source_spans[0].char_start + end,
            source_text=text,
            source_text_sha256=source_hash,
            assertion_kind=assertion,
            measurement_profile_id=profile_id,
            measurement_profile_name=profile.name,
            measurement_anchors=profile.measurements,
            object_anchors=profile.objects,
        ), None

    @staticmethod
    def _atomic_spans(text: str) -> tuple[tuple[int, int], ...]:
        spans = []
        for match in _ATOMIC_CLAUSE.finditer(text):
            start, end = match.span()
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                spans.append((start, end))
        return tuple(spans)

    @staticmethod
    def _measurement_profile(
        requirement_text: str, plan_text: str
    ) -> _MeasurementProfile | None:
        requirement_view = re.sub(r"\s+", "", requirement_text)
        specific = (
            (
                "UPRIGHT_INSERTION_LENGTH",
                ("插入", "长度"),
                ("立杆",),
            ),
            (
                "SCREW_UPRIGHT_TUBE_CLEARANCE",
                ("间隙",),
                ("螺杆", "立杆", "钢管"),
            ),
            (
                "COMPONENT_EXTENSION_LENGTH",
                ("伸出", "长度"),
                tuple(),
            ),
        )
        for name, measurements, objects in specific:
            if all(token in requirement_view for token in measurements + objects):
                if all(token in plan_text for token in measurements + objects):
                    resolved_objects = objects or tuple(
                        item
                        for item in _OBJECTS
                        if item in requirement_view and item in plan_text
                    )
                    if resolved_objects:
                        return _MeasurementProfile(
                            name, measurements, resolved_objects
                        )
                return None

        measurements = tuple(
            token
            for token in _MEASUREMENTS
            if token in requirement_view and token in plan_text
        )
        objects = tuple(
            token
            for token in _OBJECTS
            if token in requirement_view and token in plan_text
        )
        if not measurements or not objects:
            return None
        return _MeasurementProfile("SHARED_OBJECT_MEASUREMENT", measurements, objects)

    @staticmethod
    def _assertion_kind(
        text: str, profile: _MeasurementProfile
    ) -> PlanAssertionKind | None:
        if _UPPER.search(text):
            return PlanAssertionKind.UPPER_BOUND_CONTROL
        if _LOWER.search(text):
            return PlanAssertionKind.LOWER_BOUND_CONTROL
        if _DESIGN.search(text):
            return PlanAssertionKind.DESIGN_CONTROL
        if _EXACT.search(text):
            return PlanAssertionKind.EXACT_VALUE
        if any(
            re.search(re.escape(anchor) + r"[^，。；;]{0,8}\d", text)
            for anchor in profile.measurements
        ):
            return PlanAssertionKind.EXACT_VALUE
        return None

    @staticmethod
    def _compatible(
        operator: NumericOperator | None, assertion: PlanAssertionKind
    ) -> bool:
        if assertion in {
            PlanAssertionKind.EXACT_VALUE,
            PlanAssertionKind.DESIGN_CONTROL,
        }:
            return operator in {
                NumericOperator.GE,
                NumericOperator.LE,
                NumericOperator.EQ,
            }
        if assertion == PlanAssertionKind.LOWER_BOUND_CONTROL:
            return operator == NumericOperator.GE
        return operator == NumericOperator.LE

    @staticmethod
    def _result(
        *,
        candidate,
        route,
        preparation,
        decomposition,
        requirement_id,
        status,
        binding=None,
        reason=None,
    ) -> PlanFactExtractionResult:
        span = candidate.source_spans[0]
        values = dict(
            candidate_id=candidate.candidate_id,
            document_id=candidate.document_id,
            document_sha256=candidate.document_sha256,
            physical_page=span.page_number,
            review_unit_id=preparation.review_unit.review_unit_id,
            route_id=route.route_id,
            decomposition_id=decomposition.decomposition_id,
            requirement_id=requirement_id,
            status=status,
            binding=binding,
            reason=reason,
        )
        extraction_id = deterministic_extraction_id(**values)
        return PlanFactExtractionResult(
            extraction_id=extraction_id,
            extraction_version=PLAN_FACT_EXTRACTION_VERSION,
            **values,
        )
