"""Deterministic, repository-qualified requirement decomposition for D.3."""

import hashlib
import re

from app.schemas.compliance_comparison import SourceSpanSelector
from app.schemas.compliance_review import ComplianceReviewResult, ComplianceReviewStatus
from app.schemas.normative_requirement import (
    RequirementAtomicity,
    RequirementModality,
)
from app.schemas.requirement_decomposition import (
    DECOMPOSITION_VERSION,
    RequirementDecompositionMethod,
    RequirementDecompositionResult,
    RequirementDecompositionStatus,
    UnresolvedRequirementReason,
    UnresolvedRequirementSpan,
    deterministic_decomposition_id,
    deterministic_unresolved_span_id,
)
from app.schemas.review_candidate import ReviewCandidate, ReviewCandidateStatus
from app.schemas.standard_route import StandardRoute, StandardRouteStatus
from app.schemas.standards import (
    ArticleType,
    StandardIdentityStatus,
    StandardParseStatus,
)
from app.schemas.standards_retrieval import RetrievalDecision, RetrievalMethod
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.review.requirement_service import (
    RequirementSelectionError,
    RequirementService,
)
from app.services.standards.standard_repository import StandardRepository


_NORMATIVE_CUE = re.compile(r"(?:不得|不应|应当|应|必须|严禁|禁止)")
_CONDITION_CUE = re.compile(
    r"(?:^|[\s，。；;：:]|\d\s+)(?:当|若|如果|对于|仅当|在[^，。；;]{0,30}情况下)"
)
_EXCEPTION_OR_SCOPE_CUE = re.compile(
    r"(?:除[^，。；;]{0,40}外|但(?:是)?|仅当|用于|施工过程中)"
)
_LEADING_LOCATION_OR_SCOPE_CUE = re.compile(
    r"(?:^|[。；;：:\n])\s*"
    r"(?:(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+)[、.．]?\s*)?"
    r"(?:在|于)[^，,。；;：:\n]{1,32}[，,]"
)
_NUMBERED_ITEM = re.compile(
    r"(?m)^[ \t]*(?:\d+[、．]|\d+\.(?=[ \t])|\d+[ \t]+(?=(?:当|应|不|必须|严禁|禁止)))"
)
_GUARDED_AND = re.compile(r"，且(?=[^，。；;]{2,}?(?:不得|不应|应当|应|必须|严禁|禁止))")
_NUMERIC_TOKEN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|℃|°C|mm|cm|m|kg|毫米|厘米|米|千克|公斤|吨|分钟|小时|秒|度)",
    re.IGNORECASE,
)


class RequirementDecompositionError(ValueError):
    pass


class RequirementDecompositionAuthorityError(RequirementDecompositionError):
    pass


class RequirementDecompositionService:
    """Discover exact atoms without moving comparison authority into D.3."""

    def __init__(
        self,
        repository: StandardRepository | None = None,
        requirement_service: RequirementService | None = None,
    ) -> None:
        self.repository = repository or StandardRepository()
        self.requirement_service = requirement_service or RequirementService()

    def decompose(
        self,
        candidate: ReviewCandidate,
        route: StandardRoute,
        preparation: ComplianceReviewResult,
    ) -> RequirementDecompositionResult:
        scope = self._validate_candidate_route(candidate, route)
        self._validate_plan_binding(candidate, preparation, scope.standard_id)

        if preparation.retrieval_decision != RetrievalDecision.ACCEPT:
            return self._result(
                candidate=candidate,
                route=route,
                scope=scope,
                evidence_ids=(),
                requirements=(),
                unresolved=(),
                status=RequirementDecompositionStatus.SOURCE_NOT_QUALIFIED,
            )
        if preparation.status != ComplianceReviewStatus.NEEDS_COMPARISON:
            raise RequirementDecompositionAuthorityError(
                "ACCEPT evidence must be a NEEDS_COMPARISON preparation."
            )

        document = self._qualified_document(scope.standard_id, scope.source_checksum)
        articles = {
            article.article_id: article
            for article in self.repository.load_articles(scope.standard_id)
        }
        qualified = []
        for binding in preparation.evidence_bindings:
            qualified.append(
                self._qualified_binding(
                    binding=binding,
                    routed_standard_id=scope.standard_id,
                    document=document,
                    articles=articles,
                )
            )
        if not qualified:
            raise RequirementDecompositionAuthorityError(
                "ACCEPT preparation contains no qualified evidence bindings."
            )
        qualified.sort(
            key=lambda item: (
                item[0].evidence.article_number,
                item[0].evidence.id,
            )
        )

        requirements = []
        unresolved = []
        for binding, article in qualified:
            for start, end in self._atomic_spans(article.source_text):
                selected_text = article.source_text[start:end]
                forced_reason = self._forced_unresolved_reason(selected_text)
                if forced_reason is not None:
                    unresolved.append(
                        self._unresolved(binding, start, end, forced_reason)
                    )
                    continue
                selector = SourceSpanSelector(
                    char_start=start,
                    char_end=end,
                    source_text=selected_text,
                )
                try:
                    requirement = self.requirement_service.select(binding, selector)
                    self.requirement_service.verify(requirement, binding)
                except (RequirementSelectionError, ValueError):
                    unresolved.append(
                        self._unresolved(
                            binding,
                            start,
                            end,
                            UnresolvedRequirementReason.AMBIGUOUS_SPLIT,
                        )
                    )
                    continue
                if (
                    requirement.atomicity == RequirementAtomicity.PROVEN
                    and requirement.modality == RequirementModality.NUMERIC_LIMIT
                ):
                    requirements.append(requirement)
                else:
                    unresolved.append(
                        self._unresolved(
                            binding,
                            start,
                            end,
                            self._reason_for_unproven(selected_text, requirement),
                        )
                    )

        if requirements and unresolved:
            status = RequirementDecompositionStatus.PARTIAL
        elif requirements:
            status = RequirementDecompositionStatus.DECOMPOSED
        else:
            status = RequirementDecompositionStatus.UNRESOLVED
        return self._result(
            candidate=candidate,
            route=route,
            scope=scope,
            evidence_ids=tuple(item[0].evidence.id for item in qualified),
            requirements=tuple(requirements),
            unresolved=tuple(unresolved),
            status=status,
        )

    @staticmethod
    def _validate_candidate_route(candidate: ReviewCandidate, route: StandardRoute):
        if candidate.status != ReviewCandidateStatus.DISCOVERED:
            raise RequirementDecompositionAuthorityError(
                "D.3 requires a discovered ReviewCandidate."
            )
        if route.route_status != StandardRouteStatus.ROUTED:
            raise RequirementDecompositionAuthorityError(
                "D.3 requires a qualified ROUTED StandardRoute."
            )
        if (
            route.candidate_id != candidate.candidate_id
            or route.candidate_class != candidate.candidate_class
            or route.candidate_topic != candidate.topic
        ):
            raise RequirementDecompositionAuthorityError(
                "StandardRoute does not match the ReviewCandidate identity."
            )
        if len(route.scope_candidates) != 1 or route.selected_scope is None:
            raise RequirementDecompositionAuthorityError(
                "ROUTED authority must contain exactly one selected scope."
            )
        return route.scope_candidates[0]

    @staticmethod
    def _validate_plan_binding(
        candidate: ReviewCandidate,
        preparation: ComplianceReviewResult,
        routed_standard_id: str,
    ) -> None:
        unit = preparation.review_unit
        span = candidate.source_spans[0]
        if (
            unit.document_id != candidate.document_id
            or unit.page_number != span.page_number
            or unit.char_start != span.char_start
            or unit.char_end != span.char_end
            or unit.source_text != candidate.source_text
            or unit.source_text_sha256 != candidate.source_text_sha256
        ):
            raise RequirementDecompositionAuthorityError(
                "C.1 preparation does not match the ReviewCandidate source authority."
            )
        if unit.standard_scope.standard_ids != [routed_standard_id]:
            raise RequirementDecompositionAuthorityError(
                "C.1 scope does not equal the selected D.2 standard scope."
            )

    def _qualified_document(self, standard_id: str, route_checksum: str):
        document = self.repository.load_document(standard_id)
        if document is None:
            raise RequirementDecompositionAuthorityError(
                "The routed standard is not installed."
            )
        binding = document.official_source_binding
        if (
            document.identity_status != StandardIdentityStatus.CONFIRMED
            or document.parse_status != StandardParseStatus.PARSED
            or binding is None
            or not binding.confirmed
            or document.source_checksum != route_checksum
            or binding.source_checksum != document.source_checksum
            or binding.canonical_standard_code != document.canonical_standard_code
        ):
            raise RequirementDecompositionAuthorityError(
                "The routed standard lacks qualified installed/official authority."
            )
        return document

    @staticmethod
    def _qualified_binding(*, binding, routed_standard_id, document, articles):
        evidence = binding.evidence
        article = articles.get(evidence.article_id)
        if (
            binding.retrieval_decision != RetrievalDecision.ACCEPT
            or evidence.article_type != ArticleType.NORMATIVE
            or binding.standard_id != routed_standard_id
            or evidence.standard_id != routed_standard_id
            or article is None
            or article.standard_id != routed_standard_id
            or article.article_number != evidence.article_number
            or article.source_text != evidence.source_text
            or evidence.source_checksum != document.source_checksum
        ):
            raise RequirementDecompositionAuthorityError(
                "Evidence does not match the routed repository article authority."
            )
        expected = make_hit(
            RetrievalRecord(document=document, article=article),
            rank=1,
            methods=[RetrievalMethod.KEYWORD],
        ).evidence
        if expected != evidence or binding.standard_evidence_id != expected.id:
            raise RequirementDecompositionAuthorityError(
                "EvidenceEnvelope does not match deterministic repository reconstruction."
            )
        return binding, article

    @classmethod
    def _atomic_spans(cls, text: str) -> list[tuple[int, int]]:
        boundaries = {0, len(text)}
        for match in _NUMBERED_ITEM.finditer(text):
            boundaries.add(match.start())
        for index, character in enumerate(text):
            if character in "。；;":
                boundaries.add(index + 1)

        spans: list[tuple[int, int]] = []
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            start, end = cls._trim_span(text, left, right)
            if start >= end:
                continue
            spans.extend(cls._split_guarded_conjunction(text, start, end))
        return spans

    @classmethod
    def _split_guarded_conjunction(
        cls, text: str, start: int, end: int
    ) -> list[tuple[int, int]]:
        selected = text[start:end]
        if cls._has_condition(selected) or _EXCEPTION_OR_SCOPE_CUE.search(selected):
            return [(start, end)]
        match = _GUARDED_AND.search(selected)
        if match is None:
            return [(start, end)]
        left_text = selected[: match.start() + 1]
        right_text = selected[match.start() + 1 :]
        if not (_NORMATIVE_CUE.search(left_text) and _NORMATIVE_CUE.search(right_text)):
            return [(start, end)]
        split = start + match.start() + 1
        left = cls._trim_span(text, start, split)
        right = cls._trim_span(text, split, end)
        return [item for item in (left, right) if item[0] < item[1]]

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _has_condition(text: str) -> bool:
        return bool(_CONDITION_CUE.search(text))

    @classmethod
    def _forced_unresolved_reason(
        cls, text: str
    ) -> UnresolvedRequirementReason | None:
        if cls._has_condition(text):
            return UnresolvedRequirementReason.CONDITIONAL_UNSUPPORTED
        if _EXCEPTION_OR_SCOPE_CUE.search(
            text
        ) or _LEADING_LOCATION_OR_SCOPE_CUE.search(text):
            return UnresolvedRequirementReason.EXCEPTION_SCOPE_UNRESOLVED
        # A single numeric match must not hide another obligation in the same
        # exact span. Guarded splitting has already removed the narrow case in
        # which both sides carry their own explicit subject and predicate.
        if len(list(_NORMATIVE_CUE.finditer(text))) > 1:
            return UnresolvedRequirementReason.COMPOUND_STRUCTURE_UNRESOLVED
        return None

    @staticmethod
    def _reason_for_unproven(text, requirement) -> UnresolvedRequirementReason:
        if _EXCEPTION_OR_SCOPE_CUE.search(text):
            return UnresolvedRequirementReason.EXCEPTION_SCOPE_UNRESOLVED
        if _NUMERIC_TOKEN.search(text):
            return UnresolvedRequirementReason.UNSUPPORTED_NUMERIC_FORM
        cues = list(_NORMATIVE_CUE.finditer(text))
        if len(cues) > 1:
            return UnresolvedRequirementReason.COMPOUND_STRUCTURE_UNRESOLVED
        if requirement.modality in {
            RequirementModality.REQUIRED,
            RequirementModality.PROHIBITED,
            RequirementModality.UNSUPPORTED,
        } and cues:
            return UnresolvedRequirementReason.QUALITATIVE_UNSUPPORTED
        return UnresolvedRequirementReason.AMBIGUOUS_SPLIT

    @staticmethod
    def _unresolved(binding, start, end, reason):
        text = binding.evidence.source_text[start:end]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return UnresolvedRequirementSpan(
            unresolved_span_id=deterministic_unresolved_span_id(
                evidence_id=binding.evidence.id,
                article_id=binding.evidence.article_id,
                article_number=binding.evidence.article_number,
                char_start=start,
                char_end=end,
                source_text_sha256=text_hash,
                reason=reason,
            ),
            evidence_id=binding.evidence.id,
            article_id=binding.evidence.article_id,
            article_number=binding.evidence.article_number,
            char_start=start,
            char_end=end,
            source_text=text,
            source_text_sha256=text_hash,
            reason=reason,
        )

    @staticmethod
    def _result(
        *, candidate, route, scope, evidence_ids, requirements, unresolved, status
    ):
        decomposition_id = deterministic_decomposition_id(
            candidate_id=candidate.candidate_id,
            route_id=route.route_id,
            evidence_ids=evidence_ids,
            requirements=requirements,
            unresolved_spans=unresolved,
            decomposition_version=DECOMPOSITION_VERSION,
        )
        return RequirementDecompositionResult(
            decomposition_id=decomposition_id,
            candidate_id=candidate.candidate_id,
            route_id=route.route_id,
            document_id=candidate.document_id,
            document_sha256=candidate.document_sha256,
            standard_id=scope.standard_id,
            standard_code=scope.standard_code,
            canonical_standard_code=scope.canonical_standard_code,
            standard_name=scope.standard_name,
            evidence_ids=evidence_ids,
            requirements=requirements,
            unresolved_spans=unresolved,
            status=status,
            decomposition_method=(
                RequirementDecompositionMethod.EXACT_SOURCE_DETERMINISTIC
            ),
        )
