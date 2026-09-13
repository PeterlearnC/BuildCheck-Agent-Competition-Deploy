"""Conservative D.2 routing into server-qualified standard scopes."""

import hashlib
import json
import re
from dataclasses import dataclass

from app.schemas.review_candidate import ReviewCandidate, ReviewCandidateClass
from app.schemas.routing_context import RoutingContext, RoutingContextBoundaryMethod
from app.schemas.standard_route import (
    STANDARD_ROUTE_IDENTITY_VERSION,
    STANDARD_ROUTING_VERSION,
    QualifiedStandardScopeCandidate,
    StandardRoute,
    StandardRouteStatus,
    StandardRoutingMethod,
    deterministic_route_id,
)
from app.schemas.standards import (
    StandardDocument,
    StandardIdentityStatus,
    StandardParseStatus,
    StandardRegistryEntry,
    StandardRegistryStatus,
)
from app.schemas.standards_scope import StandardScope
from app.services.standards.standard_identity_service import StandardIdentityService
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository
from app.services.pdf_service import PDFServiceError
from app.services.review.routing_context_service import (
    RoutingContextError,
    RoutingContextService,
)


@dataclass(frozen=True)
class _DomainRule:
    domain: str
    canonical_standard_code: str
    pattern: re.Pattern[str]
    eligible_classes: frozenset[ReviewCandidateClass]
    confidence: float


_SCAFFOLD_CLASSES = frozenset(
    {
        ReviewCandidateClass.NUMERIC_CONTROL,
        ReviewCandidateClass.CONSTRUCTION_REQUIREMENT,
        ReviewCandidateClass.MATERIAL_REQUIREMENT,
        ReviewCandidateClass.STRUCTURAL_CONFIGURATION,
        ReviewCandidateClass.INSPECTION_REQUIREMENT,
        ReviewCandidateClass.SAFETY_REQUIREMENT,
        ReviewCandidateClass.PROCEDURAL_REQUIREMENT,
    }
)

# Standard codes here are routing metadata only. Every match must subsequently
# resolve through both an ACTIVE registry entry and an installed qualified document.
_DOMAIN_RULES = (
    _DomainRule(
        domain="scaffold_and_support",
        canonical_standard_code="GB55023-2022",
        pattern=re.compile(
            r"可调托撑|脚手架|模板支撑|支撑架|剪刀撑|连墙件|"
            r"立杆(?=[^。；，,\n]{0,30}(?:钢管|间距|水平杆|支撑))|"
            r"(?:钢管|间距|水平杆|支撑)[^。；，,\n]{0,30}立杆"
        ),
        eligible_classes=_SCAFFOLD_CLASSES,
        confidence=0.9,
    ),
)

_CONTEXT_CANDIDATE_ANCHOR = re.compile(r"立杆|钢管|螺杆")
MAX_CONTEXT_ANCHOR_DISTANCE = 64

_QUOTED_STANDARD_REFERENCE = re.compile(r"《[^》]{1,100}》")
_STANDARD_CODE_REFERENCE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:GB|JGJ|CJJ|DB|T)[A-Za-z0-9/ .-]*\d[A-Za-z0-9/ .-]*"
)
_ARTICLE_REFERENCE = re.compile(
    r"第\s*[0-9一二三四五六七八九十百千万]+"
    r"(?:\s*[.．、-]\s*[0-9一二三四五六七八九十百千万]+)*\s*条"
)
_CODE_WITH_TITLE_REFERENCE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:GB|JGJ|CJJ|DB|T)[A-Za-z0-9/ .-]*\d"
    r"[A-Za-z0-9/ .-]*[\s,，、:：-]*"
    r"[\u4e00-\u9fff\s,，、]{2,60}(?:规范|标准|规程)(?!化)"
)
_INTRODUCED_STANDARD_TITLE = re.compile(
    r"(?:依据|按照|根据|参照|遵照|执行)\s*"
    r"(?:《[^》]{1,100}》|"
    r"(?:[A-Za-z]{1,5}[A-Za-z0-9/ .-]*\d[\s,，、:：-]*)?"
    r"[\u4e00-\u9fff]{2,50}(?:规范|标准|规程)(?!化))"
)
_KNOWN_NAME_SEPARATORS = frozenset(" \t\r\n,，、;；:：·")
_UNKNOWN_BARE_TITLE_LINE = re.compile(
    r"(?m)(?:^|\n)\s*(?:\d+(?:\.\d+)*[、．.]\s*)?"
    r"(?P<title>[\u4e00-\u9fff]{2,40}(?:规范|标准|规程))"
    r"(?=\s*(?:$|\n|[，,。；;！？!?]|规定|要求))"
)
_ENGINEERING_PROPOSITION_MARKERS = (
    "应当",
    "必须",
    "不得",
    "不应",
    "不宜",
    "安装",
    "搭设",
    "设置",
    "检查",
    "验收",
    "控制",
    "采用",
    "保证",
    "符合",
    "满足",
    "调整",
    "连接",
    "布置",
    "应",
)


class StandardRoutingService:
    """Route a D.1 candidate; never retrieve articles or create findings."""

    def __init__(
        self,
        repository: StandardRepository | None = None,
        registry: StandardRegistryService | None = None,
    ) -> None:
        self.repository = repository or StandardRepository()
        self.registry = registry or StandardRegistryService()
        self.identity = StandardIdentityService()
        self._routing_context_service = RoutingContextService()

    def route(self, candidate: ReviewCandidate) -> StandardRoute:
        if not isinstance(candidate, ReviewCandidate):
            raise TypeError("D.2 input must be a qualified ReviewCandidate.")

        qualified, fingerprint = self._qualified_scopes()
        exact_text = self._eligible_domain_text(candidate.source_text, qualified)
        exact_rules = [
            rule
            for rule in _DOMAIN_RULES
            if candidate.candidate_class in rule.eligible_classes
            and rule.pattern.search(exact_text)
        ]
        if exact_rules:
            return self._route_for_rules(
                candidate=candidate,
                matched_rules=exact_rules,
                qualified=qualified,
                fingerprint=fingerprint,
                routing_method=StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES,
                routing_context_id=None,
            )

        context_rules: list[_DomainRule] = []
        verified_context: RoutingContext | None = None
        try:
            verified_context = self._routing_context_service.build_for_candidate(
                candidate=candidate
            )
        except (RoutingContextError, PDFServiceError):
            verified_context = None
        if verified_context is not None:
            context_rules = self._context_rules(candidate, verified_context, qualified)
        if context_rules and verified_context is not None:
            return self._route_for_rules(
                candidate=candidate,
                matched_rules=context_rules,
                qualified=qualified,
                fingerprint=fingerprint,
                routing_method=StandardRoutingMethod.CONTEXT_ASSISTED_DETERMINISTIC,
                routing_context_id=verified_context.routing_context_id,
            )
        return self._route_for_rules(
            candidate=candidate,
            matched_rules=[],
            qualified=qualified,
            fingerprint=fingerprint,
            routing_method=StandardRoutingMethod.DETERMINISTIC_DOMAIN_RULES,
            routing_context_id=None,
        )

    @classmethod
    def _context_rules(
        cls,
        candidate: ReviewCandidate,
        context: RoutingContext,
        qualified: tuple[QualifiedStandardScopeCandidate, ...],
    ) -> list[_DomainRule]:
        if (
            context.boundary_method != RoutingContextBoundaryMethod.NUMBERED_ITEM
            or not _CONTEXT_CANDIDATE_ANCHOR.search(candidate.source_text)
        ):
            return []
        candidate_start = context.candidate_relative_start
        candidate_end = context.candidate_relative_end
        matched: list[_DomainRule] = []
        eligible_context = cls._eligible_domain_text(context.context_text, qualified)
        for rule in _DOMAIN_RULES:
            if candidate.candidate_class not in rule.eligible_classes:
                continue
            for signal in rule.pattern.finditer(eligible_context):
                if signal.end() <= candidate_start:
                    distance = candidate_start - signal.end()
                elif signal.start() >= candidate_end:
                    distance = signal.start() - candidate_end
                else:
                    distance = 0
                if distance <= MAX_CONTEXT_ANCHOR_DISTANCE:
                    matched.append(rule)
                    break
        return matched

    @staticmethod
    def _eligible_domain_text(
        text: str, qualified: tuple[QualifiedStandardScopeCandidate, ...]
    ) -> str:
        """Blank untrusted standard-reference spans without creating authority."""

        spans: list[tuple[int, int]] = []
        patterns = (
            _QUOTED_STANDARD_REFERENCE,
            _STANDARD_CODE_REFERENCE,
            _ARTICLE_REFERENCE,
            _CODE_WITH_TITLE_REFERENCE,
            _INTRODUCED_STANDARD_TITLE,
        )
        for pattern in patterns:
            spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
        spans.extend(StandardRoutingService._unknown_bare_title_spans(text))
        for scope in qualified:
            for token in (scope.canonical_standard_code, scope.standard_code):
                start = text.find(token)
                while start >= 0:
                    spans.append((start, start + len(token)))
                    start = text.find(token, start + 1)
            spans.extend(
                StandardRoutingService._normalized_known_name_spans(
                    text, scope.standard_name
                )
            )
        if not spans:
            return text
        masked = list(text)
        for start, end in spans:
            masked[start:end] = " " * (end - start)
        return "".join(masked)

    @staticmethod
    def _unknown_bare_title_spans(text: str) -> list[tuple[int, int]]:
        """Exclude bounded nominal titles, never ordinary engineering propositions."""

        spans: list[tuple[int, int]] = []
        for match in _UNKNOWN_BARE_TITLE_LINE.finditer(text):
            title = match.group("title")
            if any(marker in title for marker in _ENGINEERING_PROPOSITION_MARKERS):
                continue
            spans.append(match.span("title"))
        return spans

    @staticmethod
    def _normalized_known_name_spans(text: str, name: str) -> list[tuple[int, int]]:
        """Find a server-known name while retaining original source offsets."""

        normalized_text: list[str] = []
        original_indexes: list[int] = []
        for index, character in enumerate(text):
            if character in _KNOWN_NAME_SEPARATORS:
                continue
            normalized_text.append(character)
            original_indexes.append(index)
        normalized_name = "".join(
            character for character in name if character not in _KNOWN_NAME_SEPARATORS
        )
        if not normalized_name:
            return []
        searchable = "".join(normalized_text)
        spans: list[tuple[int, int]] = []
        start = searchable.find(normalized_name)
        while start >= 0:
            end = start + len(normalized_name)
            spans.append((original_indexes[start], original_indexes[end - 1] + 1))
            start = searchable.find(normalized_name, start + 1)
        return spans

    def _route_for_rules(
        self,
        *,
        candidate: ReviewCandidate,
        matched_rules: list[_DomainRule],
        qualified: tuple[QualifiedStandardScopeCandidate, ...],
        fingerprint: str,
        routing_method: StandardRoutingMethod,
        routing_context_id: str | None,
    ) -> StandardRoute:
        target_codes = sorted({rule.canonical_standard_code for rule in matched_rules})
        scopes = tuple(
            item
            for item in qualified
            if item.canonical_standard_code in target_codes
        )

        if len(scopes) == 1:
            status = StandardRouteStatus.ROUTED
            selected = StandardScope(standard_ids=[scopes[0].standard_id])
            confidence = max(rule.confidence for rule in matched_rules)
            reason = "A strong domain rule resolved to one ACTIVE installed standard scope."
        elif len(scopes) > 1:
            status = StandardRouteStatus.AMBIGUOUS_STANDARD_SCOPE
            selected = None
            confidence = max(rule.confidence for rule in matched_rules)
            reason = "The supported domain resolves to multiple equally qualified scopes."
        else:
            status = StandardRouteStatus.NO_STANDARD_SCOPE
            selected = None
            confidence = 0.0
            reason = (
                "No strong routing domain was detected."
                if not matched_rules
                else "The routing domain has no uniquely qualified registry scope."
            )

        route_id = deterministic_route_id(
            candidate_id=candidate.candidate_id,
            scope_candidates=scopes,
            registry_fingerprint=fingerprint,
            routing_context_id=routing_context_id,
        )
        return StandardRoute(
            route_identity_version=STANDARD_ROUTE_IDENTITY_VERSION,
            route_id=route_id,
            candidate_id=candidate.candidate_id,
            routing_version=STANDARD_ROUTING_VERSION,
            route_status=status,
            candidate_topic=candidate.topic,
            candidate_class=candidate.candidate_class,
            scope_candidates=scopes,
            selected_scope=selected,
            routing_method=routing_method,
            routing_confidence=confidence,
            routing_reason=reason,
            registry_fingerprint=fingerprint,
            routing_context_id=routing_context_id,
        )

    def _qualified_scopes(
        self,
    ) -> tuple[tuple[QualifiedStandardScopeCandidate, ...], str]:
        entries = self.registry.list_entries()
        active_entries = [
            entry for entry in entries if entry.status == StandardRegistryStatus.ACTIVE
        ]
        documents = self.repository.list_documents()
        resolved: list[QualifiedStandardScopeCandidate] = []
        for document in documents:
            if not self._document_is_qualified(document):
                continue
            canonical = self.identity.canonicalize(
                document.canonical_standard_code or document.standard_code
            )
            matching = [
                entry
                for entry in active_entries
                if self.identity.canonicalize(entry.standard_code) == canonical
            ]
            if not canonical or not matching:
                continue
            resolved.append(
                QualifiedStandardScopeCandidate(
                    standard_id=document.standard_id,
                    standard_code=document.standard_code or matching[0].standard_code,
                    canonical_standard_code=canonical,
                    standard_name=document.standard_name or matching[0].standard_name,
                    source_checksum=document.source_checksum,
                )
            )
        ordered = tuple(
            sorted(
                resolved,
                key=lambda item: (
                    item.canonical_standard_code,
                    item.standard_id,
                    item.source_checksum,
                ),
            )
        )
        return ordered, self._registry_fingerprint(active_entries, ordered)

    @staticmethod
    def _document_is_qualified(document: StandardDocument) -> bool:
        binding = document.official_source_binding
        return bool(
            document.identity_status == StandardIdentityStatus.CONFIRMED
            and document.parse_status == StandardParseStatus.PARSED
            and document.canonical_standard_code
            and document.standard_code
            and document.standard_name
            and re.fullmatch(r"[0-9a-f]{64}", document.source_checksum)
            and binding is not None
            and binding.confirmed
            and binding.source_checksum == document.source_checksum
        )

    def _registry_fingerprint(
        self,
        active_entries: list[StandardRegistryEntry],
        scopes: tuple[QualifiedStandardScopeCandidate, ...],
    ) -> str:
        payload = {
            "active_registry": [
                {
                    "standard_code": self.identity.canonicalize(entry.standard_code),
                    "standard_name": entry.standard_name,
                    "discipline": entry.discipline,
                    "jurisdiction": entry.jurisdiction,
                    "effective_date": (
                        entry.effective_date.isoformat() if entry.effective_date else None
                    ),
                }
                for entry in sorted(
                    active_entries,
                    key=lambda item: (
                        self.identity.canonicalize(item.standard_code) or "",
                        item.standard_name,
                    ),
                )
            ],
            "qualified_installed_scopes": [item.model_dump(mode="json") for item in scopes],
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
