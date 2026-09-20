"""Standalone deterministic authority for D9 numeric internal consistency review."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import UUID

from app.core.config import Settings, get_settings
from app.schemas.internal_consistency import (
    INTERNAL_CONSISTENCY_EXTRACTOR_VERSION,
    ConsistencyAssertionClass,
    ConsistencyCoverageStatus,
    ConsistencyFact,
    ConsistencyParameterProfile,
    ConsistencyUnresolvedCount,
    ConsistencyUnresolvedReason,
    ConsistencyValueGroup,
    EngineeringObjectProfile,
    InternalConsistencyCandidate,
    InternalConsistencyReviewResult,
    NormalizedUnit,
    deterministic_candidate_id,
    deterministic_fact_id,
    deterministic_profile_id,
    deterministic_review_id,
    fact_source_sort_key,
)
from app.services.pdf_service import PDFPageText, PDFService


MAX_CONSISTENCY_CLAUSE_CHARS = 320
MAX_OBJECT_PARAMETER_GAP_CHARS = 48
MAX_PARAMETER_VALUE_GAP_CHARS = 48

_SENTENCE = re.compile(r"[^。！？!?；;\r\n]+[。！？!?；;]?")
_SUBCLAUSE = re.compile(r"[^，,]+[，,]?")
_NUMBER = re.compile(r"(?<![\d.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)")
_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\d.])(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?P<unit>MPa|kPa|kN|mm|cm|N|m|%|℃)(?![A-Za-z])",
    re.IGNORECASE,
)
_RANGE = re.compile(
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*(?:[-~～至到])\s*"
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
)
_COMPOUND_DIMENSION = re.compile(
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*[×xX*]\s*"
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
)
_UPPER_BOUND = re.compile(
    r"(?:≤|<=|不(?:得)?(?:大于|超过|高于)|不大于|不超过|至多|上限|"
    r"控制在[^，。！？；;]{0,30}以内)"
)
_LOWER_BOUND = re.compile(
    r"(?:≥|>=|不(?:得)?(?:小于|低于)|不小于|不低于|至少|下限)"
)
_EXACT_MEASURED = re.compile(
    r"(?:实测|测得|测量值|监测值)[^，。！？；;]{0,30}(?:控制为|取值为|为|=)"
)
_DECLARED_CONTROL = re.compile(
    r"(?:(?:报警值|预警值|控制值|设计值)[^，。！？；;]{0,40}"
    r"(?:控制为|取值为|为|=)|"
    r"(?:间距|步距|厚度|深度|高度|轴力|监测频率)[^，。！？；;]{0,24}"
    r"(?:控制为|取值为|为|采用|=))"
)


class InternalConsistencyReviewError(RuntimeError):
    pass


class InternalConsistencyDocumentNotFoundError(InternalConsistencyReviewError):
    pass


class InternalConsistencyDocumentChangedError(InternalConsistencyReviewError):
    pass


class InternalConsistencySourceAuthorityError(InternalConsistencyReviewError):
    pass


@dataclass(frozen=True)
class _ObjectProfileRule:
    name: EngineeringObjectProfile
    anchors: tuple[str, ...]


@dataclass(frozen=True)
class _ParameterProfileRule:
    name: ConsistencyParameterProfile
    anchors: tuple[str, ...]


_OBJECT_PROFILES: tuple[_ObjectProfileRule, ...] = (
    _ObjectProfileRule(EngineeringObjectProfile.SUPPORT_STRUCTURE, ("支护结构",)),
    _ObjectProfileRule(EngineeringObjectProfile.RETAINING_WALL, ("围护墙",)),
    _ObjectProfileRule(EngineeringObjectProfile.CROWN_BEAM, ("冠梁",)),
    _ObjectProfileRule(EngineeringObjectProfile.UPRIGHT_MEMBER, ("立杆",)),
    _ObjectProfileRule(EngineeringObjectProfile.HORIZONTAL_MEMBER, ("水平杆",)),
    _ObjectProfileRule(EngineeringObjectProfile.EXCAVATION, ("开挖",)),
    _ObjectProfileRule(EngineeringObjectProfile.STEEL_SUPPORT, ("钢支撑",)),
    _ObjectProfileRule(EngineeringObjectProfile.ANGLE_WELD, ("角焊缝",)),
)

_PARAMETER_PROFILES: tuple[_ParameterProfileRule, ...] = (
    _ParameterProfileRule(
        ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_ALARM_VALUE,
        ("水平位移", "报警值"),
    ),
    _ParameterProfileRule(
        ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_PREWARNING_VALUE,
        ("水平位移", "预警值"),
    ),
    _ParameterProfileRule(
        ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_CONTROL_VALUE,
        ("水平位移", "控制值"),
    ),
    _ParameterProfileRule(ConsistencyParameterProfile.SPACING, ("间距",)),
    _ParameterProfileRule(ConsistencyParameterProfile.STEP_SPACING, ("步距",)),
    _ParameterProfileRule(ConsistencyParameterProfile.THICKNESS, ("厚度",)),
    _ParameterProfileRule(ConsistencyParameterProfile.DEPTH, ("深度",)),
    _ParameterProfileRule(ConsistencyParameterProfile.HEIGHT, ("高度",)),
    _ParameterProfileRule(
        ConsistencyParameterProfile.SUPPORT_AXIAL_FORCE, ("轴力",)
    ),
    _ParameterProfileRule(
        ConsistencyParameterProfile.MONITORING_FREQUENCY, ("监测频率",)
    ),
)

_UNIT_NORMALIZATION: dict[str, NormalizedUnit] = {
    "mm": NormalizedUnit.MILLIMETRE,
    "cm": NormalizedUnit.CENTIMETRE,
    "m": NormalizedUnit.METRE,
    "n": NormalizedUnit.NEWTON,
    "kn": NormalizedUnit.KILONEWTON,
    "kpa": NormalizedUnit.KILOPASCAL,
    "mpa": NormalizedUnit.MEGAPASCAL,
    "%": NormalizedUnit.PERCENT,
    "℃": NormalizedUnit.CELSIUS,
}


class InternalConsistencyReviewService:
    """Review one server-stored document without creating compliance authority."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        pdf_service: PDFService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pdf_service = pdf_service or PDFService()

    def review_document(self, document_id: str) -> InternalConsistencyReviewResult:
        """Resolve, hash, parse, review, and re-hash one server-owned PDF."""

        pdf_path = self._require_stored_pdf(document_id)
        initial_sha = self._sha256(pdf_path)
        extracted = self.pdf_service.extract_text(pdf_path)
        result = self._review_verified_pages(
            document_id=document_id,
            document_sha256=initial_sha,
            pages=extracted.pages,
        )
        if self._sha256(pdf_path) != initial_sha:
            raise InternalConsistencyDocumentChangedError(
                "Stored plan PDF changed during internal consistency review."
            )
        return result

    def _review_verified_pages(
        self,
        *,
        document_id: str,
        document_sha256: str,
        pages: Sequence[PDFPageText],
    ) -> InternalConsistencyReviewResult:
        """Pure authority construction for already server-bound page text."""

        ordered_pages = self._validate_pages(document_id, document_sha256, pages)
        facts_by_id: dict[str, ConsistencyFact] = {}
        unresolved: Counter[ConsistencyUnresolvedReason] = Counter()

        for page in ordered_pages:
            page_text_sha256 = hashlib.sha256(page.text.encode("utf-8")).hexdigest()
            for start, end in self._bounded_clauses(page.text):
                fact, reason = self._fact_from_clause(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page=page,
                    page_text_sha256=page_text_sha256,
                    source_start=start,
                    source_end=end,
                )
                if fact is not None:
                    facts_by_id.setdefault(fact.fact_id, fact)
                elif reason is not None:
                    unresolved[reason] += 1

        facts = tuple(sorted(facts_by_id.values(), key=fact_source_sort_key))
        candidates, grouping_unresolved = self._group_facts(facts)
        unresolved.update(grouping_unresolved)
        unresolved_counts = tuple(
            ConsistencyUnresolvedCount(reason=reason, count=count)
            for reason, count in sorted(unresolved.items(), key=lambda item: item[0].value)
            if count > 0
        )
        coverage_status = (
            ConsistencyCoverageStatus.CANDIDATES_ESTABLISHED_WITHIN_D9_V1_COVERAGE
            if candidates
            else ConsistencyCoverageStatus.NO_INCONSISTENCY_ESTABLISHED_WITHIN_D9_V1_COVERAGE
        )
        review_id = deterministic_review_id(
            document_sha256=document_sha256,
            source_page_count=len(ordered_pages),
            fact_count=len(facts),
            candidates=candidates,
            unresolved_counts=unresolved_counts,
        )
        return InternalConsistencyReviewResult(
            review_id=review_id,
            document_id=document_id,
            document_sha256=document_sha256,
            source_page_count=len(ordered_pages),
            fact_count=len(facts),
            candidate_count=len(candidates),
            unresolved_count=sum(item.count for item in unresolved_counts),
            coverage_status=coverage_status,
            candidates=candidates,
            unresolved_counts=unresolved_counts,
            extractor_version=INTERNAL_CONSISTENCY_EXTRACTOR_VERSION,
        )

    @staticmethod
    def _validate_pages(
        document_id: str,
        document_sha256: str,
        pages: Sequence[PDFPageText],
    ) -> tuple[PDFPageText, ...]:
        try:
            parsed = UUID(document_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise InternalConsistencySourceAuthorityError(
                "Internal consistency review requires a canonical document identity."
            ) from exc
        if str(parsed) != document_id or not re.fullmatch(r"[0-9a-f]{64}", document_sha256):
            raise InternalConsistencySourceAuthorityError(
                "Internal consistency source identity is not canonical."
            )
        if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
            raise InternalConsistencySourceAuthorityError(
                "Internal consistency review requires parsed physical pages."
            )
        ordered = tuple(sorted(pages, key=lambda item: item.page_number))
        if not ordered or any(not isinstance(item, PDFPageText) for item in ordered):
            raise InternalConsistencySourceAuthorityError(
                "Internal consistency review requires parsed physical pages."
            )
        page_numbers = [item.page_number for item in ordered]
        if any(number <= 0 for number in page_numbers) or len(page_numbers) != len(
            set(page_numbers)
        ):
            raise InternalConsistencySourceAuthorityError(
                "Physical page numbers must be positive and unique."
            )
        return ordered

    @classmethod
    def _fact_from_clause(
        cls,
        *,
        document_id: str,
        document_sha256: str,
        page: PDFPageText,
        page_text_sha256: str,
        source_start: int,
        source_end: int,
    ) -> tuple[ConsistencyFact | None, ConsistencyUnresolvedReason | None]:
        source_text = page.text[source_start:source_end]
        view = cls._matching_view(source_text)
        bare_number_matches = list(_NUMBER.finditer(view))
        number_matches = list(_NUMBER_WITH_UNIT.finditer(view))
        has_object_anchor = any(
            all(anchor in view for anchor in rule.anchors) for rule in _OBJECT_PROFILES
        )
        has_parameter_anchor = any(
            all(anchor in view for anchor in rule.anchors)
            for rule in _PARAMETER_PROFILES
        )

        if not bare_number_matches or not (has_object_anchor or has_parameter_anchor):
            return None, None
        if _RANGE.search(view) or _COMPOUND_DIMENSION.search(view):
            return None, ConsistencyUnresolvedReason.INSUFFICIENT_CONTEXT
        if len(number_matches) > 1 or (
            not number_matches and len(bare_number_matches) != 1
        ):
            return None, ConsistencyUnresolvedReason.INSUFFICIENT_CONTEXT

        numeric_start = (
            number_matches[0].start("value")
            if number_matches
            else bare_number_matches[0].start()
        )
        parameter_matches = tuple(
            (rule, span)
            for rule in _PARAMETER_PROFILES
            if (
                span := cls._ordered_anchor_span(
                    view,
                    anchors=rule.anchors,
                    before=numeric_start,
                )
            )
            is not None
            and numeric_start - span[1] <= MAX_PARAMETER_VALUE_GAP_CHARS
        )
        if len(parameter_matches) != 1:
            object_before_number = any(
                cls._ordered_anchor_span(
                    view,
                    anchors=rule.anchors,
                    before=numeric_start,
                )
                is not None
                for rule in _OBJECT_PROFILES
            )
            return (
                None,
                ConsistencyUnresolvedReason.PARAMETER_IDENTITY_UNRESOLVED
                if object_before_number or parameter_matches
                else None,
            )

        parameter_rule, parameter_span = parameter_matches[0]
        object_matches = tuple(
            (rule, span)
            for rule in _OBJECT_PROFILES
            if (
                span := cls._ordered_anchor_span(
                    view,
                    anchors=rule.anchors,
                    before=parameter_span[0],
                )
            )
            is not None
            and parameter_span[0] - span[1] <= MAX_OBJECT_PARAMETER_GAP_CHARS
        )
        if len(object_matches) != 1:
            return None, ConsistencyUnresolvedReason.OBJECT_IDENTITY_UNRESOLVED
        if not number_matches:
            return None, ConsistencyUnresolvedReason.MISSING_UNIT

        assertion = cls._assertion_class(view)
        if assertion is None:
            return None, ConsistencyUnresolvedReason.INSUFFICIENT_CONTEXT
        match = number_matches[0]
        try:
            numeric_value = Decimal(match.group("value"))
        except InvalidOperation:
            return None, ConsistencyUnresolvedReason.INSUFFICIENT_CONTEXT
        unit = cls._normalize_unit(match.group("unit"))
        if unit is None:
            return None, ConsistencyUnresolvedReason.MISSING_UNIT

        object_rule = object_matches[0][0]
        object_profile_id = deterministic_profile_id(
            profile_kind="object",
            profile_name=object_rule.name.value,
            anchors=object_rule.anchors,
        )
        parameter_profile_id = deterministic_profile_id(
            profile_kind="parameter",
            profile_name=parameter_rule.name.value,
            anchors=parameter_rule.anchors,
        )
        source_text_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        fact_id = deterministic_fact_id(
            document_sha256=document_sha256,
            physical_page=page.page_number,
            page_text_sha256=page_text_sha256,
            source_start=source_start,
            source_end=source_end,
            source_text_sha256=source_text_sha256,
            object_profile_id=object_profile_id,
            parameter_profile_id=parameter_profile_id,
            numeric_value=numeric_value,
            normalized_unit=unit,
            assertion_class=assertion,
        )
        return (
            ConsistencyFact(
                fact_id=fact_id,
                document_id=document_id,
                document_sha256=document_sha256,
                physical_page=page.page_number,
                page_text_sha256=page_text_sha256,
                source_start=source_start,
                source_end=source_end,
                source_text=source_text,
                source_text_sha256=source_text_sha256,
                object_profile_id=object_profile_id,
                object_profile_name=object_rule.name,
                object_anchors=object_rule.anchors,
                parameter_profile_id=parameter_profile_id,
                parameter_profile_name=parameter_rule.name,
                parameter_anchors=parameter_rule.anchors,
                numeric_value=numeric_value,
                normalized_unit=unit,
                assertion_class=assertion,
                extractor_version=INTERNAL_CONSISTENCY_EXTRACTOR_VERSION,
            ),
            None,
        )

    @staticmethod
    def _group_facts(
        facts: tuple[ConsistencyFact, ...],
    ) -> tuple[
        tuple[InternalConsistencyCandidate, ...],
        Counter[ConsistencyUnresolvedReason],
    ]:
        unresolved: Counter[ConsistencyUnresolvedReason] = Counter()
        unique_facts = {fact.fact_id: fact for fact in facts}
        by_group: dict[
            tuple[
                str,
                str,
                str,
                NormalizedUnit,
                ConsistencyAssertionClass,
            ],
            list[ConsistencyFact],
        ] = defaultdict(list)
        assertions_by_profile: dict[tuple[str, str, str], set[ConsistencyAssertionClass]] = (
            defaultdict(set)
        )
        units_by_profile_assertion: dict[
            tuple[str, str, str, ConsistencyAssertionClass], set[NormalizedUnit]
        ] = defaultdict(set)
        for fact in unique_facts.values():
            profile_key = (
                fact.document_sha256,
                fact.object_profile_id,
                fact.parameter_profile_id,
            )
            group_key = (
                *profile_key,
                fact.normalized_unit,
                fact.assertion_class,
            )
            by_group[group_key].append(fact)
            assertions_by_profile[profile_key].add(fact.assertion_class)
            units_by_profile_assertion[(*profile_key, fact.assertion_class)].add(
                fact.normalized_unit
            )

        assertion_mismatches = sum(
            1 for values in assertions_by_profile.values() if len(values) > 1
        )
        if assertion_mismatches:
            unresolved[
                ConsistencyUnresolvedReason.ASSERTION_TYPE_MISMATCH
            ] = assertion_mismatches
        unit_mismatches = sum(
            1 for values in units_by_profile_assertion.values() if len(values) > 1
        )
        if unit_mismatches:
            unresolved[ConsistencyUnresolvedReason.UNIT_MISMATCH] = unit_mismatches

        candidates: list[InternalConsistencyCandidate] = []
        ordered_group_keys = sorted(
            by_group,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3].value,
                item[4].value,
            ),
        )
        for document_sha256, object_profile_id, parameter_profile_id, unit, assertion in (
            ordered_group_keys
        ):
            group_facts = by_group[
                (
                    document_sha256,
                    object_profile_id,
                    parameter_profile_id,
                    unit,
                    assertion,
                )
            ]
            by_value: dict[Decimal, list[ConsistencyFact]] = defaultdict(list)
            for fact in group_facts:
                by_value[fact.numeric_value].append(fact)
            if len(by_value) <= 1:
                continue

            value_groups = tuple(
                ConsistencyValueGroup(
                    numeric_value=value,
                    normalized_unit=unit,
                    facts=tuple(sorted(value_facts, key=fact_source_sort_key)),
                    fact_ids=tuple(
                        fact.fact_id
                        for fact in sorted(value_facts, key=fact_source_sort_key)
                    ),
                )
                for value, value_facts in sorted(by_value.items(), key=lambda item: item[0])
            )
            first = value_groups[0].facts[0]
            candidate_id = deterministic_candidate_id(
                document_sha256=document_sha256,
                object_profile_id=object_profile_id,
                parameter_profile_id=parameter_profile_id,
                normalized_unit=unit,
                assertion_class=assertion,
                value_groups=value_groups,
            )
            candidates.append(
                InternalConsistencyCandidate(
                    candidate_id=candidate_id,
                    document_id=first.document_id,
                    document_sha256=document_sha256,
                    object_profile_id=object_profile_id,
                    object_profile_name=first.object_profile_name,
                    object_anchors=first.object_anchors,
                    parameter_profile_id=parameter_profile_id,
                    parameter_profile_name=first.parameter_profile_name,
                    parameter_anchors=first.parameter_anchors,
                    normalized_unit=unit,
                    assertion_class=assertion,
                    value_groups=value_groups,
                )
            )
        return tuple(sorted(candidates, key=lambda item: item.candidate_id)), unresolved

    @staticmethod
    def _matching_view(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        return re.sub(r"\s+", "", normalized)

    @staticmethod
    def _ordered_anchor_span(
        value: str,
        *,
        anchors: tuple[str, ...],
        before: int,
    ) -> tuple[int, int] | None:
        """Return the nearest ordered anchor chain wholly before one authority token."""

        cursor = before
        end = before
        for index, anchor in enumerate(reversed(anchors)):
            start = value.rfind(anchor, 0, cursor)
            if start < 0:
                return None
            if index == 0:
                end = start + len(anchor)
            cursor = start
        return cursor, end

    @staticmethod
    def _normalize_unit(value: str) -> NormalizedUnit | None:
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        return _UNIT_NORMALIZATION.get(normalized)

    @staticmethod
    def _assertion_class(value: str) -> ConsistencyAssertionClass | None:
        if _UPPER_BOUND.search(value):
            return ConsistencyAssertionClass.UPPER_BOUND_CONTROL
        if _LOWER_BOUND.search(value):
            return ConsistencyAssertionClass.LOWER_BOUND_CONTROL
        if _EXACT_MEASURED.search(value):
            return ConsistencyAssertionClass.EXACT_MEASURED_VALUE
        if _DECLARED_CONTROL.search(value):
            return ConsistencyAssertionClass.DECLARED_CONTROL_VALUE
        return None

    @classmethod
    def _bounded_clauses(cls, text: str) -> tuple[tuple[int, int], ...]:
        spans: set[tuple[int, int]] = set()
        for match in _SENTENCE.finditer(text):
            start, end = cls._trim_span(text, match.start(), match.end())
            if start >= end:
                continue
            if end - start <= MAX_CONSISTENCY_CLAUSE_CHARS:
                spans.add((start, end))
                continue
            sentence = text[start:end]
            for subclause in _SUBCLAUSE.finditer(sentence):
                sub_start, sub_end = cls._trim_span(
                    text, start + subclause.start(), start + subclause.end()
                )
                if 0 < sub_end - sub_start <= MAX_CONSISTENCY_CLAUSE_CHARS:
                    spans.add((sub_start, sub_end))
        return tuple(sorted(spans))

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _require_stored_pdf(self, document_id: str) -> Path:
        try:
            parsed = UUID(document_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise InternalConsistencyDocumentNotFoundError(
                "Stored plan document was not found."
            ) from exc
        if str(parsed) != document_id:
            raise InternalConsistencyDocumentNotFoundError(
                "Stored plan document was not found."
            )
        upload_root = self.settings.upload_dir.resolve()
        path = (upload_root / f"{document_id}.pdf").resolve()
        if path.parent != upload_root or not path.is_file():
            raise InternalConsistencyDocumentNotFoundError(
                "Stored plan document was not found."
            )
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise InternalConsistencyDocumentChangedError(
                "Stored plan PDF could not be read consistently."
            ) from exc
        return digest.hexdigest()
