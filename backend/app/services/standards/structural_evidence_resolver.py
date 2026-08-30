"""Resolve source-grounded structural candidates without creating corpus text."""

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata

from app.schemas.ocr import SourceKind
from app.schemas.standards import DocumentRegionType
from app.schemas.structural import (
    StructuralEvidenceKind,
    StructuralEvidenceProducer,
    StructuralEvidenceRole,
    StructuralSourceReference,
    StructuralTransitionProvenance,
    StructuralValidationBasis,
    ValidatedStructuralEvidence,
)


STRUCTURAL_RESOLVER_VERSION = "v0.4-b.3c.3-s1-r1-structural"


@dataclass(frozen=True, slots=True)
class StructuralEvidenceCandidate:
    """Internal resolver input; a candidate is never corpus evidence."""

    source_checksum: str
    source_kind: SourceKind
    producer: StructuralEvidenceProducer
    structure_kind: StructuralEvidenceKind
    structure_number: str
    structure_title: str
    region_role: DocumentRegionType
    source_fragments: tuple[str, ...]
    source_references: tuple[StructuralSourceReference, ...]
    previous_article_number: str | None = None
    next_article_number: str | None = None
    inside_article_interval: bool = False
    transition: StructuralTransitionProvenance | None = None
    validation_basis: tuple[StructuralValidationBasis, ...] = ()


class StructuralEvidenceResolverService:
    """Fail closed unless source, region, grammar, and boundary agree."""

    NUMERIC_CHAPTER = re.compile(r"^\d+$")
    NUMERIC_SECTION = re.compile(r"^\d+\.\d+$")
    APPENDIX = re.compile(r"^[A-Z]$", re.IGNORECASE)
    APPENDIX_SECTION = re.compile(r"^[A-Z]\.\d+$", re.IGNORECASE)
    EXPLICIT_CHAPTER = re.compile(
        r"^第\s*(?P<number>\d+)\s*章\s+(?P<title>\S.*)$"
    )
    SIMPLE_HEADING = re.compile(
        r"^(?P<number>(?:\d+(?:\.\d+)?|[A-Z](?:\.\d+)?))\s+"
        r"(?P<title>\S.*)$",
        re.IGNORECASE,
    )
    COMPACT_CJK_HEADING = re.compile(
        r"^(?P<number>\d+(?:\.\d+)?)"
        r"(?P<title>[\u3400-\u9fff][\u3400-\u9fff\s]{1,60})$"
    )
    EXPLICIT_APPENDIX = re.compile(
        r"^附录\s*(?P<number>[A-Z])\s+(?P<title>\S.*)$",
        re.IGNORECASE,
    )

    def resolve(
        self, candidate: StructuralEvidenceCandidate
    ) -> ValidatedStructuralEvidence | None:
        if not self._valid_checksum(candidate.source_checksum):
            return None
        if not self._source_references_are_valid(candidate):
            return None
        if not self._producer_matches_source(candidate):
            return None
        if not self._region_is_allowed(candidate):
            return None
        if candidate.inside_article_interval:
            return None

        number = self._normalize_number(candidate.structure_number)
        title = self._normalize_text(candidate.structure_title)
        if not number or not title or not self._number_matches_kind(
            number, candidate.structure_kind
        ):
            return None

        source_text = "\n".join(candidate.source_fragments)
        normalized_text = self._normalize_text(source_text)
        parsed = self._parse_grounded_heading(normalized_text, candidate.structure_kind)
        if parsed is None or parsed != (number, title):
            return None

        bases = list(dict.fromkeys(candidate.validation_basis))
        self._append_basis(bases, StructuralValidationBasis.REGION_ROLE)
        self._append_basis(bases, StructuralValidationBasis.NUMBER_GRAMMAR)
        self._append_basis(bases, StructuralValidationBasis.SOURCE_ORDER)

        if candidate.next_article_number is not None:
            if not self.is_compatible(number, candidate.next_article_number):
                return None
            self._append_basis(bases, StructuralValidationBasis.ARTICLE_PREFIX)
        elif StructuralValidationBasis.EXPLICIT_HEADING_SYNTAX not in bases:
            return None

        if candidate.previous_article_number is not None:
            if self.is_compatible(number, candidate.previous_article_number):
                return None
            if candidate.next_article_number is None:
                return None
            self._append_basis(bases, StructuralValidationBasis.PREFIX_TRANSITION)

        if candidate.transition is not None and candidate.transition.ambiguous:
            corroborating = {
                StructuralValidationBasis.ARTICLE_PREFIX,
                StructuralValidationBasis.PREFIX_TRANSITION,
                StructuralValidationBasis.EXPLICIT_HEADING_SYNTAX,
                StructuralValidationBasis.TYPOGRAPHIC_GEOMETRY,
            }
            if not corroborating.intersection(bases):
                return None

        evidence_id = self.evidence_id(
            source_checksum=candidate.source_checksum,
            structure_kind=candidate.structure_kind,
            structure_number=number,
            structure_title=title,
            source_references=candidate.source_references,
        )
        return ValidatedStructuralEvidence(
            evidence_id=evidence_id,
            source_checksum=candidate.source_checksum.lower(),
            source_kind=candidate.source_kind,
            producer=candidate.producer,
            structure_kind=candidate.structure_kind,
            structure_number=number,
            structure_title=title,
            structure_level=self._level(candidate.structure_kind),
            region_role=candidate.region_role,
            evidence_role=StructuralEvidenceRole.STRUCTURAL_ONLY,
            accepted_as_normative=False,
            source_text=source_text,
            normalized_text=normalized_text,
            source_references=list(candidate.source_references),
            transition=candidate.transition,
            validation_basis=bases,
            resolver_version=STRUCTURAL_RESOLVER_VERSION,
        )

    def resolve_many(
        self, candidates: list[StructuralEvidenceCandidate]
    ) -> list[ValidatedStructuralEvidence]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.source_references[0].page_number
                if candidate.source_references
                else 0,
                candidate.source_references[0].source_order_index
                if candidate.source_references
                else 0,
            ),
        )
        return [evidence for item in ordered if (evidence := self.resolve(item))]

    def evidence_id(
        self,
        *,
        source_checksum: str,
        structure_kind: StructuralEvidenceKind,
        structure_number: str,
        structure_title: str,
        source_references: tuple[StructuralSourceReference, ...]
        | list[StructuralSourceReference],
        resolver_version: str = STRUCTURAL_RESOLVER_VERSION,
    ) -> str:
        payload = {
            "source_checksum": source_checksum.lower(),
            "resolver_version": resolver_version,
            "structure_kind": structure_kind.value,
            "structure_number": self._normalize_number(structure_number),
            "structure_title": self._normalize_text(structure_title),
            "source_references": [
                reference.model_dump(mode="json") for reference in source_references
            ],
        }
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return f"structure_{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"

    @staticmethod
    def is_compatible(structure_number: str, article_number: str) -> bool:
        structure = StructuralEvidenceResolverService._number_parts(structure_number)
        article = StructuralEvidenceResolverService._number_parts(article_number)
        return bool(
            structure
            and article
            and len(structure) < len(article)
            and article[: len(structure)] == structure
        )

    @staticmethod
    def _number_parts(value: str) -> tuple[str, ...] | None:
        normalized = StructuralEvidenceResolverService._normalize_number(value)
        if re.fullmatch(r"\d+(?:\.\d+){0,3}", normalized):
            return tuple(str(int(part)) for part in normalized.split("."))
        if re.fullmatch(r"[A-Z](?:\.\d+){0,3}", normalized):
            head, *tail = normalized.split(".")
            return (head, *(str(int(part)) for part in tail))
        return None

    def _parse_grounded_heading(
        self, value: str, kind: StructuralEvidenceKind
    ) -> tuple[str, str] | None:
        match = None
        if kind == StructuralEvidenceKind.CHAPTER:
            match = self.EXPLICIT_CHAPTER.fullmatch(value)
        elif kind == StructuralEvidenceKind.APPENDIX:
            match = self.EXPLICIT_APPENDIX.fullmatch(value)
        if match is None:
            match = self.SIMPLE_HEADING.fullmatch(value)
        if match is None and kind in {
            StructuralEvidenceKind.CHAPTER,
            StructuralEvidenceKind.SECTION,
        }:
            match = self.COMPACT_CJK_HEADING.fullmatch(value)
        if match is None:
            return None
        number = self._normalize_number(match.group("number"))
        title = self._normalize_text(match.group("title"))
        if not self._number_matches_kind(number, kind):
            return None
        return number, title

    def _source_references_are_valid(
        self, candidate: StructuralEvidenceCandidate
    ) -> bool:
        references = candidate.source_references
        if not references or len(references) != len(candidate.source_fragments):
            return False
        if any(not fragment.strip() for fragment in candidate.source_fragments):
            return False
        if len({item.page_number for item in references}) != 1:
            return False
        positions = [item.source_order_index for item in references]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            return False
        if len(positions) > 1 and any(
            right != left + 1 for left, right in zip(positions, positions[1:])
        ):
            return False
        if candidate.source_kind == SourceKind.OCR_TEXT:
            return all(item.raw_line_id is not None for item in references)
        if candidate.source_kind == SourceKind.PDF_TEXT:
            return all(item.parser_line_index is not None for item in references)
        return False

    @staticmethod
    def _producer_matches_source(candidate: StructuralEvidenceCandidate) -> bool:
        if candidate.source_kind == SourceKind.PDF_TEXT:
            return candidate.producer == StructuralEvidenceProducer.PDF_TEXT
        return candidate.producer in {
            StructuralEvidenceProducer.OCR_TRANSITION,
            StructuralEvidenceProducer.OCR_CONTEXT,
        }

    @staticmethod
    def _region_is_allowed(candidate: StructuralEvidenceCandidate) -> bool:
        if candidate.structure_kind in {
            StructuralEvidenceKind.APPENDIX,
            StructuralEvidenceKind.APPENDIX_SECTION,
        }:
            return candidate.region_role == DocumentRegionType.APPENDIX
        return candidate.region_role == DocumentRegionType.NORMATIVE_BODY

    def _number_matches_kind(
        self, number: str, kind: StructuralEvidenceKind
    ) -> bool:
        pattern = {
            StructuralEvidenceKind.CHAPTER: self.NUMERIC_CHAPTER,
            StructuralEvidenceKind.SECTION: self.NUMERIC_SECTION,
            StructuralEvidenceKind.APPENDIX: self.APPENDIX,
            StructuralEvidenceKind.APPENDIX_SECTION: self.APPENDIX_SECTION,
        }[kind]
        return pattern.fullmatch(number) is not None

    @staticmethod
    def _level(kind: StructuralEvidenceKind) -> int:
        return 1 if kind in {
            StructuralEvidenceKind.CHAPTER,
            StructuralEvidenceKind.APPENDIX,
        } else 2

    @staticmethod
    def _append_basis(
        values: list[StructuralValidationBasis], value: StructuralValidationBasis
    ) -> None:
        if value not in values:
            values.append(value)

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value or "")
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _normalize_number(value: str) -> str:
        normalized = StructuralEvidenceResolverService._normalize_text(value).upper()
        if re.fullmatch(r"\d+(?:\.\d+){0,3}", normalized):
            return ".".join(str(int(part)) for part in normalized.split("."))
        if re.fullmatch(r"[A-Z](?:\.\d+){0,3}", normalized):
            head, *tail = normalized.split(".")
            return ".".join((head, *(str(int(part)) for part in tail)))
        return normalized

    @staticmethod
    def _valid_checksum(value: str) -> bool:
        return re.fullmatch(r"[0-9a-fA-F]{64}", value or "") is not None
