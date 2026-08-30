"""Select and parse exact normative evidence subspans without semantic inference."""

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

from app.schemas.compliance_review import ReviewEvidenceBinding
from app.schemas.compliance_comparison import SourceSpanSelector
from app.schemas.normative_requirement import (
    NormativeRequirement,
    NumericOperator,
    RequirementAtomicity,
    RequirementExtractionMethod,
    RequirementModality,
)
from app.schemas.standards import ArticleType


REQUIREMENT_ID_VERSION = "v0.1-c.2-f1-requirement"
REQUIREMENT_EXTRACTION_VERSION = "v0.1-c.2-f1-exact-numeric"

_UNIT = r"(?:%|℃|°C|[A-Za-zµμ]+|毫米|厘米|千米|米|千克|公斤|吨|分钟|小时|秒|度)"
_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_NUMERIC_PATTERNS: tuple[tuple[NumericOperator, re.Pattern[str]], ...] = (
    (
        NumericOperator.LE,
        re.compile(
            rf"(?:不得超过|不应超过|不得大于|不应大于|不大于|至多|小于或等于|小于等于|must\s+not\s+exceed|shall\s+not\s+exceed|no\s+more\s+than)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})",
            re.IGNORECASE,
        ),
    ),
    (
        NumericOperator.GE,
        re.compile(
            rf"(?:不得低于|不应低于|不得小于|不应小于|不小于|至少|大于或等于|大于等于|at\s+least|not\s+less\s+than)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})",
            re.IGNORECASE,
        ),
    ),
    (
        NumericOperator.LT,
        re.compile(
            rf"(?:应小于|必须小于|shall\s+be\s+less\s+than|must\s+be\s+less\s+than)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})",
            re.IGNORECASE,
        ),
    ),
    (
        NumericOperator.GT,
        re.compile(
            rf"(?:应大于|必须大于|shall\s+be\s+greater\s+than|must\s+be\s+greater\s+than)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})",
            re.IGNORECASE,
        ),
    ),
    (
        NumericOperator.EQ,
        re.compile(
            rf"(?:应等于|必须等于|应为|必须为|shall\s+equal|must\s+equal)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})",
            re.IGNORECASE,
        ),
    ),
    (
        NumericOperator.LE,
        re.compile(rf"(?:<=|≤)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})"),
    ),
    (
        NumericOperator.GE,
        re.compile(rf"(?:>=|≥)\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})"),
    ),
    (
        NumericOperator.LT,
        re.compile(rf"(?<![<>=])<\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})"),
    ),
    (
        NumericOperator.GT,
        re.compile(rf"(?<![<>=])>\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})"),
    ),
    (
        NumericOperator.EQ,
        re.compile(rf"(?<![<>=])=\s*(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})"),
    ),
)
_CONDITION_PATTERN = re.compile(
    r"(?:^|[，,；;。.]\s*)(?:当|若|如果|if\b|when\b)", re.IGNORECASE
)
_NORMATIVE_CUE = re.compile(r"(?:不得|禁止|不应|应当|应|必须|shall\b|must\b)", re.IGNORECASE)


class RequirementSelectionError(ValueError):
    pass


class RequirementSpanError(RequirementSelectionError):
    pass


class RequirementService:
    def select(
        self,
        binding: ReviewEvidenceBinding,
        selector: SourceSpanSelector,
    ) -> NormativeRequirement:
        evidence = binding.evidence
        if binding.standard_evidence_id != evidence.id:
            raise RequirementSelectionError("Evidence binding identity is inconsistent.")
        if evidence.article_type != ArticleType.NORMATIVE:
            raise RequirementSelectionError(
                "Only ACCEPT-bound normative evidence may define a requirement."
            )
        text = evidence.source_text
        if selector.char_end > len(text) or not text.startswith(
            selector.source_text, selector.char_start
        ):
            raise RequirementSpanError(
                "Requirement selection does not match the EvidenceEnvelope source text."
            )

        requirement_text = selector.source_text
        matches = self._numeric_matches(requirement_text)
        condition = _CONDITION_PATTERN.search(requirement_text)
        if condition:
            modality = RequirementModality.CONDITIONAL
            atomicity = RequirementAtomicity.UNRESOLVED
            operator = None
            value = None
            unit = None
            condition_text = requirement_text
        elif len(matches) == 1:
            modality = RequirementModality.NUMERIC_LIMIT
            atomicity = RequirementAtomicity.PROVEN
            operator, value, unit = matches[0]
            condition_text = None
        elif len(matches) > 1:
            modality = RequirementModality.UNSUPPORTED
            atomicity = RequirementAtomicity.UNRESOLVED
            operator = None
            value = None
            unit = None
            condition_text = None
        else:
            modality = RequirementModality.UNSUPPORTED
            cues = list(_NORMATIVE_CUE.finditer(requirement_text))
            independently_separated = bool(re.search(r"[；;]\s*", requirement_text))
            atomicity = (
                RequirementAtomicity.PROVEN
                if len(cues) == 1 and not independently_separated
                else RequirementAtomicity.UNRESOLVED
            )
            operator = None
            value = None
            unit = None
            condition_text = None

        text_hash = hashlib.sha256(requirement_text.encode("utf-8")).hexdigest()
        requirement_id = self._requirement_id(
            evidence_id=evidence.id,
            char_start=selector.char_start,
            char_end=selector.char_end,
            text_hash=text_hash,
        )
        return NormativeRequirement(
            requirement_id=requirement_id,
            evidence_id=evidence.id,
            standard_id=evidence.standard_id,
            article_id=evidence.article_id,
            article_number=evidence.article_number,
            requirement_char_start=selector.char_start,
            requirement_char_end=selector.char_end,
            requirement_text=requirement_text,
            requirement_text_sha256=text_hash,
            modality=modality,
            atomicity=atomicity,
            condition_text=condition_text,
            operator=operator,
            value=value,
            unit=unit,
            extraction_method=RequirementExtractionMethod.EXACT_SPAN_DETERMINISTIC,
            extraction_version=REQUIREMENT_EXTRACTION_VERSION,
        )

    def verify(
        self,
        requirement: NormativeRequirement,
        binding: ReviewEvidenceBinding,
    ) -> NormativeRequirement:
        evidence = binding.evidence
        if (
            requirement.evidence_id != evidence.id
            or requirement.standard_id != evidence.standard_id
            or requirement.article_id != evidence.article_id
            or requirement.article_number != evidence.article_number
        ):
            raise RequirementSelectionError(
                "Requirement provenance does not match the selected evidence."
            )
        expected = self.select(
            binding,
            SourceSpanSelector(
                char_start=requirement.requirement_char_start,
                char_end=requirement.requirement_char_end,
                source_text=requirement.requirement_text,
            ),
        )
        if expected != requirement:
            raise RequirementSelectionError(
                "Requirement fields do not match deterministic recomputation."
            )
        return requirement

    @staticmethod
    def _numeric_matches(
        text: str,
    ) -> list[tuple[NumericOperator, Decimal, str]]:
        found: list[tuple[int, NumericOperator, Decimal, str]] = []
        occupied: list[tuple[int, int]] = []
        for operator, pattern in _NUMERIC_PATTERNS:
            for match in pattern.finditer(text):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                try:
                    value = Decimal(match.group("value"))
                except InvalidOperation:
                    continue
                found.append((span[0], operator, value, match.group("unit")))
                occupied.append(span)
        return [(operator, value, unit) for _, operator, value, unit in sorted(found)]

    @staticmethod
    def _requirement_id(
        *, evidence_id: str, char_start: int, char_end: int, text_hash: str
    ) -> str:
        payload = {
            "identity_version": REQUIREMENT_ID_VERSION,
            "evidence_id": evidence_id,
            "requirement_char_start": char_start,
            "requirement_char_end": char_end,
            "requirement_text_sha256": text_hash,
            "schema_version": REQUIREMENT_EXTRACTION_VERSION,
        }
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "requirement_" + hashlib.sha256(serialized).hexdigest()
