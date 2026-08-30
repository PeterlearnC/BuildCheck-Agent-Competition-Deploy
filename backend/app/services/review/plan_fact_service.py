"""Ground and deterministically parse exact ReviewUnit fact subspans."""

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

from app.schemas.compliance_comparison import (
    PlanFact,
    PlanFactParsingStatus,
    PlanFactType,
    SourceSpanSelector,
)
from app.schemas.review_unit import ReviewUnit
from app.services.review.review_unit_service import ReviewUnitService


PLAN_FACT_ID_VERSION = "v0.1-c.2-f1-plan-fact"
PLAN_FACT_PARSER_VERSION = "v0.1-c.2-f1-exact-numeric"
_NUMBER_WITH_UNIT = re.compile(
    r"(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?P<unit>%|℃|°C|[A-Za-zµμ]+|毫米|厘米|千米|米|千克|公斤|吨|分钟|小时|秒|度)",
    re.IGNORECASE,
)


class PlanFactError(ValueError):
    pass


class PlanFactSpanError(PlanFactError):
    pass


class PlanFactService:
    def __init__(self, review_unit_service: ReviewUnitService | None = None) -> None:
        self.review_unit_service = review_unit_service or ReviewUnitService()

    def create_many(
        self,
        review_unit: ReviewUnit,
        selectors: list[SourceSpanSelector],
    ) -> list[PlanFact]:
        verified = self.review_unit_service.verify(review_unit)
        return [self._create_verified(verified, selector) for selector in selectors]

    def verify_many(
        self, review_unit: ReviewUnit, plan_facts: list[PlanFact]
    ) -> list[PlanFact]:
        verified = self.review_unit_service.verify(review_unit)
        expected = [
            self._create_verified(
                verified,
                SourceSpanSelector(
                    char_start=fact.char_start,
                    char_end=fact.char_end,
                    source_text=fact.source_text,
                ),
            )
            for fact in plan_facts
        ]
        if expected != plan_facts:
            raise PlanFactError("PlanFact fields do not match deterministic recomputation.")
        return plan_facts

    def _create_verified(
        self, review_unit: ReviewUnit, selector: SourceSpanSelector
    ) -> PlanFact:
        if selector.char_end > len(review_unit.source_text) or not (
            review_unit.source_text.startswith(selector.source_text, selector.char_start)
        ):
            raise PlanFactSpanError(
                "PlanFact selection does not match the grounded ReviewUnit source text."
            )
        matches = list(_NUMBER_WITH_UNIT.finditer(selector.source_text))
        if len(matches) == 1:
            match = matches[0]
            try:
                normalized_value = Decimal(match.group("value"))
            except InvalidOperation as exc:
                raise PlanFactError("PlanFact numeric value is invalid.") from exc
            fact_type = PlanFactType.NUMERIC
            parsing_status = PlanFactParsingStatus.PROVEN
            unit = match.group("unit")
        elif len(matches) > 1:
            normalized_value = None
            fact_type = PlanFactType.NUMERIC
            parsing_status = PlanFactParsingStatus.AMBIGUOUS
            unit = None
        else:
            normalized_value = None
            fact_type = PlanFactType.UNSUPPORTED
            parsing_status = PlanFactParsingStatus.UNSUPPORTED
            unit = None

        source_hash = hashlib.sha256(selector.source_text.encode("utf-8")).hexdigest()
        plan_fact_id = self._plan_fact_id(
            review_unit_id=review_unit.review_unit_id,
            char_start=selector.char_start,
            char_end=selector.char_end,
            source_hash=source_hash,
        )
        return PlanFact(
            plan_fact_id=plan_fact_id,
            review_unit_id=review_unit.review_unit_id,
            char_start=selector.char_start,
            char_end=selector.char_end,
            source_text=selector.source_text,
            source_text_sha256=source_hash,
            fact_type=fact_type,
            parsing_status=parsing_status,
            normalized_value=normalized_value,
            unit=unit,
            parser_method="EXACT_SPAN_DETERMINISTIC",
            parser_version=PLAN_FACT_PARSER_VERSION,
        )

    @staticmethod
    def _plan_fact_id(
        *, review_unit_id: str, char_start: int, char_end: int, source_hash: str
    ) -> str:
        payload = {
            "identity_version": PLAN_FACT_ID_VERSION,
            "review_unit_id": review_unit_id,
            "char_start": char_start,
            "char_end": char_end,
            "source_text_sha256": source_hash,
            "parser_version": PLAN_FACT_PARSER_VERSION,
        }
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "planfact_" + hashlib.sha256(serialized).hexdigest()
