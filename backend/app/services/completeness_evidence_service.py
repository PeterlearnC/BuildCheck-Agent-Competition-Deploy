"""Evidence adapters over the existing V0.2 DocumentAnalysis model."""

import re
import unicodedata
from typing import Any, Iterable

from app.schemas.analysis import DocumentAnalysis, ProjectInfo, ScheduleInfo, SourceValue
from app.schemas.completeness_review import (
    CompletenessEvidence,
    EvidenceSourceType,
)


class CompletenessEvidenceService:
    def __init__(self, analysis: DocumentAnalysis) -> None:
        self.analysis = analysis

    def chapters(self, patterns: Iterable[str]) -> list[CompletenessEvidence]:
        expressions = tuple(re.compile(pattern) for pattern in patterns)
        result: list[CompletenessEvidence] = []
        for chapter in self.analysis.chapters:
            if not any(expression.search(chapter.title) for expression in expressions):
                continue
            result.append(
                CompletenessEvidence(
                    source_type=EvidenceSourceType.CHAPTER,
                    source_page=chapter.start_page,
                    matched_title=chapter.title,
                    field_path="chapters",
                )
            )
        return self.deduplicate(result)

    def section_presence(self, field_name: str) -> list[CompletenessEvidence]:
        """Expose an existing section-presence flag without inventing provenance."""
        if field_name not in {"quality", "safety", "environment", "emergency"}:
            raise ValueError(f"Unsupported section_presence field: {field_name}")
        if not getattr(self.analysis.section_presence, field_name):
            return []
        return [
            CompletenessEvidence(
                source_type=EvidenceSourceType.SECTION_PRESENCE,
                source_page=None,
                matched_title=field_name,
                field_path=f"section_presence.{field_name}",
            )
        ]

    @classmethod
    def deduplicate(
        cls, evidence: Iterable[CompletenessEvidence]
    ) -> list[CompletenessEvidence]:
        result: list[CompletenessEvidence] = []
        seen: set[tuple[object, ...]] = set()
        for item in evidence:
            identity = (
                item.source_type,
                item.source_page,
                cls._normalize(item.matched_title),
                item.field_path,
                cls._normalize(item.source_text),
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(item)
        return result

    @staticmethod
    def _normalize(value: str | None) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKC", value)
        return re.sub(r"\s+", "", normalized).casefold()

    @staticmethod
    def field(
        item: Any,
        field_path: str,
        display_value: str | int | None = None,
    ) -> CompletenessEvidence | None:
        if item is None:
            return None
        if isinstance(item, SourceValue):
            value = item.value
            source_page = item.source_page
            source_text = item.source_text
        elif isinstance(item, str):
            value = item.strip()
            source_page = None
            source_text = None
        else:
            value = display_value
            source_page = getattr(item, "source_page", None)
            source_text = getattr(item, "source_text", None)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return CompletenessEvidence(
            source_type=EvidenceSourceType.FIELD,
            source_page=source_page,
            source_text=source_text,
            matched_title=str(value),
            field_path=field_path,
        )

    def project_fields(self) -> list[CompletenessEvidence]:
        result: list[CompletenessEvidence] = []
        for field_name in ProjectInfo.model_fields:
            evidence = self.field(
                getattr(self.analysis.project, field_name),
                f"project.{field_name}",
            )
            if evidence is not None:
                result.append(evidence)
        return result

    def schedule_fields(self) -> list[CompletenessEvidence]:
        result: list[CompletenessEvidence] = []
        for field_name in ScheduleInfo.model_fields:
            evidence = self.field(
                getattr(self.analysis.schedule, field_name),
                f"schedule.{field_name}",
            )
            if evidence is not None:
                result.append(evidence)
        return result

    def compilation_basis(self) -> list[CompletenessEvidence]:
        return self._source_list(self.analysis.compilation_basis, "compilation_basis")

    def materials(self) -> list[CompletenessEvidence]:
        result: list[CompletenessEvidence] = []
        for index, item in enumerate(self.analysis.main_materials):
            evidence = self.field(
                item,
                f"main_materials[{index}]",
                display_value=item.name,
            )
            if evidence is not None:
                result.append(evidence)
        return result

    def methods(self) -> list[CompletenessEvidence]:
        result: list[CompletenessEvidence] = []
        for index, item in enumerate(self.analysis.main_methods):
            evidence = self.field(
                item,
                f"main_methods[{index}]",
                display_value=item.name,
            )
            if evidence is not None:
                result.append(evidence)
        return result

    def _source_list(
        self, items: list[SourceValue | str], field_name: str
    ) -> list[CompletenessEvidence]:
        result: list[CompletenessEvidence] = []
        for index, item in enumerate(items):
            evidence = self.field(item, f"{field_name}[{index}]")
            if evidence is not None:
                result.append(evidence)
        return result
